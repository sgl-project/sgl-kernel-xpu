# GEMM + Square Sum - Phase 2 Implementation TODO

## Current Status

✅ **Phase 1 Complete** - Infrastructure
- Build system integrated
- PyTorch bindings work
- Mainloop skeleton exists
- Currently uses PyTorch fallback

⚠️ **Phase 2 In Progress** - Real CUTLASS Kernel

## What Needs To Be Done

### Step 1: Configure MMA Atom and TiledMMA

**File:** `src/sycl/kernels/gemm_sqrsum/device/gemm_sqrsum_types.hpp`

**Reference:** `src/sycl/HcPreGemm.cpp` lines 94-103

```cpp
// Add after line 64 in gemm_sqrsum_types.hpp:

// MMA configuration for DPAS (XPU matrix multiply-accumulate)
using MmaAtom = MMA_Atom<XE_DPAS_TT<8, Element, Element>>;

// TiledMMA: defines how MMA atoms tile across subgroups
// For 256x256x16 tile with 8x16x8 DPAS atom:
//   - Need 32 atoms in M (256/8)
//   - Need 16 atoms in N (256/16)
//   - Need 2 atoms in K (16/8)
using TiledMma = typename TiledMMAHelper<
    MmaAtom,
    Shape<Int<TileSizeOpt::TileM>, Int<TileSizeOpt::TileN>, Int<TileSizeOpt::TileK>>,
    Layout<Shape<_16, _16, _1>, Stride<_16, _1, _0>>  // 16x16 subgroup layout
>::TiledMMA;
```

### Step 2: Update Mainloop to Use Real Types

**File:** `src/sycl/kernels/gemm_sqrsum/collective/xe_gemm_sqrsum_mainloop.hpp`

**Current Issue:** Line 126-127 use `void` placeholders

```cpp
// BEFORE (lines 126-127):
using MMA_Atom = void;  // Placeholder
using TiledMMA = void;  // Placeholder

// AFTER: Remove these lines - TiledMMA will be passed as template parameter
```

The mainloop already has the structure, but it needs:
1. Remove placeholder types
2. Ensure `TiledMMA` is passed from kernel configuration
3. Fix fragment types to match actual MMA output

### Step 3: Complete the Epilogue

**File:** `src/sycl/kernels/gemm_sqrsum/collective/xe_gemm_sqrsum_epilogue.hpp`

**Current Issue:** Lines 52-65 are just comments

The epilogue needs to:

#### 3a. Write GEMM output C to global memory

```cpp
// Partition global C tensor
auto layout_C = make_layout(
    make_shape(M, N),
    make_stride(params.stride_C_m, params.stride_C_n));
Tensor gC = make_tensor(make_gmem_ptr(params.ptr_C), layout_C);

// Get this thread's partition of C
Tensor gC_tile = local_tile(gC, TileShape{}, blk_mn, Step<_1, _1>{});

// Create copy from register to global
// ... (need to partition by thread and copy tC -> gC_tile)
```

#### 3b. Reduce and write sqrsum

The tricky part: `tSqrSum` is per-thread, but each row needs ONE final value.

```cpp
// Need to reduce across all threads that worked on same row
// Options:
//   1. Use subgroup reduce operations
//   2. Use shared memory for cross-subgroup reduction
//   3. Use atomic adds to global memory (simplest but slower)

// Example with atomics (simplest):
int global_row = blk_m * BLK_M + local_row_id;
if (global_row < M) {
  // Atomic add this thread's contribution
  sycl::atomic_ref<ElementSqrSum, ...> atomic_sqrsum(params.ptr_sqrsum[global_row]);
  atomic_sqrsum.fetch_add(tSqrSum(local_row_idx));
}
```

### Step 4: Create Kernel Wrapper

**File:** `src/sycl/kernels/gemm_sqrsum/kernel/xe_gemm_sqrsum_kernel.hpp`

**Current Issue:** Lines 157-167 - operator() is incomplete

The kernel needs:

```cpp
CUTLASS_DEVICE void operator()(Params const& params, SharedStorage& shared_storage) {
  using namespace sycl::ext::oneapi;

  int thr_id = this_work_item::get_nd_item<3>().get_local_id(0);
  int blk_m = this_work_item::get_nd_item<3>().get_group(2);
  int blk_n = this_work_item::get_nd_item<3>().get_group(1);

  // Create tensors from pointers
  auto layout_A = make_layout(make_shape(params.M, params.K), ...);
  auto layout_B = make_layout(make_shape(params.K, params.N), ...);
  Tensor A = make_tensor(make_gmem_ptr(params.ptr_A), layout_A);
  Tensor B = make_tensor(make_gmem_ptr(params.ptr_B), layout_B);

  // Create 2D slices
  auto A_2D = A(append<rank_v<decltype(A)>>(make_coord(_, _), 0));
  auto B_2D = B(append<rank_v<decltype(B)>>(make_coord(_, _), 0));

  // Initialize mainloop and epilogue
  CollectiveMainloop mainloop(params.mainloop, shared_storage.mainloop);
  CollectiveEpilogue epilogue(params.epilogue, shared_storage.epilogue);

  // Create accumulators
  typename CollectiveMainloop::FragGemm tC;
  typename CollectiveMainloop::FragSqrSum tSqrSum;

  // Run mainloop
  mainloop(A_2D, B_2D, tC, tSqrSum, make_coord(blk_m, blk_n), thr_id);

  // Run epilogue
  epilogue(tC, tSqrSum, make_coord(blk_m, blk_n), params.M, params.N, thr_id);
}
```

Also needs:

```cpp
static compat::dim3 get_grid_shape(Params const& params) {
  int grid_m = (params.M + BLK_M - 1) / BLK_M;
  int grid_n = (params.N + BLK_N - 1) / BLK_N;
  return compat::dim3(1, grid_n, grid_m);
}

static compat::dim3 get_block_shape() {
  // Number of threads = number of subgroups × subgroup size
  int num_subgroups = NumSubgroups;  // from mainloop
  int subgroup_size = intel::sg_size;  // 16 for XPU
  return compat::dim3(num_subgroups * subgroup_size, 1, 1);
}

static constexpr int SharedStorageSize = sizeof(SharedStorage);
```

### Step 5: Wire It All Together

**File:** `src/sycl/kernels/gemm_sqrsum/device/gemm_sqrsum_types.hpp`

Replace the fallback in `runGemmSqrSumImpl` with:

```cpp
using CutlassElement = typename ToCutlassElementType<Element>::type;

// Configure kernel
using TileShape = Shape<
    Int<TileSizeOpt::TileM>,
    Int<TileSizeOpt::TileN>,
    Int<TileSizeOpt::TileK>>;

using MmaAtom = MMA_Atom<XE_DPAS_TT<8, CutlassElement, CutlassElement>>;
using TiledMma = ...; // as in Step 1

// Create tensor types
using TensorA = decltype(make_tensor(make_gmem_ptr<CutlassElement const>(nullptr), 
                                      make_layout(make_shape(M, K), ...)));
using TensorB = decltype(make_tensor(make_gmem_ptr<CutlassElement const>(nullptr),
                                      make_layout(make_shape(K, N), ...)));

// Create mainloop
using DispatchPolicy = cutlass::gemm_sqrsum::XeDefault<1>;
using CollectiveMainloop = cutlass::gemm_sqrsum::collective::XeGemmSqrSumMainloop<
    DispatchPolicy, TiledMma, TensorA, TensorB>;

// Create epilogue
using TensorC = decltype(make_tensor(make_gmem_ptr<CutlassElement>(nullptr),
                                      make_layout(make_shape(M, N), ...)));
using TensorSqrSum = decltype(make_tensor(make_gmem_ptr<CutlassElement>(nullptr),
                                           make_layout(make_shape(M), ...)));
using CollectiveEpilogue = cutlass::gemm_sqrsum::collective::XeGemmSqrSumEpilogue<
    TensorC, TensorSqrSum>;

// Create kernel
using Kernel = cutlass::gemm_sqrsum::kernel::GemmSqrSumKernel<CollectiveMainloop, CollectiveEpilogue>;

// Create runner
using Runner = cutlass::gemm_sqrsum::device::GemmSqrSum<Kernel>;

// Setup arguments
typename Kernel::Arguments args;
args.mainloop = {...};  // mainloop params
args.epilogue = {       // epilogue params
    C.data_ptr<Element>(),
    N, 1,  // strides
    sqrsum.data_ptr<Element>()
};
args.M = M;
args.K = K;
args.N = N;
args.ptr_A = A.data_ptr<Element>();
args.ptr_B = B.data_ptr<Element>();
// ... set strides

// Launch
Runner gemm;
CUTLASS_CHECK(gemm.can_implement(args));
CUTLASS_CHECK(gemm.run(args, nullptr));
```

## Testing Strategy

### Stage 1: Compile Check
```bash
cd build && make sgl-ops-sycl-gemm_sqrsum
```

### Stage 2: Small Matrix Test
```python
M, K, N = 64, 32, 64  # Small enough to debug
A = torch.randn(M, K, dtype=torch.float16, device="xpu")
# ... test
```

### Stage 3: Verify Correctness
```python
# Run test_gemm_sqrsum.py with various sizes
python test_gemm_sqrsum.py
```

### Stage 4: Performance Benchmark
Compare against:
- PyTorch: `torch.matmul(A, B)`
- Separate ops: `C = A @ B; sqrsum = (A*A).sum(1)`

## Key Challenges

### Challenge 1: Fragment Layout
CUTE's fragment layout can be confusing. The `tArA` fragment is partitioned by thread in a specific pattern. Need to understand:
- How elements map to threads
- How to reduce correctly for sqrsum

### Challenge 2: Cross-Thread Reduction
Multiple threads compute parts of the same row's sqrsum. Need proper reduction:
- Subgroup-level: `sycl::reduce_over_group()`
- Cross-subgroup: shared memory or atomics

### Challenge 3: Template Instantiation
CUTLASS has heavy templates. Compile errors can be cryptic.
- Start simple (e.g., FP32 only)
- Add types incrementally
- Check each step compiles

## Estimated Effort

- **Step 1 (MMA config)**: 1-2 hours - straightforward, copy from HcPreGemm
- **Step 2 (Mainloop)**: 30 mins - mostly done
- **Step 3 (Epilogue)**: 3-4 hours - the hardest part, need to understand CUTE copy
- **Step 4 (Kernel wrapper)**: 1-2 hours - boilerplate, but detailed
- **Step 5 (Integration)**: 2-3 hours - wiring and debugging
- **Testing/Debug**: 4-6 hours - fixing inevitable issues

**Total: ~12-18 hours** of focused work for someone familiar with CUTLASS/CUTE.

## Alternative: Simpler Approach

If full CUTLASS is too complex, could do:
1. Use CUTLASS for GEMM only (existing GemmUniversal)
2. Separate kernel for sqrsum
3. Still faster than PyTorch fallback, less complex

This would be **~4-6 hours** instead.

## Next Steps

1. **Decision**: Full CUTLASS integration vs. simpler hybrid approach?
2. **Start with**: Step 1 (MMA configuration) - easiest entry point
3. **Test incrementally**: Don't wait until everything is done to compile

Would you like me to start with Step 1?
