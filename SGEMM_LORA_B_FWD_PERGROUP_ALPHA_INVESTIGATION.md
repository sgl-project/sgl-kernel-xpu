# SGEMM LoRA-B Forward — Per-Group Alpha Debugging Investigation

> **Status:** ROOT CAUSE CONFIRMED (see §10) — the `alpha_ptr_array` per-group mechanism is
> genuinely broken on this Xe grouped kernel. Fix not yet applied.
> This document is a detailed, chronological investigation log intended for educational
> reference. It captures the hypotheses, the exact commands run, what each experiment
> revealed, and where the reasoning went wrong (and was corrected). It is intentionally verbose.
>
> **TL;DR for the impatient:** The original "L is hardcoded to 0" hypothesis (§3.1) was
> **correct**. It was wrongly abandoned in §3.5 because the upstream example *appeared* to pass
> — but that "pass" was an artifact of (a) the example defaulting `--alpha=1` so it never took
> the per-group branch, and (b) its random per-group alphas colliding. When forced to use
> genuinely distinct per-group alphas, **the upstream example FAILS exactly like our kernel**
> (§10). Per-group alpha via `alpha_ptr_array` does not work on this kernel; the fix is to fold
> `scalings[l]` into the operand (pre-scale) rather than rely on the fusion (§10.4).

---

## 1. Background & Problem Statement

### 1.1 What we're building

We are implementing a new SYCL/CUTLASS kernel `sgemm_lora_b_fwd` for Intel XPU (Battlemage /
BMG, using the XMX DPAS MMA units), mirroring the structure of the already-working
`sgemm_lora_a_fwd` kernel.

**LoRA-A forward** computes:
```
D = A @ B^T          (alpha = 1, beta = 0)
```

**LoRA-B forward** computes, per LoRA segment:
```
D = scalings[l] * (input_x @ weights[l]^T) + beta * base_output
```
where:
- Each LoRA segment `s` routes to an adapter `l = weight_indices[s]`.
- `scalings[l]` is a **per-segment scalar** (this is `lora_alpha / rank`, folded in by the caller).
- `base_output` is an optional residual: when present `beta = 1` (fused add), else `beta = 0`.

The kernel is implemented as a **CUTLASS pointer-array grouped GEMM**
(`GemmUniversalMode::kGrouped`), where each LoRA segment is one "group" of the grouped GEMM.
- `K` (reduction dim) = `max_rank`
- `N` (output cols) = `output_dim`
- `M_s` (rows for segment `s`) = number of tokens routed to segment `s`

### 1.2 The bug

Multi-segment tests fail. All segments end up applying `scalings[0]` (the **first** segment's
scaling) regardless of which adapter they actually route to. Single-segment tests pass. The
GEMM math itself (routing, the `A @ B^T` product, the residual add) is all correct — **only
the per-segment alpha is wrong.**

Test results (after a clean rebuild):
```
25 failed, 31 passed
```
Every failure is a multi-segment case. Every pass is either single-segment or a case where all
scalings are equal.

---

## 2. The Mechanism We're Trying To Use

### 2.1 CUTLASS epilogue fusion

The grouped epilogue dispatch policy is `cutlass::epilogue::IntelXeGenericGroup`. It binds the
fusion `cutlass::epilogue::fusion::LinearCombination<...>`, which expands (for this policy) to
`Sm90LinearCombinationPtrArray`:

```
D = beta * C + (alpha * acc)
```

where both `alpha` and `beta` are provided through `Sm90ScalarBroadcastPtrArray<ElementScalar, Stride<_0,_0,int64_t>>`.

`Sm90ScalarBroadcastPtrArray` supports **three** ways to source the scalar (see
`update_scalar()` in
`build/_deps/repo-cutlass-sycl-src/include/cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp`):

```cpp
CUTLASS_DEVICE void
update_scalar(int l_coord = 0) {
  int l_offset = l_coord * size<2>(params_ptr->dScalar[0]);

  if (params_ptr->scalar_ptr_arrays[0] != nullptr) {
    // (A) Pointer-array variant: array of POINTERS, one per group.
    scalar = *(params_ptr->scalar_ptr_arrays[0][l_coord]);
  }
  else if (params_ptr->scalar_ptrs[0] != nullptr) {
    // (B) Strided value-buffer variant: contiguous VALUE buffer.
    scalar = params_ptr->scalar_ptrs[0][l_offset];
  }
  else {
    // (C) Literal fallback: a single compile-visible value.
    scalar = params_ptr->scalars[0];
  }
  ...
}
```

`update_scalar(l_coord)` is called with `l_coord = get<3>(tile_coord_mnkl)` from both
`get_producer_load_callbacks()` and `get_consumer_store_callbacks()`.

### 2.2 The two per-group alpha mechanisms

- **(A) `alpha_ptr_array`** — an array of *pointers*, `scalar_ptr_arrays[0][l_coord]` selects the
  pointer for group `l_coord`, then dereferences it. Stride not used (`dAlpha = {_0,_0,1}`).
- **(B) `alpha_ptr`** — a *strided contiguous value buffer*, `scalar_ptrs[0][l_coord * stride]`.

The upstream example `04_bmg_grouped_gemm.cpp` uses **(A)** with `dAlpha = {_0,_0,1}` and
`alpha = 0`.

### 2.3 What our code does

`src/sycl/kernels/lora/common/group_gemm_types.hpp` → `args_from_options()` wires:

```cpp
if (alpha_ptr_array.has_value()) {
  fusion_args.alpha = ElementScalar(0);
  fusion_args.alpha_ptr_array =
      reinterpret_cast<ElementScalar const* const*>(alpha_ptr_array->data_ptr<int64_t>());
  fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 1};
} else {
  fusion_args.alpha = alpha;
  fusion_args.alpha_ptr_array = nullptr;
  fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
}
```

The runner `sgemm_lora_b_fwd_runner.hpp` builds the pointer array:

```cpp
auto opt_i64 = torch::TensorOptions().dtype(torch::kInt64).device(device);
auto alpha_off = torch::arange(num_segments, opt_i64) * static_cast<int64_t>(sizeof(float));
auto alpha_ptrs = core::make_device_ptrs(meta.alpha, alpha_off, queue);
```

`meta.alpha` is a device fp32 buffer of length `num_segments`, filled by a SYCL kernel
(`build_grouped_gemm_meta`), one alpha per segment: `alpha[s] = scalings[weight_indices[s]]`.

So `alpha_ptrs[s]` = absolute device address of `meta.alpha[s]`. This is byte-for-byte the
same shape as what the upstream example does.

**And yet every group reads `alpha[0]`.**

---

## 3. Investigation Timeline

### 3.1 Initial (WRONG) hypothesis — "the kernel hardcodes L=0"

The grouped kernel is
`build/_deps/repo-cutlass-sycl-src/include/cutlass/gemm/kernel/xe_gemm_array_cooperative.hpp`.
Reading its main loop:

```bash
sed -n '250,340p' build/_deps/repo-cutlass-sycl-src/include/cutlass/gemm/kernel/xe_gemm_array_cooperative.hpp
```

Key lines:
```cpp
int32_t curr_group = -1;
if (work_tile_info.is_valid()) {
  curr_group = work_tile_info.L_idx;
  problem_shape_MNKL = append<4>(params.problem_shape.get_problem_shape(curr_group), 1);
}

while (work_tile_info.is_valid()) {
  ...
  auto m_coord = work_tile_info.M_idx;
  auto n_coord = work_tile_info.N_idx;
  ...
  if (did_group_change) {
    base_mainloop_params = CollectiveMainloop::Base::to_underlying_arguments(problem_shape_MNKL,
        CollectiveMainloop::to_base_arguments(params.mainloop, curr_group), params.workspace);
    base_epilogue_params = CollectiveEpilogue::Base::to_underlying_arguments(problem_shape_MNKL,
        CollectiveEpilogue::to_base_arguments(params.epilogue, curr_group), params.workspace);
    did_group_change = false;
  }
  auto tile_coord = make_coord(m_coord, n_coord, _, 0);   // <-- L HARDCODED TO 0
  ...
  epilogue(problem_shape_MNKL, subgroup_shape, tile_coord, accumulators, tiled_mma, thread_idx);
  ...
  did_group_change = curr_group != work_tile_info.L_idx;
}
```

**Initial (flawed) reasoning:** `tile_coord`'s L (4th coordinate) is hardcoded to `0`. Since the
fusion computes `l_coord = get<3>(tile_coord_mnkl)`, `update_scalar` always gets `l_coord = 0`,
so `alpha_ptr_array[0]` is always read → all groups get `alpha[0]`. Conclusion (WRONG): "the
fusion's per-group alpha is structurally defeated on this Xe grouped path."

I proposed a workaround (fold `scalings[l]` into the `A` operand by pre-scaling `input_x`), and
the edit was rejected by the user. **Good instinct on the user's part** — because the reasoning
was incomplete.

### 3.2 The critical counter-evidence — the upstream example works

The upstream example `04_bmg_grouped_gemm.cpp` uses the **exact same mechanism**. If the "L=0"
theory were correct, that shipped example would be broken too. Examining the example:

```bash
grep -n "alpha\|Alpha\|dAlpha\|scalar\|per_group\|block_alpha" \
  build/_deps/repo-cutlass-sycl-src/examples/04_bmg_grouped_gemm/04_bmg_grouped_gemm.cpp
```

Its arg construction (lines ~443-454):
```cpp
// If pointers to alpha/beta are provided, i.e., alpha/beta can differ between batches/groups.
fusion_args.alpha = 0;
fusion_args.beta = 0;
fusion_args.alpha_ptr = nullptr;
fusion_args.beta_ptr = nullptr;
fusion_args.alpha_ptr_array = alpha_device.get();   // array of pointers
fusion_args.beta_ptr_array = beta_device.get();
fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 1};    // <-- identical to ours
fusion_args.dBeta = {cute::_0{}, cute::_0{}, 1};
```

Its per-group alpha setup (lines ~367-421):
```cpp
std::vector<ElementAccumulator *> ptr_alpha_host(options.groups);
for (int32_t i = 0; i < options.groups; ++i) {
  ...
  alpha_host.push_back((options.alpha == FLT_MAX) ? static_cast<...>((rand() % 5) + 1) : options.alpha);
  ptr_alpha_host.at(i) = block_alpha.get() + i;   // pointer into per-group value block
}
alpha_device.reset(options.groups);
alpha_device.copy_from_host(ptr_alpha_host.data());   // device array-of-pointers
block_alpha.copy_from_host(alpha_host.data());        // device value block
```

And its verification defaults to ON (`--verify=1`) with **random per-group alphas**. So if the
example passes, per-group alpha works on this exact kernel. **The "L=0" theory must be wrong.**

Verified the example instantiates the SAME dispatch policy as us:
```bash
grep -n "GEMMDispatchPolicy\|MainloopXe\|EpilogueDispatchPolicy\|IntelXe" \
  build/_deps/repo-cutlass-sycl-src/examples/04_bmg_grouped_gemm/04_bmg_grouped_gemm.cpp
```
```
586:  using GEMMDispatchPolicy = cutlass::gemm::MainloopXeL1StagedGroup<PipelineStages>;
587:  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGenericGroup;
```
→ identical to `group_gemm_types.hpp`.

### 3.3 Empirical probe #1 — confirm OUR kernel is wrong

Rather than keep reasoning statically, wrote a Python probe using our built `.so`. Set
`input_x` and `weights` to all ones so each output element equals `scalings[seg] * max_rank`,
one token per segment, distinct scalings per adapter:

```python
import torch
from sgl_kernel import sgemm_lora_b_fwd

torch.manual_seed(0)
dtype = torch.float16
num_tokens = 3
max_rank = 8
output_dim = 4
num_loras = 3

seg_indptr = torch.tensor([0,1,2,3], dtype=torch.int32, device="xpu")
weight_indices = torch.tensor([0,1,2], dtype=torch.int32, device="xpu")
lora_ranks = torch.tensor([max_rank]*num_loras, dtype=torch.int32, device="xpu")
scalings = torch.tensor([0.5, 2.0, 4.0], dtype=torch.float32, device="xpu")

input_x = torch.ones(num_tokens, max_rank, dtype=dtype, device="xpu")
weights = torch.ones(num_loras, output_dim, max_rank, dtype=dtype, device="xpu")

out = sgemm_lora_b_fwd(input_x=input_x, weights=weights, seg_indptr=seg_indptr,
    weight_indices=weight_indices, lora_ranks=lora_ranks, scalings=scalings,
    seg_lens=None, base_output=None)
print("out:\n", out.float().cpu())
implied = out.float().cpu()[:,0]/max_rank
print("implied alpha per segment:", implied.tolist())
```

Output:
```
out:
 tensor([[4., 4., 4., 4.],
        [4., 4., 4., 4.],
        [4., 4., 4., 4.]])
expected per row: scalings*max_rank = [4.0, 16.0, 32.0]
implied alpha per segment: [0.5, 0.5, 0.5]
```

→ Confirmed: all 3 segments apply `alpha = 0.5 = scalings[0]`. The kernel routes correctly
(the product is right) but every group uses the first alpha.

Repeated with **large M** (300 tokens/segment, larger than the tile M=256) to rule out a
tile-size / degenerate-M effect:
```python
counts = [300, 300, 300]
# ... same all-ones setup ...
# seg 0: implied alpha = 0.5  (expected 0.5)
# seg 1: implied alpha = 0.5  (expected 2.0)
# seg 2: implied alpha = 0.5  (expected 4.0)
```
Same failure regardless of segment size.

### 3.4 Building the upstream example standalone

To get ground truth, compiled and ran the upstream example directly. First attempt:

```bash
source ~/intel/oneapi/setvars.sh
conda activate my_py312_env
cd build
CUTLASS=_deps/repo-cutlass-sycl-src
icpx -fsycl -std=c++17 -fsycl-targets=intel_gpu_bmg_g21 \
  -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET -DCUTLASS_SYCL_SWITCH_WG=0 \
  -I${CUTLASS}/include -I${CUTLASS}/tools/util/include -I${CUTLASS}/applications \
  ${CUTLASS}/examples/04_bmg_grouped_gemm/04_bmg_grouped_gemm.cpp -o /tmp/bmg04
```
→ `fatal error: 'sycl_common.hpp' file not found`

Found it and added the include dir:
```bash
find _deps/repo-cutlass-sycl-src -name "sycl_common.hpp"
# _deps/repo-cutlass-sycl-src/examples/common/sycl_common.hpp
```

Second attempt added `-I${CUTLASS}/examples/common`:
→ `llvm-spirv command failed with exit code 18: RequiresExtension: SPV_INTEL_split_barrier`

Needed the SPIR-V extension flags. Extracted the real flags our build uses from `build.ninja`:
```bash
grep -m2 "split_barrier\|Xspirv-translator\|allow-unknown" build.ninja src/build.ninja
```
Found:
```
-fsycl-targets=spir64_gen
-Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate
-Xs -device bmg -options ...
```

Working compile command:
```bash
source ~/intel/oneapi/setvars.sh
conda activate my_py312_env
cd build
CUTLASS=_deps/repo-cutlass-sycl-src
icpx -fsycl -std=c++17 -fsycl-targets=spir64_gen \
  -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate \
  -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET -DCUTLASS_SYCL_SWITCH_WG=0 \
  -I${CUTLASS}/include -I${CUTLASS}/tools/util/include -I${CUTLASS}/applications -I${CUTLASS}/examples/common \
  ${CUTLASS}/examples/04_bmg_grouped_gemm/04_bmg_grouped_gemm.cpp \
  -Xs "-device bmg" \
  -o /tmp/bmg04
```
→ `Build succeeded.`

The compile log confirmed the instantiated kernel type is **identical** to ours:
```
cutlass::gemm::kernel::GemmUniversal<
  cutlass::gemm::GroupProblemShape<cute::tuple<int,int,int>>,
  cutlass::gemm::collective::CollectiveMma<
    cutlass::gemm::MainloopXeL1StagedGroup<2, cutlass::gemm::KernelXePtrArrayCooperative>, ...>,
  cutlass::epilogue::collective::CollectiveEpilogue<cutlass::epilogue::IntelXeGenericGroup, ...
    cutlass::epilogue::fusion::FusionCallbacks<cutlass::epilogue::IntelXeGenericGroup,
      cutlass::epilogue::fusion::LinearCombination<float,float,float,float,...>, ...>, ...>,
  cutlass::gemm::GroupScheduler, void>
```

### 3.5 Running the example — it PASSES per-group alpha

```bash
/tmp/bmg04 --groups=4 --iterations=0   # no --alpha/--beta => random per-group values
# Disposition: Passed

/tmp/bmg04 --groups=3 --m=16 --n=64 --k=16 --iterations=0
# Disposition: Passed
```

So the identical kernel type, with per-group alpha via `alpha_ptr_array`, **passes** — even at
small sizes. This definitively kills the "L=0 defeats per-group alpha" theory. The mechanism
works; **our wiring must differ somehow.**

(Note: `--m=1` produced no output — likely a separate alignment/degenerate-M issue in the
example harness, not relevant to the alpha question.)

### 3.6 Instrumenting the CUTLASS fusion with device printf

To see what `update_scalar` actually receives, patched the shared header **in place** (backup
saved first):

```bash
CUTLASS=_deps/repo-cutlass-sycl-src
F=${CUTLASS}/include/cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp
cp $F /tmp/orig_sm90_visitor.hpp     # BACKUP
```

**Gotcha #1 — printf namespace.** CUTLASS defines a macro
`#define printf sycl::ext::oneapi::experimental::printf` in `cute/util/print.hpp` and
`cute/numeric/numeric_types.hpp`. So:
- Writing `sycl::ext::oneapi::experimental::printf(...)` explicitly → the inner `printf` token
  gets macro-expanded again → garbled `sycl::ext::...::sycl::ext::...` → compile error
  `no member named 'sycl' in namespace 'sycl::ext::oneapi::experimental'`.
- Writing `::sycl::ext::...` had the same problem for the same reason.
- **Fix:** just call bare `printf(...)` and let the CUTLASS macro do the work.

**Gotcha #2 — format string must live in constant address space** for OpenCL/SYCL device
printf:
```cpp
static const __attribute__((opencl_constant)) char fmt[] = "....\n";
printf(fmt, ...);
```

First instrumentation (entry of `update_scalar`, printing which branch is taken):
```cpp
CUTLASS_DEVICE void
update_scalar(int l_coord = 0) {
  int l_offset = l_coord * size<2>(params_ptr->dScalar[0]);
#if defined(__SYCL_DEVICE_ONLY__)
  {
    static const __attribute__((opencl_constant)) char fmt0[] =
        "[US_ENTRY] l_coord=%d arr=%d ptr=%d\n";
    printf(fmt0, (int)l_coord,
           (int)(params_ptr->scalar_ptr_arrays[0]!=nullptr),
           (int)(params_ptr->scalar_ptrs[0]!=nullptr));
  }
#endif
  if (params_ptr->scalar_ptr_arrays[0] != nullptr) { ... }
  ...
}
```

Recompiled the example against the patched header and ran:
```bash
/tmp/bmg04p --groups=2 --iterations=0 2>&1 | grep -i "US_ENTRY\|Disposition" | sort | uniq -c
```
Output:
```
      1 Disposition: Passed
 131071 [US_ENTRY] l_coord=0 arr=0 ptr=0
```

### 3.7 The surprising result

**This is the breakthrough and the current head-scratcher:**

In the **passing** example, on the device:
- `l_coord` is **always 0** (as the "L=0" reading predicted), AND
- `scalar_ptr_arrays[0]` is **NULL** (`arr=0`), AND
- `scalar_ptrs[0]` is **NULL** (`ptr=0`).

So the device takes neither pointer branch — it falls into the **literal fallback**
`scalar = params_ptr->scalars[0]`. Yet the example produces **correct per-group** results.

Extended the print to show `scalars[0]`:
```cpp
static const __attribute__((opencl_constant)) char fmt0[] =
    "[US_ENTRY] l_coord=%d BC=%d arrAddr=%ld scalars0=%f\n";
printf(fmt0, (int)l_coord, (int)BroadcastCount,
       (long)(params_ptr->scalar_ptr_arrays[0]),
       (double)(float)params_ptr->scalars[0]);
```
```bash
/tmp/bmg04p --groups=3 --iterations=0 2>&1 | grep -i "US_ENTRY\|Disposition" | sort | uniq -c
```
Output:
```
      1 Disposition: Passed
  43829 [US_ENTRY] l_coord=0 BC=1 arrAddr=0 scalars0=0.000000
  43552 [US_ENTRY] l_coord=0 BC=1 arrAddr=0 scalars0=1.000000
```

Interpretation (note: the ~87k prints are torn/interleaved across many threads, so exact
values are unreliable, but the shape is clear):
- `arrAddr = 0` on device — the `alpha_ptr_array` pointer that was set NON-null on the host is
  **NULL by the time the device fusion sees it.**
- `scalars[0]` takes **more than one value** across the run (0.0 and 1.0 seen), meaning the
  fusion's literal `scalars[0]` is being **rewritten per group** somewhere between host and
  device.

**Emerging (corrected) theory:** On the Xe grouped path, per-group alpha does **NOT** flow
through `l_coord` indexing into a live pointer array at all. Instead, the per-group scalar is
**resolved/baked into `scalars[0]`** during the per-group parameter rebuild that the kernel runs
on every `did_group_change`:
```cpp
if (did_group_change) {
  base_epilogue_params = CollectiveEpilogue::Base::to_underlying_arguments(
      problem_shape_MNKL,
      CollectiveEpilogue::to_base_arguments(params.epilogue, curr_group),  // <-- per group
      params.workspace);
}
```
`to_base_arguments(params.epilogue, curr_group)` is called with the actual `curr_group`. If the
host-side arguments or this rebuild path resolve `alpha_ptr_array[curr_group]` into a plain
`scalars[0]` for the group (nulling the array pointer in the process), then the example works
because the *rebuild* — not the fusion's `l_coord` — carries the group identity.

If that's the mechanism, then **our kernel must be failing to trigger or feed that per-group
resolution correctly** — e.g. our `alpha_ptr_array` values, or the way `make_device_ptrs`
constructs them, don't survive `to_base_arguments`/`to_underlying_arguments` the way the
example's `DeviceAllocation`-backed pointers do.

### 3.8 The decisive experiment (IN PROGRESS at time of writing)

The one apples-to-apples experiment not yet completed: **rebuild our own `.so` against the same
instrumented header** and run the tiny 3-segment probe (§3.3). This tells us whether our kernel:
- hits the `scalar_ptr_arrays[0] != nullptr` branch (pointer array survives) — then the bug is
  in our `l_coord`/array contents, OR
- falls into the `scalars[0]` fallback like the example — then the bug is that our per-group
  rebuild is resolving the *wrong* group's alpha (always group 0).

Instrumentation used for this (pointer-array branch only, low-noise):
```cpp
if (params_ptr->scalar_ptr_arrays[0] != nullptr) {
  scalar = *(params_ptr->scalar_ptr_arrays[0][l_coord]);
#if defined(__SYCL_DEVICE_ONLY__)
  {
    static const __attribute__((opencl_constant)) char fmt[] =
        "[ALPHA_PA] l_coord=%d scalar=%f\n";
    printf(fmt, (int)l_coord, (double)(float)scalar);
  }
#endif
}
```

Build command for the instrumented `.so` (this triggers a near-full rebuild because the header
is transitively included by fmha and other kernels):
```bash
source ~/intel/oneapi/setvars.sh
conda activate my_py312_env
cd /home/gta/frameworks.ai.pytorch.sglang/sgl-kernel-xpu-upstream
rm -f build/src/libsgl-ops-sycl-sgemm_lora_b_fwd_kernel_half_large.so \
      build/src/libsgl-ops-sycl-sgemm_lora_b_fwd_kernel_bf16_large.so
CMAKE_BUILD_PARALLEL_LEVEL=8 pip install -e . --no-build-isolation -v \
    > ~/Logs/sgemm_lora_b_fwd_instr.log 2>&1
```

**This build was in progress and then stopped by the user, who is running it themselves.**

---

## 4. Key Files & Their Roles

| File | Role |
|---|---|
| `src/sycl/SGEMMLoraBFwd.cpp` | Main API entry: validation, dtype/index casts, degenerate-K handling, dispatch. |
| `src/sycl/kernels/lora/device/sgemm_lora_b_fwd_runner.hpp` | Builds meta + pointer arrays, sets up `alpha_ptrs`/`c_ptrs`/`beta`, calls `args_from_options`, drives launcher. |
| `src/sycl/kernels/lora/common/group_gemm_types.hpp` | **SHARED with A-fwd.** `GroupGemmTypes<>` type bundle + `args_from_options()` (where alpha/beta wiring lives). Must not regress A-fwd. |
| `src/sycl/kernels/lora/common/grouped_gemm_meta.hpp` | **SHARED.** Device-side per-segment metadata build (problem sizes, strides, byte offsets, `meta.alpha[s]=scalings[weight_indices[s]]`) + `make_device_ptrs()`. |
| `src/sycl/kernels/lora/device/sgemm_lora_b_fwd_types.hpp` | Tile-option tag `LoraBFwdTileLarge` (256×256×32, 8×4×1 threads, ColumnMajor B, 2 stages). |
| `src/sycl/kernels/lora/device/sgemm_lora_b_fwd_dispatch.hpp` | Per-(dtype,tile) launch symbol declarations. |
| `src/sycl/sgemm_lora_b_fwd_kernel.cpp.in` | TU template (configure_file'd per dtype/tile). |
| `src/SGEMMLoraBFwdXe20.cmake` | Generates the per-(dtype,tile) TUs, appends to `ATen_XPU_SYCL_XE20`. |
| `tests/test_sgemm_lora_b_fwd.py` | Test suite (mirrors A-fwd). 25 fail / 31 pass currently. |

### 4.1 CUTLASS reference files (read-only, in `build/_deps/repo-cutlass-sycl-src/`)

| File | What it showed |
|---|---|
| `examples/04_bmg_grouped_gemm/04_bmg_grouped_gemm.cpp` | Reference per-group alpha usage (identical mechanism to ours). PASSES. |
| `include/cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp` | `Sm90ScalarBroadcastPtrArray::update_scalar()` — the 3-way scalar source; where we instrumented. |
| `include/cutlass/epilogue/fusion/xe_callbacks.hpp` (line ~885) | `FusionCallbacks<IntelXeGenericGroup, LinearCombination,...>` — binds `Sm90LinearCombinationPtrArray`. The `Arguments` struct with `alpha_ptr_array`/`dAlpha`. |
| `include/cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp` (line ~205) | `Sm90LinearCombinationPtrArray` definition: alpha & beta as `Sm90ScalarBroadcastPtrArray<ElementScalar, Stride<_0,_0,int64_t>>`. |
| `include/cutlass/epilogue/collective/xe_array_epilogue.hpp` | Grouped epilogue: `Arguments{thread, ptr_C, dC, ptr_D, dD}`; `to_base_arguments(args, idx)` forwards `args.thread` UNCHANGED + indexes `ptr_C[idx]/ptr_D[idx]`. |
| `include/cutlass/epilogue/collective/xe_epilogue.hpp` (line ~299) | `auto batch_idx = get<3>(tile_coord_mnkl)` used to slice `params.mD(_,_,batch_idx)` / `params.mC`. |
| `include/cutlass/gemm/kernel/xe_gemm_array_cooperative.hpp` (line ~288) | `auto tile_coord = make_coord(m_coord, n_coord, _, 0)` — L hardcoded 0; per-group rebuild via `to_base_arguments(..., curr_group)` on `did_group_change`. |
| `include/cutlass/epilogue/thread/linear_combination.h` (line ~173) | A *different* (non-visitor) epilogue op that DOES index `alpha_ptr_array[group_idx]` directly — shows the "resolve per group_idx" pattern exists elsewhere in CUTLASS. |

---

## 5. Environment & Build Reference

### 5.1 Environment setup (always needed before building)
```bash
source ~/intel/oneapi/setvars.sh          # oneAPI (icpx/icx, SPIR-V tooling)
conda activate my_py312_env               # Python 3.12 env with torch/xpu
# sanity:
which python   # /home/gta/miniforge3/envs/my_py312_env/bin/python
which icpx     # /home/gta/intel/oneapi/compiler/2025.3/bin/icpx
```

### 5.2 Full project build (editable install)
```bash
cd /home/gta/frameworks.ai.pytorch.sglang/sgl-kernel-xpu-upstream
CMAKE_BUILD_PARALLEL_LEVEL=8 pip install -e . --no-build-isolation -v \
    > ~/Logs/sgemm_lora_b_fwd_build.log 2>&1
```
- Editing **headers** (runner/types/meta) recompiles the generated TUs that include them.
- Editing a **widely-included CUTLASS header** (like `sm90_visitor_load_tma_warpspecialized.hpp`)
  triggers a near-full rebuild (fmha + all grouped kernels include it transitively) — slow.
- Editing **cmake/CMakeLists** triggers a CMake reconfigure; if it errors with
  "cannot find the XPU runtime libraries", you forgot to `source ~/intel/oneapi/setvars.sh`.

### 5.3 Removing only the b_fwd artifacts for a clean b_fwd rebuild
```bash
cd /home/gta/frameworks.ai.pytorch.sglang/sgl-kernel-xpu-upstream
find build -iname "*b_fwd*"     # inspect first
rm -rf build/src/libsgl-ops-sycl-sgemm_lora_b_fwd_kernel_bf16_large.so \
       build/src/libsgl-ops-sycl-sgemm_lora_b_fwd_kernel_half_large.so \
       build/src/CMakeFiles/sgl-ops-sycl-sgemm_lora_b_fwd_kernel_half_large.dir \
       build/src/CMakeFiles/sgl-ops-sycl-sgemm_lora_b_fwd_kernel_bf16_large.dir \
       build/src/generated/sgemm_lora_b_fwd \
       build/.cmake/api/v1/reply/target-sgl-ops-sycl-sgemm_lora_b_fwd_kernel_*.json
```

### 5.4 Compiling a standalone CUTLASS example (for isolated experiments)
```bash
source ~/intel/oneapi/setvars.sh
conda activate my_py312_env
cd build
CUTLASS=_deps/repo-cutlass-sycl-src
icpx -fsycl -std=c++17 -fsycl-targets=spir64_gen \
  -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate \
  -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET -DCUTLASS_SYCL_SWITCH_WG=0 \
  -I${CUTLASS}/include -I${CUTLASS}/tools/util/include -I${CUTLASS}/applications -I${CUTLASS}/examples/common \
  ${CUTLASS}/examples/04_bmg_grouped_gemm/04_bmg_grouped_gemm.cpp \
  -Xs "-device bmg" \
  -o /tmp/bmg04
# run:
/tmp/bmg04 --groups=4 --iterations=0        # random per-group alpha, verify on
/tmp/bmg04 --groups=3 --m=16 --n=64 --k=16 --iterations=0
```

### 5.5 Running the tests
```bash
source ~/intel/oneapi/setvars.sh
conda activate my_py312_env
cd /home/gta/frameworks.ai.pytorch.sglang/sgl-kernel-xpu-upstream
python -m pytest tests/test_sgemm_lora_b_fwd.py -q 2>&1 | tail -25
```

---

## 6. Device printf Cheat-Sheet (SYCL/CUTLASS)

Lessons learned instrumenting CUTLASS device code:

1. **Bare `printf` only.** CUTLASS `#define printf sycl::ext::oneapi::experimental::printf`
   (in `cute/util/print.hpp`). Do NOT write the qualified name yourself — the macro will
   double-expand and break. Just call `printf(...)`.
2. **Format string in constant address space:**
   ```cpp
   static const __attribute__((opencl_constant)) char fmt[] = "val=%d\n";
   printf(fmt, x);
   ```
3. **Guard with `#if defined(__SYCL_DEVICE_ONLY__)`** so host compilation is unaffected.
4. **Floats:** promote to `double` in the varargs: `printf(fmt, (double)(float)scalar)` with `%f`.
5. **Volume:** the epilogue runs per output element per thread → tens of thousands of prints.
   Prints interleave/tear across threads; use `sort | uniq -c` to aggregate and treat exact
   values cautiously. Prefer gating on a single coordinate (e.g. one work-tile) or a single
   branch to reduce noise.

---

## 7. Cleanup / State To Restore

- **Patched header (INSTRUMENTED):**
  `build/_deps/repo-cutlass-sycl-src/include/cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp`
- **Clean backup:** `/tmp/orig_sm90_visitor.hpp`
- **Restore before any production build:**
  ```bash
  cp /tmp/orig_sm90_visitor.hpp \
     build/_deps/repo-cutlass-sycl-src/include/cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp
  ```
- Standalone example binaries built to `/tmp/bmg04` (clean) and `/tmp/bmg04p` (instrumented).
- No source files under `src/` were modified during the debugging in this session (the one
  proposed edit to the runner's doc-comment was rejected). The runner still uses the
  `alpha_ptr_array` approach.

---

## 8. Current Best Understanding & Next Steps

### 8.1 What we know for certain
1. Our kernel applies `scalings[0]` to **every** group (empirically confirmed, §3.3).
2. The upstream example, using the **identical kernel type and the identical `alpha_ptr_array`
   mechanism**, applies per-group alpha **correctly** (empirically confirmed, §3.5).
3. On the device, in the passing example, the fusion sees `l_coord=0`, `alpha_ptr_array=NULL`,
   and reads a per-group-varying `scalars[0]` (§3.6-3.7).

### 8.2 The leading hypothesis
Per-group alpha on the Xe grouped path is resolved by the **per-group parameter rebuild**
(`to_base_arguments(params.epilogue, curr_group)` → `to_underlying_arguments`) that runs on each
`did_group_change`, NOT by the fusion's `l_coord` indexing. The example's pointer array gets
resolved into a per-group `scalars[0]`; our kernel is (somehow) always resolving group 0.

### 8.3 The decisive next experiment
Rebuild **our `.so`** against the instrumented header (§3.8 build command) and run the §3.3
probe. Two possible outcomes:
- **We hit `scalar_ptr_arrays[0] != nullptr`** → the array survives to the device; the bug is
  in our array *contents* or the `l_coord` we present. Compare our `alpha_ptrs` construction
  (`make_device_ptrs(meta.alpha, arange*sizeof(float))`) against the example's
  `DeviceAllocation<ElementAccumulator*>` of pointers into a value block.
- **We fall into `scalars[0]` fallback** (like the example) but with the wrong (group-0) value →
  the per-group rebuild is being fed group 0 for every group; investigate how `curr_group` /
  the problem-shape / the argument layout reaches `to_base_arguments` in our setup vs the
  example's.

### 8.4 Candidate fixes (to evaluate AFTER the decisive experiment)
1. **Match the example's argument construction exactly** — build the pointer array the same way
   (pointers into a contiguous per-group value block via a device allocation), if our
   `make_device_ptrs` layout differs materially.
2. **Verify `problem_sizes` / group count / scheduler args** are wired identically to the example
   (a mismatch could make the scheduler assign all work to group 0's params).
3. **Only as a last resort** — fold `scalings[l]` into the `A` operand on device (pre-scale
   `input_x` rows per segment) and run alpha=1. Mathematically exact
   (`scalings[l]*(x@Bᵀ) == (scalings[l]*x)@Bᵀ`), keeps the residual `beta*C` path intact, and
   sidesteps the fusion entirely. This was proposed earlier and deferred; revisit only if the
   fusion path can't be made to work.

### 8.5 Constraints to respect throughout
- **Do NOT regress the working A-fwd kernel.** `group_gemm_types.hpp` and `grouped_gemm_meta.hpp`
  are shared. A-fwd calls `args_from_options()` WITHOUT `alpha_ptr_array` (default `nullopt`) →
  scalar-broadcast branch → must remain untouched in behavior.
- A-fwd's call site (for reference):
  ```cpp
  args_from_options<Types>(meta.problem_sizes, a_ptrs, b_ptrs, /*c=*/d_ptrs, d_ptrs,
      meta.stride_A, meta.stride_B, /*stride_C=*/meta.stride_D, meta.stride_D,
      num_segments, /*alpha=*/1.0f, /*beta=*/0.0f);   // no alpha_ptr_array
  ```

---

## 9. Appendix — Chronological Command Log (abridged)

```bash
# --- Static reading of the grouped kernel (initial, flawed hypothesis) ---
sed -n '250,340p' build/_deps/repo-cutlass-sycl-src/include/cutlass/gemm/kernel/xe_gemm_array_cooperative.hpp
grep -n "tile_coord\|L_idx\|curr_group\|make_coord\|batch_idx\|l_coord" \
  build/_deps/repo-cutlass-sycl-src/include/cutlass/gemm/kernel/xe_gemm_array_cooperative.hpp

# --- Confirm example uses the same mechanism ---
grep -n "alpha\|Alpha\|dAlpha\|scalar\|block_alpha" \
  build/_deps/repo-cutlass-sycl-src/examples/04_bmg_grouped_gemm/04_bmg_grouped_gemm.cpp
grep -n "GEMMDispatchPolicy\|MainloopXe\|EpilogueDispatchPolicy\|IntelXe" \
  build/_deps/repo-cutlass-sycl-src/examples/04_bmg_grouped_gemm/04_bmg_grouped_gemm.cpp

# --- Read the fusion scalar-broadcast source ---
grep -rn "Sm90ScalarBroadcastPtrArray" build/_deps/repo-cutlass-sycl-src/include/cutlass/epilogue/fusion/
sed -n '813,1000p' build/_deps/repo-cutlass-sycl-src/include/cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp
sed -n '150,245p' build/_deps/repo-cutlass-sycl-src/include/cutlass/epilogue/collective/xe_array_epilogue.hpp
sed -n '160,180p' build/_deps/repo-cutlass-sycl-src/include/cutlass/epilogue/collective/xe_epilogue.hpp

# --- Clean rebuild of b_fwd artifacts; confirm 25 fail / 31 pass ---
find build -iname "*b_fwd*"
rm -rf build/src/libsgl-ops-sycl-sgemm_lora_b_fwd_kernel_*_large.so \
       build/src/CMakeFiles/sgl-ops-sycl-sgemm_lora_b_fwd_kernel_*_large.dir \
       build/src/generated/sgemm_lora_b_fwd \
       build/.cmake/api/v1/reply/target-sgl-ops-sycl-sgemm_lora_b_fwd_kernel_*.json
source ~/intel/oneapi/setvars.sh; conda activate my_py312_env
CMAKE_BUILD_PARALLEL_LEVEL=8 pip install -e . --no-build-isolation -v > ~/Logs/sgemm_lora_b_fwd_build.log 2>&1
python -m pytest tests/test_sgemm_lora_b_fwd.py -q 2>&1 | tail -25   # 25 failed, 31 passed

# --- Empirical probe: our kernel applies scalings[0] to every group ---
python - <<'PY'   # (all-ones probe, see §3.3)
...
PY

# --- Build the upstream example standalone ---
find _deps/repo-cutlass-sycl-src -name "sycl_common.hpp"
grep -m2 "split_barrier\|Xspirv-translator" build.ninja src/build.ninja
icpx -fsycl -std=c++17 -fsycl-targets=spir64_gen \
  -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate \
  -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET -DCUTLASS_SYCL_SWITCH_WG=0 \
  -I_deps/repo-cutlass-sycl-src/include -I_deps/repo-cutlass-sycl-src/tools/util/include \
  -I_deps/repo-cutlass-sycl-src/applications -I_deps/repo-cutlass-sycl-src/examples/common \
  _deps/repo-cutlass-sycl-src/examples/04_bmg_grouped_gemm/04_bmg_grouped_gemm.cpp \
  -Xs "-device bmg" -o /tmp/bmg04
/tmp/bmg04 --groups=4 --iterations=0                       # Disposition: Passed
/tmp/bmg04 --groups=3 --m=16 --n=64 --k=16 --iterations=0  # Disposition: Passed

# --- Instrument the fusion (patch in place, backup first) ---
cp _deps/repo-cutlass-sycl-src/include/cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp /tmp/orig_sm90_visitor.hpp
# ... add device printf to update_scalar (bare printf + opencl_constant fmt) ...
# recompile example against patched header -> /tmp/bmg04p
/tmp/bmg04p --groups=2 --iterations=0 2>&1 | grep -i "US_ENTRY\|Disposition" | sort | uniq -c
#   1 Disposition: Passed
#   131071 [US_ENTRY] l_coord=0 arr=0 ptr=0       <-- device: l=0, both pointers NULL, still correct
/tmp/bmg04p --groups=3 --iterations=0 2>&1 | grep -i "US_ENTRY\|Disposition" | sort | uniq -c
#   [US_ENTRY] l_coord=0 BC=1 arrAddr=0 scalars0=0.000000
#   [US_ENTRY] l_coord=0 BC=1 arrAddr=0 scalars0=1.000000   <-- scalars[0] varies per group

# --- (IN PROGRESS) rebuild OUR .so against instrumented header for apples-to-apples ---
rm -f build/src/libsgl-ops-sycl-sgemm_lora_b_fwd_kernel_*_large.so
CMAKE_BUILD_PARALLEL_LEVEL=8 pip install -e . --no-build-isolation -v > ~/Logs/sgemm_lora_b_fwd_instr.log 2>&1
```

---

*(Historical: §8.3 was the "decisive experiment" as understood mid-investigation. It was run —
see §10, which supersedes the open questions in §8.)*

---

## 10. RESOLUTION — The Decisive Experiments (root cause confirmed)

This section was written after the instrumented `.so` build finished. It resolves the paradox
from §3 and overturns the mid-investigation conclusion in §3.5/§8.

### 10.1 Experiment A — our instrumented `.so`, 3-segment all-ones probe

Ran the §3.3 probe against our kernel built with the `[ALPHA_PA]` device printf in the
pointer-array branch of `update_scalar`:

```bash
source ~/intel/oneapi/setvars.sh; conda activate my_py312_env
cd /home/gta/frameworks.ai.pytorch.sglang/sgl-kernel-xpu-upstream
python /tmp/probe_bfwd.py 2>&1 | grep -i "ALPHA_PA\|implied\|expected" | sort | uniq -c
```
Probe (`/tmp/probe_bfwd.py`): `scalings=[0.5, 2.0, 4.0]`, all-ones `input_x`/`weights`,
`max_rank=8`, one token per segment.

**Result:**
```
   1536 [ALPHA_PA] l_coord=0 scalar=0.500000
      1 implied alpha per segment: [0.5, 0.5, 0.5]
      1 expected alpha per segment: [0.5, 2.0, 4.0]
```

**Reading:** Our kernel *does* enter the pointer-array branch (`scalar_ptr_arrays[0] != nullptr`),
but `l_coord` is **always 0**, so it always reads `scalar_ptr_arrays[0][0] = scalings[0] = 0.5`
for **every** group. This is the bug, observed directly on-device.

### 10.2 Experiment B — the example's default never took the per-group branch

Re-reading the example's option parsing (the detail missed in §3.5):

```cpp
// 04_bmg_grouped_gemm.cpp
Options() : ... alpha(FLT_MAX), beta(FLT_MAX), ...   // struct default
void parse(...) {
  cmd.get_cmd_line_argument("alpha", alpha, 1.f);    // <-- CLI default is 1.f, NOT FLT_MAX
  cmd.get_cmd_line_argument("beta",  beta,  0.f);    // <-- CLI default is 0.f
}
...
// args_from_options():
if (options.alpha != FLT_MAX && options.beta != FLT_MAX) {   // TRUE for default run!
  // SCALAR broadcast branch — alpha_ptr_array = nullptr
} else {
  // per-group POINTER-ARRAY branch — never reached on a default `./bmg04` run
}
```

**`options.parse()` overwrites the `FLT_MAX` struct-defaults with `1.f`/`0.f`.** So *every*
plain `./bmg04 --groups=N` invocation I ran in §3.5 took the **scalar** branch. It never
exercised `alpha_ptr_array` at all — which is exactly why the §3.6 instrumentation showed
`arr=0` (NULL) on device and why there were zero `[ALPHA_PA]` prints. **The example "passing"
told us nothing about the per-group path.** This was the core mistake of the investigation.

Confirmed empirically — instrumented example, default alpha, zero pointer-array hits:
```bash
/tmp/bmg04p --groups=3 --iterations=0 2>&1 | grep -c "ALPHA_PA"   # -> 0
# Disposition: Passed  (but via the scalar branch, alpha=beta=1/0 for all groups)
```

### 10.3 Experiment C — force the per-group branch with DISTINCT alphas → example FAILS

Patched the example to (1) force the per-group branch and (2) use unmistakably distinct
per-group alphas, then rebuilt and ran with verification on:

```bash
cd build/_deps/repo-cutlass-sycl-src/examples/04_bmg_grouped_gemm
cp 04_bmg_grouped_gemm.cpp /tmp/ex_orig.cpp   # backup

# Patch 1 — force per-group branch:
#   if (options.alpha != FLT_MAX && options.beta != FLT_MAX) {
# becomes
#   if (false /*FORCED per-group branch for debug*/) {

# Patch 2 — distinct per-group alphas, beta=0:
#   alpha_host.push_back((options.alpha==FLT_MAX)? rand()%5+1 : options.alpha);
#   beta_host.push_back ((options.beta ==FLT_MAX)? rand()%5   : options.beta );
# becomes
#   alpha_host.push_back(static_cast<ElementAccumulator>((i + 1) * 10)); // 10, 20, 30, ...
#   beta_host.push_back (static_cast<ElementAccumulator>(0));
```

Also extended the header print to dump the array base + resolved element pointer to
disambiguate the alpha vs beta broadcast objects:
```cpp
static const __attribute__((opencl_constant)) char fmt[] =
    "[ALPHA_PA] l_coord=%d base=%ld elem=%ld scalar=%f\n";
printf(fmt, (int)l_coord,
       (long)(params_ptr->scalar_ptr_arrays[0]),
       (long)(params_ptr->scalar_ptr_arrays[0][l_coord]),
       (double)(float)scalar);
```

Rebuild + run:
```bash
icpx -fsycl -std=c++17 -fsycl-targets=spir64_gen \
  -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate \
  -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET -DCUTLASS_SYCL_SWITCH_WG=0 \
  -I${CUTLASS}/include -I${CUTLASS}/tools/util/include -I${CUTLASS}/applications -I${CUTLASS}/examples/common \
  ${CUTLASS}/examples/04_bmg_grouped_gemm/04_bmg_grouped_gemm.cpp \
  -Xs "-device bmg" -o /tmp/bmg04d
/tmp/bmg04d --groups=3 --m=256 --n=256 --k=32 --iterations=0 2>&1 | grep -i "ALPHA_PA\|Disposition"
```

**Result:**
```
   1536 [ALPHA_PA] l_coord=0 base=-23453394397184 scalar=0.000000   <- beta broadcast object
   1536 [ALPHA_PA] l_coord=0 base=-23453394397696 scalar=10.000000  <- alpha broadcast object
      1 Disposition: Failed
```

**The upstream example FAILS with distinct per-group alphas.** Every group reads
`alpha = 10.0` (= group 0's alpha `(0+1)*10`), exactly like our kernel reads `scalings[0]`. The
two distinct `base` values are the **alpha** vs **beta** `Sm90ScalarBroadcastPtrArray` objects
(both reuse `update_scalar`), NOT two groups — `l_coord=0` for both, always.

(Earlier §3.7's "`scalars[0]` = 0.0 and 1.0" was likewise the alpha object vs the beta object,
not per-group variation. That misread fed the wrong conclusion.)

### 10.4 Root cause (confirmed)

Per-group alpha/beta through `alpha_ptr_array` is **structurally non-functional** on the Xe
`KernelXePtrArrayCooperative` grouped kernel:

1. `xe_gemm_array_cooperative.hpp:288` — `auto tile_coord = make_coord(m_coord, n_coord, _, 0);`
   The L (batch/group) coordinate handed to the epilogue is **hardcoded to 0**.
2. `xe_array_epilogue.hpp:235-237` — `to_base_arguments(args, idx)` forwards `args.thread`
   (the fusion args, including the `alpha_ptr_array` **base** pointer and `dAlpha`) **unchanged**
   for every group; only `ptr_C[idx]` / `ptr_D[idx]` are indexed by the group `idx`.
3. `sm90_visitor_load_tma_warpspecialized.hpp` — `update_scalar(l_coord)` computes
   `scalar = *(scalar_ptr_arrays[0][l_coord])`. With `l_coord ≡ 0` and the base pointer never
   advanced per group, this is **always** `scalar_ptr_arrays[0][0]` = group 0's value.

So the group identity reaches the *mainloop/data pointers* (via `ptr_A/B/C/D[idx]` and the
per-group `to_underlying_arguments` rebuild) but **never reaches the epilogue scalar fusion**.
The `dAlpha = {_0,_0,1}` stride is meaningless here because the L coordinate that would use it
is pinned to 0.

**This is a genuine limitation of the current cutlass-sycl grouped epilogue, not a bug in our
wiring.** Our code did exactly what the (broken) example does.

### 10.5 The fix — fold `scalings[l]` into the operand (pre-scale)

Since the epilogue cannot deliver per-group alpha, apply the per-segment scaling **inside the
GEMM operands**, using the mathematical identity:
```
scalings[l] * (input_x @ weights[l]^T)  ==  (scalings[l] * input_x) @ weights[l]^T
                                        ==  input_x @ (scalings[l] * weights[l])^T
```
Recommended: **pre-scale the `A` operand rows per segment** (scale `input_x` by
`scalings[seg]`), then run the grouped GEMM with plain `alpha = 1`, keeping the `beta * C`
residual path (which uses per-group `ptr_C[idx]` — that part *does* work). This:
- is exact (fp32 accumulate; scaling folded before the MMA),
- needs no epilogue per-group scalar,
- keeps the residual add intact,
- does not touch the shared A-fwd path (A-fwd already runs `alpha=1`, no `alpha_ptr_array`).

Pre-scaling `A` (not `B`) is preferred because `input_x` is typically much smaller than the
stacked weight tensor, so the extra scaling pass is cheaper and touches fewer bytes. It also
sidesteps the ColumnMajor-B / VNNI packing of the weights.

Alternatively pre-scale a copy of the relevant `weights` slices — avoid if it means duplicating
the (larger) weight tensor.

This is precisely the approach proposed earlier (§3.1) and deferred; the §10 experiments prove
it is the correct path, not a workaround around a mis-wiring.

### 10.6 What to change (implementation sketch)

- In `sgemm_lora_b_fwd_runner.hpp`: drop the `alpha_ptrs` / `alpha_ptr_array` argument; instead
  produce a scaled input `x_scaled` where each segment `s`'s rows are multiplied by
  `scalings[weight_indices[s]]`. A small SYCL kernel keyed by `seg_indptr` (or reuse the meta
  build) can write `x_scaled[row] = scalings[seg(row)] * input_x[row]`. Call
  `args_from_options<Types>(..., /*alpha=*/1.0f, /*beta=*/beta)` with **no** `alpha_ptr_array`.
- `group_gemm_types.hpp::args_from_options` — the `alpha_ptr_array` branch can stay (harmless,
  unused) or be removed; A-fwd is unaffected either way since it never passes it.
- `grouped_gemm_meta.hpp` — `meta.alpha` is no longer consumed by the epilogue; keep it if the
  pre-scale kernel reads it, else drop.
- Tests: the existing `_reference_sgemm_lora_b_fwd` (which computes
  `scalings[l]*(x@w.T)+base_output` in fp32) is the correct oracle; multi-segment cases should
  pass once the pre-scale lands.

### 10.7 Cleanup performed at end of this session

- **Example restored** to pristine: `cp /tmp/ex_orig.cpp <.../04_bmg_grouped_gemm.cpp>`
  (verified 0 `FORCED` markers remain).
- **CUTLASS header still instrumented** with the `[ALPHA_PA] ... base=%ld elem=%ld` printf.
  Restore before any production build:
  ```bash
  cp /tmp/orig_sm90_visitor.hpp \
     build/_deps/repo-cutlass-sycl-src/include/cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp
  ```
- Standalone binaries: `/tmp/bmg04` (clean), `/tmp/bmg04p` (instrumented, default alpha),
  `/tmp/bmg04pg` (per-group forced, random alphas), `/tmp/bmg04d` (per-group forced, distinct
  alphas 10/20/30). Probe script: `/tmp/probe_bfwd.py`.

### 10.8 Lesson (for the educational record)

The investigation's biggest time sink was trusting a "Passed" from a reference example without
verifying **which code path** it exercised. Two compounding traps:
1. A CLI default (`--alpha=1`) silently routed the example around the very feature under test.
2. Random test data (`rand()%5+1`) can collide and mask a "reads index 0 for all groups" bug —
   with 3 groups drawing from {1..5} there's a real chance ≥2 groups share a value, and even
   when they differ, a torn/aggregated printf made the single distinct value look like variation.

**Takeaways:** (a) when a reference "passes," confirm it actually runs the path you care about —
instrument or force the branch; (b) use *distinct, non-random, memorable* values (10/20/30) when
testing per-index selection so an off-by-index bug is unmistakable; (c) on-device `printf` output
across thousands of threads is torn — aggregate with `sort | uniq -c` and print
disambiguating identifiers (here, the `base` pointer distinguished the alpha vs beta objects).

---

*End of investigation log. Root cause confirmed (§10.4); fix path decided (§10.5). Next action:
implement the pre-scale in `sgemm_lora_b_fwd_runner.hpp`, restore the clean header, rebuild, and
confirm 56/56 tests pass without regressing A-fwd.*
