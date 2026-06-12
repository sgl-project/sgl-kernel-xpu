# GEMM + Square Sum Kernel

This kernel performs two operations in a single fused kernel:

1. **GEMM (Matrix Multiplication)**: `C = A @ B`
2. **Row-wise Square Sum**: `sqrsum[i] = sum(A[i,:]^2)` for each row i in A

## Motivation

This kernel is derived from the MLA (Multi-Latent Attention) decode kernel but simplified to demonstrate the core concepts:
- Tiled GEMM computation using CUTLASS/CUTE primitives
- Simultaneous reduction operations during the mainloop
- Efficient memory access patterns with prefetching

## Operations

Given:
- `A`: [M, K] matrix
- `B`: [K, N] matrix

Computes:
- `C`: [M, N] = A @ B
- `sqrsum`: [M] where `sqrsum[i] = sum(A[i,0]^2 + A[i,1]^2 + ... + A[i,K-1]^2)`

## Mainloop Structure

The mainloop in `collective/xe_gemm_sqrsum_mainloop.hpp` performs:

### Loop over K tiles:
```
for each K tile:
  1. Prefetch next A and B tiles
  2. Load current A and B tiles to registers
  3. Reorder data for MMA layout
  4. Compute: C += A @ B  (matrix multiply accumulate)
  5. Compute: sqrsum += sum(A[i,:]^2)  (row-wise square and accumulate)
```

### Key Points:

1. **Tiled Computation**: Processes the GEMM in tiles (e.g., 256x256x16)
2. **Prefetching**: Issues prefetch for tile K+1 while computing tile K
3. **Fused Operations**: Square sum is computed on A tiles as they're loaded, avoiding extra memory passes
4. **Register Blocking**: All intermediate values stay in registers

## File Structure

```
gemm_sqrsum/
├── collective/
│   └── xe_gemm_sqrsum_mainloop.hpp  # Core mainloop logic
├── kernel/
│   └── xe_gemm_sqrsum_kernel.hpp    # Kernel wrapper and launch infrastructure
├── device/
│   └── gemm_sqrsum_example.hpp      # Example configuration and launch function
└── README.md                         # This file
```

## Differences from MLA Kernel

### MLA Kernel:
- Dual Q/K paths: `Score = Q_nope @ KV_c^T + Q_pe @ K_pe^T`
- Online softmax computation
- Paged KV cache access
- Three major operations: Q@K^T, softmax, P@V

### This Kernel:
- Single path: `C = A @ B`
- Row-wise reduction on A: `sqrsum[i] = sum(A[i,:]^2)`
- Direct memory access (no paging)
- Two operations: GEMM + reduction

## Usage Example

```cpp
#include "device/gemm_sqrsum_example.hpp"

sycl::queue q;
int M = 1024, K = 512, N = 2048;

// Allocate device memory
sycl::half* A = sycl::malloc_device<sycl::half>(M * K, q);
sycl::half* B = sycl::malloc_device<sycl::half>(K * N, q);
sycl::half* C = sycl::malloc_device<sycl::half>(M * N, q);
sycl::half* sqrsum = sycl::malloc_device<sycl::half>(M, q);

// Initialize A and B...

// Launch kernel
cutlass::gemm_sqrsum::launch_gemm_sqrsum(q, M, K, N, A, B, C, sqrsum);
q.wait();

// Results:
// - C contains A @ B
// - sqrsum[i] contains sum(A[i,:]^2) for each row i
```

## Next Steps

To make this kernel production-ready:

1. **Add Epilogue**: Proper output writing logic in `xe_gemm_sqrsum_kernel.hpp`
2. **Add Tests**: Unit tests for correctness
3. **Optimize**: Tune tile sizes for specific hardware
4. **Handle Edge Cases**: Remainder handling for non-tile-aligned dimensions
5. **Cross-SG Reduction**: Proper subgroup reduction for square sum accumulation
6. **Integration**: Add CMake build configuration and PyTorch binding

## Comparison with Original Question

> "can we create another kernel where we only do gemm and softmax"

This kernel shows the pattern for creating simplified kernels from the MLA kernel:
- Removed MLA-specific dual paths (Q_nope/Q_pe)
- Removed softmax and replaced with simpler row-wise square sum
- Kept the core GEMM structure and tiling approach
- Demonstrates how to add custom reductions alongside GEMM

To create a GEMM+Softmax kernel instead:
- Replace the square sum accumulation with the softmax logic from MLA kernel
- Use the online softmax algorithm (lines 462-503 in `xe_mla_mainloop.hpp`)
- Apply softmax to the GEMM output C instead of A's square sum
