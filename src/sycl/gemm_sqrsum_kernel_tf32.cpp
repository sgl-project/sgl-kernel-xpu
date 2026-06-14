/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
/*! \file
    \brief tf32 instantiation of the GEMM + Square Sum kernel.

    This is the production mhc_pre path: bf16(A) x fp32(B) -> fp32, run through a
    tf32 x tf32 -> fp32 DPAS atom (A widened to fp32, B taken as-is, both
    reinterpreted to tf32 at load). Compiled as its own translation unit so the
    heavy CUTLASS instantiation builds/relinks independently of the other dtypes.
*/
#define SYCL_INTEL_TARGET 20

#include "sycl/kernels/gemm_sqrsum/device/gemm_sqrsum_types.hpp"
#include "sycl/kernels/gemm_sqrsum/device/gemm_sqrsum_dispatch.hpp"

namespace gemm_sqrsum {

void launch_gemm_sqrsum_tf32_256x256x16(
    at::Tensor& C,
    at::Tensor& sqrsum,
    const at::Tensor& A,
    const at::Tensor& B) {
  // Tile 64 x 32 x 16. TileM=64 = 8(atom M) * 8(SG_M) * 1 iter -> SG layout
  // stays 8x2 (16 subgroups, 256 work-items); only the per-subgroup M-iteration
  // count drops 4->1 vs the old TileM=256. Smaller M-tile => 4x more workgroups
  // (grid_m = ceil(M/64)), which fills the GPU for medium/large M since N=24
  // gives grid_n=1 and all parallelism must come from grid_m. Also shrinks the
  // per-work-item accumulator (no register spill).
  runGemmSqrSum<cutlass::tfloat32_t, TileSizeOption<64, 32, 16>>(C, sqrsum, A, B);
}

}  // namespace gemm_sqrsum
