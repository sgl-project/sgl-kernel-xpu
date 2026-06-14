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
  runGemmSqrSum<cutlass::tfloat32_t, TileSizeOption<256, 32, 16>>(C, sqrsum, A, B);
}

}  // namespace gemm_sqrsum
