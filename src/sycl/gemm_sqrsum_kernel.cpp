/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
/*! \file
    \brief Instantiation of the GEMM + Square Sum kernel.

    The mhc_pre path: bf16(A) x fp32(B) -> fp32, run through a tf32 x tf32 ->
    fp32 DPAS atom (A widened to fp32, B taken as-is, both reinterpreted to tf32
    at load). Compiled as its own translation unit so the heavy CUTLASS template
    instantiation lives in one place, separate from the light host dispatcher
    (gemm_sqrsum.cpp) which only sees the launch declaration.
*/
#define SYCL_INTEL_TARGET 20

#include "sycl/kernels/gemm_sqrsum/device/gemm_sqrsum_types.hpp"
#include "sycl/kernels/gemm_sqrsum/device/gemm_sqrsum_dispatch.hpp"

namespace gemm_sqrsum {

void launch_gemm_sqrsum(
    at::Tensor& C,
    at::Tensor& sqrsum,
    const at::Tensor& A,
    const at::Tensor& B) {
  // Single concrete config (tf32, tile 64x32x16); see gemm_sqrsum_types.hpp.
  runGemmSqrSum(C, sqrsum, A, B);
}

}  // namespace gemm_sqrsum
