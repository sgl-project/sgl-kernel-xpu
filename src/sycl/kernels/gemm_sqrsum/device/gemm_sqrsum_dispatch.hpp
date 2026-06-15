/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
/*!
  \file
  \brief Forward declaration for the GEMM+SqrSum kernel launch function.
*/

#pragma once

#include <ATen/ATen.h>
#include <sycl/sycl.hpp>

namespace gemm_sqrsum {

// The kernel is instantiated in src/sycl/gemm_sqrsum_kernel.cpp as its own
// translation unit (kept separate from this declaration / the host dispatcher
// so the heavy CUTLASS template stack compiles in one place). It runs the
// mhc_pre path: bf16(A) x fp32(B) -> fp32 via a tf32 x tf32 -> fp32 DPAS atom
// (A widened to fp32, B taken as-is, both reinterpreted to tf32 at load). C and
// sqrsum are fp32.
void launch_gemm_sqrsum(
    at::Tensor& C,
    at::Tensor& sqrsum,
    const at::Tensor& A,
    const at::Tensor& B);

}  // namespace gemm_sqrsum
