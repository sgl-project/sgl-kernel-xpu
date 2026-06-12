/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
/*! \file
    \brief GEMM + Square Sum dispatch interface for PyTorch
*/
#define SYCL_INTEL_TARGET 20
#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <sycl/sycl.hpp>

#include "Utils.h"
#include "sycl/kernels/gemm_sqrsum/device/gemm_sqrsum_dispatch.hpp"
#include "sycl/kernels/gemm_sqrsum/device/gemm_sqrsum_types.hpp"

namespace {

#define DISPATCH_GEMM_SQRSUM_TILE(ELEM)                                          \
  do {                                                                           \
    gemm_sqrsum::launch_gemm_sqrsum_##ELEM##_256x256x16(C, sqrsum, A, B);      \
  } while (0)

#define DISPATCH_GEMM_SQRSUM_DTYPE()                                            \
  do {                                                                          \
    switch (dtype) {                                                            \
      case at::ScalarType::Half:                                                \
        DISPATCH_GEMM_SQRSUM_TILE(half);                                       \
        break;                                                                  \
      case at::ScalarType::BFloat16:                                            \
        DISPATCH_GEMM_SQRSUM_TILE(bf16);                                       \
        break;                                                                  \
      default:                                                                  \
        TORCH_CHECK(false, "Unsupported data type for GEMM+SqrSum. Supported: Half, BFloat16");           \
    }                                                                           \
  } while (0)

}  // namespace

/// @brief Compute C = A @ B and sqrsum[i] = sum(A[i,:]^2)
///
/// @param C Output tensor [M, N]
/// @param sqrsum Output row-wise square sum [M]
/// @param A Input tensor [M, K]
/// @param B Input tensor [K, N]
void gemm_with_sqrsum(
    at::Tensor& C,
    at::Tensor& sqrsum,
    const at::Tensor& A,
    const at::Tensor& B) {

  CHECK_INPUT(C);
  CHECK_INPUT(sqrsum);
  CHECK_INPUT(A);
  CHECK_INPUT(B);

  c10::DeviceGuard guard(A.device());

  auto dtype = A.scalar_type();

  // Verify tensor dtypes
  TORCH_CHECK(B.scalar_type() == dtype && C.scalar_type() == dtype,
              "A, B, and C must have the same data type");
  TORCH_CHECK(sqrsum.scalar_type() == at::ScalarType::Float,
              "sqrsum must be float32 for atomic operations");

  TORCH_CHECK(
      dtype == at::ScalarType::Half || dtype == at::ScalarType::BFloat16,
      "Unsupported data type for GEMM+SqrSum. Supported: Half, BFloat16 (DPAS limitation)");

  // Verify shapes
  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);

  TORCH_CHECK(B.size(0) == K, "Matrix dimensions don't match for GEMM: A.K=", K, " but B.K=", B.size(0));
  TORCH_CHECK(C.size(0) == M && C.size(1) == N, "Output C shape mismatch");
  TORCH_CHECK(sqrsum.size(0) == M, "Output sqrsum size mismatch");
  TORCH_CHECK(sqrsum.dim() == 1, "sqrsum must be 1D");

  DISPATCH_GEMM_SQRSUM_DTYPE();
}

#undef DISPATCH_GEMM_SQRSUM_TILE
#undef DISPATCH_GEMM_SQRSUM_DTYPE
#undef SYCL_INTEL_TARGET
