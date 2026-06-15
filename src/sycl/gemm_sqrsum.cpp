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
// Only the launch declarations are needed here; the heavy CUTLASS/CuTe kernel
// types live in the per-dtype generated TUs, not in this host dispatcher.
#include "sycl/kernels/gemm_sqrsum/device/gemm_sqrsum_dispatch.hpp"

namespace {

// Drop leading size-1 dims until the tensor has `rank` dims. This squeezes the
// n_splits=1 axis the mhc_pre pipeline carries on its 3D buffers
// (gemm_out_mul [1,T,N], gemm_out_sqrsum [1,T]) without touching the trailing
// dims, so a legitimate small leading dim (M, N) is never collapsed.
at::Tensor squeeze_leading_to(const at::Tensor& t, int64_t rank) {
  at::Tensor x = t;
  while (x.dim() > rank && x.size(0) == 1) {
    x = x.squeeze(0);
  }
  return x;
}

}  // namespace

/// @brief Compute C = A @ B and sqrsum[i] = sum(A[i,:]^2)
///
/// Contract (mhc_pre GEMM+sqrsum stage):
///   A      [M, K]   bf16/fp16/fp32   (residual.view(M, hc_hidden))
///   B      [N, K]   fp32             (fn = [24, 16384] = [N, K])
///   C      [M, N]   fp32             (gemm_out_mul)
///   sqrsum [M]      fp32             (gemm_out_sqrsum), sqrsum[m] = sum_k A[m,k]^2
///
/// Leading singleton (n_splits=1) axes on any argument are squeezed away.
///
/// Precision: when B is fp32 the kernel runs a tf32 x tf32 -> fp32 DPAS path
/// (A widened to fp32, B taken as-is, both reinterpreted to tf32 at load). When
/// A and B share a 16-bit dtype (half/bf16) the matching native DPAS path runs.
/// C is always fp32.
///
/// @param C Output tensor [M, N] fp32
/// @param sqrsum Output row-wise square sum [M] fp32
/// @param A Input tensor [M, K]
/// @param B Input tensor [N, K]
void gemm_with_sqrsum(
    at::Tensor& C,
    at::Tensor& sqrsum,
    const at::Tensor& A_in,
    const at::Tensor& B_in) {

  c10::DeviceGuard guard(A_in.device());

  // Squeeze the n_splits=1 leading axis off the 3D pipeline buffers. The squeezed
  // views share storage with the originals, so kernel writes to Cv/sqv land in
  // the caller's C/sqrsum buffers.
  at::Tensor A = squeeze_leading_to(A_in, 2);
  at::Tensor B = squeeze_leading_to(B_in, 2);
  at::Tensor Cv = squeeze_leading_to(C, 2);
  at::Tensor sqv = squeeze_leading_to(sqrsum, 1);

  CHECK_DEVICE(A);
  CHECK_DEVICE(B);
  CHECK_INPUT(Cv);
  CHECK_INPUT(sqv);

  auto a_dtype = A.scalar_type();
  auto b_dtype = B.scalar_type();

  TORCH_CHECK(Cv.scalar_type() == at::ScalarType::Float, "C must be float32");
  TORCH_CHECK(sqv.scalar_type() == at::ScalarType::Float,
              "sqrsum must be float32 for atomic operations");

  auto is_supported = [](at::ScalarType t) {
    return t == at::ScalarType::Half || t == at::ScalarType::BFloat16 || t == at::ScalarType::Float;
  };
  TORCH_CHECK(is_supported(a_dtype), "A must be Half, BFloat16, or Float, got ", a_dtype);
  TORCH_CHECK(is_supported(b_dtype), "B must be Half, BFloat16, or Float, got ", b_dtype);

  // Verify shapes. A is [M, K], B is [N, K] (both K-contiguous), C is [M, N].
  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(0);

  TORCH_CHECK(B.size(1) == K, "K mismatch for GEMM: A.K=", K, " but B.K=", B.size(1));
  TORCH_CHECK(Cv.size(0) == M && Cv.size(1) == N, "Output C shape mismatch: expected [", M, ",", N, "]");
  TORCH_CHECK(sqv.size(0) == M, "Output sqrsum size mismatch");
  TORCH_CHECK(sqv.dim() == 1, "sqrsum must be 1D");

  // Single tf32 x tf32 -> fp32 DPAS path. The launcher widens A and B to fp32
  // and reinterprets to tf32 at load, so it covers the production bf16(A) x
  // fp32(B) case and any half/bf16/float input combination uniformly.
  gemm_sqrsum::launch_gemm_sqrsum(Cv, sqv, A, B);
}

#undef SYCL_INTEL_TARGET
