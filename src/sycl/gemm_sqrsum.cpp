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

/// @brief Compute the K-split partials of C = A @ Bᵀ and the row-wise square sum.
///
/// Design B contract (mhc_pre GEMM+sqrsum stage). The K reduction is partitioned
/// into `n_splits` slices; each slice writes its own partial slab and the
/// downstream hc_pre_big_fuse reduces the leading axis. NOTHING is summed here.
///   A      [M, K]            bf16/fp16/fp32   (residual.view(M, hc_hidden))
///   B      [N, K]            fp32             (fn = [24, 16384] = [N, K])
///   C      [n_splits, M, N]  fp32             (gemm_out_mul partials)
///   sqrsum [n_splits, M]     fp32             (gemm_out_sqrsum partials)
///                            sqrsum[s,m] = sum_{k in split s} A[m,k]^2
///
/// n_splits is C.size(0): the caller (mhc_pre) picks it (32 for the split-k
/// path, 1 for the simple path) and pre-allocates the partial buffers.
///
/// Precision: B fp32 -> tf32 x tf32 -> fp32 DPAS (A widened to fp32, B as-is,
/// both reinterpreted to tf32 at load). C/sqrsum are always fp32.
///
/// @param C      Output partials [n_splits, M, N] fp32
/// @param sqrsum Output partials [n_splits, M] fp32
/// @param A      Input tensor [M, K]
/// @param B      Input tensor [N, K]
void gemm_with_sqrsum(
    at::Tensor& C,
    at::Tensor& sqrsum,
    const at::Tensor& A,
    const at::Tensor& B) {

  c10::DeviceGuard guard(A.device());

  CHECK_DEVICE(A);
  CHECK_DEVICE(B);
  CHECK_INPUT(C);
  CHECK_INPUT(sqrsum);

  auto a_dtype = A.scalar_type();
  auto b_dtype = B.scalar_type();

  TORCH_CHECK(C.scalar_type() == at::ScalarType::Float, "C must be float32");
  TORCH_CHECK(sqrsum.scalar_type() == at::ScalarType::Float, "sqrsum must be float32");

  auto is_supported = [](at::ScalarType t) {
    return t == at::ScalarType::Half || t == at::ScalarType::BFloat16 || t == at::ScalarType::Float;
  };
  TORCH_CHECK(is_supported(a_dtype), "A must be Half, BFloat16, or Float, got ", a_dtype);
  TORCH_CHECK(is_supported(b_dtype), "B must be Half, BFloat16, or Float, got ", b_dtype);

  // Shapes. A is [M, K], B is [N, K] (both K-contiguous); C is [n_splits, M, N]
  // and sqrsum is [n_splits, M] — the fuse-shaped partial buffers.
  TORCH_CHECK(A.dim() == 2, "A must be 2D [M, K]");
  TORCH_CHECK(B.dim() == 2, "B must be 2D [N, K]");
  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(0);

  TORCH_CHECK(B.size(1) == K, "K mismatch for GEMM: A.K=", K, " but B.K=", B.size(1));
  TORCH_CHECK(C.dim() == 3 && C.size(1) == M && C.size(2) == N,
              "Output C must be [n_splits, ", M, ", ", N, "]");
  TORCH_CHECK(sqrsum.dim() == 2 && sqrsum.size(0) == C.size(0) && sqrsum.size(1) == M,
              "Output sqrsum must be [n_splits, ", M, "] matching C's leading dim");

  // Single tf32 x tf32 -> fp32 DPAS path. The launcher widens A and B to fp32
  // and reinterprets to tf32 at load, so it covers the production bf16(A) x
  // fp32(B) case and any half/bf16/float input combination uniformly.
  gemm_sqrsum::launch_gemm_sqrsum(C, sqrsum, A, B);
}

#undef SYCL_INTEL_TARGET
