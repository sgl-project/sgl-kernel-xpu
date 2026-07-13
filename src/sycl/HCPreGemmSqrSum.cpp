/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief HC Pre Gemm Square Sum Kernel
*/

#define SYCL_INTEL_TARGET 20
#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <sycl/sycl.hpp>

#include "Utils.h"
#include "sycl/kernels/hc_pre_gemm_sqr_sum/device/hc_pre_gemm_sqr_sum_types.hpp"

void hc_pre_gemm_sqr_sum(at::Tensor& C, at::Tensor& sqr_sum, const at::Tensor& A, const at::Tensor& B) {
  c10::DeviceGuard guard(A.device());

  CHECK_INPUT(A);
  CHECK_INPUT(B);
  CHECK_INPUT(C);
  CHECK_INPUT(sqr_sum);

  TORCH_CHECK(C.scalar_type() == at::ScalarType::Float, "C must be float32");
  TORCH_CHECK(sqr_sum.scalar_type() == at::ScalarType::Float, "sqr_sum must be float32");
  TORCH_CHECK(A.scalar_type() == at::ScalarType::BFloat16, "A must be BFloat16, got ", A.scalar_type());
  TORCH_CHECK(B.scalar_type() == at::ScalarType::Float, "B must be float32, got ", B.scalar_type());
  TORCH_CHECK(A.dim() == 2, "A must be 2D [M, K]");
  TORCH_CHECK(B.dim() == 2, "B must be 2D [N, K]");

  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(0);

  TORCH_CHECK(B.size(1) == K, "K mismatch for GEMM: A.K=", K, " but B.K=", B.size(1));
  TORCH_CHECK(C.dim() == 3 && C.size(1) == M && C.size(2) == N, "Output C must be [n_splits, ", M, ", ", N, "]");
  TORCH_CHECK(
      sqr_sum.dim() == 2 && sqr_sum.size(0) == C.size(0) && sqr_sum.size(1) == M,
      "Output sqr_sum must be [n_splits, ",
      M,
      "] matching C's leading dim");

  runHcPreGemmSqrSum(C, sqr_sum, A, B);
}

#undef SYCL_INTEL_TARGET
