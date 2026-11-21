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

#undef SYCL_INTEL_TARGET
#define SYCL_INTEL_TARGET 20
#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <sycl/sycl.hpp>

#include "SGLKernelPerf.h"
#include "Utils.h"
#include "sgl_kernel_export.h"
#include "sycl/kernels/hc_pre_gemm_sqr_sum/device/hc_pre_gemm_sqr_sum_types.hpp"

SGL_KERNEL_EXPORT void
hc_pre_gemm_sqr_sum(at::Tensor& C, at::Tensor& sqr_sum, const at::Tensor& A, const at::Tensor& B) {
  c10::DeviceGuard guard(A.device());

  CHECK_INPUT(2, at::ScalarType::BFloat16, A);
  CHECK_INPUT(2, at::ScalarType::Float, B);
  CHECK_INPUT(3, at::ScalarType::Float, C);
  CHECK_INPUT(2, at::ScalarType::Float, sqr_sum);

  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(0);

  CHECK_SHAPE(B, N, K);
  CHECK_SHAPE(C, C.size(0), M, N);
  CHECK_SHAPE(sqr_sum, C.size(0), M);

#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
  // GEMM (2*M*N*K) + squared-sum reduction (~2*M*N) per split.
  const int64_t n_splits = C.size(0);
  const double flops =
      2.0 * static_cast<double>(n_splits) * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K) +
      2.0 * static_cast<double>(n_splits) * static_cast<double>(M) * static_cast<double>(N);
  const double bytes = static_cast<double>(M) * static_cast<double>(K) * static_cast<double>(A.element_size()) +
                       static_cast<double>(N) * static_cast<double>(K) * static_cast<double>(B.element_size()) +
                       static_cast<double>(C.numel()) * static_cast<double>(C.element_size()) +
                       static_cast<double>(sqr_sum.numel()) * static_cast<double>(sqr_sum.element_size());
  auto profiling_queue = at::xpu::getCurrentXPUStream().queue();
  SGL_KERNEL_PERF_SCOPE("hc_pre_gemm_sqr_sum", profiling_queue, bytes, flops);
#endif

  runHcPreGemmSqrSum(C, sqr_sum, A, B);
}

#undef SYCL_INTEL_TARGET
