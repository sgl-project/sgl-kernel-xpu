/* Copyright 2026 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Dense weight-only FP8 (W8A16) GEMM for Xe2 (BMG):
//   output[M, N] = input[M, K] @ dequant(weight[N, K]).T (+ bias[N])
// One SYCL-TLA kernel launch on the current XPU stream (kernels/gemm/xe20/w8a16_dense).

#undef SYCL_INTEL_TARGET
#define SYCL_INTEL_TARGET 20

// torch headers before cute: cute/util/print.hpp redefines printf.
#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstdint>
#include <optional>

#include "kernels/gemm/xe20/w8a16_dense/dense_w8a16.hpp"
#include "sgl_kernel_export.h"

namespace {

inline bool aligned64(const at::Tensor& t) {
  return reinterpret_cast<uintptr_t>(t.data_ptr()) % 64 == 0;
}

// Returns the scale mode: 0 per-tensor, 1 per-channel, 2 128x128 block.
int check_args(
    const at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& weight_scale,
    const std::optional<at::Tensor>& bias) {
  TORCH_CHECK(
      input.device() == weight.device() && input.device() == weight_scale.device() && input.device() == output.device(),
      "fp8_w8a16_mm_xe20: tensors must be on the same device");
  TORCH_CHECK(
      input.scalar_type() == at::kBFloat16 && input.dim() == 2 && input.is_contiguous(),
      "fp8_w8a16_mm_xe20: input must be contiguous bfloat16 [M, K]");
  TORCH_CHECK(
      weight.scalar_type() == at::kFloat8_e4m3fn && weight.dim() == 2 && weight.is_contiguous(),
      "fp8_w8a16_mm_xe20: weight must be contiguous float8_e4m3fn [N, K]");
  const int64_t M = input.size(0), K = input.size(1), N = weight.size(0);
  TORCH_CHECK(weight.size(1) == K, "fp8_w8a16_mm_xe20: weight [N, K] does not match input [M, K]");
  TORCH_CHECK(
      output.scalar_type() == at::kBFloat16 && output.dim() == 2 && output.is_contiguous() && output.size(0) == M &&
          output.size(1) == N,
      "fp8_w8a16_mm_xe20: output must be contiguous bfloat16 [M, N]");
  TORCH_CHECK(N > 0 && N % 64 == 0, "fp8_w8a16_mm_xe20: N must be a positive multiple of 64");
  // The decode tiles step K by 64.
  TORCH_CHECK(K > 0 && K % 64 == 0, "fp8_w8a16_mm_xe20: K must be a positive multiple of 64");
  TORCH_CHECK(
      M < (1 << 24) && N < (1 << 24) && K < (1 << 23), "fp8_w8a16_mm_xe20: matrix too large for block-2D access");
  TORCH_CHECK(
      weight_scale.scalar_type() == at::kFloat && weight_scale.is_contiguous(),
      "fp8_w8a16_mm_xe20: weight_scale must be contiguous float32");
  int mode = -1;
  if (weight_scale.numel() == 1) {
    mode = 0;
  } else if (
      (weight_scale.dim() == 1 && weight_scale.size(0) == N) ||
      (weight_scale.dim() == 2 && weight_scale.size(0) == N && weight_scale.size(1) == 1)) {
    mode = 1;
  } else if (
      weight_scale.dim() == 2 && weight_scale.size(0) == (N + 127) / 128 && K % 128 == 0 &&
      weight_scale.size(1) == K / 128) {
    mode = 2;
  }
  TORCH_CHECK(
      mode >= 0,
      "fp8_w8a16_mm_xe20: weight_scale must be per-tensor (one element), per-channel [N], or 128x128 block "
      "[ceil(N/128), K/128] (block scales need K % 128 == 0); got ",
      weight_scale.sizes());
  if (bias.has_value()) {
    TORCH_CHECK(bias->device() == input.device(), "fp8_w8a16_mm_xe20: bias must be on the same device");
    TORCH_CHECK(
        bias->scalar_type() == at::kFloat && bias->dim() == 1 && bias->size(0) == N && bias->is_contiguous(),
        "fp8_w8a16_mm_xe20: bias must be contiguous float32 [N]");
  }
  if (M > 0) {
    TORCH_CHECK(
        aligned64(input) && aligned64(weight) && aligned64(output),
        "fp8_w8a16_mm_xe20: input, weight and output must be 64-byte aligned");
  }
  return mode;
}

}  // namespace

SGL_KERNEL_EXPORT void fp8_w8a16_mm_xe20(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& weight_scale,
    const std::optional<at::Tensor>& bias) {
  const int mode = check_args(output, input, weight, weight_scale, bias);
  const int64_t M = input.size(0);
  if (M == 0) return;
  c10::DeviceGuard guard(input.device());
  sycl::queue& queue = c10::xpu::getCurrentXPUStream(input.device().index()).queue();
  dense_w8a16::run(
      queue,
      output.data_ptr(),
      input.data_ptr(),
      weight.data_ptr(),
      weight_scale.data_ptr<float>(),
      bias.has_value() ? bias->data_ptr<float>() : nullptr,
      static_cast<int>(M),
      static_cast<int>(weight.size(0)),
      static_cast<int>(input.size(1)),
      mode);
}

#undef SYCL_INTEL_TARGET
