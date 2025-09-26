/* Copyright 2025 SGLang Team. All Rights Reserved.

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
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>
#include <torch/library.h>

#include "sgl_kernel_ops.h"

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  /*
   * From csrc/gemm
   */
  m.def("awq_dequantize(Tensor qweight, Tensor scales, Tensor qzeros) -> Tensor");
  m.impl("awq_dequantize", torch::kXPU, &awq_dequantize);

  m.def("silu_and_mul(Tensor! out, Tensor input) -> ()");
  m.impl("silu_and_mul", torch::kXPU, &silu_and_mul);

  m.def("gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()");
  m.impl("gelu_tanh_and_mul", torch::kXPU, &gelu_tanh_and_mul);

  m.def("gelu_and_mul(Tensor! out, Tensor input) -> ()");
  m.impl("gelu_and_mul", torch::kXPU, &gelu_and_mul);

  m.def("rmsnorm(Tensor! output, Tensor input, Tensor weight, float eps) -> ()");
  m.impl("rmsnorm", torch::kXPU, &at::native::xpu::rmsnorm);

  m.def("fused_add_rmsnorm(Tensor! input, Tensor! residual, Tensor weight, float eps) -> ()");
  m.impl("fused_add_rmsnorm", torch::kXPU, &at::native::xpu::fused_add_rmsnorm);

  m.def("gemma_rmsnorm(Tensor! output, Tensor input, Tensor weight, float eps) -> ()");
  m.impl("gemma_rmsnorm", torch::kXPU, &at::native::xpu::gemma_rmsnorm);

  m.def("gemma_fused_add_rmsnorm(Tensor! input, Tensor! residual, Tensor weight, float eps) -> ()");
  m.impl("gemma_fused_add_rmsnorm", torch::kXPU, &at::native::xpu::gemma_fused_add_rmsnorm);

  m.def(
      "apply_rotary_embedding_two_qk(Tensor query, Tensor key, Tensor sin, Tensor cos, Tensor query_out, Tensor "
      "key_out) -> ()");
  m.impl("apply_rotary_embedding_two_qk", torch::kXPU, &at::native::xpu::apply_rotary_embedding_two_qk);

  m.def("apply_rotary_embedding_two(Tensor query, Tensor sin, Tensor cos, Tensor query_out) -> ()");
  m.impl("apply_rotary_embedding_two", torch::kXPU, &at::native::xpu::apply_rotary_embedding_two);

  m.def("apply_rotary_embedding_half(Tensor query, Tensor sin, Tensor cos, Tensor query_out) -> ()");
  m.impl("apply_rotary_embedding_half", torch::kXPU, &at::native::xpu::apply_rotary_embedding_half);

  m.def(
      "apply_rotary_embedding_half_qk(Tensor query, Tensor key, Tensor sin, Tensor cos, Tensor query_out, Tensor "
      "key_out) -> ()");
  m.impl("apply_rotary_embedding_half_qk", torch::kXPU, &at::native::xpu::apply_rotary_embedding_half_qk);

  m.def(
      "rotary_embedding_batched(Tensor positions, Tensor query, Tensor key, int head_size, Tensor cos_sin_cache, bool "
      "is_neox, int rot_dim, Tensor cos_sin_cache_offsets) -> ()");
  m.impl("rotary_embedding_batched", torch::kXPU, &at::native::xpu::rotary_embedding_batched);

  m.def(
      "rotary_embedding(Tensor positions, Tensor query, Tensor key, int head_size, Tensor cos_sin_cache, bool is_neox, "
      "int rot_dim) -> ()");
  m.impl("rotary_embedding", torch::kXPU, &at::native::xpu::rotary_embedding);

  m.def(
      "ds_rotary_embedding_qk(Tensor positions, Tensor query, Tensor key, Tensor? offsets_opt, Tensor cos_sin_cache, "
      "int rotary_dim, bool is_neox_style) -> (Tensor, Tensor)");
  m.impl("ds_rotary_embedding_qk", torch::kXPU, &at::native::xpu::ds_rotary_embedding_qk);

  //   m.def(
  //       "fp8_blockwise_scaled_mm(Tensor mat_a, Tensor mat_b, Tensor scales_a, Tensor scales_b, ScalarType out_dtype,
  //       -> Tensor");
  //   m.impl("fp8_blockwise_scaled_mm", torch::kXPU, &fp8_blockwise_scaled_mm);
}

REGISTER_EXTENSION(common_ops)
