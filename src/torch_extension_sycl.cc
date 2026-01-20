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

#include "sgl_flash_kernel_ops.h"
#include "sgl_kernel_ops.h"
#include "sgl_kernel_torch_shim.h"

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
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

  m.def("topk_softmax(Tensor! topk_weights, Tensor! topk_indices, Tensor gating_output, bool renormalize) -> ()");
  m.impl("topk_softmax", torch::kXPU, &at::native::xpu::topk_softmax);

  m.def("swiglu_with_alpha_and_limit(Tensor x, float alpha, float limit) -> Tensor");
  m.impl("swiglu_with_alpha_and_limit", torch::kXPU, &swiglu_with_alpha_and_limit);
  m.def(
      "moe_fused_gate(Tensor input, Tensor bias, int num_expert_group, int topk_group, int topk, int "
      "num_fused_shared_experts, float routed_scaling_factor, bool apply_routed_scaling_factor_on_output) -> "
      "(Tensor[])");
  m.impl("moe_fused_gate", torch::kXPU, &moe_fused_gate);

  m.def(
      "rotary_embedding(Tensor positions, Tensor query, Tensor key, int head_size, Tensor cos_sin_cache, "
      "bool is_neox) -> (Tensor, Tensor)");
  m.impl("rotary_embedding", torch::kXPU, &at::native::xpu::rotary_embedding);

  m.def("moe_sum_reduce(Tensor input, Tensor output, float routed_scaling_factor) -> ()");
  m.impl("moe_sum_reduce", torch::kXPU, &moe_sum_reduce);
  m.def(
      "moe_align_block_size(Tensor topk_ids, int num_experts, int block_size, Tensor! sorted_token_ids, Tensor! "
      "experts_ids, Tensor! num_tokens_post_pad, Tensor! cumsum_buffer, bool "
      "pad_sorted_token_ids) -> ()");
  m.impl("moe_align_block_size", torch::kXPU, &moe_align_block_size);

  m.def("moe_sum(Tensor input, Tensor! output) -> ()");
  m.impl("moe_sum", torch::kXPU, &moe_sum);

  m.def(
      "moe_grouped_mm_nt(Tensor output, Tensor activations, Tensor weights, Tensor total_rows_for_experts, int "
      "n_experts, bool fuse_silu) -> ()");
  m.impl("moe_grouped_mm_nt", torch::kXPU, &moe_grouped_mm_nt);

  m.def(
      "prepare_moe_input(Tensor topk_ids, Tensor expert_offsets, Tensor? blockscale_offsets, Tensor problem_sizes1,"
      " Tensor problem_sizes2, Tensor input_permutation, Tensor output_permutation, int num_experts, int n, int k) -> "
      "()");
  m.impl("prepare_moe_input", torch::kXPU, &prepare_moe_input);
  m.def("shuffle_rows(Tensor input, Tensor dst2src_map, Tensor output) -> ()");
  m.impl("shuffle_rows", torch::kXPU, &shuffle_rows);
  m.def("apply_shuffle_mul_sum(Tensor input, Tensor output, Tensor permutation, Tensor? factors) -> ()");
  m.impl("apply_shuffle_mul_sum", torch::kXPU, &apply_shuffle_mul_sum);

  /*
   * From cutlass attention
   */
  m.def(
      "fwd(Tensor!  q,"
      "    Tensor   k,"
      "    Tensor   v,"
      "    Tensor?  q_v,"
      "    Tensor  cu_seqlens_q,"
      "    Tensor  cu_seqlens_k,"
      "    int     max_seqlen_q,"
      "    Tensor  page_table,"
      "    Tensor?  kv_batch_idx,"
      "    Tensor?  leftpad_k,"
      "    Tensor?  rotary_cos,"
      "    Tensor?  rotary_sin,"
      "    Tensor?  seqlens_rotary,"
      "    Tensor?  q_descale,"
      "    Tensor?  k_descale,"
      "    Tensor?  v_descale,"
      "    float    softmax_scale,"
      "    Tensor?  sinks,"
      "    bool     is_causal,"
      "    int      window_size_left,"
      "    int      window_size_right,"
      "    float    softcap,"
      "    bool     is_rotary_interleaved,"
      "    Tensor?  scheduler_metadata,"
      "    int      num_splits,"
      "    bool?    pack_gqa,"
      "    int      sm_margin) -> Tensor[]");
  m.impl("fwd", torch::kXPU, make_pytorch_shim(&mha_fwd));

  /*
   * From quantization ops
   */
  m.def(
      "sgl_per_token_group_quant_8bit(Tensor input, Tensor output_q, Tensor output_s, int group_size,"
      " float eps, float fp8_min, float fp8_max, bool scale_ue8m0) -> ()");
  m.impl("sgl_per_token_group_quant_8bit", torch::kXPU, &at::native::xpu::sgl_per_token_group_quant_8bit);
  m.def("sgl_per_tensor_quant_fp8(Tensor input, Tensor output_q, Tensor output_s, bool is_static) -> ()");
  m.impl("sgl_per_tensor_quant_fp8", torch::kXPU, &sgl_per_tensor_quant_fp8);
}

REGISTER_EXTENSION(common_ops)
