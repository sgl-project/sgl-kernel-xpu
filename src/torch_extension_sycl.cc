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
  m.def("weak_ref_tensor(Tensor(a) tensor) -> Tensor(a)");
  m.impl("weak_ref_tensor", torch::kXPU, &weak_ref_tensor);

  m.def("awq_dequantize(Tensor qweight, Tensor scales, Tensor qzeros) -> Tensor");
  m.impl("awq_dequantize", torch::kXPU, &awq_dequantize);

  m.def("silu_and_mul(Tensor! out, Tensor input) -> ()");
  m.impl("silu_and_mul", torch::kXPU, &silu_and_mul);

  m.def("silu_and_mul_clamp(Tensor! out, Tensor input, float swiglu_limit) -> ()");
  m.impl("silu_and_mul_clamp", torch::kXPU, &silu_and_mul_clamp);

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

  m.def(
      "topk_sigmoid(Tensor! topk_weights, Tensor! topk_indices, Tensor gating_output, bool renormalize, Tensor? "
      "correction_bias, float routed_scaling_factor=1.0, int num_fused_shared_experts=0) -> ()");
  m.impl("topk_sigmoid", torch::kXPU, &at::native::xpu::topk_sigmoid);

  m.def("top_k_renorm_probs(Tensor probs, Tensor! renorm_probs, Tensor? maybe_top_k_arr, int top_k_val) -> ()");
  m.impl("top_k_renorm_probs", torch::kXPU, &top_k_renorm_probs);

  /*
   * Fast radix top-k (DeepSeek V3.2 indexer)
   */
  m.def("fast_topk(Tensor score, Tensor! indices, Tensor lengths, Tensor? row_starts) -> ()");
  m.impl("fast_topk", torch::kXPU, &fast_topk_interface);

  m.def(
      "fast_topk_transform_fused(Tensor score, Tensor lengths, Tensor! dst_page_table, Tensor src_page_table, "
      "Tensor cu_seqlens_q, Tensor? row_starts) -> ()");
  m.impl("fast_topk_transform_fused", torch::kXPU, &fast_topk_transform_interface);

  m.def(
      "fast_topk_transform_ragged_fused(Tensor score, Tensor lengths, Tensor! topk_indices_ragged, "
      "Tensor topk_indices_offset, Tensor? row_starts) -> ()");
  m.impl("fast_topk_transform_ragged_fused", torch::kXPU, &fast_topk_transform_ragged_interface);

  m.def("swiglu_gpt_oss_sigmoid_alpha(Tensor x, float alpha, float limit) -> Tensor");
  m.impl("swiglu_gpt_oss_sigmoid_alpha", torch::kXPU, &swiglu_gpt_oss_sigmoid_alpha);
  m.def(
      "moe_fused_gate(Tensor input, Tensor? bias, int num_expert_group, int topk_group, int topk, int "
      "num_fused_shared_experts, int scoring_func, bool renormalize, float routed_scaling_factor, bool "
      "apply_routed_scaling_factor_on_output) -> "
      "(Tensor[])");
  m.impl("moe_fused_gate", torch::kXPU, &moe_fused_gate);

  m.def(
      "rotary_embedding(Tensor positions, Tensor query, Tensor key, int head_size, Tensor cos_sin_cache, "
      "bool is_neox) -> (Tensor, Tensor)");
  m.impl("rotary_embedding", torch::kXPU, &at::native::xpu::rotary_embedding);

  m.def(
      "store_cache(Tensor k, Tensor v, Tensor(a!) k_cache, Tensor(b!) v_cache, "
      "Tensor indices) -> ()");
  m.impl("store_cache", torch::kXPU, &at::native::xpu::store_cache);

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
      "moe_grouped_mm_nt_xe20(Tensor! output, Tensor activations, Tensor weights, Tensor? bias, Tensor "
      "total_rows_for_experts, int n_experts, int activation_type, bool fuse_act, float gemm1_alpha=1.702, float "
      "gemm1_limit=7.0) -> ()");
  m.impl("moe_grouped_mm_nt_xe20", torch::kXPU, &moe_grouped_mm_nt_xe20);

  m.def(
      "moe_grouped_mm_nt_xe20_mxfp4_w4a16(Tensor! output, Tensor activations, Tensor packed_weights, Tensor scales, "
      "Tensor? bias, Tensor total_rows_for_experts, int n_experts, int activation_type, bool fuse_act, "
      "float gemm1_alpha=1.702, float gemm1_limit=7.0) -> ()");
  m.impl("moe_grouped_mm_nt_xe20_mxfp4_w4a16", torch::kXPU, &moe_grouped_mm_nt_xe20_mxfp4_w4a16);

  m.def(
      "prepare_moe_input(Tensor topk_ids, Tensor! expert_offsets, Tensor? blockscale_offsets, Tensor! problem_sizes1,"
      " Tensor! problem_sizes2, Tensor! input_permutation, Tensor! output_permutation, int num_experts, int n, int k)"
      " -> ()");
  m.impl("prepare_moe_input", torch::kXPU, &prepare_moe_input);
  m.def("scatter_tokens_to_experts(Tensor input, Tensor src2dst_map, Tensor! output) -> ()");
  m.impl("scatter_tokens_to_experts", torch::kXPU, &scatter_tokens_to_experts);
  m.def(
      "apply_shuffle_mul_sum(Tensor input, Tensor! output, Tensor permutation, float routed_scaling_factor, Tensor? "
      "factors) -> ()");
  m.impl("apply_shuffle_mul_sum", torch::kXPU, &apply_shuffle_mul_sum);

  m.def("merge_state_v2(Tensor v_a, Tensor s_a, Tensor v_b, Tensor s_b, Tensor! v_merged, Tensor! s_merged) -> ()");
  m.impl("merge_state_v2", torch::kXPU, &merge_state_v2);
  m.def("merge_state(Tensor v_a, Tensor s_a, Tensor v_b, Tensor s_b, Tensor! v_merged, Tensor! s_merged) -> ()");
  m.impl("merge_state", torch::kXPU, &merge_state);
  /*
   * From cutlass attention
   */
  m.def(
      "fwd(Tensor   q,"
      "    Tensor   k,"
      "    Tensor   v,"
      "    Tensor?  q_v,"
      "    Tensor  cu_seqlens_q,"
      "    Tensor  cu_seqlens_k,"
      "    int     max_seqlen_q,"
      "    int     max_seqlen_k,"
      "    Tensor?  page_table,"
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
      "    int      num_kv_splits,"
      "    bool?    pack_gqa,"
      "    int      sm_margin,"
      "    Tensor(a!)?  out=None,"
      "    Tensor?  k_new=None,"
      "    Tensor?  v_new=None,"
      "    Tensor?  cu_seqlens_k_new=None) -> (Tensor(a!), Tensor, Tensor, Tensor)");
  m.impl("fwd", torch::kXPU, make_pytorch_shim(&mha_fwd));

  m.def("flash_mla_get_workspace_size", &flash_mla_get_workspace_size);

  m.def(
      "flash_mla_decode(Tensor! out, Tensor! q_nope, Tensor! q_pe, Tensor! kv_c_and_k_pe_cache, Tensor! seq_lens, "
      "Tensor! "
      "page_table, Tensor! workspace, float sm_scale, int num_kv_splits) -> ()");
  m.impl("flash_mla_decode", torch::kXPU, &flash_mla_decode);

  m.def(
      "flash_mla_sparse_decode(Tensor! out, Tensor! lse_out, Tensor! q, Tensor! k_cache, "
      "Tensor! indices, Tensor? topk_length, "
      "Tensor? extra_k_cache, Tensor? extra_indices, Tensor? extra_topk_length, "
      "Tensor? attn_sink, float sm_scale, int head_dim_v, bool is_fp8_kvcache) -> ()");
  m.impl("flash_mla_sparse_decode", torch::kXPU, &flash_mla_sparse_decode);

  m.def("flash_mla_prefill_get_workspace_size", &flash_mla_prefill_get_workspace_size);

  m.def(
      "flash_mla_prefill(Tensor! out, Tensor! q_nope, Tensor! q_pe, Tensor! kv_c_and_k_pe_cache, "
      "Tensor! cu_seqlens_q, Tensor! seq_lens, int max_seqlen_q, "
      "Tensor! page_table, Tensor! workspace, float sm_scale, bool causal, int num_kv_splits) -> ()");
  ;
  m.impl("flash_mla_prefill", torch::kXPU, &flash_mla_prefill);

  /*
   * From quantization ops
   */
  m.def(
      "sgl_per_token_group_quant_8bit(Tensor input, Tensor output_q, Tensor output_s, int group_size,"
      " float eps, float fp8_min, float fp8_max, bool scale_ue8m0) -> ()");
  m.impl("sgl_per_token_group_quant_8bit", torch::kXPU, &at::native::xpu::sgl_per_token_group_quant_8bit);
  m.def(
      "sgl_per_token_group_quant_8bit_v2(Tensor input, Tensor output_q, Tensor output_s, int group_size,"
      " float eps, float fp8_min, float fp8_max, bool scale_ue8m0, bool fuse_silu_and_mul, Tensor? masked_m) -> ()");
  m.impl("sgl_per_token_group_quant_8bit_v2", torch::kXPU, &at::native::xpu::sgl_per_token_group_quant_8bit_v2);
  m.def(
      "sgl_per_token_group_quant_fp4(Tensor input, Tensor output_q, Tensor output_s, int group_size,"
      " float eps, Tensor? input_secondary=None) -> ()");
  m.impl("sgl_per_token_group_quant_fp4", torch::kXPU, &at::native::xpu::sgl_per_token_group_quant_fp4);
  m.def("sgl_per_tensor_quant_fp8(Tensor input, Tensor output_q, Tensor output_s, bool is_static) -> ()");
  m.impl("sgl_per_tensor_quant_fp8", torch::kXPU, &sgl_per_tensor_quant_fp8);

  /*
   * From fused qk norm rope
   */
  m.def(
      "fused_qk_norm_rope(Tensor! qkv, int num_heads_q, int num_heads_k, int num_heads_v, int head_dim, "
      "float eps, Tensor! q_weight, Tensor! k_weight, float base, bool is_neox, Tensor! position_ids, "
      "float factor, float low, float high, float attention_factor, int rotary_dim) -> ()");
  m.impl("fused_qk_norm_rope", torch::kXPU, &at::native::xpu::fused_qk_norm_rope);
  m.def(
      "fused_qk_norm_rope_with_cos_sin_cache_inplace(Tensor! q, Tensor! k, Tensor q_weight, Tensor k_weight, "
      "Tensor cos_sin_cache, Tensor positions, bool is_neox, float eps) -> ()");
  m.impl(
      "fused_qk_norm_rope_with_cos_sin_cache_inplace",
      torch::kXPU,
      &at::native::xpu::fused_qk_norm_rope_with_cos_sin_cache_inplace);
  /*
   * Fused QK RoPE (no RMS_Norm)
   */
  m.def(
      "fused_qk_rope(Tensor! qkv, int num_heads_q, int num_heads_k, int num_heads_v, int head_dim, "
      "Tensor! q_weight, Tensor! k_weight, float base, bool is_neox, Tensor! position_ids, "
      "float factor, float low, float high, float attention_factor, int rotary_dim) -> ()");
  m.impl("fused_qk_rope", torch::kXPU, &at::native::xpu::fused_qk_rope);

  m.def(
      "fused_qk_rope_with_cos_sin_cache_inplace(Tensor! q, Tensor! k, Tensor! cos_sin_cache, Tensor! positions, int "
      "rope_dim, "
      "bool is_neox) -> ()");
  m.impl(
      "fused_qk_rope_with_cos_sin_cache_inplace",
      torch::kXPU,
      &at::native::xpu::fused_qk_rope_with_cos_sin_cache_inplace);

  m.def(
      "multimodal_rotary_embedding(Tensor! query, Tensor! key, Tensor cos_sin_cache, Tensor positions, "
      "int[] mrope_section, int head_size, int rotary_dim, bool mrope_interleaved, bool mrope_interleaved_glm, "
      "bool is_neox_style, Tensor? axis_map) -> ()");
  m.impl("multimodal_rotary_embedding", torch::kXPU, &at::native::xpu::multimodal_rotary_embedding);

  /* utils */
  m.def("query_device(int device_id) -> (int, int)");
  m.impl("query_device", c10::DispatchKey::BackendSelect, &query_device);

  /* HC SPLIT SINKHORN */
  m.def(
      "hc_split_sinkhorn(Tensor mixes, Tensor hc_scale, Tensor hc_base, "
      "Tensor! pre, Tensor! post, Tensor! comb, "
      "int hc_mult, int sinkhorn_iters, float eps) -> ()");
  m.impl("hc_split_sinkhorn", torch::kXPU, &hc_split_sinkhorn);

  /* HC PRE BIG FUSE */
  m.def(
      "hc_pre_big_fuse(Tensor gemm_out_mul, Tensor gemm_out_sqrsum, "
      "Tensor hc_scale, Tensor hc_base, Tensor residual_flat, "
      "Tensor! post_mix, Tensor! comb_mix, Tensor! layer_input, "
      "int hc_mult, int sinkhorn_iters, int n_splits, "
      "float rms_eps, float hc_pre_eps, float hc_sinkhorn_eps, float hc_post_mult_value, "
      "Tensor? norm_weight=None, float? norm_eps=None) -> ()");
  m.impl("hc_pre_big_fuse", torch::kXPU, &hc_pre_big_fuse);

  /* HC PRE GEMM + SQUARE SUM */
  m.def("hc_pre_gemm_sqr_sum(Tensor! C, Tensor! sqr_sum, Tensor A, Tensor B) -> ()");
  m.impl("hc_pre_gemm_sqr_sum", torch::kXPU, &hc_pre_gemm_sqr_sum);

  /* HC POST */
  m.def("hc_post(Tensor x, Tensor residual, Tensor post_layer_mix, Tensor comb_res_mix, Tensor! out) -> ()");
  m.impl("hc_post", torch::kXPU, &hc_post);

  /*
   * From LoRA
   */
  m.def(
      "embedding_lora_a_fwd(Tensor! output, Tensor input_ids, Tensor weights, int vocab_size, Tensor seg_indptr, "
      "Tensor weight_indices, "
      "Tensor lora_ranks, Tensor? extra_embeddings, Tensor? seg_lens) -> ()");
  m.impl("embedding_lora_a_fwd", torch::kXPU, &embedding_lora_a_fwd);

  /* NSA (Native Sparse Attention) indexer scoring */
  // fp8_mqa_logits (prefill) is implemented in pure Python via sgl_kernel.nsa.
  m.def(
      "fp8_paged_mqa_logits(Tensor q_fp8, Tensor kv_cache, Tensor weights, "
      "Tensor seq_lens, Tensor block_tables, Tensor? schedule_metadata, "
      "int max_seq_len, bool clean_logits) -> Tensor");
  m.impl("fp8_paged_mqa_logits", torch::kXPU, &fp8_paged_mqa_logits);
}

REGISTER_EXTENSION(common_ops)
