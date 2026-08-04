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

  m.def(
      "hash_topk(Tensor router_logits, Tensor input_ids, Tensor tid2eid, Tensor! topk_weights, Tensor! topk_ids, "
      "float routed_scaling_factor=1.0) -> ()");
  m.impl("hash_topk", torch::kXPU, &at::native::xpu::hash_topk);

  m.def("top_k_renorm_probs(Tensor probs, Tensor! renorm_probs, Tensor? maybe_top_k_arr, int top_k_val) -> ()");
  m.impl("top_k_renorm_probs", torch::kXPU, &top_k_renorm_probs);

  m.def(
      "min_p_sampling_from_probs(Tensor probs, Tensor! output, Tensor? maybe_indices, Tensor? "
      "maybe_min_p_arr, float min_p_val, bool deterministic, Generator? gen) -> ()");
  m.impl("min_p_sampling_from_probs", torch::kXPU, &min_p_sampling_from_probs);

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
      "biased_topk(Tensor input, Tensor bias, Tensor! output, Tensor! indices, int topk, int scoring_func, int "
      "num_fused_shared_experts, bool renormalize, float routed_scaling_factor, bool "
      "apply_routed_scaling_factor_on_output) -> ()");
  m.impl("biased_topk", torch::kXPU, &at::native::xpu::biased_topk);

  m.def(
      "rotary_embedding(Tensor positions, Tensor query, Tensor key, int head_size, Tensor cos_sin_cache, "
      "bool is_neox) -> (Tensor, Tensor)");
  m.impl("rotary_embedding", torch::kXPU, &at::native::xpu::rotary_embedding);

  m.def(
      "store_cache(Tensor k, Tensor v, Tensor(a!) k_cache, Tensor(b!) v_cache, "
      "Tensor indices) -> ()");
  m.impl("store_cache", torch::kXPU, &at::native::xpu::store_cache);

  // KV cache transfer ops
  m.def(
      "transfer_kv_per_layer(Tensor src_k, Tensor(a!) dst_k, Tensor src_v, Tensor(b!) dst_v, "
      "Tensor src_indices, Tensor dst_indices, int item_size, int block_quota, int sgs_per_wg) -> ()");
  m.impl("transfer_kv_per_layer", torch::kXPU, &transfer_kv_per_layer);

  m.def(
      "transfer_kv_per_layer_mla(Tensor src, Tensor(a!) dst, "
      "Tensor src_indices, Tensor dst_indices, int item_size, int block_quota, int sgs_per_wg) -> ()");
  m.impl("transfer_kv_per_layer_mla", torch::kXPU, &transfer_kv_per_layer_mla);

  m.def(
      "transfer_kv_all_layer(Tensor src_k_layers, Tensor(a!) dst_k_layers, "
      "Tensor src_v_layers, Tensor(b!) dst_v_layers, "
      "Tensor src_indices, Tensor dst_indices, int item_size, int num_layers, "
      "int block_quota, int sgs_per_wg) -> ()");
  m.impl("transfer_kv_all_layer", torch::kXPU, &transfer_kv_all_layer);

  m.def(
      "transfer_kv_all_layer_mla(Tensor src_layers, Tensor(a!) dst_layers, "
      "Tensor src_indices, Tensor dst_indices, int item_size, int num_layers, "
      "int block_quota, int sgs_per_wg) -> ()");
  m.impl("transfer_kv_all_layer_mla", torch::kXPU, &transfer_kv_all_layer_mla);

  m.def(
      "transfer_kv_all_layer_lf_ph(Tensor src_k_layers, Tensor(a!) dst_k, "
      "Tensor src_v_layers, Tensor(b!) dst_v, "
      "Tensor src_indices, Tensor dst_indices, int item_size, int dst_layout_dim, "
      "int num_layers, int page_size, int head_num, int block_quota, int sgs_per_wg) -> ()");
  m.impl("transfer_kv_all_layer_lf_ph", torch::kXPU, &transfer_kv_all_layer_lf_ph);

  m.def(
      "transfer_kv_per_layer_ph_lf(Tensor src_k, Tensor(a!) dst_k, "
      "Tensor src_v, Tensor(b!) dst_v, "
      "Tensor src_indices, Tensor dst_indices, int layer_id, int item_size, int src_layout_dim, "
      "int page_size, int head_num, int block_quota, int sgs_per_wg) -> ()");
  m.impl("transfer_kv_per_layer_ph_lf", torch::kXPU, &transfer_kv_per_layer_ph_lf);

  m.def(
      "transfer_kv_per_layer_pf_lf(Tensor src_k, Tensor(a!) dst_k, Tensor src_v, Tensor(b!) dst_v, "
      "Tensor src_indices, Tensor dst_indices, int layer_id, int item_size, int src_layout_dim, "
      "int block_quota, int sgs_per_wg) -> ()");
  m.impl("transfer_kv_per_layer_pf_lf", torch::kXPU, &transfer_kv_per_layer_pf_lf);

  m.def(
      "transfer_kv_all_layer_lf_pf(Tensor src_k_layers, Tensor(a!) dst_k, "
      "Tensor src_v_layers, Tensor(b!) dst_v, "
      "Tensor src_indices, Tensor dst_indices, int item_size, int dst_layout_dim, "
      "int num_layers, int block_quota, int sgs_per_wg) -> ()");
  m.impl("transfer_kv_all_layer_lf_pf", torch::kXPU, &transfer_kv_all_layer_lf_pf);

  m.def(
      "transfer_kv_per_layer_mla_pf_lf(Tensor src, Tensor(a!) dst, "
      "Tensor src_indices, Tensor dst_indices, int layer_id, int item_size, int src_layout_dim, "
      "int block_quota, int sgs_per_wg) -> ()");
  m.impl("transfer_kv_per_layer_mla_pf_lf", torch::kXPU, &transfer_kv_per_layer_mla_pf_lf);

  m.def(
      "transfer_kv_all_layer_mla_lf_pf(Tensor src_layers, Tensor(a!) dst, "
      "Tensor src_indices, Tensor dst_indices, int item_size, int dst_layout_dim, "
      "int num_layers, int block_quota, int sgs_per_wg) -> ()");
  m.impl("transfer_kv_all_layer_mla_lf_pf", torch::kXPU, &transfer_kv_all_layer_mla_lf_pf);

#ifdef USE_MOE
  m.def(
      "moe_fused_gate(Tensor input, Tensor? bias, int num_expert_group, int topk_group, int topk, int "
      "num_fused_shared_experts, int scoring_func, bool renormalize, float routed_scaling_factor, bool "
      "apply_routed_scaling_factor_on_output) -> "
      "(Tensor[])");
  m.impl("moe_fused_gate", torch::kXPU, &moe_fused_gate);
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
      "moe_grouped_mm_nt_xe20_w4a16(Tensor! output, Tensor activations, Tensor packed_weights, Tensor scales, "
      "Tensor? zeros, Tensor? bias, Tensor rows_per_expert, int n_experts, bool is_int4, int group_size) -> ()");
  m.impl("moe_grouped_mm_nt_xe20_w4a16", torch::kXPU, &moe_grouped_mm_nt_xe20_w4a16);

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
#endif  // USE_MOE

  m.def("merge_state_v2(Tensor v_a, Tensor s_a, Tensor v_b, Tensor s_b, Tensor! v_merged, Tensor! s_merged) -> ()");
  m.impl("merge_state_v2", torch::kXPU, &merge_state_v2);
  m.def("merge_state(Tensor v_a, Tensor s_a, Tensor v_b, Tensor s_b, Tensor! v_merged, Tensor! s_merged) -> ()");
  m.impl("merge_state", torch::kXPU, &merge_state);

  /*
   * Inkling short convolution
   */
  m.def(
      "inkling_sconv_forward(Tensor x, Tensor weight, Tensor sconv_cache, Tensor cache_mask, Tensor safe_idx, "
      "Tensor cu, Tensor si, bool silu_activation, bool use_residual, bool is_decode) -> Tensor");
  m.impl("inkling_sconv_forward", torch::kXPU, &inkling_sconv_forward);

  m.def(
      "inkling_update_sconv_cache(Tensor x, Tensor(a!) sconv_cache, Tensor cache_indices, "
      "Tensor has_initial_state, Tensor query_start_loc) -> ()");
  m.impl("inkling_update_sconv_cache", torch::kXPU, &inkling_update_sconv_cache);

  m.def(
      "inkling_fused_decode_update_sconv(Tensor x, Tensor weight, Tensor(a!) sconv_cache, "
      "Tensor cache_indices, Tensor cache_mask, bool silu_activation, bool use_residual, "
      "Tensor? track_mask=None, Tensor? track_indices=None) -> Tensor");
  m.impl("inkling_fused_decode_update_sconv", torch::kXPU, &inkling_fused_decode_update_sconv);

  m.def(
      "inkling_gather_scatter_sconv_cache(Tensor hidden_states, Tensor(a!) sconv_cache, "
      "Tensor track_conv_indices, Tensor mask, Tensor dst_indices) -> ()");
  m.impl("inkling_gather_scatter_sconv_cache", torch::kXPU, &inkling_gather_scatter_sconv_cache);

  m.def(
      "inkling_draft_extend_sconv_cache(Tensor hidden_states, Tensor(a!) sconv_cache, Tensor cache_indices, "
      "Tensor num_accepted_tokens, int draft_token_num, bool do_tracking, Tensor? crossed=None, "
      "Tensor? track_step=None, Tensor? mamba_track_indices=None) -> ()");
  m.impl("inkling_draft_extend_sconv_cache", torch::kXPU, &inkling_draft_extend_sconv_cache);

  m.def(
      "inkling_fused_decode_sconv_metadata(int B, Tensor cache_indices, "
      "Tensor(a!)? query_start_loc_out=None, Tensor(b!)? has_initial_state_out=None, "
      "Tensor(c!)? cache_mask_out=None, Tensor(d!)? safe_idx_out=None, "
      "Tensor(e!)? cu_out=None, Tensor(f!)? si_out=None) -> "
      "(Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!), Tensor(e!), Tensor(f!))");
  m.impl("inkling_fused_decode_sconv_metadata", torch::kXPU, &inkling_fused_decode_sconv_metadata);

  m.def(
      "inkling_fused_extend_sconv_metadata(int B, int T, Tensor cache_indices, int his_mode, "
      "Tensor? extend_seq_lens=None, Tensor? his_src=None, int draft_token_num=1, "
      "Tensor(a!)? query_start_loc_out=None, Tensor(b!)? has_initial_state_out=None, "
      "Tensor(c!)? cache_mask_out=None, Tensor(d!)? safe_idx_out=None, "
      "Tensor(e!)? cu_out=None, Tensor(f!)? si_out=None) -> "
      "(Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!), Tensor(e!), Tensor(f!))");
  m.impl("inkling_fused_extend_sconv_metadata", torch::kXPU, &inkling_fused_extend_sconv_metadata);

  m.def(
      "inkling_track_conv_indices(Tensor query_start_loc, Tensor mamba_track_seqlens, Tensor extend_prefix_lens, "
      "int width_minus_one, int chunk_size, int total_tokens) -> Tensor");
  m.impl("inkling_track_conv_indices", torch::kXPU, &inkling_track_conv_indices);

  m.def(
      "inkling_save_intermediate_conv_windows(Tensor sconv_cache, Tensor hidden_states, Tensor cache_indices, "
      "Tensor(a!) intermediate_out, int batch_size, int draft_token_num) -> ()");
  m.impl("inkling_save_intermediate_conv_windows", torch::kXPU, &inkling_save_intermediate_conv_windows);

  /*
   * From cutlass attention
   */
#ifdef USE_FMHA
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
      "    Tensor(a!)?  out=None) -> (Tensor(a!), Tensor, Tensor, Tensor)");
  m.impl("fwd", torch::kXPU, make_pytorch_shim(&mha_fwd));
#endif  // USE_FMHA

#ifdef USE_MLA
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
  m.impl("flash_mla_prefill", torch::kXPU, &flash_mla_prefill);

  m.def(
      "flash_mla_sparse_prefill(Tensor! out, Tensor! max_logits, Tensor! lse, Tensor! q, Tensor! kv, "
      "Tensor! indices, float sm_scale, int head_dim_v, "
      "Tensor? attn_sink=None, Tensor? topk_length=None) -> ()");
  m.impl("flash_mla_sparse_prefill", torch::kXPU, &flash_mla_sparse_prefill);
#endif  // USE_MLA

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

  m.def("sgl_per_token_quant_fp8(Tensor input, Tensor(a!) output_q, Tensor(b!) output_s) -> ()");
  m.impl("sgl_per_token_quant_fp8", torch::kXPU, &sgl_per_token_quant_fp8);

  /*
   * From fused qk norm rope
   */
  m.def(
      "fused_qk_norm_rope(Tensor! qkv, int num_heads_q, int num_heads_k, int num_heads_v, int head_dim, "
      "float eps, Tensor! q_weight, Tensor! k_weight, float base, bool is_neox, Tensor! position_ids, "
      "float factor, float low, float high, float attention_factor, int rotary_dim) -> ()");
  m.impl("fused_qk_norm_rope", torch::kXPU, &at::native::xpu::fused_qk_norm_rope);
  m.def(
      "fused_inplace_qknorm_rope(Tensor! q, Tensor! k, Tensor q_weight, Tensor k_weight, "
      "Tensor cos_sin_cache, Tensor positions, bool is_neox, float eps, int head_dim=0, int rope_dim=0) -> ()");
  m.impl("fused_inplace_qknorm_rope", torch::kXPU, &at::native::xpu::fused_inplace_qknorm_rope);
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
  m.def(
      "sgemm_lora_a_fwd(Tensor! output, Tensor input_x, Tensor weights, int stack_num, Tensor seg_indptr, "
      "Tensor weight_indices, "
      "Tensor lora_ranks, Tensor? seg_lens) -> ()");
  m.impl("sgemm_lora_a_fwd", torch::kXPU, &sgemm_lora_a_fwd);

  /* NSA (Native Sparse Attention) indexer scoring */
  // fp8_mqa_logits (prefill) is implemented in pure Python via sgl_kernel.nsa.
  m.def(
      "fp8_paged_mqa_logits(Tensor q_fp8, Tensor kv_cache, Tensor weights, "
      "Tensor seq_lens, Tensor block_tables, Tensor? schedule_metadata, "
      "int max_seq_len, bool clean_logits) -> Tensor");
  m.impl("fp8_paged_mqa_logits", torch::kXPU, &fp8_paged_mqa_logits);

  /*
   * From GDN (Gated DeltaNet) attention (Intel Xe2)
   */
  m.def(
      "gdn_attention(Tensor! core_attn_out, Tensor! z, Tensor projected_states_qkvz, Tensor projected_states_ba, "
      "int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim, "
      "Tensor! conv_state, Tensor! ssm_state, Tensor conv_weights, Tensor? conv_bias, str activation, Tensor A_log, "
      "Tensor dt_bias, int num_prefills, int num_decodes, int num_spec_decodes, Tensor? has_initial_state, "
      "Tensor? non_spec_query_start_loc, Tensor? non_spec_token_indx, Tensor? non_spec_state_indices_tensor, "
      "Tensor? spec_query_start_loc, Tensor? spec_token_indx, Tensor? spec_state_indices_tensor, "
      "Tensor? num_accepted_tokens, int num_actual_tokens, int tp_size, bool reorder_input) -> ()");
  m.impl("gdn_attention", torch::kXPU, &gdn_attention);

  /*
   * Mamba causal conv1d (XPU)
   */
  m.def(
      "causal_conv1d_fwd(Tensor! x, Tensor weight, Tensor? bias_, Tensor(a!)? conv_states, "
      "Tensor? query_start_loc, Tensor? cache_indices, Tensor? has_initial_state, "
      "bool silu_activation, int pad_slot_id) -> ()");
  m.impl("causal_conv1d_fwd", torch::kXPU, &causal_conv1d_fwd);

  m.def(
      "causal_conv1d_update(Tensor! x, Tensor! conv_state, Tensor weight, Tensor? bias_, "
      "bool silu_activation, Tensor? cache_seqlens_, Tensor? conv_state_indices_, "
      "int pad_slot_id) -> ()");
  m.impl("causal_conv1d_update", torch::kXPU, &causal_conv1d_update);

  /*
   * Compress plan kernels
   */
  m.def(
      "plan_compress_prefill(Tensor req_pool_indices, Tensor req_to_token, Tensor full_to_state, "
      "Tensor seq_lens, Tensor extend_lens, Tensor pin_buffer, int num_q_tokens, int compress_ratio, int "
      "swa_page_size, "
      "int ring_size, bool use_cuda_graph=False) -> (Tensor, Tensor)");
  m.impl("plan_compress_prefill", torch::kXPU, &at::native::xpu::plan_compress_prefill);

  m.def(
      "plan_compress_decode(Tensor req_pool_indices, Tensor req_to_token, Tensor full_to_state, "
      "Tensor seq_lens, int compress_ratio, int swa_page_size, int ring_size) -> Tensor");
  m.impl("plan_compress_decode", torch::kXPU, &at::native::xpu::plan_compress_decode);

  m.def(
      "flash_compress128_decode(Tensor! kv_buffer, Tensor kv_input, Tensor! kv_output, Tensor ape, Tensor plan_d) "
      "-> ()");
  m.impl("flash_compress128_decode", torch::kXPU, &at::native::xpu::flash_compress128_decode);

  m.def(
      "flash_compress128_prefill(Tensor! kv_buffer, Tensor kv_input, Tensor! kv_output, Tensor ape, Tensor plan_c, "
      "Tensor plan_w) -> ()");
  m.impl("flash_compress128_prefill", torch::kXPU, &at::native::xpu::flash_compress128_prefill);
}

REGISTER_EXTENSION(common_ops)
