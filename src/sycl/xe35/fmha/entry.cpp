#include <limits>
#include "sycl/xe35/fmha/dispatch.hpp"

SGL_KERNEL_EXPORT void mha_fwd_upstream_xe35(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    std::optional<const at::Tensor>& q_v_,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    int max_seqlen_q,
    int max_seqlen_k,
    std::optional<const at::Tensor>& page_table,
    std::optional<const at::Tensor>& kv_batch_idx_,
    std::optional<const at::Tensor>& leftpad_k_,
    std::optional<const at::Tensor>& rotary_cos_,
    std::optional<const at::Tensor>& rotary_sin_,
    std::optional<const at::Tensor>& seqlens_rotary_,
    std::optional<at::Tensor>& q_descale_,
    std::optional<at::Tensor>& k_descale_,
    std::optional<at::Tensor>& v_descale_,
    const float softmax_scale_,
    std::optional<const at::Tensor>& sinks_,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    float const softcap,
    bool const is_rotary_interleaved,
    std::optional<at::Tensor>& scheduler_metadata_,
    int num_kv_splits,
    std::optional<bool> pack_gqa_,
    int const sm_margin,
    at::Tensor& out,
    std::optional<at::Tensor>& softmax_lse,
    std::optional<const at::Tensor>& rel_bias_,
    bool rel_bias_is_sheared,
    std::optional<const at::Tensor>& bidirectional_block_ids_);

SGL_KERNEL_EXPORT void mha_fwd(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    std::optional<const at::Tensor>& q_v_,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    int max_seqlen_q,
    int max_seqlen_k,
    std::optional<const at::Tensor>& page_table,
    std::optional<const at::Tensor>& kv_batch_idx_,
    std::optional<const at::Tensor>& leftpad_k_,
    std::optional<const at::Tensor>& rotary_cos_,
    std::optional<const at::Tensor>& rotary_sin_,
    std::optional<const at::Tensor>& seqlens_rotary_,
    std::optional<at::Tensor>& q_descale_,
    std::optional<at::Tensor>& k_descale_,
    std::optional<at::Tensor>& v_descale_,
    const float softmax_scale_,
    std::optional<const at::Tensor>& sinks_,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    float const softcap,
    bool const is_rotary_interleaved,
    std::optional<at::Tensor>& scheduler_metadata_,
    int num_kv_splits,
    std::optional<bool> pack_gqa_,
    int const sm_margin,
    at::Tensor& out,
    std::optional<at::Tensor>& softmax_lse,
    std::optional<const at::Tensor>& rel_bias_,
    bool rel_bias_is_sheared,
    std::optional<const at::Tensor>& bidirectional_block_ids_) {
  const int batch_xe35 = cu_seqlens_q.dim() == 1 ? static_cast<int>(cu_seqlens_q.numel()) - 1 : 0;
  const bool capacity_hint_xe35 = max_seqlen_k <= 0 && page_table.has_value() &&
      page_table->dim() == 2 && k.dim() == 4 && k.size(1) == 64 &&
      page_table->size(1) > 0 && page_table->size(1) <= std::numeric_limits<int>::max() / 64;
  const int resolved_max_seqlen_k_xe35 = capacity_hint_xe35
      ? static_cast<int>(page_table->size(1) * k.size(1)) : max_seqlen_k;
  const bool production_shape_xe35 =
      (max_seqlen_q == 1 && batch_xe35 == 8 &&
       ((resolved_max_seqlen_k_xe35 >= 4097 && resolved_max_seqlen_k_xe35 <= 4112) ||
        (capacity_hint_xe35 && resolved_max_seqlen_k_xe35 == 4160))) ||
      (max_seqlen_q == 4096 &&
       (batch_xe35 == 1 || batch_xe35 == 3 || batch_xe35 == 5 || batch_xe35 == 7 || batch_xe35 == 8) &&
       resolved_max_seqlen_k_xe35 == 4096);
  const bool use_port_xe35 = production_shape_xe35 &&
      q.dim() == 3 && q.size(0) == batch_xe35 * max_seqlen_q &&
      q.size(1) == 128 && q.size(2) == 64 && q.scalar_type() == at::kBFloat16 &&
      k.dim() == 4 && k.size(1) == 64 && k.size(2) == 16 && k.size(3) == 64 &&
      k.scalar_type() == at::kBFloat16 && v.scalar_type() == at::kBFloat16 && v.sizes() == k.sizes() &&
      q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && out.is_contiguous() &&
      out.scalar_type() == at::kBFloat16 && out.sizes() == q.sizes() && page_table.has_value() &&
      is_causal && (window_size_left == -1 || window_size_left == 2047) &&
      softcap == 0.0f && num_kv_splits == 0 && sm_margin == 0 &&
      !softmax_lse.has_value() && !q_v_.has_value() && !kv_batch_idx_.has_value() && !leftpad_k_.has_value() &&
      !rotary_cos_.has_value() && !rotary_sin_.has_value() && !seqlens_rotary_.has_value() &&
      !scheduler_metadata_.has_value() && !rel_bias_.has_value() && !bidirectional_block_ids_.has_value() &&
      !sinks_.has_value() && !q_descale_.has_value() && !k_descale_.has_value() && !v_descale_.has_value();
  if (!use_port_xe35) {
    mha_fwd_upstream_xe35(
      q, k, v, q_v_, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
      page_table, kv_batch_idx_, leftpad_k_, rotary_cos_, rotary_sin_, seqlens_rotary_,
      q_descale_, k_descale_, v_descale_, softmax_scale_, sinks_, is_causal,
      window_size_left, window_size_right, softcap, is_rotary_interleaved,
      scheduler_metadata_, num_kv_splits, pack_gqa_, sm_margin, out,
      softmax_lse, rel_bias_, rel_bias_is_sheared, bidirectional_block_ids_);
    return;
  }
  std::optional<at::Tensor> output_xe35 = out;
  mha_fwd_xe35(
      q, k, v, q_v_, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, resolved_max_seqlen_k_xe35,
      page_table, kv_batch_idx_, leftpad_k_, rotary_cos_, rotary_sin_, seqlens_rotary_,
      q_descale_, k_descale_, v_descale_, softmax_scale_, sinks_, is_causal,
      window_size_left, window_size_right, softcap, is_rotary_interleaved,
      scheduler_metadata_, num_kv_splits, pack_gqa_, sm_margin, output_xe35);
}
