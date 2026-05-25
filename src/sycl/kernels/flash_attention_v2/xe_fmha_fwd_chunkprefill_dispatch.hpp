#pragma once

#include "sycl/kernels/flash_attention_v2/xe_fmha_fwd_chunkprefill_runner.hpp"

namespace chunkprefill {

extern template struct FmhaChunkPrefillDynamicRunner<64>;
extern template struct FmhaChunkPrefillDynamicRunner<96>;
extern template struct FmhaChunkPrefillDynamicRunner<128>;
extern template struct FmhaChunkPrefillDynamicRunner<192>;
extern template struct FmhaChunkPrefillDynamicRunner<256>;
extern template struct FmhaChunkPrefillDynamicRunner<512>;

template <int HEAD_DIM>
inline void run_chunkprefill_kernel(const Arguments& params) {
  FmhaChunkPrefillDynamicRunner<HEAD_DIM>{}(params);
}

inline void dispatch_chunkprefill_kernel(const Arguments& params) {
  switch (params.d) {
    case 64:
      run_chunkprefill_kernel<64>(params);
      return;
    case 96:
      run_chunkprefill_kernel<96>(params);
      return;
    case 128:
      run_chunkprefill_kernel<128>(params);
      return;
    case 192:
      run_chunkprefill_kernel<192>(params);
      return;
    case 256:
      run_chunkprefill_kernel<256>(params);
      return;
    case 512:
      run_chunkprefill_kernel<512>(params);
      return;
    default:
      TORCH_CHECK(false, "Unsupported head size ", params.d, " for chunk-prefill MHA");
  }
}

inline int get_chunkprefill_q_tile_size(int head_size) {
  return head_size <= 96 ? 128 : 256;
}

inline at::Tensor
make_cumulative_q_blocks(const at::Tensor& cu_seqlens_q, int batch_size, int q_tile_size, int& total_q_blocks) {
  at::Tensor cu_seqlens_q_cpu = cu_seqlens_q.to(torch::kCPU);
  auto cu_seqlens_q_ptr = cu_seqlens_q_cpu.data_ptr<int>();
  at::Tensor cumulative_q_blocks_cpu = torch::empty({batch_size + 1}, torch::TensorOptions().dtype(torch::kInt32));
  auto cumulative_q_blocks_ptr = cumulative_q_blocks_cpu.data_ptr<int>();
  cumulative_q_blocks_ptr[0] = 0;
  for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    int seq_len_q = cu_seqlens_q_ptr[batch_idx + 1] - cu_seqlens_q_ptr[batch_idx];
    int q_blocks = (seq_len_q + q_tile_size - 1) / q_tile_size;
    cumulative_q_blocks_ptr[batch_idx + 1] = cumulative_q_blocks_ptr[batch_idx] + q_blocks;
  }
  total_q_blocks = cumulative_q_blocks_ptr[batch_size];
  return cumulative_q_blocks_cpu.to(cu_seqlens_q.device(), /*non_blocking=*/false);
}

std::vector<at::Tensor> mha_fwd(
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
    int const sm_margin) {
  auto q_type = q.scalar_type();
  TORCH_CHECK(
      q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16,
      "SGL Kernel XPU only supports fp16 and bf16 type");
  TORCH_CHECK(k.scalar_type() == q_type, "query and key must have the same dtype");
  TORCH_CHECK(v.scalar_type() == q_type, "query and value must have the same dtype");
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(v);
  TORCH_CHECK(q.dim() == 3, "query must be in ragged format");
  CHECK_INPUT(cu_seqlens_q);
  CHECK_INPUT(cu_seqlens_k);
  TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype torch.int32");
  TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype torch.int32");
  TORCH_CHECK(!q_v_.has_value(), "q_v is not supported yet");
  TORCH_CHECK(!rotary_cos_.has_value() && !rotary_sin_.has_value(), "rotary is not supported by v2 chunk-prefill yet");
  TORCH_CHECK(!kv_batch_idx_.has_value(), "kv_batch_idx is not supported by v2 chunk-prefill yet");
  TORCH_CHECK(!leftpad_k_.has_value(), "leftpad_k is not supported by v2 chunk-prefill yet");
  TORCH_CHECK(
      !q_descale_.has_value() && !k_descale_.has_value() && !v_descale_.has_value(),
      "fp8 scale tensors are not supported by v2 chunk-prefill yet");

  const int batch_size = cu_seqlens_q.size(0) - 1;
  const int seqlen_q = max_seqlen_q;
  const int total_q = q.size(0);
  const int num_heads = q.size(-2);
  const int head_size = q.size(-1);
  const int head_size_v = v.size(-1);
  const int num_heads_k = k.size(-2);
  const bool has_page_table = page_table.has_value();
  const int num_pages = has_page_table ? k.size(0) : 0;
  const int page_size = has_page_table ? k.size(1) : 0;
  const int max_num_pages_per_seq = has_page_table ? page_table.value().size(1) : 0;
  const int batch_size_k = has_page_table ? page_table.value().size(0) : cu_seqlens_k.size(0) - 1;
  const int seqlen_k = has_page_table ? max_num_pages_per_seq * page_size : max_seqlen_k;
  const int total_k = has_page_table ? num_pages * page_size : k.size(0);

  if (has_page_table) {
    CHECK_INPUT(page_table.value());
    TORCH_CHECK(page_table.value().dtype() == torch::kInt32, "page_table must have dtype torch.int32");
    TORCH_CHECK(page_table.value().stride(-1) == 1, "page_table must have contiguous last dimension");
    CHECK_SHAPE(page_table.value(), batch_size_k, max_num_pages_per_seq);
    CHECK_SHAPE(k, num_pages, page_size, num_heads_k, head_size);
    CHECK_SHAPE(v, num_pages, page_size, num_heads_k, head_size_v);
  } else {
    CHECK_SHAPE(k, total_k, num_heads_k, head_size);
    CHECK_SHAPE(v, total_k, num_heads_k, head_size_v);
  }
  TORCH_CHECK(batch_size == batch_size_k, "batch_size must be equal to batch_size_k");
  TORCH_CHECK(head_size <= 512, "FlashAttention forward only supports head dimension at most 512");
  TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
  TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
  TORCH_CHECK(head_size_v % 8 == 0, "head_size_v should be a multiple of 8");

  if (window_size_left >= seqlen_k - 1) {
    window_size_left = -1;
  }
  window_size_right = min(window_size_right, seqlen_q);
  if (is_causal) {
    window_size_right = 0;
  }

  at::Tensor out = torch::empty({total_q, num_heads, head_size_v}, q.options());
  at::Tensor softmax_lse = torch::empty({num_heads, total_q}, q.options().dtype(at::kFloat));
  c10::DeviceGuard device_guard(q.device());

  Arguments params;
  params.is_bf16 = q.dtype() == torch::kBFloat16;
  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  params.q_row_stride = q.stride(-3);
  params.k_row_stride = k.stride(-3);
  params.v_row_stride = v.stride(-3);
  params.q_head_stride = q.stride(-2);
  params.k_head_stride = k.stride(-2);
  params.v_head_stride = v.stride(-2);
  params.v_dim_stride = v.stride(-1);
  params.o_ptr = out.data_ptr();
  params.o_row_stride = out.stride(-3);
  params.o_head_stride = out.stride(-2);
  params.softmax_lse_ptr = softmax_lse.data_ptr();
  params.cu_seqlens_q = const_cast<int*>(cu_seqlens_q.data_ptr<int>());
  params.cu_seqlens_k = const_cast<int*>(cu_seqlens_k.data_ptr<int>());
  params.cu_seqlens_knew = nullptr;
  at::Tensor cumulative_q_blocks = make_cumulative_q_blocks(
      cu_seqlens_q, batch_size, get_chunkprefill_q_tile_size(head_size), params.total_q_blocks);
  params.cu_q_blocks = cumulative_q_blocks.data_ptr<int>();
  params.b = batch_size;
  params.h = num_heads;
  params.h_k = num_heads_k;
  params.q_group_size = 1;
  params.seqlen_q = seqlen_q;
  params.seqlen_k = seqlen_k;
  params.seqlen_knew = 0;
  params.d = head_size;
  params.d_rounded = round_up_headdim(head_size);
  params.dv = head_size_v;
  params.dv_rounded = head_size_v == head_size ? params.d_rounded : round_up_headdim(head_size_v);
  params.total_q = total_q;
  params.total_k = total_k;
  params.total_knew = 0;
  params.b_k = batch_size_k;
  params.softmax_scale = softmax_scale_;
  params.softmax_sink_ptr = sinks_.has_value() ? sinks_.value().data_ptr() : nullptr;
  params.softcap = softcap;
  params.p_dropout = 1.f;
  params.is_causal = window_size_left < 0 && window_size_right == 0;
  params.is_local = (window_size_left >= 0 || window_size_right >= 0) && !params.is_causal;
  if (window_size_left < 0) {
    window_size_left = seqlen_k - 1;
  }
  if (window_size_right < 0) {
    window_size_right = seqlen_q - 1;
  }
  params.window_size_left = window_size_left;
  params.window_size_right = window_size_right;
  params.page_table = has_page_table ? page_table.value().data_ptr<int>() : nullptr;
  params.page_table_batch_stride = has_page_table ? page_table.value().stride(0) : 0;
  params.max_num_pages_per_seq = max_num_pages_per_seq;
  params.page_size = page_size;
  params.num_pages = num_pages;
  params.tensor_opts = torch::TensorOptions().dtype(torch::kUInt8).device(q.device());
  params.num_kv_splits = std::max(1, num_kv_splits);

  dispatch_chunkprefill_kernel(params);
  at::Tensor out_accum;
  at::Tensor softmax_lse_accum;
  return {out, softmax_lse, out_accum, softmax_lse_accum};
}

}  // namespace chunkprefill
