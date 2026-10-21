#pragma once

#include "sycl/xe35/fmha/common.hpp"

namespace xe35_fmha {

template <int HeadDim, bool Paged = true, class Element = cutlass::bfloat16_t>
inline void run_xe35_prefill_xe35(
    const at::Tensor& q_dense,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& page_table,
    int num_pages_per_seq,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k_cache,
    at::Tensor& out_dense,
    int batch,
    int num_heads_q,
    int num_heads_kv,
    int max_seqlen_q,
    int max_seqlen_kv_cache,
    int page_size,
    float q_descale,
    float k_descale,
    float v_descale,
    float softmax_scale,
    bool is_causal,
    bool is_local,
    int window_size_left,
    int window_size_right,
    const cutlass::bfloat16_t* sm_sink,
    bool use_sink,
    int num_kv_splits = 1) {
  dispatch_xe3_kernel_xe35<HeadDim, false, Paged, Element, cutlass::bfloat16_t>(
      q_dense,
      k_pool,
      v_pool,
      page_table,
      num_pages_per_seq,
      cu_seqlens_q,
      cu_seqlens_k_cache,
      out_dense,
      batch,
      num_heads_q,
      num_heads_kv,
      max_seqlen_q,
      max_seqlen_kv_cache,
      page_size,
      q_descale,
      k_descale,
      v_descale,
      softmax_scale,
      is_causal,
      is_local,
      window_size_left,
      window_size_right,
      sm_sink,
      use_sink,
      num_kv_splits);
}

}  // namespace xe35_fmha
