#pragma once

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>

#include <sycl/sycl.hpp>

#include "sycl/SYCLHelpers.h"
#include "sycl/kernels/flash_attention_v2/collective/fmha_relative_bias.hpp"

namespace flash_attention_v2::relative_attention {

inline constexpr int kQTile = 256;
inline constexpr int kKTile = 32;

inline constexpr int padded_cols(int extent) {
  return cutlass::fmha::collective::rel_bias_padded_cols(extent, kQTile, kKTile);
}

inline constexpr int decode_padded_cols(int extent, int k_tile) {
  return cutlass::fmha::collective::rel_bias_padded_cols(extent, /*m_drift=*/0, k_tile);
}

template <typename Element, int QTile>
struct ShearBiasKernel {
  const Element* input;
  Element* output;
  const int* cu_q;
  const int* cache_lengths;
  int heads;
  int max_seqlen_k;
  int cols;
  int extent;
  int k_tile;
  int64_t token_stride;
  int64_t head_stride;

  void operator()(sycl::id<3> index) const {
    const int b = index[0];
    const int h = index[1];
    const int q_idx = index[2] / cols;
    const int col = index[2] % cols;
    const int q_len = cu_q[b + 1] - cu_q[b];
    const int k_len = cache_lengths[b];
    if (q_idx >= q_len || k_len > max_seqlen_k) return;
    const int row_kv = k_len - q_len + q_idx;
    const int row_kv_first = row_kv - q_idx % QTile;
    const int col_origin = cutlass::fmha::collective::rel_bias_col_origin(row_kv_first, extent, k_tile);
    const int rel = row_kv - (col_origin + col);
    if (rel < 0 || rel >= extent) return;

    const int q_global = cu_q[b] + q_idx;
    output[(q_global * heads + h) * cols + col] = input[q_global * token_stride + h * head_stride + rel];
  }
};

template <typename Element, int QTile>
at::Tensor prepare_sheared_bias(
    const at::Tensor& rel_logits,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cache_seqlens,
    int max_seqlen_q,
    int max_seqlen_k,
    int cols,
    int k_tile) {
  const int extent = rel_logits.size(-1);
  auto bias = at::zeros({rel_logits.size(0), rel_logits.size(1), cols}, rel_logits.options());
  auto queue = c10::xpu::getCurrentXPUStream().queue();
  const int heads = rel_logits.size(1);
  const int64_t token_stride = rel_logits.stride(0);
  const int64_t head_stride = rel_logits.stride(1);
  const int batch = cu_seqlens_q.size(0) - 1;
  const auto* cu_q = cu_seqlens_q.data_ptr<int>();
  const auto* cache_lengths = cache_seqlens.data_ptr<int>();
  const auto* input = rel_logits.data_ptr<Element>();
  auto* output = bias.data_ptr<Element>();

  ShearBiasKernel<Element, QTile> kernel{
      input, output, cu_q, cache_lengths, heads, max_seqlen_k, cols, extent, k_tile, token_stride, head_stride};
  sycl_kernel_submit(sycl::range<3>(batch, heads, max_seqlen_q * cols), queue, kernel);
  return bias;
}

template <typename Element>
at::Tensor prepare_bias(
    const at::Tensor& rel_logits,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cache_seqlens,
    int max_seqlen_q,
    int max_seqlen_k) {
  const int extent = rel_logits.size(-1);
  return prepare_sheared_bias<Element, kQTile>(
      rel_logits, cu_seqlens_q, cache_seqlens, max_seqlen_q, max_seqlen_k, padded_cols(extent), kKTile);
}

template <typename Element>
at::Tensor prepare_decode_bias(
    const at::Tensor& rel_logits,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cache_seqlens,
    int max_seqlen_q,
    int max_seqlen_k,
    int k_tile) {
  const int extent = rel_logits.size(-1);
  return prepare_sheared_bias<Element, 1>(
      rel_logits, cu_seqlens_q, cache_seqlens, max_seqlen_q, max_seqlen_k, decode_padded_cols(extent, k_tile), k_tile);
}

}  // namespace flash_attention_v2::relative_attention
