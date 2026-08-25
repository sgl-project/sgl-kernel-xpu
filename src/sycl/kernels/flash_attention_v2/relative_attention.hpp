#pragma once

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>

#include <sycl/sycl.hpp>

#include "sycl/kernels/flash_attention_v2/collective/fmha_relative_bias.hpp"

namespace flash_attention_v2::relative_attention {

inline constexpr int kQTile = 256;
inline constexpr int kKTile = 32;

inline constexpr int padded_cols(int extent) {
  return cutlass::fmha::collective::rel_bias_padded_cols(extent, kQTile, kKTile);
}

template <typename Element>
class ShearBiasKernel;

template <typename Element>
at::Tensor prepare_bias(
    const at::Tensor& rel_logits,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cache_seqlens,
    int max_seqlen_q,
    int max_seqlen_k) {
  const int extent = rel_logits.size(-1);
  const int cols = padded_cols(extent);
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
  queue.parallel_for<ShearBiasKernel<Element>>(
      sycl::range<3>(batch, heads, max_seqlen_q * cols), [=](sycl::id<3> index) {
        const int b = index[0];
        const int h = index[1];
        const int q_idx = index[2] / cols;
        const int col = index[2] % cols;
        const int q_len = cu_q[b + 1] - cu_q[b];
        const int k_len = cache_lengths[b];
        if (q_idx >= q_len || k_len > max_seqlen_k) return;
        const int row_kv = k_len - q_len + q_idx;
        const int row_kv_first = row_kv - q_idx % kQTile;
        const int col_origin = cutlass::fmha::collective::rel_bias_col_origin(row_kv_first, extent, kKTile);
        const int rel = row_kv - (col_origin + col);
        if (rel >= 0 && rel < extent) {
          const int q_global = cu_q[b] + q_idx;
          output[(q_global * heads + h) * cols + col] = input[q_global * token_stride + h * head_stride + rel];
        }
      });
  return bias;
}

}  // namespace flash_attention_v2::relative_attention
