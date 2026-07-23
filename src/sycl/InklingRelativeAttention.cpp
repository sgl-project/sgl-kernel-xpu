/* Copyright 2026 SGLang Team. All Rights Reserved.
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *
 * This file adapts the Inkling BMG relative-attention backend from
 * /data2/syk/cutlass-sycl/examples/17_bmg_relative_attention_backend for the
 * sgl-kernel XPU extension ABI.
 */

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <sycl/sycl.hpp>
#include <tuple>

#include "Utils.h"

namespace {

constexpr int kDefaultLocalSize = 128;
constexpr int kMaxLocalSize = 1024;
constexpr float kNegInf = -3.402823466e+38F;

template <typename scalar_t>
inline float to_float_device(scalar_t value) {
  return static_cast<float>(value);
}

template <typename scalar_t>
inline scalar_t from_float_device(float value) {
  return static_cast<scalar_t>(value);
}

inline int64_t next_power_of_2_i64(int64_t x) {
  int64_t value = 1;
  while (value < x) {
    value <<= 1;
  }
  return value;
}

inline bool is_power_of_2_i64(int64_t x) {
  return x > 0 && (x & (x - 1)) == 0;
}

template <typename scalar_t>
struct RelativeAttentionParams {
  scalar_t const* q = nullptr;
  scalar_t const* k = nullptr;
  scalar_t const* v = nullptr;
  float const* rel_bias = nullptr;
  int32_t const* q_to_seq = nullptr;
  int32_t const* q_pos = nullptr;
  int32_t const* cu_k = nullptr;
  scalar_t* out = nullptr;
  float* lse = nullptr;

  float scale = 1.0f;
  float softcap = 0.0f;
  int64_t total_q = 0;
  int64_t total_k = 0;
  int64_t batch = 0;
  int64_t heads = 0;
  int64_t kv_heads = 0;
  int64_t d = 0;
  int64_t dv = 0;
  int64_t rel_len = 0;
  int64_t q_stride_t = 0;
  int64_t q_stride_h = 0;
  int64_t k_stride_t = 0;
  int64_t k_stride_h = 0;
  int64_t v_stride_t = 0;
  int64_t v_stride_h = 0;
  int64_t o_stride_t = 0;
  int64_t o_stride_h = 0;
  int64_t bias_stride_t = 0;
  int64_t bias_stride_h = 0;
  int64_t window_left = -1;
  int64_t window_right = -1;
};

template <typename scalar_t, bool UseRelativeBias, bool UseWindow, bool UseCausal>
inline bool key_is_valid(RelativeAttentionParams<scalar_t> const& params, int64_t q_pos, int64_t k_pos) {
  bool valid = true;
  if constexpr (UseCausal) {
    valid = valid && (k_pos <= q_pos);
  }
  if constexpr (UseWindow) {
    if (params.window_left >= 0) {
      valid = valid && (k_pos >= q_pos - params.window_left);
    }
    if (params.window_right >= 0) {
      valid = valid && (k_pos <= q_pos + params.window_right);
    }
  }
  return valid;
}

template <typename scalar_t, bool UseRelativeBias, bool UseWindow, bool UseCausal>
inline float compute_score(
    RelativeAttentionParams<scalar_t> const& params,
    int64_t q_row,
    int64_t head,
    int64_t kv_head,
    int64_t q_abs_pos,
    int64_t k_global,
    int64_t k_pos) {
  if (!key_is_valid<scalar_t, UseRelativeBias, UseWindow, UseCausal>(params, q_abs_pos, k_pos)) {
    return kNegInf;
  }

  int64_t q_base = q_row * params.q_stride_t + head * params.q_stride_h;
  int64_t k_base = k_global * params.k_stride_t + kv_head * params.k_stride_h;
  float score = 0.0f;
  for (int64_t d = 0; d < params.d; ++d) {
    score = sycl::fma(to_float_device(params.q[q_base + d]), to_float_device(params.k[k_base + d]), score);
  }
  score *= params.scale;

  if constexpr (UseRelativeBias) {
    int64_t rel = q_abs_pos - k_pos;
    if (rel >= 0 && rel < params.rel_len) {
      int64_t bias_offset = q_row * params.bias_stride_t + head * params.bias_stride_h + rel;
      score += params.rel_bias[bias_offset];
    }
  }

  if (params.softcap > 0.0f) {
    score = params.softcap * sycl::tanh(score / params.softcap);
  }
  return score;
}

template <typename scalar_t, bool UseRelativeBias, bool UseWindow, bool UseCausal>
class InklingRelativeAttentionRowKernel;

template <typename scalar_t, bool UseRelativeBias, bool UseWindow, bool UseCausal>
sycl::event launch_relative_attention_static(
    sycl::queue& queue,
    RelativeAttentionParams<scalar_t> const& params,
    int local_size) {
  if (params.total_q == 0 || params.heads == 0) {
    return {};
  }

  int64_t groups = params.total_q * params.heads;
  int64_t global = groups * local_size;

  return queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> p_scratch(sycl::range<1>(static_cast<std::size_t>(local_size)), cgh);
    cgh.parallel_for<InklingRelativeAttentionRowKernel<scalar_t, UseRelativeBias, UseWindow, UseCausal>>(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<std::size_t>(global)),
            sycl::range<1>(static_cast<std::size_t>(local_size))),
        [=](sycl::nd_item<1> item) {
          sycl::sub_group sg = item.get_sub_group();
          int local_id = static_cast<int>(item.get_local_id(0));
          int sg_lane = static_cast<int>(sg.get_local_id());
          int sg_id = static_cast<int>(sg.get_group_id());
          int sg_size = static_cast<int>(sg.get_local_range()[0]);
          int sg_count = (local_size + sg_size - 1) / sg_size;
          int64_t group_id = static_cast<int64_t>(item.get_group(0));
          int64_t head = group_id % params.heads;
          int64_t q_row = group_id / params.heads;
          int64_t kv_group = params.heads / params.kv_heads;
          int64_t kv_head = head / kv_group;
          int32_t seq = params.q_to_seq[q_row];
          int64_t kv_begin = params.cu_k[seq];
          int64_t kv_end = params.cu_k[seq + 1];
          int64_t kv_len = kv_end - kv_begin;
          int64_t q_abs_pos = params.q_pos[q_row];
          int64_t valid_begin = 0;
          int64_t valid_end = kv_len;
          if constexpr (UseCausal) {
            int64_t causal_end = q_abs_pos + 1;
            valid_end = valid_end < causal_end ? valid_end : causal_end;
          }
          if constexpr (UseWindow) {
            if (params.window_left >= 0) {
              int64_t window_begin = q_abs_pos - params.window_left;
              valid_begin = valid_begin > window_begin ? valid_begin : window_begin;
            }
            if (params.window_right >= 0) {
              int64_t window_end = q_abs_pos + params.window_right + 1;
              valid_end = valid_end < window_end ? valid_end : window_end;
            }
          }
          valid_begin = valid_begin < 0 ? 0 : valid_begin;
          valid_begin = valid_begin > kv_len ? kv_len : valid_begin;
          valid_end = valid_end < valid_begin ? valid_begin : valid_end;
          valid_end = valid_end > kv_len ? kv_len : valid_end;
          int64_t valid_len = valid_end - valid_begin;

          float e_max = kNegInf;
          float denom = 0.0f;
          float acc = 0.0f;
          bool owns_value = local_id < params.dv;

          for (int64_t tile_begin = 0; tile_begin < valid_len; tile_begin += local_size) {
            int64_t remaining = valid_len - tile_begin;
            int tile_count = static_cast<int>(remaining < local_size ? remaining : local_size);
            int64_t k_local = valid_begin + tile_begin + local_id;
            float score = kNegInf;
            if (local_id < tile_count) {
              score = compute_score<scalar_t, UseRelativeBias, UseWindow, UseCausal>(
                  params, q_row, head, kv_head, q_abs_pos, kv_begin + k_local, k_local);
            }

            float sg_max = sycl::reduce_over_group(sg, score, sycl::maximum<float>());
            if (sg_lane == 0) {
              p_scratch[sg_id] = sg_max;
            }
            item.barrier(sycl::access::fence_space::local_space);
            if (local_id == 0) {
              float reduced = kNegInf;
              for (int i = 0; i < sg_count; ++i) {
                float candidate = p_scratch[i];
                reduced = candidate > reduced ? candidate : reduced;
              }
              p_scratch[0] = reduced;
            }
            item.barrier(sycl::access::fence_space::local_space);
            float tile_max = p_scratch[0];
            item.barrier(sycl::access::fence_space::local_space);

            float n_e_max = tile_max > e_max ? tile_max : e_max;
            float re_scale = sycl::exp(e_max - n_e_max);
            float p = local_id < tile_count ? sycl::exp(score - n_e_max) : 0.0f;
            float sg_sum = sycl::reduce_over_group(sg, p, sycl::plus<float>());
            if (sg_lane == 0) {
              p_scratch[sg_id] = sg_sum;
            }
            item.barrier(sycl::access::fence_space::local_space);
            if (local_id == 0) {
              float reduced = 0.0f;
              for (int i = 0; i < sg_count; ++i) {
                reduced += p_scratch[i];
              }
              p_scratch[0] = reduced;
            }
            item.barrier(sycl::access::fence_space::local_space);
            float tile_sum = p_scratch[0];
            item.barrier(sycl::access::fence_space::local_space);

            p_scratch[local_id] = p;
            item.barrier(sycl::access::fence_space::local_space);

            if (owns_value) {
              acc *= re_scale;
              for (int n = 0; n < tile_count; ++n) {
                int64_t v_k_local = valid_begin + tile_begin + n;
                int64_t v_base = (kv_begin + v_k_local) * params.v_stride_t + kv_head * params.v_stride_h;
                acc += p_scratch[n] * to_float_device(params.v[v_base + local_id]);
              }
            }

            denom = denom * re_scale + tile_sum;
            e_max = n_e_max;
            item.barrier(sycl::access::fence_space::local_space);
          }

          int64_t o_base = q_row * params.o_stride_t + head * params.o_stride_h;
          if (owns_value) {
            float value = denom > 0.0f ? acc / denom : 0.0f;
            params.out[o_base + local_id] = from_float_device<scalar_t>(value);
          }
          if (local_id == 0) {
            int64_t lse_offset = q_row * params.heads + head;
            params.lse[lse_offset] = denom > 0.0f ? sycl::log(denom) + e_max :
                -std::numeric_limits<float>::infinity();
          }
        });
  });
}

template <typename scalar_t>
sycl::event launch_relative_attention(
    sycl::queue& queue,
    RelativeAttentionParams<scalar_t> const& params,
    bool use_relative_bias,
    bool use_window,
    bool causal,
    int local_size) {
  if (use_relative_bias) {
    if (use_window) {
      return causal ? launch_relative_attention_static<scalar_t, true, true, true>(queue, params, local_size)
                    : launch_relative_attention_static<scalar_t, true, true, false>(queue, params, local_size);
    }
    return causal ? launch_relative_attention_static<scalar_t, true, false, true>(queue, params, local_size)
                  : launch_relative_attention_static<scalar_t, true, false, false>(queue, params, local_size);
  }

  if (use_window) {
    return causal ? launch_relative_attention_static<scalar_t, false, true, true>(queue, params, local_size)
                  : launch_relative_attention_static<scalar_t, false, true, false>(queue, params, local_size);
  }
  return causal ? launch_relative_attention_static<scalar_t, false, false, true>(queue, params, local_size)
                : launch_relative_attention_static<scalar_t, false, false, false>(queue, params, local_size);
}

void check_index_tensor(const at::Tensor& tensor, const char* name) {
  CHECK_INPUT(tensor);
  TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Int, name, " must be int32");
}

void check_last_dim_contiguous_3d(const at::Tensor& tensor, const char* name) {
  CHECK_DEVICE(tensor);
  TORCH_CHECK(tensor.dim() == 3, name, " must have shape [tokens, heads, dim]");
  TORCH_CHECK(tensor.stride(2) == 1, name, " must be contiguous on the last dimension");
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> inkling_relative_attention(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& q_to_seq,
    const at::Tensor& q_pos,
    const at::Tensor& cu_k,
    const std::optional<at::Tensor>& rel_bias,
    double softmax_scale,
    bool causal,
    int64_t window_size_left,
    int64_t window_size_right,
    double softcap,
    int64_t local_size_arg,
    const std::optional<at::Tensor>& out_opt) {
  check_last_dim_contiguous_3d(q, "q");
  check_last_dim_contiguous_3d(k, "k");
  check_last_dim_contiguous_3d(v, "v");
  check_index_tensor(q_to_seq, "q_to_seq");
  check_index_tensor(q_pos, "q_pos");
  check_index_tensor(cu_k, "cu_k");

  auto dtype = q.scalar_type();
  TORCH_CHECK(
      dtype == at::ScalarType::Half || dtype == at::ScalarType::BFloat16,
      "inkling_relative_attention only supports fp16/bf16 q, got ",
      dtype);
  TORCH_CHECK(k.scalar_type() == dtype, "k must have the same dtype as q");
  TORCH_CHECK(v.scalar_type() == dtype, "v must have the same dtype as q");
  TORCH_CHECK(q_to_seq.dim() == 1, "q_to_seq must be 1D");
  TORCH_CHECK(q_pos.dim() == 1, "q_pos must be 1D");
  TORCH_CHECK(cu_k.dim() == 1, "cu_k must be 1D");

  const int64_t total_q = q.size(0);
  const int64_t total_k = k.size(0);
  const int64_t heads = q.size(1);
  const int64_t kv_heads = k.size(1);
  const int64_t d = q.size(2);
  const int64_t dv = v.size(2);
  const int64_t batch = cu_k.size(0) - 1;

  TORCH_CHECK(v.size(0) == total_k, "v.size(0) must equal k.size(0)");
  TORCH_CHECK(v.size(1) == kv_heads, "v.size(1) must equal k.size(1)");
  TORCH_CHECK(q_to_seq.numel() == total_q, "q_to_seq must have one entry per query row");
  TORCH_CHECK(q_pos.numel() == total_q, "q_pos must have one entry per query row");
  TORCH_CHECK(batch >= 0, "cu_k must contain at least one element");
  TORCH_CHECK(heads > 0 && kv_heads > 0, "q/k head counts must be positive");
  TORCH_CHECK(heads % kv_heads == 0, "q heads must be divisible by kv heads");
  TORCH_CHECK(d > 0 && dv > 0, "q/k/v head dimensions must be positive");
  TORCH_CHECK(d == k.size(2), "q and k head dimensions must match");
  TORCH_CHECK(softmax_scale > 0.0, "softmax_scale must be positive");

  bool use_relative_bias = rel_bias.has_value() && rel_bias->numel() > 0;
  int64_t rel_len = 0;
  int64_t bias_stride_t = 0;
  int64_t bias_stride_h = 0;
  float const* rel_bias_ptr = nullptr;
  if (use_relative_bias) {
    const at::Tensor& bias = rel_bias.value();
    CHECK_DEVICE(bias);
    TORCH_CHECK(bias.scalar_type() == at::ScalarType::Float, "rel_bias must be float32");
    TORCH_CHECK(bias.dim() == 3, "rel_bias must have shape [total_q, heads, rel_len]");
    TORCH_CHECK(bias.size(0) == total_q, "rel_bias.size(0) must equal q.size(0)");
    TORCH_CHECK(bias.size(1) == heads, "rel_bias.size(1) must equal q.size(1)");
    TORCH_CHECK(bias.size(2) > 0, "rel_bias.size(2) must be positive");
    TORCH_CHECK(bias.stride(2) == 1, "rel_bias must be contiguous on the last dimension");
    rel_len = bias.size(2);
    bias_stride_t = bias.stride(0);
    bias_stride_h = bias.stride(1);
    rel_bias_ptr = bias.data_ptr<float>();
  }

  bool use_window = window_size_left >= 0 || window_size_right >= 0;
  int64_t local_size = local_size_arg <= 0 ? std::max<int64_t>(kDefaultLocalSize, next_power_of_2_i64(dv))
                                           : local_size_arg;
  TORCH_CHECK(is_power_of_2_i64(local_size), "local_size must be a power of two");
  TORCH_CHECK(local_size >= dv, "local_size must be >= v head dimension");
  TORCH_CHECK(local_size <= kMaxLocalSize, "local_size must be <= ", kMaxLocalSize);

  at::Tensor out;
  if (out_opt.has_value()) {
    out = out_opt.value();
    CHECK_DEVICE(out);
    TORCH_CHECK(out.scalar_type() == dtype, "out must have the same dtype as q");
    TORCH_CHECK(out.dim() == 3, "out must have shape [total_q, heads, dv]");
    TORCH_CHECK(out.size(0) == total_q && out.size(1) == heads && out.size(2) == dv, "out has invalid shape");
    TORCH_CHECK(out.stride(2) == 1, "out must be contiguous on the last dimension");
  } else {
    out = at::empty_strided({total_q, heads, dv}, {heads * dv, dv, 1}, q.options());
  }
  at::Tensor lse = at::empty_strided({total_q, heads}, {heads, 1}, q.options().dtype(at::ScalarType::Float));

  auto queue = c10::xpu::getCurrentXPUStream().queue();
  SYCL_DISPATCH_FLOATING_TYPES(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      dtype,
      "inkling_relative_attention",
      [&]() -> std::tuple<at::Tensor, at::Tensor> {
        RelativeAttentionParams<scalar_t> params{};
        params.q = q.data_ptr<scalar_t>();
        params.k = k.data_ptr<scalar_t>();
        params.v = v.data_ptr<scalar_t>();
        params.rel_bias = rel_bias_ptr;
        params.q_to_seq = q_to_seq.data_ptr<int32_t>();
        params.q_pos = q_pos.data_ptr<int32_t>();
        params.cu_k = cu_k.data_ptr<int32_t>();
        params.out = out.data_ptr<scalar_t>();
        params.lse = lse.data_ptr<float>();
        params.scale = static_cast<float>(softmax_scale);
        params.softcap = static_cast<float>(softcap);
        params.total_q = total_q;
        params.total_k = total_k;
        params.batch = batch;
        params.heads = heads;
        params.kv_heads = kv_heads;
        params.d = d;
        params.dv = dv;
        params.rel_len = rel_len;
        params.q_stride_t = q.stride(0);
        params.q_stride_h = q.stride(1);
        params.k_stride_t = k.stride(0);
        params.k_stride_h = k.stride(1);
        params.v_stride_t = v.stride(0);
        params.v_stride_h = v.stride(1);
        params.o_stride_t = out.stride(0);
        params.o_stride_h = out.stride(1);
        params.bias_stride_t = bias_stride_t;
        params.bias_stride_h = bias_stride_h;
        params.window_left = window_size_left;
        params.window_right = window_size_right;
        launch_relative_attention(
            queue, params, use_relative_bias, use_window, causal, static_cast<int>(local_size));
        return {out, lse};
      });
}
