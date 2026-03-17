/* Copyright 2026 SGLang Team. All Rights Reserved.

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

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <algorithm>
#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"

namespace {

constexpr int kSeqVecElems = 4;

template <typename scalar_t>
inline float to_float(scalar_t v) {
  return static_cast<float>(v);
}

template <typename scalar_t>
struct CausalConv1dFwdKernel {
  const bool varlen;
  const int32_t* query_start_ptr;
  const int32_t* cache_indices_ptr;
  const bool* has_initial_state_ptr;
  const int64_t seqlen;
  const int64_t width;
  const scalar_t* x_ptr;
  const scalar_t* weight_ptr;
  const scalar_t* bias_ptr;
  scalar_t* out_ptr;
  scalar_t* conv_states_ptr;
  const int64_t x_batch_stride;
  const int64_t x_c_stride;
  const int64_t x_l_stride;
  const int64_t weight_c_stride;
  const int64_t weight_width_stride;
  const int64_t out_batch_stride;
  const int64_t out_c_stride;
  const int64_t out_l_stride;
  const int64_t conv_states_batch_stride;
  const int64_t conv_states_c_stride;
  const int64_t conv_states_l_stride;
  const bool silu_activation;
  const int64_t pad_slot_id;

  void operator()(sycl::id<2> idx) const {
    const int64_t batch_id = static_cast<int64_t>(idx[0]);
    const int64_t channel_id = static_cast<int64_t>(idx[1]);

    const int cache_index = cache_indices_ptr == nullptr ? static_cast<int>(batch_id) : cache_indices_ptr[batch_id];
    if (cache_index == pad_slot_id) {
      return;
    }

    const int64_t sequence_start = varlen ? static_cast<int64_t>(query_start_ptr[batch_id]) : batch_id;
    const int64_t seq_len =
        varlen ? static_cast<int64_t>(query_start_ptr[batch_id + 1] - query_start_ptr[batch_id]) : seqlen;

    const scalar_t* x_base = x_ptr + sequence_start * x_batch_stride + channel_id * x_c_stride;
    scalar_t* out_base = out_ptr + sequence_start * out_batch_stride + channel_id * out_c_stride;

    const scalar_t* weight_base = weight_ptr + channel_id * weight_c_stride;
    const float bias_val = bias_ptr == nullptr ? 0.0f : to_float(bias_ptr[channel_id]);

    scalar_t* conv_base = conv_states_ptr == nullptr
                              ? nullptr
                              : conv_states_ptr + static_cast<int64_t>(cache_index) * conv_states_batch_stride +
                                    channel_id * conv_states_c_stride;

    const bool has_init = has_initial_state_ptr == nullptr ? false : has_initial_state_ptr[batch_id];

    scalar_t tail[4] = {scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0)};
    if (conv_base != nullptr && has_init) {
      for (int64_t i = 0; i < width - 1; ++i) {
        tail[i] = conv_base[i * conv_states_l_stride];
      }
    }

    using x_vec_t = sycl::vec<scalar_t, kSeqVecElems>;
    int64_t t = 0;
    for (; t + (kSeqVecElems - 1) < seq_len; t += kSeqVecElems) {
      x_vec_t x_vec_s;
      if (x_l_stride == 1) {
        x_vec_s.load(0, x_base + t);
      } else {
        for (int lane = 0; lane < kSeqVecElems; ++lane) {
          x_vec_s[lane] = x_base[(t + lane) * x_l_stride];
        }
      }

      const float w_last = weight_base[(width - 1) * weight_width_stride];
      sycl::vec<float, kSeqVecElems> out_vec_f = (x_vec_s * w_last).template convert<float>();
      out_vec_f += bias_val;

#pragma unroll
      for (int64_t w = 0; w < width - 1; ++w) {
        sycl::vec<float, kSeqVecElems> history_vec_f;
        for (int lane = 0; lane < kSeqVecElems; ++lane) {
          const int idx = lane + static_cast<int>(w);
          const scalar_t hist_val = idx < width - 1 ? tail[idx] : x_vec_s[idx - static_cast<int>(width - 1)];
          history_vec_f[lane] = to_float(hist_val);
        }
        out_vec_f += to_float(weight_base[w * weight_width_stride]) * history_vec_f;
      }

      if (silu_activation) {
        auto sigmoid = 1.0f / (1.0f + sycl::exp(-out_vec_f));
        out_vec_f *= sigmoid;
      }
      for (int64_t i = 0; i < width - 1; ++i) {
        tail[i] = x_vec_s[static_cast<int>(kSeqVecElems - (width - 1) + i)];
      }

      if (out_l_stride == 1) {
        x_vec_t out_data;
        out_data = out_vec_f.template convert<scalar_t>();
        out_data.store(0, out_base + t);
      } else {
        for (int lane = 0; lane < kSeqVecElems; ++lane) {
          out_base[(t + lane) * out_l_stride] = static_cast<scalar_t>(out_vec_f[lane]);
        }
      }
    }

    for (; t < seq_len; ++t) {
      const scalar_t x_t = x_base[t * x_l_stride];
      float acc = bias_val;

      for (int64_t w = 0; w < width - 1; ++w) {
        acc += to_float(weight_base[w * weight_width_stride]) * to_float(tail[w]);
      }

      acc += to_float(weight_base[(width - 1) * weight_width_stride]) * to_float(x_t);

      if (silu_activation) {
        acc = acc / (1.0f + sycl::exp(-acc));
      }

      out_base[t * out_l_stride] = static_cast<scalar_t>(acc);

      for (int64_t i = 0; i < width - 2; ++i) {
        tail[i] = tail[i + 1];
      }
      tail[width - 2] = x_t;
    }

    if (conv_base != nullptr) {
      for (int64_t i = 0; i < width - 1; ++i) {
        conv_base[i * conv_states_l_stride] = tail[i];
      }
    }
  }
};

template <typename scalar_t>
struct CausalConv1dUpdateKernel {
  const int64_t seqlen;
  const int64_t width;
  const int64_t state_len;
  const scalar_t* x_ptr;
  scalar_t* out_ptr;
  const scalar_t* weight_ptr;
  const scalar_t* bias_ptr;
  scalar_t* conv_state_ptr;
  const int32_t* cache_seqlens_ptr;
  const int32_t* conv_state_indices_ptr;
  const int64_t x_batch_stride;
  const int64_t x_c_stride;
  const int64_t x_l_stride;
  const int64_t out_batch_stride;
  const int64_t out_c_stride;
  const int64_t out_l_stride;
  const int64_t weight_c_stride;
  const int64_t weight_width_stride;
  const int64_t conv_state_batch_stride;
  const int64_t conv_state_c_stride;
  const int64_t conv_state_l_stride;
  const bool silu_activation;
  const int64_t pad_slot_id;

  void operator()(sycl::id<2> idx) const {
    const int64_t batch_id = static_cast<int64_t>(idx[0]);
    const int64_t channel_id = static_cast<int64_t>(idx[1]);

    const int conv_state_batch_coord =
        conv_state_indices_ptr == nullptr ? static_cast<int>(batch_id) : conv_state_indices_ptr[batch_id];

    if (conv_state_batch_coord == pad_slot_id) {
      return;
    }

    const scalar_t* x_base = x_ptr + batch_id * x_batch_stride + channel_id * x_c_stride;
    scalar_t* out_base = out_ptr + batch_id * out_batch_stride + channel_id * out_c_stride;
    const scalar_t* weight_base = weight_ptr + channel_id * weight_c_stride;
    scalar_t* conv_base = conv_state_ptr + static_cast<int64_t>(conv_state_batch_coord) * conv_state_batch_stride +
                          channel_id * conv_state_c_stride;

    const float bias_val = bias_ptr == nullptr ? 0.0f : to_float(bias_ptr[channel_id]);

    scalar_t history[4] = {scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0)};
    int update_idx = 0;

    if (cache_seqlens_ptr != nullptr) {
      const int cache_seqlen = cache_seqlens_ptr[batch_id] % static_cast<int>(state_len);
      int start_idx = cache_seqlen - static_cast<int>(width - 1);
      while (start_idx < 0) {
        start_idx += static_cast<int>(state_len);
      }
      for (int64_t i = 0; i < width - 1; ++i) {
        const int idx_in_state = (start_idx + static_cast<int>(i)) % static_cast<int>(state_len);
        history[i] = conv_base[idx_in_state * conv_state_l_stride];
      }
      update_idx = cache_seqlen;
    } else {
      for (int64_t i = 0; i < width - 1; ++i) {
        history[i] = conv_base[(state_len - (width - 1) + i) * conv_state_l_stride];
      }
    }

    using x_vec_t = sycl::vec<scalar_t, kSeqVecElems>;
    int64_t t = 0;
    for (; t + (kSeqVecElems - 1) < seqlen; t += kSeqVecElems) {
      x_vec_t x_vec_s;
      if (x_l_stride == 1) {
        x_vec_s.load(0, x_base + t);
      } else {
        for (int lane = 0; lane < kSeqVecElems; ++lane) {
          x_vec_s[lane] = x_base[(t + lane) * x_l_stride];
        }
      }

      const float w_last = weight_base[(width - 1) * weight_width_stride];
      sycl::vec<float, kSeqVecElems> out_vec_f = (x_vec_s * w_last).template convert<float>();
      out_vec_f += bias_val;

#pragma unroll
      for (int64_t w = 0; w < width - 1; ++w) {
        sycl::vec<float, kSeqVecElems> history_vec_f;
        for (int lane = 0; lane < kSeqVecElems; ++lane) {
          const int hist_idx = lane + static_cast<int>(w);
          const scalar_t hist_val =
              hist_idx < width - 1 ? history[hist_idx] : x_vec_s[hist_idx - static_cast<int>(width - 1)];
          history_vec_f[lane] = to_float(hist_val);
        }
        out_vec_f += to_float(weight_base[w * weight_width_stride]) * history_vec_f;
      }

      if (silu_activation) {
        auto sigmoid = 1.0f / (1.0f + sycl::exp(-out_vec_f));
        out_vec_f *= sigmoid;
      }

      if (out_l_stride == 1) {
        x_vec_t out_data = out_vec_f.template convert<scalar_t>();
        out_data.store(0, out_base + t);
      } else {
        for (int lane = 0; lane < kSeqVecElems; ++lane) {
          out_base[(t + lane) * out_l_stride] = static_cast<scalar_t>(out_vec_f[lane]);
        }
      }

      for (int lane = 0; lane < kSeqVecElems; ++lane) {
        const scalar_t x_t = x_vec_s[lane];
        if (cache_seqlens_ptr != nullptr) {
          conv_base[static_cast<int64_t>(update_idx) * conv_state_l_stride] = x_t;
          ++update_idx;
          if (update_idx >= state_len) {
            update_idx -= static_cast<int>(state_len);
          }
        } else {
          // Keep conv_state as a sliding window over original input tokens.
          for (int64_t i = 0; i < state_len - 1; ++i) {
            conv_base[i * conv_state_l_stride] = conv_base[(i + 1) * conv_state_l_stride];
          }
          conv_base[(state_len - 1) * conv_state_l_stride] = x_t;
        }
      }

      for (int64_t i = 0; i < width - 1; ++i) {
        history[i] = x_vec_s[static_cast<int>(kSeqVecElems - (width - 1) + i)];
      }
    }

    for (; t < seqlen; ++t) {
      const scalar_t x_t = x_base[t * x_l_stride];
      float acc = bias_val;
      for (int64_t w = 0; w < width - 1; ++w) {
        acc += to_float(weight_base[w * weight_width_stride]) * to_float(history[w]);
      }
      acc += to_float(weight_base[(width - 1) * weight_width_stride]) * to_float(x_t);
      if (silu_activation) {
        acc = acc / (1.0f + sycl::exp(-acc));
      }
      out_base[t * out_l_stride] = static_cast<scalar_t>(acc);

      if (cache_seqlens_ptr != nullptr) {
        conv_base[static_cast<int64_t>(update_idx) * conv_state_l_stride] = x_t;
        ++update_idx;
        if (update_idx >= state_len) {
          update_idx -= static_cast<int>(state_len);
        }
      } else {
        // Keep conv_state as a sliding window over original input tokens.
        for (int64_t i = 0; i < state_len - 1; ++i) {
          conv_base[i * conv_state_l_stride] = conv_base[(i + 1) * conv_state_l_stride];
        }
        conv_base[(state_len - 1) * conv_state_l_stride] = x_t;
      }

      for (int64_t i = 0; i < width - 2; ++i) {
        history[i] = history[i + 1];
      }
      history[width - 2] = x_t;
    }
  }
};

template <typename scalar_t>
void causal_conv1d_fwd_impl(
    at::Tensor& x,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    const std::optional<at::Tensor>& conv_states,
    const std::optional<at::Tensor>& query_start_loc,
    const std::optional<at::Tensor>& cache_indices,
    const std::optional<at::Tensor>& has_initial_state,
    bool silu_activation,
    int64_t pad_slot_id) {
  const bool varlen = query_start_loc.has_value();

  const int32_t* query_start_ptr = varlen ? query_start_loc->data_ptr<int32_t>() : nullptr;
  const int32_t* cache_indices_ptr = cache_indices.has_value() ? cache_indices->data_ptr<int32_t>() : nullptr;
  const bool* has_initial_state_ptr = has_initial_state.has_value() ? has_initial_state->data_ptr<bool>() : nullptr;

  const int64_t batch = varlen ? query_start_loc->size(0) - 1 : x.size(0);
  const int64_t dim = varlen ? x.size(0) : x.size(1);
  const int64_t seqlen = varlen ? x.size(1) : x.size(2);
  const int64_t width = weight.size(1);

  const scalar_t* x_ptr = reinterpret_cast<const scalar_t*>(x.data_ptr());
  const scalar_t* weight_ptr = reinterpret_cast<const scalar_t*>(weight.data_ptr());
  const scalar_t* bias_ptr = bias.has_value() ? reinterpret_cast<const scalar_t*>(bias->data_ptr()) : nullptr;
  scalar_t* out_ptr = reinterpret_cast<scalar_t*>(x.data_ptr());

  scalar_t* conv_states_ptr = conv_states.has_value() ? reinterpret_cast<scalar_t*>(conv_states->data_ptr()) : nullptr;

  const int64_t x_batch_stride = x.stride(varlen ? 1 : 0);
  const int64_t x_c_stride = x.stride(varlen ? 0 : 1);
  const int64_t x_l_stride = x.stride(varlen ? 1 : 2);
  const int64_t weight_c_stride = weight.stride(0);
  const int64_t weight_width_stride = weight.stride(1);
  const int64_t out_batch_stride = x.stride(varlen ? 1 : 0);
  const int64_t out_c_stride = x.stride(varlen ? 0 : 1);
  const int64_t out_l_stride = x.stride(varlen ? 1 : 2);

  const int64_t conv_states_batch_stride = conv_states.has_value() ? conv_states->stride(0) : 0;
  const int64_t conv_states_c_stride = conv_states.has_value() ? conv_states->stride(-2) : 0;
  const int64_t conv_states_l_stride = conv_states.has_value() ? conv_states->stride(-1) : 0;

  auto q = at::xpu::getCurrentXPUStream().queue();
  CausalConv1dFwdKernel<scalar_t> kernel{
      varlen,
      query_start_ptr,
      cache_indices_ptr,
      has_initial_state_ptr,
      seqlen,
      width,
      x_ptr,
      weight_ptr,
      bias_ptr,
      out_ptr,
      conv_states_ptr,
      x_batch_stride,
      x_c_stride,
      x_l_stride,
      weight_c_stride,
      weight_width_stride,
      out_batch_stride,
      out_c_stride,
      out_l_stride,
      conv_states_batch_stride,
      conv_states_c_stride,
      conv_states_l_stride,
      silu_activation,
      pad_slot_id};
  q.parallel_for<CausalConv1dFwdKernel<scalar_t>>(sycl::range<2>(batch, dim), kernel);
}

template <typename scalar_t>
void causal_conv1d_update_impl(
    at::Tensor& x,
    at::Tensor& conv_state,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    bool silu_activation,
    const std::optional<at::Tensor>& cache_seqlens,
    const std::optional<at::Tensor>& conv_state_indices,
    int64_t pad_slot_id) {
  const int64_t batch = x.size(0);
  const int64_t dim = x.size(1);
  const int64_t seqlen = x.size(2);
  const int64_t width = weight.size(1);
  const int64_t state_len = conv_state.size(2);

  const scalar_t* x_ptr = reinterpret_cast<const scalar_t*>(x.data_ptr());
  scalar_t* out_ptr = reinterpret_cast<scalar_t*>(x.data_ptr());
  const scalar_t* weight_ptr = reinterpret_cast<const scalar_t*>(weight.data_ptr());
  const scalar_t* bias_ptr = bias.has_value() ? reinterpret_cast<const scalar_t*>(bias->data_ptr()) : nullptr;
  scalar_t* conv_state_ptr = reinterpret_cast<scalar_t*>(conv_state.data_ptr());

  const int32_t* cache_seqlens_ptr =
      cache_seqlens.has_value() ? reinterpret_cast<const int32_t*>(cache_seqlens->data_ptr()) : nullptr;
  const int32_t* conv_state_indices_ptr =
      conv_state_indices.has_value() ? reinterpret_cast<const int32_t*>(conv_state_indices->data_ptr()) : nullptr;

  const int64_t x_batch_stride = x.stride(0);
  const int64_t x_c_stride = x.stride(1);
  const int64_t x_l_stride = x.stride(2);
  const int64_t out_batch_stride = x.stride(0);
  const int64_t out_c_stride = x.stride(1);
  const int64_t out_l_stride = x.stride(2);

  const int64_t weight_c_stride = weight.stride(0);
  const int64_t weight_width_stride = weight.stride(1);

  const int64_t conv_state_batch_stride = conv_state.stride(0);
  const int64_t conv_state_c_stride = conv_state.stride(1);
  const int64_t conv_state_l_stride = conv_state.stride(2);

  auto q = at::xpu::getCurrentXPUStream().queue();
  CausalConv1dUpdateKernel<scalar_t> kernel{
      seqlen,
      width,
      state_len,
      x_ptr,
      out_ptr,
      weight_ptr,
      bias_ptr,
      conv_state_ptr,
      cache_seqlens_ptr,
      conv_state_indices_ptr,
      x_batch_stride,
      x_c_stride,
      x_l_stride,
      out_batch_stride,
      out_c_stride,
      out_l_stride,
      weight_c_stride,
      weight_width_stride,
      conv_state_batch_stride,
      conv_state_c_stride,
      conv_state_l_stride,
      silu_activation,
      pad_slot_id};
  q.parallel_for<CausalConv1dUpdateKernel<scalar_t>>(sycl::range<2>(batch, dim), kernel);
}

}  // namespace

void causal_conv1d_fwd(
    at::Tensor& x,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    const std::optional<at::Tensor>& conv_states,
    const std::optional<at::Tensor>& query_start_loc,
    const std::optional<at::Tensor>& cache_indices,
    const std::optional<at::Tensor>& has_initial_state,
    bool silu_activation,
    int64_t pad_slot_id) {
  TORCH_CHECK(x.is_xpu(), "x must be on XPU");
  TORCH_CHECK(weight.is_xpu(), "weight must be on XPU");
  TORCH_CHECK(x.scalar_type() == weight.scalar_type(), "weight type must equal input type");
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16);

  const bool varlen = query_start_loc.has_value();
  const int64_t width = weight.size(-1);
  TORCH_CHECK(width >= 2 && width <= 4, "causal_conv1d only supports width between 2 and 4");

  if (bias.has_value()) {
    TORCH_CHECK(bias->is_xpu(), "bias must be on XPU");
    TORCH_CHECK(bias->scalar_type() == weight.scalar_type(), "bias type must equal weight type");
  }

  if (query_start_loc.has_value()) {
    TORCH_CHECK(query_start_loc->is_xpu(), "query_start_loc must be on XPU");
    TORCH_CHECK(query_start_loc->scalar_type() == at::kInt, "query_start_loc must be int32");
  }

  if (cache_indices.has_value()) {
    TORCH_CHECK(cache_indices->is_xpu(), "cache_indices must be on XPU");
    TORCH_CHECK(cache_indices->scalar_type() == at::kInt, "cache_indices must be int32");
  }

  if (has_initial_state.has_value()) {
    TORCH_CHECK(has_initial_state->is_xpu(), "has_initial_state must be on XPU");
    TORCH_CHECK(has_initial_state->scalar_type() == at::kBool, "has_initial_state must be bool");
  }

  if (conv_states.has_value()) {
    TORCH_CHECK(conv_states->is_xpu(), "conv_states must be on XPU");
    TORCH_CHECK(conv_states->scalar_type() == x.scalar_type(), "conv_states type must equal input type");
  }

  if (varlen) {
    TORCH_CHECK(x.dim() == 2, "varlen mode expects x to be 2D");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [dim, width]");
    TORCH_CHECK(weight.size(0) == x.size(0), "weight dim must match x dim axis");
  } else {
    TORCH_CHECK(x.dim() == 3, "dense mode expects x to be 3D [batch, dim, seqlen]");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [dim, width]");
    TORCH_CHECK(weight.size(0) == x.size(1), "weight dim must match x dim axis");
  }

  SYCL_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16, at::ScalarType::Half, x.scalar_type(), "causal_conv1d_fwd_xpu", [&] {
        causal_conv1d_fwd_impl<scalar_t>(
            x,
            weight,
            bias,
            conv_states,
            query_start_loc,
            cache_indices,
            has_initial_state,
            silu_activation,
            pad_slot_id);
      });
}

void causal_conv1d_update(
    at::Tensor& x,
    at::Tensor& conv_state,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    bool silu_activation,
    const std::optional<at::Tensor>& cache_seqlens,
    const std::optional<at::Tensor>& conv_state_indices,
    int64_t pad_slot_id) {
  TORCH_CHECK(x.is_xpu(), "x must be on XPU");
  TORCH_CHECK(conv_state.is_xpu(), "conv_state must be on XPU");
  TORCH_CHECK(weight.is_xpu(), "weight must be on XPU");
  TORCH_CHECK(x.scalar_type() == weight.scalar_type(), "weight type must equal input type");
  TORCH_CHECK(conv_state.scalar_type() == x.scalar_type(), "conv_state type must equal input type");

  TORCH_CHECK(x.dim() == 3, "x must be [batch, dim, seqlen]");
  TORCH_CHECK(conv_state.dim() == 3, "conv_state must be [entries, dim, state_len]");
  TORCH_CHECK(weight.dim() == 2, "weight must be [dim, width]");

  const int64_t batch = x.size(0);
  const int64_t dim = x.size(1);
  const int64_t width = weight.size(1);
  const int64_t state_len = conv_state.size(2);

  TORCH_CHECK(width >= 2 && width <= 4, "causal_conv1d only supports width between 2 and 4");
  TORCH_CHECK(weight.size(0) == dim, "weight dim must match x dim");
  TORCH_CHECK(conv_state.size(1) == dim, "conv_state dim must match x dim");
  TORCH_CHECK(state_len >= width - 1, "conv_state_len must be >= width - 1");

  if (bias.has_value()) {
    TORCH_CHECK(bias->is_xpu(), "bias must be on XPU");
    TORCH_CHECK(bias->scalar_type() == weight.scalar_type(), "bias type must equal weight type");
    TORCH_CHECK(bias->numel() == dim, "bias must have shape [dim]");
  }

  if (cache_seqlens.has_value()) {
    TORCH_CHECK(cache_seqlens->is_xpu(), "cache_seqlens must be on XPU");
    TORCH_CHECK(cache_seqlens->scalar_type() == at::kInt, "cache_seqlens must be int32");
    TORCH_CHECK(cache_seqlens->numel() == batch, "cache_seqlens must have shape [batch]");
  }

  if (conv_state_indices.has_value()) {
    TORCH_CHECK(conv_state_indices->is_xpu(), "conv_state_indices must be on XPU");
    TORCH_CHECK(conv_state_indices->scalar_type() == at::kInt, "conv_state_indices must be int32");
    TORCH_CHECK(conv_state_indices->numel() == batch, "conv_state_indices must have shape [batch]");
  } else {
    TORCH_CHECK(conv_state.size(0) == batch, "conv_state entries must match batch when indices are absent");
  }

  SYCL_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16, at::ScalarType::Half, x.scalar_type(), "causal_conv1d_update_xpu", [&] {
        causal_conv1d_update_impl<scalar_t>(
            x, conv_state, weight, bias, silu_activation, cache_seqlens, conv_state_indices, pad_slot_id);
      });
}
