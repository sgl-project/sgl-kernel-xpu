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

// Causal conv1d forward/update kernels for XPU (SYCL).
// Implementation aims to mirror CUDA-side behavior.

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstdint>
#include <optional>
#include <sycl/sycl.hpp>

#include "SGLKernelPerf.h"
#include "Utils.h"
#include "causal_conv1d_token_parallel.hpp"
#include "sgl_kernel_export.h"

struct ConvParamsBase {
  using index_t = uint32_t;

  int batch = 0;
  int dim = 0;
  int seqlen = 0;
  int width = 0;
  int64_t pad_slot_id = 0;
  bool silu_activation = false;

  index_t x_batch_stride = 0;
  index_t x_c_stride = 0;
  index_t x_l_stride = 0;
  index_t weight_c_stride = 0;
  index_t weight_width_stride = 0;
  index_t out_batch_stride = 0;
  index_t out_c_stride = 0;
  index_t out_l_stride = 0;

  int conv_state_len = 0;
  index_t conv_state_batch_stride = 0;
  index_t conv_state_c_stride = 0;
  index_t conv_state_l_stride = 0;

  void* x_ptr = nullptr;
  void* weight_ptr = nullptr;
  void* bias_ptr = nullptr;
  void* out_ptr = nullptr;

  void* conv_state_ptr = nullptr;
  const int* query_start_loc_ptr = nullptr;
  const bool* has_initial_state_ptr = nullptr;
  const int* cache_indices_ptr = nullptr;
  const int* cache_seqlens = nullptr;

  const int* conv_state_indices_ptr = nullptr;

  void* seq_idx_ptr = nullptr;

  void* initial_states_ptr = nullptr;
  index_t initial_states_batch_stride = 0;
  index_t initial_states_l_stride = 0;
  index_t initial_states_c_stride = 0;

  void* final_states_ptr = nullptr;
  index_t final_states_batch_stride = 0;
  index_t final_states_l_stride = 0;
  index_t final_states_c_stride = 0;

  void* conv_states_ptr = nullptr;
  index_t conv_states_batch_stride = 0;
  index_t conv_states_l_stride = 0;
  index_t conv_states_c_stride = 0;
};

static void set_conv_params_fwd(
    ConvParamsBase& params,
    int batch,
    int dim,
    int seqlen,
    int width,
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& out,
    const std::optional<at::Tensor>& bias,
    bool silu_activation,
    int64_t pad_slot_id,
    const std::optional<at::Tensor>& query_start_loc = std::nullopt,
    const std::optional<at::Tensor>& cache_indices = std::nullopt,
    const std::optional<at::Tensor>& has_initial_state = std::nullopt) {
  params = ConvParamsBase{};

  params.batch = batch;
  params.dim = dim;
  params.seqlen = seqlen;
  params.width = width;
  params.pad_slot_id = pad_slot_id;

  params.silu_activation = silu_activation;

  // Set the pointers and strides.
  params.x_ptr = x.data_ptr();
  params.weight_ptr = weight.data_ptr();
  params.bias_ptr = bias.has_value() ? bias.value().data_ptr() : nullptr;
  params.out_ptr = out.data_ptr();
  // All stride are in elements, not bytes.
  params.query_start_loc_ptr = query_start_loc.has_value() ? query_start_loc.value().data_ptr<int>() : nullptr;
  params.cache_indices_ptr = cache_indices.has_value() ? cache_indices.value().data_ptr<int>() : nullptr;
  params.has_initial_state_ptr = has_initial_state.has_value() ? has_initial_state.value().data_ptr<bool>() : nullptr;
  const bool varlen = params.query_start_loc_ptr != nullptr;
  params.x_batch_stride = x.stride(varlen ? 1 : 0);
  params.x_c_stride = x.stride(varlen ? 0 : 1);
  params.x_l_stride = x.stride(varlen ? 1 : -1);
  params.weight_c_stride = weight.stride(0);
  params.weight_width_stride = weight.stride(1);
  params.out_batch_stride = out.stride(varlen ? 1 : 0);
  params.out_c_stride = out.stride(varlen ? 0 : 1);
  params.out_l_stride = out.stride(varlen ? 1 : -1);
}

template <typename scalar_t, int kWidth>
struct CausalConv1dFwdKernel {
  ConvParamsBase params;

  [[sycl::reqd_sub_group_size(32)]]
  void operator()(sycl::nd_item<2> item) const {
    auto* x_ptr = reinterpret_cast<scalar_t*>(params.x_ptr);
    auto* w_ptr = reinterpret_cast<scalar_t*>(params.weight_ptr);
    auto* bias_ptr = reinterpret_cast<scalar_t*>(params.bias_ptr);
    auto* conv_states_ptr = reinterpret_cast<scalar_t*>(params.conv_states_ptr);
    auto* out_ptr = reinterpret_cast<scalar_t*>(params.out_ptr);
    const bool varlen = params.query_start_loc_ptr != nullptr;

    const int batch_id = item.get_group(0);
    const int channel_id = item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);
    // Guard out-of-range lanes.
    if (batch_id >= params.batch || channel_id >= params.dim) {
      return;
    }

    const int seq_start = varlen ? params.query_start_loc_ptr[batch_id] : 0;
    const int cur_seqlen = varlen ? (params.query_start_loc_ptr[batch_id + 1] - seq_start) : params.seqlen;
    // Skip empty sequences.
    if (cur_seqlen <= 0) {
      return;
    }

    const int cache_idx = (params.cache_indices_ptr != nullptr) ? params.cache_indices_ptr[batch_id] : batch_id;
    // Skip padded slots.
    if (cache_idx == static_cast<int>(params.pad_slot_id)) {
      return;
    }
    const bool has_init = (params.has_initial_state_ptr != nullptr) ? params.has_initial_state_ptr[batch_id] : false;

    // Load per-channel weights and optional bias.
    const int w_base_offset = channel_id * params.weight_c_stride;
    float wt[kWidth];
#pragma unroll
    for (int w = 0; w < kWidth; ++w) {
      wt[w] = static_cast<float>(w_ptr[w_base_offset + w * params.weight_width_stride]);
    }
    const float bias_val = (bias_ptr != nullptr) ? static_cast<float>(bias_ptr[channel_id]) : 0.f;
    const bool has_conv_states = (conv_states_ptr != nullptr);

    // Initialize the sliding window history.
    float x_vals[kWidth];
#pragma unroll
    for (int w = 0; w < kWidth - 1; ++w) {
      if (has_conv_states && has_init) {
        x_vals[w] =
            static_cast<float>(conv_states_ptr
                                   [cache_idx * params.conv_states_batch_stride +
                                    channel_id * params.conv_states_c_stride + w * params.conv_states_l_stride]);
      } else {
        x_vals[w] = 0.f;
      }
    }
    x_vals[kWidth - 1] = 0.f;

    // Precompute offset base and step to avoid per-token conditional logic.
    int x_base_offset, out_base_offset, x_step, out_step;
    if (varlen) {
      x_base_offset = seq_start * params.x_batch_stride + channel_id * params.x_c_stride;
      out_base_offset = seq_start * params.out_batch_stride + channel_id * params.out_c_stride;
      x_step = params.x_batch_stride;
      out_step = params.out_batch_stride;
    } else {
      x_base_offset = batch_id * params.x_batch_stride + channel_id * params.x_c_stride;
      out_base_offset = batch_id * params.out_batch_stride + channel_id * params.out_c_stride;
      x_step = params.x_l_stride;
      out_step = params.out_l_stride;
    }

    // Main causal conv loop.
    float x_t = static_cast<float>(x_ptr[x_base_offset]);
    for (int t = 0; t < cur_seqlen; ++t) {
      const float x_next = (t + 1 < cur_seqlen) ? static_cast<float>(x_ptr[x_base_offset + (t + 1) * x_step]) : 0.f;
      x_vals[kWidth - 1] = x_t;

      float out_val = bias_val;
#pragma unroll
      for (int w = 0; w < kWidth; ++w) {
        out_val += wt[w] * x_vals[w];
      }

      if (params.silu_activation) {
        // Use native exp here to keep SiLU on the XPU fast-math path.
        out_val = out_val / (1.f + sycl::native::exp(-out_val));
      }

      out_ptr[out_base_offset + t * out_step] = static_cast<scalar_t>(out_val);

#pragma unroll
      for (int w = 0; w < kWidth - 1; ++w) {
        x_vals[w] = x_vals[w + 1];
      }
      x_t = x_next;
    }

    // Save trailing window as next-chunk initial state.
    if (has_conv_states) {
#pragma unroll
      for (int w = 0; w < kWidth - 1; ++w) {
        conv_states_ptr
            [cache_idx * params.conv_states_batch_stride + channel_id * params.conv_states_c_stride +
             w * params.conv_states_l_stride] = static_cast<scalar_t>(x_vals[w]);
      }
    }
  }
};

template <typename scalar_t, int kWidth>
struct CausalConv1dUpdateKernel {
  ConvParamsBase params;

  [[sycl::reqd_sub_group_size(32)]]
  void operator()(sycl::nd_item<2> item) const {
    auto* x_ptr = reinterpret_cast<scalar_t*>(params.x_ptr);
    auto* w_ptr = reinterpret_cast<scalar_t*>(params.weight_ptr);
    auto* bias_ptr = reinterpret_cast<scalar_t*>(params.bias_ptr);
    auto* conv_state_ptr = reinterpret_cast<scalar_t*>(params.conv_state_ptr);
    auto* out_ptr = reinterpret_cast<scalar_t*>(params.out_ptr);

    const int batch_id = item.get_group(0);
    const int channel_id = item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);
    // Guard out-of-range lanes.
    if (batch_id >= params.batch || channel_id >= params.dim) {
      return;
    }

    const int cache_idx =
        (params.conv_state_indices_ptr != nullptr) ? params.conv_state_indices_ptr[batch_id] : batch_id;
    // Skip padded slots.
    if (cache_idx == static_cast<int>(params.pad_slot_id)) {
      return;
    }
    const bool circular = (params.cache_seqlens != nullptr);

    // Load per-channel weights and optional bias.
    const int w_base_offset = channel_id * params.weight_c_stride;
    float wt[kWidth];
#pragma unroll
    for (int w = 0; w < kWidth; ++w) {
      wt[w] = static_cast<float>(w_ptr[w_base_offset + w * params.weight_width_stride]);
    }
    const float bias_val = (bias_ptr != nullptr) ? static_cast<float>(bias_ptr[channel_id]) : 0.f;

    scalar_t* conv_state_channel =
        conv_state_ptr + cache_idx * params.conv_state_batch_stride + channel_id * params.conv_state_c_stride;
    const int state_len = params.conv_state_len;
    const int advance_len = params.seqlen;

    int cache_seqlen = circular ? (params.cache_seqlens[batch_id] % state_len) : 0;
    int update_idx = cache_seqlen - (kWidth - 1);
    update_idx = (update_idx < 0) ? (update_idx + state_len) : update_idx;

    // Sliding window: first (kWidth - 1) are history, last element is current token.
    float x_vals[kWidth] = {0.f};
    if (!circular) {
      const int shift_count = state_len - advance_len - (kWidth - 1);
      for (int i = 0; i < shift_count; ++i) {
        conv_state_channel[i * params.conv_state_l_stride] =
            conv_state_channel[(i + advance_len) * params.conv_state_l_stride];
      }

      for (int i = 0; i < kWidth - 1; ++i) {
        const scalar_t state_val = conv_state_channel[(state_len - (kWidth - 1) + i) * params.conv_state_l_stride];
        const int write_idx = state_len - advance_len - (kWidth - 1) + i;
        if (i < advance_len + (kWidth - 1) && write_idx >= 0) {
          conv_state_channel[write_idx * params.conv_state_l_stride] = state_val;
        }
        x_vals[i] = static_cast<float>(state_val);
      }
    } else {
      for (int i = 0; i < kWidth - 1; ++i) {
        const scalar_t state_val = conv_state_channel[update_idx * params.conv_state_l_stride];
        x_vals[i] = static_cast<float>(state_val);
        ++update_idx;
        update_idx = (update_idx >= state_len) ? (update_idx - state_len) : update_idx;
      }
    }

    // Main causal conv loop: write state, run causal conv, write output.
    const int x_base_offset = batch_id * params.x_batch_stride + channel_id * params.x_c_stride;
    const int out_base_offset = batch_id * params.out_batch_stride + channel_id * params.out_c_stride;
    float x_t = static_cast<float>(x_ptr[x_base_offset]);
    for (int t = 0; t < params.seqlen; ++t) {
      const float x_next =
          (t + 1 < params.seqlen) ? static_cast<float>(x_ptr[x_base_offset + (t + 1) * params.x_l_stride]) : 0.f;

      if (!circular) {
        const int write_idx = state_len - advance_len + t;
        if (t < advance_len && write_idx >= 0) {
          conv_state_channel[write_idx * params.conv_state_l_stride] = static_cast<scalar_t>(x_t);
        }
      } else {
        conv_state_channel[update_idx * params.conv_state_l_stride] = static_cast<scalar_t>(x_t);
        ++update_idx;
        update_idx = (update_idx >= state_len) ? (update_idx - state_len) : update_idx;
      }

      x_vals[kWidth - 1] = x_t;

      float out_val = bias_val;
#pragma unroll
      for (int w = 0; w < kWidth; ++w) {
        out_val += wt[w] * x_vals[w];
      }
      if (params.silu_activation) {
        // Use native exp here to keep SiLU on the XPU fast-math path.
        out_val = out_val / (1.f + sycl::native::exp(-out_val));
      }

      out_ptr[out_base_offset + t * params.out_l_stride] = static_cast<scalar_t>(out_val);

#pragma unroll
      // Slide the history window by one position.
      for (int w = 0; w < kWidth - 1; ++w) {
        x_vals[w] = x_vals[w + 1];
      }
      x_t = x_next;
    }
  }
};

template <typename scalar_t>
static void launch_causal_conv1d_fwd(const ConvParamsBase& params) {
  auto queue = c10::xpu::getCurrentXPUStream().queue();
  constexpr int kNThreads = 128;
  const int batch = params.batch;
  const int dim = params.dim;
  const int channel_groups = (dim + kNThreads - 1) / kNThreads;
  sycl::nd_range<2> range(sycl::range<2>(batch, channel_groups * kNThreads), sycl::range<2>(1, kNThreads));

#define LAUNCH_WIDTH(W)                                                   \
  if (params.width == W) {                                                \
    CausalConv1dFwdKernel<scalar_t, W> kern{params};                      \
    queue.submit([&](sycl::handler& h) { h.parallel_for(range, kern); }); \
    return;                                                               \
  }

  LAUNCH_WIDTH(2)
  LAUNCH_WIDTH(3)
  LAUNCH_WIDTH(4)
#undef LAUNCH_WIDTH
  TORCH_CHECK(false, "causal_conv1d only supports width between 2 and 4");
}

template <typename scalar_t>
static void launch_causal_conv1d_update(const ConvParamsBase& params) {
  auto queue = c10::xpu::getCurrentXPUStream().queue();
  constexpr int kNThreads = 64;
  const int channel_groups = (params.dim + kNThreads - 1) / kNThreads;
  sycl::nd_range<2> range(sycl::range<2>(params.batch, channel_groups * kNThreads), sycl::range<2>(1, kNThreads));

#define LAUNCH_WIDTH(W)                                                   \
  if (params.width == W) {                                                \
    CausalConv1dUpdateKernel<scalar_t, W> kern{params};                   \
    queue.submit([&](sycl::handler& h) { h.parallel_for(range, kern); }); \
    return;                                                               \
  }

  LAUNCH_WIDTH(2)
  LAUNCH_WIDTH(3)
  LAUNCH_WIDTH(4)
#undef LAUNCH_WIDTH
  TORCH_CHECK(false, "causal_conv1d only supports width between 2 and 4");
}

// ---------------------------------------------------------------------------
// Token-parallel implementation (see causal_conv1d_token_parallel.hpp for the
// design and its tradeoffs vs. the channel-parallel/serial-loop kernels
// above). This is now the default for the two paths that matter in
// production and that the design was benchmarked against (varlen prefill,
// single-token decode/update); the old kernels above remain as the
// implementation for the narrower cases the token-parallel kernels don't
// (yet) cover: dense (non-varlen) prefill and multi-token/circular-buffer
// decode -- see the dispatch in causal_conv1d_fwd/causal_conv1d_update below.
// ---------------------------------------------------------------------------

template <typename scalar_t, int Width>
static void launch_causal_conv1d_fwd_token_parallel_w(
    sycl::queue& queue,
    at::Tensor& out,
    const at::Tensor& x,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias_,
    const std::optional<at::Tensor>& conv_states,
    const at::Tensor& query_start_loc,
    const std::optional<at::Tensor>& cache_indices,
    const std::optional<at::Tensor>& has_initial_state,
    bool silu_activation,
    int64_t pad_slot_id,
    int batch_size,
    int num_virtual_tokens,
    int dim) {
  const bool has_conv_states = conv_states.has_value();

  // Allocated (if needed) and kept alive for the lifetime of this function,
  // which submits both kernels that touch it before returning -- same
  // lifetime pattern already used by the GDN kernel's own two-launch
  // prefill scheme.
  at::Tensor conv_states_tmp;
  if (has_conv_states) {
    conv_states_tmp = at::empty({std::max(batch_size, 1), Width - 1, dim}, x.options());
  }

  using KERNEL_MAIN = sgl_conv1d::CausalConv1dFwdTokenParallelKernel<scalar_t, Width>;
  auto range_main = KERNEL_MAIN::get_nd_range(num_virtual_tokens, dim);
  queue.submit([&](sycl::handler& cgh) {
    KERNEL_MAIN task(
        reinterpret_cast<scalar_t*>(out.data_ptr()),
        reinterpret_cast<const scalar_t*>(x.data_ptr()),
        static_cast<int>(x.stride(1)),
        static_cast<int>(x.stride(0)),
        static_cast<int>(out.stride(1)),
        static_cast<int>(out.stride(0)),
        reinterpret_cast<const scalar_t*>(weight.data_ptr()),
        static_cast<int>(weight.stride(0)),
        static_cast<int>(weight.stride(1)),
        bias_.has_value() ? reinterpret_cast<const scalar_t*>(bias_.value().data_ptr()) : nullptr,
        has_conv_states ? reinterpret_cast<scalar_t*>(conv_states.value().data_ptr()) : nullptr,
        has_conv_states ? static_cast<int>(conv_states.value().stride(0)) : 0,
        has_conv_states ? static_cast<int>(conv_states.value().stride(-1)) : 0,
        has_conv_states ? static_cast<int>(conv_states.value().stride(-2)) : 0,
        has_conv_states ? reinterpret_cast<scalar_t*>(conv_states_tmp.data_ptr()) : nullptr,
        reinterpret_cast<const int32_t*>(query_start_loc.data_ptr()),
        cache_indices.has_value() ? reinterpret_cast<const int*>(cache_indices.value().data_ptr()) : nullptr,
        has_initial_state.has_value() ? reinterpret_cast<const bool*>(has_initial_state.value().data_ptr()) : nullptr,
        silu_activation,
        static_cast<int>(pad_slot_id),
        batch_size,
        dim);
    cgh.parallel_for(range_main, task);
  });

  if (has_conv_states) {
    using KERNEL_FINALIZE = sgl_conv1d::FinalizeConvStatesKernel<scalar_t>;
    auto range_finalize = KERNEL_FINALIZE::get_nd_range(batch_size, Width, dim);
    queue.submit([&](sycl::handler& cgh) {
      KERNEL_FINALIZE task(
          reinterpret_cast<scalar_t*>(conv_states.value().data_ptr()),
          static_cast<int>(conv_states.value().stride(0)),
          static_cast<int>(conv_states.value().stride(-1)),
          static_cast<int>(conv_states.value().stride(-2)),
          reinterpret_cast<const scalar_t*>(conv_states_tmp.data_ptr()),
          cache_indices.has_value() ? reinterpret_cast<const int*>(cache_indices.value().data_ptr()) : nullptr,
          Width,
          dim,
          reinterpret_cast<const int32_t*>(query_start_loc.data_ptr()),
          static_cast<int>(pad_slot_id));
      cgh.parallel_for(range_finalize, task);
    });
  }
}

template <typename scalar_t>
static void launch_causal_conv1d_fwd_token_parallel(
    sycl::queue& queue,
    at::Tensor& out,
    const at::Tensor& x,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias_,
    const std::optional<at::Tensor>& conv_states,
    const at::Tensor& query_start_loc,
    const std::optional<at::Tensor>& cache_indices,
    const std::optional<at::Tensor>& has_initial_state,
    bool silu_activation,
    int64_t pad_slot_id,
    int batch_size,
    int num_virtual_tokens,
    int dim,
    int width) {
#define LAUNCH_WIDTH(W)                                     \
  if (width == W) {                                         \
    launch_causal_conv1d_fwd_token_parallel_w<scalar_t, W>( \
        queue,                                              \
        out,                                                \
        x,                                                  \
        weight,                                             \
        bias_,                                              \
        conv_states,                                        \
        query_start_loc,                                    \
        cache_indices,                                      \
        has_initial_state,                                  \
        silu_activation,                                    \
        pad_slot_id,                                        \
        batch_size,                                         \
        num_virtual_tokens,                                 \
        dim);                                               \
    return;                                                 \
  }
  LAUNCH_WIDTH(2)
  LAUNCH_WIDTH(3)
  LAUNCH_WIDTH(4)
#undef LAUNCH_WIDTH
  TORCH_CHECK(false, "causal_conv1d only supports width between 2 and 4");
}

template <typename scalar_t, int Width>
static void launch_causal_conv1d_update_token_parallel_w(
    sycl::queue& queue,
    at::Tensor& out,
    const at::Tensor& x,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias_,
    at::Tensor& conv_state,
    const std::optional<at::Tensor>& conv_state_indices_,
    bool silu_activation,
    int64_t pad_slot_id,
    int batch_size,
    int dim,
    int conv_state_len) {
  using KERNEL = sgl_conv1d::CausalConv1dUpdateTokenParallelKernel<scalar_t, Width>;
  auto range = KERNEL::get_nd_range(batch_size, dim);
  queue.submit([&](sycl::handler& cgh) {
    KERNEL task(
        reinterpret_cast<scalar_t*>(out.data_ptr()),
        reinterpret_cast<const scalar_t*>(x.data_ptr()),
        static_cast<int>(x.stride(0)),
        static_cast<int>(x.stride(1)),
        static_cast<int>(out.stride(0)),
        static_cast<int>(out.stride(1)),
        reinterpret_cast<const scalar_t*>(weight.data_ptr()),
        static_cast<int>(weight.stride(0)),
        static_cast<int>(weight.stride(1)),
        bias_.has_value() ? reinterpret_cast<const scalar_t*>(bias_.value().data_ptr()) : nullptr,
        reinterpret_cast<scalar_t*>(conv_state.data_ptr()),
        static_cast<int>(conv_state.stride(0)),
        static_cast<int>(conv_state.stride(2)),
        static_cast<int>(conv_state.stride(1)),
        conv_state_len,
        conv_state_indices_.has_value() ? reinterpret_cast<const int*>(conv_state_indices_.value().data_ptr())
                                        : nullptr,
        silu_activation,
        static_cast<int>(pad_slot_id),
        dim);
    cgh.parallel_for(range, task);
  });
}

template <typename scalar_t>
static void launch_causal_conv1d_update_token_parallel(
    sycl::queue& queue,
    at::Tensor& out,
    const at::Tensor& x,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias_,
    at::Tensor& conv_state,
    const std::optional<at::Tensor>& conv_state_indices_,
    bool silu_activation,
    int64_t pad_slot_id,
    int batch_size,
    int dim,
    int conv_state_len,
    int width) {
#define LAUNCH_WIDTH(W)                                        \
  if (width == W) {                                            \
    launch_causal_conv1d_update_token_parallel_w<scalar_t, W>( \
        queue,                                                 \
        out,                                                   \
        x,                                                     \
        weight,                                                \
        bias_,                                                 \
        conv_state,                                            \
        conv_state_indices_,                                   \
        silu_activation,                                       \
        pad_slot_id,                                           \
        batch_size,                                            \
        dim,                                                   \
        conv_state_len);                                       \
    return;                                                    \
  }
  LAUNCH_WIDTH(2)
  LAUNCH_WIDTH(3)
  LAUNCH_WIDTH(4)
#undef LAUNCH_WIDTH
  TORCH_CHECK(false, "causal_conv1d only supports width between 2 and 4");
}

SGL_KERNEL_EXPORT void causal_conv1d_fwd(
    at::Tensor& x,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias_,
    const std::optional<at::Tensor>& conv_states,
    const std::optional<at::Tensor>& query_start_loc,
    const std::optional<at::Tensor>& cache_indices,
    const std::optional<at::Tensor>& has_initial_state,
    bool silu_activation,
    int64_t pad_slot_id) {
  auto input_type = x.scalar_type();
  auto weight_type = weight.scalar_type();
  TORCH_CHECK(
      input_type == at::ScalarType::Float || input_type == at::ScalarType::Half ||
      input_type == at::ScalarType::BFloat16);
  TORCH_CHECK(
      weight_type == at::ScalarType::Float || weight_type == at::ScalarType::Half ||
      weight_type == at::ScalarType::BFloat16);
  TORCH_CHECK(
      weight_type == input_type,
      "weight type must equal to input type, other variations are disabled due to binary size limitations");

  TORCH_CHECK(x.is_xpu(), "x must be an XPU tensor");
  TORCH_CHECK(weight.is_xpu(), "weight must be an XPU tensor");

  const bool varlen = query_start_loc.has_value();
  const auto sizes = x.sizes();
  const int batch_size = varlen ? query_start_loc.value().size(0) - 1 : sizes[0];
  const int dim = varlen ? sizes[0] : sizes[1];
  const int seqlen = varlen ? sizes[1] : sizes[2];
  const int width = weight.size(-1);
  if (varlen) {
    CHECK_SHAPE(x, dim, seqlen);
  } else {
    CHECK_SHAPE(x, batch_size, dim, seqlen);
  }
  CHECK_SHAPE(weight, dim, width);

  TORCH_CHECK(width >= 2 && width <= 4, "causal_conv1d_fwd: width must be 2, 3, or 4");

  if (bias_.has_value()) {
    const auto& bias = bias_.value();
    TORCH_CHECK(bias.scalar_type() == weight_type, "bias dtype must match weight dtype");
    TORCH_CHECK(bias.is_xpu(), "bias must be an XPU tensor");
    TORCH_CHECK(bias.stride(-1) == 1, "bias must be contiguous on last dim");
    CHECK_SHAPE(bias, dim);
  }

  if (has_initial_state.has_value()) {
    const auto& has_initial_state_ = has_initial_state.value();
    TORCH_CHECK(has_initial_state_.scalar_type() == at::ScalarType::Bool, "has_initial_state must be bool");
    TORCH_CHECK(has_initial_state_.is_xpu(), "has_initial_state must be an XPU tensor");
    CHECK_SHAPE(has_initial_state_, batch_size);
  }

  if (query_start_loc.has_value()) {
    const auto& query_start_loc_ = query_start_loc.value();
    TORCH_CHECK(query_start_loc_.scalar_type() == at::ScalarType::Int, "query_start_loc must be int32");
    TORCH_CHECK(query_start_loc_.is_xpu(), "query_start_loc must be an XPU tensor");
  }

  if (cache_indices.has_value()) {
    const auto& cache_indices_ = cache_indices.value();
    TORCH_CHECK(cache_indices_.scalar_type() == at::ScalarType::Int, "cache_indices must be int32");
    TORCH_CHECK(cache_indices_.is_xpu(), "cache_indices must be an XPU tensor");
    CHECK_SHAPE(cache_indices_, batch_size);
  }

  if (conv_states.has_value()) {
    const auto& conv_states_ = conv_states.value();
    TORCH_CHECK(conv_states_.scalar_type() == input_type, "conv_states dtype must match x dtype");
    TORCH_CHECK(conv_states_.is_xpu(), "conv_states must be an XPU tensor");
  }

  at::Tensor out = x;

  // NOTE: prefill (varlen) intentionally does NOT dispatch to the
  // token-parallel design here, unlike decode below. Measured on real
  // hardware: the token-parallel kernel's win (see
  // conv1d_kernel_comparison_conclusions.md) depended on GDN's own
  // token-major (NLC-like) `mixed_qkv` storage, where consecutive channels
  // for one token are contiguous, giving coalesced cross-thread access.
  // This op's varlen `x` is genuinely dim-major (NCL-like, shape
  // `(dim, cu_seqlen)`, `x.stride(0) == cu_seqlen`), matching the upstream
  // causal-conv1d reference API real Mamba2 callers (Falcon-H1, plain
  // Qwen3-Next, etc.) expect. Porting the token-parallel kernel verbatim
  // onto that layout makes cross-thread channel access non-coalesced
  // (each thread reads a different row `cu_seqlen` elements apart) and
  // was measured to *regress* ~2.8x vs. this original kernel (1358us vs.
  // 490us at dim=8192, batch=1, seqlen=1024), not win -- the opposite of
  // what the token-major-only microbenchmark showed. See
  // conv1d_kernel_comparison_conclusions.md's follow-up note for the
  // isolated repro. Decode is unaffected (see below): `causal_conv1d_update`'s
  // dense `x`/`conv_state` have `seqlen == 1`, which collapses the channel
  // stride to 1 regardless of layout, so token-major and dim-major storage
  // are identical there.
  ConvParamsBase params;
  set_conv_params_fwd(
      params,
      batch_size,
      dim,
      seqlen,
      width,
      x,
      weight,
      out,
      bias_,
      silu_activation,
      pad_slot_id,
      query_start_loc,
      cache_indices,
      has_initial_state);

  if (conv_states.has_value()) {
    const auto& conv_states_ = conv_states.value();
    params.conv_states_ptr = conv_states_.data_ptr();
    params.conv_states_batch_stride = conv_states_.stride(0);
    params.conv_states_c_stride = conv_states_.stride(-2);
    params.conv_states_l_stride = conv_states_.stride(-1);
  } else {
    params.conv_states_ptr = nullptr;
  }

#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
  auto profiling_queue = at::xpu::getCurrentXPUStream().queue();
  GPU_Clock timer;
  timer.start();
#endif

  SYCL_DISPATCH_FLOATING_TYPES(at::ScalarType::Half, at::ScalarType::BFloat16, input_type, "causal_conv1d_fwd", [&]() {
    launch_causal_conv1d_fwd<scalar_t>(params);
  });

#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
  // causal_conv1d: 2 * batch * seq_len * channels * kernel_size mult-adds.
  const double flops = 2.0 * static_cast<double>(batch_size) * static_cast<double>(seqlen) * static_cast<double>(dim) *
                       static_cast<double>(width);
  const double bytes = static_cast<double>(x.numel()) * static_cast<double>(x.element_size()) +
                       static_cast<double>(weight.numel()) * static_cast<double>(weight.element_size()) +
                       static_cast<double>(out.numel()) * static_cast<double>(out.element_size());
  ::sglkernel::report_kernel_perf("causal_conv1d_fwd", profiling_queue, timer, bytes, flops);
#endif
}

SGL_KERNEL_EXPORT void causal_conv1d_update(
    at::Tensor& x,
    at::Tensor& conv_state,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias_,
    bool silu_activation,
    const std::optional<at::Tensor>& cache_seqlens_,
    const std::optional<at::Tensor>& conv_state_indices_,
    int64_t pad_slot_id) {
  auto input_type = x.scalar_type();
  auto weight_type = weight.scalar_type();
  TORCH_CHECK(
      input_type == at::ScalarType::Float || input_type == at::ScalarType::Half ||
      input_type == at::ScalarType::BFloat16);
  TORCH_CHECK(
      weight_type == at::ScalarType::Float || weight_type == at::ScalarType::Half ||
      weight_type == at::ScalarType::BFloat16);
  TORCH_CHECK(
      weight_type == input_type,
      "weight type must equal to input type, other variations are disabled due to binary size limitations");
  TORCH_CHECK(conv_state.scalar_type() == input_type);

  TORCH_CHECK(x.is_xpu(), "x must be an XPU tensor");
  TORCH_CHECK(conv_state.is_xpu(), "conv_state must be an XPU tensor");
  TORCH_CHECK(weight.is_xpu(), "weight must be an XPU tensor");

  const int batch_size = x.size(0);
  const int dim = x.size(1);
  const int seqlen = x.size(2);
  const int width = weight.size(-1);
  const int conv_state_len = conv_state.size(2);

  CHECK_SHAPE(x, batch_size, dim, seqlen);
  CHECK_SHAPE(weight, dim, width);
  TORCH_CHECK(conv_state_len >= width - 1, "conv_state_len must be >= width - 1");
  TORCH_CHECK(width >= 2 && width <= 4, "causal_conv1d only supports width between 2 and 4");

  if (bias_.has_value()) {
    const auto& bias = bias_.value();
    TORCH_CHECK(bias.scalar_type() == weight_type, "bias dtype must match weight dtype");
    TORCH_CHECK(bias.is_xpu(), "bias must be an XPU tensor");
    TORCH_CHECK(bias.stride(-1) == 1, "bias must be contiguous on last dim");
    CHECK_SHAPE(bias, dim);
  }

  if (cache_seqlens_.has_value()) {
    const auto& cache_seqlens = cache_seqlens_.value();
    TORCH_CHECK(cache_seqlens.scalar_type() == at::ScalarType::Int, "cache_seqlens must be int32");
    TORCH_CHECK(cache_seqlens.is_xpu(), "cache_seqlens must be an XPU tensor");
    TORCH_CHECK(cache_seqlens.stride(-1) == 1, "cache_seqlens must be contiguous");
    CHECK_SHAPE(cache_seqlens, batch_size);
  }

  if (conv_state_indices_.has_value()) {
    const auto& conv_state_indices = conv_state_indices_.value();
    TORCH_CHECK(conv_state_indices.scalar_type() == at::ScalarType::Int, "conv_state_indices must be int32");
    TORCH_CHECK(conv_state_indices.is_xpu(), "conv_state_indices must be an XPU tensor");
    TORCH_CHECK(conv_state_indices.stride(0) == 1, "conv_state_indices must be contiguous");
    CHECK_SHAPE(conv_state_indices, batch_size);
    const int conv_state_entries = conv_state.size(0);
    CHECK_SHAPE(conv_state, conv_state_entries, dim, conv_state_len);
  } else {
    CHECK_SHAPE(conv_state, batch_size, dim, conv_state_len);
  }

  at::Tensor out = x;

  if (seqlen == 1 && !cache_seqlens_.has_value()) {
    // Token-parallel design: see causal_conv1d_token_parallel.hpp. This is
    // the shape every current SGLang call site uses (single-token decode);
    // faster than the fallback below due to vectorized loads/stores cutting
    // total issued instructions, confirmed with unitrace hardware counters
    // -- see gdn_spec_decode_xpu_investigation/conv1d_kernel_comparison_conclusions.md.
    auto queue = c10::xpu::getCurrentXPUStream().queue();
    SYCL_DISPATCH_FLOATING_TYPES(
        at::ScalarType::Half, at::ScalarType::BFloat16, input_type, "causal_conv1d_update", [&]() {
          launch_causal_conv1d_update_token_parallel<scalar_t>(
              queue,
              out,
              x,
              weight,
              bias_,
              conv_state,
              conv_state_indices_,
              silu_activation,
              pad_slot_id,
              batch_size,
              dim,
              conv_state_len,
              width);
        });
    return;
  }

  // Multi-token (seqlen > 1) or circular-buffer (cache_seqlens) update: no
  // current SGLang call site uses either (multi-token verify goes through
  // Triton; cache_seqlens is never set), so this keeps the original
  // channel-parallel/serial-loop kernel rather than adding the token-
  // parallel design's same-row hazard handling for an untested, uncalled
  // path. See causal_conv1d_token_parallel.hpp's module doc comment.
  ConvParamsBase params;
  set_conv_params_fwd(params, batch_size, dim, seqlen, width, x, weight, out, bias_, silu_activation, pad_slot_id);
  params.conv_state_ptr = conv_state.data_ptr();
  params.conv_state_len = conv_state_len;
  params.conv_state_batch_stride = conv_state.stride(0);
  params.conv_state_c_stride = conv_state.stride(1);
  params.conv_state_l_stride = conv_state.stride(2);
  params.conv_state_indices_ptr =
      conv_state_indices_.has_value() ? conv_state_indices_.value().data_ptr<int>() : nullptr;
  params.cache_seqlens = cache_seqlens_.has_value() ? cache_seqlens_.value().data_ptr<int>() : nullptr;

#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
  auto profiling_queue = at::xpu::getCurrentXPUStream().queue();
  GPU_Clock timer;
  timer.start();
#endif

  SYCL_DISPATCH_FLOATING_TYPES(
      at::ScalarType::Half, at::ScalarType::BFloat16, input_type, "causal_conv1d_update", [&]() {
        launch_causal_conv1d_update<scalar_t>(params);
      });

#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
  const double flops = 2.0 * static_cast<double>(batch_size) * static_cast<double>(seqlen) * static_cast<double>(dim) *
                       static_cast<double>(width);
  const double bytes = static_cast<double>(x.numel()) * static_cast<double>(x.element_size()) +
                       static_cast<double>(weight.numel()) * static_cast<double>(weight.element_size()) +
                       static_cast<double>(conv_state.numel()) * static_cast<double>(conv_state.element_size()) +
                       static_cast<double>(out.numel()) * static_cast<double>(out.element_size());
  ::sglkernel::report_kernel_perf("causal_conv1d_update", profiling_queue, timer, bytes, flops);
#endif
}
