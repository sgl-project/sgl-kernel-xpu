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

// Token-parallel single-token decode for the standalone causal_conv1d_update
// op (see causal_conv1d.cpp). Varlen prefill uses the channel-parallel
// causal_conv1d_fwd kernel; its token-parallel implementation was removed
// because real dim-major inputs made it slower.
//
// Multi-token updates and updates with cache_seqlens fall back to the
// channel-parallel update kernel. Tail masking supports arbitrary dim.

#include <torch/all.h>

#include <sycl/sycl.hpp>

namespace sgl_conv1d {

// Token-parallel causal conv1d kernel for **single-token decode/update**
// (`causal_conv1d_update` with `seqlen == 1` and no `cache_seqlens`). One
// work-item group per batch entry (== per token, since seqlen == 1); no
// batch-lookup scan is needed (batch_id is the grid index directly), and
// the trailing state can be written in-place with no cross-token race
// (each batch entry contributes exactly one token to this launch).
template <typename T, int Width, int ElemsPerItem = 4>
struct CausalConv1dUpdateTokenParallelKernel {
 public:
  static constexpr int sub_group_size = 32;
  static constexpr int group_size = 256;
  static constexpr int elems_per_item = ElemsPerItem;
  static constexpr int elems_per_group = group_size * elems_per_item;

  CausalConv1dUpdateTokenParallelKernel(
      T* out,
      const T* x_in,
      const int x_batch_stride,
      const int x_c_stride,
      const int out_batch_stride,
      const int out_c_stride,
      const T* conv_weights,
      const int weight_c_stride,
      const int weight_width_stride,
      const T* conv_bias,
      T* conv_state,
      const int conv_state_batch_stride,
      const int conv_state_l_stride,
      const int conv_state_c_stride,
      const int conv_state_len,
      const int* conv_state_indices,
      const bool silu_activation,
      const int pad_slot_id,
      const int dim)
      : out(out),
        x_in(x_in),
        x_batch_stride(x_batch_stride),
        x_c_stride(x_c_stride),
        out_batch_stride(out_batch_stride),
        out_c_stride(out_c_stride),
        conv_weights(conv_weights),
        weight_c_stride(weight_c_stride),
        weight_width_stride(weight_width_stride),
        conv_bias(conv_bias),
        conv_state(conv_state),
        conv_state_batch_stride(conv_state_batch_stride),
        conv_state_l_stride(conv_state_l_stride),
        conv_state_c_stride(conv_state_c_stride),
        conv_state_len(conv_state_len),
        conv_state_indices(conv_state_indices),
        silu_activation(silu_activation),
        pad_slot_id(pad_slot_id),
        dim(dim) {}

  static inline sycl::nd_range<2> get_nd_range(const int batch_size, const int dim) {
    const int groups_per_token = (dim + elems_per_group - 1) / elems_per_group;
    sycl::range<2> local(1, group_size);
    sycl::range<2> global(batch_size, groups_per_token);
    return sycl::nd_range<2>(global * local, local);
  }

  [[sycl::reqd_sub_group_size(sub_group_size)]] void operator()(sycl::nd_item<2> item) const {
    const int batch_id = item.get_group(0);
    const int local_group_id = item.get_group(1);
    const int local_id = item.get_local_linear_id();
    const int channel_id = local_group_id * elems_per_group + local_id * elems_per_item;

    if (channel_id >= dim) {
      return;
    }
    const int active_elems = sycl::min(elems_per_item, dim - channel_id);

    const int cache_idx = (conv_state_indices != nullptr) ? conv_state_indices[batch_id] : batch_id;
    if (cache_idx == pad_slot_id) {
      return;
    }

    T local_weights[Width * elems_per_item];
#pragma unroll
    for (int i = 0; i < Width; ++i) {
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        local_weights[Width * e + i] =
            (e < active_elems) ? conv_weights[(channel_id + e) * weight_c_stride + i * weight_width_stride] : T(0);
      }
    }

    // Sliding window: the first (Width - 1) entries are history read from
    // conv_state (state_len may exceed Width - 1; only the trailing
    // Width - 1 entries are the relevant history), the last is the
    // current token read from x_in.
    T local_input[Width * elems_per_item];
#pragma unroll
    for (int i = 0; i < Width - 1; ++i) {
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        if (e < active_elems) {
          local_input[Width * e + i] = conv_state
              [cache_idx * conv_state_batch_stride + (channel_id + e) * conv_state_c_stride +
               (conv_state_len - (Width - 1) + i) * conv_state_l_stride];
        } else {
          local_input[Width * e + i] = T(0);
        }
      }
    }
#pragma unroll
    for (int e = 0; e < elems_per_item; ++e) {
      local_input[Width * e + (Width - 1)] =
          (e < active_elems) ? x_in[batch_id * x_batch_stride + (channel_id + e) * x_c_stride] : T(0);
    }

    float res[elems_per_item];
#pragma unroll
    for (int e = 0; e < elems_per_item; ++e) {
      res[e] = 0.0f;
    }
#pragma unroll
    for (int i = 0; i < Width; ++i) {
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        res[e] += static_cast<float>(local_input[Width * e + i]) * static_cast<float>(local_weights[Width * e + i]);
      }
    }

    if (conv_bias != nullptr) {
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        if (e < active_elems) {
          res[e] += static_cast<float>(conv_bias[channel_id + e]);
        }
      }
    }

    // Shift the history window by one and write back: no cross-token race
    // here since this kernel is only used when seqlen == 1 (one token per
    // batch entry per launch).
#pragma unroll
    for (int i = 0; i < Width - 1; ++i) {
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        if (e < active_elems) {
          conv_state
              [cache_idx * conv_state_batch_stride + (channel_id + e) * conv_state_c_stride +
               (conv_state_len - (Width - 1) + i) * conv_state_l_stride] = local_input[Width * e + i + 1];
        }
      }
    }

    if (silu_activation) {
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        // Use native exp here to keep SiLU on the XPU fast-math path.
        res[e] = res[e] / (1.0f + sycl::native::exp(-res[e]));
      }
    }

#pragma unroll
    for (int e = 0; e < elems_per_item; ++e) {
      if (e < active_elems) {
        out[batch_id * out_batch_stride + (channel_id + e) * out_c_stride] = static_cast<T>(res[e]);
      }
    }
  }

 private:
  T* out;
  const T* x_in;
  const int x_batch_stride;
  const int x_c_stride;
  const int out_batch_stride;
  const int out_c_stride;
  const T* conv_weights;
  const int weight_c_stride;
  const int weight_width_stride;
  const T* conv_bias;
  T* conv_state;
  const int conv_state_batch_stride;
  const int conv_state_l_stride;
  const int conv_state_c_stride;
  const int conv_state_len;
  const int* conv_state_indices;
  const bool silu_activation;
  const int pad_slot_id;
  const int dim;
};

}  // namespace sgl_conv1d
