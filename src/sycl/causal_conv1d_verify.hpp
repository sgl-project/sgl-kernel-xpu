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

// Standalone causal-conv1d op backing GDN's target-verify (speculative
// decoding) path. Mirrors SGLang's Triton `causal_conv1d_update` tree-mode
// contract (`_causal_conv1d_update_kernel` with HAS_EAGLE_TREE_CUSTOM_ATTN_MASK)
// so the SYCL path needs no XPU-special-casing in gdn_backend.py.
//
// Each draft token's conv window is built by walking up its EAGLE-tree
// ancestors (not a linear t-1,t-2,... window): `retrieve_next_token` /
// `retrieve_next_sibling` encode the tree as a first-child/next-sibling
// list. `CausalConv1dVerifyTreeDecodeKernel` decodes it into a per-token
// parent index once per batch entry (not redundantly per channel-group,
// and not redundantly per work-group either -- measured: both of
// those cost more than a tiny dedicated kernel launch does), writing both
// `retrieve_parent_token` (consumed later by `gated_delta_rule_verify`)
// and a `parent_idx` scratch buffer that `CausalConv1dUpdateVerifyKernel`
// reads while computing the convolution. Ancestor chains shorter than
// `width-1` fall back to the committed `conv_state` window (read-only;
// never written here). Every token's post-update window is also
// snapshotted into `intermediate_conv_window` so a later accept/commit step
// can seed the next decode round without replaying the tree walk.
#include <torch/all.h>

#include <sycl/sycl.hpp>

namespace sgl_conv1d {

// Draft trees in practice are far smaller; this just bounds the stack-local
// scratch used by callers that size the `parent_idx` tensor.
constexpr int kVerifyMaxCacheSteps = 128;

// Decodes the first-child/next-sibling tree encoding into a per-token
// parent index, once per batch entry. Deliberately a separate, tiny
// kernel (grid=(batch_size)) so `CausalConv1dUpdateVerifyKernel` below
// doesn't redundantly repeat this O(cache_steps) scan once per
// channel-group.
template <typename index_t>
struct CausalConv1dVerifyTreeDecodeKernel {
 public:
  CausalConv1dVerifyTreeDecodeKernel(
      const int* cache_indices,
      const index_t* retrieve_next_token,
      const index_t* retrieve_next_sibling,
      index_t* retrieve_parent_token,
      int* parent_idx,
      const int pad_slot_id,
      const int cache_steps)
      : cache_indices(cache_indices),
        retrieve_next_token(retrieve_next_token),
        retrieve_next_sibling(retrieve_next_sibling),
        retrieve_parent_token(retrieve_parent_token),
        parent_idx(parent_idx),
        pad_slot_id(pad_slot_id),
        cache_steps(cache_steps) {}

  static inline sycl::range<1> get_range(const int batch_size) {
    return sycl::range<1>(batch_size);
  }

  void operator()(sycl::item<1> item) const {
    const int batch_id = item.get_id(0);
    const int cache_idx = (cache_indices != nullptr) ? cache_indices[batch_id] : batch_id;
    if (cache_idx == pad_slot_id) {
      return;
    }
    int* row = parent_idx + batch_id * cache_steps;
#pragma unroll 1
    for (int i = 0; i < cache_steps; ++i) {
      row[i] = 0;
    }
#pragma unroll 1
    for (int t = 0; t < cache_steps; ++t) {
      const index_t child = retrieve_next_token[batch_id * cache_steps + t];
      if (child != -1) {
        row[child] = t;
      }
      const index_t sibling = retrieve_next_sibling[batch_id * cache_steps + t];
      if (sibling != -1) {
        row[sibling] = row[t];
      }
    }
#pragma unroll 1
    for (int t = 0; t < cache_steps; ++t) {
      retrieve_parent_token[batch_id * cache_steps + t] = row[t];
    }
  }

 private:
  const int* cache_indices;
  const index_t* retrieve_next_token;
  const index_t* retrieve_next_sibling;
  index_t* retrieve_parent_token;
  int* parent_idx;
  const int pad_slot_id;
  const int cache_steps;
};

template <typename T, int Width, int ElemsPerItem = 4>
struct CausalConv1dUpdateVerifyKernel {
 public:
  static constexpr int sub_group_size = 32;
  static constexpr int group_size = 256;
  static constexpr int elems_per_item = ElemsPerItem;
  static constexpr int elems_per_group = group_size * elems_per_item;

  CausalConv1dUpdateVerifyKernel(
      T* out,
      const T* x_in,
      const int x_seq_stride,
      const int x_c_stride,
      const int x_token_stride,
      const int out_seq_stride,
      const int out_c_stride,
      const int out_token_stride,
      const T* conv_weights,
      const int weight_c_stride,
      const int weight_width_stride,
      const T* conv_bias,
      const T* conv_states,
      const int conv_states_stride_0,
      const int conv_states_w_stride,
      const int conv_states_d_stride,
      T* intermediate_conv_window,
      const int inter_seq_stride,
      const int inter_step_stride,
      const int inter_dim_stride,
      const int inter_win_stride,
      const int* cache_indices,
      const int* intermediate_state_indices,
      const int* parent_idx,
      const bool silu_activation,
      const int pad_slot_id,
      const int batch_size,
      const int dim,
      const int cache_steps)
      : out(out),
        x_in(x_in),
        x_seq_stride(x_seq_stride),
        x_c_stride(x_c_stride),
        x_token_stride(x_token_stride),
        out_seq_stride(out_seq_stride),
        out_c_stride(out_c_stride),
        out_token_stride(out_token_stride),
        conv_weights(conv_weights),
        weight_c_stride(weight_c_stride),
        weight_width_stride(weight_width_stride),
        conv_bias(conv_bias),
        conv_states(conv_states),
        conv_states_stride_0(conv_states_stride_0),
        conv_states_w_stride(conv_states_w_stride),
        conv_states_d_stride(conv_states_d_stride),
        intermediate_conv_window(intermediate_conv_window),
        inter_seq_stride(inter_seq_stride),
        inter_step_stride(inter_step_stride),
        inter_dim_stride(inter_dim_stride),
        inter_win_stride(inter_win_stride),
        cache_indices(cache_indices),
        intermediate_state_indices(intermediate_state_indices),
        parent_idx(parent_idx),
        silu_activation(silu_activation),
        pad_slot_id(pad_slot_id),
        batch_size(batch_size),
        dim(dim),
        cache_steps(cache_steps) {}

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
    const int channel_id = local_group_id * elems_per_item * group_size + local_id * elems_per_item;

    const int cache_idx = (cache_indices != nullptr) ? cache_indices[batch_id] : batch_id;
    const bool padded = (cache_idx == pad_slot_id);

    if (channel_id >= dim) {
      return;
    }
    if (padded) {
#pragma unroll 1
      for (int t = 0; t < cache_steps; ++t) {
        out[batch_id * out_seq_stride + channel_id * out_c_stride + t * out_token_stride] =
            x_in[batch_id * x_seq_stride + channel_id * x_c_stride + t * x_token_stride];
      }
      return;
    }
    const int active_elems = sycl::min(elems_per_item, dim - channel_id);
    const int scratch_row = (intermediate_state_indices != nullptr) ? intermediate_state_indices[batch_id] : batch_id;
    const int* batch_parent_idx = parent_idx + batch_id * cache_steps;

    // STEP 2: preload weights and the committed window (root's ancestors).
    T local_weights[Width][ElemsPerItem];
#pragma unroll
    for (int w = 0; w < Width; ++w) {
#pragma unroll
      for (int e = 0; e < ElemsPerItem; ++e) {
        local_weights[w][e] =
            (e < active_elems) ? conv_weights[(channel_id + e) * weight_c_stride + w * weight_width_stride] : T(0);
      }
    }

    T bias_local[ElemsPerItem];
#pragma unroll
    for (int e = 0; e < ElemsPerItem; ++e) {
      bias_local[e] = (conv_bias != nullptr && e < active_elems) ? conv_bias[channel_id + e] : T(0);
    }

    // conv_states window columns, oldest (col[0]) to most recent (col[Width-2]).
    T col[Width - 1][ElemsPerItem];
#pragma unroll
    for (int w = 0; w < Width - 1; ++w) {
#pragma unroll
      for (int e = 0; e < ElemsPerItem; ++e) {
        col[w][e] = (conv_states != nullptr && e < active_elems)
                        ? conv_states
                              [cache_idx * conv_states_stride_0 + w * conv_states_w_stride +
                               (channel_id + e) * conv_states_d_stride]
                        : T(0);
      }
    }

    // STEP 3: compute each draft token's convolution by walking its
    // ancestor chain (parent -> grandparent -> ...), falling back to the
    // committed window once the chain runs out (reaches token 0's "parent").
#pragma unroll 1
    for (int idx_token = 0; idx_token < cache_steps; ++idx_token) {
      float acc[ElemsPerItem];
#pragma unroll
      for (int e = 0; e < ElemsPerItem; ++e) {
        acc[e] = static_cast<float>(bias_local[e]);
      }

      T hist[Width][ElemsPerItem];
      int cur = idx_token;
      T cur_val[ElemsPerItem];
#pragma unroll
      for (int e = 0; e < ElemsPerItem; ++e) {
        cur_val[e] = (e < active_elems)
                         ? x_in[batch_id * x_seq_stride + (channel_id + e) * x_c_stride + idx_token * x_token_stride]
                         : T(0);
      }
      int which_col = 0;
#pragma unroll
      for (int j = 0; j < Width; ++j) {
#pragma unroll
        for (int e = 0; e < ElemsPerItem; ++e) {
          hist[j][e] = cur_val[e];
          acc[e] += static_cast<float>(cur_val[e]) * static_cast<float>(local_weights[Width - 1 - j][e]);
        }
        if (cur > 0) {
          cur = batch_parent_idx[cur];
#pragma unroll
          for (int e = 0; e < ElemsPerItem; ++e) {
            cur_val[e] = (e < active_elems)
                             ? x_in[batch_id * x_seq_stride + (channel_id + e) * x_c_stride + cur * x_token_stride]
                             : T(0);
          }
        } else {
          const int col_idx = sycl::max(0, (Width - 2) + which_col);
#pragma unroll
          for (int e = 0; e < ElemsPerItem; ++e) {
            cur_val[e] = col[col_idx][e];
          }
          which_col -= 1;
        }
      }

      // Snapshot the post-token window: hist[0] (itself) is most recent ->
      // last slot; hist[Width-2] (oldest kept ancestor) -> slot 0.
      T* inter_base = intermediate_conv_window + static_cast<int64_t>(scratch_row) * inter_seq_stride +
                      idx_token * inter_step_stride;
#pragma unroll
      for (int j = 0; j <= Width - 2; ++j) {
        const int slot = Width - 2 - j;
#pragma unroll
        for (int e = 0; e < ElemsPerItem; ++e) {
          if (e < active_elems) {
            inter_base[(channel_id + e) * inter_dim_stride + slot * inter_win_stride] = hist[j][e];
          }
        }
      }

      if (silu_activation) {
#pragma unroll
        for (int e = 0; e < ElemsPerItem; ++e) {
          acc[e] = acc[e] / (1.0f + sycl::native::exp(-acc[e]));
        }
      }
#pragma unroll
      for (int e = 0; e < ElemsPerItem; ++e) {
        if (e < active_elems) {
          out[batch_id * out_seq_stride + (channel_id + e) * out_c_stride + idx_token * out_token_stride] =
              static_cast<T>(acc[e]);
        }
      }
    }
  }

 private:
  T* out;
  const T* x_in;
  const int x_seq_stride;
  const int x_c_stride;
  const int x_token_stride;
  const int out_seq_stride;
  const int out_c_stride;
  const int out_token_stride;
  const T* conv_weights;
  const int weight_c_stride;
  const int weight_width_stride;
  const T* conv_bias;
  const T* conv_states;
  const int conv_states_stride_0;
  const int conv_states_w_stride;
  const int conv_states_d_stride;
  T* intermediate_conv_window;
  const int inter_seq_stride;
  const int inter_step_stride;
  const int inter_dim_stride;
  const int inter_win_stride;
  const int* cache_indices;
  const int* intermediate_state_indices;
  const int* parent_idx;
  const bool silu_activation;
  const int pad_slot_id;
  const int batch_size;
  const int dim;
  const int cache_steps;
};

}  // namespace sgl_conv1d
