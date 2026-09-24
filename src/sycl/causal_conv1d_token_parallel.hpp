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

// Token-parallel causal conv1d kernels backing the general-purpose,
// standalone `causal_conv1d_fwd` / `causal_conv1d_update` ops
// (see causal_conv1d.cpp).
//
// Design adapted from the GDN-specific token-parallel kernel in
// gdn_attn/causal_conv1d.hpp (one work-item-group per token, vectorized
// ElemsPerItem channels/thread), which a microbenchmark comparison
// (benchmark/bench_conv1d_split_compare.py, see
// gdn_spec_decode_xpu_investigation/conv1d_kernel_comparison_conclusions.md)
// showed is faster than the channel-parallel/serial-loop design previously
// used here: ~2.1-3.65x for varlen prefill (parallelism scales with total
// tokens, not just batch*dim) and ~1.0-1.4x for single-token decode (fewer,
// wider vector instructions -> less overhead, higher achieved memory
// bandwidth; confirmed with unitrace hardware counters).
//
// Differences from the GDN-internal kernel this was adapted from (needed
// because this backs the *general-purpose* op, used by non-GDN Mamba2
// models too, not just GDN):
//  - `cache_indices` is optional here (falls back to identity: state row
//    for batch `b` is row `b`), matching causal_conv1d_fwd's contract.
//  - `conv_states`/`conv_state` may be entirely absent (no persistent
//    state kept at all), matching causal_conv1d_fwd's contract.
//  - `has_initial_state` defaults to **false** when absent, matching the
//    general-purpose op's documented contract. (Note: GDN's own internal
//    kernel defaults this to *true*, since GDN always has a persistent
//    per-request cache slot and only uses this flag to mark a request's
//    very first chunk -- an intentional, narrower convention that does not
//    apply here.)
//  - Explicit tail-masking so `dim` need not be a multiple of
//    `ElemsPerItem`.
//
// Not handled by the decode/update kernel here (falls back to the
// existing serial-loop `CausalConv1dUpdateKernel` in causal_conv1d.cpp,
// unchanged, for these cases -- see the dispatch in launch_causal_conv1d_update):
//  - `seqlen > 1` (multi-token decode/update): parallelizing per-token
//    would race the same conv_state row's trailing-window write against
//    another token's initial-state read within the same launch, since SYCL
//    provides no cross-work-group ordering guarantee. This dense
//    multi-token case isn't exercised by any current SGLang call site
//    (only single-token decode uses the compiled op; multi-token verify
//    goes through Triton), so it isn't worth the extra staged-commit
//    kernel to remove this restriction yet.
//  - `cache_seqlens` (circular conv-state buffer): no current SGLang call
//    site ever passes a non-null value (grep-confirmed); genuinely
//    unimplemented here.
//
// The varlen prefill kernel always needs a per-token batch lookup (a
// linear scan over `query_start_loc`) since sequence lengths vary; the
// dense (non-varlen, fixed `seqlen` per batch) prefill case does NOT need
// query_start_loc or a scan since batch/token math is direct division --
// but it also falls back to the old serial-loop kernel here, since dense
// prefill has the same same-row trailing-state hazard as multi-token
// decode/update above, and (like cache_seqlens) has no current caller.

#include <torch/all.h>

#include <sycl/sycl.hpp>

namespace sgl_conv1d {

// Token-parallel causal conv1d kernel for **varlen prefill**
// (`causal_conv1d_fwd` with `query_start_loc` provided). One work-item
// group per token; each work-item handles `ElemsPerItem` channels.
template <typename T, int Width, int ElemsPerItem = 4>
struct CausalConv1dFwdTokenParallelKernel {
 public:
  static constexpr int sub_group_size = 32;
  static constexpr int group_size = 256;
  static constexpr int elems_per_item = ElemsPerItem;
  static constexpr int elems_per_group = group_size * elems_per_item;

  CausalConv1dFwdTokenParallelKernel(
      T* out,
      const T* x_in,
      const int x_token_stride,
      const int x_c_stride,
      const int out_token_stride,
      const int out_c_stride,
      const T* conv_weights,
      const int weight_c_stride,
      const int weight_width_stride,
      const T* conv_bias,
      T* conv_states,
      const int conv_states_stride_0,
      const int conv_w_stride,
      const int conv_d_stride,
      T* conv_states_tmp,
      const int32_t* query_start_loc,
      const int* cache_indices,
      const bool* has_initial_state,
      const bool silu_activation,
      const int pad_slot_id,
      const int batch_size,
      const int dim)
      : out(out),
        x_in(x_in),
        x_token_stride(x_token_stride),
        x_c_stride(x_c_stride),
        out_token_stride(out_token_stride),
        out_c_stride(out_c_stride),
        conv_weights(conv_weights),
        weight_c_stride(weight_c_stride),
        weight_width_stride(weight_width_stride),
        conv_bias(conv_bias),
        conv_states(conv_states),
        conv_states_stride_0(conv_states_stride_0),
        conv_w_stride(conv_w_stride),
        conv_d_stride(conv_d_stride),
        conv_states_tmp(conv_states_tmp),
        query_start_loc(query_start_loc),
        cache_indices(cache_indices),
        has_initial_state(has_initial_state),
        silu_activation(silu_activation),
        pad_slot_id(pad_slot_id),
        batch_size(batch_size),
        dim(dim) {}

  static inline sycl::nd_range<2> get_nd_range(const int num_virtual_tokens, const int dim) {
    const int groups_per_token = (dim + elems_per_group - 1) / elems_per_group;
    sycl::range<2> local(1, group_size);
    sycl::range<2> global(num_virtual_tokens, groups_per_token);
    return sycl::nd_range<2>(global * local, local);
  }

  [[sycl::reqd_sub_group_size(sub_group_size)]] void operator()(sycl::nd_item<2> item) const {
    const int token_id = item.get_group(0);
    const int local_group_id = item.get_group(1);
    const int local_id = item.get_local_linear_id();
    const int channel_id = local_group_id * elems_per_group + local_id * elems_per_item;

    if (channel_id >= dim) {
      return;
    }
    // Tail mask: dim need not be a multiple of elems_per_item.
    const int active_elems = sycl::min(elems_per_item, dim - channel_id);

    // Per-token batch lookup: variable sequence lengths require a scan
    // (the general-purpose channel-parallel design gets batch_id for free
    // from its grid; that grid trades this scan away for occupancy that's
    // capped at batch*dim -- see the module doc comment for the tradeoff).
    int batch_id = batch_size - 1;
    int seq_start_offset = 0;
    int seq_end_offset = 0;
    for (int i = 0; i < batch_size; ++i) {
      if (token_id < query_start_loc[i + 1]) {
        batch_id = i;
        seq_start_offset = query_start_loc[i];
        seq_end_offset = query_start_loc[i + 1];
        break;
      }
    }

    const int cache_idx = (cache_indices != nullptr) ? cache_indices[batch_id] : batch_id;
    if (cache_idx == pad_slot_id) {
      return;
    }
    // Matches causal_conv1d_fwd's documented contract: no initial state is
    // used unless has_initial_state explicitly says so for this batch entry.
    const bool has_init = (has_initial_state != nullptr) && has_initial_state[batch_id];
    const bool has_conv_states = (conv_states != nullptr);

    T local_weights[Width * elems_per_item];
#pragma unroll
    for (int i = 0; i < Width; ++i) {
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        local_weights[Width * e + i] =
            (e < active_elems) ? conv_weights[(channel_id + e) * weight_c_stride + i * weight_width_stride] : T(0);
      }
    }

    T local_input[Width * elems_per_item];
#pragma unroll
    for (int i = 0; i < Width; ++i) {
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        local_input[Width * e + i] = T(0);
      }
    }

    const int seq_cu_len = token_id - seq_start_offset + 1;
    const int input_load_len = seq_cu_len >= Width ? Width : seq_cu_len;
    const int states_load_len = seq_cu_len >= Width ? 0 : Width - input_load_len;
    if (states_load_len != 0 && has_init && has_conv_states) {
      T* conv_states_ptr = conv_states + cache_idx * conv_states_stride_0;
#pragma unroll
      for (int i = 0; i < states_load_len; ++i) {
#pragma unroll
        for (int e = 0; e < elems_per_item; ++e) {
          if (e < active_elems) {
            local_input[Width * e + i] =
                conv_states_ptr[(Width - 1 - states_load_len + i) * conv_w_stride + (channel_id + e) * conv_d_stride];
          }
        }
      }
    }

#pragma unroll
    for (int i = 0; i < input_load_len; ++i) {
      const int load_token = token_id - input_load_len + 1 + i;
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        if (e < active_elems) {
          local_input[Width * e + states_load_len + i] =
              x_in[load_token * x_token_stride + (channel_id + e) * x_c_stride];
        }
      }
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

    // Save the trailing window as the next-chunk initial state. For a
    // multi-token sequence this must be staged into conv_states_tmp and
    // committed by a follow-up FinalizeConvStatesKernel launch, since
    // another work-item of the same batch entry (an earlier token) may
    // concurrently be reading conv_states_ptr as *initial* state in this
    // same launch -- SYCL gives no cross-work-group ordering guarantee, so
    // writing conv_states_ptr directly here would race. Single-token
    // sequences (decode-shaped: seq_end_offset - seq_start_offset == 1)
    // have no such reader in this launch, so they can write directly.
    if (has_conv_states) {
      if (seq_end_offset - seq_start_offset > 1) {
        if (seq_end_offset - 1 == token_id) {
#pragma unroll
          for (int i = 0; i < Width - 1; ++i) {
#pragma unroll
            for (int e = 0; e < elems_per_item; ++e) {
              if (e < active_elems) {
                conv_states_tmp[batch_id * (Width - 1) * dim + i * dim + channel_id + e] =
                    local_input[Width * e + i + 1];
              }
            }
          }
        }
      } else {
        T* conv_states_ptr = conv_states + cache_idx * conv_states_stride_0;
#pragma unroll
        for (int i = 0; i < Width - 1; ++i) {
#pragma unroll
          for (int e = 0; e < elems_per_item; ++e) {
            if (e < active_elems) {
              conv_states_ptr[i * conv_w_stride + (channel_id + e) * conv_d_stride] = local_input[Width * e + i + 1];
            }
          }
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
        out[token_id * out_token_stride + (channel_id + e) * out_c_stride] = static_cast<T>(res[e]);
      }
    }
  }

 private:
  T* out;
  const T* x_in;
  const int x_token_stride;
  const int x_c_stride;
  const int out_token_stride;
  const int out_c_stride;
  const T* conv_weights;
  const int weight_c_stride;
  const int weight_width_stride;
  const T* conv_bias;
  T* conv_states;
  const int conv_states_stride_0;
  const int conv_w_stride;
  const int conv_d_stride;
  T* conv_states_tmp;
  const int32_t* query_start_loc;
  const int* cache_indices;
  const bool* has_initial_state;
  const bool silu_activation;
  const int pad_slot_id;
  const int batch_size;
  const int dim;
};

// Commits the staged trailing-window state (written by
// CausalConv1dFwdTokenParallelKernel above into conv_states_tmp for
// multi-token sequences) into the real conv_states buffer. Must run as a
// separate, dependent kernel launch after the main kernel completes, so
// that all initial-state reads for this batch entry are guaranteed to have
// already happened (see the race explanation above).
template <typename T>
struct FinalizeConvStatesKernel {
 public:
  static constexpr int sub_group_size = 32;
  static constexpr int group_size = 256;
  static constexpr int elems_per_item = 4;
  static constexpr int elems_per_group = group_size * elems_per_item;

  FinalizeConvStatesKernel(
      T* conv_states,
      const int conv_states_stride_0,
      const int conv_w_stride,
      const int conv_d_stride,
      const T* conv_states_tmp,
      const int* cache_indices,
      const int width,
      const int dim,
      const int32_t* query_start_loc,
      const int pad_slot_id)
      : conv_states(conv_states),
        conv_states_stride_0(conv_states_stride_0),
        conv_w_stride(conv_w_stride),
        conv_d_stride(conv_d_stride),
        conv_states_tmp(conv_states_tmp),
        cache_indices(cache_indices),
        width(width),
        dim(dim),
        query_start_loc(query_start_loc),
        pad_slot_id(pad_slot_id) {}

  static inline sycl::nd_range<3> get_nd_range(const int batch_size, const int width, const int dim) {
    const int groups_per_token = (dim + elems_per_group - 1) / elems_per_group;
    sycl::range<3> local(1, 1, group_size);
    sycl::range<3> global(batch_size, (width - 1), groups_per_token);
    return sycl::nd_range<3>(global * local, local);
  }

  [[sycl::reqd_sub_group_size(sub_group_size)]] void operator()(sycl::nd_item<3> item) const {
    const int batch_id = item.get_group(0);
    const int width_id = item.get_group(1);
    const int local_group_id = item.get_group(2);
    const int local_id = item.get_local_linear_id();
    const int elems_start_offset_group = local_group_id * elems_per_group;

    const int seq_start_offset = query_start_loc[batch_id];
    const int seq_end_offset = query_start_loc[batch_id + 1];
    if (seq_end_offset - seq_start_offset <= 1) {
      // Only multi-token (prefill) sequences stage through conv_states_tmp;
      // single-token sequences already wrote conv_states directly.
      return;
    }

    const int cache_idx = (cache_indices != nullptr) ? cache_indices[batch_id] : batch_id;
    if (cache_idx == pad_slot_id) {
      return;
    }
    T* conv_states_ptr = conv_states + cache_idx * conv_states_stride_0;
    const T* conv_states_tmp_ptr = conv_states_tmp + batch_id * (width - 1) * dim;
    // Bounds check against dim is required: when dim is not an exact
    // multiple of elems_per_group, the group covering the tail would
    // otherwise write/read past the tensor end.
    for (int i = elems_start_offset_group + local_id; i < (local_group_id + 1) * elems_per_group && i < dim;
         i += group_size) {
      conv_states_ptr[width_id * conv_w_stride + i * conv_d_stride] = conv_states_tmp_ptr[width_id * dim + i];
    }
  }

 private:
  T* conv_states;
  const int conv_states_stride_0;
  const int conv_w_stride;
  const int conv_d_stride;
  const T* conv_states_tmp;
  const int* cache_indices;
  const int width;
  const int dim;
  const int32_t* query_start_loc;
  const int pad_slot_id;
};

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
