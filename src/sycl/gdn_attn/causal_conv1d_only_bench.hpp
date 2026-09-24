#pragma once

// Benchmark-only, standalone extraction of the GDN-specific causal-conv1d
// design (`causal_conv1d_kernel` in causal_conv1d.hpp), stripped of the
// fused q/k/v/z/b/a split & reorder logic that kernel also performs. This
// isolates the *conv1d-only* cost of the token-parallel / no-serial-loop
// design so it can be compared apples-to-apples against the general-purpose
// channel-parallel / serial-loop design in ../causal_conv1d.cpp
// (`causal_conv1d_fwd`/`causal_conv1d_update`).
//
// This file is NOT used by the production `gdn_attention` op and is not a
// deliverable of B.1b (the real standalone conv1d-verify op) -- it exists
// solely to produce a fair, isolated microbenchmark for choosing which
// design B.1b should build on. See benchmark/bench_conv1d_split_compare.py.
//
// Kept as close as possible (same parallelization strategy, same per-token
// batch lookup, same conv_states_tmp + update_states_kernel two-launch
// scheme for prefill, same adaptive ElemsPerItem occupancy heuristic) to the
// original so the comparison reflects the real design tradeoff, not an
// unrelated implementation difference.

#include <torch/all.h>

#include <sycl/sycl.hpp>

#include "causal_conv1d.hpp"  // reuses gdn::update_states_kernel and gdn::ActMode verbatim
#include "gdn_attn_utils.h"

namespace gdn_bench {

// SkipBatchScan=true is a decode-only fast path: since every decode
// "sequence" contributes exactly one token, batch_id == token_id trivially
// and the O(batch_size) linear scan over query_start_loc (needed for
// prefill, where multiple tokens share a batch_id) is pure waste. This
// isolates whether that scan is the cause of GDN-style's decode slowdown
// at large batch sizes (see bench_conv1d_split_compare.py's decode sweep).
template <typename T, int Width, int ElemsPerItem = 4, bool SkipBatchScan = false>
struct causal_conv1d_only_kernel {
 public:
  static constexpr int sub_group_size = 32;
  static constexpr int group_size = 256;
  static constexpr int elems_per_item = ElemsPerItem;
  static constexpr int elems_per_group = group_size * elems_per_item;

  causal_conv1d_only_kernel(
      T* out,
      const T* x_in,
      const T* conv_weights,
      const T* conv_bias,
      T* conv_states,
      const int conv_states_stride_0,
      const int conv_w_stride,
      const int conv_d_stride,
      T* conv_states_tmp,
      const int32_t* query_start_loc,
      const int* cache_indices,
      const bool* has_initial_state,
      const gdn::ActMode& act_mode,
      const int& pad_slot_id,
      const int& batch_size,
      const int& num_virtual_tokens,
      const int& dim)
      : out(out),
        x_in(x_in),
        conv_weights(conv_weights),
        conv_bias(conv_bias),
        conv_states(conv_states),
        conv_states_stride_0(conv_states_stride_0),
        conv_w_stride(conv_w_stride),
        conv_d_stride(conv_d_stride),
        conv_states_tmp(conv_states_tmp),
        query_start_loc(query_start_loc),
        cache_indices(cache_indices),
        has_initial_state(has_initial_state),
        act_mode(act_mode),
        pad_slot_id(pad_slot_id),
        batch_size(batch_size),
        num_virtual_tokens(num_virtual_tokens),
        dim(dim) {}

  static inline sycl::nd_range<2> get_nd_range(const int num_virtual_tokens, const int dim) {
    const int groups_per_token = (dim + elems_per_group - 1) / elems_per_group;
    sycl::range<2> local(1, group_size);
    sycl::range<2> global(num_virtual_tokens, groups_per_token);
    return sycl::nd_range<2>(global * local, local);
  }

  static inline void act_swish(float& x, float beta = 1.0f) {
    x = x / (1.0f + sycl::exp(-x * beta));
  }
  static inline void act_silu(float& x) {
    act_swish(x, 1.0f);
  }

  [[sycl::reqd_sub_group_size(sub_group_size)]] void operator()(sycl::nd_item<2> item) const {
    const int token_id = item.get_group(0);
    const int local_group_id = item.get_group(1);
    const int local_id = item.get_local_linear_id();
    const int channel_id = local_group_id * elems_per_group + local_id * elems_per_item;

    if (channel_id >= dim) {
      return;
    }

    int batch_id;
    int seq_start_offset;
    int seq_end_offset;
    if constexpr (SkipBatchScan) {
      // Decode fast path: one token per sequence, so batch_id == token_id
      // and the window is always [token_id, token_id + 1) -- no scan needed.
      batch_id = token_id;
      seq_start_offset = token_id;
      seq_end_offset = token_id + 1;
    } else {
      // Same per-token linear scan over batches as the original GDN kernel
      // (the general-purpose design gets batch_id for free from the grid).
      batch_id = batch_size - 1;
      seq_start_offset = 0;
      seq_end_offset = 0;
      for (int i = 0; i < batch_size; ++i) {
        if (token_id < query_start_loc[i + 1]) {
          batch_id = i;
          seq_start_offset = query_start_loc[i];
          seq_end_offset = query_start_loc[i + 1];
          break;
        }
      }
    }

    const int states_id = cache_indices[batch_id];
    if (states_id == pad_slot_id) {
      return;
    }
    const bool has_init = (has_initial_state == nullptr) || has_initial_state[batch_id];

    T* conv_states_ptr = conv_states + states_id * conv_states_stride_0;

    T local_weights[Width * elems_per_item];
#pragma unroll
    for (int i = 0; i < Width; ++i) {
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        local_weights[Width * e + i] = conv_weights[(channel_id + e) * Width + i];
      }
    }

    T local_input[Width * elems_per_item];
#pragma unroll
    for (int i = 0; i < Width; ++i) {
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        local_input[Width * e + i] = 0.0f;
      }
    }

    const int seq_cu_len = token_id - seq_start_offset + 1;
    const int input_load_len = seq_cu_len >= Width ? Width : seq_cu_len;
    const int states_load_len = seq_cu_len >= Width ? 0 : Width - input_load_len;
    if (states_load_len != 0 && has_init) {
#pragma unroll
      for (int i = 0; i < states_load_len; ++i) {
#pragma unroll
        for (int e = 0; e < elems_per_item; ++e) {
          local_input[Width * e + i] =
              conv_states_ptr[(Width - 1 - states_load_len + i) * conv_w_stride + (channel_id + e) * conv_d_stride];
        }
      }
    }

    // NOTE: no token_indx remap here (unlike the production kernel) -- this
    // isolates conv1d-only cost; token_indx handling would apply identically
    // (or not at all) to either design and is orthogonal to this comparison.
#pragma unroll
    for (int i = 0; i < input_load_len; ++i) {
      const int load_token = token_id - input_load_len + 1 + i;
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        local_input[Width * e + states_load_len + i] = x_in[load_token * dim + channel_id + e];
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
        res[e] += static_cast<float>(conv_bias[channel_id + e]);
      }
    }

    // save states: staged temp buffer for prefill (needs a follow-up
    // update_states_kernel launch), in-place for decode (single token/batch).
    if (seq_end_offset - seq_start_offset > 1) {
      if (seq_end_offset - 1 == token_id) {
#pragma unroll
        for (int i = 0; i < Width - 1; ++i) {
#pragma unroll
          for (int e = 0; e < elems_per_item; ++e) {
            conv_states_tmp[batch_id * (Width - 1) * dim + i * dim + channel_id + e] = local_input[Width * e + i + 1];
          }
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < Width - 1; ++i) {
#pragma unroll
        for (int e = 0; e < elems_per_item; ++e) {
          conv_states_ptr[i * conv_w_stride + (channel_id + e) * conv_d_stride] = local_input[Width * e + i + 1];
        }
      }
    }

    if (act_mode == gdn::ActMode::silu) {
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        act_silu(res[e]);
      }
    } else if (act_mode == gdn::ActMode::swish) {
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        act_swish(res[e]);
      }
    }

#pragma unroll
    for (int e = 0; e < elems_per_item; ++e) {
      out[token_id * dim + channel_id + e] = static_cast<T>(res[e]);
    }
  }

 private:
  T* out;
  const T* x_in;
  const T* conv_weights;
  const T* conv_bias;
  T* conv_states;
  const int conv_states_stride_0;
  const int conv_w_stride;
  const int conv_d_stride;
  T* conv_states_tmp;
  const int32_t* query_start_loc;
  const int* cache_indices;
  const bool* has_initial_state;
  const gdn::ActMode act_mode;
  const int pad_slot_id;
  const int batch_size;
  const int num_virtual_tokens;
  const int dim;
};

// Prefill variant: launches the main conv kernel plus the follow-up
// update_states_kernel to finalize the trailing conv-state window (matching
// the two-launch scheme the original design pays for prefill).
template <typename T, int Width>
void causal_conv1d_only_prefill(
    sycl::queue& queue,
    torch::Tensor& out,
    const torch::Tensor& x_in,
    const torch::Tensor& conv_weights,
    const std::optional<torch::Tensor>& conv_bias,
    torch::Tensor& conv_states,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& cache_indices,
    const std::optional<torch::Tensor>& has_initial_state,
    const gdn::ActMode& act_mode,
    const int pad_slot_id) {
  const int batch_size = query_start_loc.size(0) - 1;
  const int num_virtual_tokens = out.size(0);
  const int dim = conv_weights.size(0);
  const int conv_states_stride_0 = conv_states.stride(0);
  const int conv_w_stride = conv_states.stride(1);
  const int conv_d_stride = conv_states.stride(2);

  auto conv_states_tmp = torch::empty(
      {std::max(batch_size, 1), Width - 1, dim}, torch::dtype(conv_states.dtype()).device(conv_states.device()));

  using KERNEL_MAIN = causal_conv1d_only_kernel<T, Width, 4>;
  auto range_main = KERNEL_MAIN::get_nd_range(num_virtual_tokens, dim);
  queue.submit([&](sycl::handler& cgh) {
    KERNEL_MAIN task(
        reinterpret_cast<T*>(out.data_ptr()),
        reinterpret_cast<const T*>(x_in.data_ptr()),
        reinterpret_cast<const T*>(conv_weights.data_ptr()),
        conv_bias.has_value() ? reinterpret_cast<const T*>(conv_bias->data_ptr()) : nullptr,
        reinterpret_cast<T*>(conv_states.data_ptr()),
        conv_states_stride_0,
        conv_w_stride,
        conv_d_stride,
        reinterpret_cast<T*>(conv_states_tmp.data_ptr()),
        reinterpret_cast<const int32_t*>(query_start_loc.data_ptr()),
        reinterpret_cast<const int*>(cache_indices.data_ptr()),
        has_initial_state.has_value() ? reinterpret_cast<const bool*>(has_initial_state->data_ptr()) : nullptr,
        act_mode,
        pad_slot_id,
        batch_size,
        num_virtual_tokens,
        dim);
    cgh.parallel_for(range_main, task);
  });

  using KERNEL_UPDATE = gdn::update_states_kernel<T>;
  auto range_update = KERNEL_UPDATE::get_nd_range(batch_size, Width, dim);
  queue.submit([&](sycl::handler& cgh) {
    KERNEL_UPDATE task(
        reinterpret_cast<T*>(conv_states.data_ptr()),
        conv_states_stride_0,
        conv_w_stride,
        conv_d_stride,
        reinterpret_cast<const T*>(conv_states_tmp.data_ptr()),
        reinterpret_cast<const int*>(cache_indices.data_ptr()),
        Width,
        dim,
        reinterpret_cast<const int32_t*>(query_start_loc.data_ptr()),
        nullptr,
        batch_size);
    cgh.parallel_for(range_update, task);
  });
}

// Decode variant: every batch entry contributes exactly one token,
// so the main kernel updates conv_states in-place and no follow-up
// update_states_kernel launch is needed (matching the original design's
// `if (num_prefills > 0)` gating).
template <typename T, int Width>
void causal_conv1d_only_decode(
    sycl::queue& queue,
    torch::Tensor& out,
    const torch::Tensor& x_in,
    const torch::Tensor& conv_weights,
    const std::optional<torch::Tensor>& conv_bias,
    torch::Tensor& conv_states,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& cache_indices,
    const std::optional<torch::Tensor>& has_initial_state,
    const gdn::ActMode& act_mode,
    const int pad_slot_id) {
  const int batch_size = query_start_loc.size(0) - 1;
  const int num_virtual_tokens = out.size(0);
  const int dim = conv_weights.size(0);
  const int conv_states_stride_0 = conv_states.stride(0);
  const int conv_w_stride = conv_states.stride(1);
  const int conv_d_stride = conv_states.stride(2);

  using KERNEL_MAIN = causal_conv1d_only_kernel<T, Width, 4>;
  auto range_main = KERNEL_MAIN::get_nd_range(num_virtual_tokens, dim);
  queue.submit([&](sycl::handler& cgh) {
    KERNEL_MAIN task(
        reinterpret_cast<T*>(out.data_ptr()),
        reinterpret_cast<const T*>(x_in.data_ptr()),
        reinterpret_cast<const T*>(conv_weights.data_ptr()),
        conv_bias.has_value() ? reinterpret_cast<const T*>(conv_bias->data_ptr()) : nullptr,
        reinterpret_cast<T*>(conv_states.data_ptr()),
        conv_states_stride_0,
        conv_w_stride,
        conv_d_stride,
        /*conv_states_tmp=*/nullptr,
        reinterpret_cast<const int32_t*>(query_start_loc.data_ptr()),
        reinterpret_cast<const int*>(cache_indices.data_ptr()),
        has_initial_state.has_value() ? reinterpret_cast<const bool*>(has_initial_state->data_ptr()) : nullptr,
        act_mode,
        pad_slot_id,
        batch_size,
        num_virtual_tokens,
        dim);
    cgh.parallel_for(range_main, task);
  });
}

// Decode fast-path variant: identical to causal_conv1d_only_decode but skips
// the O(batch_size) linear scan (SkipBatchScan=true), since decode's
// batch_id == token_id trivially. Added to test whether the scan is the
// cause of GDN-style's slowdown vs. general-purpose at large decode batch.
template <typename T, int Width>
void causal_conv1d_only_decode_fast(
    sycl::queue& queue,
    torch::Tensor& out,
    const torch::Tensor& x_in,
    const torch::Tensor& conv_weights,
    const std::optional<torch::Tensor>& conv_bias,
    torch::Tensor& conv_states,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& cache_indices,
    const std::optional<torch::Tensor>& has_initial_state,
    const gdn::ActMode& act_mode,
    const int pad_slot_id) {
  const int batch_size = query_start_loc.size(0) - 1;
  const int num_virtual_tokens = out.size(0);
  const int dim = conv_weights.size(0);
  const int conv_states_stride_0 = conv_states.stride(0);
  const int conv_w_stride = conv_states.stride(1);
  const int conv_d_stride = conv_states.stride(2);

  using KERNEL_MAIN = causal_conv1d_only_kernel<T, Width, 4, /*SkipBatchScan=*/true>;
  auto range_main = KERNEL_MAIN::get_nd_range(num_virtual_tokens, dim);
  queue.submit([&](sycl::handler& cgh) {
    KERNEL_MAIN task(
        reinterpret_cast<T*>(out.data_ptr()),
        reinterpret_cast<const T*>(x_in.data_ptr()),
        reinterpret_cast<const T*>(conv_weights.data_ptr()),
        conv_bias.has_value() ? reinterpret_cast<const T*>(conv_bias->data_ptr()) : nullptr,
        reinterpret_cast<T*>(conv_states.data_ptr()),
        conv_states_stride_0,
        conv_w_stride,
        conv_d_stride,
        /*conv_states_tmp=*/nullptr,
        reinterpret_cast<const int32_t*>(query_start_loc.data_ptr()),
        reinterpret_cast<const int*>(cache_indices.data_ptr()),
        has_initial_state.has_value() ? reinterpret_cast<const bool*>(has_initial_state->data_ptr()) : nullptr,
        act_mode,
        pad_slot_id,
        batch_size,
        num_virtual_tokens,
        dim);
    cgh.parallel_for(range_main, task);
  });
}

}  // namespace gdn_bench
