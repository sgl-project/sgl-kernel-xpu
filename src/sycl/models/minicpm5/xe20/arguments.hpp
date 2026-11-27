// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

namespace sgl_kernel::minicpm5::xe20 {
inline constexpr int MaxContext = 512 * 1024;

// A scalar projection and a per-output-channel projection share the same
// packed weight representation. The channel stride is zero for a scalar.
struct ProjectionScale {
  const float* values;
  int channel_stride;

  float operator[](int channel) const {
    return values[channel_stride * channel];
  }
};

template <typename Scalar>
struct WeightPointers {
  const Scalar *input_norm, *post_norm;
  const uint8_t *qkv, *out, *w13, *w2, *shared13, *shared2;
  ProjectionScale qkv_scale, out_scale, dense13_scale, dense2_scale;
  const float *w13_scale, *w2_scale;
  const Scalar* router;
  const float* bias;
  ProjectionScale shared13_scale, shared2_scale;
  bool expert_k_major = false;

  template <typename Other>
  WeightPointers<Other> as() const {
    return {
        reinterpret_cast<const Other*>(input_norm),
        reinterpret_cast<const Other*>(post_norm),
        qkv,
        out,
        w13,
        w2,
        shared13,
        shared2,
        qkv_scale,
        out_scale,
        dense13_scale,
        dense2_scale,
        w13_scale,
        w2_scale,
        reinterpret_cast<const Other*>(router),
        bias,
        shared13_scale,
        shared2_scale,
        expert_k_major};
  }
};

struct DraftProjection {
  const uint8_t* data;
  ProjectionScale scale;
};

struct DraftWeights {
  using Scalar = sycl::ext::oneapi::bfloat16;
  const Scalar *input_norm, *post_norm, *query_norm, *key_norm;
  DraftProjection qkv, output, gate_up, down;
};

template <typename Scalar>
struct WorkspacePointers {
  Scalar *normed, *qkv, *attention, *intermediate, *expert_out;
  float *attention_partial, *router, *routing, *gate_up_partial;
  int* experts;

  template <typename Other>
  WorkspacePointers<Other> as() const {
    return {
        reinterpret_cast<Other*>(normed),
        reinterpret_cast<Other*>(qkv),
        reinterpret_cast<Other*>(attention),
        reinterpret_cast<Other*>(intermediate),
        reinterpret_cast<Other*>(expert_out),
        attention_partial,
        router,
        routing,
        gate_up_partial,
        experts};
  }
};

enum class Part : int { Full, Attention, FeedForward, AttentionPartial, FeedForwardPartial };

template <typename Scalar>
struct StepPointers {
  const Scalar *hidden, *residual;
  Scalar *output, *res_out, *key_cache, *value_cache;
  const int64_t *positions, *slots;
  const int *blocks, *lengths;
  int block_size, block_table_stride;
  int64_t cache_blocks, block_table_cols;
  int64_t cache_block_stride, cache_head_stride, cache_token_stride;
  uint32_t* control;
  uint64_t *mailbox, *peer;
  uint32_t* health = nullptr;
  const Scalar* rope_cache = nullptr;
  int64_t rope_cache_rows = 0;
  Part part = Part::Full;
  float* ar = nullptr;

  template <typename Other>
  StepPointers<Other> as() const {
    return {
        reinterpret_cast<const Other*>(hidden),
        reinterpret_cast<const Other*>(residual),
        reinterpret_cast<Other*>(output),
        reinterpret_cast<Other*>(res_out),
        reinterpret_cast<Other*>(key_cache),
        reinterpret_cast<Other*>(value_cache),
        positions,
        slots,
        blocks,
        lengths,
        block_size,
        block_table_stride,
        cache_blocks,
        block_table_cols,
        cache_block_stride,
        cache_head_stride,
        cache_token_stride,
        control,
        mailbox,
        peer,
        health,
        reinterpret_cast<const Other*>(rope_cache),
        rope_cache_rows,
        part,
        ar};
  }
};

}  // namespace sgl_kernel::minicpm5::xe20
