/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cassert>
#include <cmath>
#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"
#include "comm/Numerics.h"
#include "cutlass/float8.h"

// TODO: Remove this when sycl float8 is supported
using cutlass::float_e4m3_t;

namespace at::native::xpu {

// local to avoid ODR issues while keeping the implementation consistent.
// Uses exp2(x * log2_base) instead of pow(base, x) because exp2 lowers to a
// single hardware instruction on Intel GPUs, whereas pow goes through a slow
// polynomial path.
static inline float
compute_freq_yarn_rope(float log2_base, int rotary_dim, int half_dim, float factor, float low, float high) {
  const float exponent = -2.0f * static_cast<float>(half_dim) / static_cast<float>(rotary_dim);
  float freq = sycl::exp2(exponent * log2_base);

  if (factor != 1.0f) {
    float inv_freq_extrapolation = freq;
    float inv_freq_interpolation = freq / factor;

    float high_adj = high;
    if (sycl::fabs(low - high_adj) <= 1e-6f) {
      high_adj += 0.001f;
    }

    float dim_value = 2.0f * static_cast<float>(half_dim);
    float linear_func = (dim_value - low) / (high_adj - low);
    float ramp_func = sycl::fmin(sycl::fmax(linear_func, 0.0f), 1.0f);

    freq = inv_freq_interpolation * (1.0f - ramp_func) + inv_freq_extrapolation * ramp_func;
  }

  return freq;
}

// Applies per-dimension weight scaling followed by RoPE to Q and K heads.
template <int head_dim, bool interleave, typename scalar_t>
struct FusedQKRopeKernel {
  scalar_t* qkv;
  int num_heads_q;
  int num_heads_k;
  int num_heads_v;
  const scalar_t* q_weight;
  const scalar_t* k_weight;
  float log2_base;
  const int* position_ids;
  int num_tokens;
  float factor;
  float low;
  float high;
  float attention_factor;
  int rotary_dim;

  [[sycl::reqd_sub_group_size(32)]]
  void operator()(sycl::nd_item<1> item) const {
    using accscalar_t = float;

    const int sg_size = item.get_sub_group().get_max_local_range()[0];
    assert(sg_size == 32);
    const int warpsPerBlock = item.get_local_range(0) / sg_size;
    const int warpId = item.get_local_id(0) / sg_size;
    const int laneId = item.get_local_id(0) % sg_size;

    const int globalWarpIdx = item.get_group(0) * warpsPerBlock + warpId;
    const int total_qk_heads = num_heads_q + num_heads_k;

    const int tokenIdx = globalWarpIdx / total_qk_heads;
    const int localHeadIdx = globalWarpIdx % total_qk_heads;

    if (tokenIdx >= num_tokens) return;

    const bool isQ = localHeadIdx < num_heads_q;
    const int headIdx = isQ ? localHeadIdx : localHeadIdx - num_heads_q;
    const int num_heads = num_heads_q + num_heads_k + num_heads_v;

    constexpr int numElemsPerThread = head_dim / 32;
    accscalar_t elements[numElemsPerThread];

    int offsetWarp;
    if (isQ) {
      offsetWarp = tokenIdx * num_heads * head_dim + headIdx * head_dim;
    } else {
      offsetWarp = tokenIdx * num_heads * head_dim + num_heads_q * head_dim + headIdx * head_dim;
    }
    int offsetThread = offsetWarp + laneId * numElemsPerThread;

    // Load elements and apply per-dimension weight scaling (no RMSNorm)
    const scalar_t* weight_ptr = isQ ? q_weight : k_weight;
#pragma unroll
    for (int i = 0; i < numElemsPerThread; i++) {
      // TODO: For FP8, per-tensor or per-channel dequant scales should be applied
      // here before the weight multiply (e.g. val *= dequant_scale).
      // Currently operating at unit scale; callers must pre-scale if needed.
      const int dim = laneId * numElemsPerThread + i;
      const accscalar_t val = static_cast<accscalar_t>(qkv[offsetThread + i]);
      elements[i] = val * static_cast<accscalar_t>(weight_ptr[dim]);
    }

    // Apply RoPE to weighted elements.
    //
    // Interleave mode pairs element (2k, 2k+1); both pair-members share the
    // same half_dim and therefore the same sin/cos. We exploit this by
    // computing sin/cos only at `numElemsPerThread/2` unique pair indices
    // instead of `numElemsPerThread` per-element values, cutting transcendental
    // work in half (exp2 + native::sin + native::cos + compute_freq branch).
    //
    // Neox mode uses a different dim_idx per element so each slot needs its
    // own sin/cos — no change there.
    accscalar_t elements2[numElemsPerThread];
    constexpr int numPairs = (numElemsPerThread + 1) / 2;
    accscalar_t cos_pair[numPairs];
    accscalar_t sin_pair[numPairs];
    accscalar_t cos_vals[numElemsPerThread];
    accscalar_t sin_vals[numElemsPerThread];
    float pos_id = static_cast<float>(position_ids[tokenIdx]);
    const int rotary_lanes = rotary_dim / numElemsPerThread;
    const bool applyRotary = (laneId < rotary_lanes);

    auto sg = item.get_sub_group();

    if (applyRotary) {
      if constexpr (interleave) {
        // Pair-swap sign pattern: e2[2k] = -e[2k+1], e2[2k+1] = e[2k]
#pragma unroll
        for (int i = 0; i < numElemsPerThread; i++) {
          elements2[i] = (i % 2 == 0) ? -elements[i + 1] : elements[i - 1];
        }
        // One transcendental per pair, shared by both pair members.
#pragma unroll
        for (int p = 0; p < numPairs; p++) {
          int dim_idx = laneId * numElemsPerThread + 2 * p;
          int half_dim = dim_idx / 2;
          float freq = compute_freq_yarn_rope(log2_base, rotary_dim, half_dim, factor, low, high);
          float theta = pos_id * freq;
          // native::{sin,cos} map to hardware transcendentals on Intel GPUs —
          // substantially faster than sycl::sin / sycl::cos.
          sin_pair[p] = sycl::native::sin(theta);
          cos_pair[p] = sycl::native::cos(theta);
        }
#pragma unroll
        for (int i = 0; i < numElemsPerThread; i++) {
          sin_vals[i] = sin_pair[i / 2];
          cos_vals[i] = cos_pair[i / 2];
        }
      }
    }

    if constexpr (!interleave) {
      // Neox style: each element maps to a distinct half_dim, so sin/cos are
      // computed per element (no pair sharing available here).
      sycl::group_barrier(sg);
      const int half_rotary_lanes = rotary_lanes / 2;

#pragma unroll
      for (int i = 0; i < numElemsPerThread; i++) {
        auto permuted = sycl::permute_group_by_xor(sg, elements[i], half_rotary_lanes);

        if (applyRotary) {
          elements2[i] = permuted;
          if (laneId < half_rotary_lanes) {
            elements2[i] = -elements2[i];
          }

          int dim_idx = laneId * numElemsPerThread + i;
          dim_idx = (dim_idx * 2) % rotary_dim;
          int half_dim = dim_idx / 2;
          float freq = compute_freq_yarn_rope(log2_base, rotary_dim, half_dim, factor, low, high);
          float theta = pos_id * freq;
          sin_vals[i] = sycl::native::sin(theta);
          cos_vals[i] = sycl::native::cos(theta);
        }
      }
      sycl::group_barrier(sg);
    }

    // Apply rotation with attention_factor
    if (applyRotary) {
#pragma unroll
      for (int i = 0; i < numElemsPerThread; i++) {
        elements[i] = (elements[i] * cos_vals[i] + elements2[i] * sin_vals[i]) * attention_factor;
      }
    }

    // Store results
#pragma unroll
    for (int i = 0; i < numElemsPerThread; i++) {
      qkv[offsetThread + i] = static_cast<scalar_t>(elements[i]);
    }
  }
};

template <int head_dim, bool interleave, typename scalar_t>
void launchFusedQKRopeImpl(
    void* qkv,
    int num_tokens,
    int num_heads_q,
    int num_heads_k,
    int num_heads_v,
    const void* q_weight,
    const void* k_weight,
    float base,
    const int* position_ids,
    float factor,
    float low,
    float high,
    float attention_factor,
    int rotary_dim,
    sycl::queue& q) {
  constexpr int blockSize = 256;
  const int warpsPerBlock = blockSize / 32;
  const int totalQKHeads = num_heads_q + num_heads_k;
  const int totalWarps = num_tokens * totalQKHeads;
  const int gridSize = CeilDiv(totalWarps, warpsPerBlock);

  const float log2_base = std::log2(base);

  FusedQKRopeKernel<head_dim, interleave, scalar_t> kernel{
      static_cast<scalar_t*>(qkv),
      num_heads_q,
      num_heads_k,
      num_heads_v,
      static_cast<const scalar_t*>(q_weight),
      static_cast<const scalar_t*>(k_weight),
      log2_base,
      position_ids,
      num_tokens,
      factor,
      low,
      high,
      attention_factor,
      rotary_dim};

  sycl_kernel_submit(sycl::range<1>(gridSize * blockSize), sycl::range<1>(blockSize), q, kernel);
}

void fused_qk_rope(
    torch::Tensor& qkv,
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t num_heads_v,
    int64_t head_dim,
    torch::Tensor& q_weight,
    torch::Tensor& k_weight,
    double base,
    bool is_neox,
    torch::Tensor& position_ids,
    double factor,
    double low,
    double high,
    double attention_factor,
    int64_t rotary_dim) {
  // Input validation
  TORCH_CHECK(qkv.dim() == 2, "QKV tensor must be 2D: [num_tokens, (num_heads_q+num_heads_k+num_heads_v)*head_dim]");
  TORCH_CHECK(position_ids.dim() == 1, "Position IDs must be 1D: [num_tokens]");
  TORCH_CHECK(q_weight.dim() == 1, "Query weights must be 1D: [head_dim]");
  TORCH_CHECK(k_weight.dim() == 1, "Key weights must be 1D: [head_dim]");
  TORCH_CHECK(q_weight.size(0) == head_dim, "Query weights size must match head dimension");
  TORCH_CHECK(k_weight.size(0) == head_dim, "Key weights size must match head dimension");
  TORCH_CHECK(
      q_weight.scalar_type() == qkv.scalar_type(),
      "Query weights dtype must match QKV dtype: expected ",
      qkv.scalar_type(),
      " but got ",
      q_weight.scalar_type());
  TORCH_CHECK(
      k_weight.scalar_type() == qkv.scalar_type(),
      "Key weights dtype must match QKV dtype: expected ",
      qkv.scalar_type(),
      " but got ",
      k_weight.scalar_type());
  TORCH_CHECK(
      position_ids.scalar_type() == at::kInt,
      "Position IDs dtype must be int32 (at::kInt) to match data_ptr<int>() usage; got ",
      position_ids.scalar_type());
  TORCH_CHECK(
      head_dim == 64 || head_dim == 128 || head_dim == 256, "head_dim must be one of {64, 128, 256}; got ", head_dim);
  TORCH_CHECK(
      rotary_dim > 0 && rotary_dim <= head_dim,
      "rotary_dim must be in the range (0, head_dim], got ",
      rotary_dim,
      " with head_dim ",
      head_dim);
  TORCH_CHECK(rotary_dim % 2 == 0, "rotary_dim must be even for RoPE, got ", rotary_dim);
  TORCH_CHECK(rotary_dim % (head_dim / 32) == 0, "rotary_dim must be divisible by numElemsPerThread");

  if (is_neox) {
    int64_t half_rotary_lanes = rotary_dim / (head_dim / 32) / 2;
    TORCH_CHECK(
        half_rotary_lanes >= 1 && half_rotary_lanes < 32 && (half_rotary_lanes & (half_rotary_lanes - 1)) == 0,
        "half_rotary_lanes must be a power of 2 and less than 32 for neox style, got ",
        half_rotary_lanes);
  }

  CHECK_INPUT(qkv);
  CHECK_INPUT(position_ids);
  CHECK_INPUT(q_weight);
  CHECK_INPUT(k_weight);

  TORCH_CHECK(
      position_ids.scalar_type() == at::kInt,
      "position_ids must be an int32 tensor, but got ",
      position_ids.scalar_type());

  int64_t num_tokens = qkv.size(0);
  TORCH_CHECK(position_ids.size(0) == num_tokens, "Number of tokens in position_ids must match QKV");

  int64_t total_heads = num_heads_q + num_heads_k + num_heads_v;
  TORCH_CHECK(
      qkv.size(1) == total_heads * head_dim, "QKV tensor size must match total number of heads and head dimension");

  auto queue = dpcppGetCurrentQueue();
  bool interleave = !is_neox;

#define FUSED_QK_ROPE_LAUNCH_ARGS                                                                                  \
  qkv.data_ptr(), static_cast<int>(num_tokens), static_cast<int>(num_heads_q), static_cast<int>(num_heads_k),      \
      static_cast<int>(num_heads_v), q_weight.data_ptr(), k_weight.data_ptr(), static_cast<float>(base),           \
      position_ids.data_ptr<int>(), static_cast<float>(factor), static_cast<float>(low), static_cast<float>(high), \
      static_cast<float>(attention_factor), static_cast<int>(rotary_dim), queue

#define LAUNCH_QK_ROPE_KERNEL(HEAD_DIM, INTERLEAVE)                                                                \
  AT_DISPATCH_SWITCH(                                                                                              \
      qkv.scalar_type(), "fused_qk_rope", DISPATCH_CASE_FLOAT_TYPES(([&] {                                         \
        launchFusedQKRopeImpl<HEAD_DIM, INTERLEAVE, scalar_t>(FUSED_QK_ROPE_LAUNCH_ARGS);                          \
      })) AT_DISPATCH_CASE(at::ScalarType::Float8_e4m3fn, ([&] {                                                   \
                             launchFusedQKRopeImpl<HEAD_DIM, INTERLEAVE, float_e4m3_t>(FUSED_QK_ROPE_LAUNCH_ARGS); \
                           })))

  switch (head_dim) {
    case 64:
      if (interleave) {
        LAUNCH_QK_ROPE_KERNEL(64, true);
      } else {
        LAUNCH_QK_ROPE_KERNEL(64, false);
      }
      break;
    case 128:
      if (interleave) {
        LAUNCH_QK_ROPE_KERNEL(128, true);
      } else {
        LAUNCH_QK_ROPE_KERNEL(128, false);
      }
      break;
    case 256:
      if (interleave) {
        LAUNCH_QK_ROPE_KERNEL(256, true);
      } else {
        LAUNCH_QK_ROPE_KERNEL(256, false);
      }
      break;
    default:
      TORCH_CHECK(false, "Unsupported head dimension for fused_qk_rope: ", head_dim);
  }

#undef LAUNCH_QK_ROPE_KERNEL
#undef FUSED_QK_ROPE_LAUNCH_ARGS
}

template <bool is_neox, int64_t rope_dim, typename scalar_t, typename pos_t>
struct FusedRopeCacheKernel {
  scalar_t* query;
  scalar_t* key;
  const scalar_t* cos_sin_cache;
  const pos_t* positions;
  int64_t q_stride;
  int64_t k_stride;
  int64_t q_head_stride;
  int64_t k_head_stride;
  int64_t num_tokens;
  int64_t num_q_heads;
  int64_t num_k_heads;
  int64_t cache_stride;
  int64_t workers_per_work_group;
  int64_t total_workers;

  [[sycl::reqd_sub_group_size(16)]]
  void operator()(sycl::nd_item<1> item) const {
    constexpr int64_t sg_size = 16;
    assert(sg_size == 16);
    const int64_t local_id = item.get_local_id(0);
    const int64_t worker_in_work_group = local_id / sg_size;
    const int64_t lane_id = local_id % sg_size;
    const int64_t worker_id = item.get_group(0) * workers_per_work_group + worker_in_work_group;

    const int64_t num_qk_heads = num_q_heads + num_k_heads;
    const int64_t num_works = num_tokens * num_qk_heads;
    constexpr int64_t half_rope = rope_dim / 2;

    using storage_t = std::conditional_t<
        std::is_same_v<scalar_t, c10::Half>,
        sycl::half,
        sycl::ext::oneapi::bfloat16>;  // fallback is bf16 since only 2 types supported

    constexpr int64_t max_vec_bytes = 16;  // 128 bits
    constexpr int64_t max_vec_elems = max_vec_bytes / (int64_t)sizeof(storage_t);

    // We use different heuristic vec_size choosing from cuda.
    // The following config would get the best performance:
    // head_dim: 64 128 256 512
    // vec_size: 2    2   8   8

    constexpr int64_t vec_size =
        is_neox ? (rope_dim < 256 ? 2 : max_vec_elems) : (rope_dim < 256 ? 1 : max_vec_elems / 2);

    using vec_t = sycl::vec<storage_t, vec_size>;
    using vec2_t = sycl::vec<storage_t, vec_size * 2>;  // for interleave pairs

    for (int64_t idx = worker_id; idx < num_works; idx += total_workers) {
      const int64_t token_id = idx / num_qk_heads;
      const int64_t head_id = idx % num_qk_heads;
      const bool is_q_head = head_id < num_q_heads;

      const int64_t pos = static_cast<int64_t>(positions[token_id]);
      scalar_t* base_ptr = nullptr;
      int64_t head_index = 0;

      if (is_q_head) {
        head_index = head_id;
        base_ptr = query + token_id * q_stride + head_index * q_head_stride;
      } else {
        head_index = head_id - num_q_heads;
        base_ptr = key + token_id * k_stride + head_index * k_head_stride;
      }

      const scalar_t* cos_ptr = cos_sin_cache + pos * cache_stride;
      const scalar_t* sin_ptr = cos_ptr + half_rope;

      // Reinterpret scalar_t* as storage_t* for vectorized load/store.
      // Safe: c10::Half and sycl::half share identical IEEE binary16 layout;
      //       same holds for c10::BFloat16 and sycl::ext::oneapi::bfloat16.
      auto* base_sptr = reinterpret_cast<storage_t*>(base_ptr);
      auto* cos_sptr = reinterpret_cast<const storage_t*>(cos_ptr);
      auto* sin_sptr = reinterpret_cast<const storage_t*>(sin_ptr);

      if constexpr (is_neox) {
        auto* x_sptr = base_sptr;
        auto* y_sptr = base_sptr + half_rope;

#pragma unroll
        for (int64_t i = lane_id * vec_size; i < half_rope; i += sg_size * vec_size) {
          vec_t x_vec = *reinterpret_cast<const vec_t*>(x_sptr + i);
          vec_t y_vec = *reinterpret_cast<const vec_t*>(y_sptr + i);
          vec_t c_vec = *reinterpret_cast<const vec_t*>(cos_sptr + i);
          vec_t s_vec = *reinterpret_cast<const vec_t*>(sin_sptr + i);
          vec_t out_x, out_y;

#pragma unroll
          for (int j = 0; j < vec_size; j++) {
            const storage_t x = x_vec[j];
            const storage_t y = y_vec[j];
            const storage_t c = c_vec[j];
            const storage_t s = s_vec[j];
            out_x[j] = x * c - y * s;
            out_y[j] = x * s + y * c;
          }
          *reinterpret_cast<vec_t*>(x_sptr + i) = out_x;
          *reinterpret_cast<vec_t*>(y_sptr + i) = out_y;
        }
      } else {
#pragma unroll
        for (int64_t i = lane_id * vec_size; i < half_rope; i += sg_size * vec_size) {
          vec2_t v = *reinterpret_cast<const vec2_t*>(base_sptr + 2 * i);
          vec_t c_vec = *reinterpret_cast<const vec_t*>(cos_sptr + i);
          vec_t s_vec = *reinterpret_cast<const vec_t*>(sin_sptr + i);

#pragma unroll
          for (int j = 0; j < vec_size; j++) {
            const storage_t x = v[2 * j];
            const storage_t y = v[2 * j + 1];
            const storage_t c = c_vec[j];
            const storage_t s = s_vec[j];
            v[2 * j] = x * c - y * s;
            v[2 * j + 1] = x * s + y * c;
          }
          *reinterpret_cast<vec2_t*>(base_sptr + 2 * i) = v;
        }
      }
    }
  }
};

template <bool is_neox, int64_t rope_dim, typename scalar_t, typename pos_t>
void launch_fused_rope_cache_kernel_scalar(
    at::Tensor& query, at::Tensor& key, const at::Tensor& cos_sin_cache, const at::Tensor& positions) {
  constexpr int kWorkGroupSize = 128;
  constexpr int sg_size = 16;
  static_assert(kWorkGroupSize % sg_size == 0, "kWorkGroupSize must be divisible by subgroup size");

  const int64_t workers_per_work_group = kWorkGroupSize / sg_size;
  const int64_t num_tokens = query.size(0);
  const int64_t num_q_heads = query.size(1);
  const int64_t num_k_heads = key.size(1);
  const int64_t num_works = num_tokens * (num_q_heads + num_k_heads);
  const int64_t num_groups = CeilDiv(num_works, workers_per_work_group);
  const int64_t total_workers = num_groups * workers_per_work_group;

  FusedRopeCacheKernel<is_neox, rope_dim, scalar_t, pos_t> kernel{
      query.data_ptr<scalar_t>(),
      key.data_ptr<scalar_t>(),
      cos_sin_cache.data_ptr<scalar_t>(),
      positions.data_ptr<pos_t>(),
      query.stride(0),
      key.stride(0),
      query.stride(1),
      key.stride(1),
      num_tokens,
      num_q_heads,
      num_k_heads,
      cos_sin_cache.stride(0),
      workers_per_work_group,
      total_workers};

  sycl_kernel_submit(
      sycl::range<1>(num_groups * kWorkGroupSize), sycl::range<1>(kWorkGroupSize), dpcppGetCurrentQueue(), kernel);
}

void fused_qk_rope_with_cos_sin_cache_inplace(
    at::Tensor& query,
    at::Tensor& key,
    at::Tensor& cos_sin_cache,
    at::Tensor& positions,
    int64_t rope_dim,
    bool is_neox) {
  const auto input_dim = query.dim();
  // Note that query and key were truncated to rope_dim here if head_dim is larger.
  TORCH_CHECK(
      input_dim == 3,
      "fused_qk_rope_with_cos_sin_cache_inplace only supports 3D input [num_tokens, num_heads, rope_dim]");

#define LAUNCH_ROPE_CACHE_KERNEL(IS_NEOX, POS_T)                                                                  \
  switch (rope_dim) {                                                                                             \
    case 64:                                                                                                      \
      launch_fused_rope_cache_kernel_scalar<IS_NEOX, 64, scalar_t, POS_T>(query, key, cos_sin_cache, positions);  \
      break;                                                                                                      \
    case 128:                                                                                                     \
      launch_fused_rope_cache_kernel_scalar<IS_NEOX, 128, scalar_t, POS_T>(query, key, cos_sin_cache, positions); \
      break;                                                                                                      \
    case 256:                                                                                                     \
      launch_fused_rope_cache_kernel_scalar<IS_NEOX, 256, scalar_t, POS_T>(query, key, cos_sin_cache, positions); \
      break;                                                                                                      \
    case 512:                                                                                                     \
      launch_fused_rope_cache_kernel_scalar<IS_NEOX, 512, scalar_t, POS_T>(query, key, cos_sin_cache, positions); \
      break;                                                                                                      \
    default:                                                                                                      \
      TORCH_CHECK(false, "Unsupported rope_dim: ", rope_dim);                                                     \
  }

#define DISPATCH_ROPE_CACHE_KERNEL_BY_LAYOUT(POS_T) \
  if (is_neox) {                                    \
    LAUNCH_ROPE_CACHE_KERNEL(true, POS_T);          \
  } else {                                          \
    LAUNCH_ROPE_CACHE_KERNEL(false, POS_T);         \
  }

  SYCL_DISPATCH_FLOATING_TYPES(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query.scalar_type(),
      "fused_qk_rope_with_cos_sin_cache_inplace",
      [&]() {
        if (positions.scalar_type() == at::kInt) {
          DISPATCH_ROPE_CACHE_KERNEL_BY_LAYOUT(int32_t);
        } else {
          DISPATCH_ROPE_CACHE_KERNEL_BY_LAYOUT(int64_t);
        }
      });

#undef DISPATCH_ROPE_CACHE_KERNEL_BY_LAYOUT
#undef LAUNCH_ROPE_CACHE_KERNEL
}

}  // namespace at::native::xpu
