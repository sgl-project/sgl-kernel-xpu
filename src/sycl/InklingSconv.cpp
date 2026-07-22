/* Copyright 2025 SGLang Team. All Rights Reserved.
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *
 * This file adapts the Inkling BMG short-convolution kernels from
 * /data2/syk/cutlass-sycl/examples/14_bmg_sconv for the sgl-kernel XPU
 * extension ABI.
 */

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstdint>
#include <optional>
#include <sycl/sycl.hpp>
#include <tuple>

#include "Utils.h"

namespace {

constexpr int kPadSlotId = -1;
constexpr int kThreads = 256;
constexpr int kMetaThreads = 128;
constexpr int kForwardBlockT = 4;
constexpr int kUpdateCopyBytes = 16;
constexpr int kUpdateCopyWords = kUpdateCopyBytes / static_cast<int>(sizeof(uint32_t));
constexpr int kFusedW4FastVec = 4;
constexpr int kFusedW4WideVec = 8;
constexpr int kPackedCopyMaxWindow = 16;
constexpr int kWindowPacksPerLane = 1;

template <typename scalar_t>
inline float to_float_device(scalar_t value) {
  return static_cast<float>(value);
}

template <typename scalar_t>
inline scalar_t from_float_device(float value) {
  return static_cast<scalar_t>(value);
}

template <typename scalar_t>
inline scalar_t from_raw16_device(uint16_t value) {
  static_assert(sizeof(scalar_t) == sizeof(uint16_t));
  return sycl::bit_cast<scalar_t>(value);
}

template <typename scalar_t>
inline uint16_t to_raw16_device(scalar_t value) {
  static_assert(sizeof(scalar_t) == sizeof(uint16_t));
  return sycl::bit_cast<uint16_t>(value);
}

template <typename scalar_t>
inline uint64_t load_pack4_device(scalar_t const* ptr) {
  return *reinterpret_cast<uint64_t const*>(ptr);
}

template <typename scalar_t>
inline void store_pack4_device(scalar_t* ptr, uint64_t value) {
  *reinterpret_cast<uint64_t*>(ptr) = value;
}

template <typename scalar_t>
inline scalar_t element_from_pack4_device(uint64_t raw, int lane) {
  return from_raw16_device<scalar_t>(static_cast<uint16_t>(raw >> (16 * lane)));
}

template <typename scalar_t, int Vec>
inline uint64_t pack4_from_floats_device(float const (&values)[Vec], int base) {
  uint64_t raw = 0;
#pragma unroll
  for (int v = 0; v < kFusedW4FastVec; ++v) {
    raw |= static_cast<uint64_t>(to_raw16_device(from_float_device<scalar_t>(values[base + v]))) << (16 * v);
  }
  return raw;
}

inline int64_t div_up_i64(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

template <typename scalar_t>
struct SconvForwardParams {
  scalar_t const* x;
  scalar_t const* cache;
  bool const* cache_mask;
  int64_t const* safe_idx;
  int64_t const* cu;
  int32_t const* si;
  scalar_t const* weight;
  scalar_t* y;

  int64_t T;
  int64_t D;
  int64_t W;
  int64_t x_stride_t;
  int64_t x_stride_d;
  int64_t cache_stride_slot;
  int64_t cache_stride_w;
  int64_t cache_stride_d;
  int64_t cache_mask_stride_seq;
  int64_t weight_stride_d;
  int64_t weight_stride_w;
  int64_t y_stride_t;
  int64_t y_stride_d;
  bool use_silu;
  bool use_residual;
  bool is_decode;
  bool weight_current_first;
};

template <typename scalar_t>
struct SconvForwardKernel {
  SconvForwardParams<scalar_t> p;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t linear = static_cast<int64_t>(item.get_global_linear_id());
    const int64_t total = p.T * p.D;
    if (linear >= total) {
      return;
    }

    const int64_t t = linear / p.D;
    const int64_t d = linear - t * p.D;
    const int32_t seq = p.si[t];
    const int64_t bos = p.cu[seq];
    const int64_t slot = p.safe_idx[seq];
    const float mask = (p.is_decode || p.cache_mask[seq * p.cache_mask_stride_seq]) ? 1.0f : 0.0f;

    float acc = 0.0f;
    for (int64_t iw = 0; iw < p.W; ++iw) {
      const int64_t shifted = t - iw;
      float tap = 0.0f;
      if (shifted >= bos && shifted < p.T) {
        tap = to_float_device(p.x[shifted * p.x_stride_t + d * p.x_stride_d]);
      } else {
        const int64_t prefix_pos = shifted - bos + (p.W - 1);
        if (shifted < bos && prefix_pos >= 0 && prefix_pos < p.W - 1) {
          tap = mask *
              to_float_device(
                    p.cache
                        [slot * p.cache_stride_slot + prefix_pos * p.cache_stride_w + d * p.cache_stride_d]);
        }
      }
      const int64_t weight_iw = p.weight_current_first ? iw : p.W - 1 - iw;
      const float w = to_float_device(p.weight[d * p.weight_stride_d + weight_iw * p.weight_stride_w]);
      acc += tap * w;
    }

    if (p.use_silu) {
      acc = acc / (1.0f + sycl::native::exp(-acc));
    }
    if (p.use_residual) {
      acc += to_float_device(p.x[t * p.x_stride_t + d * p.x_stride_d]);
    }
    p.y[t * p.y_stride_t + d * p.y_stride_d] = from_float_device<scalar_t>(acc);
  }
};

template <typename scalar_t>
void launch_sconv_forward(sycl::queue& q, SconvForwardParams<scalar_t> const& params) {
  const int64_t total = params.T * params.D;
  if (total == 0) {
    return;
  }
  const int64_t global = div_up_i64(total, kThreads) * kThreads;
  SconvForwardKernel<scalar_t> kernel{params};
  q.parallel_for<SconvForwardKernel<scalar_t>>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kThreads)), kernel);
}

template <typename scalar_t, int W, bool UseSilu, bool UseResidual, bool IsDecode>
struct SconvForwardBlockKernel {
  SconvForwardParams<scalar_t> p;

  void operator()(sycl::nd_item<2> item) const {
    const int64_t d = static_cast<int64_t>(item.get_global_id(0));
    const int64_t tb = static_cast<int64_t>(item.get_global_id(1));
    if (d >= p.D) {
      return;
    }

    float weights[W];
#pragma unroll
    for (int iw = 0; iw < W; ++iw) {
      const int64_t weight_iw = p.weight_current_first ? iw : W - 1 - iw;
      weights[iw] = to_float_device(p.weight[d * p.weight_stride_d + weight_iw * p.weight_stride_w]);
    }

    const int64_t t0 = tb * kForwardBlockT;
#pragma unroll
    for (int j = 0; j < kForwardBlockT; ++j) {
      const int64_t t = t0 + j;
      if (t >= p.T) {
        return;
      }

      const int32_t seq = p.si[t];
      const int64_t bos = p.cu[seq];
      const int64_t slot = p.safe_idx[seq];
      const float mask = (IsDecode || p.cache_mask[seq * p.cache_mask_stride_seq]) ? 1.0f : 0.0f;

      float acc = 0.0f;
#pragma unroll
      for (int iw = 0; iw < W; ++iw) {
        const int64_t shifted = t - iw;
        float tap = 0.0f;
        if (shifted >= bos && shifted < p.T) {
          tap = to_float_device(p.x[shifted * p.x_stride_t + d * p.x_stride_d]);
        } else {
          const int64_t prefix_pos = shifted - bos + (W - 1);
          if (shifted < bos && prefix_pos >= 0 && prefix_pos < W - 1) {
            tap = mask *
                to_float_device(
                      p.cache
                          [slot * p.cache_stride_slot + prefix_pos * p.cache_stride_w + d * p.cache_stride_d]);
          }
        }
        acc += tap * weights[iw];
      }

      if constexpr (UseSilu) {
        acc = acc / (1.0f + sycl::native::exp(-acc));
      }
      if constexpr (UseResidual) {
        acc += to_float_device(p.x[t * p.x_stride_t + d * p.x_stride_d]);
      }
      p.y[t * p.y_stride_t + d * p.y_stride_d] = from_float_device<scalar_t>(acc);
    }
  }
};

template <typename scalar_t, int W, bool UseSilu, bool UseResidual, bool IsDecode>
void launch_sconv_forward_block(sycl::queue& q, SconvForwardParams<scalar_t> const& params) {
  if (params.T == 0 || params.D == 0) {
    return;
  }
  const int64_t channel_global = div_up_i64(params.D, kThreads) * kThreads;
  const int64_t token_blocks = div_up_i64(params.T, kForwardBlockT);
  SconvForwardBlockKernel<scalar_t, W, UseSilu, UseResidual, IsDecode> kernel{params};
  q.parallel_for<SconvForwardBlockKernel<scalar_t, W, UseSilu, UseResidual, IsDecode>>(
      sycl::nd_range<2>(
          sycl::range<2>(channel_global, token_blocks),
          sycl::range<2>(kThreads, 1)),
      kernel);
}

template <typename scalar_t, int W, bool UseSilu, bool UseResidual>
void launch_sconv_forward_block_decode_selected(
    sycl::queue& q,
    SconvForwardParams<scalar_t> const& params) {
  if (params.is_decode) {
    launch_sconv_forward_block<scalar_t, W, UseSilu, UseResidual, true>(q, params);
  } else {
    launch_sconv_forward_block<scalar_t, W, UseSilu, UseResidual, false>(q, params);
  }
}

template <typename scalar_t, int W, bool UseSilu>
void launch_sconv_forward_block_residual_selected(
    sycl::queue& q,
    SconvForwardParams<scalar_t> const& params) {
  if (params.use_residual) {
    launch_sconv_forward_block_decode_selected<scalar_t, W, UseSilu, true>(q, params);
  } else {
    launch_sconv_forward_block_decode_selected<scalar_t, W, UseSilu, false>(q, params);
  }
}

template <typename scalar_t, int W>
void launch_sconv_forward_block_activation_selected(
    sycl::queue& q,
    SconvForwardParams<scalar_t> const& params) {
  if (params.use_silu) {
    launch_sconv_forward_block_residual_selected<scalar_t, W, true>(q, params);
  } else {
    launch_sconv_forward_block_residual_selected<scalar_t, W, false>(q, params);
  }
}

template <typename scalar_t>
bool try_launch_sconv_forward_block(sycl::queue& q, SconvForwardParams<scalar_t> const& params) {
  if (params.W == 3) {
    launch_sconv_forward_block_activation_selected<scalar_t, 3>(q, params);
    return true;
  }
  if (params.W == 4) {
    launch_sconv_forward_block_activation_selected<scalar_t, 4>(q, params);
    return true;
  }
  return false;
}

template <typename scalar_t>
struct UpdateSconvCacheParams {
  scalar_t const* x;
  scalar_t* cache;
  int32_t const* cache_indices;
  bool const* has_initial_state;
  int32_t const* query_start_loc;

  int64_t B;
  int64_t D;
  int64_t W1;
  int64_t x_stride_t;
  int64_t x_stride_d;
  int64_t cache_stride_slot;
  int64_t cache_stride_w;
  int64_t cache_stride_d;
};

template <typename scalar_t>
struct UpdateSconvCacheKernel {
  UpdateSconvCacheParams<scalar_t> p;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t linear = static_cast<int64_t>(item.get_global_linear_id());
    const int64_t total = p.B * p.D;
    if (linear >= total) {
      return;
    }

    const int64_t b = linear / p.D;
    const int64_t d = linear - b * p.D;
    const int32_t slot = p.cache_indices[b];
    const int32_t start = p.query_start_loc[b];
    const int32_t end = p.query_start_loc[b + 1];
    const int32_t qlen = end - start;
    if (slot == kPadSlotId || qlen <= 0) {
      return;
    }

    const bool has_state = p.has_initial_state[b];
    const int64_t cache_base = static_cast<int64_t>(slot) * p.cache_stride_slot + d * p.cache_stride_d;
    for (int64_t w = 0; w < p.W1; ++w) {
      scalar_t value = from_float_device<scalar_t>(0.0f);
      if (qlen >= p.W1 - w) {
        const int64_t x_idx = static_cast<int64_t>(end) - p.W1 + w;
        value = p.x[x_idx * p.x_stride_t + d * p.x_stride_d];
      } else if (has_state) {
        value = p.cache[cache_base + (w + qlen) * p.cache_stride_w];
      }
      p.cache[cache_base + w * p.cache_stride_w] = value;
    }
  }
};

template <typename scalar_t>
void launch_update_sconv_cache_scalar(sycl::queue& q, UpdateSconvCacheParams<scalar_t> const& params) {
  const int64_t total = params.B * params.D;
  if (total == 0) {
    return;
  }
  const int64_t global = div_up_i64(total, kThreads) * kThreads;
  UpdateSconvCacheKernel<scalar_t> kernel{params};
  q.parallel_for<UpdateSconvCacheKernel<scalar_t>>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kThreads)), kernel);
}

template <typename scalar_t, int StaticW1>
struct UpdateSconvCachePackedKernel {
  UpdateSconvCacheParams<scalar_t> p;
  int64_t lanes_per_batch;
  int64_t vec_count;
  int64_t pack_elems;

  void copy_zero_pack(scalar_t* row_base, int64_t pack_idx) const {
    using pack_t = sycl::vec<uint32_t, kUpdateCopyWords>;
    pack_t zero(0u);
    zero.store(static_cast<std::size_t>(pack_idx), reinterpret_cast<uint32_t*>(row_base));
  }

  void copy_pack(scalar_t const* src_row_base, scalar_t* dst_row_base, int64_t pack_idx) const {
    using pack_t = sycl::vec<uint32_t, kUpdateCopyWords>;
    pack_t value;
    value.load(static_cast<std::size_t>(pack_idx), reinterpret_cast<uint32_t const*>(src_row_base));
    value.store(static_cast<std::size_t>(pack_idx), reinterpret_cast<uint32_t*>(dst_row_base));
  }

  void operator()(sycl::nd_item<1> item) const {
    constexpr bool kStaticWidth = StaticW1 > 0;
    const int64_t width_minus_one = kStaticWidth ? StaticW1 : p.W1;
    const int64_t linear = static_cast<int64_t>(item.get_global_linear_id());
    const int64_t total = p.B * lanes_per_batch;
    if (linear >= total) {
      return;
    }

    const int64_t b = linear / lanes_per_batch;
    const int64_t lane = linear - b * lanes_per_batch;
    const int32_t slot = p.cache_indices[b];
    const int32_t start = p.query_start_loc[b];
    const int32_t end = p.query_start_loc[b + 1];
    const int32_t qlen = end - start;
    if (slot == kPadSlotId || qlen <= 0) {
      return;
    }

    const bool is_vec_lane = lane < vec_count;
    const int64_t channel =
        is_vec_lane ? lane * pack_elems : vec_count * pack_elems + (lane - vec_count);
    const int64_t cache_base = static_cast<int64_t>(slot) * p.cache_stride_slot;
    const bool has_state = p.has_initial_state[b];

    for (int64_t w = 0; w < width_minus_one; ++w) {
      scalar_t* dst_row = p.cache + cache_base + w * p.cache_stride_w;
      if (qlen >= width_minus_one - w) {
        const int64_t x_idx = static_cast<int64_t>(end) - width_minus_one + w;
        scalar_t const* src_row = p.x + x_idx * p.x_stride_t;
        if (is_vec_lane) {
          copy_pack(src_row, dst_row, lane);
        } else {
          dst_row[channel] = src_row[channel];
        }
      } else if (has_state) {
        scalar_t const* src_row = p.cache + cache_base + (w + qlen) * p.cache_stride_w;
        if (is_vec_lane) {
          copy_pack(src_row, dst_row, lane);
        } else {
          dst_row[channel] = src_row[channel];
        }
      } else {
        if (is_vec_lane) {
          copy_zero_pack(dst_row, lane);
        } else {
          dst_row[channel] = from_float_device<scalar_t>(0.0f);
        }
      }
    }
  }
};

template <typename scalar_t, int StaticW1>
bool launch_update_sconv_cache_packed_static(
    sycl::queue& q,
    UpdateSconvCacheParams<scalar_t> const& params) {
  if (params.B == 0 || params.D == 0 || params.W1 == 0) {
    return true;
  }
  if (params.x_stride_d != 1 || params.cache_stride_d != 1) {
    return false;
  }

  const int64_t pack_elems = kUpdateCopyBytes / static_cast<int64_t>(sizeof(scalar_t));
  const bool aligned =
      params.x_stride_t % pack_elems == 0 &&
      params.cache_stride_slot % pack_elems == 0 &&
      params.cache_stride_w % pack_elems == 0 &&
      reinterpret_cast<std::uintptr_t>(params.x) % kUpdateCopyBytes == 0 &&
      reinterpret_cast<std::uintptr_t>(params.cache) % kUpdateCopyBytes == 0;
  const int64_t vec_count = aligned ? params.D / pack_elems : 0;
  const int64_t scalar_tail = params.D - vec_count * pack_elems;
  const int64_t lanes_per_batch = vec_count + scalar_tail;
  if (lanes_per_batch == 0) {
    return true;
  }

  const int64_t total_lanes = params.B * lanes_per_batch;
  const int64_t global = div_up_i64(total_lanes, kThreads) * kThreads;
  UpdateSconvCachePackedKernel<scalar_t, StaticW1> kernel{
      params, lanes_per_batch, vec_count, pack_elems};
  q.parallel_for<UpdateSconvCachePackedKernel<scalar_t, StaticW1>>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kThreads)), kernel);
  return true;
}

template <typename scalar_t>
bool launch_update_sconv_cache_packed(sycl::queue& q, UpdateSconvCacheParams<scalar_t> const& params) {
  switch (params.W1) {
    case 1:
      return launch_update_sconv_cache_packed_static<scalar_t, 1>(q, params);
    case 2:
      return launch_update_sconv_cache_packed_static<scalar_t, 2>(q, params);
    case 3:
      return launch_update_sconv_cache_packed_static<scalar_t, 3>(q, params);
    case 5:
      return launch_update_sconv_cache_packed_static<scalar_t, 5>(q, params);
    case 7:
      return launch_update_sconv_cache_packed_static<scalar_t, 7>(q, params);
    case 8:
      return launch_update_sconv_cache_packed_static<scalar_t, 8>(q, params);
    default:
      return launch_update_sconv_cache_packed_static<scalar_t, 0>(q, params);
  }
}

template <typename scalar_t>
void launch_update_sconv_cache(sycl::queue& q, UpdateSconvCacheParams<scalar_t> const& params) {
  if (!launch_update_sconv_cache_packed<scalar_t>(q, params)) {
    launch_update_sconv_cache_scalar<scalar_t>(q, params);
  }
}

template <typename scalar_t>
struct FusedDecodeUpdateParams {
  scalar_t const* x;
  scalar_t* cache;
  int32_t const* cache_indices;
  bool const* cache_mask;
  scalar_t const* weight;
  scalar_t* y;
  bool const* track_mask;
  int64_t const* track_indices;

  int64_t T;
  int64_t D;
  int64_t W;
  int64_t x_stride_t;
  int64_t x_stride_d;
  int64_t cache_stride_slot;
  int64_t cache_stride_w;
  int64_t cache_stride_d;
  int64_t weight_stride_d;
  int64_t weight_stride_w;
  int64_t y_stride_t;
  int64_t y_stride_d;
  int64_t track_idx_stride;
  bool use_silu;
  bool use_residual;
  bool do_track;
  bool weight_current_first;
};

template <typename scalar_t>
struct FusedDecodeUpdateKernel {
  FusedDecodeUpdateParams<scalar_t> p;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t linear = static_cast<int64_t>(item.get_global_linear_id());
    const int64_t total = p.T * p.D;
    if (linear >= total) {
      return;
    }

    const int64_t t = linear / p.D;
    const int64_t d = linear - t * p.D;
    const int32_t ci = p.cache_indices[t];
    const bool valid = ci != kPadSlotId;
    const int64_t slot = valid ? static_cast<int64_t>(ci) : 0;
    const float mask = p.cache_mask[t] ? 1.0f : 0.0f;
    const int64_t cache_base = slot * p.cache_stride_slot + d * p.cache_stride_d;

    const scalar_t x_value = p.x[t * p.x_stride_t + d * p.x_stride_d];
    const float x_float = to_float_device(x_value);
    const int64_t current_weight_iw = p.weight_current_first ? 0 : p.W - 1;
    float acc = x_float * to_float_device(p.weight[d * p.weight_stride_d + current_weight_iw * p.weight_stride_w]);
    for (int64_t iw = 1; iw < p.W; ++iw) {
      const int64_t cache_pos = p.W - 1 - iw;
      const float h = mask * to_float_device(p.cache[cache_base + cache_pos * p.cache_stride_w]);
      const int64_t weight_iw = p.weight_current_first ? iw : p.W - 1 - iw;
      const float w = to_float_device(p.weight[d * p.weight_stride_d + weight_iw * p.weight_stride_w]);
      acc += h * w;
    }

    if (p.use_silu) {
      acc = acc / (1.0f + sycl::native::exp(-acc));
    }
    if (p.use_residual) {
      acc += x_float;
    }
    p.y[t * p.y_stride_t + d * p.y_stride_d] = from_float_device<scalar_t>(acc);

    if (!valid) {
      return;
    }

    bool do_track = false;
    int64_t track_base = 0;
    if (p.do_track && p.track_mask[t]) {
      do_track = true;
      track_base = p.track_indices[t * p.track_idx_stride] * p.cache_stride_slot + d * p.cache_stride_d;
    }
    for (int64_t iw = 0; iw < p.W - 1; ++iw) {
      scalar_t next =
          (iw < p.W - 2)
              ? (mask != 0.0f ? p.cache[cache_base + (iw + 1) * p.cache_stride_w]
                              : from_float_device<scalar_t>(0.0f))
              : x_value;
      p.cache[cache_base + iw * p.cache_stride_w] = next;
      if (do_track) {
        p.cache[track_base + iw * p.cache_stride_w] = next;
      }
    }
  }
};

template <typename scalar_t>
void launch_fused_decode_update(sycl::queue& q, FusedDecodeUpdateParams<scalar_t> const& params) {
  const int64_t total = params.T * params.D;
  if (total == 0) {
    return;
  }
  const int64_t global = div_up_i64(total, kThreads) * kThreads;
  FusedDecodeUpdateKernel<scalar_t> kernel{params};
  q.parallel_for<FusedDecodeUpdateKernel<scalar_t>>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kThreads)), kernel);
}

template <typename scalar_t, int Vec, bool UseSilu, bool UseResidual, bool DoTrack>
struct FusedDecodeUpdateW4PackedKernel {
  FusedDecodeUpdateParams<scalar_t> p;

  void operator()(sycl::nd_item<1> item) const {
    static_assert(Vec == kFusedW4FastVec || Vec == kFusedW4WideVec);
    constexpr int kPackCount = Vec / kFusedW4FastVec;
    const int64_t channel_blocks = p.D / Vec;
    const int64_t linear = static_cast<int64_t>(item.get_global_linear_id());
    const int64_t total = p.T * channel_blocks;
    if (linear >= total) {
      return;
    }

    const int64_t t = linear / channel_blocks;
    const int64_t d0 = (linear - t * channel_blocks) * Vec;
    const int32_t ci = p.cache_indices[t];
    const bool valid = ci != kPadSlotId;
    const int64_t slot = valid ? static_cast<int64_t>(ci) : 0;
    const bool mask = p.cache_mask[t];
    const int64_t cache_base = slot * p.cache_stride_slot + d0;

    uint64_t x_pack[kPackCount];
    float acc[Vec];
    scalar_t x_value[Vec];
#pragma unroll
    for (int pack = 0; pack < kPackCount; ++pack) {
      x_pack[pack] = load_pack4_device(p.x + t * p.x_stride_t + d0 + pack * kFusedW4FastVec);
    }
#pragma unroll
    for (int v = 0; v < Vec; ++v) {
      x_value[v] = element_from_pack4_device<scalar_t>(
          x_pack[v / kFusedW4FastVec],
          v % kFusedW4FastVec);
      acc[v] = 0.0f;
    }

    uint64_t weight_pack[Vec];
#pragma unroll
    for (int v = 0; v < Vec; ++v) {
      weight_pack[v] = load_pack4_device(p.weight + (d0 + v) * p.weight_stride_d);
    }

#pragma unroll
    for (int cache_pos = 0; cache_pos < 3; ++cache_pos) {
      uint64_t history_pack[kPackCount];
#pragma unroll
      for (int pack = 0; pack < kPackCount; ++pack) {
        history_pack[pack] =
            mask ? load_pack4_device(p.cache + cache_base + cache_pos * p.cache_stride_w +
                                     pack * kFusedW4FastVec)
                 : 0;
      }
#pragma unroll
      for (int v = 0; v < Vec; ++v) {
        const scalar_t h = element_from_pack4_device<scalar_t>(
            history_pack[v / kFusedW4FastVec],
            v % kFusedW4FastVec);
        const scalar_t w = element_from_pack4_device<scalar_t>(weight_pack[v], cache_pos);
        acc[v] += to_float_device(h) * to_float_device(w);
      }
    }

#pragma unroll
    for (int v = 0; v < Vec; ++v) {
      const scalar_t w = element_from_pack4_device<scalar_t>(weight_pack[v], 3);
      acc[v] += to_float_device(x_value[v]) * to_float_device(w);
      if constexpr (UseSilu) {
        acc[v] = acc[v] / (1.0f + sycl::native::exp(-acc[v]));
      }
      if constexpr (UseResidual) {
        acc[v] += to_float_device(x_value[v]);
      }
    }

#pragma unroll
    for (int pack = 0; pack < kPackCount; ++pack) {
      store_pack4_device(
          p.y + t * p.y_stride_t + d0 + pack * kFusedW4FastVec,
          pack4_from_floats_device<scalar_t, Vec>(acc, pack * kFusedW4FastVec));
    }

    if (!valid) {
      return;
    }

    bool do_track = false;
    int64_t track_base = 0;
    if constexpr (DoTrack) {
      if (p.track_mask[t]) {
        do_track = true;
        track_base = p.track_indices[t * p.track_idx_stride] * p.cache_stride_slot + d0;
      }
    }

#pragma unroll
    for (int cache_pos = 0; cache_pos < 3; ++cache_pos) {
#pragma unroll
      for (int pack = 0; pack < kPackCount; ++pack) {
        const uint64_t next_pack =
            cache_pos < 2
                ? (mask ? load_pack4_device(p.cache + cache_base + (cache_pos + 1) * p.cache_stride_w +
                                            pack * kFusedW4FastVec)
                        : 0)
                : x_pack[pack];
        store_pack4_device(
            p.cache + cache_base + cache_pos * p.cache_stride_w + pack * kFusedW4FastVec,
            next_pack);
        if constexpr (DoTrack) {
          if (do_track) {
            store_pack4_device(
                p.cache + track_base + cache_pos * p.cache_stride_w + pack * kFusedW4FastVec,
                next_pack);
          }
        }
      }
    }
  }
};

template <typename scalar_t, int Vec, bool UseSilu, bool UseResidual, bool DoTrack>
void launch_fused_decode_update_w4_packed(
    sycl::queue& q,
    FusedDecodeUpdateParams<scalar_t> const& params) {
  const int64_t total = params.T * (params.D / Vec);
  if (total == 0) {
    return;
  }
  const int64_t global = div_up_i64(total, kThreads) * kThreads;
  FusedDecodeUpdateW4PackedKernel<scalar_t, Vec, UseSilu, UseResidual, DoTrack> kernel{params};
  q.parallel_for<FusedDecodeUpdateW4PackedKernel<scalar_t, Vec, UseSilu, UseResidual, DoTrack>>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kThreads)), kernel);
}

template <typename scalar_t, int Vec, bool UseSilu, bool UseResidual>
void launch_fused_decode_update_w4_packed_track_selected(
    sycl::queue& q,
    FusedDecodeUpdateParams<scalar_t> const& params) {
  if (params.do_track) {
    launch_fused_decode_update_w4_packed<scalar_t, Vec, UseSilu, UseResidual, true>(q, params);
  } else {
    launch_fused_decode_update_w4_packed<scalar_t, Vec, UseSilu, UseResidual, false>(q, params);
  }
}

template <typename scalar_t, int Vec, bool UseSilu>
void launch_fused_decode_update_w4_packed_residual_selected(
    sycl::queue& q,
    FusedDecodeUpdateParams<scalar_t> const& params) {
  if (params.use_residual) {
    launch_fused_decode_update_w4_packed_track_selected<scalar_t, Vec, UseSilu, true>(q, params);
  } else {
    launch_fused_decode_update_w4_packed_track_selected<scalar_t, Vec, UseSilu, false>(q, params);
  }
}

template <typename scalar_t, int Vec>
void launch_fused_decode_update_w4_packed_activation_selected(
    sycl::queue& q,
    FusedDecodeUpdateParams<scalar_t> const& params) {
  if (params.use_silu) {
    launch_fused_decode_update_w4_packed_residual_selected<scalar_t, Vec, true>(q, params);
  } else {
    launch_fused_decode_update_w4_packed_residual_selected<scalar_t, Vec, false>(q, params);
  }
}

template <typename scalar_t>
bool try_launch_fused_decode_update_w4_packed(
    sycl::queue& q,
    FusedDecodeUpdateParams<scalar_t> const& params) {
  if constexpr (sizeof(scalar_t) != sizeof(uint16_t)) {
    return false;
  } else {
    if (params.W != 4 || params.weight_current_first || params.D <= 0) {
      return false;
    }
    if (params.x_stride_d != 1 || params.cache_stride_d != 1 || params.y_stride_d != 1) {
      return false;
    }
    if (params.weight_stride_w != 1 || params.weight_stride_d != 4) {
      return false;
    }
    if (params.x_stride_t % kFusedW4FastVec != 0 ||
        params.y_stride_t % kFusedW4FastVec != 0 ||
        params.cache_stride_slot % kFusedW4FastVec != 0 ||
        params.cache_stride_w % kFusedW4FastVec != 0) {
      return false;
    }
    if (reinterpret_cast<std::uintptr_t>(params.x) % sizeof(uint64_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(params.cache) % sizeof(uint64_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(params.weight) % sizeof(uint64_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(params.y) % sizeof(uint64_t) != 0) {
      return false;
    }
    if (params.D <= 512 && params.D % kFusedW4WideVec == 0) {
      launch_fused_decode_update_w4_packed_activation_selected<scalar_t, kFusedW4WideVec>(q, params);
      return true;
    }
    if (params.D % kFusedW4FastVec == 0) {
      launch_fused_decode_update_w4_packed_activation_selected<scalar_t, kFusedW4FastVec>(q, params);
      return true;
    }
  }
  return false;
}

template <typename scalar_t>
struct GatherScatterParams {
  scalar_t const* hidden;
  scalar_t* cache;
  int32_t const* track_idx;
  bool const* mask;
  int64_t const* dst;
  int64_t B;
  int64_t D;
  int64_t W1;
  int64_t hidden_stride_t;
  int64_t hidden_stride_d;
  int64_t cache_stride_slot;
  int64_t cache_stride_w;
  int64_t cache_stride_d;
  int64_t track_stride_b;
  int64_t track_stride_w;
  int64_t dst_stride_b;
};

template <typename scalar_t>
struct GatherScatterKernel {
  GatherScatterParams<scalar_t> p;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t linear = static_cast<int64_t>(item.get_global_linear_id());
    const int64_t total = p.B * p.W1 * p.D;
    if (linear >= total) {
      return;
    }

    const int64_t d = linear % p.D;
    const int64_t tmp = linear / p.D;
    const int64_t w = tmp % p.W1;
    const int64_t b = tmp / p.W1;
    if (!p.mask[b]) {
      return;
    }
    const int64_t dst_slot = p.dst[b * p.dst_stride_b];
    if (dst_slot == kPadSlotId) {
      return;
    }
    const int64_t src_t = p.track_idx[b * p.track_stride_b + w * p.track_stride_w];
    p.cache[dst_slot * p.cache_stride_slot + w * p.cache_stride_w + d * p.cache_stride_d] =
        p.hidden[src_t * p.hidden_stride_t + d * p.hidden_stride_d];
  }
};

template <typename scalar_t>
void launch_gather_scatter_scalar(sycl::queue& q, GatherScatterParams<scalar_t> const& params) {
  const int64_t total = params.B * params.W1 * params.D;
  if (total == 0) {
    return;
  }
  const int64_t global = div_up_i64(total, kThreads) * kThreads;
  GatherScatterKernel<scalar_t> kernel{params};
  q.parallel_for<GatherScatterKernel<scalar_t>>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kThreads)), kernel);
}

template <typename scalar_t, int StaticW1>
struct GatherScatterPackedKernel {
  GatherScatterParams<scalar_t> p;
  int64_t lanes_per_batch;
  int64_t vec_count;
  int64_t pack_elems;

  void copy_pack(scalar_t const* src_row, scalar_t* dst_row, int64_t pack_idx) const {
    using pack_t = sycl::vec<uint32_t, kUpdateCopyWords>;
    pack_t value;
    value.load(static_cast<std::size_t>(pack_idx), reinterpret_cast<uint32_t const*>(src_row));
    value.store(static_cast<std::size_t>(pack_idx), reinterpret_cast<uint32_t*>(dst_row));
  }

  void operator()(sycl::nd_item<1> item) const {
    constexpr bool kStaticWidth = StaticW1 > 0;
    const int64_t width = kStaticWidth ? StaticW1 : p.W1;
    const int64_t linear = static_cast<int64_t>(item.get_global_linear_id());
    const int64_t total = p.B * lanes_per_batch;
    if (linear >= total) {
      return;
    }

    const int64_t b = linear / lanes_per_batch;
    const int64_t lane = linear - b * lanes_per_batch;
    if (!p.mask[b]) {
      return;
    }
    const int64_t dst_slot = p.dst[b * p.dst_stride_b];
    if (dst_slot == kPadSlotId) {
      return;
    }

    const bool is_vec_lane = lane < vec_count;
    const int64_t channel = is_vec_lane ? lane * pack_elems : vec_count * pack_elems + (lane - vec_count);
    const int64_t cache_base = dst_slot * p.cache_stride_slot;
    const int64_t track_base = b * p.track_stride_b;
    for (int64_t w = 0; w < width; ++w) {
      const int64_t src_t = p.track_idx[track_base + w * p.track_stride_w];
      scalar_t const* src_row = p.hidden + src_t * p.hidden_stride_t;
      scalar_t* dst_row = p.cache + cache_base + w * p.cache_stride_w;
      if (is_vec_lane) {
        copy_pack(src_row, dst_row, lane);
      } else {
        dst_row[channel] = src_row[channel];
      }
    }
  }
};

template <typename scalar_t, int StaticW1>
bool launch_gather_scatter_packed_static(sycl::queue& q, GatherScatterParams<scalar_t> const& params) {
  if (params.B == 0 || params.W1 == 0 || params.D == 0) {
    return true;
  }
  if (params.hidden_stride_d != 1 || params.cache_stride_d != 1) {
    return false;
  }

  const int64_t pack_elems = kUpdateCopyBytes / static_cast<int64_t>(sizeof(scalar_t));
  const bool aligned =
      params.hidden_stride_t % pack_elems == 0 &&
      params.cache_stride_slot % pack_elems == 0 &&
      params.cache_stride_w % pack_elems == 0 &&
      reinterpret_cast<std::uintptr_t>(params.hidden) % kUpdateCopyBytes == 0 &&
      reinterpret_cast<std::uintptr_t>(params.cache) % kUpdateCopyBytes == 0;
  const int64_t vec_count = aligned ? params.D / pack_elems : 0;
  const int64_t scalar_tail = params.D - vec_count * pack_elems;
  const int64_t lanes_per_batch = vec_count + scalar_tail;
  if (lanes_per_batch == 0) {
    return true;
  }

  const int64_t total_lanes = params.B * lanes_per_batch;
  const int64_t global = div_up_i64(total_lanes, kThreads) * kThreads;
  GatherScatterPackedKernel<scalar_t, StaticW1> kernel{params, lanes_per_batch, vec_count, pack_elems};
  q.parallel_for<GatherScatterPackedKernel<scalar_t, StaticW1>>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kThreads)), kernel);
  return true;
}

template <typename scalar_t>
bool launch_gather_scatter_packed(sycl::queue& q, GatherScatterParams<scalar_t> const& params) {
  switch (params.W1) {
    case 1:
      return launch_gather_scatter_packed_static<scalar_t, 1>(q, params);
    case 2:
      return launch_gather_scatter_packed_static<scalar_t, 2>(q, params);
    case 3:
      return launch_gather_scatter_packed_static<scalar_t, 3>(q, params);
    case 5:
      return launch_gather_scatter_packed_static<scalar_t, 5>(q, params);
    case 7:
      return launch_gather_scatter_packed_static<scalar_t, 7>(q, params);
    case 8:
      return launch_gather_scatter_packed_static<scalar_t, 8>(q, params);
    default:
      return launch_gather_scatter_packed_static<scalar_t, 0>(q, params);
  }
}

template <typename scalar_t>
void launch_gather_scatter(sycl::queue& q, GatherScatterParams<scalar_t> const& params) {
  if (!launch_gather_scatter_packed<scalar_t>(q, params)) {
    launch_gather_scatter_scalar<scalar_t>(q, params);
  }
}

template <typename scalar_t>
struct DraftExtendParams {
  scalar_t const* hidden;
  scalar_t* cache;
  int32_t const* cache_indices;
  int32_t const* num_accepted;
  bool const* crossed;
  int32_t const* track_step;
  int64_t const* track_indices;
  int64_t B;
  int64_t D;
  int64_t W1;
  int64_t hidden_stride_b;
  int64_t hidden_stride_t;
  int64_t hidden_stride_d;
  int64_t cache_stride_slot;
  int64_t cache_stride_w;
  int64_t cache_stride_d;
  bool do_track;
};

template <typename scalar_t>
struct DraftExtendKernel {
  DraftExtendParams<scalar_t> p;

  scalar_t load_virtual(int64_t b, int64_t slot, int64_t pos, int64_t d) const {
    if (pos < p.W1) {
      return p.cache[slot * p.cache_stride_slot + pos * p.cache_stride_w + d * p.cache_stride_d];
    }
    return p.hidden[b * p.hidden_stride_b + (pos - p.W1) * p.hidden_stride_t + d * p.hidden_stride_d];
  }

  void store_window(int64_t b, int64_t src_slot, int64_t at, int64_t dst_slot, int64_t d) const {
    if (dst_slot == kPadSlotId) {
      return;
    }
    for (int64_t w = 0; w < p.W1; ++w) {
      p.cache[dst_slot * p.cache_stride_slot + w * p.cache_stride_w + d * p.cache_stride_d] =
          load_virtual(b, src_slot, at + w, d);
    }
  }

  void operator()(sycl::nd_item<1> item) const {
    const int64_t linear = static_cast<int64_t>(item.get_global_linear_id());
    const int64_t total = p.B * p.D;
    if (linear >= total) {
      return;
    }

    const int64_t b = linear / p.D;
    const int64_t d = linear - b * p.D;
    const int32_t slot = p.cache_indices[b];
    const int32_t accepted = p.num_accepted[b];
    if (slot == kPadSlotId || accepted < 0) {
      return;
    }

    if (p.do_track && p.crossed[b]) {
      store_window(b, slot, p.track_step[b], p.track_indices[b], d);
    }
    store_window(b, slot, accepted, slot, d);
  }
};

template <typename scalar_t>
void launch_draft_extend_scalar(sycl::queue& q, DraftExtendParams<scalar_t> const& params) {
  const int64_t total = params.B * params.D;
  if (total == 0) {
    return;
  }
  const int64_t global = div_up_i64(total, kThreads) * kThreads;
  DraftExtendKernel<scalar_t> kernel{params};
  q.parallel_for<DraftExtendKernel<scalar_t>>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kThreads)), kernel);
}

template <typename scalar_t, int StaticW1, bool DoTrack>
struct DraftExtendPackedKernel {
  DraftExtendParams<scalar_t> p;
  int64_t lanes_per_batch;
  int64_t vec_count;
  int64_t pack_elems;

  using pack_t = sycl::vec<uint32_t, kUpdateCopyWords>;

  pack_t load_pack(scalar_t const* row_base, int64_t pack_idx) const {
    pack_t value;
    value.load(static_cast<std::size_t>(pack_idx), reinterpret_cast<uint32_t const*>(row_base));
    return value;
  }

  void store_pack(scalar_t* row_base, int64_t pack_idx, pack_t value) const {
    value.store(static_cast<std::size_t>(pack_idx), reinterpret_cast<uint32_t*>(row_base));
  }

  scalar_t load_virtual_scalar(
      scalar_t const (&init)[kPackedCopyMaxWindow],
      scalar_t const* hidden_base,
      int64_t pos,
      int64_t channel,
      int64_t width) const {
    if (pos < width) {
      return init[pos];
    }
    return hidden_base[(pos - width) * p.hidden_stride_t + channel];
  }

  pack_t load_virtual_pack(
      pack_t const (&init)[kPackedCopyMaxWindow],
      scalar_t const* hidden_base,
      int64_t pos,
      int64_t pack_idx,
      int64_t width) const {
    if (pos < width) {
      return init[pos];
    }
    return load_pack(hidden_base + (pos - width) * p.hidden_stride_t, pack_idx);
  }

  void operator()(sycl::nd_item<1> item) const {
    constexpr bool kStaticWidth = StaticW1 > 0;
    const int64_t width = kStaticWidth ? StaticW1 : p.W1;
    const int64_t linear = static_cast<int64_t>(item.get_global_linear_id());
    const int64_t total = p.B * lanes_per_batch;
    if (linear >= total) {
      return;
    }

    const int64_t b = linear / lanes_per_batch;
    const int64_t lane = linear - b * lanes_per_batch;
    const int32_t slot = p.cache_indices[b];
    const int32_t accepted = p.num_accepted[b];
    if (slot == kPadSlotId || accepted < 0) {
      return;
    }

    const bool is_vec_lane = lane < vec_count;
    const int64_t channel = is_vec_lane ? lane * pack_elems : vec_count * pack_elems + (lane - vec_count);
    const int64_t cache_base = static_cast<int64_t>(slot) * p.cache_stride_slot;
    scalar_t const* hidden_base = p.hidden + b * p.hidden_stride_b;

    bool do_track_write = false;
    int64_t track_at = 0;
    int64_t track_slot = 0;
    if constexpr (DoTrack) {
      do_track_write = p.crossed[b];
      if (do_track_write) {
        track_at = p.track_step[b];
        track_slot = p.track_indices[b];
        do_track_write = track_slot != kPadSlotId;
      }
    }

    const bool need_init = accepted < width || (do_track_write && track_at < width);
    if (is_vec_lane) {
      pack_t init[kPackedCopyMaxWindow];
      if (need_init) {
        for (int64_t w = 0; w < width; ++w) {
          scalar_t const* row = p.cache + cache_base + w * p.cache_stride_w;
          init[w] = load_pack(row, lane);
        }
      }

      if constexpr (DoTrack) {
        if (do_track_write) {
          const int64_t dst_base = track_slot * p.cache_stride_slot;
          for (int64_t w = 0; w < width; ++w) {
            scalar_t* dst_row = p.cache + dst_base + w * p.cache_stride_w;
            store_pack(dst_row, lane, load_virtual_pack(init, hidden_base, track_at + w, lane, width));
          }
        }
      }

      for (int64_t w = 0; w < width; ++w) {
        scalar_t* dst_row = p.cache + cache_base + w * p.cache_stride_w;
        store_pack(dst_row, lane, load_virtual_pack(init, hidden_base, static_cast<int64_t>(accepted) + w, lane, width));
      }
    } else {
      scalar_t init[kPackedCopyMaxWindow];
      if (need_init) {
        for (int64_t w = 0; w < width; ++w) {
          init[w] = p.cache[cache_base + w * p.cache_stride_w + channel];
        }
      }

      if constexpr (DoTrack) {
        if (do_track_write) {
          const int64_t dst_base = track_slot * p.cache_stride_slot;
          for (int64_t w = 0; w < width; ++w) {
            p.cache[dst_base + w * p.cache_stride_w + channel] =
                load_virtual_scalar(init, hidden_base, track_at + w, channel, width);
          }
        }
      }

      for (int64_t w = 0; w < width; ++w) {
        p.cache[cache_base + w * p.cache_stride_w + channel] =
            load_virtual_scalar(init, hidden_base, static_cast<int64_t>(accepted) + w, channel, width);
      }
    }
  }
};

template <typename scalar_t, int StaticW1, bool DoTrack>
bool launch_draft_extend_packed_static(sycl::queue& q, DraftExtendParams<scalar_t> const& params) {
  if (params.B == 0 || params.W1 == 0 || params.D == 0) {
    return true;
  }
  if (params.W1 > kPackedCopyMaxWindow) {
    return false;
  }
  if (params.hidden_stride_d != 1 || params.cache_stride_d != 1) {
    return false;
  }

  const int64_t pack_elems = kUpdateCopyBytes / static_cast<int64_t>(sizeof(scalar_t));
  const bool aligned =
      params.hidden_stride_b % pack_elems == 0 &&
      params.hidden_stride_t % pack_elems == 0 &&
      params.cache_stride_slot % pack_elems == 0 &&
      params.cache_stride_w % pack_elems == 0 &&
      reinterpret_cast<std::uintptr_t>(params.hidden) % kUpdateCopyBytes == 0 &&
      reinterpret_cast<std::uintptr_t>(params.cache) % kUpdateCopyBytes == 0;
  const int64_t vec_count = aligned ? params.D / pack_elems : 0;
  const int64_t scalar_tail = params.D - vec_count * pack_elems;
  const int64_t lanes_per_batch = vec_count + scalar_tail;
  if (lanes_per_batch == 0) {
    return true;
  }

  const int64_t total_lanes = params.B * lanes_per_batch;
  const int64_t global = div_up_i64(total_lanes, kThreads) * kThreads;
  DraftExtendPackedKernel<scalar_t, StaticW1, DoTrack> kernel{
      params, lanes_per_batch, vec_count, pack_elems};
  q.parallel_for<DraftExtendPackedKernel<scalar_t, StaticW1, DoTrack>>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kThreads)), kernel);
  return true;
}

template <typename scalar_t, int StaticW1>
bool launch_draft_extend_packed_track_selected(sycl::queue& q, DraftExtendParams<scalar_t> const& params) {
  if (params.do_track) {
    return launch_draft_extend_packed_static<scalar_t, StaticW1, true>(q, params);
  }
  return launch_draft_extend_packed_static<scalar_t, StaticW1, false>(q, params);
}

template <typename scalar_t>
bool launch_draft_extend_packed(sycl::queue& q, DraftExtendParams<scalar_t> const& params) {
  switch (params.W1) {
    case 1:
      return launch_draft_extend_packed_track_selected<scalar_t, 1>(q, params);
    case 2:
      return launch_draft_extend_packed_track_selected<scalar_t, 2>(q, params);
    case 3:
      return launch_draft_extend_packed_track_selected<scalar_t, 3>(q, params);
    case 5:
      return launch_draft_extend_packed_track_selected<scalar_t, 5>(q, params);
    case 7:
      return launch_draft_extend_packed_track_selected<scalar_t, 7>(q, params);
    case 8:
      return launch_draft_extend_packed_track_selected<scalar_t, 8>(q, params);
    default:
      return launch_draft_extend_packed_track_selected<scalar_t, 0>(q, params);
  }
}

template <typename scalar_t>
void launch_draft_extend(sycl::queue& q, DraftExtendParams<scalar_t> const& params) {
  if (!launch_draft_extend_packed<scalar_t>(q, params)) {
    launch_draft_extend_scalar<scalar_t>(q, params);
  }
}

struct DecodeMetadataParams {
  int32_t const* cache_indices;
  int32_t* query_start_loc;
  bool* has_initial_state;
  bool* cache_mask;
  int64_t* safe_idx;
  int64_t* cu;
  int32_t* si;
  int64_t B;
};

struct DecodeMetadataKernel {
  DecodeMetadataParams p;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t off = static_cast<int64_t>(item.get_global_linear_id());
    if (off > p.B) {
      return;
    }
    p.query_start_loc[off] = static_cast<int32_t>(off);
    p.cu[off] = off;
    if (off < p.B) {
      const int32_t ci = p.cache_indices[off];
      p.has_initial_state[off] = true;
      p.cache_mask[off] = ci != kPadSlotId;
      p.safe_idx[off] = ci < 0 ? 0 : static_cast<int64_t>(ci);
      p.si[off] = static_cast<int32_t>(off);
    }
  }
};

void launch_decode_metadata(sycl::queue& q, DecodeMetadataParams const& params) {
  const int64_t total = params.B + 1;
  const int64_t global = div_up_i64(total, kMetaThreads) * kMetaThreads;
  DecodeMetadataKernel kernel{params};
  q.parallel_for<DecodeMetadataKernel>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kMetaThreads)), kernel);
}

enum HisMode {
  kHisZeros = 0,
  kHisPrefix = 1,
  kHisSeqMinusExt = 2,
  kHisOnes = 3,
};

struct ExtendMetadataParams {
  int32_t const* cache_indices;
  int32_t const* extend_seq_lens;
  int32_t const* his_src;
  int32_t* query_start_loc;
  bool* has_initial_state;
  bool* cache_mask;
  int64_t* safe_idx;
  int64_t* cu;
  int32_t* si;
  int64_t B;
  int64_t T;
  int64_t draft_token_num;
  int his_mode;
};

struct ExtendMetadataKernel {
  ExtendMetadataParams p;

  int64_t sequence_start(int64_t group) const {
    if (p.his_mode == kHisOnes) {
      return group * p.draft_token_num;
    }
    int64_t start = 0;
    for (int64_t i = 0; i < group; ++i) {
      start += static_cast<int64_t>(p.extend_seq_lens[i]);
    }
    return start;
  }

  int64_t sequence_length(int64_t group) const {
    if (group >= p.B) {
      return 0;
    }
    return p.his_mode == kHisOnes ? p.draft_token_num : static_cast<int64_t>(p.extend_seq_lens[group]);
  }

  bool has_initial_state(int64_t group, int64_t len) const {
    if (p.his_mode == kHisZeros) {
      return false;
    }
    if (p.his_mode == kHisPrefix) {
      return p.his_src[group] > 0;
    }
    if (p.his_mode == kHisSeqMinusExt) {
      return (p.his_src[group] - len) > 0;
    }
    return true;
  }

  void operator()(sycl::nd_item<1> item) const {
    const int64_t group = static_cast<int64_t>(item.get_group(0));
    const int64_t lane = static_cast<int64_t>(item.get_local_id(0));
    const int64_t start = sequence_start(group < p.B ? group : p.B);
    const int64_t len = sequence_length(group);
    const int64_t end = group < p.B ? start + len : p.T;

    if (group < p.B && lane == 0) {
      const bool his = has_initial_state(group, len);
      const int32_t ci = p.cache_indices[group];
      p.query_start_loc[group] = static_cast<int32_t>(start);
      p.query_start_loc[group + 1] = static_cast<int32_t>(start + len);
      p.cu[group] = start;
      p.cu[group + 1] = start + len;
      p.has_initial_state[group] = his;
      p.cache_mask[group] = his && ci != kPadSlotId;
      p.safe_idx[group] = ci < 0 ? 0 : static_cast<int64_t>(ci);
    }

    if (p.B <= 0) {
      return;
    }
    const int32_t seq = static_cast<int32_t>(group < p.B ? group : p.B - 1);
    const int64_t fill_begin = sycl::max<int64_t>(0, start);
    const int64_t fill_end = sycl::min<int64_t>(p.T, end);
    for (int64_t t = fill_begin + lane; t < fill_end; t += kMetaThreads) {
      p.si[t] = seq;
    }
  }
};

void launch_extend_metadata(sycl::queue& q, ExtendMetadataParams const& params) {
  const int64_t groups = params.B + 1;
  if (groups <= 0) {
    return;
  }
  ExtendMetadataKernel kernel{params};
  q.parallel_for<ExtendMetadataKernel>(
      sycl::nd_range<1>(
          sycl::range<1>(static_cast<std::size_t>(groups * kMetaThreads)),
          sycl::range<1>(kMetaThreads)),
      kernel);
}

struct TrackIndicesParams {
  int32_t const* query_start_loc;
  int32_t const* mamba_track_seqlens;
  int32_t const* extend_prefix_lens;
  int32_t* track_indices;
  int64_t B;
  int64_t W1;
  int64_t chunk_size;
  int64_t total_tokens;
};

struct TrackIndicesKernel {
  TrackIndicesParams p;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t linear = static_cast<int64_t>(item.get_global_linear_id());
    const int64_t total = p.B * p.W1;
    if (linear >= total) {
      return;
    }
    const int64_t b = linear / p.W1;
    const int64_t w = linear - b * p.W1;
    int64_t lens_to_track =
        static_cast<int64_t>(p.mamba_track_seqlens[b]) - static_cast<int64_t>(p.extend_prefix_lens[b]);
    if (lens_to_track < 0) {
      lens_to_track = 0;
    }
    const int64_t aligned = (lens_to_track / p.chunk_size) * p.chunk_size;
    int64_t idx = static_cast<int64_t>(p.query_start_loc[b]) + aligned - p.W1 + w;
    const int64_t max_idx = sycl::max<int64_t>(0, p.total_tokens - 1);
    idx = sycl::max<int64_t>(0, sycl::min<int64_t>(idx, max_idx));
    p.track_indices[linear] = static_cast<int32_t>(idx);
  }
};

void launch_track_indices(sycl::queue& q, TrackIndicesParams const& params) {
  const int64_t total = params.B * params.W1;
  if (total == 0) {
    return;
  }
  const int64_t global = div_up_i64(total, kThreads) * kThreads;
  TrackIndicesKernel kernel{params};
  q.parallel_for<TrackIndicesKernel>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kThreads)), kernel);
}

template <typename scalar_t>
struct SaveWindowsParams {
  scalar_t const* cache;
  scalar_t const* hidden;
  int32_t const* cache_indices;
  scalar_t* out;
  int64_t B;
  int64_t draft_tokens;
  int64_t W1;
  int64_t D;
  int64_t cache_stride_slot;
  int64_t cache_stride_w;
  int64_t cache_stride_d;
  int64_t hidden_stride_b;
  int64_t hidden_stride_t;
  int64_t hidden_stride_d;
  int64_t out_stride_b;
  int64_t out_stride_t;
  int64_t out_stride_w;
  int64_t out_stride_d;
};

template <typename scalar_t>
struct SaveWindowsKernel {
  SaveWindowsParams<scalar_t> p;

  scalar_t load_virtual(int64_t b, int64_t slot, int64_t pos, int64_t d) const {
    if (pos < p.W1) {
      return p.cache[slot * p.cache_stride_slot + pos * p.cache_stride_w + d * p.cache_stride_d];
    }
    return p.hidden[b * p.hidden_stride_b + (pos - p.W1) * p.hidden_stride_t + d * p.hidden_stride_d];
  }

  void operator()(sycl::nd_item<1> item) const {
    const int64_t linear = static_cast<int64_t>(item.get_global_linear_id());
    const int64_t total = p.B * p.draft_tokens * p.W1 * p.D;
    if (linear >= total) {
      return;
    }

    const int64_t d = linear % p.D;
    int64_t tmp = linear / p.D;
    const int64_t w = tmp % p.W1;
    tmp /= p.W1;
    const int64_t t = tmp % p.draft_tokens;
    const int64_t b = tmp / p.draft_tokens;
    const int32_t slot = p.cache_indices[b];
    if (slot == kPadSlotId) {
      return;
    }
    p.out[b * p.out_stride_b + t * p.out_stride_t + w * p.out_stride_w + d * p.out_stride_d] =
        load_virtual(b, slot, t + 1 + w, d);
  }
};

template <typename scalar_t>
void launch_save_windows_scalar(sycl::queue& q, SaveWindowsParams<scalar_t> const& params) {
  const int64_t total = params.B * params.draft_tokens * params.W1 * params.D;
  if (total == 0) {
    return;
  }
  const int64_t global = div_up_i64(total, kThreads) * kThreads;
  SaveWindowsKernel<scalar_t> kernel{params};
  q.parallel_for<SaveWindowsKernel<scalar_t>>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kThreads)), kernel);
}

template <typename scalar_t, int StaticW1>
struct SaveWindowsPackedKernel {
  SaveWindowsParams<scalar_t> p;
  int64_t lanes_per_row;
  int64_t vec_count;
  int64_t pack_elems;

  using pack_t = sycl::vec<uint32_t, kUpdateCopyWords>;

  void copy_pack(scalar_t const* src_row, scalar_t* dst_row, int64_t pack_idx) const {
#pragma unroll
    for (int i = 0; i < kWindowPacksPerLane; ++i) {
      pack_t value;
      value.load(static_cast<std::size_t>(pack_idx + i), reinterpret_cast<uint32_t const*>(src_row));
      value.store(static_cast<std::size_t>(pack_idx + i), reinterpret_cast<uint32_t*>(dst_row));
    }
  }

  scalar_t const* source_row(int64_t b, int64_t slot, int64_t position, int64_t width) const {
    if (position < width) {
      return p.cache + slot * p.cache_stride_slot + position * p.cache_stride_w;
    }
    return p.hidden + b * p.hidden_stride_b + (position - width) * p.hidden_stride_t;
  }

  scalar_t* destination_row(int64_t b, int64_t t, int64_t w) const {
    return p.out + b * p.out_stride_b + t * p.out_stride_t + w * p.out_stride_w;
  }

  void operator()(sycl::nd_item<1> item) const {
    constexpr bool kStaticWidth = StaticW1 > 0;
    const int64_t width = kStaticWidth ? StaticW1 : p.W1;
    const int64_t linear = static_cast<int64_t>(item.get_global_linear_id());
    const int64_t rows_per_batch = p.draft_tokens * width;
    const int64_t total = p.B * rows_per_batch * lanes_per_row;
    if (linear >= total) {
      return;
    }

    const int64_t row_lane = linear / lanes_per_row;
    const int64_t lane = linear - row_lane * lanes_per_row;
    const int64_t b = row_lane / rows_per_batch;
    const int64_t local_row = row_lane - b * rows_per_batch;
    const int64_t t = local_row / width;
    const int64_t w = local_row - t * width;
    const int32_t slot = p.cache_indices[b];
    if (slot == kPadSlotId) {
      return;
    }

    const bool is_vec_lane = lane < vec_count;
    const int64_t lane_elems = pack_elems * kWindowPacksPerLane;
    const int64_t channel = is_vec_lane ? lane * lane_elems : vec_count * lane_elems + (lane - vec_count);
    scalar_t const* src_row = source_row(b, slot, t + 1 + w, width);
    scalar_t* dst_row = destination_row(b, t, w);
    if (is_vec_lane) {
      copy_pack(src_row, dst_row, lane * kWindowPacksPerLane);
    } else {
      dst_row[channel] = src_row[channel];
    }
  }
};

template <typename scalar_t, int StaticW1>
bool launch_save_windows_packed_static(sycl::queue& q, SaveWindowsParams<scalar_t> const& params) {
  if (params.B == 0 || params.draft_tokens == 0 || params.W1 == 0 || params.D == 0) {
    return true;
  }
  if (params.cache_stride_d != 1 || params.hidden_stride_d != 1 || params.out_stride_d != 1) {
    return false;
  }

  const int64_t pack_elems = kUpdateCopyBytes / static_cast<int64_t>(sizeof(scalar_t));
  const int64_t lane_elems = pack_elems * kWindowPacksPerLane;
  const bool aligned =
      params.cache_stride_slot % pack_elems == 0 &&
      params.cache_stride_w % pack_elems == 0 &&
      params.hidden_stride_b % pack_elems == 0 &&
      params.hidden_stride_t % pack_elems == 0 &&
      params.out_stride_b % pack_elems == 0 &&
      params.out_stride_t % pack_elems == 0 &&
      params.out_stride_w % pack_elems == 0 &&
      reinterpret_cast<std::uintptr_t>(params.cache) % kUpdateCopyBytes == 0 &&
      reinterpret_cast<std::uintptr_t>(params.hidden) % kUpdateCopyBytes == 0 &&
      reinterpret_cast<std::uintptr_t>(params.out) % kUpdateCopyBytes == 0;
  const int64_t vec_count = aligned ? params.D / lane_elems : 0;
  const int64_t scalar_tail = params.D - vec_count * lane_elems;
  const int64_t lanes_per_row = vec_count + scalar_tail;
  if (lanes_per_row == 0) {
    return true;
  }

  const int64_t width = StaticW1 > 0 ? StaticW1 : params.W1;
  const int64_t total_lanes = params.B * params.draft_tokens * width * lanes_per_row;
  const int64_t global = div_up_i64(total_lanes, kThreads) * kThreads;
  SaveWindowsPackedKernel<scalar_t, StaticW1> kernel{
      params, lanes_per_row, vec_count, pack_elems};
  q.parallel_for<SaveWindowsPackedKernel<scalar_t, StaticW1>>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kThreads)), kernel);
  return true;
}

template <typename scalar_t>
bool launch_save_windows_packed(sycl::queue& q, SaveWindowsParams<scalar_t> const& params) {
  switch (params.W1) {
    case 1:
      return launch_save_windows_packed_static<scalar_t, 1>(q, params);
    case 2:
      return launch_save_windows_packed_static<scalar_t, 2>(q, params);
    case 3:
      return launch_save_windows_packed_static<scalar_t, 3>(q, params);
    case 5:
      return launch_save_windows_packed_static<scalar_t, 5>(q, params);
    case 7:
      return launch_save_windows_packed_static<scalar_t, 7>(q, params);
    case 8:
      return launch_save_windows_packed_static<scalar_t, 8>(q, params);
    default:
      return launch_save_windows_packed_static<scalar_t, 0>(q, params);
  }
}

template <typename scalar_t>
void launch_save_windows(sycl::queue& q, SaveWindowsParams<scalar_t> const& params) {
  if (!launch_save_windows_packed<scalar_t>(q, params)) {
    launch_save_windows_scalar<scalar_t>(q, params);
  }
}

void check_xpu_tensor(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_xpu(), name, " must be an XPU tensor");
}

void check_sconv_dtype(const at::Tensor& x, const at::Tensor& other, const char* name) {
  TORCH_CHECK(other.scalar_type() == x.scalar_type(), name, " dtype must match x dtype");
}

struct WeightLayout {
  int64_t W;
  int64_t stride_d;
  int64_t stride_w;
  bool current_first;
};

WeightLayout resolve_weight_layout(const at::Tensor& weight, int64_t D, int64_t W, bool d_w_current_first) {
  if (weight.size(0) == D && weight.size(1) == W) {
    return WeightLayout{W, weight.stride(0), weight.stride(1), d_w_current_first};
  }
  if (weight.size(0) == W && weight.size(1) == D) {
    return WeightLayout{W, weight.stride(1), weight.stride(0), true};
  }
  TORCH_CHECK(
      false,
      "weight must have shape [D, W] or [W, D] with D=",
      D,
      " and W=",
      W,
      ", got [",
      weight.size(0),
      ", ",
      weight.size(1),
      "]");
  return WeightLayout{W, 0, 0, false};
}

}  // namespace

at::Tensor inkling_sconv_forward(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& sconv_cache,
    const at::Tensor& cache_mask,
    const at::Tensor& safe_idx,
    const at::Tensor& cu,
    const at::Tensor& si,
    bool silu_activation,
    bool use_residual,
    bool is_decode) {
  check_xpu_tensor(x, "x");
  check_xpu_tensor(weight, "weight");
  check_xpu_tensor(sconv_cache, "sconv_cache");
  check_xpu_tensor(cache_mask, "cache_mask");
  check_xpu_tensor(safe_idx, "safe_idx");
  check_xpu_tensor(cu, "cu");
  check_xpu_tensor(si, "si");
  TORCH_CHECK(x.dim() == 2, "x must have shape [T, D]");
  TORCH_CHECK(weight.dim() == 2, "weight must have shape [D, W] or [W, D]");
  TORCH_CHECK(sconv_cache.dim() == 3, "sconv_cache must have shape [slots, W-1, D]");
  TORCH_CHECK(cache_mask.dim() >= 1, "cache_mask must have at least one dimension");
  TORCH_CHECK(safe_idx.scalar_type() == at::ScalarType::Long, "safe_idx must be int64");
  TORCH_CHECK(cu.scalar_type() == at::ScalarType::Long, "cu must be int64");
  TORCH_CHECK(si.scalar_type() == at::ScalarType::Int, "si must be int32");
  TORCH_CHECK(cache_mask.scalar_type() == at::ScalarType::Bool, "cache_mask must be bool");
  check_sconv_dtype(x, weight, "weight");
  check_sconv_dtype(x, sconv_cache, "sconv_cache");
  TORCH_CHECK(sconv_cache.size(2) == x.size(1), "sconv_cache.size(2) must equal x.size(1)");
  TORCH_CHECK(si.numel() == x.size(0), "si must have one entry per token");
  TORCH_CHECK(x.stride(1) == 1, "x must be contiguous on D");
  TORCH_CHECK(sconv_cache.stride(2) == 1, "sconv_cache must be contiguous on D");
  const WeightLayout weight_layout = resolve_weight_layout(weight, x.size(1), sconv_cache.size(1) + 1, false);

  at::Tensor y = at::empty_strided({x.size(0), x.size(1)}, {x.size(1), 1}, x.options());
  auto queue = c10::xpu::getCurrentXPUStream().queue();
  const auto input_type = x.scalar_type();
  SYCL_DISPATCH_FLOATING_TYPES(at::ScalarType::Half, at::ScalarType::BFloat16, input_type, "inkling_sconv_forward", [&]() -> at::Tensor {
    SconvForwardParams<scalar_t> params{
        x.data_ptr<scalar_t>(),
        sconv_cache.data_ptr<scalar_t>(),
        cache_mask.data_ptr<bool>(),
        safe_idx.data_ptr<int64_t>(),
        cu.data_ptr<int64_t>(),
        si.data_ptr<int32_t>(),
        weight.data_ptr<scalar_t>(),
        y.data_ptr<scalar_t>(),
        x.size(0),
        x.size(1),
        weight_layout.W,
        x.stride(0),
        x.stride(1),
        sconv_cache.stride(0),
        sconv_cache.stride(1),
        sconv_cache.stride(2),
        cache_mask.stride(0),
        weight_layout.stride_d,
        weight_layout.stride_w,
        y.stride(0),
        y.stride(1),
        silu_activation,
        use_residual,
        is_decode,
        weight_layout.current_first};
    if (!try_launch_sconv_forward_block<scalar_t>(queue, params)) {
      launch_sconv_forward<scalar_t>(queue, params);
    }
    return y;
  });
  return y;
}

void inkling_update_sconv_cache(
    const at::Tensor& x,
    at::Tensor& sconv_cache,
    const at::Tensor& cache_indices,
    const at::Tensor& has_initial_state,
    const at::Tensor& query_start_loc) {
  check_xpu_tensor(x, "x");
  check_xpu_tensor(sconv_cache, "sconv_cache");
  check_xpu_tensor(cache_indices, "cache_indices");
  check_xpu_tensor(has_initial_state, "has_initial_state");
  check_xpu_tensor(query_start_loc, "query_start_loc");
  TORCH_CHECK(x.dim() == 2, "x must have shape [T, D]");
  TORCH_CHECK(sconv_cache.dim() == 3, "sconv_cache must have shape [slots, W-1, D]");
  TORCH_CHECK(cache_indices.scalar_type() == at::ScalarType::Int, "cache_indices must be int32");
  TORCH_CHECK(has_initial_state.scalar_type() == at::ScalarType::Bool, "has_initial_state must be bool");
  TORCH_CHECK(query_start_loc.scalar_type() == at::ScalarType::Int, "query_start_loc must be int32");
  check_sconv_dtype(x, sconv_cache, "sconv_cache");
  TORCH_CHECK(sconv_cache.size(2) == x.size(1), "sconv_cache.size(2) must equal x.size(1)");
  TORCH_CHECK(cache_indices.numel() + 1 == query_start_loc.numel(), "query_start_loc must have B + 1 entries");
  TORCH_CHECK(has_initial_state.numel() == cache_indices.numel(), "has_initial_state must have B entries");
  TORCH_CHECK(x.stride(1) == 1, "x must be contiguous on D");
  TORCH_CHECK(sconv_cache.stride(2) == 1, "sconv_cache must be contiguous on D");

  auto queue = c10::xpu::getCurrentXPUStream().queue();
  const auto input_type = x.scalar_type();
  SYCL_DISPATCH_FLOATING_TYPES(at::ScalarType::Half, at::ScalarType::BFloat16, input_type, "inkling_update_sconv_cache", [&]() {
    UpdateSconvCacheParams<scalar_t> params{
        x.data_ptr<scalar_t>(),
        sconv_cache.data_ptr<scalar_t>(),
        cache_indices.data_ptr<int32_t>(),
        has_initial_state.data_ptr<bool>(),
        query_start_loc.data_ptr<int32_t>(),
        cache_indices.numel(),
        x.size(1),
        sconv_cache.size(1),
        x.stride(0),
        x.stride(1),
        sconv_cache.stride(0),
        sconv_cache.stride(1),
        sconv_cache.stride(2)};
    launch_update_sconv_cache<scalar_t>(queue, params);
  });
}

at::Tensor inkling_fused_decode_update_sconv(
    const at::Tensor& x,
    const at::Tensor& weight,
    at::Tensor& sconv_cache,
    const at::Tensor& cache_indices,
    const at::Tensor& cache_mask,
    bool silu_activation,
    bool use_residual,
    const std::optional<at::Tensor>& track_mask,
    const std::optional<at::Tensor>& track_indices) {
  check_xpu_tensor(x, "x");
  check_xpu_tensor(weight, "weight");
  check_xpu_tensor(sconv_cache, "sconv_cache");
  check_xpu_tensor(cache_indices, "cache_indices");
  check_xpu_tensor(cache_mask, "cache_mask");
  TORCH_CHECK(x.dim() == 2, "x must have shape [T, D]");
  TORCH_CHECK(weight.dim() == 2, "weight must have shape [D, W] or [W, D]");
  TORCH_CHECK(sconv_cache.dim() == 3, "sconv_cache must have shape [slots, W-1, D]");
  TORCH_CHECK(cache_indices.scalar_type() == at::ScalarType::Int, "cache_indices must be int32");
  TORCH_CHECK(cache_mask.scalar_type() == at::ScalarType::Bool, "cache_mask must be bool");
  check_sconv_dtype(x, weight, "weight");
  check_sconv_dtype(x, sconv_cache, "sconv_cache");
  TORCH_CHECK(cache_indices.numel() == x.size(0), "cache_indices must have one entry per token");
  TORCH_CHECK(cache_mask.numel() >= x.size(0), "cache_mask must have at least T entries");
  TORCH_CHECK(sconv_cache.size(2) == x.size(1), "sconv_cache.size(2) must equal x.size(1)");
  TORCH_CHECK(x.stride(1) == 1, "x must be contiguous on D");
  TORCH_CHECK(sconv_cache.stride(2) == 1, "sconv_cache must be contiguous on D");
  const WeightLayout weight_layout = resolve_weight_layout(weight, x.size(1), sconv_cache.size(1) + 1, false);
  const bool do_track = track_mask.has_value();
  if (do_track) {
    TORCH_CHECK(track_indices.has_value(), "track_indices must be provided when track_mask is provided");
    check_xpu_tensor(track_mask.value(), "track_mask");
    check_xpu_tensor(track_indices.value(), "track_indices");
    TORCH_CHECK(track_mask.value().scalar_type() == at::ScalarType::Bool, "track_mask must be bool");
    TORCH_CHECK(track_indices.value().scalar_type() == at::ScalarType::Long, "track_indices must be int64");
    TORCH_CHECK(track_mask.value().numel() >= x.size(0), "track_mask must have at least T entries");
    TORCH_CHECK(track_indices.value().numel() >= x.size(0), "track_indices must have at least T entries");
  }

  at::Tensor y = at::empty_strided({x.size(0), x.size(1)}, {x.size(1), 1}, x.options());
  auto queue = c10::xpu::getCurrentXPUStream().queue();
  const auto input_type = x.scalar_type();
  SYCL_DISPATCH_FLOATING_TYPES(at::ScalarType::Half, at::ScalarType::BFloat16, input_type, "inkling_fused_decode_update_sconv", [&]() -> at::Tensor {
    FusedDecodeUpdateParams<scalar_t> params{
        x.data_ptr<scalar_t>(),
        sconv_cache.data_ptr<scalar_t>(),
        cache_indices.data_ptr<int32_t>(),
        cache_mask.data_ptr<bool>(),
        weight.data_ptr<scalar_t>(),
        y.data_ptr<scalar_t>(),
        do_track ? track_mask.value().data_ptr<bool>() : nullptr,
        do_track ? track_indices.value().data_ptr<int64_t>() : nullptr,
        x.size(0),
        x.size(1),
        weight_layout.W,
        x.stride(0),
        x.stride(1),
        sconv_cache.stride(0),
        sconv_cache.stride(1),
        sconv_cache.stride(2),
        weight_layout.stride_d,
        weight_layout.stride_w,
        y.stride(0),
        y.stride(1),
        do_track ? track_indices.value().stride(0) : 0,
        silu_activation,
        use_residual,
        do_track,
        weight_layout.current_first};
    if (!try_launch_fused_decode_update_w4_packed<scalar_t>(queue, params)) {
      launch_fused_decode_update<scalar_t>(queue, params);
    }
    return y;
  });
  return y;
}

void inkling_gather_scatter_sconv_cache(
    const at::Tensor& hidden_states,
    at::Tensor& sconv_cache,
    const at::Tensor& track_conv_indices,
    const at::Tensor& mask,
    const at::Tensor& dst_indices) {
  check_xpu_tensor(hidden_states, "hidden_states");
  check_xpu_tensor(sconv_cache, "sconv_cache");
  check_xpu_tensor(track_conv_indices, "track_conv_indices");
  check_xpu_tensor(mask, "mask");
  check_xpu_tensor(dst_indices, "dst_indices");
  TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must have shape [T, D]");
  TORCH_CHECK(sconv_cache.dim() == 3, "sconv_cache must have shape [slots, W-1, D]");
  TORCH_CHECK(track_conv_indices.dim() == 2, "track_conv_indices must have shape [B, W-1]");
  TORCH_CHECK(mask.scalar_type() == at::ScalarType::Bool, "mask must be bool");
  TORCH_CHECK(track_conv_indices.scalar_type() == at::ScalarType::Int, "track_conv_indices must be int32");
  TORCH_CHECK(dst_indices.scalar_type() == at::ScalarType::Long, "dst_indices must be int64");
  check_sconv_dtype(hidden_states, sconv_cache, "sconv_cache");
  TORCH_CHECK(track_conv_indices.size(1) == sconv_cache.size(1), "track_conv_indices.size(1) must equal W-1");
  TORCH_CHECK(track_conv_indices.size(0) == mask.numel(), "mask must have B entries");
  TORCH_CHECK(dst_indices.numel() >= mask.numel(), "dst_indices must have at least B entries");
  TORCH_CHECK(sconv_cache.size(2) == hidden_states.size(1), "sconv_cache.size(2) must equal hidden_states.size(1)");
  TORCH_CHECK(hidden_states.stride(1) == 1, "hidden_states must be contiguous on D");
  TORCH_CHECK(sconv_cache.stride(2) == 1, "sconv_cache must be contiguous on D");

  auto queue = c10::xpu::getCurrentXPUStream().queue();
  const auto input_type = hidden_states.scalar_type();
  SYCL_DISPATCH_FLOATING_TYPES(at::ScalarType::Half, at::ScalarType::BFloat16, input_type, "inkling_gather_scatter_sconv_cache", [&]() {
    GatherScatterParams<scalar_t> params{
        hidden_states.data_ptr<scalar_t>(),
        sconv_cache.data_ptr<scalar_t>(),
        track_conv_indices.data_ptr<int32_t>(),
        mask.data_ptr<bool>(),
        dst_indices.data_ptr<int64_t>(),
        mask.numel(),
        hidden_states.size(1),
        sconv_cache.size(1),
        hidden_states.stride(0),
        hidden_states.stride(1),
        sconv_cache.stride(0),
        sconv_cache.stride(1),
        sconv_cache.stride(2),
        track_conv_indices.stride(0),
        track_conv_indices.stride(1),
        dst_indices.stride(0)};
    launch_gather_scatter<scalar_t>(queue, params);
  });
}

void inkling_draft_extend_sconv_cache(
    const at::Tensor& hidden_states,
    at::Tensor& sconv_cache,
    const at::Tensor& cache_indices,
    const at::Tensor& num_accepted_tokens,
    int64_t draft_token_num,
    bool do_tracking,
    const std::optional<at::Tensor>& crossed,
    const std::optional<at::Tensor>& track_step,
    const std::optional<at::Tensor>& mamba_track_indices) {
  check_xpu_tensor(hidden_states, "hidden_states");
  check_xpu_tensor(sconv_cache, "sconv_cache");
  check_xpu_tensor(cache_indices, "cache_indices");
  check_xpu_tensor(num_accepted_tokens, "num_accepted_tokens");
  TORCH_CHECK(hidden_states.dim() == 2 || hidden_states.dim() == 3, "hidden_states must have shape [B*T, D] or [B, T, D]");
  TORCH_CHECK(sconv_cache.dim() == 3, "sconv_cache must have shape [slots, W-1, D]");
  TORCH_CHECK(cache_indices.scalar_type() == at::ScalarType::Int, "cache_indices must be int32");
  TORCH_CHECK(num_accepted_tokens.scalar_type() == at::ScalarType::Int, "num_accepted_tokens must be int32");
  check_sconv_dtype(hidden_states, sconv_cache, "sconv_cache");
  const int64_t B = cache_indices.numel();
  const int64_t D = sconv_cache.size(2);
  TORCH_CHECK(num_accepted_tokens.numel() == B, "num_accepted_tokens must have B entries");
  TORCH_CHECK(draft_token_num >= 0, "draft_token_num must be non-negative");
  TORCH_CHECK(sconv_cache.stride(2) == 1, "sconv_cache must be contiguous on D");
  TORCH_CHECK(hidden_states.stride(hidden_states.dim() - 1) == 1, "hidden_states must be contiguous on D");
  if (hidden_states.dim() == 2) {
    TORCH_CHECK(hidden_states.size(1) == D, "hidden_states.size(1) must equal cache D");
    TORCH_CHECK(hidden_states.size(0) >= B * draft_token_num, "hidden_states has too few rows");
  } else {
    TORCH_CHECK(hidden_states.size(0) >= B, "hidden_states.size(0) must be >= B");
    TORCH_CHECK(hidden_states.size(1) >= draft_token_num, "hidden_states.size(1) must be >= draft_token_num");
    TORCH_CHECK(hidden_states.size(2) == D, "hidden_states.size(2) must equal cache D");
  }
  if (do_tracking) {
    TORCH_CHECK(crossed.has_value() && track_step.has_value() && mamba_track_indices.has_value(), "tracking tensors are required when do_tracking=True");
    check_xpu_tensor(crossed.value(), "crossed");
    check_xpu_tensor(track_step.value(), "track_step");
    check_xpu_tensor(mamba_track_indices.value(), "mamba_track_indices");
    TORCH_CHECK(crossed.value().scalar_type() == at::ScalarType::Bool, "crossed must be bool");
    TORCH_CHECK(track_step.value().scalar_type() == at::ScalarType::Int, "track_step must be int32");
    TORCH_CHECK(mamba_track_indices.value().scalar_type() == at::ScalarType::Long, "mamba_track_indices must be int64");
    TORCH_CHECK(crossed.value().numel() >= B, "crossed must have at least B entries");
    TORCH_CHECK(track_step.value().numel() >= B, "track_step must have at least B entries");
    TORCH_CHECK(mamba_track_indices.value().numel() >= B, "mamba_track_indices must have at least B entries");
  }

  const int64_t hidden_stride_b =
      hidden_states.dim() == 2 ? draft_token_num * hidden_states.stride(0) : hidden_states.stride(0);
  const int64_t hidden_stride_t = hidden_states.dim() == 2 ? hidden_states.stride(0) : hidden_states.stride(1);
  const int64_t hidden_stride_d = hidden_states.dim() == 2 ? hidden_states.stride(1) : hidden_states.stride(2);

  auto queue = c10::xpu::getCurrentXPUStream().queue();
  const auto input_type = hidden_states.scalar_type();
  SYCL_DISPATCH_FLOATING_TYPES(at::ScalarType::Half, at::ScalarType::BFloat16, input_type, "inkling_draft_extend_sconv_cache", [&]() {
    DraftExtendParams<scalar_t> params{
        hidden_states.data_ptr<scalar_t>(),
        sconv_cache.data_ptr<scalar_t>(),
        cache_indices.data_ptr<int32_t>(),
        num_accepted_tokens.data_ptr<int32_t>(),
        do_tracking ? crossed.value().data_ptr<bool>() : nullptr,
        do_tracking ? track_step.value().data_ptr<int32_t>() : nullptr,
        do_tracking ? mamba_track_indices.value().data_ptr<int64_t>() : nullptr,
        B,
        D,
        sconv_cache.size(1),
        hidden_stride_b,
        hidden_stride_t,
        hidden_stride_d,
        sconv_cache.stride(0),
        sconv_cache.stride(1),
        sconv_cache.stride(2),
        do_tracking};
    launch_draft_extend<scalar_t>(queue, params);
  });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
inkling_fused_decode_sconv_metadata(int64_t B, const at::Tensor& cache_indices) {
  check_xpu_tensor(cache_indices, "cache_indices");
  TORCH_CHECK(cache_indices.scalar_type() == at::ScalarType::Int, "cache_indices must be int32");
  TORCH_CHECK(cache_indices.numel() >= B, "cache_indices must have at least B entries");
  auto int32_opts = cache_indices.options().dtype(at::ScalarType::Int);
  auto bool_opts = cache_indices.options().dtype(at::ScalarType::Bool);
  auto int64_opts = cache_indices.options().dtype(at::ScalarType::Long);
  at::Tensor query_start_loc = at::empty({B + 1}, int32_opts);
  at::Tensor has_initial_state = at::empty({B}, bool_opts);
  at::Tensor cache_mask = at::empty({B, 1, 1}, bool_opts);
  at::Tensor safe_idx = at::empty({B}, int64_opts);
  at::Tensor cu = at::empty({B + 1}, int64_opts);
  at::Tensor si = at::empty({B}, int32_opts);
  auto queue = c10::xpu::getCurrentXPUStream().queue();
  DecodeMetadataParams params{
      cache_indices.data_ptr<int32_t>(),
      query_start_loc.data_ptr<int32_t>(),
      has_initial_state.data_ptr<bool>(),
      cache_mask.data_ptr<bool>(),
      safe_idx.data_ptr<int64_t>(),
      cu.data_ptr<int64_t>(),
      si.data_ptr<int32_t>(),
      B};
  launch_decode_metadata(queue, params);
  return {query_start_loc, has_initial_state, cache_mask, safe_idx, cu, si};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
inkling_fused_extend_sconv_metadata(
    int64_t B,
    int64_t T,
    const at::Tensor& cache_indices,
    int64_t his_mode,
    const std::optional<at::Tensor>& extend_seq_lens,
    const std::optional<at::Tensor>& his_src,
    int64_t draft_token_num) {
  check_xpu_tensor(cache_indices, "cache_indices");
  TORCH_CHECK(cache_indices.scalar_type() == at::ScalarType::Int, "cache_indices must be int32");
  TORCH_CHECK(cache_indices.numel() >= B, "cache_indices must have at least B entries");
  TORCH_CHECK(his_mode >= kHisZeros && his_mode <= kHisOnes, "unsupported his_mode");
  TORCH_CHECK(T >= 0, "T must be non-negative");
  const bool is_verify = his_mode == kHisOnes;
  if (!is_verify) {
    TORCH_CHECK(extend_seq_lens.has_value(), "extend_seq_lens is required unless his_mode=HIS_ONES");
    check_xpu_tensor(extend_seq_lens.value(), "extend_seq_lens");
    TORCH_CHECK(extend_seq_lens.value().scalar_type() == at::ScalarType::Int, "extend_seq_lens must be int32");
    TORCH_CHECK(extend_seq_lens.value().numel() >= B, "extend_seq_lens must have at least B entries");
  }
  if (his_mode == kHisPrefix || his_mode == kHisSeqMinusExt) {
    TORCH_CHECK(his_src.has_value(), "his_src is required for this his_mode");
    check_xpu_tensor(his_src.value(), "his_src");
    TORCH_CHECK(his_src.value().scalar_type() == at::ScalarType::Int, "his_src must be int32");
    TORCH_CHECK(his_src.value().numel() >= B, "his_src must have at least B entries");
  }
  if (is_verify) {
    TORCH_CHECK(draft_token_num > 0, "draft_token_num must be positive for HIS_ONES");
  }

  auto int32_opts = cache_indices.options().dtype(at::ScalarType::Int);
  auto bool_opts = cache_indices.options().dtype(at::ScalarType::Bool);
  auto int64_opts = cache_indices.options().dtype(at::ScalarType::Long);
  at::Tensor query_start_loc = at::empty({B + 1}, int32_opts);
  at::Tensor has_initial_state = at::empty({B}, bool_opts);
  at::Tensor cache_mask = at::empty({B, 1, 1}, bool_opts);
  at::Tensor safe_idx = at::empty({B}, int64_opts);
  at::Tensor cu = at::empty({B + 1}, int64_opts);
  at::Tensor si = at::empty({T}, int32_opts);
  auto queue = c10::xpu::getCurrentXPUStream().queue();
  ExtendMetadataParams params{
      cache_indices.data_ptr<int32_t>(),
      (!is_verify && extend_seq_lens.has_value()) ? extend_seq_lens.value().data_ptr<int32_t>() : nullptr,
      (his_mode == kHisPrefix || his_mode == kHisSeqMinusExt) ? his_src.value().data_ptr<int32_t>() : nullptr,
      query_start_loc.data_ptr<int32_t>(),
      has_initial_state.data_ptr<bool>(),
      cache_mask.data_ptr<bool>(),
      safe_idx.data_ptr<int64_t>(),
      cu.data_ptr<int64_t>(),
      si.data_ptr<int32_t>(),
      B,
      T,
      draft_token_num,
      static_cast<int>(his_mode)};
  launch_extend_metadata(queue, params);
  return {query_start_loc, has_initial_state, cache_mask, safe_idx, cu, si};
}

at::Tensor inkling_track_conv_indices(
    const at::Tensor& query_start_loc,
    const at::Tensor& mamba_track_seqlens,
    const at::Tensor& extend_prefix_lens,
    int64_t width_minus_one,
    int64_t chunk_size,
    int64_t total_tokens) {
  check_xpu_tensor(query_start_loc, "query_start_loc");
  check_xpu_tensor(mamba_track_seqlens, "mamba_track_seqlens");
  check_xpu_tensor(extend_prefix_lens, "extend_prefix_lens");
  TORCH_CHECK(query_start_loc.scalar_type() == at::ScalarType::Int, "query_start_loc must be int32");
  TORCH_CHECK(mamba_track_seqlens.scalar_type() == at::ScalarType::Int, "mamba_track_seqlens must be int32");
  TORCH_CHECK(extend_prefix_lens.scalar_type() == at::ScalarType::Int, "extend_prefix_lens must be int32");
  TORCH_CHECK(width_minus_one >= 0, "width_minus_one must be non-negative");
  TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
  const int64_t B = mamba_track_seqlens.numel();
  TORCH_CHECK(query_start_loc.numel() >= B + 1, "query_start_loc must have at least B + 1 entries");
  TORCH_CHECK(extend_prefix_lens.numel() >= B, "extend_prefix_lens must have at least B entries");
  at::Tensor out = at::empty({B, width_minus_one}, query_start_loc.options().dtype(at::ScalarType::Int));
  auto queue = c10::xpu::getCurrentXPUStream().queue();
  TrackIndicesParams params{
      query_start_loc.data_ptr<int32_t>(),
      mamba_track_seqlens.data_ptr<int32_t>(),
      extend_prefix_lens.data_ptr<int32_t>(),
      out.data_ptr<int32_t>(),
      B,
      width_minus_one,
      chunk_size,
      total_tokens};
  launch_track_indices(queue, params);
  return out;
}

void inkling_save_intermediate_conv_windows(
    const at::Tensor& sconv_cache,
    const at::Tensor& hidden_states,
    const at::Tensor& cache_indices,
    at::Tensor& intermediate_out,
    int64_t batch_size,
    int64_t draft_token_num) {
  check_xpu_tensor(sconv_cache, "sconv_cache");
  check_xpu_tensor(hidden_states, "hidden_states");
  check_xpu_tensor(cache_indices, "cache_indices");
  check_xpu_tensor(intermediate_out, "intermediate_out");
  TORCH_CHECK(sconv_cache.dim() == 3, "sconv_cache must have shape [slots, W-1, D]");
  TORCH_CHECK(hidden_states.dim() == 2 || hidden_states.dim() == 3, "hidden_states must have shape [B*T, D] or [B, T, D]");
  TORCH_CHECK(intermediate_out.dim() == 4, "intermediate_out must have shape [B, T, W-1, D]");
  TORCH_CHECK(cache_indices.scalar_type() == at::ScalarType::Int, "cache_indices must be int32");
  check_sconv_dtype(sconv_cache, hidden_states, "hidden_states");
  check_sconv_dtype(sconv_cache, intermediate_out, "intermediate_out");
  TORCH_CHECK(batch_size >= 0, "batch_size must be non-negative");
  TORCH_CHECK(draft_token_num >= 0, "draft_token_num must be non-negative");
  TORCH_CHECK(cache_indices.numel() >= batch_size, "cache_indices must have at least batch_size entries");
  TORCH_CHECK(intermediate_out.size(0) >= batch_size, "intermediate_out.size(0) must be >= batch_size");
  TORCH_CHECK(intermediate_out.size(1) >= draft_token_num, "intermediate_out.size(1) must be >= draft_token_num");
  TORCH_CHECK(intermediate_out.size(2) == sconv_cache.size(1), "intermediate_out.size(2) must equal W-1");
  TORCH_CHECK(intermediate_out.size(3) == sconv_cache.size(2), "intermediate_out.size(3) must equal D");
  TORCH_CHECK(sconv_cache.stride(2) == 1, "sconv_cache must be contiguous on D");
  TORCH_CHECK(hidden_states.stride(hidden_states.dim() - 1) == 1, "hidden_states must be contiguous on D");
  TORCH_CHECK(intermediate_out.stride(3) == 1, "intermediate_out must be contiguous on D");
  const int64_t D = sconv_cache.size(2);
  if (hidden_states.dim() == 2) {
    TORCH_CHECK(hidden_states.size(1) == D, "hidden_states.size(1) must equal D");
    TORCH_CHECK(hidden_states.size(0) >= batch_size * draft_token_num, "hidden_states has too few rows");
  } else {
    TORCH_CHECK(hidden_states.size(0) >= batch_size, "hidden_states.size(0) must be >= batch_size");
    TORCH_CHECK(hidden_states.size(1) >= draft_token_num, "hidden_states.size(1) must be >= draft_token_num");
    TORCH_CHECK(hidden_states.size(2) == D, "hidden_states.size(2) must equal D");
  }

  const int64_t hidden_stride_b =
      hidden_states.dim() == 2 ? draft_token_num * hidden_states.stride(0) : hidden_states.stride(0);
  const int64_t hidden_stride_t = hidden_states.dim() == 2 ? hidden_states.stride(0) : hidden_states.stride(1);
  const int64_t hidden_stride_d = hidden_states.dim() == 2 ? hidden_states.stride(1) : hidden_states.stride(2);

  auto queue = c10::xpu::getCurrentXPUStream().queue();
  const auto input_type = sconv_cache.scalar_type();
  SYCL_DISPATCH_FLOATING_TYPES(at::ScalarType::Half, at::ScalarType::BFloat16, input_type, "inkling_save_intermediate_conv_windows", [&]() {
    SaveWindowsParams<scalar_t> params{
        sconv_cache.data_ptr<scalar_t>(),
        hidden_states.data_ptr<scalar_t>(),
        cache_indices.data_ptr<int32_t>(),
        intermediate_out.data_ptr<scalar_t>(),
        batch_size,
        draft_token_num,
        sconv_cache.size(1),
        D,
        sconv_cache.stride(0),
        sconv_cache.stride(1),
        sconv_cache.stride(2),
        hidden_stride_b,
        hidden_stride_t,
        hidden_stride_d,
        intermediate_out.stride(0),
        intermediate_out.stride(1),
        intermediate_out.stride(2),
        intermediate_out.stride(3)};
    launch_save_windows<scalar_t>(queue, params);
  });
}
