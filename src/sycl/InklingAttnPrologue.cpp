/* Copyright 2026 SGLang Team. All Rights Reserved.
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *
 * This file adapts the Inkling BMG attention-prologue kernels from
 * /data2/syk/cutlass-sycl/examples/15_bmg_attn_prologue for the
 * sgl-kernel XPU extension ABI.
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

constexpr int64_t kPadSlotId = -1;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kVecElems = 8;
constexpr int64_t kMXFP8Block = 32;
constexpr int64_t kThreads = 128;
constexpr float kE4M3Max = 448.0f;

template <typename scalar_t>
inline float to_float_device(scalar_t value) {
  return static_cast<float>(value);
}

template <typename scalar_t>
inline scalar_t from_float_device(float value) {
  return static_cast<scalar_t>(value);
}

template <typename scalar_t>
inline float round_to_scalar_float(float value) {
  return to_float_device(from_float_device<scalar_t>(value));
}

inline int64_t div_up_i64(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

inline float maybe_silu(float x, bool use_silu) {
  return use_silu ? x / (1.0f + sycl::native::exp(-x)) : x;
}

inline uint32_t float_bits_device(float x) {
  return sycl::bit_cast<uint32_t>(x);
}

inline int round_even_positive(float x) {
  const float floor_x = sycl::floor(x);
  const int base = static_cast<int>(floor_x);
  const float frac = x - floor_x;
  if (frac > 0.5f || (frac == 0.5f && (base & 1))) {
    return base + 1;
  }
  return base;
}

inline uint8_t float_to_e4m3fn_byte(float x) {
  if (!(x == x)) {
    return 0x7fu;
  }
  const uint8_t sign = x < 0.0f ? 0x80u : 0u;
  const float ax = sycl::fabs(x);
  if (ax == 0.0f) {
    return sign;
  }
  if (ax >= kE4M3Max) {
    return static_cast<uint8_t>(sign | 0x7eu);
  }

  constexpr float kMinNormal = 0x1p-6f;
  if (ax < kMinNormal) {
    const int mant = round_even_positive(ax * 512.0f);
    if (mant <= 0) {
      return sign;
    }
    if (mant >= 8) {
      return static_cast<uint8_t>(sign | 0x08u);
    }
    return static_cast<uint8_t>(sign | static_cast<uint8_t>(mant));
  }

  const uint32_t bits = float_bits_device(ax);
  int e = static_cast<int>((bits >> 23) & 0xffu) - 127;
  if (e > 8) {
    return static_cast<uint8_t>(sign | 0x7eu);
  }

  const uint32_t frac = bits & 0x007fffffu;
  int mant = static_cast<int>(frac >> 20);
  const uint32_t rem = frac & ((1u << 20) - 1u);
  constexpr uint32_t halfway = 1u << 19;
  if (rem > halfway || (rem == halfway && (mant & 1))) {
    ++mant;
  }
  if (mant == 8) {
    mant = 0;
    ++e;
  }
  if (e > 8) {
    return static_cast<uint8_t>(sign | 0x7eu);
  }

  int exp_field = e + 7;
  if (exp_field >= 15 && mant > 6) {
    exp_field = 15;
    mant = 6;
  }
  return static_cast<uint8_t>(sign | static_cast<uint8_t>((exp_field << 3) | mant));
}

inline uint8_t mxfp8_scale_byte(float amax, float& descale) {
  const float safe_amax = amax > 1.0e-30f ? amax : 1.0e-30f;
  float biased = sycl::ceil(sycl::log2(safe_amax / kE4M3Max)) + 127.0f;
  if (biased < 0.0f) {
    biased = 0.0f;
  }
  if (biased > 254.0f) {
    biased = 254.0f;
  }
  const int byte = static_cast<int>(biased);
  descale = sycl::exp2(static_cast<float>(byte - 127));
  return static_cast<uint8_t>(byte);
}

inline int64_t kv_scale_offset(int64_t kv_slot, int64_t ch, int64_t dkv, int64_t page_size) {
  const int64_t hkv = dkv / kHeadDim;
  const int64_t page_chunks = page_size / kMXFP8Block;
  const int64_t sf_dim = kHeadDim / kMXFP8Block;
  const int64_t page = kv_slot / page_size;
  const int64_t po = kv_slot % page_size;
  const int64_t head = ch / kHeadDim;
  const int64_t block = (ch % kHeadDim) / kMXFP8Block;
  return ((page * hkv + head) * (kMXFP8Block * page_chunks * sf_dim)) + ((po % kMXFP8Block) * (page_chunks * sf_dim)) +
         ((po / kMXFP8Block) * sf_dim) + block;
}

inline void store_mxfp8_block(const float* values, uint8_t* dst, uint8_t* sf) {
  float amax = 0.0f;
  for (int64_t i = 0; i < kMXFP8Block; ++i) {
    const float ax = sycl::fabs(values[i]);
    amax = ax > amax ? ax : amax;
  }

  float descale = 1.0f;
  *sf = mxfp8_scale_byte(amax, descale);
  for (int64_t i = 0; i < kMXFP8Block; ++i) {
    float scaled = values[i] / descale;
    if (scaled > kE4M3Max) {
      scaled = kE4M3Max;
    }
    if (scaled < -kE4M3Max) {
      scaled = -kE4M3Max;
    }
    dst[i] = float_to_e4m3fn_byte(scaled);
  }
}

inline void
store_q_mxfp8_head(const float* values, uint8_t* q_mxfp8, uint8_t* sfq, int64_t t, int64_t base_d, int64_t dq) {
  for (int64_t block = 0; block < kHeadDim / kMXFP8Block; ++block) {
    const int64_t ch = base_d + block * kMXFP8Block;
    store_mxfp8_block(
        values + block * kMXFP8Block, q_mxfp8 + t * dq + ch, sfq + t * (dq / kMXFP8Block) + ch / kMXFP8Block);
  }
}

inline void store_kv_mxfp8_head(
    const float* values,
    uint8_t* buf,
    uint8_t* sf,
    int64_t kv_slot,
    int64_t base_d,
    int64_t dkv,
    int64_t kv_buf_stride,
    int64_t page_size) {
  for (int64_t block = 0; block < kHeadDim / kMXFP8Block; ++block) {
    const int64_t ch = base_d + block * kMXFP8Block;
    store_mxfp8_block(
        values + block * kMXFP8Block,
        buf + kv_slot * kv_buf_stride + ch,
        sf + kv_scale_offset(kv_slot, ch, dkv, page_size));
  }
}

struct WeightLayout {
  int64_t W;
  int64_t stride_d;
  int64_t stride_w;
  bool current_first;
};

WeightLayout resolve_weight_layout(const at::Tensor& weight, int64_t D, int64_t W, const char* name) {
  TORCH_CHECK(weight.dim() == 2, name, " must have shape [D, W] or [W, D]");
  if (weight.size(0) == D && weight.size(1) == W) {
    return WeightLayout{W, weight.stride(0), weight.stride(1), false};
  }
  if (weight.size(0) == W && weight.size(1) == D) {
    return WeightLayout{W, weight.stride(1), weight.stride(0), true};
  }
  TORCH_CHECK(
      false,
      name,
      " must have shape [D, W] or [W, D] with D=",
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

template <typename scalar_t>
inline float weight_at(const scalar_t* weight, int64_t d, int64_t iw_oldest, const WeightLayout& layout) {
  const int64_t iw = layout.current_first ? (layout.W - 1 - iw_oldest) : iw_oldest;
  return to_float_device(weight[d * layout.stride_d + iw * layout.stride_w]);
}

void check_xpu_tensor(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_xpu(), name, " must be an XPU tensor");
}

void check_same_dtype(const at::Tensor& ref, const at::Tensor& other, const char* name) {
  TORCH_CHECK(other.scalar_type() == ref.scalar_type(), name, " dtype must match qkvr dtype");
}

void check_bool_tensor(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Bool, name, " must be bool");
}

void check_int32_tensor(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Int, name, " must be int32");
}

void check_int64_tensor(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Long, name, " must be int64");
}

void check_common_inputs(
    const at::Tensor& qkvr,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    const at::Tensor& k_weight,
    const at::Tensor& v_weight,
    const at::Tensor& q_gamma,
    const at::Tensor& k_gamma,
    const at::Tensor& loc,
    const at::Tensor& k_buf,
    const at::Tensor& v_buf,
    int64_t q_off,
    int64_t k_off,
    int64_t v_off,
    int64_t dq,
    int64_t dkv,
    bool check_kv_buf_dtype = true) {
  check_xpu_tensor(qkvr, "qkvr");
  check_xpu_tensor(k_cache, "k_cache");
  check_xpu_tensor(v_cache, "v_cache");
  check_xpu_tensor(k_weight, "k_weight");
  check_xpu_tensor(v_weight, "v_weight");
  check_xpu_tensor(q_gamma, "q_gamma");
  check_xpu_tensor(k_gamma, "k_gamma");
  check_xpu_tensor(loc, "loc");
  check_xpu_tensor(k_buf, "k_buf");
  check_xpu_tensor(v_buf, "v_buf");
  TORCH_CHECK(qkvr.dim() == 2, "qkvr must be a 2D row-major tensor/view");
  TORCH_CHECK(qkvr.stride(1) == 1, "qkvr must be contiguous on the last dimension");
  TORCH_CHECK(k_cache.dim() == 3, "k_cache must have shape [slots, W-1, Dkv]");
  TORCH_CHECK(v_cache.dim() == 3, "v_cache must have shape [slots, W-1, Dkv]");
  TORCH_CHECK(k_cache.size(1) == v_cache.size(1), "k/v cache windows must match");
  TORCH_CHECK(k_cache.size(2) == dkv && v_cache.size(2) == dkv, "cache D must match dkv");
  TORCH_CHECK(k_cache.stride(2) == 1 && v_cache.stride(2) == 1, "conv caches must be contiguous on D");
  TORCH_CHECK(dq > 0 && dkv > 0, "dq and dkv must be positive");
  TORCH_CHECK(dq % kHeadDim == 0 && dkv % kHeadDim == 0, "dq and dkv must be multiples of 128");
  TORCH_CHECK(q_off >= 0 && k_off >= 0 && v_off >= 0, "slice offsets must be non-negative");
  TORCH_CHECK(q_gamma.numel() >= kHeadDim, "q_gamma must have at least 128 elements");
  TORCH_CHECK(k_gamma.numel() >= kHeadDim, "k_gamma must have at least 128 elements");
  TORCH_CHECK(k_buf.dim() == 2 && v_buf.dim() == 2, "k_buf/v_buf must be flattened [slots, dkv] views");
  TORCH_CHECK(k_buf.size(1) == dkv && v_buf.size(1) == dkv, "k_buf/v_buf second dimension must equal dkv");
  TORCH_CHECK(k_buf.stride(1) == 1 && v_buf.stride(1) == 1, "k_buf/v_buf must be contiguous on D");
  check_same_dtype(qkvr, k_cache, "k_cache");
  check_same_dtype(qkvr, v_cache, "v_cache");
  check_same_dtype(qkvr, k_weight, "k_weight");
  check_same_dtype(qkvr, v_weight, "v_weight");
  check_same_dtype(qkvr, q_gamma, "q_gamma");
  check_same_dtype(qkvr, k_gamma, "k_gamma");
  if (check_kv_buf_dtype) {
    check_same_dtype(qkvr, k_buf, "k_buf");
    check_same_dtype(qkvr, v_buf, "v_buf");
  }
  check_int64_tensor(loc, "loc");
}

void check_log_tau(const at::Tensor& log_tau, int64_t T) {
  check_xpu_tensor(log_tau, "log_scaling_tau");
  TORCH_CHECK(log_tau.dim() <= 1 || log_tau.is_contiguous(), "log_scaling_tau must be flattenable and contiguous");
  if (log_tau.numel() == 0) {
    return;
  }
  TORCH_CHECK(log_tau.scalar_type() == at::ScalarType::Float, "log_scaling_tau must be float32");
  TORCH_CHECK(log_tau.is_contiguous(), "log_scaling_tau must be contiguous");
  TORCH_CHECK(log_tau.numel() >= T, "log_scaling_tau must have at least T entries");
}

void check_mxfp8_inputs(
    const at::Tensor& qkvr,
    const at::Tensor& loc,
    const at::Tensor& k_buf,
    const at::Tensor& v_buf,
    const at::Tensor& sfk,
    const at::Tensor& sfv,
    const at::Tensor& log_tau,
    int64_t T,
    int64_t dq,
    int64_t dkv,
    int64_t page_size) {
  check_xpu_tensor(qkvr, "qkvr");
  check_xpu_tensor(loc, "loc");
  check_xpu_tensor(k_buf, "k_buf");
  check_xpu_tensor(v_buf, "v_buf");
  check_xpu_tensor(sfk, "sfk");
  check_xpu_tensor(sfv, "sfv");
  check_log_tau(log_tau, T);
  TORCH_CHECK(k_buf.scalar_type() == at::ScalarType::Float8_e4m3fn, "MXFP8 k_buf must be float8_e4m3fn");
  TORCH_CHECK(v_buf.scalar_type() == at::ScalarType::Float8_e4m3fn, "MXFP8 v_buf must be float8_e4m3fn");
  TORCH_CHECK(sfk.scalar_type() == at::ScalarType::Byte, "sfk must be a uint8 view");
  TORCH_CHECK(sfv.scalar_type() == at::ScalarType::Byte, "sfv must be a uint8 view");
  TORCH_CHECK(k_buf.dim() == 2 && v_buf.dim() == 2, "MXFP8 k_buf/v_buf must be flattened [slots, dkv] views");
  TORCH_CHECK(k_buf.size(1) == dkv && v_buf.size(1) == dkv, "MXFP8 k_buf/v_buf second dimension must equal dkv");
  TORCH_CHECK(k_buf.stride(1) == 1 && v_buf.stride(1) == 1, "MXFP8 k_buf/v_buf must be contiguous on D");
  TORCH_CHECK(k_buf.size(0) == v_buf.size(0), "MXFP8 k_buf/v_buf slot counts must match");
  TORCH_CHECK(dq % kMXFP8Block == 0 && dkv % kMXFP8Block == 0, "MXFP8 dims must tile 32-element scale blocks");
  TORCH_CHECK(page_size > 0 && page_size % kMXFP8Block == 0, "page_size must be a positive multiple of 32");
  TORCH_CHECK(k_buf.size(0) % page_size == 0, "MXFP8 k_buf slots must be divisible by page_size");

  const int64_t pages = k_buf.size(0) / page_size;
  const int64_t hkv = dkv / kHeadDim;
  const int64_t page_chunks = page_size / kMXFP8Block;
  const int64_t sf_dim = kHeadDim / kMXFP8Block;
  TORCH_CHECK(sfk.dim() == 5 && sfv.dim() == 5, "SFK/SFV must be 5D [pages, Hkv, 32, page/32, 4]");
  TORCH_CHECK(
      sfk.size(0) == pages && sfv.size(0) == pages && sfk.size(1) == hkv && sfv.size(1) == hkv &&
          sfk.size(2) == kMXFP8Block && sfv.size(2) == kMXFP8Block && sfk.size(3) == page_chunks &&
          sfv.size(3) == page_chunks && sfk.size(4) == sf_dim && sfv.size(4) == sf_dim,
      "SFK/SFV must have shape [",
      pages,
      ", ",
      hkv,
      ", 32, ",
      page_chunks,
      ", 4]");
  TORCH_CHECK(sfk.is_contiguous() && sfv.is_contiguous(), "SFK/SFV must be contiguous");
}

template <typename scalar_t>
struct AttnPrologueVerifyParams {
  const scalar_t* qkvr;
  const scalar_t* k_cache;
  const scalar_t* v_cache;
  const int32_t* cache_indices;
  const bool* cache_mask;
  const scalar_t* k_weight;
  const scalar_t* v_weight;
  scalar_t* k_inter;
  scalar_t* v_inter;
  const scalar_t* q_gamma;
  const scalar_t* k_gamma;
  const float* log_tau;
  scalar_t* q_out;
  scalar_t* k_out;
  scalar_t* v_out;
  const int64_t* loc;
  scalar_t* k_buf;
  scalar_t* v_buf;
  uint8_t* q_mxfp8;
  uint8_t* k_buf_mxfp8;
  uint8_t* v_buf_mxfp8;
  uint8_t* sfq;
  uint8_t* sfk;
  uint8_t* sfv;
  WeightLayout k_layout;
  WeightLayout v_layout;
  float eps;
  int64_t T;
  int64_t B;
  int64_t draft_tokens;
  int64_t dq;
  int64_t dkv;
  int64_t qkvr_stride_t;
  int64_t q_off;
  int64_t k_off;
  int64_t v_off;
  int64_t cache_stride_slot;
  int64_t cache_stride_w;
  int64_t inter_stride_b;
  int64_t inter_stride_t;
  int64_t inter_stride_w;
  int64_t kv_buf_stride;
  int64_t loc_stride;
  int64_t page_size;
  bool use_silu;
  bool use_residual;
  bool do_store;
  bool mxfp8_quant;
};

template <typename scalar_t>
struct AttnPrologueDecodeParams {
  const scalar_t* qkvr;
  scalar_t* k_cache;
  scalar_t* v_cache;
  const int32_t* cache_indices;
  const bool* cache_mask;
  const scalar_t* k_weight;
  const scalar_t* v_weight;
  const bool* track_mask;
  const int64_t* track_indices;
  const scalar_t* q_gamma;
  const scalar_t* k_gamma;
  const float* log_tau;
  scalar_t* q_out;
  scalar_t* k_out;
  scalar_t* v_out;
  const int64_t* loc;
  scalar_t* k_buf;
  scalar_t* v_buf;
  uint8_t* q_mxfp8;
  uint8_t* k_buf_mxfp8;
  uint8_t* v_buf_mxfp8;
  uint8_t* sfq;
  uint8_t* sfk;
  uint8_t* sfv;
  WeightLayout k_layout;
  WeightLayout v_layout;
  float eps;
  int64_t T;
  int64_t dq;
  int64_t dkv;
  int64_t qkvr_stride_t;
  int64_t q_off;
  int64_t k_off;
  int64_t v_off;
  int64_t cache_stride_slot;
  int64_t cache_stride_w;
  int64_t kv_buf_stride;
  int64_t loc_stride;
  int64_t track_idx_stride;
  int64_t page_size;
  bool use_silu;
  bool use_residual;
  bool do_track;
  bool do_store;
  bool mxfp8_quant;
};

template <typename scalar_t>
struct AttnPrologueExtendParams {
  const scalar_t* qkvr;
  scalar_t* k_cache;
  scalar_t* v_cache;
  const int32_t* cache_indices;
  const bool* cache_mask;
  const bool* has_initial_state;
  const int64_t* cu;
  const int32_t* si;
  const scalar_t* k_weight;
  const scalar_t* v_weight;
  const int64_t* track_rows;
  const bool* track_mask;
  const int64_t* track_dst;
  const scalar_t* q_gamma;
  const scalar_t* k_gamma;
  const float* log_tau;
  scalar_t* q_out;
  scalar_t* k_out;
  scalar_t* v_out;
  const int64_t* loc;
  scalar_t* k_buf;
  scalar_t* v_buf;
  uint8_t* q_mxfp8;
  uint8_t* k_buf_mxfp8;
  uint8_t* v_buf_mxfp8;
  uint8_t* sfq;
  uint8_t* sfk;
  uint8_t* sfv;
  WeightLayout k_layout;
  WeightLayout v_layout;
  float eps;
  int64_t T;
  int64_t B;
  int64_t dq;
  int64_t dkv;
  int64_t qkvr_stride_t;
  int64_t q_off;
  int64_t k_off;
  int64_t v_off;
  int64_t cache_stride_slot;
  int64_t cache_stride_w;
  int64_t kv_buf_stride;
  int64_t loc_stride;
  int64_t track_rows_stride_b;
  int64_t track_rows_stride_w;
  int64_t track_dst_stride;
  int64_t page_size;
  bool use_silu;
  bool use_residual;
  bool do_track;
  bool do_store;
  bool do_cache_update;
  bool mxfp8_quant;
};

template <typename scalar_t, typename Params>
static inline void compute_q_head(const Params& p, int64_t t, int64_t head) {
  const int64_t base_d = head * kHeadDim;
  float ss = 0.0f;
  for (int64_t i = 0; i < kHeadDim; ++i) {
    const float x = to_float_device(p.qkvr[t * p.qkvr_stride_t + p.q_off + base_d + i]);
    ss += x * x;
  }
  const float inv = 1.0f / sycl::sqrt(ss / static_cast<float>(kHeadDim) + p.eps);
  float values[kHeadDim];
  const float tau = p.log_tau == nullptr ? 1.0f : p.log_tau[t];
  for (int64_t i = 0; i < kHeadDim; ++i) {
    const int64_t d = base_d + i;
    const float x = to_float_device(p.qkvr[t * p.qkvr_stride_t + p.q_off + d]);
    const float g = to_float_device(p.q_gamma[i]);
    scalar_t rounded = from_float_device<scalar_t>(x * inv * g);
    if (p.log_tau != nullptr) {
      rounded = from_float_device<scalar_t>(to_float_device(rounded) * tau);
    }
    values[i] = to_float_device(rounded);
    if (!p.mxfp8_quant) {
      p.q_out[t * p.dq + d] = rounded;
    }
  }
  if (p.mxfp8_quant) {
    store_q_mxfp8_head(values, p.q_mxfp8, p.sfq, t, base_d, p.dq);
  }
}

template <typename scalar_t>
static inline float verify_or_extend_conv_value(
    const scalar_t* qkvr,
    const scalar_t* cache,
    const scalar_t* weight,
    const WeightLayout& layout,
    int64_t t,
    int64_t bos,
    int64_t d,
    int64_t ch,
    int64_t slot,
    float cache_multiplier,
    int64_t qkvr_stride_t,
    int64_t x_off,
    int64_t cache_stride_slot,
    int64_t cache_stride_w,
    bool use_silu,
    bool use_residual) {
  const int64_t W = layout.W;
  const int64_t W1 = W - 1;
  const float xj = to_float_device(qkvr[t * qkvr_stride_t + x_off + d]);
  float acc = 0.0f;
  for (int64_t iw = 0; iw < W1; ++iw) {
    const int64_t shifted = t - W1 + iw;
    float tap = 0.0f;
    if (shifted >= bos) {
      tap = to_float_device(qkvr[shifted * qkvr_stride_t + x_off + d]);
    } else {
      const int64_t prefix_pos = shifted - bos + W1;
      if (prefix_pos >= 0 && cache_multiplier != 0.0f) {
        tap = cache_multiplier * to_float_device(cache[slot * cache_stride_slot + prefix_pos * cache_stride_w + ch]);
      }
    }
    acc += tap * weight_at(weight, d, iw, layout);
  }
  acc += xj * weight_at(weight, d, W1, layout);
  acc = maybe_silu(acc, use_silu);
  if (use_residual) {
    acc += xj;
  }
  return acc;
}

template <bool SaveWindows, typename scalar_t, typename Params>
static inline void
compute_kv_head_from_prefix(const Params& p, int64_t t, int64_t seq, int64_t bos, int64_t head, bool is_k) {
  const int64_t base_d = head * kHeadDim;
  const int32_t ci = p.cache_indices[seq];
  const bool valid = ci != kPadSlotId;
  const int64_t slot = valid ? static_cast<int64_t>(ci) : 0;
  const float cache_multiplier = (valid && p.cache_mask[seq]) ? 1.0f : 0.0f;
  const auto* cache = is_k ? p.k_cache : p.v_cache;
  const auto* weight = is_k ? p.k_weight : p.v_weight;
  const auto& layout = is_k ? p.k_layout : p.v_layout;
  const int64_t x_off = is_k ? p.k_off : p.v_off;
  auto* out = is_k ? p.k_out : p.v_out;

  float y[kHeadDim];
  float ss = 0.0f;
  for (int64_t i = 0; i < kHeadDim; ++i) {
    const int64_t d = base_d + i;
    const float acc = verify_or_extend_conv_value(
        p.qkvr,
        cache,
        weight,
        layout,
        t,
        bos,
        d,
        d,
        slot,
        cache_multiplier,
        p.qkvr_stride_t,
        x_off,
        p.cache_stride_slot,
        p.cache_stride_w,
        p.use_silu,
        p.use_residual);
    if (is_k) {
      y[i] = round_to_scalar_float<scalar_t>(acc);
      ss += y[i] * y[i];
    } else {
      y[i] = round_to_scalar_float<scalar_t>(acc);
      out[t * p.dkv + d] = from_float_device<scalar_t>(y[i]);
    }
  }

  if constexpr (SaveWindows) {
    const int64_t W1 = layout.W - 1;
    if (valid) {
      const int64_t tq = t - bos;
      auto* inter = is_k ? p.k_inter : p.v_inter;
      for (int64_t w = 0; w < W1; ++w) {
        const int64_t position = tq + 1 + w;
        for (int64_t i = 0; i < kHeadDim; ++i) {
          const int64_t d = base_d + i;
          scalar_t val = from_float_device<scalar_t>(0.0f);
          if (position < W1) {
            val = cache[slot * p.cache_stride_slot + position * p.cache_stride_w + d];
          } else {
            const int64_t g = bos + position - W1;
            val = p.qkvr[g * p.qkvr_stride_t + x_off + d];
          }
          inter[seq * p.inter_stride_b + tq * p.inter_stride_t + w * p.inter_stride_w + d] = val;
        }
      }
    }
  }

  if (is_k) {
    const float inv = 1.0f / sycl::sqrt(ss / static_cast<float>(kHeadDim) + p.eps);
    for (int64_t i = 0; i < kHeadDim; ++i) {
      const int64_t d = base_d + i;
      const float g = to_float_device(p.k_gamma[i]);
      y[i] = round_to_scalar_float<scalar_t>(y[i] * inv * g);
      out[t * p.dkv + d] = from_float_device<scalar_t>(y[i]);
    }
  }

  if (p.do_store) {
    const int64_t kv_slot = p.loc[t * p.loc_stride];
    if (kv_slot >= 0) {
      if (p.mxfp8_quant) {
        store_kv_mxfp8_head(
            y,
            is_k ? p.k_buf_mxfp8 : p.v_buf_mxfp8,
            is_k ? p.sfk : p.sfv,
            kv_slot,
            base_d,
            p.dkv,
            p.kv_buf_stride,
            p.page_size);
      } else {
        auto* buf = is_k ? p.k_buf : p.v_buf;
        for (int64_t i = 0; i < kHeadDim; ++i) {
          const int64_t d = base_d + i;
          buf[kv_slot * p.kv_buf_stride + d] = out[t * p.dkv + d];
        }
      }
    }
  }
}

template <typename scalar_t>
struct AttnPrologueVerifyKernel {
  AttnPrologueVerifyParams<scalar_t> p;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t linear = static_cast<int64_t>(item.get_global_linear_id());
    const int64_t hq = p.dq / kHeadDim;
    const int64_t hkv = p.dkv / kHeadDim;
    const int64_t roles = hq + 2 * hkv;
    const int64_t total = p.T * roles;
    if (linear >= total) {
      return;
    }
    const int64_t t = linear / roles;
    const int64_t role = linear - t * roles;
    if (role < hq) {
      compute_q_head<scalar_t>(p, t, role);
      return;
    }
    const int64_t kv_role = role - hq;
    const bool is_k = kv_role < hkv;
    const int64_t head = is_k ? kv_role : kv_role - hkv;
    const int64_t seq = t / p.draft_tokens;
    const int64_t bos = seq * p.draft_tokens;
    compute_kv_head_from_prefix<true, scalar_t>(p, t, seq, bos, head, is_k);
  }
};

template <typename scalar_t>
static inline float decode_conv_value(
    const scalar_t* qkvr,
    scalar_t* cache,
    const scalar_t* weight,
    const WeightLayout& layout,
    int64_t t,
    int64_t d,
    int64_t slot,
    float cache_multiplier,
    int64_t qkvr_stride_t,
    int64_t x_off,
    int64_t cache_stride_slot,
    int64_t cache_stride_w,
    bool use_silu,
    bool use_residual) {
  const int64_t W = layout.W;
  const int64_t W1 = W - 1;
  const float xj = to_float_device(qkvr[t * qkvr_stride_t + x_off + d]);
  float acc = 0.0f;
  for (int64_t iw = 0; iw < W1; ++iw) {
    const float h = to_float_device(cache[slot * cache_stride_slot + iw * cache_stride_w + d]);
    acc += h * cache_multiplier * weight_at(weight, d, iw, layout);
  }
  acc += xj * weight_at(weight, d, W1, layout);
  acc = maybe_silu(acc, use_silu);
  if (use_residual) {
    acc += xj;
  }
  return acc;
}

template <typename scalar_t>
struct AttnPrologueDecodeKernel {
  AttnPrologueDecodeParams<scalar_t> p;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t linear = static_cast<int64_t>(item.get_global_linear_id());
    const int64_t hq = p.dq / kHeadDim;
    const int64_t hkv = p.dkv / kHeadDim;
    const int64_t roles = hq + 2 * hkv;
    const int64_t total = p.T * roles;
    if (linear >= total) {
      return;
    }
    const int64_t t = linear / roles;
    const int64_t role = linear - t * roles;
    if (role < hq) {
      compute_q_head<scalar_t>(p, t, role);
      return;
    }

    const int64_t kv_role = role - hq;
    const bool is_k = kv_role < hkv;
    const int64_t head = is_k ? kv_role : kv_role - hkv;
    const int64_t base_d = head * kHeadDim;
    auto* cache = is_k ? p.k_cache : p.v_cache;
    const auto* weight = is_k ? p.k_weight : p.v_weight;
    const auto& layout = is_k ? p.k_layout : p.v_layout;
    const int64_t x_off = is_k ? p.k_off : p.v_off;
    auto* out = is_k ? p.k_out : p.v_out;
    const int64_t W1 = layout.W - 1;
    const int32_t ci = p.cache_indices[t];
    const bool valid = ci != kPadSlotId;
    const int64_t slot = valid ? static_cast<int64_t>(ci) : 0;
    const float cache_multiplier = (valid && p.cache_mask[t]) ? 1.0f : 0.0f;

    float y[kHeadDim];
    float ss = 0.0f;
    for (int64_t i = 0; i < kHeadDim; ++i) {
      const int64_t d = base_d + i;
      const float acc = decode_conv_value(
          p.qkvr,
          cache,
          weight,
          layout,
          t,
          d,
          slot,
          cache_multiplier,
          p.qkvr_stride_t,
          x_off,
          p.cache_stride_slot,
          p.cache_stride_w,
          p.use_silu,
          p.use_residual);
      if (is_k) {
        y[i] = round_to_scalar_float<scalar_t>(acc);
        ss += y[i] * y[i];
      } else {
        y[i] = round_to_scalar_float<scalar_t>(acc);
        out[t * p.dkv + d] = from_float_device<scalar_t>(y[i]);
      }
    }

    if (valid) {
      const bool do_track = p.do_track && p.track_mask[t];
      const int64_t track_slot = do_track ? p.track_indices[t * p.track_idx_stride] : -1;
      for (int64_t iw = 0; iw < W1; ++iw) {
        for (int64_t i = 0; i < kHeadDim; ++i) {
          const int64_t d = base_d + i;
          scalar_t nv;
          if (iw < W1 - 1) {
            nv = cache_multiplier != 0.0f ? cache[slot * p.cache_stride_slot + (iw + 1) * p.cache_stride_w + d]
                                          : from_float_device<scalar_t>(0.0f);
          } else {
            nv = p.qkvr[t * p.qkvr_stride_t + x_off + d];
          }
          cache[slot * p.cache_stride_slot + iw * p.cache_stride_w + d] = nv;
          if (do_track && track_slot >= 0) {
            cache[track_slot * p.cache_stride_slot + iw * p.cache_stride_w + d] = nv;
          }
        }
      }
    }

    if (is_k) {
      const float inv = 1.0f / sycl::sqrt(ss / static_cast<float>(kHeadDim) + p.eps);
      for (int64_t i = 0; i < kHeadDim; ++i) {
        const int64_t d = base_d + i;
        const float g = to_float_device(p.k_gamma[i]);
        y[i] = round_to_scalar_float<scalar_t>(y[i] * inv * g);
        out[t * p.dkv + d] = from_float_device<scalar_t>(y[i]);
      }
    }

    if (p.do_store && valid) {
      const int64_t kv_slot = p.loc[t * p.loc_stride];
      if (kv_slot >= 0) {
        if (p.mxfp8_quant) {
          store_kv_mxfp8_head(
              y,
              is_k ? p.k_buf_mxfp8 : p.v_buf_mxfp8,
              is_k ? p.sfk : p.sfv,
              kv_slot,
              base_d,
              p.dkv,
              p.kv_buf_stride,
              p.page_size);
        } else {
          auto* buf = is_k ? p.k_buf : p.v_buf;
          for (int64_t i = 0; i < kHeadDim; ++i) {
            const int64_t d = base_d + i;
            buf[kv_slot * p.kv_buf_stride + d] = out[t * p.dkv + d];
          }
        }
      }
    }
  }
};

template <typename scalar_t>
struct AttnPrologueExtendKernel {
  AttnPrologueExtendParams<scalar_t> p;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t linear = static_cast<int64_t>(item.get_global_linear_id());
    const int64_t hq = p.dq / kHeadDim;
    const int64_t hkv = p.dkv / kHeadDim;
    const int64_t roles = hq + 2 * hkv;
    const int64_t total = p.T * roles;
    if (linear >= total) {
      return;
    }
    const int64_t t = linear / roles;
    const int64_t role = linear - t * roles;
    if (role < hq) {
      compute_q_head<scalar_t>(p, t, role);
      return;
    }
    const int64_t kv_role = role - hq;
    const bool is_k = kv_role < hkv;
    const int64_t head = is_k ? kv_role : kv_role - hkv;
    const int64_t seq = static_cast<int64_t>(p.si[t]);
    const int64_t bos = p.cu[seq];
    compute_kv_head_from_prefix<false, scalar_t>(p, t, seq, bos, head, is_k);
  }
};

template <typename scalar_t>
struct AttnPrologueExtendUpdateKernel {
  AttnPrologueExtendParams<scalar_t> p;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t linear = static_cast<int64_t>(item.get_global_linear_id());
    const int64_t hkv = p.dkv / kHeadDim;
    const int64_t roles = 2 * hkv;
    const int64_t total = p.B * roles;
    if (linear >= total) {
      return;
    }
    const int64_t b = linear / roles;
    const int64_t role = linear - b * roles;
    const bool is_k = role < hkv;
    const int64_t head = is_k ? role : role - hkv;
    const int64_t base_d = head * kHeadDim;
    const int64_t x_off = is_k ? p.k_off : p.v_off;
    auto* cache = is_k ? p.k_cache : p.v_cache;
    const auto& layout = is_k ? p.k_layout : p.v_layout;
    const int64_t W1 = layout.W - 1;
    const int64_t qlen = p.cu[b + 1] - p.cu[b];
    const int32_t ci = p.cache_indices[b];

    if (ci != kPadSlotId && qlen > 0) {
      const int64_t slot = static_cast<int64_t>(ci);
      const bool has_init = p.has_initial_state[b];
      for (int64_t iw = 0; iw < W1; ++iw) {
        for (int64_t i = 0; i < kHeadDim; ++i) {
          const int64_t d = base_d + i;
          scalar_t nv;
          if (qlen >= W1 - iw) {
            const int64_t row = p.cu[b + 1] - W1 + iw;
            nv = p.qkvr[row * p.qkvr_stride_t + x_off + d];
          } else if (has_init) {
            nv = cache[slot * p.cache_stride_slot + (iw + qlen) * p.cache_stride_w + d];
          } else {
            nv = from_float_device<scalar_t>(0.0f);
          }
          cache[slot * p.cache_stride_slot + iw * p.cache_stride_w + d] = nv;
        }
      }
    }

    if (p.do_track && p.track_mask[b]) {
      const int64_t dst = p.track_dst[b * p.track_dst_stride];
      if (dst >= 0) {
        for (int64_t iw = 0; iw < W1; ++iw) {
          const int64_t row = p.track_rows[b * p.track_rows_stride_b + iw * p.track_rows_stride_w];
          for (int64_t i = 0; i < kHeadDim; ++i) {
            const int64_t d = base_d + i;
            cache[dst * p.cache_stride_slot + iw * p.cache_stride_w + d] = p.qkvr[row * p.qkvr_stride_t + x_off + d];
          }
        }
      }
    }
  }
};

template <typename scalar_t>
void launch_verify(sycl::queue& q, AttnPrologueVerifyParams<scalar_t> const& params) {
  const int64_t roles = params.dq / kHeadDim + 2 * (params.dkv / kHeadDim);
  const int64_t total = params.T * roles;
  if (total == 0) {
    return;
  }
  const int64_t global = div_up_i64(total, kThreads) * kThreads;
  AttnPrologueVerifyKernel<scalar_t> kernel{params};
  q.parallel_for<AttnPrologueVerifyKernel<scalar_t>>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kThreads)), kernel);
}

template <typename scalar_t>
void launch_decode(sycl::queue& q, AttnPrologueDecodeParams<scalar_t> const& params) {
  const int64_t roles = params.dq / kHeadDim + 2 * (params.dkv / kHeadDim);
  const int64_t total = params.T * roles;
  if (total == 0) {
    return;
  }
  const int64_t global = div_up_i64(total, kThreads) * kThreads;
  AttnPrologueDecodeKernel<scalar_t> kernel{params};
  q.parallel_for<AttnPrologueDecodeKernel<scalar_t>>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kThreads)), kernel);
}

template <typename scalar_t>
void launch_extend(sycl::queue& q, AttnPrologueExtendParams<scalar_t> const& params) {
  const int64_t roles = params.dq / kHeadDim + 2 * (params.dkv / kHeadDim);
  const int64_t total = params.T * roles;
  if (total != 0) {
    const int64_t global = div_up_i64(total, kThreads) * kThreads;
    AttnPrologueExtendKernel<scalar_t> kernel{params};
    q.parallel_for<AttnPrologueExtendKernel<scalar_t>>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kThreads)), kernel);
  }

  if (!params.do_cache_update) {
    return;
  }
  const int64_t update_total = params.B * 2 * (params.dkv / kHeadDim);
  if (update_total == 0) {
    return;
  }
  const int64_t update_global = div_up_i64(update_total, kThreads) * kThreads;
  AttnPrologueExtendUpdateKernel<scalar_t> update_kernel{params};
  q.parallel_for<AttnPrologueExtendUpdateKernel<scalar_t>>(
      sycl::nd_range<1>(sycl::range<1>(update_global), sycl::range<1>(kThreads)), update_kernel);
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> inkling_attn_prologue_verify(
    const at::Tensor& qkvr,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    const at::Tensor& cache_indices,
    const at::Tensor& cache_mask,
    const at::Tensor& k_weight,
    const at::Tensor& v_weight,
    at::Tensor& k_inter,
    at::Tensor& v_inter,
    const at::Tensor& q_gamma,
    const at::Tensor& k_gamma,
    double eps,
    const at::Tensor& loc,
    at::Tensor& k_buf,
    at::Tensor& v_buf,
    int64_t q_off,
    int64_t k_off,
    int64_t v_off,
    int64_t dq,
    int64_t dkv,
    int64_t draft_token_num,
    bool silu_activation,
    bool use_residual,
    bool do_store,
    const at::Tensor& log_tau) {
  check_common_inputs(
      qkvr, k_cache, v_cache, k_weight, v_weight, q_gamma, k_gamma, loc, k_buf, v_buf, q_off, k_off, v_off, dq, dkv);
  check_xpu_tensor(cache_indices, "cache_indices");
  check_xpu_tensor(cache_mask, "cache_mask");
  check_xpu_tensor(k_inter, "k_inter");
  check_xpu_tensor(v_inter, "v_inter");
  check_int32_tensor(cache_indices, "cache_indices");
  check_bool_tensor(cache_mask, "cache_mask");
  check_same_dtype(qkvr, k_inter, "k_inter");
  check_same_dtype(qkvr, v_inter, "v_inter");
  TORCH_CHECK(draft_token_num > 0, "draft_token_num must be positive");
  const int64_t T = qkvr.size(0);
  const int64_t B = cache_indices.numel();
  check_log_tau(log_tau, T);
  TORCH_CHECK(T == B * draft_token_num, "qkvr.size(0) must equal cache_indices.numel() * draft_token_num");
  TORCH_CHECK(cache_mask.numel() >= B, "cache_mask must have at least B entries");
  TORCH_CHECK(loc.numel() >= T, "loc must have at least T entries");
  TORCH_CHECK(k_inter.dim() == 4 && v_inter.dim() == 4, "k_inter/v_inter must have shape [B, q, W-1, dkv]");
  TORCH_CHECK(k_inter.size(0) >= B && v_inter.size(0) >= B, "inter buffers must cover B");
  TORCH_CHECK(k_inter.size(1) >= draft_token_num && v_inter.size(1) >= draft_token_num, "inter buffers must cover q");
  TORCH_CHECK(k_inter.size(2) == k_cache.size(1) && v_inter.size(2) == v_cache.size(1), "inter W-1 must match cache");
  TORCH_CHECK(k_inter.size(3) == dkv && v_inter.size(3) == dkv, "inter D must match dkv");
  TORCH_CHECK(k_inter.stride(3) == 1 && v_inter.stride(3) == 1, "inter buffers must be contiguous on D");

  const int64_t W = k_cache.size(1) + 1;
  const WeightLayout k_layout = resolve_weight_layout(k_weight, dkv, W, "k_weight");
  const WeightLayout v_layout = resolve_weight_layout(v_weight, dkv, W, "v_weight");
  at::Tensor q_out = at::empty_strided({T, dq}, {dq, 1}, qkvr.options());
  at::Tensor k_out = at::empty_strided({T, dkv}, {dkv, 1}, qkvr.options());
  at::Tensor v_out = at::empty_strided({T, dkv}, {dkv, 1}, qkvr.options());

  auto queue = c10::xpu::getCurrentXPUStream().queue();
  const auto input_type = qkvr.scalar_type();
  SYCL_DISPATCH_FLOATING_TYPES(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input_type,
      "inkling_attn_prologue_verify",
      [&]() -> std::tuple<at::Tensor, at::Tensor, at::Tensor> {
        AttnPrologueVerifyParams<scalar_t> params{
            qkvr.data_ptr<scalar_t>(),
            k_cache.data_ptr<scalar_t>(),
            v_cache.data_ptr<scalar_t>(),
            cache_indices.data_ptr<int32_t>(),
            cache_mask.data_ptr<bool>(),
            k_weight.data_ptr<scalar_t>(),
            v_weight.data_ptr<scalar_t>(),
            k_inter.data_ptr<scalar_t>(),
            v_inter.data_ptr<scalar_t>(),
            q_gamma.data_ptr<scalar_t>(),
            k_gamma.data_ptr<scalar_t>(),
            log_tau.numel() > 0 ? log_tau.data_ptr<float>() : nullptr,
            q_out.data_ptr<scalar_t>(),
            k_out.data_ptr<scalar_t>(),
            v_out.data_ptr<scalar_t>(),
            loc.data_ptr<int64_t>(),
            k_buf.data_ptr<scalar_t>(),
            v_buf.data_ptr<scalar_t>(),
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            k_layout,
            v_layout,
            static_cast<float>(eps),
            T,
            B,
            draft_token_num,
            dq,
            dkv,
            qkvr.stride(0),
            q_off,
            k_off,
            v_off,
            k_cache.stride(0),
            k_cache.stride(1),
            k_inter.stride(0),
            k_inter.stride(1),
            k_inter.stride(2),
            k_buf.stride(0),
            loc.stride(0),
            0,
            silu_activation,
            use_residual,
            do_store,
            false};
        launch_verify<scalar_t>(queue, params);
        return {q_out, k_out, v_out};
      });
  return {q_out, k_out, v_out};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> inkling_attn_prologue_decode(
    const at::Tensor& qkvr,
    at::Tensor& k_cache,
    at::Tensor& v_cache,
    const at::Tensor& cache_indices,
    const at::Tensor& cache_mask,
    const at::Tensor& k_weight,
    const at::Tensor& v_weight,
    const std::optional<at::Tensor>& track_mask,
    const std::optional<at::Tensor>& track_indices,
    const at::Tensor& q_gamma,
    const at::Tensor& k_gamma,
    double eps,
    const at::Tensor& loc,
    at::Tensor& k_buf,
    at::Tensor& v_buf,
    int64_t q_off,
    int64_t k_off,
    int64_t v_off,
    int64_t dq,
    int64_t dkv,
    bool silu_activation,
    bool use_residual,
    bool do_store,
    const at::Tensor& log_tau) {
  check_common_inputs(
      qkvr, k_cache, v_cache, k_weight, v_weight, q_gamma, k_gamma, loc, k_buf, v_buf, q_off, k_off, v_off, dq, dkv);
  check_xpu_tensor(cache_indices, "cache_indices");
  check_xpu_tensor(cache_mask, "cache_mask");
  check_int32_tensor(cache_indices, "cache_indices");
  check_bool_tensor(cache_mask, "cache_mask");
  const int64_t T = qkvr.size(0);
  check_log_tau(log_tau, T);
  TORCH_CHECK(cache_indices.numel() >= T, "cache_indices must have at least T entries");
  TORCH_CHECK(cache_mask.numel() >= T, "cache_mask must have at least T entries");
  TORCH_CHECK(loc.numel() >= T, "loc must have at least T entries");
  const bool do_track = track_mask.has_value();
  if (do_track) {
    TORCH_CHECK(track_indices.has_value(), "track_indices is required when track_mask is provided");
    check_xpu_tensor(track_mask.value(), "track_mask");
    check_xpu_tensor(track_indices.value(), "track_indices");
    check_bool_tensor(track_mask.value(), "track_mask");
    check_int64_tensor(track_indices.value(), "track_indices");
    TORCH_CHECK(track_mask.value().numel() >= T, "track_mask must have at least T entries");
    TORCH_CHECK(track_indices.value().numel() >= T, "track_indices must have at least T entries");
  }

  const int64_t W = k_cache.size(1) + 1;
  const WeightLayout k_layout = resolve_weight_layout(k_weight, dkv, W, "k_weight");
  const WeightLayout v_layout = resolve_weight_layout(v_weight, dkv, W, "v_weight");
  at::Tensor q_out = at::empty_strided({T, dq}, {dq, 1}, qkvr.options());
  at::Tensor k_out = at::empty_strided({T, dkv}, {dkv, 1}, qkvr.options());
  at::Tensor v_out = at::empty_strided({T, dkv}, {dkv, 1}, qkvr.options());

  auto queue = c10::xpu::getCurrentXPUStream().queue();
  const auto input_type = qkvr.scalar_type();
  SYCL_DISPATCH_FLOATING_TYPES(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input_type,
      "inkling_attn_prologue_decode",
      [&]() -> std::tuple<at::Tensor, at::Tensor, at::Tensor> {
        AttnPrologueDecodeParams<scalar_t> params{
            qkvr.data_ptr<scalar_t>(),
            k_cache.data_ptr<scalar_t>(),
            v_cache.data_ptr<scalar_t>(),
            cache_indices.data_ptr<int32_t>(),
            cache_mask.data_ptr<bool>(),
            k_weight.data_ptr<scalar_t>(),
            v_weight.data_ptr<scalar_t>(),
            do_track ? track_mask.value().data_ptr<bool>() : nullptr,
            do_track ? track_indices.value().data_ptr<int64_t>() : nullptr,
            q_gamma.data_ptr<scalar_t>(),
            k_gamma.data_ptr<scalar_t>(),
            log_tau.numel() > 0 ? log_tau.data_ptr<float>() : nullptr,
            q_out.data_ptr<scalar_t>(),
            k_out.data_ptr<scalar_t>(),
            v_out.data_ptr<scalar_t>(),
            loc.data_ptr<int64_t>(),
            k_buf.data_ptr<scalar_t>(),
            v_buf.data_ptr<scalar_t>(),
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            k_layout,
            v_layout,
            static_cast<float>(eps),
            T,
            dq,
            dkv,
            qkvr.stride(0),
            q_off,
            k_off,
            v_off,
            k_cache.stride(0),
            k_cache.stride(1),
            k_buf.stride(0),
            loc.stride(0),
            do_track ? track_indices.value().stride(0) : 0,
            0,
            silu_activation,
            use_residual,
            do_track,
            do_store,
            false};
        launch_decode<scalar_t>(queue, params);
        return {q_out, k_out, v_out};
      });
  return {q_out, k_out, v_out};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> inkling_attn_prologue_extend(
    const at::Tensor& qkvr,
    at::Tensor& k_cache,
    at::Tensor& v_cache,
    const at::Tensor& cache_indices,
    const at::Tensor& cache_mask,
    const at::Tensor& has_initial_state,
    const at::Tensor& cu,
    const at::Tensor& si,
    const at::Tensor& k_weight,
    const at::Tensor& v_weight,
    const std::optional<at::Tensor>& track_rows,
    const std::optional<at::Tensor>& track_mask,
    const std::optional<at::Tensor>& track_dst,
    const at::Tensor& q_gamma,
    const at::Tensor& k_gamma,
    double eps,
    const at::Tensor& loc,
    at::Tensor& k_buf,
    at::Tensor& v_buf,
    int64_t q_off,
    int64_t k_off,
    int64_t v_off,
    int64_t dq,
    int64_t dkv,
    bool silu_activation,
    bool use_residual,
    bool do_store,
    bool do_cache_update,
    const at::Tensor& log_tau) {
  check_common_inputs(
      qkvr, k_cache, v_cache, k_weight, v_weight, q_gamma, k_gamma, loc, k_buf, v_buf, q_off, k_off, v_off, dq, dkv);
  check_xpu_tensor(cache_indices, "cache_indices");
  check_xpu_tensor(cache_mask, "cache_mask");
  check_xpu_tensor(has_initial_state, "has_initial_state");
  check_xpu_tensor(cu, "cu");
  check_xpu_tensor(si, "si");
  check_int32_tensor(cache_indices, "cache_indices");
  check_bool_tensor(cache_mask, "cache_mask");
  check_bool_tensor(has_initial_state, "has_initial_state");
  check_int64_tensor(cu, "cu");
  check_int32_tensor(si, "si");
  const int64_t T = qkvr.size(0);
  const int64_t B = cache_indices.numel();
  check_log_tau(log_tau, T);
  TORCH_CHECK(cache_mask.numel() >= B, "cache_mask must have at least B entries");
  TORCH_CHECK(has_initial_state.numel() >= B, "has_initial_state must have at least B entries");
  TORCH_CHECK(cu.numel() >= B + 1, "cu must have at least B + 1 entries");
  TORCH_CHECK(si.numel() >= T, "si must have at least T entries");
  TORCH_CHECK(loc.numel() >= T, "loc must have at least T entries");
  const bool do_track = track_mask.has_value() && track_mask.value().numel() > 0;
  if (do_track) {
    TORCH_CHECK(
        track_rows.has_value() && track_dst.has_value(), "track_rows and track_dst are required with track_mask");
    check_xpu_tensor(track_rows.value(), "track_rows");
    check_xpu_tensor(track_mask.value(), "track_mask");
    check_xpu_tensor(track_dst.value(), "track_dst");
    check_int64_tensor(track_rows.value(), "track_rows");
    check_bool_tensor(track_mask.value(), "track_mask");
    check_int64_tensor(track_dst.value(), "track_dst");
    TORCH_CHECK(track_rows.value().dim() == 2, "track_rows must have shape [B, W-1]");
    TORCH_CHECK(track_rows.value().size(0) >= B, "track_rows must cover B");
    TORCH_CHECK(track_mask.value().numel() >= B, "track_mask must cover B");
    TORCH_CHECK(track_dst.value().numel() >= B, "track_dst must cover B");
  }

  const int64_t W = k_cache.size(1) + 1;
  const WeightLayout k_layout = resolve_weight_layout(k_weight, dkv, W, "k_weight");
  const WeightLayout v_layout = resolve_weight_layout(v_weight, dkv, W, "v_weight");
  if (do_track) {
    TORCH_CHECK(track_rows.value().size(1) == W - 1, "track_rows.size(1) must equal W-1");
  }
  at::Tensor q_out = at::empty_strided({T, dq}, {dq, 1}, qkvr.options());
  at::Tensor k_out = at::empty_strided({T, dkv}, {dkv, 1}, qkvr.options());
  at::Tensor v_out = at::empty_strided({T, dkv}, {dkv, 1}, qkvr.options());

  auto queue = c10::xpu::getCurrentXPUStream().queue();
  const auto input_type = qkvr.scalar_type();
  SYCL_DISPATCH_FLOATING_TYPES(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input_type,
      "inkling_attn_prologue_extend",
      [&]() -> std::tuple<at::Tensor, at::Tensor, at::Tensor> {
        AttnPrologueExtendParams<scalar_t> params{
            qkvr.data_ptr<scalar_t>(),
            k_cache.data_ptr<scalar_t>(),
            v_cache.data_ptr<scalar_t>(),
            cache_indices.data_ptr<int32_t>(),
            cache_mask.data_ptr<bool>(),
            has_initial_state.data_ptr<bool>(),
            cu.data_ptr<int64_t>(),
            si.data_ptr<int32_t>(),
            k_weight.data_ptr<scalar_t>(),
            v_weight.data_ptr<scalar_t>(),
            do_track ? track_rows.value().data_ptr<int64_t>() : nullptr,
            do_track ? track_mask.value().data_ptr<bool>() : nullptr,
            do_track ? track_dst.value().data_ptr<int64_t>() : nullptr,
            q_gamma.data_ptr<scalar_t>(),
            k_gamma.data_ptr<scalar_t>(),
            log_tau.numel() > 0 ? log_tau.data_ptr<float>() : nullptr,
            q_out.data_ptr<scalar_t>(),
            k_out.data_ptr<scalar_t>(),
            v_out.data_ptr<scalar_t>(),
            loc.data_ptr<int64_t>(),
            k_buf.data_ptr<scalar_t>(),
            v_buf.data_ptr<scalar_t>(),
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            k_layout,
            v_layout,
            static_cast<float>(eps),
            T,
            B,
            dq,
            dkv,
            qkvr.stride(0),
            q_off,
            k_off,
            v_off,
            k_cache.stride(0),
            k_cache.stride(1),
            k_buf.stride(0),
            loc.stride(0),
            do_track ? track_rows.value().stride(0) : 0,
            do_track ? track_rows.value().stride(1) : 0,
            do_track ? track_dst.value().stride(0) : 0,
            0,
            silu_activation,
            use_residual,
            do_track,
            do_store,
            do_cache_update,
            false};
        launch_extend<scalar_t>(queue, params);
        return {q_out, k_out, v_out};
      });
  return {q_out, k_out, v_out};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> inkling_attn_prologue_verify_mxfp8(
    const at::Tensor& qkvr,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    const at::Tensor& cache_indices,
    const at::Tensor& cache_mask,
    const at::Tensor& k_weight,
    const at::Tensor& v_weight,
    at::Tensor& k_inter,
    at::Tensor& v_inter,
    const at::Tensor& q_gamma,
    const at::Tensor& k_gamma,
    double eps,
    const at::Tensor& loc,
    at::Tensor& k_buf,
    at::Tensor& v_buf,
    at::Tensor& sfk,
    at::Tensor& sfv,
    int64_t q_off,
    int64_t k_off,
    int64_t v_off,
    int64_t dq,
    int64_t dkv,
    int64_t draft_token_num,
    bool silu_activation,
    bool use_residual,
    bool do_store,
    int64_t page_size,
    const at::Tensor& log_tau) {
  check_common_inputs(
      qkvr,
      k_cache,
      v_cache,
      k_weight,
      v_weight,
      q_gamma,
      k_gamma,
      loc,
      k_buf,
      v_buf,
      q_off,
      k_off,
      v_off,
      dq,
      dkv,
      false);
  check_xpu_tensor(cache_indices, "cache_indices");
  check_xpu_tensor(cache_mask, "cache_mask");
  check_xpu_tensor(k_inter, "k_inter");
  check_xpu_tensor(v_inter, "v_inter");
  check_int32_tensor(cache_indices, "cache_indices");
  check_bool_tensor(cache_mask, "cache_mask");
  check_same_dtype(qkvr, k_inter, "k_inter");
  check_same_dtype(qkvr, v_inter, "v_inter");
  TORCH_CHECK(draft_token_num > 0, "draft_token_num must be positive");
  const int64_t T = qkvr.size(0);
  const int64_t B = cache_indices.numel();
  check_mxfp8_inputs(qkvr, loc, k_buf, v_buf, sfk, sfv, log_tau, T, dq, dkv, page_size);
  TORCH_CHECK(T == B * draft_token_num, "qkvr.size(0) must equal cache_indices.numel() * draft_token_num");
  TORCH_CHECK(cache_mask.numel() >= B, "cache_mask must have at least B entries");
  TORCH_CHECK(loc.numel() >= T, "loc must have at least T entries");
  TORCH_CHECK(k_inter.dim() == 4 && v_inter.dim() == 4, "k_inter/v_inter must have shape [B, q, W-1, dkv]");
  TORCH_CHECK(k_inter.size(0) >= B && v_inter.size(0) >= B, "inter buffers must cover B");
  TORCH_CHECK(k_inter.size(1) >= draft_token_num && v_inter.size(1) >= draft_token_num, "inter buffers must cover q");
  TORCH_CHECK(k_inter.size(2) == k_cache.size(1) && v_inter.size(2) == v_cache.size(1), "inter W-1 must match cache");
  TORCH_CHECK(k_inter.size(3) == dkv && v_inter.size(3) == dkv, "inter D must match dkv");
  TORCH_CHECK(k_inter.stride(3) == 1 && v_inter.stride(3) == 1, "inter buffers must be contiguous on D");

  const int64_t W = k_cache.size(1) + 1;
  const WeightLayout k_layout = resolve_weight_layout(k_weight, dkv, W, "k_weight");
  const WeightLayout v_layout = resolve_weight_layout(v_weight, dkv, W, "v_weight");
  at::Tensor q_out = at::empty_strided({T, dq}, {dq, 1}, qkvr.options().dtype(at::ScalarType::Byte));
  at::Tensor q_scale = at::empty_strided(
      {T, dq / kHeadDim, kHeadDim / kMXFP8Block},
      {dq / kMXFP8Block, kHeadDim / kMXFP8Block, 1},
      qkvr.options().dtype(at::ScalarType::Byte));
  at::Tensor k_out = at::empty_strided({T, dkv}, {dkv, 1}, qkvr.options());
  at::Tensor v_out = at::empty_strided({T, dkv}, {dkv, 1}, qkvr.options());

  auto queue = c10::xpu::getCurrentXPUStream().queue();
  const auto input_type = qkvr.scalar_type();
  SYCL_DISPATCH_FLOATING_TYPES(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input_type,
      "inkling_attn_prologue_verify_mxfp8",
      [&]() -> std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> {
        AttnPrologueVerifyParams<scalar_t> params{
            qkvr.data_ptr<scalar_t>(),
            k_cache.data_ptr<scalar_t>(),
            v_cache.data_ptr<scalar_t>(),
            cache_indices.data_ptr<int32_t>(),
            cache_mask.data_ptr<bool>(),
            k_weight.data_ptr<scalar_t>(),
            v_weight.data_ptr<scalar_t>(),
            k_inter.data_ptr<scalar_t>(),
            v_inter.data_ptr<scalar_t>(),
            q_gamma.data_ptr<scalar_t>(),
            k_gamma.data_ptr<scalar_t>(),
            log_tau.numel() > 0 ? log_tau.data_ptr<float>() : nullptr,
            nullptr,
            k_out.data_ptr<scalar_t>(),
            v_out.data_ptr<scalar_t>(),
            loc.data_ptr<int64_t>(),
            nullptr,
            nullptr,
            q_out.data_ptr<uint8_t>(),
            static_cast<uint8_t*>(k_buf.data_ptr()),
            static_cast<uint8_t*>(v_buf.data_ptr()),
            q_scale.data_ptr<uint8_t>(),
            sfk.data_ptr<uint8_t>(),
            sfv.data_ptr<uint8_t>(),
            k_layout,
            v_layout,
            static_cast<float>(eps),
            T,
            B,
            draft_token_num,
            dq,
            dkv,
            qkvr.stride(0),
            q_off,
            k_off,
            v_off,
            k_cache.stride(0),
            k_cache.stride(1),
            k_inter.stride(0),
            k_inter.stride(1),
            k_inter.stride(2),
            k_buf.stride(0),
            loc.stride(0),
            page_size,
            silu_activation,
            use_residual,
            do_store,
            true};
        launch_verify<scalar_t>(queue, params);
        return {q_out, k_out, v_out, q_scale};
      });
  return {q_out, k_out, v_out, q_scale};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> inkling_attn_prologue_decode_mxfp8(
    const at::Tensor& qkvr,
    at::Tensor& k_cache,
    at::Tensor& v_cache,
    const at::Tensor& cache_indices,
    const at::Tensor& cache_mask,
    const at::Tensor& k_weight,
    const at::Tensor& v_weight,
    const std::optional<at::Tensor>& track_mask,
    const std::optional<at::Tensor>& track_indices,
    const at::Tensor& q_gamma,
    const at::Tensor& k_gamma,
    double eps,
    const at::Tensor& loc,
    at::Tensor& k_buf,
    at::Tensor& v_buf,
    at::Tensor& sfk,
    at::Tensor& sfv,
    int64_t q_off,
    int64_t k_off,
    int64_t v_off,
    int64_t dq,
    int64_t dkv,
    bool silu_activation,
    bool use_residual,
    bool do_store,
    int64_t page_size,
    const at::Tensor& log_tau) {
  check_common_inputs(
      qkvr,
      k_cache,
      v_cache,
      k_weight,
      v_weight,
      q_gamma,
      k_gamma,
      loc,
      k_buf,
      v_buf,
      q_off,
      k_off,
      v_off,
      dq,
      dkv,
      false);
  check_xpu_tensor(cache_indices, "cache_indices");
  check_xpu_tensor(cache_mask, "cache_mask");
  check_int32_tensor(cache_indices, "cache_indices");
  check_bool_tensor(cache_mask, "cache_mask");
  const int64_t T = qkvr.size(0);
  check_mxfp8_inputs(qkvr, loc, k_buf, v_buf, sfk, sfv, log_tau, T, dq, dkv, page_size);
  TORCH_CHECK(cache_indices.numel() >= T, "cache_indices must have at least T entries");
  TORCH_CHECK(cache_mask.numel() >= T, "cache_mask must have at least T entries");
  TORCH_CHECK(loc.numel() >= T, "loc must have at least T entries");
  const bool do_track = track_mask.has_value();
  if (do_track) {
    TORCH_CHECK(track_indices.has_value(), "track_indices is required when track_mask is provided");
    check_xpu_tensor(track_mask.value(), "track_mask");
    check_xpu_tensor(track_indices.value(), "track_indices");
    check_bool_tensor(track_mask.value(), "track_mask");
    check_int64_tensor(track_indices.value(), "track_indices");
    TORCH_CHECK(track_mask.value().numel() >= T, "track_mask must have at least T entries");
    TORCH_CHECK(track_indices.value().numel() >= T, "track_indices must have at least T entries");
  }

  const int64_t W = k_cache.size(1) + 1;
  const WeightLayout k_layout = resolve_weight_layout(k_weight, dkv, W, "k_weight");
  const WeightLayout v_layout = resolve_weight_layout(v_weight, dkv, W, "v_weight");
  at::Tensor q_out = at::empty_strided({T, dq}, {dq, 1}, qkvr.options().dtype(at::ScalarType::Byte));
  at::Tensor q_scale = at::empty_strided(
      {T, dq / kHeadDim, kHeadDim / kMXFP8Block},
      {dq / kMXFP8Block, kHeadDim / kMXFP8Block, 1},
      qkvr.options().dtype(at::ScalarType::Byte));
  at::Tensor k_out = at::empty_strided({T, dkv}, {dkv, 1}, qkvr.options());
  at::Tensor v_out = at::empty_strided({T, dkv}, {dkv, 1}, qkvr.options());

  auto queue = c10::xpu::getCurrentXPUStream().queue();
  const auto input_type = qkvr.scalar_type();
  SYCL_DISPATCH_FLOATING_TYPES(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input_type,
      "inkling_attn_prologue_decode_mxfp8",
      [&]() -> std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> {
        AttnPrologueDecodeParams<scalar_t> params{
            qkvr.data_ptr<scalar_t>(),
            k_cache.data_ptr<scalar_t>(),
            v_cache.data_ptr<scalar_t>(),
            cache_indices.data_ptr<int32_t>(),
            cache_mask.data_ptr<bool>(),
            k_weight.data_ptr<scalar_t>(),
            v_weight.data_ptr<scalar_t>(),
            do_track ? track_mask.value().data_ptr<bool>() : nullptr,
            do_track ? track_indices.value().data_ptr<int64_t>() : nullptr,
            q_gamma.data_ptr<scalar_t>(),
            k_gamma.data_ptr<scalar_t>(),
            log_tau.numel() > 0 ? log_tau.data_ptr<float>() : nullptr,
            nullptr,
            k_out.data_ptr<scalar_t>(),
            v_out.data_ptr<scalar_t>(),
            loc.data_ptr<int64_t>(),
            nullptr,
            nullptr,
            q_out.data_ptr<uint8_t>(),
            static_cast<uint8_t*>(k_buf.data_ptr()),
            static_cast<uint8_t*>(v_buf.data_ptr()),
            q_scale.data_ptr<uint8_t>(),
            sfk.data_ptr<uint8_t>(),
            sfv.data_ptr<uint8_t>(),
            k_layout,
            v_layout,
            static_cast<float>(eps),
            T,
            dq,
            dkv,
            qkvr.stride(0),
            q_off,
            k_off,
            v_off,
            k_cache.stride(0),
            k_cache.stride(1),
            k_buf.stride(0),
            loc.stride(0),
            do_track ? track_indices.value().stride(0) : 0,
            page_size,
            silu_activation,
            use_residual,
            do_track,
            do_store,
            true};
        launch_decode<scalar_t>(queue, params);
        return {q_out, k_out, v_out, q_scale};
      });
  return {q_out, k_out, v_out, q_scale};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> inkling_attn_prologue_extend_mxfp8(
    const at::Tensor& qkvr,
    at::Tensor& k_cache,
    at::Tensor& v_cache,
    const at::Tensor& cache_indices,
    const at::Tensor& cache_mask,
    const at::Tensor& has_initial_state,
    const at::Tensor& cu,
    const at::Tensor& si,
    const at::Tensor& k_weight,
    const at::Tensor& v_weight,
    const std::optional<at::Tensor>& track_rows,
    const std::optional<at::Tensor>& track_mask,
    const std::optional<at::Tensor>& track_dst,
    const at::Tensor& q_gamma,
    const at::Tensor& k_gamma,
    double eps,
    const at::Tensor& loc,
    at::Tensor& k_buf,
    at::Tensor& v_buf,
    at::Tensor& sfk,
    at::Tensor& sfv,
    int64_t q_off,
    int64_t k_off,
    int64_t v_off,
    int64_t dq,
    int64_t dkv,
    bool silu_activation,
    bool use_residual,
    bool do_store,
    bool do_cache_update,
    int64_t page_size,
    const at::Tensor& log_tau) {
  check_common_inputs(
      qkvr,
      k_cache,
      v_cache,
      k_weight,
      v_weight,
      q_gamma,
      k_gamma,
      loc,
      k_buf,
      v_buf,
      q_off,
      k_off,
      v_off,
      dq,
      dkv,
      false);
  check_xpu_tensor(cache_indices, "cache_indices");
  check_xpu_tensor(cache_mask, "cache_mask");
  check_xpu_tensor(has_initial_state, "has_initial_state");
  check_xpu_tensor(cu, "cu");
  check_xpu_tensor(si, "si");
  check_int32_tensor(cache_indices, "cache_indices");
  check_bool_tensor(cache_mask, "cache_mask");
  check_bool_tensor(has_initial_state, "has_initial_state");
  check_int64_tensor(cu, "cu");
  check_int32_tensor(si, "si");
  const int64_t T = qkvr.size(0);
  const int64_t B = cache_indices.numel();
  check_mxfp8_inputs(qkvr, loc, k_buf, v_buf, sfk, sfv, log_tau, T, dq, dkv, page_size);
  TORCH_CHECK(cache_mask.numel() >= B, "cache_mask must have at least B entries");
  TORCH_CHECK(has_initial_state.numel() >= B, "has_initial_state must have at least B entries");
  TORCH_CHECK(cu.numel() >= B + 1, "cu must have at least B + 1 entries");
  TORCH_CHECK(si.numel() >= T, "si must have at least T entries");
  TORCH_CHECK(loc.numel() >= T, "loc must have at least T entries");
  const bool do_track = track_mask.has_value() && track_mask.value().numel() > 0;
  if (do_track) {
    TORCH_CHECK(
        track_rows.has_value() && track_dst.has_value(), "track_rows and track_dst are required with track_mask");
    check_xpu_tensor(track_rows.value(), "track_rows");
    check_xpu_tensor(track_mask.value(), "track_mask");
    check_xpu_tensor(track_dst.value(), "track_dst");
    check_int64_tensor(track_rows.value(), "track_rows");
    check_bool_tensor(track_mask.value(), "track_mask");
    check_int64_tensor(track_dst.value(), "track_dst");
    TORCH_CHECK(track_rows.value().dim() == 2, "track_rows must have shape [B, W-1]");
    TORCH_CHECK(track_rows.value().size(0) >= B, "track_rows must cover B");
    TORCH_CHECK(track_mask.value().numel() >= B, "track_mask must cover B");
    TORCH_CHECK(track_dst.value().numel() >= B, "track_dst must cover B");
  }

  const int64_t W = k_cache.size(1) + 1;
  const WeightLayout k_layout = resolve_weight_layout(k_weight, dkv, W, "k_weight");
  const WeightLayout v_layout = resolve_weight_layout(v_weight, dkv, W, "v_weight");
  if (do_track) {
    TORCH_CHECK(track_rows.value().size(1) == W - 1, "track_rows.size(1) must equal W-1");
  }
  at::Tensor q_out = at::empty_strided({T, dq}, {dq, 1}, qkvr.options().dtype(at::ScalarType::Byte));
  at::Tensor q_scale = at::empty_strided(
      {T, dq / kHeadDim, kHeadDim / kMXFP8Block},
      {dq / kMXFP8Block, kHeadDim / kMXFP8Block, 1},
      qkvr.options().dtype(at::ScalarType::Byte));
  at::Tensor k_out = at::empty_strided({T, dkv}, {dkv, 1}, qkvr.options());
  at::Tensor v_out = at::empty_strided({T, dkv}, {dkv, 1}, qkvr.options());

  auto queue = c10::xpu::getCurrentXPUStream().queue();
  const auto input_type = qkvr.scalar_type();
  SYCL_DISPATCH_FLOATING_TYPES(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input_type,
      "inkling_attn_prologue_extend_mxfp8",
      [&]() -> std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> {
        AttnPrologueExtendParams<scalar_t> params{
            qkvr.data_ptr<scalar_t>(),
            k_cache.data_ptr<scalar_t>(),
            v_cache.data_ptr<scalar_t>(),
            cache_indices.data_ptr<int32_t>(),
            cache_mask.data_ptr<bool>(),
            has_initial_state.data_ptr<bool>(),
            cu.data_ptr<int64_t>(),
            si.data_ptr<int32_t>(),
            k_weight.data_ptr<scalar_t>(),
            v_weight.data_ptr<scalar_t>(),
            do_track ? track_rows.value().data_ptr<int64_t>() : nullptr,
            do_track ? track_mask.value().data_ptr<bool>() : nullptr,
            do_track ? track_dst.value().data_ptr<int64_t>() : nullptr,
            q_gamma.data_ptr<scalar_t>(),
            k_gamma.data_ptr<scalar_t>(),
            log_tau.numel() > 0 ? log_tau.data_ptr<float>() : nullptr,
            nullptr,
            k_out.data_ptr<scalar_t>(),
            v_out.data_ptr<scalar_t>(),
            loc.data_ptr<int64_t>(),
            nullptr,
            nullptr,
            q_out.data_ptr<uint8_t>(),
            static_cast<uint8_t*>(k_buf.data_ptr()),
            static_cast<uint8_t*>(v_buf.data_ptr()),
            q_scale.data_ptr<uint8_t>(),
            sfk.data_ptr<uint8_t>(),
            sfv.data_ptr<uint8_t>(),
            k_layout,
            v_layout,
            static_cast<float>(eps),
            T,
            B,
            dq,
            dkv,
            qkvr.stride(0),
            q_off,
            k_off,
            v_off,
            k_cache.stride(0),
            k_cache.stride(1),
            k_buf.stride(0),
            loc.stride(0),
            do_track ? track_rows.value().stride(0) : 0,
            do_track ? track_rows.value().stride(1) : 0,
            do_track ? track_dst.value().stride(0) : 0,
            page_size,
            silu_activation,
            use_residual,
            do_track,
            do_store,
            do_cache_update,
            true};
        launch_extend<scalar_t>(queue, params);
        return {q_out, k_out, v_out, q_scale};
      });
  return {q_out, k_out, v_out, q_scale};
}
