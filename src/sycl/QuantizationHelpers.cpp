/* Copyright 2025 SGLang Team. All Rights Reserved.
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *
 * This file adapts the Inkling BMG FP4 quantization helpers from
 * /data2/syk/cutlass-sycl/examples/21_bmg_quantization for the sgl-kernel
 * XPU extension ABI.
 */

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <sycl/sycl.hpp>
#include <tuple>
#include <type_traits>
#include <vector>

#include "sgl_kernel_ops.h"
#include "Utils.h"

namespace {

template <typename T, int N>
using NativeVec = sycl::vec<T, N>;

constexpr int kDefaultBlock = 256;
constexpr int kMxfp4GroupSize = 32;
constexpr int kNvfp4GroupSize = 16;
constexpr float kE2M1Max = 6.0f;
constexpr float kE4M3FnMax = 448.0f;

inline int ceil_div_int(int x, int y) {
  return (x + y - 1) / y;
}

inline int round_up_int(int x, int multiple) {
  return ceil_div_int(x, multiple) * multiple;
}

inline int checked_int64_to_int(int64_t value, char const* name) {
  TORCH_CHECK(value > 0 && value <= std::numeric_limits<int>::max(), name, " must be positive and fit in int32");
  return static_cast<int>(value);
}

template <typename scalar_t>
inline float to_float_device(scalar_t value) {
  return static_cast<float>(value);
}

inline float abs_f(float x) {
  return sycl::fabs(x);
}

inline float pow2_int(int exponent) {
  if (exponent < -126) {
    return 0.0f;
  }
  if (exponent > 127) {
    exponent = 127;
  }
  uint32_t bits = static_cast<uint32_t>(exponent + 127) << 23;
  return sycl::bit_cast<float>(bits);
}

inline int floor_log2_positive(float x) {
  uint32_t bits = sycl::bit_cast<uint32_t>(x);
  int exponent = static_cast<int>((bits >> 23) & 0xffu);
  if (exponent == 0) {
    return -126;
  }
  return exponent - 127;
}

inline int round_nearest_even_int(float x) {
  float base_f = sycl::floor(x);
  int base = static_cast<int>(base_f);
  float frac = x - base_f;
  if (frac > 0.5f || (frac == 0.5f && (base & 1))) {
    ++base;
  }
  return base;
}

inline uint8_t quantize_e2m1_code(float value) {
  uint32_t bits = sycl::bit_cast<uint32_t>(value);
  uint8_t sign = static_cast<uint8_t>((bits >> 28) & 0x8u);
  float x = sycl::bit_cast<float>(bits & 0x7fffffffu);
  constexpr float kTieTol = 1.0e-6f;
  uint8_t mag = 0u;
  mag += static_cast<uint8_t>(x > 0.25f + kTieTol);
  mag += static_cast<uint8_t>(x >= 0.75f - kTieTol);
  mag += static_cast<uint8_t>(x > 1.25f + kTieTol);
  mag += static_cast<uint8_t>(x >= 1.75f - kTieTol);
  mag += static_cast<uint8_t>(x > 2.5f + kTieTol);
  mag += static_cast<uint8_t>(x >= 3.5f - kTieTol);
  mag += static_cast<uint8_t>(x > 5.0f + kTieTol);
  return mag == 0u ? 0u : static_cast<uint8_t>(sign | mag);
}

inline uint8_t quantize_e2m1_pair(float first, float second) {
  uint32_t bits0 = sycl::bit_cast<uint32_t>(first);
  uint32_t bits1 = sycl::bit_cast<uint32_t>(second);
  uint8_t sign0 = static_cast<uint8_t>((bits0 >> 28) & 0x8u);
  uint8_t sign1 = static_cast<uint8_t>((bits1 >> 28) & 0x8u);
  float x0 = sycl::bit_cast<float>(bits0 & 0x7fffffffu);
  float x1 = sycl::bit_cast<float>(bits1 & 0x7fffffffu);
  constexpr float kTieTol = 1.0e-6f;
  uint8_t mag0 = 0u;
  uint8_t mag1 = 0u;
  mag0 += static_cast<uint8_t>(x0 > 0.25f + kTieTol);
  mag1 += static_cast<uint8_t>(x1 > 0.25f + kTieTol);
  mag0 += static_cast<uint8_t>(x0 >= 0.75f - kTieTol);
  mag1 += static_cast<uint8_t>(x1 >= 0.75f - kTieTol);
  mag0 += static_cast<uint8_t>(x0 > 1.25f + kTieTol);
  mag1 += static_cast<uint8_t>(x1 > 1.25f + kTieTol);
  mag0 += static_cast<uint8_t>(x0 >= 1.75f - kTieTol);
  mag1 += static_cast<uint8_t>(x1 >= 1.75f - kTieTol);
  mag0 += static_cast<uint8_t>(x0 > 2.5f + kTieTol);
  mag1 += static_cast<uint8_t>(x1 > 2.5f + kTieTol);
  mag0 += static_cast<uint8_t>(x0 >= 3.5f - kTieTol);
  mag1 += static_cast<uint8_t>(x1 >= 3.5f - kTieTol);
  mag0 += static_cast<uint8_t>(x0 > 5.0f + kTieTol);
  mag1 += static_cast<uint8_t>(x1 > 5.0f + kTieTol);
  uint8_t code0 = mag0 == 0u ? 0u : static_cast<uint8_t>(sign0 | mag0);
  uint8_t code1 = mag1 == 0u ? 0u : static_cast<uint8_t>(sign1 | mag1);
  return static_cast<uint8_t>(code0 | (code1 << 4));
}

inline int clamp_exponent_to_ue8m0(int exponent) {
  return exponent < -127 ? -127 : (exponent > 127 ? 127 : exponent);
}

inline uint8_t encode_ue8m0_exponent(int exponent) {
  return static_cast<uint8_t>(clamp_exponent_to_ue8m0(exponent) + 127);
}

inline float e4m3fn_mantissa_inv(int mantissa) {
  switch (mantissa) {
    case 0:
      return 1.0f;
    case 1:
      return 0.8888888888888888f;
    case 2:
      return 0.8f;
    case 3:
      return 0.7272727272727273f;
    case 4:
      return 0.6666666666666666f;
    case 5:
      return 0.6153846153846154f;
    case 6:
      return 0.5714285714285714f;
    default:
      return 0.5333333333333333f;
  }
}

inline float e4m3fn_subnormal_inv(int mantissa) {
  switch (mantissa) {
    case 1:
      return 512.0f;
    case 2:
      return 256.0f;
    case 3:
      return 170.66666666666666f;
    case 4:
      return 128.0f;
    case 5:
      return 102.4f;
    case 6:
      return 85.33333333333333f;
    default:
      return 73.14285714285714f;
  }
}

struct E4M3FnEncodeInvResult {
  uint8_t code = 0;
  float inv_decoded = 0.0f;
};

inline E4M3FnEncodeInvResult e4m3fn_encode_positive_with_inv_decode(float x) {
  if (!(x > 0.0f)) {
    return {};
  }
  if (x >= kE4M3FnMax) {
    return {0x7eu, 0.002232142857142857f};
  }

  constexpr float kMinNormal = 0.015625f;
  constexpr float kSubnormalStep = 0.001953125f;
  if (x < kMinNormal) {
    int mantissa = round_nearest_even_int(x / kSubnormalStep);
    if (mantissa <= 0) {
      return {};
    }
    if (mantissa >= 8) {
      return {0x08u, 64.0f};
    }
    return {static_cast<uint8_t>(mantissa), e4m3fn_subnormal_inv(mantissa)};
  }

  uint32_t bits = sycl::bit_cast<uint32_t>(x);
  int exponent = static_cast<int>((bits >> 23) & 0xffu) - 127;
  int decoded_exponent = exponent;
  uint32_t mantissa_bits = bits & 0x007fffffu;
  int mantissa = static_cast<int>(mantissa_bits >> 20);
  uint32_t round_bits = mantissa_bits & 0x000fffffu;
  if (round_bits > 0x00080000u || (round_bits == 0x00080000u && (mantissa & 1))) {
    ++mantissa;
  }
  int exponent_field = exponent + 7;
  if (mantissa >= 8) {
    mantissa = 0;
    ++exponent_field;
    ++decoded_exponent;
  }
  if (exponent_field >= 15 && mantissa > 6) {
    return {0x7eu, 0.002232142857142857f};
  }
  if (exponent_field > 15) {
    return {0x7eu, 0.002232142857142857f};
  }
  uint8_t code = static_cast<uint8_t>((exponent_field << 3) | mantissa);
  float inv_decoded = e4m3fn_mantissa_inv(mantissa) * pow2_int(-decoded_exponent);
  return {code, inv_decoded};
}

inline int nvfp4_swizzled_scale_index(int row, int group, int rounded_groups) {
  int row_block = row / 128;
  int row_rem = row - row_block * 128;
  int e = row_rem / 32;
  int d = row_rem - e * 32;
  int c = group / 4;
  int f = group - c * 4;
  int groups4 = rounded_groups / 4;
  return (((row_block * groups4 + c) * 32 + d) * 4 + e) * 4 + f;
}

template <typename scalar_t>
struct Mxfp4Params {
  scalar_t const* __restrict x = nullptr;
  uint8_t* __restrict packed = nullptr;
  uint8_t* __restrict scales = nullptr;
  int rows = 0;
  int cols = 0;
  int groups = 0;
  int total_groups = 0;
  float eps = 1.0e-10f;
  int eps_exp = -34;
  bool column_major_scales = false;
};

inline int choose_group_block(int groups) {
  if (groups >= 256 && groups % 256 == 0) {
    return 256;
  }
  if (groups >= 128 && groups % 128 == 0) {
    return 128;
  }
  if (groups >= 64 && groups % 64 == 0) {
    return 64;
  }
  if (groups >= 32 && groups % 32 == 0) {
    return 32;
  }
  if (groups >= 16 && groups % 16 == 0) {
    return 16;
  }
  if (groups >= 8 && groups % 8 == 0) {
    return 8;
  }
  return std::min(kDefaultBlock, groups);
}

inline int choose_row_block_for_group_tile(int groups) {
  int rows = std::max(1, kDefaultBlock / std::max(1, groups));
  while (rows > 1 && (rows * groups) % 16 != 0) {
    --rows;
  }
  return rows;
}

template <typename scalar_t>
inline constexpr bool is_mxfp4_raw16_v =
    std::is_same_v<scalar_t, sycl::half> || std::is_same_v<scalar_t, sycl::ext::oneapi::bfloat16>;

inline uint16_t raw16_abs_bits(uint16_t raw) {
  return static_cast<uint16_t>(raw & 0x7fffu);
}

template <typename scalar_t>
inline float raw16_to_float(uint16_t raw) {
  if constexpr (std::is_same_v<scalar_t, sycl::ext::oneapi::bfloat16>) {
    return sycl::bit_cast<float>(static_cast<uint32_t>(raw) << 16);
  } else {
    return static_cast<float>(sycl::bit_cast<sycl::half>(raw));
  }
}

template <typename scalar_t>
inline int raw16_floor_log2_positive(uint16_t raw_abs) {
  if constexpr (std::is_same_v<scalar_t, sycl::ext::oneapi::bfloat16>) {
    int exponent = static_cast<int>((raw_abs >> 7) & 0xffu);
    return exponent == 0 ? -126 : exponent - 127;
  } else {
    int exponent = static_cast<int>((raw_abs >> 10) & 0x1fu);
    if (exponent != 0) {
      return exponent - 15;
    }
    int mantissa = static_cast<int>(raw_abs & 0x03ffu);
    if (mantissa == 0) {
      return -126;
    }
    int leading_bit = 0;
    if (mantissa >= 512) {
      leading_bit = 9;
    } else if (mantissa >= 256) {
      leading_bit = 8;
    } else if (mantissa >= 128) {
      leading_bit = 7;
    } else if (mantissa >= 64) {
      leading_bit = 6;
    } else if (mantissa >= 32) {
      leading_bit = 5;
    } else if (mantissa >= 16) {
      leading_bit = 4;
    } else if (mantissa >= 8) {
      leading_bit = 3;
    } else if (mantissa >= 4) {
      leading_bit = 2;
    } else if (mantissa >= 2) {
      leading_bit = 1;
    }
    return -24 + leading_bit;
  }
}

template <typename scalar_t>
inline uint32_t quantize_e2m1_raw_word_pairs(uint64_t raw, float scale) {
  uint16_t bits0 = static_cast<uint16_t>(raw);
  uint16_t bits1 = static_cast<uint16_t>(raw >> 16);
  uint16_t bits2 = static_cast<uint16_t>(raw >> 32);
  uint16_t bits3 = static_cast<uint16_t>(raw >> 48);
  uint32_t packed01 = quantize_e2m1_pair(raw16_to_float<scalar_t>(bits0) * scale, raw16_to_float<scalar_t>(bits1) * scale);
  uint32_t packed23 = quantize_e2m1_pair(raw16_to_float<scalar_t>(bits2) * scale, raw16_to_float<scalar_t>(bits3) * scale);
  return packed01 | (packed23 << 8);
}

template <typename scalar_t, bool StoreScale, int StaticGroups = 0>
uint8_t process_mxfp4_group_impl(Mxfp4Params<scalar_t> const& p, int row, int group) {
  int col0 = group * kMxfp4GroupSize;
  int groups = StaticGroups > 0 ? StaticGroups : p.groups;
  int cols = StaticGroups > 0 ? StaticGroups * kMxfp4GroupSize : p.cols;
  int base = row * cols + col0;
  int packed_base = (row * cols) / 2 + group * (kMxfp4GroupSize / 2);

  if constexpr (is_mxfp4_raw16_v<scalar_t>) {
    using RawWords = sycl::vec<uint64_t, kMxfp4GroupSize / 4>;
    scalar_t const* input_ptr = static_cast<scalar_t const*>(__builtin_assume_aligned(p.x + base, 64));
    RawWords raw_words = *reinterpret_cast<RawWords const*>(input_ptr);
    uint16_t local_absmax_raw = 0;
#pragma unroll
    for (int word = 0; word < kMxfp4GroupSize / 4; ++word) {
      uint64_t raw = raw_words[word];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        uint16_t bits = static_cast<uint16_t>(raw >> (16 * j));
        uint16_t abs_bits = raw16_abs_bits(bits);
        local_absmax_raw = abs_bits > local_absmax_raw ? abs_bits : local_absmax_raw;
      }
    }

    int raw_exp = raw16_floor_log2_positive<scalar_t>(local_absmax_raw);
    int max_exp = raw_exp;
    if (local_absmax_raw == 0 || raw_exp <= p.eps_exp) {
      float local_absmax = sycl::fmax(p.eps, raw16_to_float<scalar_t>(local_absmax_raw));
      max_exp = floor_log2_positive(local_absmax);
    }

    int shared_exp = clamp_exponent_to_ue8m0(max_exp - 2);
    float inv_scale = pow2_int(-shared_exp);
    uint8_t* packed_ptr = static_cast<uint8_t*>(__builtin_assume_aligned(p.packed + packed_base, 16));

    uint32_t packed_0 =
        quantize_e2m1_raw_word_pairs<scalar_t>(raw_words[0], inv_scale) |
        (quantize_e2m1_raw_word_pairs<scalar_t>(raw_words[1], inv_scale) << 16);
    uint32_t packed_1 =
        quantize_e2m1_raw_word_pairs<scalar_t>(raw_words[2], inv_scale) |
        (quantize_e2m1_raw_word_pairs<scalar_t>(raw_words[3], inv_scale) << 16);
    uint32_t packed_2 =
        quantize_e2m1_raw_word_pairs<scalar_t>(raw_words[4], inv_scale) |
        (quantize_e2m1_raw_word_pairs<scalar_t>(raw_words[5], inv_scale) << 16);
    uint32_t packed_3 =
        quantize_e2m1_raw_word_pairs<scalar_t>(raw_words[6], inv_scale) |
        (quantize_e2m1_raw_word_pairs<scalar_t>(raw_words[7], inv_scale) << 16);
    sycl::vec<uint32_t, 4> packed_words;
    packed_words[0] = packed_0;
    packed_words[1] = packed_1;
    packed_words[2] = packed_2;
    packed_words[3] = packed_3;
    *reinterpret_cast<sycl::vec<uint32_t, 4>*>(packed_ptr) = packed_words;

    uint8_t scale_byte = encode_ue8m0_exponent(shared_exp);
    if constexpr (StoreScale) {
      int scale_idx = p.column_major_scales ? group * p.rows + row : row * groups + group;
      p.scales[scale_idx] = scale_byte;
    }
    return scale_byte;
  }

  float values[kMxfp4GroupSize];
  float local_absmax = p.eps;
#pragma unroll
  for (int i = 0; i < kMxfp4GroupSize; ++i) {
    float value = to_float_device(p.x[base + i]);
    values[i] = value;
    local_absmax = sycl::fmax(local_absmax, abs_f(value));
  }

  int shared_exp = clamp_exponent_to_ue8m0(floor_log2_positive(local_absmax) - 2);
  float inv_scale = pow2_int(-shared_exp);
  uint32_t packed_0 = 0;
  uint32_t packed_1 = 0;
  uint32_t packed_2 = 0;
  uint32_t packed_3 = 0;
#pragma unroll
  for (int i = 0; i < kMxfp4GroupSize; i += 2) {
    uint32_t packed_pair = quantize_e2m1_pair(values[i] * inv_scale, values[i + 1] * inv_scale);
    int pair = i / 2;
    uint32_t shifted_pair = packed_pair << (8 * (pair & 3));
    if (pair < 4) {
      packed_0 |= shifted_pair;
    } else if (pair < 8) {
      packed_1 |= shifted_pair;
    } else if (pair < 12) {
      packed_2 |= shifted_pair;
    } else {
      packed_3 |= shifted_pair;
    }
  }
  sycl::vec<uint32_t, 4> packed_words;
  packed_words[0] = packed_0;
  packed_words[1] = packed_1;
  packed_words[2] = packed_2;
  packed_words[3] = packed_3;
  *reinterpret_cast<sycl::vec<uint32_t, 4>*>(p.packed + packed_base) = packed_words;

  uint8_t scale_byte = encode_ue8m0_exponent(shared_exp);
  if constexpr (StoreScale) {
    int scale_idx = p.column_major_scales ? group * p.rows + row : row * groups + group;
    p.scales[scale_idx] = scale_byte;
  }
  return scale_byte;
}

template <typename scalar_t, int StaticGroups = 0>
inline void process_mxfp4_group(Mxfp4Params<scalar_t> const& p, int row, int group) {
  (void)process_mxfp4_group_impl<scalar_t, true, StaticGroups>(p, row, group);
}

template <typename scalar_t>
struct Mxfp4MappingKernel1D {
  Mxfp4Params<scalar_t> p;

  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> item) const {
    int global_group = static_cast<int>(item.get_global_id(0));
    if (global_group >= p.total_groups) {
      return;
    }
    int row = global_group / p.groups;
    int group = global_group - row * p.groups;
    process_mxfp4_group(p, row, group);
  }
};

template <typename scalar_t, int Groups>
struct Mxfp4MappingKernel1DStaticGroups {
  Mxfp4Params<scalar_t> p;

  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> item) const {
    int global_group = static_cast<int>(item.get_global_id(0));
    int total_groups = p.rows * Groups;
    if (global_group >= total_groups) {
      return;
    }
    int row = global_group / Groups;
    int group = global_group - row * Groups;
    process_mxfp4_group<scalar_t, Groups>(p, row, group);
  }
};

template <typename scalar_t>
struct Mxfp4MappingKernelTiled2D {
  Mxfp4Params<scalar_t> p;

  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<2> item) const {
    int row = static_cast<int>(item.get_global_id(0));
    if (row >= p.rows) {
      return;
    }
    int group = static_cast<int>(item.get_global_id(1));
    process_mxfp4_group(p, row, group);
  }
};

template <typename scalar_t, int Groups>
struct Mxfp4MappingKernelTiled2DStaticGroups {
  Mxfp4Params<scalar_t> p;

  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<2> item) const {
    int row = static_cast<int>(item.get_global_id(0));
    if (row >= p.rows) {
      return;
    }
    int group = static_cast<int>(item.get_global_id(1));
    process_mxfp4_group<scalar_t, Groups>(p, row, group);
  }
};

template <typename scalar_t>
struct Mxfp4MappingKernel2D {
  Mxfp4Params<scalar_t> p;

  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<2> item) const {
    int row = static_cast<int>(item.get_global_id(0));
    int group = static_cast<int>(item.get_global_id(1));
    if (group >= p.groups) {
      return;
    }
    process_mxfp4_group(p, row, group);
  }
};

template <typename scalar_t, int Groups>
struct Mxfp4MappingKernel2DStaticGroups {
  Mxfp4Params<scalar_t> p;

  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<2> item) const {
    int row = static_cast<int>(item.get_global_id(0));
    int group = static_cast<int>(item.get_global_id(1));
    if (group >= Groups) {
      return;
    }
    process_mxfp4_group<scalar_t, Groups>(p, row, group);
  }
};

template <typename scalar_t, int Groups, int GroupsPerItem>
struct Mxfp4MappingKernel2DStaticGroupTile {
  Mxfp4Params<scalar_t> p;

  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<2> item) const {
    int row = static_cast<int>(item.get_global_id(0));
    int group_tile = static_cast<int>(item.get_global_id(1));
    int group = group_tile * GroupsPerItem;
#pragma unroll
    for (int i = 0; i < GroupsPerItem; ++i) {
      process_mxfp4_group<scalar_t, Groups>(p, row, group + i);
    }
  }
};

template <typename scalar_t>
void launch_mxfp4_mapping_1d(sycl::queue& queue, Mxfp4Params<scalar_t> const& p) {
  int global = round_up_int(p.total_groups, kDefaultBlock);
  queue.parallel_for<Mxfp4MappingKernel1D<scalar_t>>(
      sycl::nd_range<1>(
          sycl::range<1>(static_cast<std::size_t>(global)),
          sycl::range<1>(static_cast<std::size_t>(kDefaultBlock))),
      Mxfp4MappingKernel1D<scalar_t>{p});
}

template <typename scalar_t, int Groups>
void launch_mxfp4_mapping_1d_static(sycl::queue& queue, Mxfp4Params<scalar_t> const& p) {
  int total_groups = p.rows * Groups;
  int global = round_up_int(total_groups, kDefaultBlock);
  queue.parallel_for<Mxfp4MappingKernel1DStaticGroups<scalar_t, Groups>>(
      sycl::nd_range<1>(
          sycl::range<1>(static_cast<std::size_t>(global)),
          sycl::range<1>(static_cast<std::size_t>(kDefaultBlock))),
      Mxfp4MappingKernel1DStaticGroups<scalar_t, Groups>{p});
}

template <typename scalar_t>
void launch_mxfp4_mapping_tiled_2d(sycl::queue& queue, Mxfp4Params<scalar_t> const& p, int row_block) {
  int rows_global = round_up_int(p.rows, row_block);
  queue.parallel_for<Mxfp4MappingKernelTiled2D<scalar_t>>(
      sycl::nd_range<2>(
          sycl::range<2>(static_cast<std::size_t>(rows_global), static_cast<std::size_t>(p.groups)),
          sycl::range<2>(static_cast<std::size_t>(row_block), static_cast<std::size_t>(p.groups))),
      Mxfp4MappingKernelTiled2D<scalar_t>{p});
}

template <typename scalar_t, int Groups>
void launch_mxfp4_mapping_tiled_2d_static(sycl::queue& queue, Mxfp4Params<scalar_t> const& p, int row_block) {
  int rows_global = round_up_int(p.rows, row_block);
  queue.parallel_for<Mxfp4MappingKernelTiled2DStaticGroups<scalar_t, Groups>>(
      sycl::nd_range<2>(
          sycl::range<2>(static_cast<std::size_t>(rows_global), static_cast<std::size_t>(Groups)),
          sycl::range<2>(static_cast<std::size_t>(row_block), static_cast<std::size_t>(Groups))),
      Mxfp4MappingKernelTiled2DStaticGroups<scalar_t, Groups>{p});
}

template <typename scalar_t, int Groups>
void launch_mxfp4_mapping_2d_static(sycl::queue& queue, Mxfp4Params<scalar_t> const& p) {
  int group_block = choose_group_block(Groups);
  int groups_global = round_up_int(Groups, group_block);
  queue.parallel_for<Mxfp4MappingKernel2DStaticGroups<scalar_t, Groups>>(
      sycl::nd_range<2>(
          sycl::range<2>(static_cast<std::size_t>(p.rows), static_cast<std::size_t>(groups_global)),
          sycl::range<2>(1, static_cast<std::size_t>(group_block))),
      Mxfp4MappingKernel2DStaticGroups<scalar_t, Groups>{p});
}

template <typename scalar_t, int Groups, int GroupsPerItem>
void launch_mxfp4_mapping_2d_static_group_tile(sycl::queue& queue, Mxfp4Params<scalar_t> const& p) {
  static_assert(Groups % GroupsPerItem == 0);
  constexpr int kGroupTiles = Groups / GroupsPerItem;
  int group_block = std::max(1, choose_group_block(Groups) / GroupsPerItem);
  int groups_global = round_up_int(kGroupTiles, group_block);
  queue.parallel_for<Mxfp4MappingKernel2DStaticGroupTile<scalar_t, Groups, GroupsPerItem>>(
      sycl::nd_range<2>(
          sycl::range<2>(static_cast<std::size_t>(p.rows), static_cast<std::size_t>(groups_global)),
          sycl::range<2>(1, static_cast<std::size_t>(group_block))),
      Mxfp4MappingKernel2DStaticGroupTile<scalar_t, Groups, GroupsPerItem>{p});
}

template <typename scalar_t>
void launch_mxfp4_mapping(sycl::queue& queue, Mxfp4Params<scalar_t> const& p) {
  if (p.total_groups <= 0) {
    return;
  }
  TORCH_CHECK(p.total_groups <= std::numeric_limits<int>::max(), "MXFP4 launch grid is too large");

  if (p.groups <= 8) {
    if (p.groups == 6) {
      launch_mxfp4_mapping_1d_static<scalar_t, 6>(queue, p);
    } else {
      launch_mxfp4_mapping_1d(queue, p);
    }
    return;
  }

  if (p.groups < 128) {
    int row_block = choose_row_block_for_group_tile(p.groups);
    if (p.groups == 96) {
      launch_mxfp4_mapping_tiled_2d_static<scalar_t, 96>(queue, p, row_block);
    } else if (p.groups == 48) {
      launch_mxfp4_mapping_tiled_2d_static<scalar_t, 48>(queue, p, row_block);
    } else if (p.groups == 24) {
      launch_mxfp4_mapping_tiled_2d_static<scalar_t, 24>(queue, p, row_block);
    } else if (p.groups == 12) {
      launch_mxfp4_mapping_tiled_2d_static<scalar_t, 12>(queue, p, row_block);
    } else {
      launch_mxfp4_mapping_tiled_2d(queue, p, row_block);
    }
    return;
  }

  if (p.groups == 192) {
    launch_mxfp4_mapping_2d_static_group_tile<scalar_t, 192, 2>(queue, p);
    return;
  }
  if (p.groups == 384) {
    launch_mxfp4_mapping_2d_static_group_tile<scalar_t, 384, 2>(queue, p);
    return;
  }

  int group_block = choose_group_block(p.groups);
  int groups_global = round_up_int(p.groups, group_block);
  queue.parallel_for<Mxfp4MappingKernel2D<scalar_t>>(
      sycl::nd_range<2>(
          sycl::range<2>(static_cast<std::size_t>(p.rows), static_cast<std::size_t>(groups_global)),
          sycl::range<2>(1, static_cast<std::size_t>(group_block))),
      Mxfp4MappingKernel2D<scalar_t>{p});
}

template <bool RowsMultiple8>
struct Mxfp4ScaleTransposeKernel {
  uint8_t const* row_major = nullptr;
  uint8_t* column_major = nullptr;
  int rows = 0;
  int groups = 0;

  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<2> item) const {
    constexpr int kRowsPerItem = RowsMultiple8 ? 8 : 4;
    int row = static_cast<int>(item.get_global_id(0)) * kRowsPerItem;
    int group = static_cast<int>(item.get_global_id(1));
    if (group >= groups || row >= rows) {
      return;
    }

    int dst_idx = group * rows + row;
    int src_idx = row * groups + group;
    if constexpr (RowsMultiple8) {
      uint8_t scale0 = row_major[src_idx];
      uint8_t scale1 = row_major[src_idx + groups];
      uint8_t scale2 = row_major[src_idx + 2 * groups];
      uint8_t scale3 = row_major[src_idx + 3 * groups];
      uint8_t scale4 = row_major[src_idx + 4 * groups];
      uint8_t scale5 = row_major[src_idx + 5 * groups];
      uint8_t scale6 = row_major[src_idx + 6 * groups];
      uint8_t scale7 = row_major[src_idx + 7 * groups];
      uint32_t scale_quad_lo = static_cast<uint32_t>(scale0) |
          (static_cast<uint32_t>(scale1) << 8) |
          (static_cast<uint32_t>(scale2) << 16) |
          (static_cast<uint32_t>(scale3) << 24);
      uint32_t scale_quad_hi = static_cast<uint32_t>(scale4) |
          (static_cast<uint32_t>(scale5) << 8) |
          (static_cast<uint32_t>(scale6) << 16) |
          (static_cast<uint32_t>(scale7) << 24);
      uint64_t scale_oct = static_cast<uint64_t>(scale_quad_lo) | (static_cast<uint64_t>(scale_quad_hi) << 32);
      uint8_t* dst_ptr = static_cast<uint8_t*>(__builtin_assume_aligned(column_major + dst_idx, 8));
      *reinterpret_cast<uint64_t*>(dst_ptr) = scale_oct;
    } else {
#pragma unroll
      for (int i = 0; i < kRowsPerItem; ++i) {
        if (row + i < rows) {
          column_major[dst_idx + i] = row_major[(row + i) * groups + group];
        }
      }
    }
  }
};

template <bool RowsMultiple8>
void launch_mxfp4_scale_transpose(
    sycl::queue& queue,
    uint8_t const* row_major,
    uint8_t* column_major,
    int rows,
    int groups) {
  if (rows <= 0 || groups <= 0) {
    return;
  }
  constexpr int kRowsPerItem = RowsMultiple8 ? 8 : 4;
  int group_block = choose_group_block(groups);
  int row_tile_block = std::max(1, kDefaultBlock / group_block);
  int row_tiles = ceil_div_int(rows, kRowsPerItem);
  int row_tiles_global = round_up_int(row_tiles, row_tile_block);
  int groups_global = round_up_int(groups, group_block);
  queue.parallel_for<Mxfp4ScaleTransposeKernel<RowsMultiple8>>(
      sycl::nd_range<2>(
          sycl::range<2>(static_cast<std::size_t>(row_tiles_global), static_cast<std::size_t>(groups_global)),
          sycl::range<2>(static_cast<std::size_t>(row_tile_block), static_cast<std::size_t>(group_block))),
      Mxfp4ScaleTransposeKernel<RowsMultiple8>{row_major, column_major, rows, groups});
}

template <typename scalar_t>
struct Nvfp4Params {
  scalar_t const* __restrict x = nullptr;
  uint8_t* __restrict packed = nullptr;
  uint8_t* __restrict scales = nullptr;
  uint64_t const* __restrict raw_scale_output_lut = nullptr;
  int rows = 0;
  int cols = 0;
  int groups = 0;
  int rounded_groups = 0;
  float global_scale = 1.0f;
  float scale_factor = 1.0f;
};

template <int Groups>
inline int nvfp4_swizzled_scale_quad_index_static(int row, int group) {
  constexpr int kGroups4 = Groups / 4;
  int row_block = row >> 7;
  int row_rem = row & 127;
  int e = row_rem >> 5;
  int d = row_rem & 31;
  int c = group >> 2;
  return ((row_block * kGroups4 + c) * 32 + d) * 16 + (e << 2);
}

template <typename scalar_t, int StaticGroups = 0, bool StoreScale = true, int StaticScaleGroups = StaticGroups>
uint8_t process_nvfp4_group_impl(Nvfp4Params<scalar_t> const& p, int row, int group) {
  int col0 = group * kNvfp4GroupSize;
  int cols = StaticGroups > 0 ? StaticGroups * kNvfp4GroupSize : p.cols;
  int input_base = row * cols + col0;
  int rounded_groups = StaticScaleGroups == 0 ? p.rounded_groups : StaticScaleGroups;

  if constexpr (is_mxfp4_raw16_v<scalar_t>) {
    using RawWords = NativeVec<uint64_t, kNvfp4GroupSize / 4>;
    scalar_t const* input_ptr = static_cast<scalar_t const*>(__builtin_assume_aligned(p.x + input_base, 32));
    RawWords raw_words = *reinterpret_cast<RawWords const*>(input_ptr);
    uint16_t max_abs_raw = 0;
#pragma unroll
    for (int word = 0; word < kNvfp4GroupSize / 4; ++word) {
      uint64_t raw = raw_words[word];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        uint16_t bits = static_cast<uint16_t>(raw >> (16 * j));
        uint16_t abs_bits = raw16_abs_bits(bits);
        max_abs_raw = abs_bits > max_abs_raw ? abs_bits : max_abs_raw;
      }
    }

    uint8_t scale_byte = 0;
    float output_scale = 0.0f;
    if (p.raw_scale_output_lut != nullptr) {
      uint64_t packed_scale = p.raw_scale_output_lut[max_abs_raw];
      scale_byte = static_cast<uint8_t>(packed_scale);
      output_scale = sycl::bit_cast<float>(static_cast<uint32_t>(packed_scale >> 32));
    } else {
      float max_abs = raw16_to_float<scalar_t>(max_abs_raw);
      E4M3FnEncodeInvResult scale = e4m3fn_encode_positive_with_inv_decode(max_abs * p.scale_factor);
      scale_byte = scale.code;
      output_scale = p.global_scale * scale.inv_decoded;
    }

    if constexpr (StoreScale) {
      int scale_idx = 0;
      if constexpr (StaticScaleGroups > 0) {
        scale_idx = nvfp4_swizzled_scale_quad_index_static<StaticScaleGroups>(row, group) + (group & 3);
      } else {
        scale_idx = nvfp4_swizzled_scale_index(row, group, rounded_groups);
      }
      p.scales[scale_idx] = scale_byte;
    }

    uint32_t packed_lo =
        quantize_e2m1_raw_word_pairs<scalar_t>(raw_words[0], output_scale) |
        (quantize_e2m1_raw_word_pairs<scalar_t>(raw_words[1], output_scale) << 16);
    uint32_t packed_hi =
        quantize_e2m1_raw_word_pairs<scalar_t>(raw_words[2], output_scale) |
        (quantize_e2m1_raw_word_pairs<scalar_t>(raw_words[3], output_scale) << 16);
    NativeVec<uint32_t, 2> packed_words;
    packed_words[0] = packed_lo;
    packed_words[1] = packed_hi;
    int packed_base = (row * cols + col0) / 2;
    uint8_t* packed_ptr = static_cast<uint8_t*>(__builtin_assume_aligned(p.packed + packed_base, 8));
    *reinterpret_cast<NativeVec<uint32_t, 2>*>(packed_ptr) = packed_words;
    return scale_byte;
  }

  float values[kNvfp4GroupSize];
  float max_abs = 0.0f;
#pragma unroll
  for (int i = 0; i < kNvfp4GroupSize; ++i) {
    float value = to_float_device(p.x[input_base + i]);
    values[i] = value;
    max_abs = sycl::fmax(max_abs, abs_f(value));
  }

  E4M3FnEncodeInvResult scale = e4m3fn_encode_positive_with_inv_decode(max_abs * p.scale_factor);
  uint8_t scale_byte = scale.code;
  float output_scale = p.global_scale * scale.inv_decoded;

  if constexpr (StoreScale) {
    int scale_idx = 0;
    if constexpr (StaticScaleGroups > 0) {
      scale_idx = nvfp4_swizzled_scale_quad_index_static<StaticScaleGroups>(row, group) + (group & 3);
    } else {
      scale_idx = nvfp4_swizzled_scale_index(row, group, rounded_groups);
    }
    p.scales[scale_idx] = scale_byte;
  }

  uint32_t packed_lo = 0;
  uint32_t packed_hi = 0;
#pragma unroll
  for (int i = 0; i < kNvfp4GroupSize; i += 2) {
    uint32_t packed_pair = quantize_e2m1_pair(values[i] * output_scale, values[i + 1] * output_scale);
    int pair = i / 2;
    uint32_t shifted_pair = packed_pair << (8 * (pair & 3));
    if (pair < 4) {
      packed_lo |= shifted_pair;
    } else {
      packed_hi |= shifted_pair;
    }
  }
  NativeVec<uint32_t, 2> packed_words;
  packed_words[0] = packed_lo;
  packed_words[1] = packed_hi;
  int packed_base = (row * cols + col0) / 2;
  *reinterpret_cast<NativeVec<uint32_t, 2>*>(p.packed + packed_base) = packed_words;
  return scale_byte;
}

template <typename scalar_t, int StaticGroups = 0, int StaticScaleGroups = StaticGroups>
inline void process_nvfp4_group(Nvfp4Params<scalar_t> const& p, int row, int group) {
  (void)process_nvfp4_group_impl<scalar_t, StaticGroups, true, StaticScaleGroups>(p, row, group);
}

template <typename scalar_t, int Groups, int GroupsPerItem>
inline void process_nvfp4_static_group_tile_store(Nvfp4Params<scalar_t> const& p, int row, int group_tile) {
  int group = group_tile * GroupsPerItem;
  int static_scale_idx = nvfp4_swizzled_scale_quad_index_static<Groups>(row, group);
#pragma unroll
  for (int i = 0; i < GroupsPerItem; i += 4) {
    uint8_t scale0 = process_nvfp4_group_impl<scalar_t, Groups, false, Groups>(p, row, group + i);
    uint8_t scale1 = process_nvfp4_group_impl<scalar_t, Groups, false, Groups>(p, row, group + i + 1);
    uint8_t scale2 = process_nvfp4_group_impl<scalar_t, Groups, false, Groups>(p, row, group + i + 2);
    uint8_t scale3 = process_nvfp4_group_impl<scalar_t, Groups, false, Groups>(p, row, group + i + 3);
    uint32_t scale_quad = static_cast<uint32_t>(scale0) | (static_cast<uint32_t>(scale1) << 8) |
        (static_cast<uint32_t>(scale2) << 16) | (static_cast<uint32_t>(scale3) << 24);
    int scale_idx = static_scale_idx;
    static_scale_idx += 512;
    uint8_t* scale_ptr = static_cast<uint8_t*>(__builtin_assume_aligned(p.scales + scale_idx, 4));
    *reinterpret_cast<uint32_t*>(scale_ptr) = scale_quad;
  }
}

template <typename scalar_t>
struct Nvfp4LayoutKernel {
  Nvfp4Params<scalar_t> p;

  void operator()(sycl::nd_item<1> item) const {
    int global_group = static_cast<int>(item.get_global_id(0));
    int total_groups = p.rows * p.groups;
    if (global_group >= total_groups) {
      return;
    }
    int row = global_group / p.groups;
    int group = global_group - row * p.groups;
    int col0 = group * kNvfp4GroupSize;
    int base = row * p.cols + col0;

    float max_abs = 0.0f;
#pragma unroll
    for (int i = 0; i < kNvfp4GroupSize; ++i) {
      max_abs = sycl::fmax(max_abs, abs_f(to_float_device(p.x[base + i])));
    }

    E4M3FnEncodeInvResult scale = e4m3fn_encode_positive_with_inv_decode(max_abs * p.scale_factor);
    float output_scale = p.global_scale * scale.inv_decoded;
    int scale_idx = nvfp4_swizzled_scale_index(row, group, p.rounded_groups);
    p.scales[scale_idx] = scale.code;

    int packed_base = (row * p.cols + col0) / 2;
#pragma unroll
    for (int i = 0; i < kNvfp4GroupSize; i += 2) {
      float v0 = to_float_device(p.x[base + i]) * output_scale;
      float v1 = to_float_device(p.x[base + i + 1]) * output_scale;
      p.packed[packed_base + i / 2] = quantize_e2m1_pair(v0, v1);
    }
  }
};

template <typename scalar_t>
struct Nvfp4LayoutKernel1DOptimized {
  Nvfp4Params<scalar_t> p;

  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> item) const {
    int global_group = static_cast<int>(item.get_global_id(0));
    int total_groups = p.rows * p.groups;
    if (global_group >= total_groups) {
      return;
    }
    int row = global_group / p.groups;
    int group = global_group - row * p.groups;
    process_nvfp4_group(p, row, group);
  }
};

template <typename scalar_t>
struct Nvfp4LayoutKernel2DOptimized {
  Nvfp4Params<scalar_t> p;

  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<2> item) const {
    int row = static_cast<int>(item.get_global_id(0));
    int group = static_cast<int>(item.get_global_id(1));
    if (group >= p.groups) {
      return;
    }
    process_nvfp4_group(p, row, group);
  }
};

template <typename scalar_t, int Groups>
struct Nvfp4LayoutKernel2DStaticGroupsOptimized {
  Nvfp4Params<scalar_t> p;

  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<2> item) const {
    int row = static_cast<int>(item.get_global_id(0));
    int group = static_cast<int>(item.get_global_id(1));
    if (group >= Groups) {
      return;
    }
    process_nvfp4_group<scalar_t, Groups>(p, row, group);
  }
};

template <typename scalar_t, int Groups, int GroupsPerItem, int ScaleGroups = Groups>
struct Nvfp4LayoutKernel2DStaticGroupTileOptimized {
  Nvfp4Params<scalar_t> p;

  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<2> item) const {
    int row = static_cast<int>(item.get_global_id(0));
    if (row >= p.rows) {
      return;
    }
    int group_tile = static_cast<int>(item.get_global_id(1));
    int group = group_tile * GroupsPerItem;
    int static_scale_idx = nvfp4_swizzled_scale_quad_index_static<ScaleGroups>(row, group);
    if constexpr (GroupsPerItem == 1) {
      uint8_t scale0 = process_nvfp4_group_impl<scalar_t, Groups, false, ScaleGroups>(p, row, group);
      p.scales[static_scale_idx + (group & 3)] = scale0;
    } else if constexpr (GroupsPerItem == 2) {
      uint8_t scale0 = process_nvfp4_group_impl<scalar_t, Groups, false, ScaleGroups>(p, row, group);
      uint8_t scale1 = process_nvfp4_group_impl<scalar_t, Groups, false, ScaleGroups>(p, row, group + 1);
      uint16_t scale_pair = static_cast<uint16_t>(scale0) | (static_cast<uint16_t>(scale1) << 8);
      uint8_t* scale_ptr =
          static_cast<uint8_t*>(__builtin_assume_aligned(p.scales + static_scale_idx + (group & 3), 2));
      *reinterpret_cast<uint16_t*>(scale_ptr) = scale_pair;
    } else {
#pragma unroll
      for (int i = 0; i < GroupsPerItem; i += 4) {
        uint8_t scale0 = process_nvfp4_group_impl<scalar_t, Groups, false, ScaleGroups>(p, row, group + i);
        uint8_t scale1 = process_nvfp4_group_impl<scalar_t, Groups, false, ScaleGroups>(p, row, group + i + 1);
        uint8_t scale2 = process_nvfp4_group_impl<scalar_t, Groups, false, ScaleGroups>(p, row, group + i + 2);
        uint8_t scale3 = process_nvfp4_group_impl<scalar_t, Groups, false, ScaleGroups>(p, row, group + i + 3);
        uint32_t scale_quad = static_cast<uint32_t>(scale0) | (static_cast<uint32_t>(scale1) << 8) |
            (static_cast<uint32_t>(scale2) << 16) | (static_cast<uint32_t>(scale3) << 24);
        int scale_idx = static_scale_idx;
        static_scale_idx += 512;
        uint8_t* scale_ptr = static_cast<uint8_t*>(__builtin_assume_aligned(p.scales + scale_idx, 4));
        *reinterpret_cast<uint32_t*>(scale_ptr) = scale_quad;
      }
    }
  }
};

template <typename scalar_t>
void launch_nvfp4_layout(sycl::queue& queue, Nvfp4Params<scalar_t> const& params) {
  const int64_t total_groups = static_cast<int64_t>(params.rows) * params.groups;
  if (total_groups <= 0) {
    return;
  }
  TORCH_CHECK(total_groups <= std::numeric_limits<int>::max(), "NVFP4 launch grid is too large");
  int64_t global = round_up_int(static_cast<int>(total_groups), kDefaultBlock);
  queue.parallel_for<Nvfp4LayoutKernel<scalar_t>>(
      sycl::nd_range<1>(
          sycl::range<1>(static_cast<std::size_t>(global)),
          sycl::range<1>(static_cast<std::size_t>(kDefaultBlock))),
      Nvfp4LayoutKernel<scalar_t>{params});
}

template <typename scalar_t>
void launch_nvfp4_layout_optimized_1d(sycl::queue& queue, Nvfp4Params<scalar_t> const& p) {
  int total_groups = p.rows * p.groups;
  int global = round_up_int(total_groups, kDefaultBlock);
  queue.parallel_for<Nvfp4LayoutKernel1DOptimized<scalar_t>>(
      sycl::nd_range<1>(
          sycl::range<1>(static_cast<std::size_t>(global)),
          sycl::range<1>(static_cast<std::size_t>(kDefaultBlock))),
      Nvfp4LayoutKernel1DOptimized<scalar_t>{p});
}

template <typename scalar_t, int Groups>
void launch_nvfp4_layout_optimized_2d_static(sycl::queue& queue, Nvfp4Params<scalar_t> const& p) {
  int group_block = choose_group_block(Groups);
  int groups_global = round_up_int(Groups, group_block);
  queue.parallel_for<Nvfp4LayoutKernel2DStaticGroupsOptimized<scalar_t, Groups>>(
      sycl::nd_range<2>(
          sycl::range<2>(static_cast<std::size_t>(p.rows), static_cast<std::size_t>(groups_global)),
          sycl::range<2>(1, static_cast<std::size_t>(group_block))),
      Nvfp4LayoutKernel2DStaticGroupsOptimized<scalar_t, Groups>{p});
}

template <typename scalar_t, int Groups, int GroupsPerItem, int ScaleGroups = Groups>
void launch_nvfp4_layout_optimized_2d_static_group_tile(
    sycl::queue& queue,
    Nvfp4Params<scalar_t> const& p,
    int row_block) {
  static_assert(
      GroupsPerItem == 1 || GroupsPerItem == 2 || GroupsPerItem % 4 == 0,
      "GroupsPerItem must preserve 1-, 2-, or 4-scale swizzle adjacency");
  constexpr int kGroupTiles = Groups / GroupsPerItem;
  int rows_global = round_up_int(p.rows, row_block);
  queue.parallel_for<Nvfp4LayoutKernel2DStaticGroupTileOptimized<scalar_t, Groups, GroupsPerItem, ScaleGroups>>(
      sycl::nd_range<2>(
          sycl::range<2>(static_cast<std::size_t>(rows_global), static_cast<std::size_t>(kGroupTiles)),
          sycl::range<2>(static_cast<std::size_t>(row_block), static_cast<std::size_t>(kGroupTiles))),
      Nvfp4LayoutKernel2DStaticGroupTileOptimized<scalar_t, Groups, GroupsPerItem, ScaleGroups>{p});
}

template <typename scalar_t>
void launch_nvfp4_layout_optimized(sycl::queue& queue, Nvfp4Params<scalar_t> const& p) {
  if (p.groups < 64) {
    if (p.groups == 48) {
      launch_nvfp4_layout_optimized_2d_static_group_tile<scalar_t, 48, 4>(queue, p, 8);
      return;
    }
    if (p.groups == 24) {
      if constexpr (std::is_same_v<scalar_t, sycl::half>) {
        launch_nvfp4_layout_optimized_2d_static_group_tile<scalar_t, 24, 4>(queue, p, 16);
      } else {
        launch_nvfp4_layout_optimized_2d_static_group_tile<scalar_t, 24, 2>(queue, p, 16);
      }
      return;
    }
    if (p.groups == 12) {
      launch_nvfp4_layout_optimized_2d_static_group_tile<scalar_t, 12, 2>(queue, p, 16);
      return;
    }
    launch_nvfp4_layout_optimized_1d(queue, p);
    return;
  }
  if (p.groups == 96) {
    launch_nvfp4_layout_optimized_2d_static_group_tile<scalar_t, 96, 1>(queue, p, 2);
    return;
  }
  if (p.groups == 192) {
    launch_nvfp4_layout_optimized_2d_static<scalar_t, 192>(queue, p);
    return;
  }
  if (p.groups == 384) {
    launch_nvfp4_layout_optimized_2d_static<scalar_t, 384>(queue, p);
    return;
  }

  int group_block = choose_group_block(p.groups);
  int groups_global = round_up_int(p.groups, group_block);
  queue.parallel_for<Nvfp4LayoutKernel2DOptimized<scalar_t>>(
      sycl::nd_range<2>(
          sycl::range<2>(static_cast<std::size_t>(p.rows), static_cast<std::size_t>(groups_global)),
          sycl::range<2>(1, static_cast<std::size_t>(group_block))),
      Nvfp4LayoutKernel2DOptimized<scalar_t>{p});
}

template <typename scalar_t>
void launch_mxfp4_mapping_for_ptr(
    sycl::queue& queue,
    scalar_t const* input,
    at::Tensor& packed,
    at::Tensor& scales,
    int rows,
    int cols,
    int groups,
    double eps) {
  Mxfp4Params<scalar_t> params;
  params.x = input;
  params.packed = packed.data_ptr<uint8_t>();
  params.scales = scales.data_ptr<uint8_t>();
  params.rows = rows;
  params.cols = cols;
  params.groups = groups;
  params.total_groups = rows * groups;
  params.eps = static_cast<float>(eps);
  params.eps_exp = floor_log2_positive(static_cast<float>(eps));
  params.column_major_scales = false;
  launch_mxfp4_mapping(queue, params);
}

template <typename scalar_t>
void launch_nvfp4_layout_for_ptr(
    sycl::queue& queue,
    scalar_t const* input,
    at::Tensor& packed,
    at::Tensor& scales,
    uint64_t const* raw_scale_output_lut,
    int rows,
    int cols,
    int groups,
    int rounded_groups,
    double global_scale) {
  Nvfp4Params<scalar_t> params;
  params.x = input;
  params.packed = packed.data_ptr<uint8_t>();
  params.scales = scales.data_ptr<uint8_t>();
  params.raw_scale_output_lut = raw_scale_output_lut;
  params.rows = rows;
  params.cols = cols;
  params.groups = groups;
  params.rounded_groups = rounded_groups;
  params.global_scale = static_cast<float>(global_scale);
  params.scale_factor = static_cast<float>(global_scale / kE2M1Max);
  launch_nvfp4_layout_optimized(queue, params);
}

uint64_t make_nvfp4_bf16_lut_entry(uint16_t raw, double global_scale) {
  uint16_t raw_abs = raw16_abs_bits(raw);
  float max_abs = sycl::bit_cast<float>(static_cast<uint32_t>(raw_abs) << 16);
  E4M3FnEncodeInvResult scale = e4m3fn_encode_positive_with_inv_decode(max_abs * static_cast<float>(global_scale / kE2M1Max));
  float output_scale = static_cast<float>(global_scale) * scale.inv_decoded;
  uint32_t output_bits = sycl::bit_cast<uint32_t>(output_scale);
  return static_cast<uint64_t>(scale.code) | (static_cast<uint64_t>(output_bits) << 32);
}

at::Tensor get_nvfp4_bf16_raw_scale_output_lut(at::Tensor const& x, double global_scale) {
  struct CacheEntry {
    double global_scale = 0.0;
    c10::DeviceIndex device = -1;
    at::Tensor lut;
  };
  static std::vector<CacheEntry> cache;
  c10::DeviceIndex device = x.get_device();
  for (CacheEntry const& entry : cache) {
    if (entry.lut.defined() && entry.device == device && entry.global_scale == global_scale) {
      return entry.lut;
    }
  }

  at::Tensor host_lut = at::empty({65536}, at::TensorOptions().dtype(at::ScalarType::Long).device(at::kCPU));
  int64_t* host_ptr = host_lut.data_ptr<int64_t>();
  for (int raw = 0; raw < 65536; ++raw) {
    host_ptr[raw] = static_cast<int64_t>(make_nvfp4_bf16_lut_entry(static_cast<uint16_t>(raw), global_scale));
  }
  at::Tensor device_lut = host_lut.to(x.device(), /*non_blocking=*/false);
  cache.push_back(CacheEntry{global_scale, device, device_lut});
  return cache.back().lut;
}

void check_quant_input(at::Tensor const& x, int group_size, char const* name) {
  CHECK_INPUT(x);
  TORCH_CHECK(
      x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half ||
          x.scalar_type() == at::ScalarType::BFloat16,
      name,
      " must be float32, float16, or bfloat16");
  TORCH_CHECK(x.dim() == 2, name, " must be 2D [rows, cols]");
  TORCH_CHECK(x.size(0) > 0, name, " rows must be positive");
  TORCH_CHECK(x.size(1) > 0 && x.size(1) % group_size == 0, name, " cols must be divisible by group size");
  TORCH_CHECK(x.size(1) % 2 == 0, name, " cols must be even for FP4 packing");
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> inkling_mxfp4_mapping(
    const at::Tensor& x,
    bool column_major_scales,
    double eps) {
  check_quant_input(x, kMxfp4GroupSize, "x");
  const int rows = checked_int64_to_int(x.size(0), "rows");
  const int cols = checked_int64_to_int(x.size(1), "cols");
  const int groups = cols / kMxfp4GroupSize;
  TORCH_CHECK(eps > 0.0, "eps must be positive");

  at::Tensor packed = at::empty({rows, cols / 2}, x.options().dtype(at::ScalarType::Byte));
  at::Tensor scales = column_major_scales
      ? at::empty_strided({rows, groups}, {1, rows}, x.options().dtype(at::ScalarType::Byte))
      : at::empty({rows, groups}, x.options().dtype(at::ScalarType::Byte));

  sycl::queue& queue = dpcppGetCurrentQueue();
  at::Tensor quant_scales = scales;
  if (column_major_scales) {
    quant_scales = at::empty({rows, groups}, x.options().dtype(at::ScalarType::Byte));
  }

  if (x.scalar_type() == at::ScalarType::Float) {
    launch_mxfp4_mapping_for_ptr<float>(
        queue, x.data_ptr<float>(), packed, quant_scales, rows, cols, groups, eps);
  } else if (x.scalar_type() == at::ScalarType::Half) {
    launch_mxfp4_mapping_for_ptr<sycl::half>(
        queue,
        reinterpret_cast<sycl::half const*>(x.data_ptr<at::Half>()),
        packed,
        quant_scales,
        rows,
        cols,
        groups,
        eps);
  } else if (x.scalar_type() == at::ScalarType::BFloat16) {
    launch_mxfp4_mapping_for_ptr<sycl::ext::oneapi::bfloat16>(
        queue,
        reinterpret_cast<sycl::ext::oneapi::bfloat16 const*>(x.data_ptr<at::BFloat16>()),
        packed,
        quant_scales,
        rows,
        cols,
        groups,
        eps);
  }

  if (column_major_scales) {
    if ((rows & 7) == 0) {
      launch_mxfp4_scale_transpose<true>(
          queue, quant_scales.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(), rows, groups);
    } else {
      launch_mxfp4_scale_transpose<false>(
          queue, quant_scales.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(), rows, groups);
    }
  }
  return {packed, scales};
}

std::tuple<at::Tensor, at::Tensor> inkling_nvfp4_layout(
    const at::Tensor& x,
    double global_scale) {
  check_quant_input(x, kNvfp4GroupSize, "x");
  TORCH_CHECK(global_scale > 0.0, "global_scale must be positive");
  const int rows = checked_int64_to_int(x.size(0), "rows");
  const int cols = checked_int64_to_int(x.size(1), "cols");
  const int groups = cols / kNvfp4GroupSize;
  const int rounded_rows = round_up_int(rows, 128);
  const int rounded_groups = round_up_int(groups, 4);

  at::Tensor packed = at::empty({rows, cols / 2}, x.options().dtype(at::ScalarType::Byte));
  at::Tensor scales = (rounded_rows == rows && rounded_groups == groups)
      ? at::empty({rounded_rows, rounded_groups}, x.options().dtype(at::ScalarType::Byte))
      : at::zeros({rounded_rows, rounded_groups}, x.options().dtype(at::ScalarType::Byte));

  sycl::queue& queue = dpcppGetCurrentQueue();
  uint64_t const* raw_scale_output_lut = nullptr;
  at::Tensor raw_scale_output_lut_tensor;
  if (x.scalar_type() == at::ScalarType::BFloat16 && groups >= 48) {
    raw_scale_output_lut_tensor = get_nvfp4_bf16_raw_scale_output_lut(x, global_scale);
    raw_scale_output_lut = reinterpret_cast<uint64_t const*>(raw_scale_output_lut_tensor.data_ptr<int64_t>());
  }
  if (x.scalar_type() == at::ScalarType::Float) {
    launch_nvfp4_layout_for_ptr<float>(
        queue, x.data_ptr<float>(), packed, scales, nullptr, rows, cols, groups, rounded_groups, global_scale);
  } else if (x.scalar_type() == at::ScalarType::Half) {
    launch_nvfp4_layout_for_ptr<sycl::half>(
        queue,
        reinterpret_cast<sycl::half const*>(x.data_ptr<at::Half>()),
        packed,
        scales,
        nullptr,
        rows,
        cols,
        groups,
        rounded_groups,
        global_scale);
  } else if (x.scalar_type() == at::ScalarType::BFloat16) {
    launch_nvfp4_layout_for_ptr<sycl::ext::oneapi::bfloat16>(
        queue,
        reinterpret_cast<sycl::ext::oneapi::bfloat16 const*>(x.data_ptr<at::BFloat16>()),
        packed,
        scales,
        raw_scale_output_lut,
        rows,
        cols,
        groups,
        rounded_groups,
        global_scale);
  }
  return {packed, scales};
}
