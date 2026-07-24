/* Copyright 2025 SGLang Team. All Rights Reserved.
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *
 * This file adapts the Inkling BMG mel-bin embedding lookup + sum kernel from
 * /data2/syk/cutlass-sycl/examples/23_bmg_mel_embedding_sum for the
 * sgl-kernel XPU extension ABI.
 */

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <sycl/sycl.hpp>
#include <type_traits>
#include <vector>

#include "Utils.h"

namespace {

constexpr int kChannelBlock = 256;
constexpr int kDefaultChunkSize = 512;

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
inline void load_add_vec8_device(scalar_t const* ptr, float* accum) {
  uint64_t raw0 = *reinterpret_cast<uint64_t const*>(ptr);
  uint64_t raw1 = *reinterpret_cast<uint64_t const*>(ptr + 4);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    accum[i] += to_float_device(from_raw16_device<scalar_t>(static_cast<uint16_t>(raw0 >> (16 * i))));
    accum[i + 4] += to_float_device(from_raw16_device<scalar_t>(static_cast<uint16_t>(raw1 >> (16 * i))));
  }
}

template <typename scalar_t>
inline void store_vec8_device(scalar_t* ptr, float const* accum) {
  uint64_t raw0 = 0;
  uint64_t raw1 = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    raw0 |= static_cast<uint64_t>(to_raw16_device(from_float_device<scalar_t>(accum[i]))) << (16 * i);
    raw1 |= static_cast<uint64_t>(to_raw16_device(from_float_device<scalar_t>(accum[i + 4]))) << (16 * i);
  }
  *reinterpret_cast<uint64_t*>(ptr) = raw0;
  *reinterpret_cast<uint64_t*>(ptr + 4) = raw1;
}

inline int ceil_div(int x, int y) {
  return (x + y - 1) / y;
}

template <typename scalar_t>
struct MelEmbeddingSumParams {
  int32_t const* __restrict__ features = nullptr;
  scalar_t const* __restrict__ weight = nullptr;
  scalar_t* __restrict__ out = nullptr;
  int tokens = 0;
  int n_mel_bins = 0;
  int mel_vocab_size = 0;
  int hidden = 0;
  int token_offset = 0;
  int chunk_tokens = 0;
};

template <typename scalar_t, int ChannelsPerItem>
struct MelEmbeddingSumKernel {
  MelEmbeddingSumParams<scalar_t> params;
  sycl::local_accessor<int32_t, 1> local_features;

  void operator()(sycl::nd_item<2> item) const {
    constexpr int kChannelsPerGroup = kChannelBlock * ChannelsPerItem;
    const int token_in_chunk = static_cast<int>(item.get_group(0));
    const int channel_tile = static_cast<int>(item.get_group(1));
    const int lane = static_cast<int>(item.get_local_id(1));
    const int token = params.token_offset + token_in_chunk;

    for (int mel = lane; mel < params.n_mel_bins; mel += kChannelBlock) {
      local_features[mel] = params.features[token * params.n_mel_bins + mel];
    }
    item.barrier(sycl::access::fence_space::local_space);

    const int channel_base = channel_tile * kChannelsPerGroup + lane * ChannelsPerItem;
    float accum[ChannelsPerItem];
#pragma unroll
    for (int i = 0; i < ChannelsPerItem; ++i) {
      accum[i] = 0.0f;
    }

    for (int mel = 0; mel < params.n_mel_bins; ++mel) {
      const int feature = local_features[mel];
      const int64_t row = static_cast<int64_t>(mel) * params.mel_vocab_size + feature;
      scalar_t const* weight_row = params.weight + row * params.hidden;
      if constexpr (ChannelsPerItem == 8 && !std::is_same_v<scalar_t, float>) {
        if ((params.hidden % 8) == 0 && channel_base + ChannelsPerItem - 1 < params.hidden) {
          load_add_vec8_device(weight_row + channel_base, accum);
          continue;
        }
      }
#pragma unroll
      for (int i = 0; i < ChannelsPerItem; ++i) {
        const int channel = channel_base + i;
        if (channel < params.hidden) {
          accum[i] += to_float_device(weight_row[channel]);
        }
      }
    }

    scalar_t* out_row = params.out + static_cast<int64_t>(token) * params.hidden;
    if constexpr (ChannelsPerItem == 8 && !std::is_same_v<scalar_t, float>) {
      if ((params.hidden % 8) == 0 && channel_base + ChannelsPerItem - 1 < params.hidden) {
        store_vec8_device(out_row + channel_base, accum);
        return;
      }
    }
#pragma unroll
    for (int i = 0; i < ChannelsPerItem; ++i) {
      const int channel = channel_base + i;
      if (channel < params.hidden) {
        out_row[channel] = from_float_device<scalar_t>(accum[i]);
      }
    }
  }
};

template <typename scalar_t, int ChannelsPerItem>
void launch_chunk_kernel(sycl::queue& queue, MelEmbeddingSumParams<scalar_t> params) {
  constexpr int kChannelsPerGroup = kChannelBlock * ChannelsPerItem;
  const int channel_tiles = ceil_div(params.hidden, kChannelsPerGroup);
  const int global_channels = channel_tiles * kChannelBlock;
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<int32_t, 1> local_features(
        sycl::range<1>(static_cast<std::size_t>(params.n_mel_bins)), cgh);
    MelEmbeddingSumKernel<scalar_t, ChannelsPerItem> kernel{params, local_features};
    cgh.parallel_for<MelEmbeddingSumKernel<scalar_t, ChannelsPerItem>>(
        sycl::nd_range<2>(
            sycl::range<2>(
                static_cast<std::size_t>(params.chunk_tokens), static_cast<std::size_t>(global_channels)),
            sycl::range<2>(1, kChannelBlock)),
        kernel);
  });
}

template <typename scalar_t>
void launch_mel_embedding_sum(
    sycl::queue& queue,
    MelEmbeddingSumParams<scalar_t> base_params,
    int chunk_size,
    int channels_per_item) {
  if (base_params.tokens == 0 || base_params.hidden == 0) {
    return;
  }
  for (int start = 0; start < base_params.tokens; start += chunk_size) {
    MelEmbeddingSumParams<scalar_t> params = base_params;
    params.token_offset = start;
    params.chunk_tokens = std::min(chunk_size, base_params.tokens - start);
    if (channels_per_item == 8) {
      launch_chunk_kernel<scalar_t, 8>(queue, params);
    } else if (channels_per_item == 4) {
      launch_chunk_kernel<scalar_t, 4>(queue, params);
    } else if (channels_per_item == 2) {
      launch_chunk_kernel<scalar_t, 2>(queue, params);
    } else {
      launch_chunk_kernel<scalar_t, 1>(queue, params);
    }
  }
}

int choose_channels_per_item(int tokens, int n_mel_bins, int mel_vocab_size, int hidden, int channels_per_item) {
  if (channels_per_item != 0) {
    return channels_per_item;
  }
  if (hidden >= 1536 && n_mel_bins >= 64) {
    return 8;
  }
  if (hidden >= 4096 && (tokens >= 8192 || mel_vocab_size >= 128)) {
    return 4;
  }
  return hidden >= 2048 ? 2 : 1;
}

int checked_int64_to_int(int64_t value, char const* name) {
  TORCH_CHECK(value >= 0 && value <= std::numeric_limits<int>::max(), name, " must fit in int32, got ", value);
  return static_cast<int>(value);
}

}  // namespace

at::Tensor inkling_mel_embedding_sum(
    const at::Tensor& features,
    const at::Tensor& weight,
    int64_t chunk_size,
    int64_t channels_per_item) {
  CHECK_INPUT(features);
  CHECK_INPUT(weight);
  TORCH_CHECK(features.scalar_type() == at::ScalarType::Int, "features must be int32");
  TORCH_CHECK(
      weight.scalar_type() == at::ScalarType::Float || weight.scalar_type() == at::ScalarType::Half ||
          weight.scalar_type() == at::ScalarType::BFloat16,
      "weight must be float32, float16, or bfloat16");
  TORCH_CHECK(features.dim() == 2, "features must be 2D [tokens, n_mel_bins]");
  TORCH_CHECK(weight.dim() == 2, "weight must be 2D [n_mel_bins * mel_vocab_size, hidden]");
  TORCH_CHECK(features.size(1) > 0, "features must have at least one mel bin");
  TORCH_CHECK(weight.size(0) > 0, "weight must have at least one row");
  TORCH_CHECK(weight.size(0) % features.size(1) == 0, "weight rows must be divisible by n_mel_bins");
  TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
  TORCH_CHECK(
      channels_per_item == 0 || channels_per_item == 1 || channels_per_item == 2 || channels_per_item == 4 ||
          channels_per_item == 8,
      "channels_per_item must be 0, 1, 2, 4, or 8");

  const int tokens = checked_int64_to_int(features.size(0), "tokens");
  const int n_mel_bins = checked_int64_to_int(features.size(1), "n_mel_bins");
  const int mel_vocab_size = checked_int64_to_int(weight.size(0) / features.size(1), "mel_vocab_size");
  const int hidden = checked_int64_to_int(weight.size(1), "hidden");
  const int chunk = checked_int64_to_int(chunk_size, "chunk_size");
  const int cpi = choose_channels_per_item(tokens, n_mel_bins, mel_vocab_size, hidden, static_cast<int>(channels_per_item));

  at::Tensor out = at::empty({features.size(0), weight.size(1)}, weight.options());
  if (tokens == 0 || hidden == 0) {
    return out;
  }

  sycl::queue& queue = dpcppGetCurrentQueue();
  DISPATCH_FLOAT_TYPES(weight.scalar_type(), "inkling_mel_embedding_sum", [&] {
    MelEmbeddingSumParams<scalar_t> params;
    params.features = features.data_ptr<int32_t>();
    params.weight = weight.data_ptr<scalar_t>();
    params.out = out.data_ptr<scalar_t>();
    params.tokens = tokens;
    params.n_mel_bins = n_mel_bins;
    params.mel_vocab_size = mel_vocab_size;
    params.hidden = hidden;
    launch_mel_embedding_sum(queue, params, chunk, cpi);
  });
  return out;
}
