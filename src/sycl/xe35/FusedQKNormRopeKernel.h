#pragma once

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "FusedQKNormRopeFrequency.h"
#include "../MemoryAccess.h"
#include "../SYCLHelpers.h"
#include "../Utils.h"

namespace at::native::xpu::qk_norm_rope_kernel_xe35 {

constexpr int kSubgroupSize = 16;
constexpr int kBytesPerLane = 32;
constexpr int kTargetWorkgroup = 128;
constexpr int kMaxWorkgroup = 256;
constexpr int kMaxTokensPerWg = 4;
constexpr int64_t kCacheTrafficBytes = 48ll << 20;
constexpr int64_t kDramTrafficBytes = 100ll << 20;
// The first repeatable BF16 128D sparse-row split gain starts at 300 MiB.
constexpr int64_t kSparseRowSplitBytes = 300ll << 20;
constexpr int64_t kSparseRowAlignmentBytes = 4ll << 10;
constexpr int kCacheRoundCap = 9;
constexpr int kDramRoundCap = 3;

struct LaunchConfig_xe35 {
  int wg_lanes;
  int tokens_per_wg;
  int clusters_per_token;
  int heads_per_chunk;
  int chunks_per_token;
  int rounds;
  int64_t groups;
};

inline LaunchConfig_xe35
make_launch_xe35(int64_t num_tokens, int total_qk_heads, int lanes_per_head_xe35, int wg_lanes, int tokens_per_wg, int chunks) {
  LaunchConfig_xe35 config{};
  config.wg_lanes = wg_lanes;
  config.tokens_per_wg = tokens_per_wg;
  config.clusters_per_token = std::max(wg_lanes / lanes_per_head_xe35 / tokens_per_wg, 1);
  config.chunks_per_token = chunks;
  config.rounds = ((total_qk_heads + chunks - 1) / chunks + config.clusters_per_token - 1) / config.clusters_per_token;
  config.heads_per_chunk = config.rounds * config.clusters_per_token;
  config.chunks_per_token = (total_qk_heads + config.heads_per_chunk - 1) / config.heads_per_chunk;
  config.groups = ((num_tokens + tokens_per_wg - 1) / tokens_per_wg) * config.chunks_per_token;
  return config;
}

template <typename scalar_t, int HeadDim, bool Interleave>
inline LaunchConfig_xe35 choose_launch_xe35(int64_t num_tokens, int total_qk_heads, int lanes_per_head_xe35, int row_heads) {
  const int64_t slots_per_sg = std::max<int64_t>(kSubgroupSize / lanes_per_head_xe35, 1);
  const int64_t wave = std::max<int64_t>(dpcppGpuEuCount() * dpcppGpuHWThreadsPerEU(), 1);
  const int64_t work = num_tokens * total_qk_heads;
  const int rounds = static_cast<int>(std::max<int64_t>(1, (work + slots_per_sg * wave - 1) / (slots_per_sg * wave)));
  const int64_t traffic = 2 * work * lanes_per_head_xe35 * kBytesPerLane;
  if constexpr (Interleave) {
    // Keep the existing 64D split. Extend it only to measured BF16 128D rows.
    const bool sparse_row = row_heads >= 2 * total_qk_heads;
    const bool original_split = lanes_per_head_xe35 == 4 && traffic > kDramTrafficBytes;
    const int64_t row_bytes = static_cast<int64_t>(row_heads) * HeadDim * sizeof(scalar_t);
    const bool bf16_128d_split = std::is_same_v<scalar_t, sycl::ext::oneapi::bfloat16> && HeadDim == 128 &&
                                 row_bytes % kSparseRowAlignmentBytes == 0 && traffic >= kSparseRowSplitBytes;
    if (sparse_row && (original_split || bf16_128d_split)) {
      return make_launch_xe35(num_tokens, total_qk_heads, lanes_per_head_xe35, 192, 1, 2);
    }
  }
  const int round_cap = traffic > kDramTrafficBytes    ? kDramRoundCap
                        : traffic > kCacheTrafficBytes ? kCacheRoundCap
                                                       : rounds;

  if (rounds > round_cap) {
    const int cluster_cap = std::max(kMaxWorkgroup / lanes_per_head_xe35, 1);
    const int clusters = std::min((total_qk_heads + round_cap - 1) / round_cap, cluster_cap);
    return make_launch_xe35(num_tokens, total_qk_heads, lanes_per_head_xe35, clusters * lanes_per_head_xe35, 1, 1);
  }

  const int cluster_cap = std::max(kTargetWorkgroup / lanes_per_head_xe35, 1);
  int clusters = 1;
  int best_waste = -1;
  for (int candidate = 1; candidate <= cluster_cap; ++candidate) {
    const int chunks = (total_qk_heads + rounds * candidate - 1) / (rounds * candidate);
    const int waste = chunks * rounds * candidate - total_qk_heads;
    if (best_waste < 0 || waste <= best_waste) {
      clusters = candidate;
      best_waste = waste;
    }
  }

  int tokens_per_wg = 1;
  while (tokens_per_wg < kMaxTokensPerWg && clusters * lanes_per_head_xe35 * tokens_per_wg < kSubgroupSize) {
    tokens_per_wg <<= 1;
  }
  const int chunks = (total_qk_heads + rounds * clusters - 1) / (rounds * clusters);
  return make_launch_xe35(
      num_tokens, total_qk_heads, lanes_per_head_xe35, clusters * lanes_per_head_xe35 * tokens_per_wg, tokens_per_wg, chunks);
}

inline int partner_lanes_for_xe35(int rotary_dim, int elements_per_lane) {
  const int half = rotary_dim / 2;
  if (half <= 0 || half % elements_per_lane != 0) return 0;
  const int lanes = half / elements_per_lane;
  return (lanes & (lanes - 1)) == 0 ? lanes : 0;
}

template <typename scalar_t>
inline int lanes_per_head_xe35(int head_dim) {
  const int wide = kBytesPerLane / sizeof(scalar_t);
  return head_dim / std::min(head_dim, wide);
}

inline bool prefer_baseline_xe35(int64_t num_tokens, int total_qk_heads, int lanes_per_head_xe35) {
  const int64_t slots_per_sg = std::max<int64_t>(kSubgroupSize / lanes_per_head_xe35, 1);
  const int64_t subgroups = (num_tokens * total_qk_heads + slots_per_sg - 1) / slots_per_sg;
  const int64_t wave = std::max<int64_t>(dpcppGpuEuCount() * dpcppGpuHWThreadsPerEU(), 1);
  return subgroups * 4 < wave;
}

template <typename scalar_t, int HeadDim, bool Interleave, bool UseTable>
struct Kernel_xe35 {
  static constexpr int kElementsPerLane = kBytesPerLane / sizeof(scalar_t);
  static constexpr int kLanesPerHead = HeadDim / kElementsPerLane;
  using vec_t = aligned_vector_loop<scalar_t, kElementsPerLane>;

  static_assert(HeadDim % kElementsPerLane == 0);
  static_assert(kSubgroupSize % kLanesPerHead == 0);

  scalar_t* qkv;
  const scalar_t* q_weight;
  const scalar_t* k_weight;
  const int* position_ids;
  float eps;
  float base;
  float factor;
  float low;
  float high;
  float attention_factor;
  int rotary_dim;
  int num_tokens;
  int num_heads_q;
  int num_heads;
  int total_qk_heads;
  LaunchConfig_xe35 config;
  int partner_lanes;
  QKRopeFrequencyStorage_xe35<HeadDim, UseTable> frequencies;

  template <int Delta>
  void rotate_neox(
      const sycl::sub_group& sg,
      const float (&values)[kElementsPerLane],
      const float (&cos_values)[kElementsPerLane],
      const float (&sin_values)[kElementsPerLane],
      int first_dim,
      int half_rotary,
      int rotary_count,
      vec_t& output) const {
#pragma unroll
    for (int i = 0; i < kElementsPerLane; ++i) {
      const float peer = sycl::permute_group_by_xor(sg, values[i], static_cast<uint32_t>(Delta));
      const float rotated_peer = first_dim + i < half_rotary ? -peer : peer;
      output[i] = static_cast<scalar_t>(
          i < rotary_count ? (values[i] * cos_values[i] + rotated_peer * sin_values[i]) * attention_factor : values[i]);
    }
  }

  [[sycl::reqd_sub_group_size(kSubgroupSize)]] void operator()(sycl::nd_item<1> item) const {
    const auto sg = item.get_sub_group();
    const int local_id = static_cast<int>(item.get_local_id(0));
    const int group_id = static_cast<int>(item.get_group(0));
    const int token_group = group_id / config.chunks_per_token;
    const int chunk = group_id - token_group * config.chunks_per_token;
    const int cluster = local_id / kLanesPerHead;
    const int first_dim = (local_id - cluster * kLanesPerHead) * kElementsPerLane;
    const int local_token = cluster / config.clusters_per_token;
    const int slot = cluster - local_token * config.clusters_per_token;
    const int half_rotary = rotary_dim / 2;
    const int token_raw = token_group * config.tokens_per_wg + local_token;
    const bool token_live = token_raw < num_tokens;
    const int token = token_live ? token_raw : num_tokens - 1;

    auto local =
        sycl::ext::oneapi::group_local_memory_for_overwrite<float[kMaxTokensPerWg * 2 * HeadDim]>(item.get_group());
    float* table = *local;
    for (int index = local_id; index < config.tokens_per_wg * HeadDim; index += config.wg_lanes) {
      const int local_table_token = index / HeadDim;
      const int dim = index - local_table_token * HeadDim;
      const int table_token = std::min(token_group * config.tokens_per_wg + local_table_token, num_tokens - 1);
      const int half_dim = Interleave ? dim / 2 : (dim < half_rotary ? dim : dim - half_rotary);
      const float theta =
          dim < rotary_dim ? static_cast<float>(position_ids[table_token]) * frequencies[half_dim] : 0.0f;
      table[local_table_token * 2 * HeadDim + dim] = sycl::sin(theta);
      table[local_table_token * 2 * HeadDim + HeadDim + dim] = sycl::cos(theta);
    }
    sycl::group_barrier(item.get_group());

    float sin_values[kElementsPerLane];
    float cos_values[kElementsPerLane];
    const float* token_table = table + local_token * 2 * HeadDim;
#pragma unroll
    for (int i = 0; i < kElementsPerLane; ++i) {
      sin_values[i] = token_table[first_dim + i];
      cos_values[i] = token_table[HeadDim + first_dim + i];
    }

    const int rotary_count = rotary_dim - first_dim;
    const int64_t token_offset = static_cast<int64_t>(token) * num_heads * HeadDim;
    const int base_head = chunk * config.heads_per_chunk;
    const int heads_here = std::min(total_qk_heads - base_head, config.heads_per_chunk);
    const int rounds = (heads_here + config.clusters_per_token - 1) / config.clusters_per_token;
    auto head_for = [&](int round) {
      return std::min(base_head + round * config.clusters_per_token + slot, base_head + heads_here - 1);
    };
    auto row_for = [&](int head) { return qkv + token_offset + static_cast<int64_t>(head) * HeadDim; };

    float weights[kElementsPerLane];
    bool weight_is_q = false;
    bool weight_valid = false;
    for (int round = 0; round < rounds; ++round) {
      const int head = head_for(round);
      const vec_t input = *reinterpret_cast<const vec_t*>(row_for(head) + first_dim);
      const bool store = token_live && base_head + round * config.clusters_per_token + slot < base_head + heads_here;

      float square_sum = 0.0f;
#pragma unroll
      for (int i = 0; i < kElementsPerLane; ++i) {
        const float value = static_cast<float>(input[i]);
        square_sum += value * value;
      }
#pragma unroll
      for (int step = 1; step < kLanesPerHead; step <<= 1) {
        square_sum += sycl::permute_group_by_xor(sg, square_sum, static_cast<uint32_t>(step));
      }

      const float reciprocal_rms = sycl::rsqrt(square_sum / static_cast<float>(HeadDim) + eps);
      const bool is_q = head < num_heads_q;
      if (!weight_valid || is_q != weight_is_q) {
        const vec_t weight = *reinterpret_cast<const vec_t*>((is_q ? q_weight : k_weight) + first_dim);
#pragma unroll
        for (int i = 0; i < kElementsPerLane; ++i)
          weights[i] = static_cast<float>(weight[i]);
        weight_is_q = is_q;
        weight_valid = true;
      }

      float values[kElementsPerLane];
#pragma unroll
      for (int i = 0; i < kElementsPerLane; ++i) {
        values[i] = static_cast<float>(input[i]) * reciprocal_rms * weights[i];
      }

      vec_t output;
      if constexpr (Interleave) {
#pragma unroll
        for (int i = 0; i < kElementsPerLane; i += 2) {
          const float first = values[i];
          const float second = values[i + 1];
          output[i] = static_cast<scalar_t>(
              i < rotary_count ? (first * cos_values[i] - second * sin_values[i]) * attention_factor : first);
          output[i + 1] = static_cast<scalar_t>(
              i + 1 < rotary_count ? (second * cos_values[i + 1] + first * sin_values[i + 1]) * attention_factor
                                   : second);
        }
      } else {
        switch (partner_lanes) {
          case 1:
            rotate_neox<1>(sg, values, cos_values, sin_values, first_dim, half_rotary, rotary_count, output);
            break;
          case 2:
            rotate_neox<2>(sg, values, cos_values, sin_values, first_dim, half_rotary, rotary_count, output);
            break;
          case 4:
            rotate_neox<4>(sg, values, cos_values, sin_values, first_dim, half_rotary, rotary_count, output);
            break;
          case 8:
            rotate_neox<8>(sg, values, cos_values, sin_values, first_dim, half_rotary, rotary_count, output);
            break;
          default: {
            const scalar_t* row = row_for(head);
            const scalar_t* weight = is_q ? q_weight : k_weight;
#pragma unroll
            for (int i = 0; i < kElementsPerLane; ++i) {
              if (i >= rotary_count) {
                output[i] = static_cast<scalar_t>(values[i]);
                continue;
              }
              const int dim = first_dim + i;
              const bool lower = dim < half_rotary;
              const int peer_dim = lower ? dim + half_rotary : dim - half_rotary;
              const float peer =
                  static_cast<float>(row[peer_dim]) * reciprocal_rms * static_cast<float>(weight[peer_dim]);
              output[i] = static_cast<scalar_t>(
                  (values[i] * cos_values[i] + (lower ? -peer : peer) * sin_values[i]) * attention_factor);
            }
            break;
          }
        }
      }
      if (store) *reinterpret_cast<vec_t*>(row_for(head) + first_dim) = output;
    }
  }
};

template <typename scalar_t, int HeadDim, bool Interleave>
void launch_xe35(
    scalar_t* qkv,
    int num_tokens,
    int num_heads_q,
    int num_heads_k,
    int num_heads_v,
    float eps,
    const scalar_t* q_weight,
    const scalar_t* k_weight,
    float base,
    const int* position_ids,
    float factor,
    float low,
    float high,
    float attention_factor,
    int rotary_dim,
    sycl::queue& queue) {
  constexpr int lanes = Kernel_xe35<scalar_t, HeadDim, Interleave, true>::kLanesPerHead;
  const LaunchConfig_xe35 config = choose_launch_xe35<scalar_t, HeadDim, Interleave>(
      num_tokens, num_heads_q + num_heads_k, lanes, num_heads_q + num_heads_k + num_heads_v);
  const float* table = qk_rope_cached_frequencies_xe35(queue, base, rotary_dim, factor, low, high);
  auto submit = [&]<bool UseTable>() {
    Kernel_xe35<scalar_t, HeadDim, Interleave, UseTable> kernel{
        qkv,
        q_weight,
        k_weight,
        position_ids,
        eps,
        base,
        factor,
        low,
        high,
        attention_factor,
        rotary_dim,
        num_tokens,
        num_heads_q,
        num_heads_q + num_heads_k + num_heads_v,
        num_heads_q + num_heads_k,
        config,
        partner_lanes_for_xe35(rotary_dim, Kernel_xe35<scalar_t, HeadDim, Interleave, UseTable>::kElementsPerLane),
        qk_rope_frequency_storage_xe35<HeadDim, UseTable>(table, base, rotary_dim, factor, low, high)};
    sycl_kernel_submit(sycl::range<1>(config.groups * config.wg_lanes), sycl::range<1>(config.wg_lanes), queue, kernel);
  };
  if (table)
    submit.template operator()<true>();
  else
    submit.template operator()<false>();
}

}  // namespace at::native::xpu::qk_norm_rope_kernel_xe35
