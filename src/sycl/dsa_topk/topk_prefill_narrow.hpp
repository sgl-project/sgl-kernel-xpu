// Adapted from vllm-project/vllm-xpu-kernels, csrc/topk_per_row.cpp.
#pragma once

#include "topk_prefill_medium.hpp"

namespace sgl_kernel::prefill_topk::narrow {

constexpr int kThreads = 512;
constexpr int kCopies = 4;
constexpr int kMaxCandidates = 128;
constexpr int kFinalItems = 512;
constexpr int kRowCapacity = kThreads * kItems;
constexpr int kRefineShift = 8;
constexpr uint32_t kRefineSlack = 1u << 13;
constexpr int kRefineBuckets = 1536;

struct Scratch {
  int indices[kFinalItems];
  uint32_t keys[kFinalItems];
  alignas(16) int bins[kBins * kCopies];
  int tmp[32];
  int vars[5];
  int output[2048];
};

inline bool reduce(const sycl::nd_item<3>& item, const float* input, int length, int* output, Scratch& s, int topk) {
  const auto group = item.get_group();
  const auto sg = item.get_sub_group();
  const int tid = item.get_local_id(0);
  const int lane = tid % kSubgroupSize;
  const int sg_id = tid / kSubgroupSize;
  constexpr int kBinsPerThread = kBins / kThreads;
  const int bin_base = tid * kBinsPerThread;
  const int hist_copy = lane % kCopies;
  int* accept_cursor = s.tmp + 28;
  int* candidate_cursor = s.tmp + 24;
  const int reg_items = (length - tid + kThreads - 1) / kThreads;
  float values[kItems];
  int bins[kItems];

  int zero[kBinsPerThread * kCopies] = {};
  bins_store(s.bins + bin_base * kCopies, zero);
  sycl::group_barrier(group);
#pragma unroll
  for (int i = 0; i < kItems; ++i) {
    if (i >= reg_items) break;
    values[i] = input[tid + i * kThreads];
  }
#pragma unroll
  for (int i = 0; i < kItems; ++i) {
    if (i >= reg_items) break;
    bins[i] = bin_of(values[i]);
    atomic_add(s.bins + bins[i] * kCopies + hist_copy, 1);
  }
  sycl::group_barrier(group);

  int raw[kBinsPerThread * kCopies];
  bins_load(s.bins + bin_base * kCopies, raw);
  int counts[kBinsPerThread];
  int sum = 0;
#pragma unroll
  for (int b = 0; b < kBinsPerThread; ++b) {
    counts[b] = 0;
#pragma unroll
    for (int c = 0; c < kCopies; ++c)
      counts[b] += raw[b * kCopies + c];
    sum += counts[b];
  }
  int total = 0;
  int prefix = block_exclusive_scan<kThreads>(group, sg, lane, sg_id, s.tmp, sum, total);
#pragma unroll
  for (int b = 0; b < kBinsPerThread; ++b) {
    const int inclusive = prefix + counts[b];
    if (prefix < topk && topk <= inclusive) {
      s.vars[0] = bin_base + b;
      s.vars[1] = counts[b];
      s.vars[2] = prefix;
      *accept_cursor = 0;
      *candidate_cursor = prefix;
    }
    prefix = inclusive;
  }
  sycl::group_barrier(group);
  if (s.vars[1] > kMaxCandidates) return false;

  const int threshold = s.vars[0];
  const int accepted = s.vars[2];
  const int sg_items = sycl::group_broadcast(sg, reg_items, 0);
  const uint32_t lower_lanes = (1u << lane) - 1u;
  uint32_t accepted_mask[kItems];
  int sg_accepted = 0;
#pragma unroll
  for (int i = 0; i < kItems; ++i) {
    if (i >= sg_items) break;
    const bool live = i < reg_items;
    sycl::ext::oneapi::group_ballot(sg, live && bins[i] < threshold).extract_bits(accepted_mask[i]);
    sg_accepted += sycl::popcount(accepted_mask[i]);
    if (live && bins[i] == threshold) {
      const int slot = atomic_add(candidate_cursor, 1) - accepted;
      s.keys[slot] = float_key(values[i]);
      s.indices[slot] = tid + i * kThreads;
    }
  }
  int base = 0;
  if (lane == 0 && sg_accepted > 0) base = atomic_add(accept_cursor, sg_accepted);
  base = sycl::group_broadcast(sg, base, 0);
#pragma unroll
  for (int i = 0; i < kItems; ++i) {
    if (i >= sg_items) break;
    if ((accepted_mask[i] >> lane) & 1u)
      output[base + sycl::popcount(accepted_mask[i] & lower_lanes)] = tid + i * kThreads;
    base += sycl::popcount(accepted_mask[i]);
  }
  sycl::group_barrier(group);

  const int candidates = s.vars[1];
  const int remaining = topk - accepted;
  for (int i = sg_id; i < candidates; i += kThreads / kSubgroupSize) {
    const uint32_t value_key = s.keys[i];
    int rank = 0;
    for (int j = lane; j < candidates; j += kSubgroupSize) {
      const uint32_t other_key = s.keys[j];
      if (value_key > other_key || (value_key == other_key && i < j)) ++rank;
    }
    rank = sycl::reduce_over_group(sg, rank, sycl::plus<int>());
    if (lane == 0 && rank < remaining) output[accepted + rank] = s.indices[i];
  }
  return true;
}

template <typename Func>
inline void vectorized_walk(const sycl::nd_item<3>& item, const float* input, int length, Func fn) {
  constexpr int kDepth = 4;
  const int tid = item.get_local_id(0);
  const int misalign = reinterpret_cast<uintptr_t>(input) % sizeof(sycl::float4);
  const int head =
      sycl::min(misalign ? static_cast<int>((sizeof(sycl::float4) - misalign) / sizeof(float)) : 0, length);
  const auto* in4 = reinterpret_cast<const sycl::float4*>(input + head);
  const int len4 = (length - head) / 4;
  constexpr int kBlock = kDepth * kThreads;
  const int blocks = (len4 + kBlock - 1) / kBlock;
  const int first = blocks > 0 ? item.get_group_linear_id() % blocks : 0;
  for (int b = 0; b < blocks; ++b) {
    const int rotated = b + first;
    const int base = (rotated < blocks ? rotated : rotated - blocks) * kBlock + tid;
    sycl::float4 values[kDepth];
#pragma unroll
    for (int d = 0; d < kDepth; ++d)
      if (base + d * kThreads < len4) values[d] = in4[base + d * kThreads];
#pragma unroll
    for (int d = 0; d < kDepth; ++d) {
      if (base + d * kThreads < len4) {
        const int offset = head + (base + d * kThreads) * 4;
        fn(values[d].x(), offset);
        fn(values[d].y(), offset + 1);
        fn(values[d].z(), offset + 2);
        fn(values[d].w(), offset + 3);
      }
    }
  }
  if (tid < head) fn(input[tid], tid);
  const int tail = head + len4 * 4 + tid;
  if (tail < length) fn(input[tail], tail);
}

inline uint16_t bin_bits(int bin) {
  const uint32_t key = static_cast<uint32_t>(bin) << 5;
  return static_cast<uint16_t>((key & 0x8000u) ? key : (~key & 0x7fffu));
}

inline bool refinable(int bin) {
  const int exponent = (bin_bits(bin) >> 10) & 0x1f;
  return exponent != 0 && exponent != 0x1f;
}

inline bool refine(
    const sycl::nd_item<3>& item,
    const float* input,
    int length,
    int fast_bin,
    int accepted,
    int* output,
    Scratch& s,
    int topk) {
  const auto group = item.get_group();
  const auto sg = item.get_sub_group();
  const int tid = item.get_local_id(0);
  const int lane = tid % kSubgroupSize;
  const int sg_id = tid / kSubgroupSize;
  constexpr int kBinsPerThread = kRefineBuckets / kThreads;
  static_assert(
      kRefineBuckets == (((1 << 18) + 2 * kRefineSlack) / (1 << kRefineShift) + kThreads - 1) / kThreads * kThreads);
  const uint32_t base = float_key(static_cast<float>(sycl::bit_cast<sycl::half>(bin_bits(fast_bin)))) - kRefineSlack;
  auto bucket_of = [&](float x) {
    return static_cast<int>(sycl::min((float_key(x) - base) >> kRefineShift, uint32_t{kRefineBuckets - 1}));
  };
  for (int i = tid; i < kRefineBuckets; i += kThreads)
    s.bins[i] = 0;
  if (tid == 0) {
    s.vars[3] = 0;
    s.vars[4] = 0;
  }
  sycl::group_barrier(group);
  vectorized_walk(item, input, length, [&](float x, int) {
    if (bin_of(x) == fast_bin) atomic_add(s.bins + bucket_of(x), 1);
  });
  sycl::group_barrier(group);

  const int bin_base = tid * kBinsPerThread;
  int sum = 0;
  for (int i = 0; i < kBinsPerThread; ++i)
    sum += s.bins[bin_base + i];
  int total = 0;
  int prefix = accepted + block_exclusive_scan<kThreads>(group, sg, lane, sg_id, s.tmp, sum, total);
  for (int i = 0; i < kBinsPerThread; ++i) {
    const int count = s.bins[bin_base + i];
    s.bins[bin_base + i] = prefix;
    if (prefix < topk && topk <= prefix + count) {
      s.vars[0] = bin_base + i;
      s.vars[1] = count;
      s.vars[2] = prefix;
    }
    prefix += count;
  }
  sycl::group_barrier(group);
  if (s.vars[1] > kFinalItems) return false;

  const int threshold = s.vars[0];
  vectorized_walk(item, input, length, [&](float x, int index) {
    const int bin = bin_of(x);
    if (bin < fast_bin) {
      s.output[atomic_add(s.vars + 4, 1)] = index;
    } else if (bin == fast_bin) {
      const int bucket = bucket_of(x);
      if (bucket < threshold) {
        s.output[atomic_add(s.bins + bucket, 1)] = index;
      } else if (bucket == threshold) {
        const int slot = atomic_add(s.vars + 3, 1);
        s.keys[slot] = float_key(x);
        s.indices[slot] = index;
      }
    }
  });
  sycl::group_barrier(group);

  const int candidates = s.vars[3];
  const int final_base = s.vars[2];
  for (int i = tid; i < candidates; i += kThreads) {
    const uint32_t value_key = s.keys[i];
    int rank = 0;
    for (int j = 0; j < candidates; ++j) {
      const uint32_t other_key = s.keys[j];
      if (value_key > other_key || (value_key == other_key && i < j)) ++rank;
    }
    if (rank < topk - final_base) s.output[final_base + rank] = s.indices[i];
  }
  sycl::group_barrier(group);
  for (int i = tid; i < topk; i += kThreads)
    output[i] = s.output[i];
  return true;
}

inline void select(const sycl::nd_item<3>& item, const float* input, int length, int* output, Scratch& s, int topk) {
  if (length <= topk) {
    for (int i = item.get_local_id(0); i < topk; i += kThreads)
      output[i] = i < length ? i : -1;
    return;
  }
  if (reduce(item, input, length, output, s, topk)) return;
  const int threshold = s.vars[0];
  const int accepted = s.vars[2];
  if (refinable(threshold) && refine(item, input, length, threshold, accepted, output, s, topk)) return;
  tie_select<kThreads, kBins, 12>(item, input, length, threshold, accepted, output, s.bins, s.tmp, s.vars, topk);
}

}  // namespace sgl_kernel::prefill_topk::narrow
