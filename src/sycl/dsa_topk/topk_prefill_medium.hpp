// Adapted from vllm-project/vllm-xpu-kernels, csrc/topk_per_row.cpp.
#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

namespace sgl_kernel::prefill_topk {

constexpr int kSubgroupSize = 32;
constexpr int kThreads = 128;
constexpr int kBins = 2048;
constexpr int kItems = 8;
// Avoid full-row rescans for moderately populated threshold bins.
constexpr int kMaxCandidates = 256;
constexpr int kRowCapacity = 10240;

struct Scratch {
  alignas(16) int bins[kBins];
  int tmp[32];
  int indices[kMaxCandidates];
  uint32_t keys[kMaxCandidates];
  int vars[5];
};

inline int atomic_add(int* address, int value) {
  sycl::
      atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::local_space>
          atom(*address);
  return atom.fetch_add(value);
}

inline uint32_t float_key(float x) {
  const uint32_t bits = sycl::bit_cast<uint32_t>(x);
  if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0) return 0;
  return (bits & 0x80000000u) ? bits : (~bits & 0x7fffffffu);
}

inline int bin_of(float x) {
  const uint16_t bits = sycl::bit_cast<uint16_t>(static_cast<sycl::half>(x));
  const bool nan = ((bits & 0x7C00u) == 0x7C00u) & ((bits & 0x03FFu) != 0);
  const uint16_t key = (bits & 0x8000u) ? bits : (~bits & 0x7fffu);
  const uint16_t mask = static_cast<uint16_t>(0u - static_cast<uint16_t>(!nan));
  return (key >> 5) & mask;
}

template <int N>
inline void bins_load(const int* base, int (&out)[N]) {
  static_assert(N % 4 == 0);
#pragma unroll
  for (int j = 0; j < N; j += 4) {
    const sycl::vec<int, 4> v = *reinterpret_cast<const sycl::vec<int, 4>*>(base + j);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      out[j + i] = v[i];
  }
}

template <int N>
inline void bins_store(int* base, const int (&in)[N]) {
  static_assert(N % 4 == 0);
#pragma unroll
  for (int j = 0; j < N; j += 4) {
    sycl::vec<int, 4> v;
#pragma unroll
    for (int i = 0; i < 4; ++i)
      v[i] = in[j + i];
    *reinterpret_cast<sycl::vec<int, 4>*>(base + j) = v;
  }
}

template <int Threads>
inline int block_exclusive_scan(
    const sycl::group<3>& group, const sycl::sub_group& sg, int lane, int sg_id, int* tmp, int value, int& total) {
  constexpr int kSubgroups = Threads / kSubgroupSize;
  int inclusive = value;
  for (int offset = 1; offset < kSubgroupSize; offset *= 2) {
    const int other = sycl::shift_group_right(sg, inclusive, offset);
    if (lane >= offset) inclusive += other;
  }
  if (lane == kSubgroupSize - 1) tmp[sg_id] = inclusive;
  sycl::group_barrier(group);
  if (sg_id == 0) {
    const int v = lane < kSubgroups ? tmp[lane] : 0;
    int inc = v;
    for (int offset = 1; offset < kSubgroupSize; offset *= 2) {
      const int other = sycl::shift_group_right(sg, inc, offset);
      if (lane >= offset) inc += other;
    }
    if (lane < kSubgroups) {
      tmp[lane] = inc - v;
      if (lane == kSubgroups - 1) tmp[kSubgroups] = inc;
    }
  }
  sycl::group_barrier(group);
  total = tmp[kSubgroups];
  return tmp[sg_id] + inclusive - value;
}

inline bool reduce(const sycl::nd_item<3>& item, const float* input, int length, int* output, Scratch& s, int topk) {
  const auto group = item.get_group();
  const auto sg = item.get_sub_group();
  const int tid = item.get_local_id(0);
  const int lane = tid % kSubgroupSize;
  const int sg_id = tid / kSubgroupSize;
  constexpr int kBinsPerThread = kBins / kThreads;
  constexpr int kChunk = kItems * kThreads;
  const int bin_base = tid * kBinsPerThread;
  int* accept_cursor = s.tmp + 28;
  int* candidate_cursor = s.tmp + 24;
  const int num_chunks = (length + kChunk - 1) / kChunk;
  const int first_chunk = item.get_group_linear_id() % num_chunks;
  auto chunk_at = [&](int c) {
    const int k = c + first_chunk;
    return (k < num_chunks ? k : k - num_chunks) * kChunk;
  };
  auto load_chunk = [&](int chunk, float(&values)[kItems], int(&offsets)[kItems]) {
#pragma unroll
    for (int i = 0; i < kItems; ++i) {
      offsets[i] = chunk + tid + i * kThreads;
      values[i] = offsets[i] < length ? input[offsets[i]] : 0.f;
      offsets[i] = offsets[i] < length ? offsets[i] : length;
    }
  };

  int zero[kBinsPerThread] = {};
  bins_store(s.bins + bin_base, zero);
  sycl::group_barrier(group);

  constexpr int kHotBins = 4;
  constexpr int kHotBits = 16;
  int hot_lo = 0;
  int best = -1;
  const int sample = bin_of(input[tid < length ? tid : 0]);
  const int lead = sycl::group_broadcast(sg, sample, 0);
#pragma unroll
  for (int w = -(kHotBins - 1); w <= kHotBins - 1; ++w) {
    const int lo = sycl::clamp(lead + w, 0, kBins - kHotBins);
    uint32_t mask;
    sycl::ext::oneapi::group_ballot(sg, static_cast<uint32_t>(sample - lo) < kHotBins).extract_bits(mask);
    const int covered = sycl::popcount(mask);
    if (covered > best) {
      best = covered;
      hot_lo = lo;
    }
  }
  const bool hot_loop = best >= 16;
  if (hot_loop) {
    uint64_t hot_count = 0;
    for (int c = 0; c < num_chunks; ++c) {
      float values[kItems];
      int offsets[kItems];
      load_chunk(chunk_at(c), values, offsets);
#pragma unroll
      for (int i = 0; i < kItems; ++i) {
        const int bin = offsets[i] < length ? bin_of(values[i]) : -1;
        const uint32_t d = static_cast<uint32_t>(bin - hot_lo);
        if (d < kHotBins)
          hot_count += uint64_t{1} << (d * kHotBits);
        else if (bin >= 0)
          atomic_add(s.bins + bin, 1);
      }
    }
#pragma unroll
    for (int k = 0; k < kHotBins; ++k) {
      const int count = (hot_count >> (k * kHotBits)) & ((uint64_t{1} << kHotBits) - 1);
      const int total = sycl::reduce_over_group(sg, count, sycl::plus<int>());
      if (lane == 0 && total > 0) atomic_add(s.bins + hot_lo + k, total);
    }
  } else {
    for (int c = 0; c < num_chunks; ++c) {
      float values[kItems];
      int offsets[kItems];
      load_chunk(chunk_at(c), values, offsets);
#pragma unroll
      for (int i = 0; i < kItems; ++i)
        if (offsets[i] < length) atomic_add(s.bins + bin_of(values[i]), 1);
    }
  }
  sycl::group_barrier(group);

  int counts[kBinsPerThread];
  bins_load(s.bins + bin_base, counts);
  int sum = 0;
#pragma unroll
  for (int b = 0; b < kBinsPerThread; ++b)
    sum += counts[b];
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
  // No output has been written when the exact fallback takes over.
  if (s.vars[1] > kMaxCandidates) return false;

  const int threshold = s.vars[0];
  const int accepted = s.vars[2];
  const uint32_t lower_lanes = (1u << lane) - 1u;
  auto ballot = [&](bool predicate) {
    uint32_t mask;
    sycl::ext::oneapi::group_ballot(sg, predicate).extract_bits(mask);
    return mask;
  };
  for (int c = num_chunks - 1; c >= 0; --c) {
    float values[kItems];
    int offsets[kItems];
    load_chunk(chunk_at(c), values, offsets);
    uint32_t accepted_mask[kItems];
    uint32_t candidate_mask[kItems];
    int sg_accepted = 0;
    int sg_candidates = 0;
#pragma unroll
    for (int i = 0; i < kItems; ++i) {
      const int bin = offsets[i] < length ? bin_of(values[i]) : kBins;
      accepted_mask[i] = ballot(bin < threshold);
      candidate_mask[i] = ballot(bin == threshold);
      sg_accepted += sycl::popcount(accepted_mask[i]);
      sg_candidates += sycl::popcount(candidate_mask[i]);
    }
    int base = 0;
    if (lane == 0 && sg_accepted > 0) base = atomic_add(accept_cursor, sg_accepted);
    base = sycl::group_broadcast(sg, base, 0);
#pragma unroll
    for (int i = 0; i < kItems; ++i) {
      if ((accepted_mask[i] >> lane) & 1u) output[base + sycl::popcount(accepted_mask[i] & lower_lanes)] = offsets[i];
      base += sycl::popcount(accepted_mask[i]);
    }
    if (sg_candidates > 0) {
      int candidate = 0;
      if (lane == 0) candidate = atomic_add(candidate_cursor, sg_candidates);
      candidate = sycl::group_broadcast(sg, candidate, 0) - accepted;
#pragma unroll
      for (int i = 0; i < kItems; ++i) {
        if ((candidate_mask[i] >> lane) & 1u) {
          const int slot = candidate + sycl::popcount(candidate_mask[i] & lower_lanes);
          s.keys[slot] = float_key(values[i]);
          s.indices[slot] = offsets[i];
        }
        candidate += sycl::popcount(candidate_mask[i]);
      }
    }
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

template <int Threads, int Bins, int DigitBits>
inline void tie_select(
    const sycl::nd_item<3>& item,
    const float* input,
    int length,
    int fast_bin,
    int accepted,
    int* output,
    int* hist,
    int* tmp,
    int* vars,
    int topk) {
  constexpr int kWalkItems = 4;
  constexpr int kBuckets = 1 << DigitBits;
  constexpr int kBucketsPerThread = kBuckets / Threads;
  static_assert(kBuckets % Threads == 0);
  static_assert(Bins == kBins);
  const auto group = item.get_group();
  const auto sg = item.get_sub_group();
  const int tid = item.get_local_id(0);
  const int lane = tid % kSubgroupSize;
  const int sg_id = tid / kSubgroupSize;
  int* in_cursor = vars + 3;
  int* tie_cursor = vars + 4;
  if (tid == 0) {
    *in_cursor = 0;
    *tie_cursor = 0;
  }
  sycl::group_barrier(group);

  const int iterations = (length + Threads * kWalkItems - 1) / (Threads * kWalkItems);
  auto walk = [&](auto&& fn) {
    for (int it = 0; it < iterations; ++it) {
      const int base = it * Threads * kWalkItems + tid;
      float values[kWalkItems];
#pragma unroll
      for (int k = 0; k < kWalkItems; ++k) {
        const int j = base + k * Threads;
        values[k] = j < length ? input[j] : 0.f;
      }
#pragma unroll
      for (int k = 0; k < kWalkItems; ++k) {
        const int j = base + k * Threads;
        fn(values[k], j, j < length);
      }
    }
  };
  auto aggregated_inc = [&](bool active, int bucket) {
    if (!sycl::any_of_group(sg, active)) return;
    const int leader = sycl::reduce_over_group(sg, active ? lane : kSubgroupSize, sycl::minimum<int>());
    const int leader_bucket = sycl::select_from_group(sg, bucket, leader);
    const bool match = active && bucket == leader_bucket;
    const int n = sycl::reduce_over_group(sg, match ? 1 : 0, sycl::plus<int>());
    if (lane == leader)
      atomic_add(hist + leader_bucket, n);
    else if (active && !match)
      atomic_add(hist + bucket, 1);
  };
  auto claim = [&](int* cursor, bool active, int& end) {
    const int flag = active ? 1 : 0;
    const int n = sycl::reduce_over_group(sg, flag, sycl::plus<int>());
    if (n == 0) return 0;
    const int rank = sycl::exclusive_scan_over_group(sg, flag, sycl::plus<int>());
    int first = 0;
    if (lane == 0) first = atomic_add(cursor, n);
    first = sycl::group_broadcast(sg, first, 0);
    end = first + n;
    return first + rank;
  };

  const int need0 = topk - accepted;
  uint32_t lo = 0xffffffffu;
  uint32_t hi = 0;
  bool ties_open = true;
  walk([&](float x, int j, bool valid) {
    bool in = false;
    bool tie = false;
    if (valid) {
      const int bin = bin_of(x);
      in = bin < fast_bin;
      if (bin == fast_bin) {
        const uint32_t key = float_key(x);
        lo = sycl::min(lo, key);
        hi = sycl::max(hi, key);
        tie = ties_open;
      }
    }
    int end = 0;
    const int slot = claim(in_cursor, in, end);
    if (in) output[slot] = j;
    end = 0;
    const int tie_slot = claim(tie_cursor, tie, end);
    if (tie && tie_slot < need0) output[accepted + tie_slot] = j;
    ties_open = ties_open && end < need0;
  });
  const uint32_t min_key = sycl::reduce_over_group(group, lo, sycl::minimum<uint32_t>());
  const uint32_t range = sycl::reduce_over_group(group, hi, sycl::maximum<uint32_t>()) - min_key;
  if (range == 0) {
    sycl::group_barrier(group);
    return;
  }
  sycl::group_barrier(group);
  if (tid == 0) {
    *in_cursor = accepted;
    *tie_cursor = 0;
  }

  int need = need0;
  int shift = sycl::max(0, 32 - static_cast<int>(sycl::clz(range)) - DigitBits);
  uint32_t prefix = 0;
  int hi_shift = 32;
  while (true) {
    int* my_buckets = hist + tid * kBucketsPerThread;
    int counts[kBucketsPerThread] = {};
    bins_store(my_buckets, counts);
    sycl::group_barrier(group);
    const uint32_t digit_mask = hi_shift == 32 ? kBuckets - 1u : (uint64_t{1} << (hi_shift - shift)) - 1u;
    walk([&](float x, int, bool valid) {
      bool active = valid && bin_of(x) == fast_bin;
      uint32_t offset = 0;
      if (active) {
        offset = float_key(x) - min_key;
        active = hi_shift == 32 || (uint64_t{offset} >> hi_shift) == prefix;
      }
      aggregated_inc(active, (offset >> shift) & digit_mask);
    });
    sycl::group_barrier(group);
    bins_load(my_buckets, counts);
    int sum = 0;
#pragma unroll
    for (int i = 0; i < kBucketsPerThread; ++i)
      sum += counts[i];
    int total = 0;
    int before = block_exclusive_scan<Threads>(group, sg, lane, sg_id, tmp, sum, total);
    if (before < need && before + sum >= need) {
#pragma unroll
      for (int i = 0; i < kBucketsPerThread; ++i) {
        if (before < need && before + counts[i] >= need) {
          vars[0] = tid * kBucketsPerThread + i;
          vars[1] = before;
          vars[2] = counts[i];
        }
        before += counts[i];
      }
    }
    sycl::group_barrier(group);
    const uint32_t bucket = vars[0];
    prefix = hi_shift == 32 ? bucket : (uint64_t{prefix} << (hi_shift - shift)) | bucket;
    need -= vars[1];
    const int count = vars[2];
    hi_shift = shift;
    if (count == need || shift == 0) break;
    shift = sycl::max(0, shift - DigitBits);
  }

  const int tie_base = topk - need;
  ties_open = true;
  walk([&](float x, int j, bool valid) {
    bool in = false;
    bool tie = false;
    if (valid && bin_of(x) == fast_bin) {
      const uint32_t top = (float_key(x) - min_key) >> shift;
      in = top < prefix;
      tie = ties_open && top == prefix;
    }
    int end = 0;
    const int slot = claim(in_cursor, in, end);
    if (in) output[slot] = j;
    end = 0;
    const int tie_slot = claim(tie_cursor, tie, end);
    if (tie && tie_slot < need) output[tie_base + tie_slot] = j;
    ties_open = ties_open && end < need;
  });
  sycl::group_barrier(group);
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
  tie_select<kThreads, kBins, 11>(item, input, length, threshold, accepted, output, s.bins, s.tmp, s.vars, topk);
}

}  // namespace sgl_kernel::prefill_topk
