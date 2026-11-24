// Adapted from vllm-xpu-kernels csrc/dsa_topk (5bc6a53a).
#pragma once

// Exact top-k of one row per 1024-thread work-group, for long rows and small
// batches. A 12-bit histogram of the fp16-rounded scores locates the bin that
// holds the k-th largest score; scores above it are emitted directly, and the
// bin's ties are staged in SLM and ranked exactly. If the bin holds more than
// kMaxNumTie scores (or the row has a NaN) the row is selected by a slower
// exact radix select over the whole row.

#include <cstdint>
#include <limits>
#include <sycl/sycl.hpp>

namespace sgl_kernel::dsa_topk {

constexpr uint32_t kBlockSize = 1024;
constexpr uint32_t kWarpSize = 32;
constexpr uint32_t kNumWarps = kBlockSize / kWarpSize;
static_assert(kNumWarps == kWarpSize, "tie ranking assumes as many subgroups as lanes");
constexpr uint32_t kVecSize = 4;
constexpr uint32_t kHistBits = 12;
constexpr uint32_t kHistSize = 1 << kHistBits;
constexpr uint32_t kHistItems = kHistSize / kBlockSize;
constexpr uint32_t kRadixSize = 256;
constexpr uint32_t kMaxNumTie = 2048;
constexpr uint32_t kTieItems = kMaxNumTie / kBlockSize;

using SlmAtomic = sycl::atomic_ref<
    uint32_t,
    sycl::memory_order::relaxed,
    sycl::memory_scope::work_group,
    sycl::access::address_space::local_space>;

// Order-preserving fp32 key (larger float, larger key); every NaN is the largest.
inline uint32_t extract_exact_bin(float x) {
  const uint32_t bits = sycl::bit_cast<uint32_t>(x);
  if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0) return 0xFFFFFFFFu;
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// fp32 -> coarse bin via the ordered fp16 key.
inline uint32_t extract_coarse_bin(float x) {
  const uint32_t bits = static_cast<uint32_t>(sycl::bit_cast<uint16_t>(static_cast<sycl::half>(x))) << 16;
  const uint32_t sign = static_cast<uint32_t>(static_cast<int32_t>(bits) >> 31);
  return (bits ^ (sign | 0x80000000u)) >> (32 - kHistBits);
}

// Ordered 16-bit half-precision key (low 16 bits) -> fp32.
inline float ordered16_to_fp32(uint32_t k16) {
  const uint16_t bits = static_cast<uint16_t>(k16 ^ 0x8000u ^ (((~k16 >> 15) & 0x1u) * 0x7FFFu));
  return static_cast<float>(sycl::bit_cast<sycl::half>(bits));
}

// fp32 bounds {lo, hi} of a coarse bin: "above" iff v >= hi, "tie" iff v >= lo
// and not above.
inline sycl::float2 coarse_bin_bounds(uint32_t bin) {
  constexpr uint32_t kShift = 16 - kHistBits;
  static_assert(kShift >= 1, "edge keys must be even for the rounding rule below");
  constexpr uint32_t kNegInfBin = 0x03FFu >> kShift;  // bin holding the -inf key
  constexpr uint32_t kInfBin = 0xFC00u >> kShift;     // bin holding the +inf key

  const uint32_t key_lo16 = bin << kShift;
  const uint32_t key_hi16 = (bin + 1) << kShift;
  const float v_lo = ordered16_to_fp32(key_lo16);
  const float v_hi = ordered16_to_fp32(key_hi16);
  const float u_lo = ordered16_to_fp32(key_lo16 - 1u);
  const float u_hi = ordered16_to_fp32(key_hi16 - 1u);

  const auto midpoint = [](float a, float b) { return sycl::fmin(sycl::fmax(0.5f * (a + b), -65520.0f), 65520.0f); };
  const int32_t lo = sycl::bit_cast<int32_t>(midpoint(v_lo, u_lo));
  const int32_t hi = sycl::bit_cast<int32_t>(midpoint(v_hi, u_hi));
  // Ties round to even and edge keys are even, so negative edges sit one ulp
  // above the midpoint. A zero hi steps past +0.
  const float lo_edge = sycl::bit_cast<float>(lo + (lo >> 31));
  const float hi_edge = sycl::bit_cast<float>(hi + (hi >> 31) + (hi == 0));
  const float nan = sycl::bit_cast<float>(0x7F800001);
  const float lo_final = bin <= kNegInfBin ? -std::numeric_limits<float>::infinity() : (bin > kInfBin ? nan : lo_edge);
  const float hi_final = bin >= kInfBin ? nan : hi_edge;
  return sycl::float2(lo_final, hi_final);
}

struct alignas(8) TieValue {
  float value;
  uint32_t idx;
  static constexpr TieValue invalid() {
    return TieValue{std::numeric_limits<float>::quiet_NaN(), 0xFFFFFFFFu};
  }
};

// value desc, index asc on equal values.
inline bool tie_greater(const TieValue& a, const TieValue& b) {
  return (a.value > b.value) || (a.value == b.value && a.idx < b.idx);
}

struct TopKProblem {
  const float* in;
  int32_t* out;  // selected indices [topk]
  uint32_t topk;
  uint32_t seq_len;
};

struct MatchBin {
  uint32_t bin;
  uint32_t above_count;
  uint32_t equal_count;
};

struct Smem {
  uint32_t count_eq;
  uint32_t count_gt;
  uint32_t has_nan;
  uint32_t overflow_prefix;
  float v_hi;
  float v_lo;
  uint32_t warp_sum[kNumWarps];
  // The coarse histogram and the tie-ranking state are live in disjoint phases.
  union {
    alignas(16) uint32_t histogram[kHistSize];
    struct {
      TieValue tie_values[kMaxNumTie];
      uint32_t counter;
      uint32_t counter_final;
      MatchBin match;
      uint32_t tie_warp_sum[kNumWarps];
      uint32_t tie_histogram[2][kRadixSize];
    };
  };
};

// Four-element input vector. Rows are offset by an arbitrary row_start, so the
// 16-byte load is used only when the address is aligned.
struct Vec4 {
  sycl::vec<float, 4> v;
  void load(const float* ptr, int64_t vi) {
    const float* p = ptr + vi * 4;
    if ((reinterpret_cast<uintptr_t>(p) & 0xF) == 0) {
      v = *reinterpret_cast<const sycl::vec<float, 4>*>(p);
    } else {
      for (int j = 0; j < 4; ++j)
        v[j] = p[j];
    }
  }
  // Final partial vector: read only the n valid elements.
  void load_n(const float* ptr, int64_t vi, uint32_t n) {
    const float* p = ptr + vi * 4;
    for (int j = 0; j < 4; ++j)
      v[j] = static_cast<uint32_t>(j) < n ? p[j] : 0.f;
  }
  float operator[](uint32_t j) const {
    return v[j];
  }
};

// Scan one row with the work-group, invoking fn(value, index).
template <typename F>
inline void for_each_input(const sycl::nd_item<1>& item, const float* in, uint32_t seq_len, F&& fn) {
  const uint32_t num_full = seq_len / kVecSize;
  uint32_t vi = item.get_local_linear_id();
  if (vi < num_full) {
    Vec4 next_vec;
    next_vec.load(in, vi);
    do {
      const Vec4 cur = next_vec;
      vi += kBlockSize;
      if (vi < num_full) next_vec.load(in, vi);
      const uint32_t base = (vi - kBlockSize) * kVecSize;
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j)
        fn(cur[j], base + j);
    } while (vi < num_full);
  }
  if (vi == num_full && num_full * kVecSize != seq_len) {
    const uint32_t base = vi * kVecSize;
    Vec4 cur;
    cur.load_n(in, vi, seq_len - base);
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j)
      if (base + j < seq_len) fn(cur[j], base + j);
  }
}

// Rescan the row exactly: four 8-bit radix passes find the k-th key, then two
// passes emit the larger keys and the first ties.
inline void select_overflow(const sycl::nd_item<1>& item, const TopKProblem& problem, Smem* smem) {
  const uint32_t tx = item.get_local_linear_id();
  auto group = item.get_group();
  uint32_t prefix = 0;
  uint32_t remaining = problem.topk;
  for (int shift = 24; shift >= 0; shift -= 8) {
    if (tx < 256) smem->histogram[tx] = 0;
    sycl::group_barrier(group);
    for (uint32_t i = tx; i < problem.seq_len; i += kBlockSize) {
      const uint32_t key = extract_exact_bin(problem.in[i]);
      if (((key >> shift) >> 8) == prefix) SlmAtomic(smem->histogram[(key >> shift) & 255]).fetch_add(1);
    }
    sycl::group_barrier(group);
    if (tx == 0) {
      uint32_t need = remaining;
      for (int bin = 255; bin >= 0; --bin) {
        const uint32_t count = smem->histogram[bin];
        if (need > count) {
          need -= count;
        } else {
          smem->overflow_prefix = (prefix << 8) | static_cast<uint32_t>(bin);
          smem->count_eq = need;
          break;
        }
      }
    }
    sycl::group_barrier(group);
    prefix = smem->overflow_prefix;
    remaining = smem->count_eq;
    sycl::group_barrier(group);
  }

  if (tx == 0) smem->count_gt = smem->count_eq = 0;
  sycl::group_barrier(group);
  for (uint32_t i = tx; i < problem.seq_len; i += kBlockSize) {
    if (extract_exact_bin(problem.in[i]) > prefix) {
      const uint32_t pos = SlmAtomic(smem->count_gt).fetch_add(1);
      if (pos < problem.topk) problem.out[pos] = i;
    }
  }
  sycl::group_barrier(group);
  const uint32_t above = smem->count_gt;
  for (uint32_t i = tx; i < problem.seq_len; i += kBlockSize) {
    if (extract_exact_bin(problem.in[i]) == prefix) {
      const uint32_t pos = above + SlmAtomic(smem->count_eq).fetch_add(1);
      if (pos < problem.topk) problem.out[pos] = i;
    }
  }
}

// Exact radix select over the staged ties; each thread owns kTieItems strided
// ties. Writes the `topk` largest to out[base + ...].
inline void radix_tie_select(
    const sycl::nd_item<1>& item,
    const TopKProblem& problem,
    const uint32_t base,
    const uint32_t num_ties,
    const uint32_t topk,
    Smem* smem) {
  auto group = item.get_group();
  auto sg = item.get_sub_group();
  const uint32_t tx = item.get_local_linear_id();
  const uint32_t lane_id = tx % kWarpSize;
  const uint32_t warp_id = tx / kWarpSize;

  bool active[kTieItems];
  uint32_t key[kTieItems], idx[kTieItems], write_pos[kTieItems];
#pragma unroll
  for (uint32_t i = 0; i < kTieItems; ++i) {
    const uint32_t t = tx + i * kBlockSize;
    active[i] = t < num_ties;
    const TieValue tie = active[i] ? smem->tie_values[t] : TieValue::invalid();
    key[i] = extract_exact_bin(tie.value);
    idx[i] = tie.idx;
    write_pos[i] = topk;
  }
  uint32_t topk_remain = topk;
  if (tx < kRadixSize) smem->tie_histogram[0][tx] = 0;
  if (tx == kRadixSize) smem->counter = smem->counter_final = 0;
  sycl::group_barrier(group);
  uint32_t total_active = num_ties;

#pragma unroll
  for (int round = 0; round < 4; round++) {
    const uint32_t shift = 24 - round * 8;
    uint32_t* histogram = smem->tie_histogram[round % 2];
#pragma unroll
    for (uint32_t i = 0; i < kTieItems; ++i) {
      if (active[i]) SlmAtomic(histogram[(key[i] >> shift) & 0xFFu]).fetch_add(1);
    }
    if (round < 3 && tx < kRadixSize) smem->tie_histogram[(round % 2) ^ 1][tx] = 0;
    sycl::group_barrier(group);

    uint32_t hist_val = 0, warp_inc = 0;
    if (tx < kRadixSize) {
      hist_val = histogram[tx];
      warp_inc = sycl::inclusive_scan_over_group(sg, hist_val, sycl::plus<uint32_t>());
      if (lane_id == kWarpSize - 1) smem->tie_warp_sum[warp_id] = warp_inc;
    }
    sycl::group_barrier(group);
    if (tx < kRadixSize) {
      const uint32_t inter =
          sycl::reduce_over_group(sg, lane_id < warp_id ? smem->tie_warp_sum[lane_id] : 0u, sycl::plus<uint32_t>());
      const uint32_t above = total_active - (inter + warp_inc);
      if (above < topk_remain && above + hist_val >= topk_remain) smem->match = {tx, above, hist_val};
    }
    sycl::group_barrier(group);

    const MatchBin m = smem->match;
    total_active = m.equal_count;
    topk_remain -= m.above_count;
#pragma unroll
    for (uint32_t i = 0; i < kTieItems; ++i) {
      if (!active[i]) continue;
      const uint32_t bin = (key[i] >> shift) & 0xFFu;
      if (bin > m.bin) {
        write_pos[i] = SlmAtomic(smem->counter).fetch_add(1);
        active[i] = false;
      } else if (bin < m.bin) {
        active[i] = false;
      } else if (round == 3) {
        write_pos[i] = topk - topk_remain + SlmAtomic(smem->counter_final).fetch_add(1);
      }
    }
    if (round == 3 || topk_remain == 0) break;
  }

#pragma unroll
  for (uint32_t i = 0; i < kTieItems; ++i) {
    if (write_pos[i] < topk) problem.out[base + write_pos[i]] = idx[i];
  }
}

// Resolve the threshold bin's staged ties exactly. `base` strictly-above
// scores are already emitted; `topk` is the number of slots left to fill.
inline void handle_tie(
    const sycl::nd_item<1>& item,
    const TopKProblem& problem,
    const uint32_t base,
    const uint32_t num_ties,
    const uint32_t topk,
    Smem* smem) {
  const TieValue* ties = smem->tie_values;
  const uint32_t tx = item.get_local_linear_id();
  if (num_ties <= topk) {
    for (uint32_t t = tx; t < num_ties; t += kBlockSize)
      problem.out[base + t] = ties[t].idx;
    for (uint32_t t = num_ties + tx; t < topk; t += kBlockSize)
      problem.out[base + t] = base + t;
  } else if (num_ties <= 4 * kWarpSize) {
    // Each subgroup ranks its targets against all ties, one lane per tie.
    auto sg = item.get_sub_group();
    for (uint32_t t = tx / kWarpSize; t < num_ties; t += kNumWarps) {
      const TieValue target = ties[t];
      uint32_t rank = 0;
      for (uint32_t j = tx % kWarpSize; j < num_ties; j += kWarpSize)
        rank += tie_greater(ties[j], target);
      rank = sycl::reduce_over_group(sg, rank, sycl::plus<uint32_t>());
      if (tx % kWarpSize == 0 && rank < topk) problem.out[base + rank] = target.idx;
    }
  } else {
    radix_tie_select(item, problem, base, num_ties, topk, smem);
  }
}

// Select the problem.topk largest scores of the row; problem.seq_len > topk.
inline void select(const sycl::nd_item<1>& item, const TopKProblem& problem, Smem* smem) {
  auto group = item.get_group();
  auto sg = item.get_sub_group();
  const uint32_t tx = item.get_local_linear_id();
  const uint32_t topk = problem.topk;

  for (uint32_t i = 0; i < kHistItems; ++i)
    smem->histogram[tx * kHistItems + i] = 0;
  if (tx == 0) smem->count_eq = smem->count_gt = smem->has_nan = 0;
  sycl::group_barrier(group);

  // Pass 1: coarse histogram.
  for_each_input(item, problem.in, problem.seq_len, [&](float val, uint32_t) {
    if (sycl::isnan(val)) SlmAtomic(smem->has_nan).fetch_or(1);
    SlmAtomic(smem->histogram[extract_coarse_bin(val)]).fetch_add(1);
  });
  sycl::group_barrier(group);
  if (smem->has_nan) {
    select_overflow(item, problem, smem);
    return;
  }

  // Threshold bin: the one holding the topk-th largest score. Each thread owns
  // kHistItems contiguous bins.
  uint32_t local_exc_sum[kHistItems + 1];
  local_exc_sum[0] = 0;
  for (uint32_t i = 0; i < kHistItems; ++i)
    local_exc_sum[i + 1] = smem->histogram[tx * kHistItems + i] + local_exc_sum[i];
  const uint32_t local_sum = local_exc_sum[kHistItems];
  const uint32_t lane_id = tx % kWarpSize;
  const uint32_t warp_id = tx / kWarpSize;
  const uint32_t warp_inc_sum = sycl::inclusive_scan_over_group(sg, local_sum, sycl::plus<uint32_t>());
  if (lane_id == kWarpSize - 1) smem->warp_sum[warp_id] = warp_inc_sum;
  sycl::group_barrier(group);
  const uint32_t warp_prefix_sum =
      sycl::reduce_over_group(sg, lane_id < warp_id ? smem->warp_sum[lane_id] : 0u, sycl::plus<uint32_t>());
  const int32_t exc_sum = static_cast<int32_t>(warp_prefix_sum + warp_inc_sum - local_sum);
  const int32_t remained = static_cast<int32_t>(problem.seq_len - topk - exc_sum);
  if (remained >= 0 && remained < static_cast<int32_t>(local_sum)) {
    for (uint32_t i = 0; i < kHistItems; ++i) {
      if (remained >= static_cast<int32_t>(local_exc_sum[i]) && remained < static_cast<int32_t>(local_exc_sum[i + 1])) {
        const auto bounds = coarse_bin_bounds(tx * kHistItems + i);
        smem->v_lo = bounds[0];
        smem->v_hi = bounds[1];
      }
    }
  }
  sycl::group_barrier(group);

  // Pass 2: emit the scores above the bin, stage the bin's.
  const float v_hi = smem->v_hi;
  const float v_lo = smem->v_lo;
  for_each_input(item, problem.in, problem.seq_len, [&](float val, uint32_t idx) {
    if (val >= v_hi) {
      const uint32_t pos = SlmAtomic(smem->count_gt).fetch_add(1);
      if (pos < topk) problem.out[pos] = idx;
    } else if (val >= v_lo) {
      const uint32_t pos = SlmAtomic(smem->count_eq).fetch_add(1);
      if (pos < kMaxNumTie) smem->tie_values[pos] = {val, idx};
    }
  });
  sycl::group_barrier(group);

  if (smem->count_eq > kMaxNumTie) {
    select_overflow(item, problem, smem);
    return;
  }
  const uint32_t count_gt = smem->count_gt;
  handle_tie(item, problem, count_gt, smem->count_eq, count_gt < topk ? topk - count_gt : 0, smem);
}

}  // namespace sgl_kernel::dsa_topk
