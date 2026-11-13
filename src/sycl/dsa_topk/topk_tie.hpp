// Adapted from vllm-xpu-kernels csrc/dsa_topk (5bc6a53a).
#pragma once

// handle_tie family: exact radix select over the threshold bin's ties.
// Direct port of topk_impl.cuh (handle_tie, radix_tie_select,
// refine_find_threshold, select_smallest_indices). The deterministic-mode
// branches compile out under DSA_TOPK_MODE=0/1.

#include <cstdint>
#include <sycl/sycl.hpp>

#include "topk_config.hpp"
#include "topk_warp.hpp"

namespace vllm {
namespace dsa_topk {

struct TieOps : TopKConfig {
  using TieHandleSmem = TopKConfig::TieHandleSmem;

  // is_greater over TieValue: value desc, index asc on equal values.
  static inline bool tie_greater(const TieValue& a, const TieValue& b) {
    return (a.value > b.value) || (a.value == b.value && a.idx < b.idx);
  }

  // refine_find_threshold: threshold-digit search over a kRefineSize-bin
  // histogram. One thread owns kRefineItems contiguous bins. Block-wide.
  static void refine_find_threshold(
      const sycl::nd_item<1>& item,
      const uint32_t* hist,
      const uint32_t total_active,
      const uint32_t topk_remain,
      TieHandleSmem* smem) {
    auto group = item.get_group();
    auto sg = item.get_sub_group();
    const uint32_t tx = item.get_local_linear_id();
    const uint32_t lane_id = tx % kWarpSize;
    const uint32_t warp_id = tx / kWarpSize;

    uint32_t counts[kRefineItems];
    uint32_t local_sum = 0;
#pragma unroll
    for (uint32_t i = 0; i < kRefineItems; ++i) {
      counts[i] = hist[tx * kRefineItems + i];
      local_sum += counts[i];
    }
    const uint32_t warp_inc = warp_inclusive_sum(sg, local_sum);
    if (lane_id == kWarpSize - 1) smem->warp_sum[warp_id] = warp_inc;
    sycl::group_barrier(group);

    uint32_t prefix = warp_reduce_sum(sg, lane_id < warp_id ? smem->warp_sum[lane_id] : 0u);
    prefix += warp_inc - local_sum;
#pragma unroll
    for (uint32_t i = 0; i < kRefineItems; ++i) {
      prefix += counts[i];
      const uint32_t above = total_active - prefix;
      if (above < topk_remain && above + counts[i] >= topk_remain) {
        smem->match = {tx * kRefineItems + i, above, counts[i]};
      }
    }
    sycl::group_barrier(group);
  }

  // select_smallest_indices: emit the `need` smallest of `count` distinct
  // indices in `list` at [pos, pos+need). Deterministic-mode only.
  static void select_smallest_indices(
      const sycl::nd_item<1>& item,
      const uint32_t* list,
      const uint32_t count,
      uint32_t need,
      const TopKProblem problem,
      const uint32_t pos,
      TieHandleSmem* smem) {
    auto group = item.get_group();
    auto sg = item.get_sub_group();
    const uint32_t tx = item.get_local_linear_id();
    const uint32_t lane_id = tx % kWarpSize;
    const uint32_t warp_id = tx / kWarpSize;
    const uint32_t top = problem.seq_len - 1;
    uint32_t ikey[kTieItems];
    bool active[kTieItems];
#pragma unroll
    for (uint32_t i = 0; i < kTieItems; ++i) {
      const uint32_t t = tx + i * kBlockSize;
      active[i] = t < count;
      ikey[i] = active[i] ? top - list[t] : 0u;
    }
    if (tx < kRadixSize) smem->histogram[0][tx] = 0;
    if (tx == kRadixSize) smem->counter = 0;
    sycl::group_barrier(group);
    uint32_t total = count;
    uint32_t parity = 0;
    const uint32_t bits = top == 0 ? 1u : 32u - sycl::clz(top);
    for (int shift = (int)((bits - 1) / 8 * 8); shift >= 0; shift -= 8) {
      uint32_t* histogram = smem->histogram[parity];
#pragma unroll
      for (uint32_t i = 0; i < kTieItems; ++i) {
        if (active[i]) slm_atomic_add(histogram[(ikey[i] >> shift) & 0xFFu], 1);
      }
      if (tx < kRadixSize) smem->histogram[parity ^ 1][tx] = 0;
      parity ^= 1;
      sycl::group_barrier(group);
      uint32_t hist_val = 0, warp_inc = 0;
      if (tx < kRadixSize) {
        hist_val = histogram[tx];
        warp_inc = warp_inclusive_sum(sg, hist_val);
        if (lane_id == kWarpSize - 1) smem->warp_sum[warp_id] = warp_inc;
      }
      sycl::group_barrier(group);
      if (tx < kRadixSize) {
        const uint32_t inter = warp_reduce_sum(sg, lane_id < warp_id ? smem->warp_sum[lane_id] : 0u);
        const uint32_t above = total - (inter + warp_inc);
        if (above < need && above + hist_val >= need) smem->match = {tx, above, hist_val};
      }
      sycl::group_barrier(group);
      const auto m = smem->match;
      const uint32_t threshold_bin = m.bin, above_count = m.above_count, equal_count = m.equal_count;
      total = equal_count;
      need -= above_count;
#pragma unroll
      for (uint32_t i = 0; i < kTieItems; ++i) {
        if (!active[i]) continue;
        const uint32_t bin = (ikey[i] >> shift) & 0xFFu;
        if (bin > threshold_bin || (bin == threshold_bin && shift == 0)) {
          problem.emit(pos + slm_atomic_add(smem->counter, 1), top - ikey[i]);
          active[i] = false;
        } else if (bin < threshold_bin) {
          active[i] = false;
        }
      }
      if (need == 0 || (equal_count == need)) {
#pragma unroll
        for (uint32_t i = 0; i < kTieItems; ++i) {
          if (active[i]) problem.emit(pos + slm_atomic_add(smem->counter, 1), top - ikey[i]);
        }
        return;
      }
    }
  }

  // radix_tie_select: exact radix select over tie candidates. Each thread owns
  // kItems strided elements. Requires num_ties <= kItems * kBlockSize.
  template <uint32_t kItems>
  static void radix_tie_select(
      const sycl::nd_item<1>& item,
      const TieValue* tie_buffer,
      const TopKProblem& problem,
      const uint32_t base,
      const uint32_t num_ties,
      const uint32_t topk,
      TieHandleSmem* smem) {
    auto group = item.get_group();
    auto sg = item.get_sub_group();
    const uint32_t tx = item.get_local_linear_id();
    const uint32_t lane_id = tx % kWarpSize;
    const uint32_t warp_id = tx / kWarpSize;

    bool active[kItems];
    uint32_t key[kItems], idx[kItems], write_pos[kItems];
#pragma unroll
    for (uint32_t i = 0; i < kItems; ++i) {
      const uint32_t t = tx + i * kBlockSize;
      active[i] = t < num_ties;
      const TieValue tie = active[i] ? tie_buffer[t] : TieValue::invalid();
      key[i] = extract_tie_key(tie.value);
      idx[i] = tie.idx;
      write_pos[i] = topk;
    }
    uint32_t topk_remain = topk;
    if (tx < kRadixSize) smem->histogram[0][tx] = 0;
    if (tx == kRadixSize) smem->counter = smem->counter_final = 0;
    sycl::group_barrier(group);
    uint32_t total_active = num_ties;
    uint32_t* list = reinterpret_cast<uint32_t*>(const_cast<TieValue*>(tie_buffer));
    bool straddle = false;

#pragma unroll
    for (int round = 0; round < 4; round++) {
      const uint32_t shift = 24 - round * 8;
      const uint32_t hist_idx = round % 2;
      uint32_t* histogram = smem->histogram[hist_idx];
#pragma unroll
      for (uint32_t i = 0; i < kItems; ++i) {
        if (active[i]) slm_atomic_add(histogram[(key[i] >> shift) & 0xFFu], 1);
      }
      if (round < 3 && tx < kRadixSize) smem->histogram[hist_idx ^ 1][tx] = 0;
      sycl::group_barrier(group);

      uint32_t hist_val = 0, warp_inc = 0;
      if (tx < kRadixSize) {
        hist_val = histogram[tx];
        warp_inc = warp_inclusive_sum(sg, hist_val);
        if (lane_id == kWarpSize - 1) smem->warp_sum[warp_id] = warp_inc;
      }
      sycl::group_barrier(group);
      if (tx < kRadixSize) {
        const uint32_t inter = warp_reduce_sum(sg, lane_id < warp_id ? smem->warp_sum[lane_id] : 0u);
        const uint32_t prefix = inter + warp_inc;
        const uint32_t above = total_active - prefix;
        if (above < topk_remain && above + hist_val >= topk_remain) smem->match = {tx, above, hist_val};
      }
      sycl::group_barrier(group);

      const auto m = smem->match;
      const uint32_t threshold_bin = m.bin, above_count = m.above_count, equal_count = m.equal_count;
      total_active = equal_count;
      topk_remain -= above_count;
      if (round == 3 && kDeterministic) straddle = equal_count > topk_remain;

#pragma unroll
      for (uint32_t i = 0; i < kItems; ++i) {
        if (!active[i]) continue;
        const uint32_t bin = (key[i] >> shift) & 0xFFu;
        if (bin > threshold_bin) {
          write_pos[i] = slm_atomic_add(smem->counter, 1);
          active[i] = false;
        } else if (bin < threshold_bin) {
          active[i] = false;
        } else if (round == 3) {
          const uint32_t slot = slm_atomic_add(smem->counter_final, 1);
          if constexpr (kDeterministic) {
            if (straddle) {
              list[slot] = idx[i];
              continue;
            }
          }
          write_pos[i] = topk - topk_remain + slot;
        }
      }
      if (round == 3 || topk_remain == 0) break;
    }

#pragma unroll
    for (uint32_t i = 0; i < kItems; ++i) {
      if (write_pos[i] < topk) problem.emit(base + write_pos[i], idx[i]);
    }
    if constexpr (kDeterministic) {
      if (straddle) {
        sycl::group_barrier(group);
        const uint32_t pos = base + topk - topk_remain;
        if (total_active <= kWarpSize) {
          if (warp_id == 0 && lane_id < total_active) {
            const uint32_t mine = list[lane_id];
            uint32_t rank = 0;
            for (uint32_t j = 0; j < total_active; ++j)
              rank += list[j] < mine ? 1u : 0u;
            if (rank < topk_remain) problem.emit(pos + rank, mine);
          }
        } else {
          select_smallest_indices(item, list, total_active, topk_remain, problem, pos, smem);
        }
      }
    }
  }

  // handle_tie: resolve the threshold bin's ties exactly. `base` strictly-above
  // elements already emitted; `topk` here is remaining slots to fill.
  static void handle_tie(
      const sycl::nd_item<1>& item,
      const TieValue* tie_buffer,
      const TopKProblem& problem,
      const uint32_t base,
      const uint32_t num_ties,
      const uint32_t topk,
      TieHandleSmem* smem) {
    auto sg = item.get_sub_group();
    const uint32_t tx = item.get_local_linear_id();
    const uint32_t lane_id = tx % kWarpSize;
    const uint32_t warp_id = tx / kWarpSize;

    if (num_ties <= topk) {
      for (uint32_t t = tx; t < num_ties; t += kBlockSize)
        problem.emit(base + t, tie_buffer[t].idx);
      for (uint32_t t = num_ties + tx; t < topk; t += kBlockSize)
        problem.emit(base + t, base + t);
    } else if (num_ties <= kWarpSize) {
      // Every lane of each subgroup must reach warp_sum_bool_masked (it calls
      // reduce_over_group, a subgroup collective). Do not early-return lanes;
      // invalid lanes feed an invalid tie that never counts, and only valid
      // (lane 0 of a warp with warp_id < num_ties) emits.
      const TieValue invalid = TieValue::invalid();
      const TieValue tie = lane_id < num_ties ? tie_buffer[lane_id] : invalid;
      const TieValue target = warp_id < num_ties ? tie_buffer[warp_id] : invalid;
      const uint32_t rank = warp_sum_bool_masked(sg, tie_greater(tie, target), num_ties);
      if (warp_id < num_ties && lane_id == 0 && rank < topk) problem.emit(base + rank, target.idx);
    } else if (num_ties <= kWarpSize * 2) {
      const uint32_t warp_id_1 = warp_id + kWarpSize;
      const uint32_t lane_id_1 = lane_id + kWarpSize;
      const TieValue invalid = TieValue::invalid();
      const TieValue tie_0 = tie_buffer[lane_id];
      const TieValue tie_1 = lane_id_1 < num_ties ? tie_buffer[lane_id_1] : invalid;
      const TieValue target_0 = tie_buffer[warp_id];
      const TieValue target_1 = tie_buffer[warp_id_1];
      {
        const uint32_t rank =
            warp_sum_bool(sg, tie_greater(tie_0, target_0)) + warp_sum_bool(sg, tie_greater(tie_1, target_0));
        if (lane_id == 0 && rank < topk) problem.emit(base + rank, target_0.idx);
      }
      if (warp_id_1 < num_ties) {
        const uint32_t rank =
            warp_sum_bool(sg, tie_greater(tie_0, target_1)) + warp_sum_bool(sg, tie_greater(tie_1, target_1));
        if (lane_id == 0 && rank < topk) problem.emit(base + rank, target_1.idx);
      }
    } else if (num_ties <= kWarpSize * 4) {
      const TieValue invalid = TieValue::invalid();
      const TieValue tie[] = {
          tie_buffer[lane_id + 0 * kWarpSize],
          tie_buffer[lane_id + 1 * kWarpSize],
          lane_id + 2 * kWarpSize < num_ties ? tie_buffer[lane_id + 2 * kWarpSize] : invalid,
          lane_id + 3 * kWarpSize < num_ties ? tie_buffer[lane_id + 3 * kWarpSize] : invalid,
      };
      const TieValue target[] = {
          tie_buffer[warp_id + 0 * kWarpSize],
          tie_buffer[warp_id + 1 * kWarpSize],
          tie_buffer[warp_id + 2 * kWarpSize],
          tie_buffer[warp_id + 3 * kWarpSize],
      };
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if (i >= 2 && warp_id + i * kWarpSize >= num_ties) break;
        uint32_t rank = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j)
          rank += warp_sum_bool(sg, tie_greater(tie[j], target[i]));
        if (lane_id == 0 && rank < topk) problem.emit(base + rank, target[i].idx);
      }
    } else if (num_ties <= kBlockSize) {
      radix_tie_select<1>(item, tie_buffer, problem, base, num_ties, topk, smem);
    } else {
      static_assert(kTieItems == 2);
      radix_tie_select<2>(item, tie_buffer, problem, base, num_ties, topk, smem);
    }
  }
};

}  // namespace dsa_topk
}  // namespace vllm
