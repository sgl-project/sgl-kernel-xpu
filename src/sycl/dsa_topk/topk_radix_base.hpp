// Adapted from vllm-xpu-kernels csrc/dsa_topk (5bc6a53a).
#pragma once

// TopKRadixBase: histogram storage + vectorized input iteration + threshold-bin
// search. Includes an exact fallback for threshold-bin overflow.

#include <cstdint>
#include <sycl/sycl.hpp>

#include "topk_config.hpp"
#include "topk_tie.hpp"
#include "topk_warp.hpp"

namespace sgl_kernel {
namespace dsa_topk {

// Four-element input vector supporting aligned and scalar loads.
struct Vec4 {
  sycl::vec<float, 4> v;
  inline void load(const float* ptr, int64_t vi) {
    const float* p = ptr + vi * 4;
    // The vectorized (16 B) load needs a 16-byte-aligned address, but ragged
    // callers offset the row pointer by an arbitrary row_start, so p may be
    // unaligned. Use the single-message load only when it is safe; otherwise
    // fall back to scalar element loads (no alignment requirement).
    if ((reinterpret_cast<uintptr_t>(p) & 0xF) == 0) {
      v = *reinterpret_cast<const sycl::vec<float, 4>*>(p);
    } else {
#pragma unroll
      for (int j = 0; j < 4; ++j)
        v[j] = p[j];
    }
  }
  // Partial load of the final vector: reads only the `n` (1..4) valid elements,
  // the rest are filled with `pad`. Prevents the 4-float read from running past
  // the row when seq_len is not a multiple of 4 (e.g. [1, 513]).
  inline void load_n(const float* ptr, int64_t vi, uint32_t n, float pad) {
    const float* p = ptr + vi * 4;
#pragma unroll
    for (int j = 0; j < 4; ++j)
      v[j] = static_cast<uint32_t>(j) < n ? p[j] : pad;
  }
  // Read four bf16 values and widen them to fp32.
  inline void load(const sycl::ext::oneapi::bfloat16* ptr, int64_t vi) {
    const uint16_t* p = reinterpret_cast<const uint16_t*>(ptr) + vi * 4;
#pragma unroll
    for (int j = 0; j < 4; ++j)
      v[j] = sycl::bit_cast<float>(static_cast<uint32_t>(p[j]) << 16);
  }
  // Partial widening bf16 load (final vector), rest filled with `pad`.
  inline void load_n(const sycl::ext::oneapi::bfloat16* ptr, int64_t vi, uint32_t n, float pad) {
    const uint16_t* p = reinterpret_cast<const uint16_t*>(ptr) + vi * 4;
#pragma unroll
    for (int j = 0; j < 4; ++j)
      v[j] = static_cast<uint32_t>(j) < n ? sycl::bit_cast<float>(static_cast<uint32_t>(p[j]) << 16) : pad;
  }
  inline float operator[](uint32_t j) const {
    return v[j];
  }
  inline float& operator[](uint32_t j) {
    return reinterpret_cast<float*>(&v)[j];
  }
};

template <uint32_t kHistBits_>
struct TopKRadixBase : TieOps {
  static constexpr uint32_t kVecSize = 4;
  static constexpr uint32_t kHistBits = kHistBits_;
  static constexpr uint32_t kHistSize = 1 << kHistBits;
  using vec_t = Vec4;

  struct Smem {
    uint32_t count_eq;
    uint32_t count_gt;
    float v_hi;
    float v_lo;
    uint32_t thr_count;
    union {
      uint32_t warp_sum[kNumWarps];
      uint32_t tie_nonzero;
    };
    // Coarse histogram overlays the tie machinery (live in disjoint phases).
    union {
      alignas(16) uint32_t histogram[kHistSize];
      struct {
        union {
          TieValue tie_values[kMaxNumTie];
          uint32_t refine_hist[kRefineSize];
        };
        TieHandleSmem tie_handle;
      };
    };
  };

  // Scan one row with a work-group, invoking fn(value, index).
  template <typename F>
  static void for_each_input(const sycl::nd_item<1>& item, const float* in, uint32_t seq_len, F&& fn) {
    const uint32_t tx = item.get_local_linear_id();
    const uint32_t num_full = seq_len / kVecSize;
    constexpr uint32_t kStride = kBlockSize;
    uint32_t vi = tx;
    if (vi < num_full) {
      vec_t next_vec;
      next_vec.load(in, vi);
      do {
        const vec_t cur = next_vec;
        vi += kStride;
        if (vi < num_full) next_vec.load(in, vi);
        const uint32_t base = (vi - kStride) * kVecSize;
#pragma unroll
        for (uint32_t j = 0; j < kVecSize; ++j)
          fn(cur[j], base + j);
      } while (vi < num_full);
    }
    if (vi == num_full) {
      const uint32_t base = vi * kVecSize;
      if (base == seq_len) return;
      // Final partial vector: load only the valid elements so the 16 B read
      // never runs past the row (guarding the callback alone does not).
      vec_t cur;
      cur.load_n(in, vi, seq_len - base, padding_value());
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j)
        if (base + j < seq_len) fn(cur[j], base + j);
    }
  }

  static void init_histogram(const sycl::nd_item<1>& item, uint32_t (&histogram)[kHistSize]) {
    const uint32_t tx = item.get_local_linear_id();
    constexpr uint32_t kItems = kHistSize / kBlockSize;
#pragma unroll
    for (uint32_t i = 0; i < kItems; ++i)
      histogram[tx * kItems + i] = 0;
  }

  // Rescan the row when the bounded tie buffer overflows.
  static void select_overflow(const sycl::nd_item<1>& item, const TopKProblem& problem, Smem* smem) {
    const uint32_t tx = item.get_local_linear_id();
    auto group = item.get_group();
    uint32_t prefix = 0;
    uint32_t remaining = problem.topk;
    for (int shift = 24; shift >= 0; shift -= 8) {
      if (tx < 256) smem->histogram[tx] = 0;
      sycl::group_barrier(group);
      for (uint32_t i = tx; i < problem.seq_len; i += kBlockSize) {
        const uint32_t key = extract_exact_bin(problem.in[i]);
        if (((key >> shift) >> 8) == prefix) slm_atomic_add(smem->histogram[(key >> shift) & 255], 1);
      }
      sycl::group_barrier(group);
      if (tx == 0) {
        uint32_t need = remaining;
        for (int bin = 255; bin >= 0; --bin) {
          const uint32_t count = smem->histogram[bin];
          if (need > count) {
            need -= count;
          } else {
            smem->thr_count = (prefix << 8) | static_cast<uint32_t>(bin);
            smem->count_eq = need;
            break;
          }
        }
      }
      sycl::group_barrier(group);
      prefix = smem->thr_count;
      remaining = smem->count_eq;
      sycl::group_barrier(group);
    }

    if (tx == 0) smem->count_gt = smem->count_eq = 0;
    sycl::group_barrier(group);
    for (uint32_t i = tx; i < problem.seq_len; i += kBlockSize) {
      if (extract_exact_bin(problem.in[i]) > prefix) {
        const uint32_t pos = slm_atomic_add(smem->count_gt, 1);
        if (pos < problem.topk) problem.emit(pos, i);
      }
    }
    sycl::group_barrier(group);
    const uint32_t above = smem->count_gt;
    for (uint32_t i = tx; i < problem.seq_len; i += kBlockSize) {
      if (extract_exact_bin(problem.in[i]) == prefix) {
        const uint32_t pos = above + slm_atomic_add(smem->count_eq, 1);
        if (pos < problem.topk) problem.emit(pos, i);
      }
    }
  }

  // Fills the slots after the count_gt "above" ones from the staged ties.
  static void handle_staged_ties(const sycl::nd_item<1>& item, const TopKProblem& problem, Smem* smem) {
    const bool overflow = smem->count_eq > kMaxNumTie;
    sycl::group_barrier(item.get_group());
    if (overflow) {
      select_overflow(item, problem, smem);
      return;
    }
    const uint32_t topk = problem.topk;
    const uint32_t count_gt = smem->count_gt;
    const uint32_t remain_topk = count_gt < topk ? topk - count_gt : 0;
    const uint32_t tie_count = sycl::min(smem->count_eq, kMaxNumTie);
    handle_tie(item, smem->tie_values, problem, count_gt, tie_count, remain_topk, &smem->tie_handle);
  }

  // find_threshold: the coarse bin holding the topk-th largest score; calls
  // fn(bin) on the one thread that owns it. Block-wide.
  template <typename Fn>
  static void
  find_threshold(const sycl::nd_item<1>& item, const uint32_t topk, const uint32_t seq_len, Smem* smem, Fn fn) {
    auto group = item.get_group();
    auto sg = item.get_sub_group();
    const uint32_t tx = item.get_local_linear_id();
    constexpr uint32_t kItems = kHistSize / kBlockSize;
    uint32_t local_exc_sum[kItems + 1];
    local_exc_sum[0] = 0;
#pragma unroll
    for (uint32_t i = 0; i < kItems; ++i)
      local_exc_sum[i + 1] = smem->histogram[tx * kItems + i] + local_exc_sum[i];

    const uint32_t local_sum = local_exc_sum[kItems];
    const uint32_t lane_id = tx % kWarpSize;
    const uint32_t warp_id = tx / kWarpSize;
    const uint32_t warp_inc_sum = warp_inclusive_sum(sg, local_sum);
    const uint32_t warp_exc_sum = warp_inc_sum - local_sum;
    if (lane_id == kWarpSize - 1) smem->warp_sum[warp_id] = warp_inc_sum;

    sycl::group_barrier(group);

    const uint32_t lane_warp_sum = smem->warp_sum[lane_id];
    const uint32_t warp_prefix_sum = warp_reduce_sum(sg, lane_id < warp_id ? lane_warp_sum : 0u);
    const int32_t exc_sum = static_cast<int32_t>(warp_prefix_sum + warp_exc_sum);
    const int32_t remained = static_cast<int32_t>(seq_len - topk - exc_sum);
    if (remained >= 0 && remained < static_cast<int32_t>(local_sum)) {
      uint32_t target = 0;
#pragma unroll
      for (uint32_t i = 0; i < kItems; ++i) {
        const int32_t prev = static_cast<int32_t>(local_exc_sum[i + 0]);
        const int32_t next = static_cast<int32_t>(local_exc_sum[i + 1]);
        if (remained >= prev && remained < next) target = tx * kItems + i;
      }
      fn(target);
    }
    sycl::group_barrier(group);
  }
};

}  // namespace dsa_topk
}  // namespace sgl_kernel
