#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <sycl/sycl.hpp>

#include "MemoryAccess.h"
#include "SYCLHelpers.h"
#include "Utils.h"
#include "comm/Sampling.h"

namespace {

// Work-group size shared by the kernel and its launch geometry, so the
// device-side loop stride (kWgSize) and the host-side local range stay in sync.
constexpr uint32_t kTopPRenormWgSize = 1024;

// Map a float onto a uint32 whose unsigned ordering matches the float ordering,
// so radix digits can be compared without decoding the float back. Positives
// get the sign bit set; negatives are inverted. Same transform as the top-k
// radix select in TopKRenormProbs.cpp.
inline uint32_t top_p_to_ordered(float val) {
  const uint32_t bits = sycl::bit_cast<uint32_t>(val);
  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

inline float top_p_from_ordered(uint32_t ordered) {
  const uint32_t bits = (ordered & 0x80000000u) ? (ordered ^ 0x80000000u) : ~ordered;
  return sycl::bit_cast<float>(bits);
}

//----------------- single-cta kernel implementation --------------------//
// One work-group processes one row.
template <uint32_t kVecSize>
struct TopPRenormProbsSingleCTA {
  static constexpr uint32_t kWgSize = kTopPRenormWgSize;

  const float* probs;
  float* renorm_probs;
  const float* maybe_top_p_arr;
  float top_p_val;
  int batch_size;
  int vocab_size;

  TopPRenormProbsSingleCTA(
      const float* probs,
      float* renorm_probs,
      const float* maybe_top_p_arr,
      float top_p_val,
      int batch_size,
      int vocab_size)
      : probs(probs),
        renorm_probs(renorm_probs),
        maybe_top_p_arr(maybe_top_p_arr),
        top_p_val(top_p_val),
        batch_size(batch_size),
        vocab_size(vocab_size) {}

  [[sycl::reqd_sub_group_size(32)]]
  void operator()(sycl::nd_item<1> item) const {
    auto grp = item.get_group();
    const uint32_t row_idx = item.get_group(0);
    if (row_idx >= static_cast<uint32_t>(batch_size)) return;

    const uint32_t tid = item.get_local_id(0);
    const uint32_t vocab_u32 = static_cast<uint32_t>(vocab_size);
    const size_t row_offset = static_cast<size_t>(row_idx) * static_cast<size_t>(vocab_u32);

    const float p = maybe_top_p_arr ? maybe_top_p_arr[row_idx] : top_p_val;

    using vec_io = vec_t<float, kVecSize>;
    const uint32_t num_vec_elems = vocab_u32 / kVecSize;
    const uint32_t vec_tail_start = num_vec_elems * kVecSize;

    // Fast path: p >= 1.0 keeps every element, so just renormalize.
    if (p >= 1.0f) {
      float thread_sum = 0.0f;
#pragma unroll 2
      for (uint32_t i = tid; i < num_vec_elems; i += kWgSize) {
        vec_io v;
        v.load(
            0,
            sycl::multi_ptr<const float, sycl::access::address_space::global_space>(probs + row_offset + i * kVecSize));
#pragma unroll
        for (uint32_t j = 0; j < kVecSize; ++j) {
          thread_sum += v[j];
        }
      }
      for (uint32_t col = vec_tail_start + tid; col < vocab_u32; col += kWgSize) {
        thread_sum += probs[row_offset + col];
      }

      const float row_sum = sycl::reduce_over_group(grp, thread_sum, sycl::plus<float>());
      const float normalizer = (row_sum <= 1e-8f) ? 1.0f : sycl::native::recip(row_sum);

#pragma unroll 2
      for (uint32_t i = tid; i < num_vec_elems; i += kWgSize) {
        vec_io v;
        v.load(
            0,
            sycl::multi_ptr<const float, sycl::access::address_space::global_space>(probs + row_offset + i * kVecSize));
        vec_io out;
#pragma unroll
        for (uint32_t j = 0; j < kVecSize; ++j) {
          out[j] = v[j] * normalizer;
        }
        out.store(
            0,
            sycl::multi_ptr<float, sycl::access::address_space::global_space>(
                renorm_probs + row_offset + i * kVecSize));
      }
      for (uint32_t col = vec_tail_start + tid; col < vocab_u32; col += kWgSize) {
        renorm_probs[row_offset + col] = probs[row_offset + col] * normalizer;
      }
      return;
    }

    // Compute the maximum probability in the row.
    const float max_val = sgl::sampling::get_max_value<float, kVecSize, kWgSize>(grp, probs, row_idx, tid, vocab_u32);

    // TERNARY SEARCH:  for the pivot threshold `low` such that keeping probs > low
    // yields cumulative mass >= p while being the minimal such nucleus.
    // WINDOW:      low, high              ← lower bound of the window; also the final answer (the pivot cutoff), and
    // the upper bound of the window PROBES:      pivot_0, pivot_1       ← the ⅓ and ⅔ test points MASSES:      agg0,
    // agg1            ← f(pivot_0) = sum of probs > pivot_0, f(pivot_1) = sum of probs > pivot_1 BRACKETS: min_gt_low,
    // max_le_high ← 	smallest actual prob still > low, largest actual prob still <= high NORMALIZER:  sum_low ←
    // surviving mass, saved for renorm
    double low = 0.0, high = static_cast<double>(max_val);
    float min_gt_low = static_cast<float>(high);
    float max_le_high = static_cast<float>(low);
    float sum_low = 1.0f;

    do {
      const double pivot_0 = (high + 2.0 * low) / 3.0;
      const double pivot_1 = (2.0 * high + low) / 3.0;

      float thr_agg0 = 0.0f, thr_agg1 = 0.0f;
      float thr_min_gt_low = static_cast<float>(high);
      float thr_max_le_high = static_cast<float>(low);

#pragma unroll 2
      for (uint32_t i = tid; i < num_vec_elems; i += kWgSize) {
        vec_io v;
        v.load(
            0,
            sycl::multi_ptr<const float, sycl::access::address_space::global_space>(probs + row_offset + i * kVecSize));
#pragma unroll
        for (uint32_t j = 0; j < kVecSize; ++j) {
          const float val = v[j];
          if (val > pivot_0) thr_agg0 += val;
          if (val > pivot_1) thr_agg1 += val;
          if (val > low) thr_min_gt_low = (val < thr_min_gt_low) ? val : thr_min_gt_low;
          if (val <= high) thr_max_le_high = (val > thr_max_le_high) ? val : thr_max_le_high;
        }
      }
      for (uint32_t col = vec_tail_start + tid; col < vocab_u32; col += kWgSize) {
        const float val = probs[row_offset + col];
        if (val > pivot_0) thr_agg0 += val;
        if (val > pivot_1) thr_agg1 += val;
        if (val > low) thr_min_gt_low = (val < thr_min_gt_low) ? val : thr_min_gt_low;
        if (val <= high) thr_max_le_high = (val > thr_max_le_high) ? val : thr_max_le_high;
      }

      const float agg0 = sycl::reduce_over_group(grp, thr_agg0, sycl::plus<float>());
      const float agg1 = sycl::reduce_over_group(grp, thr_agg1, sycl::plus<float>());
      min_gt_low = sycl::reduce_over_group(grp, thr_min_gt_low, sycl::minimum<float>());
      max_le_high = sycl::reduce_over_group(grp, thr_max_le_high, sycl::maximum<float>());

      // The goal each iteration: shrink [low, high] toward the pivot, where f(x) = sum(probs > x)
      // is non-increasing, and we want the highest cutoff where f(x) ≥ p still holds.
      // case A:  f(pivot_1) ≥ p, so the pivot is too low; shrink the lower bound to pivot_1.
      // case B:  f(pivot_1) < p ≤ f(pivot_0), so the pivot is too low; shrink the lower bound to pivot_0,
      //          and shrink the upper bound to the largest prob ≤ high (which is still ≥ pivot_1).
      // case C:  f(pivot_0) < p, so the pivot is too high; shrink the upper bound to pivot_0,
      //          and shrink the lower bound to the smallest prob > low (which is still ≤ pivot_0).
      const bool keep1 = agg1 >= p;  // case A
      const bool keep0 = agg0 >= p;  // case A or B
      const double mlh = static_cast<double>(max_le_high);
      const double hi_shrink_src = keep0 ? pivot_1 : pivot_0;
      const double hi_shrunk = (hi_shrink_src < mlh) ? hi_shrink_src : mlh;

      low = keep1 ? pivot_1 : (keep0 ? pivot_0 : low);
      high = keep1 ? high : hi_shrunk;
      sum_low = keep1 ? agg1 : (keep0 ? agg0 : sum_low);
    } while (min_gt_low < max_le_high && sycl::nextafter(min_gt_low, max_le_high) < max_le_high);

    const float normalizer = sycl::native::recip((sum_low > 1e-8f) ? sum_low : 1e-8f);

#pragma unroll 2
    for (uint32_t i = tid; i < num_vec_elems; i += kWgSize) {
      vec_io v;
      v.load(
          0,
          sycl::multi_ptr<const float, sycl::access::address_space::global_space>(probs + row_offset + i * kVecSize));
      vec_io out;
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        const float val = v[j];
        out[j] = (val > low) ? val * normalizer : 0.0f;
      }
      out.store(
          0,
          sycl::multi_ptr<float, sycl::access::address_space::global_space>(renorm_probs + row_offset + i * kVecSize));
    }
    for (uint32_t col = vec_tail_start + tid; col < vocab_u32; col += kWgSize) {
      const float val = probs[row_offset + col];
      renorm_probs[row_offset + col] = (val > low) ? val * normalizer : 0.0f;
    }
  }
};

//----------------- radix-select kernel implementation --------------------//
// One work-group processes one row.
//
// The ternary-search kernel above rescans the row once per bisection step, and
// the step count depends on how many *distinct* values the row holds (it stops
// once the bracket closes to one ULP): ~13 rescans at vocab=128256 for fp32
// inputs. The kernel is bandwidth-bound, so that read amplification is the
// whole cost.
//
// This variant runs a radix select over the order-preserving bit patterns
// instead, which is a fixed 4 rescans regardless of vocab or input
// distribution. Where the top-k select in TopKRenormProbs.cpp descends on
// bucket *counts*, top-p descends on bucket *sums*: narrow to the bucket where
// the accumulated mass first reaches p. The descent tracks the surviving mass
// as it goes, so the normalizer falls out for free and no extra sum pass is
// needed.
template <uint32_t kVecSize>
struct TopPRenormProbsRadixCTA : public __SYCL_KER_CONFIG_CONVENTION__ {
  static constexpr uint32_t kWgSize = kTopPRenormWgSize;
  static constexpr uint32_t kRadixBits = 8;
  static constexpr uint32_t kRadix = 1u << kRadixBits;
  static constexpr uint32_t kNumRounds = 32u / kRadixBits;
  static constexpr uint32_t kSubGroupSize = 32;
  // One histogram per sub-group. Rounds 1-3 jam every element straight into local
  // memory with a plain atomic, so the replicas are what keep those atomics from
  // all landing on the same address.
  static constexpr uint32_t kNumHistCopies = kWgSize / kSubGroupSize;
  static constexpr uint32_t kHistElems = kNumHistCopies * kRadix;

  using vec_io = vec_t<float, kVecSize>;

  const float* probs;
  float* renorm_probs;
  const float* maybe_top_p_arr;
  float top_p_val;
  int batch_size;
  int vocab_size;

  // kNumHistCopies histograms, so the accumulating atomics spread over several
  // addresses instead of all landing on one; folded together after each pass.
  sycl::local_accessor<float, 1> sg_hist_;
  // prefix_[0]: the ordered-bit prefix fixed by the rounds so far.
  sycl::local_accessor<uint32_t, 1> prefix_;
  // mass_[0]: mass strictly above the chosen bucket. mass_[1]: surviving mass.
  sycl::local_accessor<float, 1> mass_;

  void sycl_ker_config_convention(sycl::handler& cgh) {
    sg_hist_ = sycl::local_accessor<float, 1>(sycl::range<1>(kHistElems), cgh);
    prefix_ = sycl::local_accessor<uint32_t, 1>(sycl::range<1>(1), cgh);
    mass_ = sycl::local_accessor<float, 1>(sycl::range<1>(2), cgh);
  }

  TopPRenormProbsRadixCTA(
      const float* probs,
      float* renorm_probs,
      const float* maybe_top_p_arr,
      float top_p_val,
      int batch_size,
      int vocab_size)
      : probs(probs),
        renorm_probs(renorm_probs),
        maybe_top_p_arr(maybe_top_p_arr),
        top_p_val(top_p_val),
        batch_size(batch_size),
        vocab_size(vocab_size) {}

  [[sycl::reqd_sub_group_size(32)]]
  void operator()(sycl::nd_item<1> item) const {
    auto grp = item.get_group();
    const uint32_t row_idx = item.get_group(0);
    if (row_idx >= static_cast<uint32_t>(batch_size)) return;

    const uint32_t tid = item.get_local_id(0);
    const uint32_t vocab_u32 = static_cast<uint32_t>(vocab_size);
    const size_t row_offset = static_cast<size_t>(row_idx) * static_cast<size_t>(vocab_u32);
    const uint32_t num_vec_elems = vocab_u32 / kVecSize;
    const uint32_t vec_tail_start = num_vec_elems * kVecSize;

    const float p = maybe_top_p_arr ? maybe_top_p_arr[row_idx] : top_p_val;

    // Fast path: p >= 1.0 keeps every element, so only the renormalize is left.
    // Two passes instead of five, skipping the select entirely.
    if (p >= 1.0f) {
      float thread_sum = 0.0f;
#pragma unroll 2
      for (uint32_t i = tid; i < num_vec_elems; i += kWgSize) {
        vec_io v;
        load_vec(v, row_offset, i);
#pragma unroll
        for (uint32_t j = 0; j < kVecSize; ++j) {
          thread_sum += v[j];
        }
      }
      for (uint32_t col = vec_tail_start + tid; col < vocab_u32; col += kWgSize) {
        thread_sum += probs[row_offset + col];
      }

      const float row_sum = sycl::reduce_over_group(grp, thread_sum, sycl::plus<float>());
      const float normalizer = (row_sum <= 1e-8f) ? 1.0f : sycl::native::recip(row_sum);
      // -inf keeps every element, matching the p >= 1.0 semantics.
      write_scaled(
          tid,
          row_offset,
          vocab_u32,
          num_vec_elems,
          vec_tail_start,
          -std::numeric_limits<float>::infinity(),
          normalizer);
      return;
    }

    sycl::sub_group sg = item.get_sub_group();
    const uint32_t hist_base = (sg.get_group_id()[0] % kNumHistCopies) * kRadix;

    if (tid == 0) {
      prefix_[0] = 0u;
      mass_[0] = 0.0f;
      mass_[1] = 1.0f;
    }
    item.barrier(sycl::access::fence_space::local_space);

    for (uint32_t round = 0; round < kNumRounds; ++round) {
      const uint32_t cur_prefix = prefix_[0];
      const float mass_above = mass_[0];
      const uint32_t shift = 32u - (round + 1u) * kRadixBits;
      // Bits fixed by earlier rounds; round 0 accepts every element.
      const uint32_t prefix_mask = (round == 0u) ? 0u : (~0u << (32u - round * kRadixBits));

      for (uint32_t i = tid; i < kHistElems; i += kWgSize) {
        sg_hist_[i] = 0.0f;
      }
      item.barrier(sycl::access::fence_space::local_space);

      // Round 0's leading digit is degenerate, so it wants the ballot; the mantissa
      // rounds want plain atomics. See accumulate().
      if (round == 0u) {
        histogram_pass<true>(
            sg, tid, row_offset, vocab_u32, num_vec_elems, vec_tail_start, cur_prefix, prefix_mask, shift, hist_base);
      } else {
        histogram_pass<false>(
            sg, tid, row_offset, vocab_u32, num_vec_elems, vec_tail_start, cur_prefix, prefix_mask, shift, hist_base);
      }
      item.barrier(sycl::access::fence_space::local_space);

      // Fold the copies down into bins [0, kRadix).
      for (uint32_t bin = tid; bin < kRadix; bin += kWgSize) {
        float total = 0.0f;
        for (uint32_t s = 1; s < kNumHistCopies; ++s) {
          total += sg_hist_[s * kRadix + bin];
        }
        sg_hist_[bin] += total;
      }
      item.barrier(sycl::access::fence_space::local_space);

      // Walk buckets high to low; the nucleus closes in the first bucket where
      // the running mass reaches p. If it never does (the whole row sums under
      // p) `chosen` stays 0, which keeps everything.
      if (tid == 0) {
        uint32_t chosen = 0u;
        float suffix = 0.0f;
        for (int bin = static_cast<int>(kRadix) - 1; bin >= 0; --bin) {
          suffix += sg_hist_[bin];
          if (mass_above + suffix >= p) {
            chosen = static_cast<uint32_t>(bin);
            break;
          }
        }
        prefix_[0] = cur_prefix | (chosen << shift);
        mass_[0] = mass_above + (suffix - sg_hist_[chosen]);
        mass_[1] = mass_above + suffix;
      }
      item.barrier(sycl::access::fence_space::local_space);
    }

    // Every bit is fixed now, so the pivot is an actual element of the row and
    // its bucket holds all values tied with it. Keeping `val >= pivot` therefore
    // keeps the whole tie group, matching the top-k convention.
    const float pivot = top_p_from_ordered(prefix_[0]);
    const float nucleus = mass_[1];
    const float normalizer = sycl::native::recip((nucleus > 1e-8f) ? nucleus : 1e-8f);

    write_scaled(tid, row_offset, vocab_u32, num_vec_elems, vec_tail_start, pivot, normalizer);
  }

  inline void load_vec(vec_io& v, size_t row_offset, uint32_t i) const {
    v.load(
        0, sycl::multi_ptr<const float, sycl::access::address_space::global_space>(probs + row_offset + i * kVecSize));
  }

  // Accumulate the whole row into sg_hist_ for one radix round.
  //
  // The ballot variant of accumulate() reduces across the sub-group, so every lane
  // has to reach it the same number of times: the loops below therefore run a trip
  // count uniform over the work-group and pass 0.0f for out-of-range lanes, which
  // lands in no bucket and contributes nothing.
  template <bool kBallot>
  inline void histogram_pass(
      sycl::sub_group sg,
      uint32_t tid,
      size_t row_offset,
      uint32_t vocab_u32,
      uint32_t num_vec_elems,
      uint32_t vec_tail_start,
      uint32_t cur_prefix,
      uint32_t prefix_mask,
      uint32_t shift,
      uint32_t hist_base) const {
    const uint32_t vec_iters = div_up(num_vec_elems, kWgSize);
    for (uint32_t it = 0; it < vec_iters; ++it) {
      const uint32_t i = it * kWgSize + tid;
      vec_io v(0.0f);
      if (i < num_vec_elems) load_vec(v, row_offset, i);
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        accumulate<kBallot>(sg, i < num_vec_elems ? v[j] : 0.0f, cur_prefix, prefix_mask, shift, hist_base);
      }
    }
    const uint32_t tail_iters = div_up(vocab_u32 - vec_tail_start, kWgSize);
    for (uint32_t it = 0; it < tail_iters; ++it) {
      const uint32_t col = vec_tail_start + it * kWgSize + tid;
      accumulate<kBallot>(
          sg, col < vocab_u32 ? probs[row_offset + col] : 0.0f, cur_prefix, prefix_mask, shift, hist_base);
    }
  }

  // Add `val` into its bucket, if it still matches the prefix fixed so far.
  //
  // Two strategies, because the bucket distribution differs sharply by round.
  //
  // Round 0 keys on the sign plus 7 exponent bits, and probabilities cluster into a
  // couple of exponent octaves, so ~half the row lands in one bucket and a plain
  // atomic would serialize the whole sub-group onto one address. There the
  // sub-group first agrees on which buckets it holds and reduces each locally, so a
  // bucket costs one atomic however many lanes feed it. `kSentinel` marks a lane as
  // done; real buckets are always < kRadix, so it can never collide with one.
  //
  // Rounds 1-3 key on mantissa bits, which are close to uniform over the 256
  // buckets: 32 lanes hold ~32 distinct buckets, so that ballot runs ~32 iterations
  // of two sub-group reductions each -- roughly 10-20 shuffle ops per element to
  // save a single uncontended atomic. There a direct atomic is far cheaper, and the
  // per-sub-group replicas keep the addresses spread.
  template <bool kBallot>
  inline void accumulate(
      sycl::sub_group sg, float val, uint32_t cur_prefix, uint32_t prefix_mask, uint32_t shift, uint32_t hist_base)
      const {
    const uint32_t ordered = top_p_to_ordered(val);
    const bool active = (ordered & prefix_mask) == cur_prefix;

    if constexpr (!kBallot) {
      // `val == 0.0f` marks an out-of-range lane and contributes nothing, so it can
      // take the atomic unconditionally rather than diverging.
      if (active && val != 0.0f) {
        const uint32_t bucket = (ordered >> shift) & (kRadix - 1u);
        sycl::atomic_ref<
            float,
            sycl::memory_order::relaxed,
            sycl::memory_scope::work_group,
            sycl::access::address_space::local_space>(sg_hist_[hist_base + bucket])
            .fetch_add(val);
      }
      return;
    } else {
      constexpr uint32_t kSentinel = ~0u;
      uint32_t bucket = active ? ((ordered >> shift) & (kRadix - 1u)) : kSentinel;

      // Uniform across the sub-group, so every lane runs the same trip count and
      // the reductions below stay convergent.
      while (true) {
        const uint32_t lead = sycl::reduce_over_group(sg, bucket, sycl::minimum<uint32_t>());
        if (lead == kSentinel) break;
        const bool mine = (bucket == lead);
        const float bucket_sum = sycl::reduce_over_group(sg, mine ? val : 0.0f, sycl::plus<float>());
        if (sg.leader()) {
          sycl::atomic_ref<
              float,
              sycl::memory_order::relaxed,
              sycl::memory_scope::work_group,
              sycl::access::address_space::local_space>(sg_hist_[hist_base + lead])
              .fetch_add(bucket_sum);
        }
        if (mine) bucket = kSentinel;
      }
    }
  }

  // Scale the row by `normalizer`, zeroing everything below `pivot`.
  inline void write_scaled(
      uint32_t tid,
      size_t row_offset,
      uint32_t vocab_u32,
      uint32_t num_vec_elems,
      uint32_t vec_tail_start,
      float pivot,
      float normalizer) const {
#pragma unroll 2
    for (uint32_t i = tid; i < num_vec_elems; i += kWgSize) {
      vec_io v;
      load_vec(v, row_offset, i);
      vec_io out;
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        const float val = v[j];
        out[j] = (val >= pivot) ? val * normalizer : 0.0f;
      }
      out.store(
          0,
          sycl::multi_ptr<float, sycl::access::address_space::global_space>(renorm_probs + row_offset + i * kVecSize));
    }
    for (uint32_t col = vec_tail_start + tid; col < vocab_u32; col += kWgSize) {
      const float val = probs[row_offset + col];
      renorm_probs[row_offset + col] = (val >= pivot) ? val * normalizer : 0.0f;
    }
  }
};

void launch_top_p_renorm_kernel(
    at::Tensor& renorm_probs,
    const at::Tensor& probs,
    const float* maybe_top_p_ptr,
    float top_p_val,
    int batch_size,
    int vocab_size,
    sycl::queue& queue) {
  const float* probs_ptr = probs.data_ptr<float>();
  float* renorm_probs_ptr = renorm_probs.data_ptr<float>();

  const int local_size = kTopPRenormWgSize;
  const int global_size = batch_size * local_size;

  // Pick the vectorization width the device prefers for a float element,
  // matching the approach used by per_tensor_quant_fp8, instead of hardcoding 4.
  // Then clamp it to what the actual buffers support: probs_ptr/renorm_probs_ptr
  // are only guaranteed element-aligned (row_offset is a plain element count), so
  // an unlucky base address can leave them short of 16B/8B alignment even when the
  // device prefers a wide vector, matching the alignment check RMSNorm.cpp uses via
  // get_min_vec_size.
  int vec_size = preferred_vector_width(dpcppGetDeviceIdOfCurrentQueue(), sizeof(float));
  vec_size = get_min_vec_size(vec_size, const_cast<float*>(probs_ptr), renorm_probs_ptr);

#define LAUNCH_TOP_P(VEC_SIZE)           \
  sycl_kernel_submit(                    \
      global_size,                       \
      local_size,                        \
      queue,                             \
      TopPRenormProbsRadixCTA<VEC_SIZE>( \
          probs_ptr, renorm_probs_ptr, maybe_top_p_ptr, top_p_val, batch_size, vocab_size))

  switch (vec_size) {
    case 16:
      LAUNCH_TOP_P(16);
      break;
    case 8:
      LAUNCH_TOP_P(8);
      break;
    case 4:
      LAUNCH_TOP_P(4);
      break;
    case 2:
      LAUNCH_TOP_P(2);
      break;
    default:
      LAUNCH_TOP_P(1);
  }

#undef LAUNCH_TOP_P
}

}  // namespace

void top_p_renorm_probs(
    const at::Tensor& probs,
    at::Tensor& renorm_probs,
    const std::optional<at::Tensor>& maybe_top_p_arr,
    double top_p_val) {
  CHECK_INPUT(probs);
  CHECK_INPUT(renorm_probs);
  TORCH_CHECK(probs.dim() == 2, "probs must be a 2D tensor [batch_size, vocab_size]");
  TORCH_CHECK(renorm_probs.dim() == 2, "renorm_probs must be a 2D tensor [batch_size, vocab_size]");
  TORCH_CHECK(probs.sizes() == renorm_probs.sizes(), "Input tensors must have the same shape");
  TORCH_CHECK(probs.scalar_type() == torch::kFloat32, "probs must be float32");
  TORCH_CHECK(renorm_probs.scalar_type() == torch::kFloat32, "renorm_probs must be float32");

  if (maybe_top_p_arr.has_value()) {
    CHECK_INPUT((*maybe_top_p_arr));
    TORCH_CHECK(maybe_top_p_arr->dim() == 1, "maybe_top_p_arr must be a 1D tensor [batch_size]");
    TORCH_CHECK(maybe_top_p_arr->size(0) == probs.size(0), "maybe_top_p_arr size must match batch_size");
    TORCH_CHECK(maybe_top_p_arr->scalar_type() == torch::kFloat32, "maybe_top_p_arr must be float32");
  } else {
    TORCH_CHECK(top_p_val > 0.0 && top_p_val <= 1.0, "top_p_val must be in (0, 1]");
  }

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();
  int batch_size = probs.size(0);
  int vocab_size = probs.size(1);

  const float* maybe_top_p_ptr = maybe_top_p_arr.has_value() ? maybe_top_p_arr->data_ptr<float>() : nullptr;

  launch_top_p_renorm_kernel(
      renorm_probs, probs, maybe_top_p_ptr, static_cast<float>(top_p_val), batch_size, vocab_size, queue);
}
