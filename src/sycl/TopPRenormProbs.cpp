#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <sycl/sycl.hpp>

#include "MemoryAccess.h"
#include "SYCLHelpers.h"
#include "Utils.h"
#include "sgl_kernel_export.h"

namespace {

// SLM/WG constants
constexpr uint32_t kTopPRenormWgSize = 1024;
constexpr uint32_t kTopPRenormSubGroupSize = 32;
constexpr uint32_t kTopPRenormRadixBits = 8;
constexpr uint32_t kTopPRenormRadix = 1u << kTopPRenormRadixBits;
constexpr uint32_t kTopPRenormHistCopies = kTopPRenormWgSize / kTopPRenormSubGroupSize;
constexpr uint32_t kTopPRenormHistElems = kTopPRenormHistCopies * kTopPRenormRadix;

//----------------- bit-pattern transformation --------------------//

inline uint32_t top_p_to_ordered(float val) {
  const uint32_t bits = sycl::bit_cast<uint32_t>(val);
  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

inline float top_p_from_ordered(uint32_t ordered) {
  const uint32_t bits = (ordered & 0x80000000u) ? (ordered ^ 0x80000000u) : ~ordered;
  return sycl::bit_cast<float>(bits);
}

//----------------- radix-select kernel implementation --------------------//
// One work-group processes one row.

template <uint32_t kVecSize>
struct TopPRenormProbsRadixCTA : public __SYCL_KER_CONFIG_CONVENTION__ {
  static constexpr uint32_t kWgSize = kTopPRenormWgSize;
  static constexpr uint32_t kRadixBits = kTopPRenormRadixBits;
  static constexpr uint32_t kRadix = kTopPRenormRadix;
  static constexpr uint32_t kNumRounds = 32u / kRadixBits;
  static constexpr uint32_t kSubGroupSize = kTopPRenormSubGroupSize;
  static constexpr uint32_t kNumHistCopies = kTopPRenormHistCopies;
  static constexpr uint32_t kHistElems = kTopPRenormHistElems;

  const float* probs;
  float* renorm_probs;
  const float* maybe_top_p_arr;
  float top_p_val;
  int batch_size;
  int vocab_size;

  sycl::local_accessor<float, 1> sg_hist_;
  sycl::local_accessor<uint32_t, 1> prefix_;
  // Mass above the bucket being refined, excluding and including that bucket.
  sycl::local_accessor<float, 1> mass_excl_;
  sycl::local_accessor<float, 1> mass_incl_;

  void sycl_ker_config_convention(sycl::handler& cgh) {
    sg_hist_ = sycl::local_accessor<float, 1>(sycl::range<1>(kHistElems), cgh);
    prefix_ = sycl::local_accessor<uint32_t, 1>(sycl::range<1>(1), cgh);
    mass_excl_ = sycl::local_accessor<float, 1>(sycl::range<1>(1), cgh);
    mass_incl_ = sycl::local_accessor<float, 1>(sycl::range<1>(1), cgh);
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

  [[sycl::reqd_sub_group_size(kTopPRenormSubGroupSize)]]
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
    if (p >= 1.0f) {
      float thread_sum = 0.0f;
#pragma unroll 2
      for (uint32_t i = tid; i < num_vec_elems; i += kWgSize) {
        sycl::vec<float, kVecSize> v;
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
      mass_excl_[0] = 0.0f;
      mass_incl_[0] = 1.0f;
    }
    item.barrier(sycl::access::fence_space::local_space);

    for (uint32_t round = 0; round < kNumRounds; ++round) {
      const uint32_t cur_prefix = prefix_[0];
      const float mass_above = mass_excl_[0];
      const uint32_t shift = 32u - (round + 1u) * kRadixBits;
      const uint32_t prefix_mask = (round == 0u) ? 0u : (~0u << (32u - round * kRadixBits));

      for (uint32_t i = tid; i < kHistElems; i += kWgSize) {
        sg_hist_[i] = 0.0f;
      }
      item.barrier(sycl::access::fence_space::local_space);

      // Round 0 has every element active, so atomic contention on the bins peaks;
      // aggregate per sub-group first. Later rounds touch only the elements under
      // the fixed prefix, few enough that plain atomics are cheaper.
      if (round == 0u) {
        histogram_pass<true>(
            sg, tid, row_offset, vocab_u32, num_vec_elems, vec_tail_start, cur_prefix, prefix_mask, shift, hist_base);
      } else {
        histogram_pass<false>(
            sg, tid, row_offset, vocab_u32, num_vec_elems, vec_tail_start, cur_prefix, prefix_mask, shift, hist_base);
      }
      item.barrier(sycl::access::fence_space::local_space);

      for (uint32_t bin = tid; bin < kRadix; bin += kWgSize) {
        float total = 0.0f;
        for (uint32_t s = 1; s < kNumHistCopies; ++s) {
          total += sg_hist_[s * kRadix + bin];
        }
        sg_hist_[bin] += total;
      }
      item.barrier(sycl::access::fence_space::local_space);

      // Walk buckets high to low; the nucleus closes in the first bucket where the running mass reaches p.
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
        mass_excl_[0] = mass_above + (suffix - sg_hist_[chosen]);
        mass_incl_[0] = mass_above + suffix;
      }
      item.barrier(sycl::access::fence_space::local_space);
    }

    const float pivot = top_p_from_ordered(prefix_[0]);
    const float nucleus = mass_incl_[0];
    const float normalizer = sycl::native::recip((nucleus > 1e-8f) ? nucleus : 1e-8f);

    write_scaled(tid, row_offset, vocab_u32, num_vec_elems, vec_tail_start, pivot, normalizer);
  }

  inline void load_vec(sycl::vec<float, kVecSize>& v, size_t row_offset, uint32_t i) const {
    const sycl::vec<float, kVecSize>* q = (const sycl::vec<float, kVecSize>*)(&(probs[row_offset + i * kVecSize]));
    v = *q;
  }

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
      sycl::vec<float, kVecSize> v(0.0f);
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
  template <bool kBallot>
  inline void accumulate(
      sycl::sub_group sg, float val, uint32_t cur_prefix, uint32_t prefix_mask, uint32_t shift, uint32_t hist_base)
      const {
    const uint32_t ordered = top_p_to_ordered(val);
    const bool active = (ordered & prefix_mask) == cur_prefix;

    if constexpr (!kBallot) {
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
      // One sub-group reduction per distinct bucket, so each bucket costs a single
      // atomic from the leader instead of one per lane.
      constexpr uint32_t kSentinel = ~0u;
      uint32_t bucket = active ? ((ordered >> shift) & (kRadix - 1u)) : kSentinel;

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
      sycl::vec<float, kVecSize> v;
      load_vec(v, row_offset, i);
      sycl::vec<float, kVecSize> out;
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        const float val = v[j];
        out[j] = (val >= pivot) ? val * normalizer : 0.0f;
      }
      sycl::vec<float, kVecSize>* r = (sycl::vec<float, kVecSize>*)(&(renorm_probs[row_offset + i * kVecSize]));
      *r = out;
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

  int vec_size = preferred_vector_width(dpcppGetDeviceIdOfCurrentQueue(), sizeof(float));
  vec_size = get_min_vec_size(vec_size, const_cast<float*>(probs_ptr), renorm_probs_ptr);

  // align vec_size with vocab_size
  vec_size = std::gcd(vec_size, vocab_size);

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

SGL_KERNEL_EXPORT void top_p_renorm_probs(
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
