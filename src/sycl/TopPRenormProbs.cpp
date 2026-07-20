#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstdint>
#include <optional>
#include <sycl/sycl.hpp>

#include "MemoryAccess.h"
#include "SYCLHelpers.h"
#include "Utils.h"

namespace {

//----------------- set element type options --------------------//

template <typename T>
struct ToSyclElementTypeP {
  using type = T;
};

template <>
struct ToSyclElementTypeP<at::Half> {
  using type = sycl::half;
};

template <>
struct ToSyclElementTypeP<at::BFloat16> {
  using type = sycl::ext::oneapi::bfloat16;
};

// Work-group size shared by both kernels and their launch geometry, so the
// device-side loop stride (kWgSize) and the host-side local range stay in sync.
constexpr uint32_t kTopPRenormWgSize = 1024;

// Vocabularies at or below this size get exact tie handling. Small vocabs are
// where a mishandled boundary tie carries non-negligible mass (each element is
// ~1/vocab), and where the extra O(vocab) tie scan is cheap. Used by both the
// ternary kernel's tie-aware final pass and the exact kernel's dispatch.
constexpr int kExactVocabThreshold = 1024;

//----------------- shared kernel parameters --------------------//
// Both top-p renorm kernels take the same inputs, so the common data members
// and constructor live in this base. It holds no SYCL accessors and has no
// virtual members, so it stays trivially copyable (i.e. device-copyable).
// Because the base is a template, derived kernels bring the members back into
// unqualified scope with `using Base::member;` declarations.

template <typename DType>
struct TopPRenormProbsParams {
  static constexpr uint32_t kWgSize = kTopPRenormWgSize;

  const DType* probs;
  DType* renorm_probs;
  const float* maybe_top_p_arr;
  float top_p_val;
  int batch_size;
  int vocab_size;

  TopPRenormProbsParams(
      DType* renorm_probs,
      const DType* probs,
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
};

//----------------- single-cta kernel implementation --------------------//
// One work-group processes one row. The pivot threshold that defines the
// top-p nucleus is found via a ternary search on f(x) = sum(probs[probs > x]),
// which is non-increasing. This mirrors the flashinfer TopPRenormProb kernel.

template <typename DType, uint32_t kVecSize>
struct TopPRenormProbsSingleCTA : public TopPRenormProbsParams<DType> {
  using Base = TopPRenormProbsParams<DType>;
  using Base::Base;  // inherit the constructor
  using Base::batch_size;
  using Base::kWgSize;
  using Base::maybe_top_p_arr;
  using Base::probs;
  using Base::renorm_probs;
  using Base::top_p_val;
  using Base::vocab_size;

  [[sycl::reqd_sub_group_size(32)]]
  void operator()(sycl::nd_item<1> item) const {
    auto grp = item.get_group();
    const uint32_t row_idx = item.get_group(0);
    const uint32_t tid = item.get_local_id(0);
    const uint32_t vocab_u32 = static_cast<uint32_t>(vocab_size);
    const size_t row_offset = static_cast<size_t>(row_idx) * static_cast<size_t>(vocab_u32);

    const float p = maybe_top_p_arr ? maybe_top_p_arr[row_idx] : top_p_val;

    using vec_io = vec_t<DType, kVecSize>;
    const uint32_t num_vec_elems = vocab_u32 / kVecSize;
    const uint32_t vec_tail_start = num_vec_elems * kVecSize;

    // Fast path: p >= 1.0 keeps every element, so just renormalize.
    if (p >= 1.0f) {
      float thread_sum = 0.0f;
      for (uint32_t i = tid; i < num_vec_elems; i += kWgSize) {
        vec_io v;
        v.load(
            0,
            sycl::multi_ptr<const DType, sycl::access::address_space::global_space>(probs + row_offset + i * kVecSize));
#pragma unroll
        for (uint32_t j = 0; j < kVecSize; ++j) {
          thread_sum += static_cast<float>(v[j]);
        }
      }
      for (uint32_t col = vec_tail_start + tid; col < vocab_u32; col += kWgSize) {
        thread_sum += static_cast<float>(probs[row_offset + col]);
      }

      const float row_sum = sycl::reduce_over_group(grp, thread_sum, sycl::plus<float>());
      const float normalizer = (row_sum <= 1e-8f) ? 1.0f : 1.0f / row_sum;

      for (uint32_t i = tid; i < num_vec_elems; i += kWgSize) {
        vec_io v;
        v.load(
            0,
            sycl::multi_ptr<const DType, sycl::access::address_space::global_space>(probs + row_offset + i * kVecSize));
        vec_io out;
#pragma unroll
        for (uint32_t j = 0; j < kVecSize; ++j) {
          out[j] = static_cast<DType>(static_cast<float>(v[j]) * normalizer);
        }
        out.store(
            0,
            sycl::multi_ptr<DType, sycl::access::address_space::global_space>(
                renorm_probs + row_offset + i * kVecSize));
      }
      for (uint32_t col = vec_tail_start + tid; col < vocab_u32; col += kWgSize) {
        renorm_probs[row_offset + col] = static_cast<DType>(static_cast<float>(probs[row_offset + col]) * normalizer);
      }
      return;
    }

    // Compute the maximum probability in the row.
    float thread_max = 0.0f;
    for (uint32_t i = tid; i < num_vec_elems; i += kWgSize) {
      vec_io v;
      v.load(
          0,
          sycl::multi_ptr<const DType, sycl::access::address_space::global_space>(probs + row_offset + i * kVecSize));
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        const float val = static_cast<float>(v[j]);
        thread_max = (val > thread_max) ? val : thread_max;
      }
    }
    for (uint32_t col = vec_tail_start + tid; col < vocab_u32; col += kWgSize) {
      const float val = static_cast<float>(probs[row_offset + col]);
      thread_max = (val > thread_max) ? val : thread_max;
    }
    const float max_val = sycl::reduce_over_group(grp, thread_max, sycl::maximum<float>());

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

      for (uint32_t i = tid; i < num_vec_elems; i += kWgSize) {
        vec_io v;
        v.load(
            0,
            sycl::multi_ptr<const DType, sycl::access::address_space::global_space>(probs + row_offset + i * kVecSize));
#pragma unroll
        for (uint32_t j = 0; j < kVecSize; ++j) {
          const float val = static_cast<float>(v[j]);
          if (val > pivot_0) thr_agg0 += val;
          if (val > pivot_1) thr_agg1 += val;
          if (val > low) thr_min_gt_low = (val < thr_min_gt_low) ? val : thr_min_gt_low;
          if (val <= high) thr_max_le_high = (val > thr_max_le_high) ? val : thr_max_le_high;
        }
      }
      for (uint32_t col = vec_tail_start + tid; col < vocab_u32; col += kWgSize) {
        const float val = static_cast<float>(probs[row_offset + col]);
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
      if (agg1 >= p) {
        low = pivot_1;
        sum_low = agg1;
      } else if (agg0 >= p) {
        low = pivot_0;
        const double mlh = static_cast<double>(max_le_high);
        high = (pivot_1 < mlh) ? pivot_1 : mlh;
        sum_low = agg0;
      } else {
        const double mlh = static_cast<double>(max_le_high);
        high = (pivot_0 < mlh) ? pivot_0 : mlh;
      }
    } while (min_gt_low != max_le_high);

    const float normalizer = 1.0f / ((sum_low > 1e-8f) ? sum_low : 1e-8f);

    for (uint32_t i = tid; i < num_vec_elems; i += kWgSize) {
      vec_io v;
      v.load(
          0,
          sycl::multi_ptr<const DType, sycl::access::address_space::global_space>(probs + row_offset + i * kVecSize));
      vec_io out;
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        const float val = static_cast<float>(v[j]);
        out[j] = static_cast<DType>((val > low) ? val * normalizer : 0.0f);
      }
      out.store(
          0,
          sycl::multi_ptr<DType, sycl::access::address_space::global_space>(renorm_probs + row_offset + i * kVecSize));
    }
    for (uint32_t col = vec_tail_start + tid; col < vocab_u32; col += kWgSize) {
      const float val = static_cast<float>(probs[row_offset + col]);
      renorm_probs[row_offset + col] = static_cast<DType>((val > low) ? val * normalizer : 0.0f);
    }
  }
};

//----------------- exact small-vocab kernel implementation --------------------//
// One work-group processes one row. This reproduces the reference top-p rule
// exactly (including tie-breaking): sort ascending with ties ordered by
// ascending original index, take the suffix whose cumulative mass >= 1 - p.
//
// Equivalently, without any sort, element `e` is kept iff
//   sum(probs[j] : probs[j] < probs[e])
//     + sum(probs[j] : probs[j] == probs[e] && j <= e)   >= 1 - p
//
// The row is small enough (<= kExactVocabThreshold) to cache in shared local
// memory, so the O(vocab^2) comparison stays cheap. Unlike the ternary-search
// kernel this splits tied values individually, matching the reference bit for
// bit under bf16 rounding where a tie can straddle the nucleus boundary.
//
// WHY A SEPARATE KERNEL INSTEAD OF ONE UNIFIED KERNEL?
// The two vocab regimes want structurally opposite kernels, so one body cannot
// be efficient for both:
//   * Large vocab (32k-152k): sorting or an O(V^2) scan is far too expensive,
//     and the row does not fit in shared local memory (SLM). The ternary-search
//     kernel is O(V*iters) with iters~20 (constant in V), streams from global
//     memory with coalesced vectorized loads, and needs zero SLM. Exact
//     index-tie-splitting is unnecessary here: each element carries ~1/V mass,
//     so a mishandled boundary tie is below rounding tolerance anyway.
//   * Small vocab (<= 1024): exact tie-splitting DOES matter (bf16/fp16 create
//     many exact ties and each element's ~1/V mass is large enough to exceed
//     tolerance when a tie flips), and the row fits entirely in SLM. This kernel
//     caches the row once and does the O(V^2) cdf scan from fast SLM in a single
//     pass -- worst-case-stable even when the tie group is O(V), which is
//     precisely the failing bf16 regime.
// Folding both into one kernel would mean either (a) paying the ternary kernel's
// global-memory tie-correction pass, which recomputes ranks and degrades to
// O(V^2) in *global* memory on heavy-tie bf16 rows -- worse than this kernel's
// SLM O(V^2); or (b) allocating SLM and running the O(V^2) scan for large vocab
// too, which blows the SLM budget and the runtime. The launcher therefore
// dispatches by vocab size and each kernel stays specialized for its regime.

template <typename DType, uint32_t kVecSize>
struct TopPRenormProbsExactSingleCTA : public TopPRenormProbsParams<DType>, public __SYCL_KER_CONFIG_CONVENTION__ {
  using Base = TopPRenormProbsParams<DType>;
  using Base::Base;  // inherit the constructor
  using Base::batch_size;
  using Base::kWgSize;
  using Base::maybe_top_p_arr;
  using Base::probs;
  using Base::renorm_probs;
  using Base::top_p_val;
  using Base::vocab_size;

  sycl::local_accessor<float, 1> sh_prob_;
  sycl::local_accessor<uint8_t, 1> sh_keep_;

  void sycl_ker_config_convention(sycl::handler& cgh) {
    sh_prob_ = sycl::local_accessor<float, 1>(sycl::range<1>(vocab_size), cgh);
    sh_keep_ = sycl::local_accessor<uint8_t, 1>(sycl::range<1>(vocab_size), cgh);
  }

  [[sycl::reqd_sub_group_size(32)]]
  void operator()(sycl::nd_item<1> item) const {
    auto grp = item.get_group();
    const uint32_t row_idx = item.get_group(0);
    if (row_idx >= static_cast<uint32_t>(batch_size)) return;

    const uint32_t tid = item.get_local_id(0);
    const uint32_t vocab_u32 = static_cast<uint32_t>(vocab_size);
    const size_t row_offset = static_cast<size_t>(row_idx) * static_cast<size_t>(vocab_u32);

    const float p = maybe_top_p_arr ? maybe_top_p_arr[row_idx] : top_p_val;
    const float threshold = 1.0f - p;

    using vec_in = vec_t<DType, kVecSize>;
    const uint32_t num_vec_elems = vocab_u32 / kVecSize;
    const uint32_t vec_tail_start = num_vec_elems * kVecSize;

    // Cache the row in shared local memory (promoted to float, matching the
    // reference which runs entirely in float32). Vectorized global load.
    for (uint32_t i = tid; i < num_vec_elems; i += kWgSize) {
      vec_in v;
      v.load(
          0,
          sycl::multi_ptr<const DType, sycl::access::address_space::global_space>(probs + row_offset + i * kVecSize));
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        sh_prob_[i * kVecSize + j] = static_cast<float>(v[j]);
      }
    }
    for (uint32_t col = vec_tail_start + tid; col < vocab_u32; col += kWgSize) {
      sh_prob_[col] = static_cast<float>(probs[row_offset + col]);
    }
    item.barrier(sycl::access::fence_space::local_space);

    // For each owned element decide whether it belongs to the nucleus, and
    // accumulate the mass of the kept elements for renormalization. The inner
    // cdf loop reads shared memory in vectors of kVecSize, accumulating in the
    // same ascending-index order as the scalar version so the float result is
    // bit-identical.
    float thread_kept_sum = 0.0f;
    for (uint32_t col = tid; col < vocab_u32; col += kWgSize) {
      const float val_e = sh_prob_[col];
      float cdf = 0.0f;
      for (uint32_t i = 0; i < num_vec_elems; ++i) {
        const uint32_t base = i * kVecSize;
#pragma unroll
        for (uint32_t j = 0; j < kVecSize; ++j) {
          const uint32_t idx = base + j;
          const float val_j = sh_prob_[idx];
          // ascending-sort prefix with ties broken by ascending index
          if (val_j < val_e || (val_j == val_e && idx <= col)) {
            cdf += val_j;
          }
        }
      }
      for (uint32_t idx = vec_tail_start; idx < vocab_u32; ++idx) {
        const float val_j = sh_prob_[idx];
        if (val_j < val_e || (val_j == val_e && idx <= col)) {
          cdf += val_j;
        }
      }
      const bool keep = cdf >= threshold;
      sh_keep_[col] = keep ? uint8_t(1) : uint8_t(0);
      if (keep) thread_kept_sum += val_e;
    }

    const float kept_sum = sycl::reduce_over_group(grp, thread_kept_sum, sycl::plus<float>());
    const float normalizer = 1.0f / ((kept_sum > 1e-8f) ? kept_sum : 1e-8f);

    // Vectorized global store.
    using vec_out = vec_t<DType, kVecSize>;
    for (uint32_t i = tid; i < num_vec_elems; i += kWgSize) {
      vec_out out;
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        const uint32_t idx = i * kVecSize + j;
        out[j] = static_cast<DType>(sh_keep_[idx] ? sh_prob_[idx] * normalizer : 0.0f);
      }
      out.store(
          0,
          sycl::multi_ptr<DType, sycl::access::address_space::global_space>(renorm_probs + row_offset + i * kVecSize));
    }
    for (uint32_t col = vec_tail_start + tid; col < vocab_u32; col += kWgSize) {
      renorm_probs[row_offset + col] = static_cast<DType>(sh_keep_[col] ? sh_prob_[col] * normalizer : 0.0f);
    }
  }
};

// Single templated entry point. Picks the exact O(vocab^2) kernel for small
// vocabularies (where the ternary kernel's group-wise tie handling can flip a
// boundary element carrying non-negligible mass) and the fast ternary-search
// kernel otherwise. Both kernels share the same launch geometry and functor
// constructor signature, so the only difference is which functor is submitted.
template <typename TensorDType>
void launch_top_p_renorm_kernel(
    at::Tensor& renorm_probs,
    const at::Tensor& probs,
    const float* maybe_top_p_ptr,
    float top_p_val,
    int batch_size,
    int vocab_size,
    sycl::queue& queue) {
  using KernelDType = typename ToSyclElementTypeP<TensorDType>::type;

  const KernelDType* probs_ptr = reinterpret_cast<const KernelDType*>(probs.data_ptr<TensorDType>());
  auto* renorm_probs_ptr = reinterpret_cast<KernelDType*>(renorm_probs.data_ptr<TensorDType>());

  const int local_size = kTopPRenormWgSize;
  const int global_size = batch_size * local_size;

  // Pick the vectorization width the device prefers for this element size,
  // matching the approach used by per_tensor_quant_fp8, instead of hardcoding 4.
  // Both kernels share the same launch geometry, so only the functor differs:
  // the exact O(vocab^2) kernel for small vocabularies (where the ternary
  // kernel's group-wise tie handling can flip a boundary element carrying
  // non-negligible mass) and the fast ternary-search kernel otherwise.
  const int vec_size = preferred_vector_width(dpcppGetDeviceIdOfCurrentQueue(), sizeof(TensorDType));
  const bool use_exact = vocab_size <= kExactVocabThreshold;

#define LAUNCH_TOP_P(VEC_SIZE)                                                                   \
  do {                                                                                           \
    if (use_exact) {                                                                             \
      sycl_kernel_submit(                                                                        \
          global_size,                                                                           \
          local_size,                                                                            \
          queue,                                                                                 \
          TopPRenormProbsExactSingleCTA<KernelDType, VEC_SIZE>(                                  \
              renorm_probs_ptr, probs_ptr, maybe_top_p_ptr, top_p_val, batch_size, vocab_size)); \
    } else {                                                                                     \
      sycl_kernel_submit(                                                                        \
          global_size,                                                                           \
          local_size,                                                                            \
          queue,                                                                                 \
          TopPRenormProbsSingleCTA<KernelDType, VEC_SIZE>(                                       \
              renorm_probs_ptr, probs_ptr, maybe_top_p_ptr, top_p_val, batch_size, vocab_size)); \
    }                                                                                            \
  } while (0)

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
    at::Tensor& renorm_probs,
    const at::Tensor& probs,
    const std::optional<at::Tensor>& maybe_top_p_arr,
    double top_p_val) {
  CHECK_INPUT(probs);
  CHECK_INPUT(renorm_probs);
  TORCH_CHECK(probs.dim() == 2, "probs must be a 2D tensor [batch_size, vocab_size]");
  TORCH_CHECK(renorm_probs.dim() == 2, "renorm_probs must be a 2D tensor [batch_size, vocab_size]");
  TORCH_CHECK(probs.sizes() == renorm_probs.sizes(), "Input tensors must have the same shape");
  TORCH_CHECK(probs.scalar_type() == renorm_probs.scalar_type(), "Input tensors must have the same dtype");

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

  auto dtype = probs.scalar_type();

#define LAUNCH_TOP_P_RENORM(TENSOR_DTYPE)   \
  launch_top_p_renorm_kernel<TENSOR_DTYPE>( \
      renorm_probs, probs, maybe_top_p_ptr, static_cast<float>(top_p_val), batch_size, vocab_size, queue)

  if (dtype == torch::kFloat32) {
    LAUNCH_TOP_P_RENORM(float);
  } else if (dtype == torch::kHalf) {
    LAUNCH_TOP_P_RENORM(at::Half);
  } else if (dtype == torch::kBFloat16) {
    LAUNCH_TOP_P_RENORM(at::BFloat16);
  } else {
    TORCH_CHECK(false, "Unsupported data type for top_p_renorm_probs");
  }

#undef LAUNCH_TOP_P_RENORM
}
