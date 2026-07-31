#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstdint>
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
  const int vec_size = preferred_vector_width(dpcppGetDeviceIdOfCurrentQueue(), sizeof(float));

#define LAUNCH_TOP_P(VEC_SIZE)            \
  sycl_kernel_submit(                     \
      global_size,                        \
      local_size,                         \
      queue,                              \
      TopPRenormProbsSingleCTA<VEC_SIZE>( \
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
