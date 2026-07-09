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

//----------------- single-cta kernel implementation --------------------//
// One work-group processes one row. The pivot threshold that defines the
// top-p nucleus is found via a ternary search on f(x) = sum(probs[probs > x]),
// which is non-increasing. This mirrors the flashinfer TopPRenormProb kernel.

template <typename DType>
struct TopPRenormProbsSingleCTA {
  static constexpr uint32_t kWgSize = 1024;
  static constexpr uint32_t kVecSize = 4;

  const DType* probs;
  DType* renorm_probs;
  const float* maybe_top_p_arr;
  float top_p_val;
  int batch_size;
  int vocab_size;

  TopPRenormProbsSingleCTA(
      const DType* probs,
      DType* renorm_probs,
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
            sycl::multi_ptr<const DType, sycl::access::address_space::global_space>(
                probs + row_offset + i * kVecSize));
#pragma unroll
        for (uint32_t j = 0; j < kVecSize; ++j) {
          thread_sum += static_cast<float>(v[j]);
        }
      }
      for (uint32_t col = vec_tail_start + tid; col < vocab_u32; col += kWgSize) {
        thread_sum += static_cast<float>(probs[row_offset + col]);
      }

      const float row_sum = sycl::reduce_over_group(grp, thread_sum, sycl::plus<float>());
      const float denom = (row_sum <= 1e-8f) ? 1.0f : row_sum;
      const float normalizer = 1.0f / denom;

      for (uint32_t i = tid; i < num_vec_elems; i += kWgSize) {
        vec_io v;
        v.load(
            0,
            sycl::multi_ptr<const DType, sycl::access::address_space::global_space>(
                probs + row_offset + i * kVecSize));
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
        renorm_probs[row_offset + col] =
            static_cast<DType>(static_cast<float>(probs[row_offset + col]) * normalizer);
      }
      return;
    }

    // Compute the maximum probability in the row.
    float thread_max = 0.0f;
    for (uint32_t i = tid; i < num_vec_elems; i += kWgSize) {
      vec_io v;
      v.load(
          0,
          sycl::multi_ptr<const DType, sycl::access::address_space::global_space>(
              probs + row_offset + i * kVecSize));
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

    // Ternary search for the pivot threshold `low` such that keeping probs > low
    // yields cumulative mass >= p while being the minimal such nucleus.
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
            sycl::multi_ptr<const DType, sycl::access::address_space::global_space>(
                probs + row_offset + i * kVecSize));
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
          sycl::multi_ptr<const DType, sycl::access::address_space::global_space>(
              probs + row_offset + i * kVecSize));
      vec_io out;
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        const float val = static_cast<float>(v[j]);
        out[j] = static_cast<DType>((val > low) ? val * normalizer : 0.0f);
      }
      out.store(
          0,
          sycl::multi_ptr<DType, sycl::access::address_space::global_space>(
              renorm_probs + row_offset + i * kVecSize));
    }
    for (uint32_t col = vec_tail_start + tid; col < vocab_u32; col += kWgSize) {
      const float val = static_cast<float>(probs[row_offset + col]);
      renorm_probs[row_offset + col] = static_cast<DType>((val > low) ? val * normalizer : 0.0f);
    }
  }
};

template <typename TensorDType>
void launch_top_p_renorm_kernel(
    at::Tensor probs,
    at::Tensor renorm_probs,
    const float* maybe_top_p_ptr,
    float top_p_val,
    int batch_size,
    int vocab_size,
    sycl::queue& queue) {
  using KernelDType = typename ToSyclElementTypeP<TensorDType>::type;

  const KernelDType* probs_ptr = reinterpret_cast<const KernelDType*>(probs.data_ptr<TensorDType>());
  auto* renorm_probs_ptr = reinterpret_cast<KernelDType*>(renorm_probs.data_ptr<TensorDType>());

  const int local_size = 1024;
  const int global_size = batch_size * local_size;

  auto kernel = TopPRenormProbsSingleCTA<KernelDType>(
      probs_ptr, renorm_probs_ptr, maybe_top_p_ptr, top_p_val, batch_size, vocab_size);

  sycl_kernel_submit(global_size, local_size, queue, kernel);
}

}  // namespace

void top_p_renorm_probs(
    at::Tensor probs, at::Tensor renorm_probs, std::optional<at::Tensor> maybe_top_p_arr, double top_p_val) {
  CHECK_INPUT(probs);
  CHECK_INPUT(renorm_probs);
  TORCH_CHECK(probs.dim() == 2, "probs must be a 2D tensor [batch_size, vocab_size]");
  TORCH_CHECK(renorm_probs.dim() == 2, "renorm_probs must be a 2D tensor [batch_size, vocab_size]");
  TORCH_CHECK(probs.sizes() == renorm_probs.sizes(), "Input tensors must have the same shape");
  TORCH_CHECK(probs.scalar_type() == renorm_probs.scalar_type(), "Input tensors must have the same dtype");
  TORCH_CHECK(
      probs.scalar_type() == torch::kFloat32 || probs.scalar_type() == torch::kHalf ||
          probs.scalar_type() == torch::kBFloat16,
      "Input tensors must be float32, float16, or bfloat16");

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

  if (dtype == torch::kFloat32) {
    launch_top_p_renorm_kernel<float>(
        probs, renorm_probs, maybe_top_p_ptr, static_cast<float>(top_p_val), batch_size, vocab_size, queue);
  } else if (dtype == torch::kHalf) {
    launch_top_p_renorm_kernel<at::Half>(
        probs, renorm_probs, maybe_top_p_ptr, static_cast<float>(top_p_val), batch_size, vocab_size, queue);
  } else if (dtype == torch::kBFloat16) {
    launch_top_p_renorm_kernel<at::BFloat16>(
        probs, renorm_probs, maybe_top_p_ptr, static_cast<float>(top_p_val), batch_size, vocab_size, queue);
  }
}
