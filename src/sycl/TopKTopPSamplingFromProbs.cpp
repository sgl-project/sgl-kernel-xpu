#include <ATen/ATen.h>
#include <ATen/xpu/XPUGeneratorImpl.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstdint>
#include <numeric>
#include <optional>
#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"
#include "comm/Random.h"
#include "comm/Sampling.h"

namespace {

constexpr int kTopKTopPMaxRounds = 32;

#define DISPATCH_TOPKTOPP_VEC_SIZE(vec_size, VEC_SIZE_VAR, ...) \
  switch (vec_size) {                                           \
    case 16: {                                                  \
      constexpr uint32_t VEC_SIZE_VAR = 16;                     \
      __VA_ARGS__;                                              \
      break;                                                    \
    }                                                           \
    case 8: {                                                   \
      constexpr uint32_t VEC_SIZE_VAR = 8;                      \
      __VA_ARGS__;                                              \
      break;                                                    \
    }                                                           \
    case 4: {                                                   \
      constexpr uint32_t VEC_SIZE_VAR = 4;                      \
      __VA_ARGS__;                                              \
      break;                                                    \
    }                                                           \
    case 2: {                                                   \
      constexpr uint32_t VEC_SIZE_VAR = 2;                      \
      __VA_ARGS__;                                              \
      break;                                                    \
    }                                                           \
    default: {                                                  \
      constexpr uint32_t VEC_SIZE_VAR = 1;                      \
      __VA_ARGS__;                                              \
      break;                                                    \
    }                                                           \
  }

//----------------- joint top-k / top-p rejection sampling --------------------//
// One work-group processes one request row.

constexpr uint32_t kTopKTopPWgSize = 1024;

template <uint32_t VEC_SIZE, bool DETERMINISTIC>
struct TopKTopPSamplingKernel : public __SYCL_KER_CONFIG_CONVENTION__ {
  static constexpr uint32_t kWgSize = kTopKTopPWgSize;
  static constexpr uint32_t kNumWarps = kWgSize / 32;
  static constexpr int kMaxRounds = kTopKTopPMaxRounds;

  const float* probs;
  int32_t* output;
  const int64_t* maybe_indices;
  const int32_t* maybe_top_k_arr;
  const float* maybe_top_p_arr;
  int top_k_val;
  float top_p_val;
  int batch_size;
  int vocab_size;
  uint64_t philox_seed;
  uint64_t philox_offset;

  sycl::local_accessor<int32_t, 1> sampled_id_;
  sycl::local_accessor<int32_t, 1> last_valid_id_;
  sycl::local_accessor<float, 1> smem_prefix_sum_;

  void sycl_ker_config_convention(sycl::handler& cgh) {
    sampled_id_ = sycl::local_accessor<int32_t, 1>(sycl::range<1>(1), cgh);
    last_valid_id_ = sycl::local_accessor<int32_t, 1>(sycl::range<1>(1), cgh);
    smem_prefix_sum_ = sycl::local_accessor<float, 1>(sycl::range<1>(kNumWarps), cgh);
  }

  TopKTopPSamplingKernel(
      const float* probs,
      int32_t* output,
      const int64_t* maybe_indices,
      const int32_t* maybe_top_k_arr,
      const float* maybe_top_p_arr,
      int top_k_val,
      float top_p_val,
      int batch_size,
      int vocab_size,
      uint64_t philox_seed,
      uint64_t philox_offset)
      : probs(probs),
        output(output),
        maybe_indices(maybe_indices),
        maybe_top_k_arr(maybe_top_k_arr),
        maybe_top_p_arr(maybe_top_p_arr),
        top_k_val(top_k_val),
        top_p_val(top_p_val),
        batch_size(batch_size),
        vocab_size(vocab_size),
        philox_seed(philox_seed),
        philox_offset(philox_offset) {}

  [[sycl::reqd_sub_group_size(32)]]
  void operator()(sycl::nd_item<1> item) const {
    auto grp = item.get_group();
    const uint32_t bx = item.get_group(0);
    if (bx >= static_cast<uint32_t>(batch_size)) return;

    const uint32_t tx = item.get_local_id(0);
    const uint32_t d = static_cast<uint32_t>(vocab_size);
    const int32_t d_int = static_cast<int32_t>(d);
    const uint32_t row_idx = (maybe_indices != nullptr) ? static_cast<uint32_t>(maybe_indices[bx]) : bx;
    const size_t row_offset = static_cast<size_t>(row_idx) * static_cast<size_t>(d);

    const int k = (maybe_top_k_arr != nullptr) ? static_cast<int>(maybe_top_k_arr[bx]) : top_k_val;
    const float p = (maybe_top_p_arr != nullptr) ? maybe_top_p_arr[bx] : top_p_val;

    // vec_size is chosen as gcd(16/sizeof(float), vocab_size) on the host, so
    // VEC_SIZE always divides d exactly: every chunk is a full vector.
    const uint32_t num_chunks = div_up(d, kWgSize * VEC_SIZE);
    using vec_in = vec_t<float, VEC_SIZE>;

    // The pivot bracket stays in fp32: this is a consumer Xe part where fp64 is
    // not full-rate, and the acceptance sums below are fp32 anyway, so widening
    // only the comparisons bought no accuracy. kMaxRounds bounds the loop, and
    // the `!(low < high)` guard catches a bracket that stops narrowing.
    float low = 0.0f, high = 1.0f;
    float q = 1.0f;
    int32_t result_id = 0;

    for (int round = 0; round < kMaxRounds; ++round) {
      if (tx == 0) {
        sampled_id_[0] = d_int;
        last_valid_id_[0] = -1;
      }
      item.barrier(sycl::access::fence_space::local_space);

      const float u = sgl::random::philox_uniform(philox_seed, philox_offset, bx, static_cast<uint32_t>(round)) * q;

      // --- sample one index proportional to the retained (prob > low) mass ---
      auto pred = [low](float x) { return x > low; };
      float aggregate = 0.0f;
      for (uint32_t i = 0; i < num_chunks; ++i) {
        vec_in v(0.0f);
        if ((i * kWgSize + tx) * VEC_SIZE < d) {
          v.load(
              0,
              sycl::multi_ptr<const float, sycl::access::address_space::global_space>(
                  probs + row_offset + (i * kWgSize + tx) * VEC_SIZE));
        }
        sgl::sampling::device_sampling_from_prob<float, VEC_SIZE, kWgSize, DETERMINISTIC>(
            item, i, d, pred, u, v, aggregate, sampled_id_, last_valid_id_, smem_prefix_sum_);
        if (aggregate > u) break;
      }

      int32_t sampled_id = sampled_id_[0];
      const int32_t last_valid = last_valid_id_[0];
      if (sampled_id == d_int) {
        // u fell beyond the retained mass; fall back to the last valid index.
        sampled_id = last_valid;
        if (last_valid == -1) {
          if (tx == 0) output[bx] = 0;
          return;
        }
      }
      result_id = sampled_id;

      // --- joint acceptance test on pivots derived from the candidate ---
      const float pivot_0 = probs[row_offset + sampled_id];
      const float pivot_1 = (pivot_0 + high) * 0.5f;

      // Same vectorized traversal as the sampling pass above: wide loads, fp32
      // compares. Both pivots are tested in one pass over the row.
      float tsum0 = 0.0f, tsum1 = 0.0f;
      int tcnt0 = 0, tcnt1 = 0;
      for (uint32_t i = 0; i < num_chunks; ++i) {
        const uint32_t col_base = (i * kWgSize + tx) * VEC_SIZE;
        vec_in v(0.0f);
        if (col_base < d) {
          v.load(
              0,
              sycl::multi_ptr<const float, sycl::access::address_space::global_space>(probs + row_offset + col_base));
        }
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j) {
          if (col_base + j < d) {
            const float x = v[j];
            if (x > pivot_0) {
              tsum0 += x;
              ++tcnt0;
            }
            if (x > pivot_1) {
              tsum1 += x;
              ++tcnt1;
            }
          }
        }
      }
      const float sum0 = sycl::reduce_over_group(grp, tsum0, sycl::plus<float>());
      const float sum1 = sycl::reduce_over_group(grp, tsum1, sycl::plus<float>());
      const int cnt0 = sycl::reduce_over_group(grp, tcnt0, sycl::plus<int>());
      const int cnt1 = sycl::reduce_over_group(grp, tcnt1, sycl::plus<int>());

      if (cnt0 < k && sum0 < p) {
        // candidate accepted: fewer than k tokens and less than p mass rank above it.
        break;
      } else if (cnt1 < k && sum1 < p) {
        low = pivot_0;
        high = pivot_1;
        q = sum0;
      } else {
        low = pivot_1;
        q = sum1;
      }

      if (!(low < high)) break;
    }

    if (tx == 0) output[bx] = result_id;
  }
};

void launch_top_k_top_p_sampling(
    const float* probs,
    int32_t* output,
    const int64_t* maybe_indices,
    const int32_t* maybe_top_k_arr,
    const float* maybe_top_p_arr,
    int top_k_val,
    float top_p_val,
    int batch_size,
    int vocab_size,
    uint64_t philox_seed,
    uint64_t philox_offset,
    bool deterministic,
    sycl::queue& queue) {
  const int local_size = kTopKTopPWgSize;
  const int global_size = batch_size * local_size;

  const uint32_t vec_size = std::gcd(16 / sizeof(float), static_cast<uint32_t>(vocab_size));

  DISPATCH_TOPKTOPP_VEC_SIZE(vec_size, VEC_SIZE, {
    AT_DISPATCH_BOOL_NO_RETURN(deterministic, DETERMINISTIC, {
      auto kernel = TopKTopPSamplingKernel<VEC_SIZE, DETERMINISTIC>(
          probs,
          output,
          maybe_indices,
          maybe_top_k_arr,
          maybe_top_p_arr,
          top_k_val,
          top_p_val,
          batch_size,
          vocab_size,
          philox_seed,
          philox_offset);
      sycl_kernel_submit(global_size, local_size, queue, kernel);
    });
  });
}

}  // namespace

void top_k_top_p_sampling_from_probs(
    at::Tensor probs,
    at::Tensor output,
    std::optional<at::Tensor> maybe_indices,
    std::optional<at::Tensor> maybe_top_k_arr,
    int64_t top_k_val,
    std::optional<at::Tensor> maybe_top_p_arr,
    double top_p_val,
    bool deterministic,
    std::optional<at::Generator> gen) {
  CHECK_INPUT(probs);
  CHECK_INPUT(output);
  TORCH_CHECK(probs.dim() == 2, "probs must be a 2D tensor [batch_size, vocab_size]");
  TORCH_CHECK(probs.scalar_type() == torch::kFloat32, "probs must be float32");
  TORCH_CHECK(output.dim() == 1, "output must be a 1D tensor [batch_size]");
  TORCH_CHECK(output.scalar_type() == torch::kInt32, "output must be int32");

  const int batch_size = output.size(0);
  const int vocab_size = probs.size(1);

  const int64_t* indices_ptr = nullptr;
  if (maybe_indices.has_value()) {
    CHECK_INPUT((*maybe_indices));
    TORCH_CHECK(maybe_indices->scalar_type() == torch::kInt64, "maybe_indices must be int64");
    TORCH_CHECK(maybe_indices->size(0) == batch_size, "maybe_indices size must match batch_size");
    indices_ptr = maybe_indices->data_ptr<int64_t>();
  } else {
    TORCH_CHECK(
        probs.size(0) == batch_size, "probs.size(0) must match output.size(0) when maybe_indices is not provided");
  }

  const int32_t* top_k_ptr = nullptr;
  if (maybe_top_k_arr.has_value()) {
    CHECK_INPUT((*maybe_top_k_arr));
    TORCH_CHECK(maybe_top_k_arr->dim() == 1, "maybe_top_k_arr must be a 1D tensor");
    TORCH_CHECK(maybe_top_k_arr->scalar_type() == torch::kInt32, "maybe_top_k_arr must be int32");
    TORCH_CHECK(maybe_top_k_arr->size(0) == batch_size, "maybe_top_k_arr size must match batch_size");
    top_k_ptr = maybe_top_k_arr->data_ptr<int32_t>();
  } else {
    TORCH_CHECK(top_k_val > 0 && top_k_val <= vocab_size, "top_k_val must be within (0, vocab_size]");
  }

  const float* top_p_ptr = nullptr;
  if (maybe_top_p_arr.has_value()) {
    CHECK_INPUT((*maybe_top_p_arr));
    TORCH_CHECK(maybe_top_p_arr->dim() == 1, "maybe_top_p_arr must be a 1D tensor");
    TORCH_CHECK(maybe_top_p_arr->scalar_type() == torch::kFloat32, "maybe_top_p_arr must be float32");
    TORCH_CHECK(maybe_top_p_arr->size(0) == batch_size, "maybe_top_p_arr size must match batch_size");
    top_p_ptr = maybe_top_p_arr->data_ptr<float>();
  } else {
    TORCH_CHECK(top_p_val > 0.0 && top_p_val <= 1.0, "top_p_val must be within (0, 1]");
  }

  // Resolve the Philox seed/offset from the (default) XPU generator.
  auto generator = at::get_generator_or_default<at::XPUGeneratorImpl>(gen, at::xpu::detail::getDefaultXPUGenerator());
  uint64_t philox_seed, philox_offset;
  {
    std::lock_guard<std::mutex> lock(generator->mutex_);
    auto philox = generator->philox_engine_inputs(static_cast<uint64_t>(kTopKTopPMaxRounds));
    philox_seed = philox.first;
    philox_offset = philox.second;
  }

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  launch_top_k_top_p_sampling(
      probs.data_ptr<float>(),
      output.data_ptr<int32_t>(),
      indices_ptr,
      top_k_ptr,
      top_p_ptr,
      static_cast<int>(top_k_val),
      static_cast<float>(top_p_val),
      batch_size,
      vocab_size,
      philox_seed,
      philox_offset,
      deterministic,
      queue);
}
