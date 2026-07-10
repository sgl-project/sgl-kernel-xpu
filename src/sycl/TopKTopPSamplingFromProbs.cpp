#include <ATen/ATen.h>
#include <ATen/xpu/XPUGeneratorImpl.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstdint>
#include <optional>
#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"
#include "comm/Random.h"

namespace {

//----------------- joint top-k / top-p rejection sampling --------------------//
// One work-group processes one request row. Each round: draw u, sample a
// candidate index proportional to the retained probs (> low) via an inclusive
// scan, then test whether that candidate lies inside both the top-k and top-p
// sets; if not, shrink the [low, high] pivot bracket and retry. Mirrors the
// flashinfer TopKTopPSamplingFromProbKernel (joint filter order).

struct TopKTopPSamplingKernel : public __SYCL_KER_CONFIG_CONVENTION__ {
  static constexpr uint32_t kWgSize = 1024;
  static constexpr int kMaxRounds = 32;

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

  sycl::local_accessor<int32_t, 1> shared_ids_;  // [0]=sampled_id, [1]=last_valid_id

  void sycl_ker_config_convention(sycl::handler& cgh) {
    shared_ids_ = sycl::local_accessor<int32_t, 1>(sycl::range<1>(2), cgh);
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

    const int k = (maybe_top_k_arr != nullptr) ? static_cast<int>(maybe_top_k_arr[row_idx]) : top_k_val;
    const float p = (maybe_top_p_arr != nullptr) ? maybe_top_p_arr[row_idx] : top_p_val;

    const uint32_t num_chunks = (d + kWgSize - 1) / kWgSize;

    double low = 0.0, high = 1.0;
    float q = 1.0f;
    int32_t result_id = 0;

    for (int round = 0; round < kMaxRounds; ++round) {
      if (tx == 0) {
        shared_ids_[0] = d_int;  // sampled_id sentinel
        shared_ids_[1] = -1;     // last_valid_id
      }
      item.barrier(sycl::access::fence_space::local_space);

      const float u = sgl::random::philox_uniform(philox_seed, philox_offset, bx, static_cast<uint32_t>(round)) * q;

      // --- sample one index proportional to the retained (prob > low) mass ---
      float aggregate = 0.0f;
      for (uint32_t i = 0; i < num_chunks; ++i) {
        const uint32_t col = i * kWgSize + tx;
        const bool inb = col < d;
        const float x = inb ? probs[row_offset + col] : 0.0f;
        const bool keep = inb && (static_cast<double>(x) > low);
        const float pgt = keep ? x : 0.0f;

        const float block_total = sycl::reduce_over_group(grp, pgt, sycl::plus<float>());

        if (aggregate + block_total > u) {
          const float excl = sycl::exclusive_scan_over_group(grp, pgt, sycl::plus<float>());
          const float cdf = excl + pgt;
          if (keep && (aggregate + cdf > u)) {
            sycl::atomic_ref<
                int32_t,
                sycl::memory_order::relaxed,
                sycl::memory_scope::work_group,
                sycl::access::address_space::local_space>(shared_ids_[0])
                .fetch_min(static_cast<int32_t>(col));
          }
        }
        if (keep) {
          sycl::atomic_ref<
              int32_t,
              sycl::memory_order::relaxed,
              sycl::memory_scope::work_group,
              sycl::access::address_space::local_space>(shared_ids_[1])
              .fetch_max(static_cast<int32_t>(col));
        }

        aggregate += block_total;
        if (aggregate > u) break;
      }
      item.barrier(sycl::access::fence_space::local_space);

      int32_t sampled_id = shared_ids_[0];
      const int32_t last_valid = shared_ids_[1];
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
      const double pivot_0 = static_cast<double>(probs[row_offset + sampled_id]);
      const double pivot_1 = (pivot_0 + high) * 0.5;

      float tsum0 = 0.0f, tsum1 = 0.0f;
      int tcnt0 = 0, tcnt1 = 0;
      for (uint32_t i = 0; i < num_chunks; ++i) {
        const uint32_t col = i * kWgSize + tx;
        if (col < d) {
          const double x = static_cast<double>(probs[row_offset + col]);
          if (x > pivot_0) {
            tsum0 += static_cast<float>(x);
            ++tcnt0;
          }
          if (x > pivot_1) {
            tsum1 += static_cast<float>(x);
            ++tcnt1;
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
    sycl::queue& queue) {
  const int local_size = 1024;
  const int global_size = batch_size * local_size;

  auto kernel = TopKTopPSamplingKernel(
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
}

}  // namespace

void top_k_top_p_sampling_from_probs(
    at::Tensor probs,
    at::Tensor output,
    std::optional<at::Tensor> maybe_indices,
    std::optional<at::Tensor> maybe_top_k_arr,
    double top_k_val,
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
  }

  const int32_t* top_k_ptr = nullptr;
  if (maybe_top_k_arr.has_value()) {
    CHECK_INPUT((*maybe_top_k_arr));
    TORCH_CHECK(maybe_top_k_arr->dim() == 1, "maybe_top_k_arr must be a 1D tensor");
    TORCH_CHECK(maybe_top_k_arr->scalar_type() == torch::kInt32, "maybe_top_k_arr must be int32");
    top_k_ptr = maybe_top_k_arr->data_ptr<int32_t>();
  }

  const float* top_p_ptr = nullptr;
  if (maybe_top_p_arr.has_value()) {
    CHECK_INPUT((*maybe_top_p_arr));
    TORCH_CHECK(maybe_top_p_arr->dim() == 1, "maybe_top_p_arr must be a 1D tensor");
    TORCH_CHECK(maybe_top_p_arr->scalar_type() == torch::kFloat32, "maybe_top_p_arr must be float32");
    top_p_ptr = maybe_top_p_arr->data_ptr<float>();
  }

  // Resolve the Philox seed/offset from the (default) XPU generator.
  auto generator = at::get_generator_or_default<at::XPUGeneratorImpl>(gen, at::xpu::detail::getDefaultXPUGenerator());
  uint64_t philox_seed, philox_offset;
  {
    std::lock_guard<std::mutex> lock(generator->mutex_);
    auto philox = generator->philox_engine_inputs(static_cast<uint64_t>(TopKTopPSamplingKernel::kMaxRounds));
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
      queue);
}
