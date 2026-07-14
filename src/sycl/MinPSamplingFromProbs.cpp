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

//----------------- min-p rejection sampling --------------------//
// One work-group processes one request row. Threshold = min_p * row_max, no
// pivot search rounds needed (unlike top-k/top-p, min_p's cutoff is known in
// closed form). Draw u, then walk the row to find the index where the
// cumulative sum of retained (>= pivot) probs first crosses u. Mirrors the
// flashinfer MinPSamplingFromProbKernel.

struct MinPSamplingKernel : public __SYCL_KER_CONFIG_CONVENTION__ {
  static constexpr uint32_t kWgSize = 1024;

  const float* probs;
  int32_t* output;
  const int64_t* maybe_indices;
  const float* maybe_min_p_arr;
  float min_p_val;
  int batch_size;
  int vocab_size;
  uint64_t philox_seed;
  uint64_t philox_offset;

  sycl::local_accessor<int32_t, 1> shared_ids_;  // [0]=sampled_id, [1]=last_valid_id
  sycl::local_accessor<float, 1> shared_scalars_;  // [0]=pivot, [1]=q (normalizing mass)

  void sycl_ker_config_convention(sycl::handler& cgh) {
    shared_ids_ = sycl::local_accessor<int32_t, 1>(sycl::range<1>(2), cgh);
    shared_scalars_ = sycl::local_accessor<float, 1>(sycl::range<1>(2), cgh);
  }

  MinPSamplingKernel(
      const float* probs,
      int32_t* output,
      const int64_t* maybe_indices,
      const float* maybe_min_p_arr,
      float min_p_val,
      int batch_size,
      int vocab_size,
      uint64_t philox_seed,
      uint64_t philox_offset)
      : probs(probs),
        output(output),
        maybe_indices(maybe_indices),
        maybe_min_p_arr(maybe_min_p_arr),
        min_p_val(min_p_val),
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
    const uint32_t row_idx = (maybe_indices != nullptr) ? static_cast<uint32_t>(maybe_indices[bx]) : bx;
    const size_t row_offset = static_cast<size_t>(row_idx) * static_cast<size_t>(d);

    const float p = (maybe_min_p_arr != nullptr) ? maybe_min_p_arr[row_idx] : min_p_val;
    const uint32_t num_chunks = (d + kWgSize - 1) / kWgSize;

    // --- pass 1: row max ---
    float thread_max = -std::numeric_limits<float>::infinity();
    for (uint32_t i = 0; i < num_chunks; ++i) {
      const uint32_t col = i * kWgSize + tx;
      if (col < d) {
        thread_max = sycl::max(thread_max, probs[row_offset + col]);
      }
    }
    const float row_max = sycl::reduce_over_group(grp, thread_max, sycl::maximum<float>());
    const float pivot = p * row_max;

    if (tx == 0) shared_scalars_[0] = pivot;
    item.barrier(sycl::access::fence_space::local_space);

    // --- pass 2: normalizing mass q over retained (>= pivot) entries ---
    float thread_sum = 0.0f;
    for (uint32_t i = 0; i < num_chunks; ++i) {
      const uint32_t col = i * kWgSize + tx;
      if (col < d) {
        const float x = probs[row_offset + col];
        if (x >= pivot) thread_sum += x;
      }
    }
    const float q = sycl::reduce_over_group(grp, thread_sum, sycl::plus<float>());

    if (tx == 0) {
      shared_ids_[0] = static_cast<int32_t>(d);  // sampled_id sentinel
      shared_ids_[1] = -1;                       // last_valid_id
    }
    item.barrier(sycl::access::fence_space::local_space);

    const float u = sgl::random::philox_uniform(philox_seed, philox_offset, bx, /*round=*/0) * q;

    // --- pass 3: locate the crossing index ---
    float aggregate = 0.0f;
    for (uint32_t i = 0; i < num_chunks; ++i) {
      const uint32_t col = i * kWgSize + tx;
      const bool inb = col < d;
      const float x = inb ? probs[row_offset + col] : 0.0f;
      const bool keep = inb && (x >= pivot);
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

    if (tx == 0) {
      int32_t sampled_id = shared_ids_[0];
      if (sampled_id == static_cast<int32_t>(d)) {
        // u fell beyond the retained mass (floating-point rounding); fall
        // back to the last retained index.
        sampled_id = shared_ids_[1];
      }
      output[bx] = (sampled_id == -1) ? 0 : sampled_id;
    }
  }
};

void launch_min_p_sampling(
    const float* probs,
    int32_t* output,
    const int64_t* maybe_indices,
    const float* maybe_min_p_arr,
    float min_p_val,
    int batch_size,
    int vocab_size,
    uint64_t philox_seed,
    uint64_t philox_offset,
    sycl::queue& queue) {
  const int local_size = 1024;
  const int global_size = batch_size * local_size;

  auto kernel = MinPSamplingKernel(
      probs,
      output,
      maybe_indices,
      maybe_min_p_arr,
      min_p_val,
      batch_size,
      vocab_size,
      philox_seed,
      philox_offset);

  sycl_kernel_submit(global_size, local_size, queue, kernel);
}

}  // namespace

void min_p_sampling_from_probs(
    at::Tensor probs,
    at::Tensor output,
    std::optional<at::Tensor> maybe_indices,
    std::optional<at::Tensor> maybe_min_p_arr,
    double min_p_val,
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

  const float* min_p_ptr = nullptr;
  if (maybe_min_p_arr.has_value()) {
    CHECK_INPUT((*maybe_min_p_arr));
    TORCH_CHECK(maybe_min_p_arr->dim() == 1, "maybe_min_p_arr must be a 1D tensor");
    TORCH_CHECK(maybe_min_p_arr->scalar_type() == torch::kFloat32, "maybe_min_p_arr must be float32");
    TORCH_CHECK(maybe_min_p_arr->size(0) == batch_size, "maybe_min_p_arr size must match batch_size");
    min_p_ptr = maybe_min_p_arr->data_ptr<float>();
  }

  // Resolve the Philox seed/offset from the (default) XPU generator. Only one
  // round is consumed per row since min_p needs no pivot-search rounds.
  auto generator = at::get_generator_or_default<at::XPUGeneratorImpl>(gen, at::xpu::detail::getDefaultXPUGenerator());
  uint64_t philox_seed, philox_offset;
  {
    std::lock_guard<std::mutex> lock(generator->mutex_);
    auto philox = generator->philox_engine_inputs(1);
    philox_seed = philox.first;
    philox_offset = philox.second;
  }

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  launch_min_p_sampling(
      probs.data_ptr<float>(),
      output.data_ptr<int32_t>(),
      indices_ptr,
      min_p_ptr,
      static_cast<float>(min_p_val),
      batch_size,
      vocab_size,
      philox_seed,
      philox_offset,
      queue);
}
