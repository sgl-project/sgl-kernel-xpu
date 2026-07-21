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

template <typename T>
struct ToSyclElementType {
  using type = T;
};

template <>
struct ToSyclElementType<at::Half> {
  using type = sycl::half;
};

template <>
struct ToSyclElementType<at::BFloat16> {
  using type = sycl::ext::oneapi::bfloat16;
};

//----------------- min-p rejection sampling --------------------//
// One work-group processes one request row.

template <typename DType>
struct MinPSamplingKernel : public __SYCL_KER_CONFIG_CONVENTION__ {
  static constexpr uint32_t kWgSize = 1024;

  const DType* probs;
  int32_t* output;
  const int64_t* maybe_indices;
  const float* maybe_min_p_arr;
  float min_p_val;
  int batch_size;
  int vocab_size;
  uint64_t philox_seed;
  uint64_t philox_offset;

  sycl::local_accessor<int32_t, 1> shared_ids_;

  void sycl_ker_config_convention(sycl::handler& cgh) {
    shared_ids_ = sycl::local_accessor<int32_t, 1>(sycl::range<1>(2), cgh);
  }

  MinPSamplingKernel(
      const DType* probs,
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

    constexpr uint32_t kVecSize = 4;
    using vec_in = vec_t<DType, kVecSize>;
    const uint32_t num_vec_elems = d / kVecSize;
    const uint32_t vec_tail_start = num_vec_elems * kVecSize;

    float thread_max = -std::numeric_limits<float>::infinity();
    for (uint32_t i = tx; i < num_vec_elems; i += kWgSize) {
      vec_in v;
      v.load(
          0,
          sycl::multi_ptr<const DType, sycl::access::address_space::global_space>(probs + row_offset + i * kVecSize));
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        thread_max = sycl::max(thread_max, static_cast<float>(v[j]));
      }
    }
    for (uint32_t col = vec_tail_start + tx; col < d; col += kWgSize) {
      thread_max = sycl::max(thread_max, static_cast<float>(probs[row_offset + col]));
    }
    const float row_max = sycl::reduce_over_group(grp, thread_max, sycl::maximum<float>());
    const float pivot = p * row_max;

    float thread_sum = 0.0f;
    const uint32_t n_valid = (tx < d) ? ((d - tx - 1) / kWgSize + 1) : 0;
    uint32_t i = 0;
    for (; i + 4 <= n_valid; i += 4) {
      const float a = static_cast<float>(probs[row_offset + (i + 0) * kWgSize + tx]);
      const float b = static_cast<float>(probs[row_offset + (i + 1) * kWgSize + tx]);
      const float c = static_cast<float>(probs[row_offset + (i + 2) * kWgSize + tx]);
      const float e = static_cast<float>(probs[row_offset + (i + 3) * kWgSize + tx]);
      if (a >= pivot) thread_sum += a;
      if (b >= pivot) thread_sum += b;
      if (c >= pivot) thread_sum += c;
      if (e >= pivot) thread_sum += e;
    }
    for (; i < n_valid; ++i) {
      const float x = static_cast<float>(probs[row_offset + i * kWgSize + tx]);
      if (x >= pivot) thread_sum += x;
    }
    const float q = sycl::reduce_over_group(grp, thread_sum, sycl::plus<float>());

    if (tx == 0) {
      shared_ids_[0] = static_cast<int32_t>(d);
      shared_ids_[1] = -1;
    }
    item.barrier(sycl::access::fence_space::local_space);

    const float u = sgl::random::philox_uniform(philox_seed, philox_offset, bx, /*round=*/0) * q;

    float aggregate = 0.0f;
    for (uint32_t i = 0; i < num_chunks; ++i) {
      const uint32_t col = i * kWgSize + tx;
      const bool inb = col < d;
      const float x = inb ? static_cast<float>(probs[row_offset + col]) : 0.0f;
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
        sampled_id = shared_ids_[1];
      }
      output[bx] = (sampled_id == -1) ? 0 : sampled_id;
    }
  }
};

template <typename TensorDType>
void launch_min_p_sampling(
    at::Tensor probs,
    int32_t* output,
    const int64_t* maybe_indices,
    const float* maybe_min_p_arr,
    float min_p_val,
    int batch_size,
    int vocab_size,
    uint64_t philox_seed,
    uint64_t philox_offset,
    sycl::queue& queue) {
  using KernelDType = typename ToSyclElementType<TensorDType>::type;

  const KernelDType* probs_ptr = reinterpret_cast<const KernelDType*>(probs.data_ptr<TensorDType>());

  const int local_size = 1024;
  const int global_size = batch_size * local_size;

  auto kernel = MinPSamplingKernel<KernelDType>(
      probs_ptr, output, maybe_indices, maybe_min_p_arr, min_p_val, batch_size, vocab_size, philox_seed, philox_offset);

  sycl_kernel_submit(global_size, local_size, queue, kernel);
}

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
  TORCH_CHECK(
      probs.scalar_type() == torch::kFloat32 || probs.scalar_type() == torch::kHalf ||
          probs.scalar_type() == torch::kBFloat16,
      "probs must be float32, float16, or bfloat16");
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

  auto dtype = probs.scalar_type();
  if (dtype == torch::kFloat32) {
    launch_min_p_sampling<float>(
        probs,
        output.data_ptr<int32_t>(),
        indices_ptr,
        min_p_ptr,
        static_cast<float>(min_p_val),
        batch_size,
        vocab_size,
        philox_seed,
        philox_offset,
        queue);
  } else if (dtype == torch::kHalf) {
    launch_min_p_sampling<at::Half>(
        probs,
        output.data_ptr<int32_t>(),
        indices_ptr,
        min_p_ptr,
        static_cast<float>(min_p_val),
        batch_size,
        vocab_size,
        philox_seed,
        philox_offset,
        queue);
  } else if (dtype == torch::kBFloat16) {
    launch_min_p_sampling<at::BFloat16>(
        probs,
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
}
