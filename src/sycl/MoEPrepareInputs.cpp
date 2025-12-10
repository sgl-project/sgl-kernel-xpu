#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cmath>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

#include "MemoryAccess.h"
#include "SYCLHelpers.h"
#include "Utils.h"

constexpr int block_size = 128;

struct compute_problem_sizes_sycl_K {
  compute_problem_sizes_sycl_K(
    const int* topk_ids,
    int32_t* problem_sizes1,
    int32_t* problem_sizes2,
    int32_t* atomic_buffer,
    const uint32_t num_experts,
    const uint32_t topk_length,
    const uint32_t n,
    const uint32_t k,
    const uint32_t max_tokens_per_expert)
      : topk_ids_(topk_ids),
        problem_sizes1_(problem_sizes1),
        problem_sizes2_(problem_sizes2),
        atomic_buffer_(atomic_buffer),
        num_experts_(num_experts),
        topk_length_(topk_length),
        n_(n),
        k_(k),
        max_tokens_per_expert_(max_tokens_per_expert) {}

    void operator()(sycl::nd_item<1> item) const {
          int thread_id = item.get_local_linear_id();
          if (thread_id < topk_length_) {
            int expert_id = item.get_group(0);

            int occurrences = 0;
            for (int i = thread_id; i < topk_length_; i += max_tokens_per_expert_) {
              occurrences += (topk_ids_[i] == expert_id);
            }

            sycl::atomic_ref<
              int32_t,
              sycl::memory_order::relaxed,
              sycl::memory_scope::work_group,
              sycl::access::address_space::generic_space
            > atomic_counter(atomic_buffer_[expert_id]);

            atomic_counter.fetch_add(occurrences);

            item.barrier(sycl::access::fence_space::local_space);

            if (thread_id == 0) {
              int final_occurrences = atomic_buffer_[expert_id];
              problem_sizes1_[expert_id * 3] = final_occurrences;
              problem_sizes1_[expert_id * 3 + 1] = static_cast<int32_t>(2 * n_);
              problem_sizes1_[expert_id * 3 + 2] = static_cast<int32_t>(k_);
              problem_sizes2_[expert_id * 3] = final_occurrences;
              problem_sizes2_[expert_id * 3 + 1] = static_cast<int32_t>(k_);
              problem_sizes2_[expert_id * 3 + 2] = static_cast<int32_t>(n_);
            }
          }
    }

    const int* topk_ids_;
    int32_t* problem_sizes1_;
    int32_t* problem_sizes2_;
    int32_t* atomic_buffer_;
    const uint32_t num_experts_;
    const uint32_t topk_length_;
    const uint32_t n_;
    const uint32_t k_;
    const uint32_t max_tokens_per_expert_;
};


void compute_problem_sizes_sycl(
    const int* topk_ids,
    int32_t* problem_sizes1,
    int32_t* problem_sizes2,
    int32_t* atomic_buffer,
    const uint32_t num_experts,
    const uint32_t topk_length,
    const uint32_t n,
    const uint32_t k) {

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  using Kernel = compute_problem_sizes_sycl_K;

  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  uint32_t max_wg_size = dpcppMaxWorkGroupSize(dev_id);
  uint32_t max_tokens_per_expert = static_cast<uint32_t>(sycl::min(max_wg_size, topk_length));

  sycl::range<1> global_range{ num_experts * max_tokens_per_expert };
  sycl::range<1> local_range{ max_tokens_per_expert };

  Kernel task(topk_ids, problem_sizes1, problem_sizes2, atomic_buffer, num_experts, topk_length, n, k, max_tokens_per_expert);

  sycl_kernel_submit(global_range, local_range, queue, task);
  return;
}


struct compute_expert_offsets_sycl_k {
  compute_expert_offsets_sycl_k(
  int32_t* expert_offsets,
  int32_t* atomic_buffer,
  const uint32_t num_experts)
      : expert_offsets_(expert_offsets),
        atomic_buffer_(atomic_buffer),
        num_experts_(num_experts) {}

    void operator()(sycl::nd_item<1> it) const {
      int lid = it.get_local_id(0);
      int x = (lid < num_experts_) ? expert_offsets_[lid] : 0;
      int scanned = exclusive_scan_over_group(it.get_group(), x, sycl::plus<int>());
      if (lid < num_experts_)
        atomic_buffer_[lid] = scanned;
    }

  int32_t* expert_offsets_;
  int32_t* atomic_buffer_;
  const uint32_t num_experts_;
};

void compute_expert_offsets_sycl(
    int32_t* expert_offsets,
    int32_t* atomic_buffer,
    const uint32_t num_experts) {

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  using Kernel = compute_expert_offsets_sycl_k;

  Kernel task(expert_offsets, atomic_buffer, num_experts);

  sycl_kernel_submit(num_experts, num_experts, queue, task);
  return;
}

struct compute_expert_blockscale_offsets_sycl_K {
  compute_expert_blockscale_offsets_sycl_K(
  const int32_t* problem_sizes1,
  int32_t* expert_offsets,
  int32_t* blockscale_offsets,
  int32_t* atomic_buffer,
  const uint32_t num_experts)
      : problem_sizes1_(problem_sizes1),
        expert_offsets_(expert_offsets),
        blockscale_offsets_(blockscale_offsets),
        atomic_buffer_(atomic_buffer),
        num_experts_(num_experts) {}

    void operator()(sycl::nd_item<1> item) const {
      int32_t tot_offset = 0;
      int32_t tot_rounded_offset = 0;
      expert_offsets_[0] = 0;
      blockscale_offsets_[0] = 0;
      for (int i = 0; i < num_experts_; ++i) {
        atomic_buffer_[i] = tot_offset;
        int num_tokens = problem_sizes1_[i * 3];
        int rounded_num_tokens = (num_tokens + (block_size - 1)) / block_size * block_size;
        tot_offset += num_tokens;
        tot_rounded_offset += rounded_num_tokens;
        expert_offsets_[i + 1] = tot_offset;
        blockscale_offsets_[i + 1] = tot_rounded_offset;
      }
    }

  const int32_t* problem_sizes1_;
  int32_t* expert_offsets_;
  int32_t* blockscale_offsets_;
  int32_t* atomic_buffer_;
  const uint32_t num_experts_;
};

void compute_expert_blockscale_offsets_sycl(
  const int32_t* problem_sizes1,
  int32_t* expert_offsets,
  int32_t* blockscale_offsets,
  int32_t* atomic_buffer,
  const int64_t num_experts) {

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  using Kernel = compute_expert_blockscale_offsets_sycl_K;

  Kernel task(problem_sizes1, expert_offsets, blockscale_offsets, atomic_buffer, num_experts);

  sycl_kernel_submit(1, 1, queue, task);
  return;
}


struct compute_arg_sorts_sycl_K {
  compute_arg_sorts_sycl_K(
    const int32_t* topk_ids,
    int32_t* input_permutation,
    int32_t* output_permutation,
    int32_t* atomic_buffer,
    const int32_t topk_length,
    const int32_t topk,
    const int32_t num_experts,
    const int32_t max_tokens_per_expert)
      : topk_ids_(topk_ids),
        input_permutation_(input_permutation),
        output_permutation_(output_permutation),
        atomic_buffer_(atomic_buffer),
        topk_length_(topk_length),
        topk_(topk),
        num_experts_(num_experts),
        max_tokens_per_expert_(max_tokens_per_expert) {}

    void operator()(sycl::nd_item<1> item) const {
      int expert_id = item.get_group(0);

      sycl::atomic_ref<
        int32_t,
        sycl::memory_order::relaxed,
        sycl::memory_scope::work_group,
        sycl::access::address_space::generic_space
      > atomic_counter(atomic_buffer_[expert_id]);

      for (int32_t i = item.get_local_id(0); i < topk_length_; i += max_tokens_per_expert_) {
        if (topk_ids_[i] == expert_id) {
          int32_t start = atomic_counter.fetch_add(1);
          input_permutation_[start] = i / topk_;
          output_permutation_[i] = start;
        }
      }
    }

    const int32_t* topk_ids_;
    int32_t* input_permutation_;
    int32_t* output_permutation_;
    int32_t* atomic_buffer_;
    const uint32_t topk_length_;
    const uint32_t topk_;
    const uint32_t num_experts_;
    const uint32_t max_tokens_per_expert_;
};

void compute_arg_sorts_sycl(
    const int32_t* topk_ids,
    int32_t* input_permutation,
    int32_t* output_permutation,
    int32_t* atomic_buffer,
    const uint32_t topk_length,
    const uint32_t topk,
    const uint32_t num_experts) {


  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  using Kernel = compute_arg_sorts_sycl_K;

  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  uint32_t max_wg_size = dpcppMaxWorkGroupSize(dev_id);
  uint32_t max_tokens_per_expert = static_cast<uint32_t>(sycl::min(max_wg_size, topk_length));

  sycl::range<1> global_range{ num_experts * max_tokens_per_expert };
  sycl::range<1> local_range{ max_tokens_per_expert };

  Kernel task(topk_ids, input_permutation, output_permutation, atomic_buffer, topk_length, topk, num_experts, max_tokens_per_expert);

  sycl_kernel_submit(global_range, local_range, queue, task);
  return;

}

void prepare_moe_input(
    const torch::Tensor& topk_ids,
    torch::Tensor& expert_offsets,
    const std::optional<torch::Tensor>& blockscale_offsets,
    torch::Tensor& problem_sizes1,
    torch::Tensor& problem_sizes2,
    torch::Tensor& input_permutation,
    torch::Tensor& output_permutation,
    const int64_t num_experts,
    const int64_t n,
    const int64_t k) {
  TORCH_CHECK(topk_ids.dtype() == torch::kInt32, "Expected topk_ids to be int32");

  auto options_int32 = torch::TensorOptions().dtype(torch::kInt32).device(topk_ids.device());
  torch::Tensor atomic_buffer = torch::zeros(num_experts + 1, options_int32);

  compute_problem_sizes_sycl(
      static_cast<const int32_t*>(topk_ids.data_ptr()),
      static_cast<int32_t*>(problem_sizes1.data_ptr()),
      static_cast<int32_t*>(problem_sizes2.data_ptr()),
      static_cast<int32_t*>(expert_offsets.data_ptr()),
      num_experts,
      topk_ids.numel(),
      n,
      k);

  if (blockscale_offsets.has_value()) {
    compute_expert_blockscale_offsets_sycl(
        static_cast<const int32_t*>(problem_sizes1.data_ptr()),
        static_cast<int32_t*>(expert_offsets.data_ptr()),
        static_cast<int32_t*>(blockscale_offsets.value().data_ptr()),
        static_cast<int32_t*>(expert_offsets.data_ptr()),
        num_experts);
  } else {
    compute_expert_offsets_sycl(
        static_cast<int32_t*>(expert_offsets.data_ptr()),
        static_cast<int32_t*>(atomic_buffer.data_ptr()),
        num_experts);
  }

  compute_arg_sorts_sycl(
    static_cast<const int32_t*>(topk_ids.data_ptr()),
    static_cast<int32_t*>(input_permutation.data_ptr()),
    static_cast<int32_t*>(output_permutation.data_ptr()),
    static_cast<int32_t*>(atomic_buffer.data_ptr()),
    topk_ids.numel(),
    topk_ids.size(1),
    num_experts);

  return;
}

template <typename T>
struct ShuffleRows {
  ShuffleRows(
      const T* input,
      const int32_t* dst2src_map,
      T* output,
      const uint32_t num_src_rows,
      const uint32_t num_dest_rows,
      const uint32_t num_cols,
      const uint32_t bs_num_cols)
      : input_(input),
        dst2src_map_(dst2src_map),
        output_(output),
        num_src_rows_(num_src_rows),
        num_dest_rows_(num_dest_rows),
        num_cols_(num_cols),
        bs_num_cols_(bs_num_cols) {}

  void operator()(sycl::nd_item<1> item) const {
      int gid = item.get_global_linear_id();
      int tid = item.get_local_linear_id();
    // Leave it to compiler for simd sub-group
    if (gid < num_dest_rows_ * bs_num_cols_) {
      uint32_t dest_token_idx = item.get_group(0);
      uint32_t source_token_idx = dst2src_map_[dest_token_idx];
      for (int i = tid; i < num_cols_; i += bs_num_cols_) {
        auto source_val = input_[source_token_idx * num_cols_ + i];
        output_[dest_token_idx * num_cols_ + i] = source_val;
      }
    }
  }
  const T* input_;
  const int32_t* dst2src_map_;
  T* output_;
  const uint32_t num_src_rows_;
  const uint32_t num_dest_rows_;
  const uint32_t num_cols_;
  const uint32_t bs_num_cols_;
};

template <typename T>
void shuffle_rows_kernel_impl(const torch::Tensor& input_tensor, const torch::Tensor& dst2src_map, torch::Tensor& output_tensor) {
  auto input = reinterpret_cast<T*>(input_tensor.data_ptr());
  auto dst2srcmap = reinterpret_cast<const int32_t*>(dst2src_map.data_ptr());
  auto output = reinterpret_cast<T*>(output_tensor.data_ptr());

  uint32_t num_src_rows = input_tensor.size(0);
  uint32_t num_dest_rows = output_tensor.size(0);
  uint32_t num_cols = input_tensor.size(1);

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  using Kernel = ShuffleRows<T>;

  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  uint32_t max_wg_size = dpcppMaxWorkGroupSize(dev_id);
  uint32_t max_num_cols = static_cast<uint32_t>(sycl::min(max_wg_size, num_cols));

  sycl::range<1> global_range{ num_dest_rows * max_num_cols };
  sycl::range<1> local_range{ max_num_cols };

  Kernel task(input, dst2srcmap, output, num_src_rows, num_dest_rows, num_cols, max_num_cols);

  sycl_kernel_submit(global_range, local_range, queue, task);
  return;

}

void shuffle_rows(const torch::Tensor& input_tensor, const torch::Tensor& dst2src_map, torch::Tensor& output_tensor) {
  TORCH_CHECK(
      input_tensor.scalar_type() == output_tensor.scalar_type(),
      "Input and output tensors must have the same data type");
    SYCL_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16, at::ScalarType::Half, input_tensor.scalar_type(), "shuffle_rows_kernel_impl", [&]() {
        shuffle_rows_kernel_impl<scalar_t>(input_tensor, dst2src_map, output_tensor);
      });
  return;
}

template <typename T, typename T1>
struct ApplyShuffleMulSum {
  ApplyShuffleMulSum(
      const T* input,
      T* output,
      const int32_t* dst2src_map,
      const T1* factors,
      const int32_t topk,
      const int32_t hidden_dim,
      const int32_t bs_hidden_dim)
      : input_(input),
        output_(output),
        dst2src_map_(dst2src_map),
        factors_(factors),
        topk_(topk),
        hidden_dim_(hidden_dim),
        bs_hidden_dim_(bs_hidden_dim) {}

  void operator()(sycl::nd_item<1> item) const {

    int out_tkn_id = item.get_group(0);
    float sum_val = 0;

    for (int i = item.get_local_id(0); i < hidden_dim_; i += bs_hidden_dim_) {
      for (int k = 0; k < topk_; ++k) {
        int src_perm_offset = out_tkn_id * topk_ + k;
        int src_index = dst2src_map_[src_perm_offset];
        float src_val = static_cast<float>(input_[src_index * hidden_dim_ + i]);
        float weight = 0;
        if (factors_ != nullptr) {
          weight = static_cast<float>(factors_[out_tkn_id * topk_ + k]);
        }
        sum_val += weight * src_val;
      }
      output_[out_tkn_id * hidden_dim_ + i] = sum_val;
    }
  }
  const T* input_;
  const int32_t* dst2src_map_;
  T* output_;
  const T1* factors_;
  const int32_t topk_;
  const int32_t hidden_dim_;
  const int32_t bs_hidden_dim_;
};

template <typename T, typename T1>
void apply_shuffle_mul_sum_impl(
      const T* input,
      T* output,
      const int32_t* dst2src_map,
      const T1* factors,
      const uint32_t out_tkns,
      const uint32_t out_hidden_dims,
      const int topk) {

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  using Kernel = ApplyShuffleMulSum<T, T1>;

  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  uint32_t max_wg_size = dpcppMaxWorkGroupSize(dev_id);
  uint32_t max_out_hidden_dims = static_cast<uint32_t>(sycl::min(max_wg_size, out_hidden_dims));

  sycl::range<1> global_range{ out_tkns * max_out_hidden_dims };
  sycl::range<1> local_range{ max_out_hidden_dims };

  Kernel task(input, output, dst2src_map, factors, topk, out_hidden_dims, max_out_hidden_dims);

  sycl_kernel_submit(global_range, local_range, queue, task);
  return;

}

void apply_shuffle_mul_sum(
    const torch::Tensor& input,
    torch::Tensor& output,
    const torch::Tensor& permutation,
    const std::optional<torch::Tensor>& factors) {
  int m = output.size(0);
  int topk = int(permutation.size(0) / m);
  SYCL_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16, at::ScalarType::Half, input.scalar_type(), "apply_shuffle_mul_sum", [&]() {
    using input_t = scalar_t;
    if (factors.has_value()) {
      SYCL_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::BFloat16, at::ScalarType::Half, factors.value().scalar_type(), "factors dispatch", [&]() {
        using factors_t = scalar_t;
        apply_shuffle_mul_sum_impl<input_t, factors_t>(
            reinterpret_cast<input_t*>(input.data_ptr()),
            reinterpret_cast<input_t*>(output.data_ptr()),
            reinterpret_cast<int32_t*>(permutation.data_ptr()),
            reinterpret_cast<factors_t*>(factors->data_ptr()),
            output.size(0),
            output.size(1),
            topk
          );
      });
    } else {
        apply_shuffle_mul_sum_impl<input_t, input_t>(
            reinterpret_cast<input_t*>(input.data_ptr()),
            reinterpret_cast<input_t*>(output.data_ptr()),
            reinterpret_cast<int32_t*>(permutation.data_ptr()),
            nullptr,
            output.size(0),
            output.size(1),
            topk
          );
    }
  });
  return;
}
