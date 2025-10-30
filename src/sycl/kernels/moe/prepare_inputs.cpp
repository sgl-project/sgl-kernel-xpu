#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

#include "MemoryAccess.h"
#include "SYCLHelpers.h"
#include "Utils.h"

constexpr uint64_t THREADS_PER_EXPERT = 512;
constexpr int block_size = 128

void compute_problem_sizes_sycl(
    sycl::queue& q,
    const int* topk_ids,
    int32_t* problem_sizes1,
    int32_t* problem_sizes2,
    int32_t* atomic_buffer,
    const int64_t num_experts,
    const int64_t topk_length,
    const int64_t n,
    const int64_t k) {

  sycl::range<1> global_range{ num_experts * topk_length };
  sycl::range<1> local_range{ topk_length }; 

  // Launch kernel
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
      sycl::nd_range<1>(global_range, local_range),
      [=](sycl::nd_item<1> item) {
        int expert_id = item.get_group(0);
        int occurrences = 0;
        size_t local_id = item.get_local_id(0);
        for (int i = local_id; i < topk_length; i += THREADS_PER_EXPERT) {
          occurrences += (topk_ids[i] == expert_id);
        }

        atomic_ref<
          int32_t,
          sycl::memory_order::relaxed,
          sycl::memory_scope::work_group,
          sycl::access::address_space::generic_space
        > atomic_counter(atomic_buffer[expert_id]);

        atomic_counter.fetch_add(occurrences);

        if (local_id == 0) {
          int final_occurrences = atomic_buffer[expert_id];
          problem_sizes1[expert_id * 3] = final_occurrences;
          problem_sizes1[expert_id * 3 + 1] = static_cast<int32_t>(2 * n);
          problem_sizes1[expert_id * 3 + 2] = static_cast<int32_t>(k);
          problem_sizes2[expert_id * 3] = final_occurrences;
          problem_sizes2[expert_id * 3 + 1] = static_cast<int32_t>(k);
          problem_sizes2[expert_id * 3 + 2] = static_cast<int32_t>(n);
        }          
    });
  });          
}

void compute_expert_offsets_sycl(
    sycl::queue& q,
    const int32_t* problem_sizes1,
    int32_t* expert_offsets,
    int32_t* atomic_buffer,
    const int64_t num_experts) {

  // Launch kernel
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
      sycl::nd_range<1>(1, 1),
      [=](sycl::nd_item<1> item) {
      int32_t tot_offset = 0;
      expert_offsets[0] = 0;
      for (int i = 0; i < num_experts; ++i) {
        atomic_buffer[i] = tot_offset;
        tot_offset += problem_sizes1[i * 3];
        expert_offsets[i + 1] = tot_offset;
      }
    });
  });
}

void compute_expert_blockscale_offsets_sycl(
  sycl::queue& q,
  const int32_t* problem_sizes1,
  int32_t* expert_offsets,
  int32_t* blockscale_offsets,
  int32_t* atomic_buffer,
  const int64_t num_experts) {

  // Launch kernel
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
      sycl::nd_range<1>(1, 1),
      [=](sycl::nd_item<1> item) {
      int32_t tot_offset = 0;
      int32_t tot_rounded_offset = 0;
      expert_offsets[0] = 0;
      blockscale_offsets[0] = 0;
      for (int i = 0; i < num_experts; ++i) {
        atomic_buffer[i] = tot_offset;
        int num_tokens = problem_sizes1[i * 3];
        int rounded_num_tokens = (num_tokens + (block_size - 1)) / block_size * block_size;
        tot_offset += num_tokens;
        tot_rounded_offset += rounded_num_tokens;
        expert_offsets[i + 1] = tot_offset;
        blockscale_offsets[i + 1] = tot_rounded_offset;
      }
    });
  });
}

void compute_arg_sorts_sycl(
    sycl::queue& q,
    const int32_t* topk_ids,
    int32_t* input_permutation,
    int32_t* output_permutation,
    int32_t* atomic_buffer,
    const int64_t topk_length,
    const int64_t topk) {

  sycl::range<1> global_range{ num_experts * topk_length };
  sycl::range<1> local_range{ topk_length }; 

  // Launch kernel
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
      sycl::nd_range<1>(global_range, local_range),
      [=](sycl::nd_item<1> item) {
        int expert_id = item.get_group(0);

        atomic_ref<
          int32_t,
          sycl::memory_order::relaxed,
          sycl::memory_scope::work_group,
          sycl::access::address_space::generic_space
        > atomic_counter(atomic_buffer[expert_id]);

        for (int i = threadIdx.x; i < topk_length; i += THREADS_PER_EXPERT) {
          if (topk_ids[i] == expert_id) {
            int start = atomic_counter.fetch_add(1);
            input_permutation[start] = i / topk;
            output_permutation[i] = start;
          }
        }
    });
  });          
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
  TORCH_CHECK(topk_ids.dtype() == torch::kInt32);
  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  auto options_int32 = torch::TensorOptions().dtype(torch::kInt32).device(topk_ids.device());
  torch::Tensor atomic_buffer = torch::zeros(num_experts, options_int32);

  uint32_t num_threads = static_cast<uint32_t>(min(THREADS_PER_EXPERT, topk_ids.numel()));
  uint32_t num_blocks = static_cast<uint32_t>(num_experts);

  compute_problem_sizes_sycl(
      queue,
      static_cast<const int32_t*>(topk_ids.data_ptr()),
      static_cast<int32_t*>(problem_sizes1.data_ptr()),
      static_cast<int32_t*>(problem_sizes2.data_ptr()),
      static_cast<int32_t*>(atomic_buffer.data_ptr()),
      num_experts,
      topk_ids.numel(),
      n,
      k);

  if (blockscale_offsets.has_value()) {
    compute_expert_blockscale_offsets_sycl(
        static_cast<const int32_t*>(problem_sizes1.data_ptr()),
        static_cast<int32_t*>(expert_offsets.data_ptr()),
        static_cast<int32_t*>(blockscale_offsets.value().data_ptr()),
        static_cast<int32_t*>(atomic_buffer.data_ptr()),
        num_experts);
  } else {
    compute_expert_offsets_sycl(
        queue,
        static_cast<int32_t*>(problem_sizes1),
        static_cast<int32_t*>(expert_offsets),
        static_cast<int32_t*>(atomic_buffer),
        num_experts);
  }

  compute_arg_sorts(
      queue,
      static_cast<const int32_t*>(topk_ids.data_ptr()),
      static_cast<int32_t*>(input_permutation.data_ptr()),
      static_cast<int32_t*>(output_permutation.data_ptr()),
      static_cast<int32_t*>(atomic_buffer.data_ptr()),
      topk_ids.numel(),
      topk_ids.size(1));

  return;
}
