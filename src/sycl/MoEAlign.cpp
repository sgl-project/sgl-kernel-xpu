#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"

#define VEC_SIZE 4
static constexpr int sub_group_size = 32;

using Vec = sycl::int4;

// Utility function: atomic add for SYCL
template <typename T>
T atomic_add_sycl(
    T* ptr, T value, sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device> atomic_ref) {
  return atomic_ref.fetch_add(value);
}

// Utility: next power of 2
inline size_t next_pow2(size_t n) {
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n |= n >> 32;
  return n + 1;
}

#define CEILDIV(x, y) ((x + y - 1) / y)

template <typename scalar_t>
struct CountAndSortExpertTokensFunctor {
  CountAndSortExpertTokensFunctor(
      const scalar_t* topk_ids, int32_t* sorted_token_ids, int32_t* cumsum_buffer, size_t numel)
      : topk_ids(topk_ids), sorted_token_ids(sorted_token_ids), cumsum_buffer(cumsum_buffer), numel(numel) {}

  [[sycl::reqd_sub_group_size(sub_group_size)]] void operator()(sycl::nd_item<1> item) const {
    const size_t tid = item.get_global_id(0);
    const size_t stride = item.get_global_range(0);

    for (size_t i = tid; i < numel; i += stride) {
      int32_t expert_id = topk_ids[i] + 1;
      sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> atomicRef(
          cumsum_buffer[expert_id]);
      int32_t rank_post_pad = atomicRef.fetch_add(1);
      sorted_token_ids[rank_post_pad] = i;
    }
  }

  const scalar_t* topk_ids;
  int32_t* sorted_token_ids;
  int32_t* cumsum_buffer;
  size_t numel;
};

template <typename scalar_t>
struct MOEAlignBlockSizeFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  MOEAlignBlockSizeFunctor(
      const scalar_t* topk_ids,
      int32_t* sorted_token_ids,
      int32_t* expert_ids,
      int32_t* total_tokens_post_pad,
      int32_t num_experts,
      int32_t block_size,
      size_t numel,
      int32_t* cumsum,
      bool pad_sorted_token_ids,
      const int32_t scan_size)
      : topk_ids(topk_ids),
        sorted_token_ids(sorted_token_ids),
        expert_ids(expert_ids),
        total_tokens_post_pad(total_tokens_post_pad),
        num_experts(num_experts),
        block_size(block_size),
        numel(numel),
        cumsum(cumsum),
        pad_sorted_token_ids(pad_sorted_token_ids),
        scan_size(scan_size) {}

  [[sycl::reqd_sub_group_size(sub_group_size)]] void operator()(sycl::nd_item<1> item) const {
    const size_t tid = item.get_local_id(0);
    const size_t stride = item.get_local_range(0);

    int32_t* shared_counts = (int32_t*)(slm_.template get_multi_ptr<sycl::access::decorated::no>().get());
    int32_t* prefix = shared_counts + num_experts;
    int32_t* scan_buf = prefix + num_experts + 1;
    int32_t* s_total_tokens_post_pad =
        (int32_t*)(total_token_.template get_multi_ptr<sycl::access::decorated::no>().get());
    *s_total_tokens_post_pad = 0;

    if (tid < num_experts) {
      shared_counts[tid] = 0;
    }
    item.barrier(sycl::access::fence_space::local_space);

    for (size_t i = tid; i < numel; i += stride) {
      int expert_id = topk_ids[i] + 1;
      sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group> atomicRef(
          shared_counts[expert_id]);
      atomicRef.fetch_add(1);
    }

    item.barrier(sycl::access::fence_space::local_space);

    int32_t padded_count = 0;
    if (tid < num_experts) {
      int32_t count = shared_counts[tid];
      padded_count = (count + block_size - 1) / block_size * block_size;
      scan_buf[tid] = padded_count;
    }

    // Blelloch scan
    if (tid >= num_experts && tid < scan_size) {
      scan_buf[tid] = 0;
    }
    item.barrier(sycl::access::fence_space::local_space);

    int offset = 1;
    for (int d = scan_size >> 1; d > 0; d >>= 1) {
      if (tid < d) {
        int ai = offset * (2 * tid + 1) - 1;
        int bi = offset * (2 * tid + 2) - 1;
        scan_buf[bi] += scan_buf[ai];
      }
      offset <<= 1;
      item.barrier(sycl::access::fence_space::local_space);
    }

    // down-sweep
    if (tid == 0) {
      prefix[num_experts] = scan_buf[scan_size - 1];
      scan_buf[scan_size - 1] = 0;
    }
    item.barrier(sycl::access::fence_space::local_space);

    for (int d = 1; d < scan_size; d <<= 1) {
      offset >>= 1;
      if (tid < d) {
        int ai = offset * (2 * tid + 1) - 1;
        int bi = offset * (2 * tid + 2) - 1;
        if (bi < scan_size) {
          int temp = scan_buf[ai];
          scan_buf[ai] = scan_buf[bi];
          scan_buf[bi] += temp;
        }
      }
      item.barrier(sycl::access::fence_space::local_space);
    }

    if (tid < num_experts) {
      prefix[tid] = scan_buf[tid];
    }

    if (tid == 0) {
      *s_total_tokens_post_pad = prefix[num_experts];
      *total_tokens_post_pad = *s_total_tokens_post_pad;
    }
    item.barrier(sycl::access::fence_space::local_space);

    // Write cumsum
    if (tid <= num_experts) {
      cumsum[tid] = prefix[tid];
    }

    // Fill expert_ids
    const int32_t num_blocks = *s_total_tokens_post_pad / block_size;
    for (int32_t i = tid; i < num_blocks; i += stride) {
      int32_t block_start = i * block_size;
      int left = 0, right = num_experts;
      while (left < right) {
        int mid = (left + right) >> 1;
        if (prefix[mid] <= block_start) {
          left = mid + 1;
        } else {
          right = mid;
        }
      }
      expert_ids[i] = left - 2;
    }

    if (pad_sorted_token_ids) {
      Vec fill_vec{(int)numel, (int)numel, (int)numel, (int)numel};
      int32_t total_vecs = (*s_total_tokens_post_pad + VEC_SIZE - 1) / VEC_SIZE;
      Vec* out_ptr = reinterpret_cast<Vec*>(sorted_token_ids);
      for (int32_t i = tid; i < total_vecs; i += stride) {
        out_ptr[i] = fill_vec;
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    const size_t scan_size = next_pow2(num_experts);
    const size_t shared_mem_size = num_experts + (num_experts + 1) + scan_size + sub_group_size;
    slm_ = sycl::local_accessor<int32_t>(shared_mem_size, cgh);
    total_token_ = sycl::local_accessor<int32_t>(1, cgh);
  }

  const scalar_t* topk_ids;
  int32_t* sorted_token_ids;
  int32_t* expert_ids;
  int32_t* total_tokens_post_pad;
  int32_t num_experts;
  int32_t block_size;
  size_t numel;
  int32_t* cumsum;
  bool pad_sorted_token_ids;
  const int32_t scan_size;
  sycl::local_accessor<int32_t> slm_;
  sycl::local_accessor<int32_t> total_token_;
  // int32_t* shared_counts;
  // int32_t* prefix;
  // int32_t* scan_buf;
  // int32_t* s_total_tokens_post_pad;
};

template <typename scalar_t>
struct MOEAlignBlockSizeSmallBatchExpertFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  MOEAlignBlockSizeSmallBatchExpertFunctor(
      const scalar_t* topk_ids,
      int32_t* sorted_token_ids,
      int32_t* expert_ids,
      int32_t* total_tokens_post_pad,
      int32_t num_experts,
      int32_t block_size,
      size_t numel,
      bool pad_sorted_token_ids)
      : topk_ids(topk_ids),
        sorted_token_ids(sorted_token_ids),
        expert_ids(expert_ids),
        total_tokens_post_pad(total_tokens_post_pad),
        num_experts(num_experts),
        block_size(block_size),
        numel(numel),
        pad_sorted_token_ids(pad_sorted_token_ids) {}

  [[sycl::reqd_sub_group_size(sub_group_size)]] void operator()(sycl::nd_item<1> item) const {
    const size_t tid = item.get_local_id(0);
    const size_t stride = item.get_local_range(0);
    const size_t block_dim = item.get_local_range(0);

    int32_t* shared_mem = (int32_t*)(slm_.template get_multi_ptr<sycl::access::decorated::no>().get());
    int32_t* cumsum = shared_mem;
    int32_t* tokens_cnts = shared_mem + num_experts + 1;

    for (int i = 0; i < num_experts; ++i) {
      tokens_cnts[num_experts + i] = 0;
    }
    item.barrier(sycl::access::fence_space::local_space);

    for (size_t i = tid; i < numel; i += stride) {
      ++tokens_cnts[(tid + 1) * num_experts + topk_ids[i] + 1];
    }
    item.barrier(sycl::access::fence_space::local_space);

    if (tid < num_experts) {
      tokens_cnts[tid] = 0;
      for (int i = 1; i <= block_dim; ++i) {
        tokens_cnts[i * num_experts + tid] += tokens_cnts[(i - 1) * num_experts + tid];
      }
    }
    item.barrier(sycl::access::fence_space::local_space);

    if (tid == 0) {
      cumsum[0] = 0;
      for (int i = 1; i <= num_experts; ++i) {
        cumsum[i] = cumsum[i - 1] + CEILDIV(tokens_cnts[block_dim * num_experts + i - 1], block_size) * block_size;
      }
      *total_tokens_post_pad = static_cast<int32_t>(cumsum[num_experts]);
    }
    item.barrier(sycl::access::fence_space::local_space);

    if (tid < num_experts) {
      for (int i = cumsum[tid]; i < cumsum[tid + 1]; i += block_size) {
        expert_ids[i / block_size] = tid - 1;
      }
    }

    if (pad_sorted_token_ids) {
      Vec fill_vec{(int)numel, (int)numel, (int)numel, (int)numel};
      int32_t total_vecs = (*total_tokens_post_pad + VEC_SIZE - 1) / VEC_SIZE;
      Vec* out_ptr = reinterpret_cast<Vec*>(sorted_token_ids);
      for (int32_t i = tid; i < total_vecs; i += stride) {
        out_ptr[i] = fill_vec;
      }
    }
    item.barrier(sycl::access::fence_space::local_space);

    for (size_t i = tid; i < numel; i += stride) {
      int32_t expert_id = topk_ids[i] + 1;
      int32_t rank_post_pad = tokens_cnts[tid * num_experts + expert_id] + cumsum[expert_id];
      sorted_token_ids[rank_post_pad] = i;
      ++tokens_cnts[tid * num_experts + expert_id];
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    const int32_t threads_local = std::max((int32_t)num_experts, sub_group_size);
    const int32_t shared_mem_size = ((threads_local + 1) * num_experts + (num_experts + 1));
    slm_ = sycl::local_accessor<int32_t>(shared_mem_size, cgh);
  }

  const scalar_t* topk_ids;
  int32_t* sorted_token_ids;
  int32_t* expert_ids;
  int32_t* total_tokens_post_pad;
  int32_t num_experts;
  int32_t block_size;
  size_t numel;
  bool pad_sorted_token_ids;
  sycl::local_accessor<int32_t> slm_;
};

void moe_align_block_size(
    torch::Tensor topk_ids,
    int64_t num_experts,
    int64_t block_size,
    torch::Tensor sorted_token_ids,
    torch::Tensor experts_ids,
    torch::Tensor num_tokens_post_pad,
    torch::Tensor cumsum_buffer,
    bool pad_sorted_token_ids) {
  auto q = sycl::queue();

  int threads = 1024;
  threads = ((threads + sub_group_size - 1) / sub_group_size) * sub_group_size;

  DISPATCH_INTEGRAL_TYPES(topk_ids.scalar_type(), "moe_align_block_size_kernel", [&] {
    auto stream = at::xpu::getCurrentXPUStream();
    auto queue = stream.queue();
    bool small_batch_expert_mode = (topk_ids.numel() < 1024) && (num_experts <= 64);

    if (small_batch_expert_mode) {
      const int32_t threads_local = std::max((int32_t)num_experts, sub_group_size);
      auto range = sycl::nd_range<1>(sycl::range<1>(threads_local), sycl::range<1>(threads_local));
      using SmallKernel = MOEAlignBlockSizeSmallBatchExpertFunctor<scalar_t>;
      SmallKernel kernel(
          topk_ids.data_ptr<scalar_t>(),
          sorted_token_ids.data_ptr<int32_t>(),
          experts_ids.data_ptr<int32_t>(),
          num_tokens_post_pad.data_ptr<int32_t>(),
          num_experts,
          block_size,
          topk_ids.numel(),
          pad_sorted_token_ids);
      sycl_kernel_submit(range.get_global_range(), range.get_local_range(), queue, kernel);
    } else {
      const size_t scan_size = next_pow2(num_experts);
      const size_t shared_mem_size = (num_experts + (num_experts + 1) + scan_size + sub_group_size) * sizeof(int32_t);
      using Kernel = MOEAlignBlockSizeFunctor<scalar_t>;
      Kernel kernel(
          topk_ids.data_ptr<scalar_t>(),
          sorted_token_ids.data_ptr<int32_t>(),
          experts_ids.data_ptr<int32_t>(),
          num_tokens_post_pad.data_ptr<int32_t>(),
          num_experts,
          block_size,
          topk_ids.numel(),
          cumsum_buffer.data_ptr<int32_t>(),
          pad_sorted_token_ids,
          scan_size);
      auto range = sycl::nd_range<1>(sycl::range<1>(threads), sycl::range<1>(threads));
      auto local_range = range.get_local_range();
      sycl_kernel_submit(range.get_global_range(), range.get_local_range(), queue, kernel);

      const int block_threads = std::min(256, (int)threads);
      const int num_blocks = (topk_ids.numel() + block_threads - 1) / block_threads;

      using SortKernel = CountAndSortExpertTokensFunctor<scalar_t>;
      SortKernel count_and_sort_kernel(
          topk_ids.data_ptr<scalar_t>(),
          sorted_token_ids.data_ptr<int32_t>(),
          cumsum_buffer.data_ptr<int32_t>(),
          topk_ids.numel());
      auto sort_range = sycl::nd_range<1>(sycl::range<1>(num_blocks * block_threads), sycl::range<1>(block_threads));
      sycl_kernel_submit(sort_range.get_global_range(), sort_range.get_local_range(), queue, count_and_sort_kernel);
    }
  });
}
