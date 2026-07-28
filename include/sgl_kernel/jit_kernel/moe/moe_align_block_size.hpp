/**
 * MoE Align Block Size SYCL JIT Kernel for SGLang XPU
 *
 * Prepares MoE routing for block-wise grouped GEMM. Given topk_ids (the expert
 * each token-slot selected), it pads each expert's token count up to a multiple
 * of block_size and produces:
 *   sorted_token_ids       : tokens grouped/sorted by expert (padded with numel)
 *   expert_ids             : expert index owning each output block
 *   num_tokens_post_pad    : total padded token count
 *   cumsum_buffer          : per-expert padded prefix sums
 *
 * Integer-only (counting + Blelloch prefix-sum + scatter-sort); no float math.
 *
 * Self-contained JIT port of the AOT kernel in src/sycl/MoEAlign.cpp. Keeps both
 * AOT code paths, selected at runtime (dtype of topk_ids is the only compile-time
 * specialization, so one .so per integer dtype serves every shape — mirrors the
 * CUDA JIT moe_align_kernel design):
 *   - small-batch path (numel < 1024 && num_experts <= 64): single work-group,
 *     per-thread private counts + sequential cumsum + scatter, all in SLM.
 *   - general path: work-group Blelloch scan of padded counts + a separate
 *     count_and_sort pass using device-scope atomics into cumsum_buffer.
 *
 * The +1 expert offset (topk_ids[i] + 1, so -1 padding maps to bucket 0) and the
 * int4-vectorized sorted_token_ids fill are preserved from the AOT kernel.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>

namespace sgl {
namespace sycl_kernel {
namespace moe_align {

static constexpr int kSubGroupSize = 32;
static constexpr int kVecSize = 4;
using Vec = ::sycl::int4;

inline int host_div_up(int a, int b) {
  return (a + b - 1) / b;
}

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

// ---------------------------------------------------------------------------
// count_and_sort: device-atomic scatter of each token into its expert bucket
// using the (exclusive) cumsum offsets in cumsum_buffer.
// ---------------------------------------------------------------------------
template <typename scalar_t>
struct CountAndSortExpertTokensFunctor {
  const scalar_t* topk_ids;
  int32_t* sorted_token_ids;
  int32_t* cumsum_buffer;
  size_t numel;

  [[sycl::reqd_sub_group_size(kSubGroupSize)]] void operator()(::sycl::nd_item<1> item) const {
    const size_t tid = item.get_global_id(0);
    const size_t stride = item.get_global_range(0);
    for (size_t i = tid; i < numel; i += stride) {
      int32_t expert_id = static_cast<int32_t>(topk_ids[i]) + 1;
      ::sycl::atomic_ref<int32_t, ::sycl::memory_order::relaxed, ::sycl::memory_scope::device> atomicRef(
          cumsum_buffer[expert_id]);
      int32_t rank_post_pad = atomicRef.fetch_add(1);
      sorted_token_ids[rank_post_pad] = static_cast<int32_t>(i);
    }
  }
};

// ---------------------------------------------------------------------------
// General path: work-group Blelloch scan over padded per-expert counts, then
// binary-search fill of expert_ids and (optional) vectorized padding fill.
// ---------------------------------------------------------------------------
template <typename scalar_t>
struct MOEAlignBlockSizeFunctor {
  const scalar_t* topk_ids;
  int32_t* sorted_token_ids;
  int32_t* expert_ids;
  int32_t* total_tokens_post_pad;
  int32_t num_experts;
  int32_t block_size;
  size_t numel;
  int32_t* cumsum;
  bool pad_sorted_token_ids;
  int32_t scan_size;
  ::sycl::local_accessor<int32_t> slm_;
  ::sycl::local_accessor<int32_t> total_token_;

  [[sycl::reqd_sub_group_size(kSubGroupSize)]] void operator()(::sycl::nd_item<1> item) const {
    const size_t tid = item.get_local_id(0);
    const size_t stride = item.get_local_range(0);

    int32_t* shared_counts = slm_.template get_multi_ptr<::sycl::access::decorated::no>().get();
    int32_t* prefix = shared_counts + num_experts;
    int32_t* scan_buf = prefix + num_experts + 1;
    int32_t* s_total_tokens_post_pad = total_token_.template get_multi_ptr<::sycl::access::decorated::no>().get();
    *s_total_tokens_post_pad = 0;

    if (tid < static_cast<size_t>(num_experts)) {
      shared_counts[tid] = 0;
    }
    item.barrier(::sycl::access::fence_space::local_space);

    for (size_t i = tid; i < numel; i += stride) {
      int expert_id = static_cast<int32_t>(topk_ids[i]) + 1;
      ::sycl::atomic_ref<int32_t, ::sycl::memory_order::relaxed, ::sycl::memory_scope::work_group> atomicRef(
          shared_counts[expert_id]);
      atomicRef.fetch_add(1);
    }
    item.barrier(::sycl::access::fence_space::local_space);

    int32_t padded_count = 0;
    if (tid < static_cast<size_t>(num_experts)) {
      int32_t count = shared_counts[tid];
      padded_count = (count + block_size - 1) / block_size * block_size;
      scan_buf[tid] = padded_count;
    }

    if (tid >= static_cast<size_t>(num_experts) && tid < static_cast<size_t>(scan_size)) {
      scan_buf[tid] = 0;
    }
    item.barrier(::sycl::access::fence_space::local_space);

    // Blelloch up-sweep
    int offset = 1;
    for (int d = scan_size >> 1; d > 0; d >>= 1) {
      if (tid < static_cast<size_t>(d)) {
        int ai = offset * (2 * tid + 1) - 1;
        int bi = offset * (2 * tid + 2) - 1;
        scan_buf[bi] += scan_buf[ai];
      }
      offset <<= 1;
      item.barrier(::sycl::access::fence_space::local_space);
    }

    // down-sweep
    if (tid == 0) {
      prefix[num_experts] = scan_buf[scan_size - 1];
      scan_buf[scan_size - 1] = 0;
    }
    item.barrier(::sycl::access::fence_space::local_space);

    for (int d = 1; d < scan_size; d <<= 1) {
      offset >>= 1;
      if (tid < static_cast<size_t>(d)) {
        int ai = offset * (2 * tid + 1) - 1;
        int bi = offset * (2 * tid + 2) - 1;
        if (bi < scan_size) {
          int temp = scan_buf[ai];
          scan_buf[ai] = scan_buf[bi];
          scan_buf[bi] += temp;
        }
      }
      item.barrier(::sycl::access::fence_space::local_space);
    }

    if (tid < static_cast<size_t>(num_experts)) {
      prefix[tid] = scan_buf[tid];
    }

    if (tid == 0) {
      *s_total_tokens_post_pad = prefix[num_experts];
      *total_tokens_post_pad = *s_total_tokens_post_pad;
    }
    item.barrier(::sycl::access::fence_space::local_space);

    if (tid <= static_cast<size_t>(num_experts)) {
      cumsum[tid] = prefix[tid];
    }

    // Fill expert_ids via binary search over the padded prefix boundaries.
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
      int32_t total_vecs = (*s_total_tokens_post_pad + kVecSize - 1) / kVecSize;
      Vec* out_ptr = reinterpret_cast<Vec*>(sorted_token_ids);
      for (int32_t i = tid; i < total_vecs; i += stride) {
        out_ptr[i] = fill_vec;
      }
    }
  }
};

// ---------------------------------------------------------------------------
// Small-batch path: single work-group, per-thread private counts in SLM, then
// sequential cumsum + in-SLM scatter (no device atomics, no separate pass).
// ---------------------------------------------------------------------------
template <typename scalar_t>
struct MOEAlignBlockSizeSmallBatchExpertFunctor {
  const scalar_t* topk_ids;
  int32_t* sorted_token_ids;
  int32_t* expert_ids;
  int32_t* total_tokens_post_pad;
  int32_t num_experts;
  int32_t block_size;
  size_t numel;
  bool pad_sorted_token_ids;
  ::sycl::local_accessor<int32_t> slm_;

  [[sycl::reqd_sub_group_size(kSubGroupSize)]] void operator()(::sycl::nd_item<1> item) const {
    const size_t tid = item.get_local_id(0);
    const size_t stride = item.get_local_range(0);
    const size_t block_dim = item.get_local_range(0);

    int32_t* shared_mem = slm_.template get_multi_ptr<::sycl::access::decorated::no>().get();
    int32_t* cumsum = shared_mem;
    int32_t* tokens_cnts = shared_mem + num_experts + 1;

    for (int i = 0; i < num_experts; ++i) {
      tokens_cnts[num_experts + i] = 0;
    }
    item.barrier(::sycl::access::fence_space::local_space);

    for (size_t i = tid; i < numel; i += stride) {
      ++tokens_cnts[(tid + 1) * num_experts + static_cast<int32_t>(topk_ids[i]) + 1];
    }
    item.barrier(::sycl::access::fence_space::local_space);

    if (tid < static_cast<size_t>(num_experts)) {
      tokens_cnts[tid] = 0;
      for (size_t i = 1; i <= block_dim; ++i) {
        tokens_cnts[i * num_experts + tid] += tokens_cnts[(i - 1) * num_experts + tid];
      }
    }
    item.barrier(::sycl::access::fence_space::local_space);

    if (tid == 0) {
      cumsum[0] = 0;
      for (int i = 1; i <= num_experts; ++i) {
        cumsum[i] =
            cumsum[i - 1] + (tokens_cnts[block_dim * num_experts + i - 1] + block_size - 1) / block_size * block_size;
      }
      *total_tokens_post_pad = static_cast<int32_t>(cumsum[num_experts]);
    }
    item.barrier(::sycl::access::fence_space::local_space);

    if (tid < static_cast<size_t>(num_experts)) {
      for (int i = cumsum[tid]; i < cumsum[tid + 1]; i += block_size) {
        expert_ids[i / block_size] = tid - 1;
      }
    }

    if (pad_sorted_token_ids) {
      Vec fill_vec{(int)numel, (int)numel, (int)numel, (int)numel};
      int32_t total_vecs = (*total_tokens_post_pad + kVecSize - 1) / kVecSize;
      Vec* out_ptr = reinterpret_cast<Vec*>(sorted_token_ids);
      for (int32_t i = tid; i < total_vecs; i += stride) {
        out_ptr[i] = fill_vec;
      }
    }
    item.barrier(::sycl::access::fence_space::local_space);

    for (size_t i = tid; i < numel; i += stride) {
      int32_t expert_id = static_cast<int32_t>(topk_ids[i]) + 1;
      int32_t rank_post_pad = tokens_cnts[tid * num_experts + expert_id] + cumsum[expert_id];
      sorted_token_ids[rank_post_pad] = static_cast<int32_t>(i);
      ++tokens_cnts[tid * num_experts + expert_id];
    }
  }
};

// ---------------------------------------------------------------------------
// Host launcher: runtime dispatch between the small-batch and general paths.
// ---------------------------------------------------------------------------
template <typename scalar_t>
void moe_align_block_size_launcher(
    ::sycl::queue& queue,
    const void* topk_ids,
    void* sorted_token_ids,
    void* expert_ids,
    void* num_tokens_post_pad,
    void* cumsum_buffer,
    int64_t num_experts,
    int64_t block_size,
    int64_t numel,
    bool pad_sorted_token_ids) {
  const scalar_t* topk_ids_ptr = static_cast<const scalar_t*>(topk_ids);
  int32_t* sorted_token_ids_ptr = static_cast<int32_t*>(sorted_token_ids);
  int32_t* expert_ids_ptr = static_cast<int32_t*>(expert_ids);
  int32_t* total_tokens_post_pad_ptr = static_cast<int32_t*>(num_tokens_post_pad);
  int32_t* cumsum_buffer_ptr = static_cast<int32_t*>(cumsum_buffer);

  int threads = 1024;
  threads = ((threads + kSubGroupSize - 1) / kSubGroupSize) * kSubGroupSize;

  const bool small_batch_expert_mode = (numel < 1024) && (num_experts <= 64);

  if (small_batch_expert_mode) {
    const int32_t threads_local = std::max(static_cast<int32_t>(num_experts), kSubGroupSize);
    const int32_t shared_mem_size = ((threads_local + 1) * num_experts + (num_experts + 1));

    queue.submit([&](::sycl::handler& cgh) {
      ::sycl::local_accessor<int32_t> slm(shared_mem_size, cgh);
      MOEAlignBlockSizeSmallBatchExpertFunctor<scalar_t> kernel{
          topk_ids_ptr,
          sorted_token_ids_ptr,
          expert_ids_ptr,
          total_tokens_post_pad_ptr,
          static_cast<int32_t>(num_experts),
          static_cast<int32_t>(block_size),
          static_cast<size_t>(numel),
          pad_sorted_token_ids,
          slm};
      cgh.parallel_for(::sycl::nd_range<1>(::sycl::range<1>(threads_local), ::sycl::range<1>(threads_local)), kernel);
    });
  } else {
    const int32_t scan_size = static_cast<int32_t>(next_pow2(num_experts));
    const size_t shared_mem_size = num_experts + (num_experts + 1) + scan_size + kSubGroupSize;

    queue.submit([&](::sycl::handler& cgh) {
      ::sycl::local_accessor<int32_t> slm(shared_mem_size, cgh);
      ::sycl::local_accessor<int32_t> total_token(1, cgh);
      MOEAlignBlockSizeFunctor<scalar_t> kernel{
          topk_ids_ptr,
          sorted_token_ids_ptr,
          expert_ids_ptr,
          total_tokens_post_pad_ptr,
          static_cast<int32_t>(num_experts),
          static_cast<int32_t>(block_size),
          static_cast<size_t>(numel),
          cumsum_buffer_ptr,
          pad_sorted_token_ids,
          scan_size,
          slm,
          total_token};
      cgh.parallel_for(::sycl::nd_range<1>(::sycl::range<1>(threads), ::sycl::range<1>(threads)), kernel);
    });

    const int block_threads = std::min(256, threads);
    const int num_blocks = (numel + block_threads - 1) / block_threads;

    queue.submit([&](::sycl::handler& cgh) {
      CountAndSortExpertTokensFunctor<scalar_t> kernel{
          topk_ids_ptr, sorted_token_ids_ptr, cumsum_buffer_ptr, static_cast<size_t>(numel)};
      cgh.parallel_for(
          ::sycl::nd_range<1>(
              ::sycl::range<1>(static_cast<size_t>(num_blocks) * block_threads), ::sycl::range<1>(block_threads)),
          kernel);
    });
  }
  // NOTE: no .wait() -- PyTorch owns stream synchronization.
}

}  // namespace moe_align

// ---------------------------------------------------------------------------
// C API for the Python ctypes wrapper. One symbol per topk_ids integer dtype;
// all shape/config values are runtime arguments.
// ---------------------------------------------------------------------------
#define DEFINE_MOE_ALIGN_FORWARD(DTYPE_SUFFIX, DTYPE)          \
  extern "C" void moe_align_block_size_forward_##DTYPE_SUFFIX( \
      void* queue_ptr,                                         \
      const void* topk_ids,                                    \
      void* sorted_token_ids,                                  \
      void* expert_ids,                                        \
      void* num_tokens_post_pad,                               \
      void* cumsum_buffer,                                     \
      int64_t num_experts,                                     \
      int64_t block_size,                                      \
      int64_t numel,                                           \
      int32_t pad_sorted_token_ids) {                          \
    auto& queue = *static_cast<::sycl::queue*>(queue_ptr);     \
    moe_align::moe_align_block_size_launcher<DTYPE>(           \
        queue,                                                 \
        topk_ids,                                              \
        sorted_token_ids,                                      \
        expert_ids,                                            \
        num_tokens_post_pad,                                   \
        cumsum_buffer,                                         \
        num_experts,                                           \
        block_size,                                            \
        numel,                                                 \
        pad_sorted_token_ids != 0);                            \
  }

#if defined(SGL_MOE_ALIGN_DTYPE_i32)
DEFINE_MOE_ALIGN_FORWARD(i32, int32_t)
#elif defined(SGL_MOE_ALIGN_DTYPE_i64)
DEFINE_MOE_ALIGN_FORWARD(i64, int64_t)
#else
DEFINE_MOE_ALIGN_FORWARD(i32, int32_t)
DEFINE_MOE_ALIGN_FORWARD(i64, int64_t)
#endif

#undef DEFINE_MOE_ALIGN_FORWARD

}  // namespace sycl_kernel
}  // namespace sgl
