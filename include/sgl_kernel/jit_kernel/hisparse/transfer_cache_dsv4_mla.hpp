/**
 * HiSparse: transfer_cache_dsv4_mla SYCL kernel (Intel XPU).
 *
 * Ports transfer_cache_dsv4_mla_kernel<BLOCK_SIZE> from the CUDA source
 * (sglang jit_kernel/csrc/hisparse.cuh). Bulk-copies DSv4-MLA C4 tokens between
 * two sets of page-padded C4 buffers, one set per model layer.
 *
 * Mapping to the CUDA original:
 *   - CUDA "warp" (32 lanes) -> SYCL sub-group (pinned to kSubGroupSize).
 *   - One sub-group copies one item, iterating over all layers.
 *   - Grid-stride loop over items across all sub-groups.
 *
 * src_caches / dst_caches are device arrays of `num_layers` raw cache base
 * pointers (uint64_t values), one per layer.
 */

#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

#include "c4_layout.hpp"

namespace sgl {
namespace sycl_kernel {
namespace hisparse {

// Sub-group width used for the cooperative item copy. Intel GPUs support 16/32;
// 32 mirrors the CUDA warp the kernel was written against. The strided copy in
// transfer_item is correct for any width, so this only affects granularity.
static constexpr int kSubGroupSize = 32;

template <int BLOCK_SIZE>
class TransferCacheDsv4MlaKernel {
 public:
  static constexpr int kNumSubGroups = BLOCK_SIZE / kSubGroupSize;

  TransferCacheDsv4MlaKernel(
      void** src_caches,
      void** dst_caches,
      const int64_t* src_indices,
      const int64_t* dst_indices,
      uint32_t num_items,
      uint32_t num_layers,
      uint32_t total_sub_groups)
      : src_caches_(src_caches),
        dst_caches_(dst_caches),
        src_indices_(src_indices),
        dst_indices_(dst_indices),
        num_items_(num_items),
        num_layers_(num_layers),
        total_sub_groups_(total_sub_groups) {}

  [[sycl::reqd_sub_group_size(kSubGroupSize)]] void operator()(::sycl::nd_item<1> item) const {
    const ::sycl::sub_group sg = item.get_sub_group();
    const int lane_id = static_cast<int>(sg.get_local_linear_id());
    const int sg_size = static_cast<int>(sg.get_max_local_range()[0]);

    // Global sub-group index: group * subgroups_per_group + local subgroup index.
    const uint32_t global_sg =
        static_cast<uint32_t>(item.get_group(0)) * kNumSubGroups + static_cast<uint32_t>(sg.get_group_linear_id());

    for (uint32_t i = global_sg; i < num_items_; i += total_sub_groups_) {
      const int32_t src_index = static_cast<int32_t>(src_indices_[i]);
      const int32_t dst_index = static_cast<int32_t>(dst_indices_[i]);
      for (uint32_t layer_id = 0; layer_id < num_layers_; ++layer_id) {
        transfer_item(lane_id, sg_size, dst_caches_[layer_id], src_caches_[layer_id], dst_index, src_index);
      }
    }
  }

 private:
  void** src_caches_;
  void** dst_caches_;
  const int64_t* src_indices_;
  const int64_t* dst_indices_;
  uint32_t num_items_;
  uint32_t num_layers_;
  uint32_t total_sub_groups_;
};

template <int BLOCK_SIZE>
void transfer_cache_dsv4_mla_launcher(
    ::sycl::queue& queue,
    void** src_caches,
    void** dst_caches,
    const int64_t* src_indices,
    const int64_t* dst_indices,
    uint32_t num_items,
    uint32_t num_layers) {
  if (num_items == 0) {
    return;
  }
  constexpr int kNumSubGroups = BLOCK_SIZE / kSubGroupSize;
  const uint32_t num_groups = (num_items + kNumSubGroups - 1) / kNumSubGroups;
  const uint32_t total_sub_groups = num_groups * kNumSubGroups;

  queue.submit([&](::sycl::handler& cgh) {
    cgh.parallel_for(
        ::sycl::nd_range<1>(
            ::sycl::range<1>(static_cast<size_t>(num_groups) * BLOCK_SIZE), ::sycl::range<1>(BLOCK_SIZE)),
        TransferCacheDsv4MlaKernel<BLOCK_SIZE>(
            src_caches, dst_caches, src_indices, dst_indices, num_items, num_layers, total_sub_groups));
  });
}

// ============================================================================
// C API for Python (ctypes) binding
// ============================================================================

#define _DEFINE_TRANSFER_CACHE_DSV4_MLA(BLOCK_SIZE)         \
  extern "C" void transfer_cache_dsv4_mla_##BLOCK_SIZE(     \
      void* queue_ptr,                                      \
      void* src_caches,                                     \
      void* dst_caches,                                     \
      const void* src_indices,                              \
      const void* dst_indices,                              \
      uint32_t num_items,                                   \
      uint32_t num_layers) {                                \
    auto& queue = *static_cast<::sycl::queue*>(queue_ptr);  \
    transfer_cache_dsv4_mla_launcher<BLOCK_SIZE>(           \
        queue,                                              \
        static_cast<void**>(src_caches),                    \
        static_cast<void**>(dst_caches),                    \
        static_cast<const int64_t*>(src_indices),           \
        static_cast<const int64_t*>(dst_indices),           \
        num_items,                                          \
        num_layers);                                        \
  }
#define DEFINE_TRANSFER_CACHE_DSV4_MLA(BLOCK_SIZE) _DEFINE_TRANSFER_CACHE_DSV4_MLA(BLOCK_SIZE)

#ifdef SGL_HISPARSE_BLOCK_SIZE
DEFINE_TRANSFER_CACHE_DSV4_MLA(SGL_HISPARSE_BLOCK_SIZE)
#else
DEFINE_TRANSFER_CACHE_DSV4_MLA(256)
DEFINE_TRANSFER_CACHE_DSV4_MLA(512)
DEFINE_TRANSFER_CACHE_DSV4_MLA(1024)
#endif

#undef DEFINE_TRANSFER_CACHE_DSV4_MLA
#undef _DEFINE_TRANSFER_CACHE_DSV4_MLA

}  // namespace hisparse
}  // namespace sycl_kernel
}  // namespace sgl
