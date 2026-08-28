/**
 * HiSparse: transfer_cache_dsv4_mla SYCL kernel.
 *
 * Bulk-copies DSv4-MLA C4 tokens between two sets of page-padded C4 buffers, one
 * set per model layer. One sub-group copies one item across all layers, with a
 * global-stride loop over items.
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

// Aggregate-initialized so the launcher can name the fields.
template <int BLOCK_SIZE>
struct TransferCacheDsv4MlaKernel {
  static_assert(BLOCK_SIZE % kSubGroupSize == 0, "BLOCK_SIZE must be a multiple of the sub-group size (32).");
  static constexpr int kNumSubGroups = BLOCK_SIZE / kSubGroupSize;

  void** src_caches_;
  void** dst_caches_;
  const int64_t* src_indices_;
  const int64_t* dst_indices_;
  uint32_t num_items_;
  uint32_t num_layers_;
  uint32_t total_sub_groups_;

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
};

}  // namespace hisparse
}  // namespace sycl_kernel
}  // namespace sgl
