#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <algorithm>
#include <limits>
#include <sycl/sycl.hpp>

#include "sycl/SYCLHelpers.h"
#include "sgl_kernel/xe35/mxfp4_permutation_ops.h"

namespace at::native::xpu {
namespace {
constexpr int kWorkgroup_xe35 = 256;

template <typename Index>
struct Mxfp4SmallPermutationKernel_xe35 : __SYCL_KER_CONFIG_CONVENTION__ {
  Mxfp4SmallPermutationKernel_xe35(
      const Index* topk_ids,
      int32_t* padded_counts,
      int32_t* perm_src,
      int32_t* expanded_to_permuted,
      int32_t* scale_map,
      int num_pairs,
      int num_experts,
      int num_rows,
      int topk,
      int alignment)
      : topk_ids(topk_ids),
        padded_counts(padded_counts),
        perm_src(perm_src),
        expanded_to_permuted(expanded_to_permuted),
        scale_map(scale_map),
        num_pairs(num_pairs),
        num_experts(num_experts),
        num_rows(num_rows),
        topk(topk),
        alignment(alignment) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {
    counts = sycl::local_accessor<int32_t, 1>(sycl::range<1>(num_experts), cgh);
    starts = sycl::local_accessor<int32_t, 1>(sycl::range<1>(num_experts), cgh);
    padded = sycl::local_accessor<int32_t, 1>(sycl::range<1>(num_experts), cgh);
    expert_ids = sycl::local_accessor<int32_t, 1>(sycl::range<1>(kWorkgroup_xe35), cgh);
    total_rows = sycl::local_accessor<int32_t, 1>(sycl::range<1>(1), cgh);
  }

  void operator()(sycl::nd_item<1> item) const {
    const int lane = static_cast<int>(item.get_local_id(0));
    for (int expert = lane; expert < num_experts; expert += kWorkgroup_xe35) {
      counts[expert] = 0;
    }
    if (lane < num_pairs) {
      expert_ids[lane] = static_cast<int32_t>(topk_ids[lane]);
    }
    item.barrier(sycl::access::fence_space::local_space);
    if (lane < num_pairs) {
      sycl::atomic_ref<
          int32_t,
          sycl::memory_order::relaxed,
          sycl::memory_scope::work_group,
          sycl::access::address_space::local_space>
          count(counts[expert_ids[lane]]);
      count.fetch_add(1);
    }
    item.barrier(sycl::access::fence_space::local_space);
    if (num_experts <= kWorkgroup_xe35) {
      const int rows = lane < num_experts ? ((counts[lane] + alignment - 1) / alignment) * alignment : 0;
      const int base = sycl::exclusive_scan_over_group(item.get_group(), rows, sycl::plus<int>());
      if (lane < num_experts) {
        starts[lane] = base;
        padded[lane] = rows;
        padded_counts[lane] = rows;
      }
      if (lane == kWorkgroup_xe35 - 1) total_rows[0] = base + rows;
    } else {
      const int per_lane = num_experts / kWorkgroup_xe35;
      const int first = lane * per_lane;
      int sum = 0;
      for (int expert_offset = 0; expert_offset < per_lane; ++expert_offset) {
        const int count = counts[first + expert_offset];
        sum += ((count + alignment - 1) / alignment) * alignment;
      }
      int row = sycl::exclusive_scan_over_group(item.get_group(), sum, sycl::plus<int>());
      for (int expert_offset = 0; expert_offset < per_lane; ++expert_offset) {
        const int expert = first + expert_offset;
        const int rows = ((counts[expert] + alignment - 1) / alignment) * alignment;
        starts[expert] = row;
        padded[expert] = rows;
        padded_counts[expert] = rows;
        row += rows;
      }
      if (lane == kWorkgroup_xe35 - 1) total_rows[0] = row;
    }
    item.barrier(sycl::access::fence_space::local_space);
    for (int64_t row = total_rows[0] + lane; row < num_rows; row += kWorkgroup_xe35) {
      perm_src[row] = 0;
      scale_map[2 * row] = static_cast<int32_t>(row);
      scale_map[2 * row + 1] = 0;
    }
    if (lane >= num_pairs) return;
    const int expert = expert_ids[lane];
    int rank = 0;
    int earlier = 0;
    for (; earlier + 3 < lane; earlier += 4) {
      rank += (expert_ids[earlier] == expert) + (expert_ids[earlier + 1] == expert) +
              (expert_ids[earlier + 2] == expert) + (expert_ids[earlier + 3] == expert);
    }
    for (; earlier < lane; ++earlier) {
      rank += expert_ids[earlier] == expert;
    }
    const int base = starts[expert];
    const int rows = padded[expert];
    const int dst = base + rank;
    expanded_to_permuted[lane] = dst;
    perm_src[dst] = lane / topk;
    scale_map[2 * dst] = base;
    scale_map[2 * dst + 1] = rows;
    if (rank == counts[expert] - 1) {
      for (int padding_offset = 1; padding_offset < alignment && dst + padding_offset < base + rows; ++padding_offset) {
        const int pad = dst + padding_offset;
        perm_src[pad] = 0;
        scale_map[2 * pad] = base;
        scale_map[2 * pad + 1] = -rows;
      }
    }
  }

  const Index* topk_ids;
  int32_t* padded_counts;
  int32_t* perm_src;
  int32_t* expanded_to_permuted;
  int32_t* scale_map;
  int num_pairs;
  int num_experts;
  int num_rows;
  int topk;
  int alignment;
  sycl::local_accessor<int32_t, 1> counts;
  sycl::local_accessor<int32_t, 1> starts;
  sycl::local_accessor<int32_t, 1> padded;
  sycl::local_accessor<int32_t, 1> expert_ids;
  sycl::local_accessor<int32_t, 1> total_rows;
};

template <typename Index>
void launch_small_permutation_xe35(
    const Index* topk_ids,
    int32_t* padded_counts,
    int32_t* perm_src,
    int32_t* expanded_to_permuted,
    int32_t* scale_map,
    int num_pairs,
    int num_experts,
    int num_rows,
    int topk,
    int alignment,
    sycl::queue& queue) {
  Mxfp4SmallPermutationKernel_xe35<Index> kernel(
      topk_ids,
      padded_counts,
      perm_src,
      expanded_to_permuted,
      scale_map,
      num_pairs,
      num_experts,
      num_rows,
      topk,
      alignment);
  sycl_kernel_submit(sycl::range<1>(kWorkgroup_xe35), sycl::range<1>(kWorkgroup_xe35), queue, kernel);
}
}  // namespace

void mxfp4_small_permutation_xe35(
    torch::Tensor topk_ids,
    torch::Tensor padded_counts,
    torch::Tensor perm_src,
    torch::Tensor expanded_to_permuted,
    torch::Tensor scale_map,
    int64_t alignment) {
  TORCH_CHECK(
      topk_ids.is_xpu() && topk_ids.dim() == 2 && topk_ids.is_contiguous() &&
          (topk_ids.scalar_type() == at::kInt || topk_ids.scalar_type() == at::kLong),
      "topk_ids must be contiguous XPU int32 or int64 [tokens, topk]");
  TORCH_CHECK(
      padded_counts.is_xpu() && padded_counts.dim() == 1 && padded_counts.is_contiguous() &&
          padded_counts.scalar_type() == at::kInt,
      "padded_counts must be contiguous XPU int32 [experts]");
  TORCH_CHECK(
      perm_src.is_xpu() && perm_src.dim() == 1 && perm_src.is_contiguous() && perm_src.scalar_type() == at::kInt,
      "perm_src must be contiguous XPU int32 [padded_rows]");
  TORCH_CHECK(
      expanded_to_permuted.is_xpu() && expanded_to_permuted.dim() == 1 && expanded_to_permuted.is_contiguous() &&
          expanded_to_permuted.scalar_type() == at::kInt,
      "expanded_to_permuted must be contiguous XPU int32 [pairs]");
  TORCH_CHECK(
      scale_map.is_xpu() && scale_map.dim() == 2 && scale_map.is_contiguous() && scale_map.scalar_type() == at::kInt &&
          scale_map.size(1) == 2,
      "scale_map must be contiguous XPU int32 [padded_rows, 2]");
  const auto pairs = topk_ids.numel();
  const auto experts = padded_counts.numel();
  TORCH_CHECK(
      padded_counts.device() == topk_ids.device() && perm_src.device() == topk_ids.device() &&
          expanded_to_permuted.device() == topk_ids.device() && scale_map.device() == topk_ids.device(),
      "all permutation tensors must be on the same XPU device");
  TORCH_CHECK(
      topk_ids.size(1) > 0 && topk_ids.size(1) <= std::numeric_limits<int>::max() &&
          perm_src.numel() <= std::numeric_limits<int>::max() && pairs <= kWorkgroup_xe35 &&
          experts > 0 && experts <= 1024 && (experts & (experts - 1)) == 0 &&
          alignment > 0 && alignment <= 64 && expanded_to_permuted.numel() == pairs &&
          scale_map.size(0) == perm_src.numel() &&
          perm_src.numel() >= pairs + (alignment - 1) * std::min(experts, pairs),
      "invalid small permutation dimensions or alignment");
  c10::DeviceGuard guard(topk_ids.device());
  auto& queue = at::xpu::getCurrentXPUStream().queue();
  if (topk_ids.scalar_type() == at::kInt) {
    launch_small_permutation_xe35(
        topk_ids.data_ptr<int32_t>(),
        padded_counts.data_ptr<int32_t>(),
        perm_src.data_ptr<int32_t>(),
        expanded_to_permuted.data_ptr<int32_t>(),
        scale_map.data_ptr<int32_t>(),
        static_cast<int>(pairs),
        static_cast<int>(experts),
        static_cast<int>(perm_src.numel()),
        static_cast<int>(topk_ids.size(1)),
        static_cast<int>(alignment),
        queue);
  } else {
    launch_small_permutation_xe35(
        topk_ids.data_ptr<int64_t>(),
        padded_counts.data_ptr<int32_t>(),
        perm_src.data_ptr<int32_t>(),
        expanded_to_permuted.data_ptr<int32_t>(),
        scale_map.data_ptr<int32_t>(),
        static_cast<int>(pairs),
        static_cast<int>(experts),
        static_cast<int>(perm_src.numel()),
        static_cast<int>(topk_ids.size(1)),
        static_cast<int>(alignment),
        queue);
  }
}
}  // namespace at::native::xpu
