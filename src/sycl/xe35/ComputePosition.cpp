#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <algorithm>
#include <sycl/sycl.hpp>

#include "../SYCLHelpers.h"
#include "sgl_kernel/xe35/position_ops.h"

namespace at::native::xpu {
namespace {

constexpr int kPositionWorkgroup = 256;

template <typename PrefixScalar, typename SeqScalar>
struct ComputePositionKernel_xe35 {
  int64_t* __restrict__ positions;
  int32_t* __restrict__ extend_start_loc;
  const PrefixScalar* __restrict__ prefix_lens;
  const SeqScalar* __restrict__ seq_lens;
  int64_t batch;
  int64_t groups_per_request;
  bool has_prefix;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t req = item.get_group(0) / groups_per_request;
    const int64_t tile = item.get_group(0) % groups_per_request;
    const int64_t lane = item.get_local_id(0);
    if (req >= batch) {
      return;
    }
    int64_t start = 0;
    for (int64_t i = 0; i < req; ++i) {
      start += seq_lens[i];
    }
    const int64_t prefix = has_prefix ? prefix_lens[req] : 0;
    const int64_t extend = seq_lens[req];
    for (int64_t i = tile * kPositionWorkgroup + lane; i < extend; i += groups_per_request * kPositionWorkgroup) {
      positions[start + i] = prefix + i;
    }
    if (tile == 0 && lane == 0) {
      extend_start_loc[req] = static_cast<int32_t>(start);
    }
  }
};

}  // namespace

void compute_position_impl_xe35(
    torch::Tensor positions,
    torch::Tensor extend_start_loc,
    torch::Tensor prefix_lens,
    torch::Tensor seq_lens,
    int64_t max_seq_len_hint) {
  TORCH_CHECK(
      positions.is_xpu() && positions.scalar_type() == at::kLong && positions.is_contiguous(),
      "positions must be contiguous XPU int64");
  TORCH_CHECK(
      extend_start_loc.is_xpu() && extend_start_loc.scalar_type() == at::kInt && extend_start_loc.is_contiguous(),
      "extend_start_loc must be contiguous XPU int32");
  TORCH_CHECK(
      prefix_lens.is_xpu() && (prefix_lens.scalar_type() == at::kInt || prefix_lens.scalar_type() == at::kLong) &&
          prefix_lens.is_contiguous() && seq_lens.is_xpu() &&
          (seq_lens.scalar_type() == at::kInt || seq_lens.scalar_type() == at::kLong) && seq_lens.is_contiguous(),
      "lengths must be contiguous XPU int32 or int64");
  const int64_t batch = seq_lens.numel();
  TORCH_CHECK(
      extend_start_loc.numel() == batch && (prefix_lens.numel() == 0 || prefix_lens.numel() == batch),
      "lengths must match batch");
  if (batch == 0) {
    return;
  }
  auto queue = at::xpu::getCurrentXPUStream().queue();
  const bool has_prefix = prefix_lens.numel() == batch;
  // Spread long requests across workgroups; keep one group for short requests.
  // The model already has a host length list. Its maximum avoids stranding a
  // long request when the average length is diluted by empty/short requests.
  // A stale or missing hint only changes occupancy: the kernel's strided loop
  // still covers every sequence element.
  const int64_t average_groups = (positions.numel() + batch * 512 - 1) / (batch * 512);
  const int64_t hinted_groups = max_seq_len_hint > 0 ? (max_seq_len_hint + 511) / 512 : 1;
  // There is no benefit in launching more groups per request than there are
  // workgroup-sized chunks in the entire positions output.
  const int64_t useful_groups = std::max<int64_t>(1, (positions.numel() + kPositionWorkgroup - 1) / kPositionWorkgroup);
  const int64_t groups_per_request = std::min(useful_groups, std::max({int64_t{1}, average_groups, hinted_groups}));
  const auto global_size = sycl::range<1>(batch * groups_per_request * kPositionWorkgroup);
  const auto local_size = sycl::range<1>(kPositionWorkgroup);
  if (prefix_lens.scalar_type() == at::kInt && seq_lens.scalar_type() == at::kInt) {
    ComputePositionKernel_xe35<int32_t, int32_t> kernel{
        positions.data_ptr<int64_t>(),
        extend_start_loc.data_ptr<int32_t>(),
        prefix_lens.data_ptr<int32_t>(),
        seq_lens.data_ptr<int32_t>(),
        batch,
        groups_per_request,
        has_prefix};
    sycl_kernel_submit(global_size, local_size, queue, kernel);
  } else if (prefix_lens.scalar_type() == at::kLong && seq_lens.scalar_type() == at::kLong) {
    ComputePositionKernel_xe35<int64_t, int64_t> kernel{
        positions.data_ptr<int64_t>(),
        extend_start_loc.data_ptr<int32_t>(),
        prefix_lens.data_ptr<int64_t>(),
        seq_lens.data_ptr<int64_t>(),
        batch,
        groups_per_request,
        has_prefix};
    sycl_kernel_submit(global_size, local_size, queue, kernel);
  } else if (prefix_lens.scalar_type() == at::kInt) {
    ComputePositionKernel_xe35<int32_t, int64_t> kernel{
        positions.data_ptr<int64_t>(),
        extend_start_loc.data_ptr<int32_t>(),
        prefix_lens.data_ptr<int32_t>(),
        seq_lens.data_ptr<int64_t>(),
        batch,
        groups_per_request,
        has_prefix};
    sycl_kernel_submit(global_size, local_size, queue, kernel);
  } else {
    ComputePositionKernel_xe35<int64_t, int32_t> kernel{
        positions.data_ptr<int64_t>(),
        extend_start_loc.data_ptr<int32_t>(),
        prefix_lens.data_ptr<int64_t>(),
        seq_lens.data_ptr<int32_t>(),
        batch,
        groups_per_request,
        has_prefix};
    sycl_kernel_submit(global_size, local_size, queue, kernel);
  }
}

void compute_position_xe35(
    torch::Tensor positions, torch::Tensor extend_start_loc, torch::Tensor prefix_lens, torch::Tensor seq_lens) {
  compute_position_impl_xe35(positions, extend_start_loc, prefix_lens, seq_lens, -1);
}

void compute_position_with_hint_xe35(
    torch::Tensor positions,
    torch::Tensor extend_start_loc,
    torch::Tensor prefix_lens,
    torch::Tensor seq_lens,
    int64_t max_seq_len_hint) {
  TORCH_CHECK(max_seq_len_hint >= 0, "max_seq_len_hint must be nonnegative");
  compute_position_impl_xe35(positions, extend_start_loc, prefix_lens, seq_lens, max_seq_len_hint);
}

}  // namespace at::native::xpu
