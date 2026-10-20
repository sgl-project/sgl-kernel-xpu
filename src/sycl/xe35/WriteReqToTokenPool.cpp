#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include "sgl_kernel/xe35/memory_ops.h"

#include <algorithm>
#include <sycl/sycl.hpp>

#include "../SYCLHelpers.h"

namespace at::native::xpu {
namespace {

constexpr int kWorkgroup = 256;

struct WriteReqToTokenPoolKernel {
  int32_t* __restrict__ req_to_token;
  const int64_t* __restrict__ req_pool_indices;
  const uint64_t* __restrict__ prefix_pointers;
  const int64_t* __restrict__ prefix_lens;
  const int64_t* __restrict__ seq_lens;
  const int64_t* __restrict__ extend_lens;
  const int64_t* __restrict__ out_cache_loc;
  int64_t num_reqs;
  int64_t groups_per_request;
  int64_t pool_row_stride;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t req = item.get_group(0) / groups_per_request;
    const int64_t tile = item.get_group(0) % groups_per_request;
    const int64_t lane = item.get_local_id(0);
    if (req >= num_reqs) {
      return;
    }
    const int64_t pool_row = req_pool_indices[req];
    const int64_t prefix_len = prefix_lens[req];
    const int64_t extend_len = seq_lens[req] - prefix_len;
    const auto* prefix = reinterpret_cast<const int64_t*>(static_cast<uintptr_t>(prefix_pointers[req]));
    const int64_t pool_base = pool_row * pool_row_stride;
    for (int64_t pos = tile * kWorkgroup + lane; pos < prefix_len; pos += groups_per_request * kWorkgroup) {
      req_to_token[pool_base + pos] = static_cast<int32_t>(prefix[pos]);
    }
    int64_t extend_start = 0;
    for (int64_t prior = 0; prior < req; ++prior) {
      extend_start += extend_lens[prior];
    }
    for (int64_t pos = tile * kWorkgroup + lane; pos < extend_len; pos += groups_per_request * kWorkgroup) {
      req_to_token[pool_base + prefix_len + pos] = static_cast<int32_t>(out_cache_loc[extend_start + pos]);
    }
  }
};

void check_xpu_vector(const torch::Tensor& tensor, const char* name, at::ScalarType dtype) {
  TORCH_CHECK(
      tensor.is_xpu() && tensor.dim() == 1 && tensor.is_contiguous() && tensor.scalar_type() == dtype,
      name,
      " must be a contiguous XPU vector with the expected dtype");
}

}  // namespace

void write_req_to_token_pool_xe35(
    torch::Tensor req_to_token,
    torch::Tensor req_pool_indices,
    torch::Tensor prefix_pointers,
    torch::Tensor prefix_lens,
    torch::Tensor seq_lens,
    torch::Tensor extend_lens,
    torch::Tensor out_cache_loc) {
  TORCH_CHECK(
      req_to_token.is_xpu() && req_to_token.dim() == 2 && req_to_token.is_contiguous() &&
          req_to_token.scalar_type() == at::kInt,
      "req_to_token must be a contiguous XPU int32 matrix");
  check_xpu_vector(req_pool_indices, "req_pool_indices", at::kLong);
  check_xpu_vector(prefix_pointers, "prefix_pointers", at::kUInt64);
  check_xpu_vector(prefix_lens, "prefix_lens", at::kLong);
  check_xpu_vector(seq_lens, "seq_lens", at::kLong);
  check_xpu_vector(extend_lens, "extend_lens", at::kLong);
  check_xpu_vector(out_cache_loc, "out_cache_loc", at::kLong);
  const int64_t num_reqs = req_pool_indices.numel();
  TORCH_CHECK(
      prefix_pointers.numel() == num_reqs && prefix_lens.numel() == num_reqs && seq_lens.numel() == num_reqs &&
          extend_lens.numel() == num_reqs,
      "request metadata lengths must match");
  if (num_reqs == 0) {
    return;
  }
  auto queue = at::xpu::getCurrentXPUStream().queue();
  const int64_t groups_per_request =
      std::min<int64_t>(8, std::max<int64_t>(1, (req_to_token.size(1) + 511) / 512));
  WriteReqToTokenPoolKernel kernel{
      req_to_token.data_ptr<int32_t>(),
      req_pool_indices.data_ptr<int64_t>(),
      prefix_pointers.data_ptr<uint64_t>(),
      prefix_lens.data_ptr<int64_t>(),
      seq_lens.data_ptr<int64_t>(),
      extend_lens.data_ptr<int64_t>(),
      out_cache_loc.data_ptr<int64_t>(),
      num_reqs,
      groups_per_request,
      req_to_token.stride(0)};
  sycl_kernel_submit(
      sycl::range<1>(num_reqs * groups_per_request * kWorkgroup), sycl::range<1>(kWorkgroup), queue, kernel);
}

}  // namespace at::native::xpu
