#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <sycl/sycl.hpp>

#include "MemoryAccess.h"
#include "SYCLHelpers.h"
#include "Utils.h"

template <typename T, int VEC = 1>
struct moe_sum_reduce_impl_sycl_k {
  moe_sum_reduce_impl_sycl_k(
      const T* input,
      T* output,
      const uint32_t token_num,
      const uint32_t topk_num,
      const uint32_t hidden_dim,
      const uint32_t in_stride_token,
      const uint32_t in_stride_topk,
      const uint32_t out_stride_token,
      const uint32_t max_wg_dims,
      double routed_scaling_factor)
      : input_(input),
        output_(output),
        token_num_(token_num),
        topk_num_(topk_num),
        hidden_dim_(hidden_dim),
        in_stride_token_(in_stride_token),
        in_stride_topk_(in_stride_topk),
        out_stride_token_(out_stride_token),
        max_wg_dims_(max_wg_dims),
        routed_scaling_factor_(routed_scaling_factor) {}

  void operator()(sycl::nd_item<1> it) const {
    int tkn_id = it.get_group(0);
    int thread_id = it.get_local_id(0);
    const int src_offset = tkn_id * in_stride_token_;
    const int dst_offset = tkn_id * out_stride_token_;
    for (int i = thread_id; i < hidden_dim_; i += max_wg_dims_) {
      float acc = 0;
#pragma unroll
      for (int k = 0; k < topk_num_; ++k) {
        float src_val = static_cast<float>(input_[src_offset + k * in_stride_topk_ + i]);
        acc += src_val;
      }
      acc *= static_cast<float>(routed_scaling_factor_);
      output_[dst_offset + i] = static_cast<T>(acc);

    }
  }

  const T* input_;
  T* output_;
  const uint32_t token_num_;
  const uint32_t topk_num_;
  const uint32_t hidden_dim_;
  const uint32_t in_stride_token_;
  const uint32_t in_stride_topk_;
  const uint32_t out_stride_token_;
  const uint32_t max_wg_dims_;
  double routed_scaling_factor_;
};

template <typename T>
void moe_sum_reduce_impl(at::Tensor& input_tensor, at::Tensor& output_tensor, double routed_scaling_factor) {
  auto input = reinterpret_cast<T*>(input_tensor.data_ptr());
  auto output = reinterpret_cast<T*>(output_tensor.data_ptr());

  const uint32_t token_num = input_tensor.size(0);
  const uint32_t topk_num = input_tensor.size(1);
  const uint32_t hidden_dim = input_tensor.size(2);

  const uint32_t in_stride_token = input_tensor.stride(0);
  const uint32_t in_stride_topk = input_tensor.stride(1);
  const int64_t out_stride_token = output_tensor.stride(0);

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  constexpr int VEC = 1;
  using Kernel = moe_sum_reduce_impl_sycl_k<T, VEC>;

  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  uint32_t max_wg_size = dpcppMaxWorkGroupSize(dev_id);
  uint32_t max_wg_dims = static_cast<uint32_t>(sycl::min(max_wg_size, hidden_dim));

  sycl::range<1> global_range{token_num * max_wg_size };
  sycl::range<1> local_range{max_wg_size};

  Kernel task(
      input,
      output,
      token_num,
      topk_num,
      hidden_dim,
      in_stride_token,
      in_stride_topk,
      out_stride_token,
      max_wg_dims,
      routed_scaling_factor);

  sycl_kernel_submit(global_range, local_range, queue, task);
  return;
}

void moe_sum_reduce(at::Tensor& input, at::Tensor& output, double routed_scaling_factor) {
  TORCH_CHECK(input.dim() == 3, "input must be a 3D tensor like [token_num, topk_num, hidden_dim]");
  TORCH_CHECK(output.dim() == 2, "output must be [token_num, hidden_dim]");
  TORCH_CHECK(input.size(0) == output.size(0), "token dim mismatch");
  TORCH_CHECK(input.size(2) == output.size(1), "hidden_dim mismatch");

  TORCH_CHECK(input.is_contiguous(), "expect input to be contiguous");
  TORCH_CHECK(output.is_contiguous(), "expect output to be contiguous");

  SYCL_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16, at::ScalarType::Half, input.scalar_type(), "moe_sum_reduce_impl", [&]() {
        moe_sum_reduce_impl<scalar_t>(input, output, routed_scaling_factor);
      });
}
