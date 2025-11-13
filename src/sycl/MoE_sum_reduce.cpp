#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"

void moe_sum_reduce(at::Tensor& input, at::Tensor& output, double routed_scaling_factor) {
  TORCH_CHECK(input.dim() == 3, "input must be a 3D tensor like [token_num, topk_num, hidden_dim]");
  TORCH_CHECK(output.dim() == 2, "output must be [token_num, hidden_dim]");
  TORCH_CHECK(input.size(0) == output.size(0), "token dim mismatch");
  TORCH_CHECK(input.size(2) == output.size(1), "hidden_dim mismatch");

  TORCH_CHECK(input.is_contiguous(), "expect input to be contiguous");
  TORCH_CHECK(output.is_contiguous(), "expect output to be contiguous");

  const int64_t token_num = input.size(0);
  const int64_t topk_num = input.size(1);
  const int64_t hidden_dim = input.size(2);

  const int64_t in_stride_token = input.stride(0);
  const int64_t in_stride_topk = input.stride(1);
  const int64_t out_stride_token = output.stride(0);

  const bool fast_bf16_vec_ok = (input.scalar_type() == at::kBFloat16) && (token_num > 256) && (hidden_dim % 8 == 0);

  // Both Fast path and few tokens
  // [toDo] per token can be reused...
  if (fast_bf16_vec_ok || !(token_num > 128)) {

    auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
    uint32_t max_wg_size = dpcppMaxWorkGroupSize(dev_id);
    uint32_t max_tokens = static_cast<uint32_t>(sycl::min(max_wg_size, hidden_dim));

    // max tokens may not work for all condtions - [toDo]
    const float scale = static_cast<float>(routed_scaling_factor);
    /* singel SYCL kernel */
#if 0
    moe_sum_reduce_per_token_kernel_sycl(
        reinterpret_cast<const at::BFloat16*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<at::BFloat16*>(output.data_ptr<at::BFloat16>()),
        token_num,
        hidden_dim,
        topk_num,
        in_stride_token,
        in_stride_topk,
        out_stride_token,
        scale,
        max_tokens);
#endif
    return;
  }
}

