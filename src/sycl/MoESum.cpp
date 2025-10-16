#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"

template <typename scalar_t, int TOPK>
struct MoeSumKernel {
  MoeSumKernel(scalar_t* out_, const scalar_t* input_, int hidden_size_)
      : out(out_), input(input_), hidden_size(hidden_size_) {}

  void operator()(sycl::nd_item<1> item) const {
    int64_t global_idx = item.get_global_id(0);
    int64_t token_idx = global_idx / hidden_size;
    int idx = global_idx % hidden_size;

    scalar_t x = 0.0;
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      x += input[token_idx * 2 * hidden_size + k * hidden_size + idx];
    }
    out[token_idx * hidden_size + idx] = x;
  }

  scalar_t* out;
  const scalar_t* input;
  int hidden_size;
};

void moe_sum(
    torch::Tensor& input,   // [num_tokens, topk, hidden_size]
    torch::Tensor& output)  // [num_tokens, hidden_size]
{
  const int hidden_size = input.size(-1);
  const auto num_tokens = output.numel() / hidden_size;
  const int topk = input.size(1);

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();
  sycl::range<1> global(num_tokens);
  sycl::range<1> local(std::min(hidden_size, 1024));
  auto range = sycl::nd_range<1>(global * local, local);

  switch (topk) {
    case 2: {
      DISPATCH_FLOAT_TYPES(input.scalar_type(), "moe_sum", [&] {
        using Kernel = MoeSumKernel<scalar_t, 2>;
        Kernel kernel(output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), hidden_size);
        sycl_kernel_submit(range.get_global_range(), range.get_local_range(), queue, kernel);
      });
      break;
    }
    case 3: {
      DISPATCH_FLOAT_TYPES(input.scalar_type(), "moe_sum", [&] {
        using Kernel = MoeSumKernel<scalar_t, 3>;
        Kernel kernel(output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), hidden_size);
        sycl_kernel_submit(range.get_global_range(), range.get_local_range(), queue, kernel);
      });
      break;
    }
    case 4: {
      DISPATCH_FLOAT_TYPES(input.scalar_type(), "moe_sum", [&] {
        using Kernel = MoeSumKernel<scalar_t, 4>;
        Kernel kernel(output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), hidden_size);
        sycl_kernel_submit(range.get_global_range(), range.get_local_range(), queue, kernel);
      });
      break;
    }
    default:
      at::sum_out(output, input, 1);
      break;
  }
}
