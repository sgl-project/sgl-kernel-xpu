#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include "sgl_kernel/xe35/logits_vocab_ops.h"

#include <algorithm>
#include <sycl/sycl.hpp>

#include "../MemoryAccess.h"
#include "../SYCLHelpers.h"

namespace at::native::xpu {
namespace {

constexpr int kSoftcapWorkgroup = 128;

template <typename Scalar, bool Contiguous>
struct SoftcapLogitsKernel {
  Scalar* __restrict__ logits;
  int64_t nrows;
  int64_t ncols;
  int64_t row_stride;
  float cap;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t index = item.get_global_id(0);
    if (index >= nrows * ncols) {
      return;
    }
    int64_t address;
    if constexpr (Contiguous) {
      address = index;
    } else {
      const int64_t row = index / ncols;
      const int64_t col = index % ncols;
      address = row * row_stride + col;
    }
    const float value = static_cast<float>(logits[address]);
    logits[address] = static_cast<Scalar>(sycl::tanh(value / cap) * cap);
  }
};

struct SoftcapLogitsFloat4Kernel {
  float* __restrict__ logits;
  int64_t numel;
  float cap;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t index = item.get_global_id(0) * 4;
    if (index >= numel) {
      return;
    }
    using Vec = aligned_vector_loop<float, 4>;
    const Vec x = *reinterpret_cast<const Vec*>(logits + index);
    Vec y;
    for (int i = 0; i < 4; ++i) {
      y[i] = sycl::tanh(x[i] / cap) * cap;
    }
    *reinterpret_cast<Vec*>(logits + index) = y;
  }
};

}  // namespace

void softcap_logits_xe35(torch::Tensor logits, double cap) {
  TORCH_CHECK(logits.is_xpu(), "logits must be on XPU");
  TORCH_CHECK(
      logits.scalar_type() == at::kFloat || logits.scalar_type() == at::kBFloat16 || logits.scalar_type() == at::kHalf,
      "logits must be float32, bfloat16, or float16");
  TORCH_CHECK(
      logits.is_contiguous() || (logits.dim() == 2 && logits.stride(1) == 1),
      "logits must be contiguous or a matrix with contiguous columns");
  TORCH_CHECK(cap > 0.0, "cap must be positive");
  if (logits.numel() == 0) {
    return;
  }
  const bool contiguous = logits.is_contiguous();
  const int64_t nrows = contiguous ? 1 : logits.size(0);
  const int64_t ncols = contiguous ? logits.numel() : logits.size(1);
  const int64_t row_stride = contiguous ? ncols : logits.stride(0);
  auto queue = at::xpu::getCurrentXPUStream().queue();
  const auto global_size = ((logits.numel() + kSoftcapWorkgroup - 1) / kSoftcapWorkgroup) * kSoftcapWorkgroup;
  if (logits.scalar_type() == at::kFloat) {
    if (contiguous && logits.numel() % 4 == 0 && reinterpret_cast<uintptr_t>(logits.data_ptr()) % 16 == 0) {
      const auto vector_groups = (logits.numel() / 4 + kSoftcapWorkgroup - 1) / kSoftcapWorkgroup;
      SoftcapLogitsFloat4Kernel kernel{logits.data_ptr<float>(), logits.numel(), static_cast<float>(cap)};
      sycl_kernel_submit(
          sycl::range<1>(vector_groups * kSoftcapWorkgroup), sycl::range<1>(kSoftcapWorkgroup), queue, kernel);
    } else if (contiguous) {
      SoftcapLogitsKernel<float, true> kernel{
          logits.data_ptr<float>(), nrows, ncols, row_stride, static_cast<float>(cap)};
      sycl_kernel_submit(sycl::range<1>(global_size), sycl::range<1>(kSoftcapWorkgroup), queue, kernel);
    } else {
      SoftcapLogitsKernel<float, false> kernel{
          logits.data_ptr<float>(), nrows, ncols, row_stride, static_cast<float>(cap)};
      sycl_kernel_submit(sycl::range<1>(global_size), sycl::range<1>(kSoftcapWorkgroup), queue, kernel);
    }
  } else if (logits.scalar_type() == at::kBFloat16) {
    if (contiguous) {
      SoftcapLogitsKernel<at::BFloat16, true> kernel{
          logits.data_ptr<at::BFloat16>(), nrows, ncols, row_stride, static_cast<float>(cap)};
      sycl_kernel_submit(sycl::range<1>(global_size), sycl::range<1>(kSoftcapWorkgroup), queue, kernel);
    } else {
      SoftcapLogitsKernel<at::BFloat16, false> kernel{
          logits.data_ptr<at::BFloat16>(), nrows, ncols, row_stride, static_cast<float>(cap)};
      sycl_kernel_submit(sycl::range<1>(global_size), sycl::range<1>(kSoftcapWorkgroup), queue, kernel);
    }
  } else {
    if (contiguous) {
      SoftcapLogitsKernel<at::Half, true> kernel{
          logits.data_ptr<at::Half>(), nrows, ncols, row_stride, static_cast<float>(cap)};
      sycl_kernel_submit(sycl::range<1>(global_size), sycl::range<1>(kSoftcapWorkgroup), queue, kernel);
    } else {
      SoftcapLogitsKernel<at::Half, false> kernel{
          logits.data_ptr<at::Half>(), nrows, ncols, row_stride, static_cast<float>(cap)};
      sycl_kernel_submit(sycl::range<1>(global_size), sycl::range<1>(kSoftcapWorkgroup), queue, kernel);
    }
  }
}

}  // namespace at::native::xpu
