#include <torch/all.h>

#include <cstdio>

#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

#include "sgl_kernel_export.h"

namespace {

struct EntryProbeFunctor {
  void operator()(sycl::nd_item<1>) const {}
};

}  // namespace

SGL_KERNEL_EXPORT void gptq_entry_probe(torch::Tensor device_index) {
  TORCH_CHECK(device_index.is_xpu(), "gptq_entry_probe requires an XPU tensor");
  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();
  queue.parallel_for(sycl::nd_range<1>(sycl::range<1>(1), sycl::range<1>(1)), EntryProbeFunctor{}).wait();
  // One line, once per process, so the hal smoke log can search for it.
  static bool logged = false;
  if (!logged) {
    std::printf("SGL_KERNEL_XPU_GPTQ_ENTRY\n");
    std::fflush(stdout);
    logged = true;
  }
}
