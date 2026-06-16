#define SYCL_INTEL_TARGET 20
#define VLLM_XPU_ENABLE_XE2
#include "l2norm.h"

#include <torch/all.h>

#include <sycl/sycl.hpp>

#include "l2norm_kernel.hpp"

void l2norm(sycl::queue& queue, const torch::Tensor& q, const torch::Tensor& k) {
  gdn::l2norm_impl(queue, q, k);
}
