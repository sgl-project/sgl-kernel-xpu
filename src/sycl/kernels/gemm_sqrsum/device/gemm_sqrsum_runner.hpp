#pragma once

#include <c10/xpu/XPUStream.h>

#include <sycl/sycl.hpp>

#include "cutlass/cutlass.h"
#include "sycl/comm/common.h"

namespace cutlass::gemm_sqrsum::device {
using namespace cute;

template <class Kernel_>
class GemmSqrSum {
 public:
  using Kernel = Kernel_;
  using Arguments = typename Kernel::Arguments;
  using Params = typename Kernel::Params;

 private:
  Params params_;
  bool initialized_ = false;

 public:
  GemmSqrSum() = default;

  Params const& params() const {
    return params_;
  }

  static cutlass::Status can_implement(Arguments const& args) {
    return Kernel::can_implement(args) ? cutlass::Status::kSuccess : cutlass::Status::kErrorInvalidProblem;
  }

  static size_t get_workspace_size(Arguments const& args) {
    return Kernel::get_workspace_size(args);
  }

  cutlass::Status initialize(
      Arguments const& args, void* workspace = nullptr, sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue()) {
    params_ = Kernel::to_underlying_arguments(args, workspace);
    initialized_ = true;
    return cutlass::Status::kSuccess;
  }

  cutlass::Status update(Arguments const& args, void* workspace = nullptr) {
    return initialize(args, workspace);
  }

  static cutlass::Status run(Params& params, sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue()) {
    launch<Kernel, 256>(params);
    return cutlass::Status::kSuccess;
  }

  cutlass::Status
  run(Arguments const& args, void* workspace = nullptr, sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue()) {
    cutlass::Status status = initialize(args, workspace, queue);
    if (cutlass::Status::kSuccess == status) {
      status = run(params_, queue);
    }
    return status;
  }

  cutlass::Status operator()(
      Arguments const& args, void* workspace = nullptr, sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue()) {
    return run(args, workspace, queue);
  }

  cutlass::Status run(sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue()) {
    return run(params_, queue);
  }

  cutlass::Status operator()(sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue()) {
    return run(params_, queue);
  }
};

}  // namespace cutlass::gemm_sqrsum::device
