/***************************************************************************************************
 * Copyright 2025 SGLang Team. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 **************************************************************************************************/
/*! \file
    \brief Device Runner for Row-wise Square Sum
*/

#pragma once

#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "sycl/kernels/reduction/kernel/xe_row_square_sum_kernel.hpp"

namespace cutlass::reduction::device {
using namespace cute;

///////////////////////////////////////////////////////////////////////////////
/// RowSquareSum Device Adapter: manages kernel launch and lifecycle
///
/// Template Parameters:
///   Kernel_: The kernel type (XeRowSquareSumKernel)
///////////////////////////////////////////////////////////////////////////////
template <class Kernel_>
class RowSquareSum {
 public:
  using Kernel = Kernel_;
  using Arguments = typename Kernel::Arguments;
  using Params = typename Kernel::Params;
  using ProblemShape = typename Kernel::ProblemShape;

 private:
  Params params_;
  bool initialized_ = false;

 public:
  RowSquareSum() = default;

  ///////////////////////////////////////////////////////////////////////////////
  /// Accessors
  ///////////////////////////////////////////////////////////////////////////////
  Params const& params() const {
    return params_;
  }

  ///////////////////////////////////////////////////////////////////////////////
  /// can_implement: validate problem
  ///////////////////////////////////////////////////////////////////////////////
  static cutlass::Status can_implement(Arguments const& args) {
    if (!Kernel::can_implement(args)) {
      return cutlass::Status::kErrorInvalidProblem;
    }
    return cutlass::Status::kSuccess;
  }

  ///////////////////////////////////////////////////////////////////////////////
  /// get_workspace_size: calculate workspace requirements
  ///////////////////////////////////////////////////////////////////////////////
  static size_t get_workspace_size(Arguments const& args) {
    return Kernel::get_workspace_size(args);
  }

  ///////////////////////////////////////////////////////////////////////////////
  /// initialize: prepare kernel parameters
  ///////////////////////////////////////////////////////////////////////////////
  cutlass::Status initialize(
      Arguments const& args,
      void* /* workspace */ = nullptr,
      sycl::queue& /* queue */ = c10::xpu::getCurrentXPUStream().queue()) {

    // Validate
    cutlass::Status status = can_implement(args);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }

    // Convert arguments to params
    params_ = Kernel::to_underlying_arguments(args);
    initialized_ = true;

    return cutlass::Status::kSuccess;
  }

  ///////////////////////////////////////////////////////////////////////////////
  /// run: execute the kernel
  ///////////////////////////////////////////////////////////////////////////////
  cutlass::Status run(sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue()) {
    if (!initialized_) {
      return cutlass::Status::kErrorInternal;
    }

    const int M = params_.shape.M;
    constexpr int kThreadsPerRow = 256;

    // Launch configuration: M work-groups, 256 threads each
    sycl::range<1> global(M * kThreadsPerRow);
    sycl::range<1> local(kThreadsPerRow);

    // Launch kernel using CUTLASS device_kernel utilities
    // Capture params by value to avoid 'this' capture issues
    Params params_copy = params_;

    queue.submit([&](sycl::handler& cgh) {
      // Allocate shared memory
      sycl::local_accessor<typename Kernel::SharedStorage, 0> shared_storage(cgh);

      // Launch kernel - capture params_copy by value
      cgh.parallel_for<Kernel>(
          sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> item) {
            Kernel kernel_instance;
            auto* shared_ptr = shared_storage.template get_multi_ptr<sycl::access::decorated::no>().get();
            kernel_instance(params_copy, *shared_ptr, item);
          });
    });

    return cutlass::Status::kSuccess;
  }

  ///////////////////////////////////////////////////////////////////////////////
  /// operator(): combined initialize + run
  ///////////////////////////////////////////////////////////////////////////////
  cutlass::Status operator()(Arguments const& args, void* workspace = nullptr, sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue()) {
    cutlass::Status status = initialize(args, workspace, queue);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }
    return run(queue);
  }
};

}  // namespace cutlass::reduction::device
