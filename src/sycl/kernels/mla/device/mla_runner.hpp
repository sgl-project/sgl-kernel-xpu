/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*!
  \file
  \brief Device layer for XE MLA kernels interface
*/

#pragma once

#include <c10/xpu/XPUStream.h>

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/util/sycl_event_manager.hpp"
#include "sycl/comm/common.h"
#include "sycl/kernels/mla/kernel/xe_mla_kernel.hpp"

namespace cutlass::flash_attention::device {
using namespace cute;

template <typename Kernel>
class KernelCur {};

////////////////////////////////////////////////////////////////////////////////
template <class Kernel_>
class MLA {
 public:
  //
  // Type Aliases
  //
  using Kernel = Kernel_;
  using KernelArguments = typename Kernel::KernelArguments;
  using Arguments = typename Kernel::Arguments;
  using KernelParams = typename Kernel::Params;
  //
  // Params
  //
  struct Params {
    KernelParams fmla_params;
    Params() = default;
    explicit Params(KernelParams const& kp) : fmla_params(kp) {}
  };

 private:
  //
  // data members
  //
  Params params_;
  bool initialized_ = false;

  //
  // methods
  //
  bool is_initialized(bool set = false) {
    if (set) {
      initialized_ = true;
    }
    return initialized_;
  }

 public:
  //
  // Default constructor
  //
  MLA() = default;

  //
  // methods
  //
  Params const& params() const {
    return params_;
  }

  static void set_split_kv(KernelArguments& args) {
    // TODO: set split_kv in args if needed
    assert(false && "set_split_kv not implemented yet.");
    return;
  }

  static cutlass::Status can_implement(Arguments const& args) {
    return Kernel_::can_implement(args) ? cutlass::Status::kSuccess : cutlass::Status::kErrorInvalidProblem;
    // return cutlass::Status::kSuccess;
  }

  static size_t get_workspace_size(Arguments const& args) {
    size_t workspace_bytes = 0;
    workspace_bytes += Kernel::get_workspace_size(args);
    // TODO: add Reduction workspace size when splitkv > 1
    return workspace_bytes;
  }

  static int maximum_active_blocks(int /* smem_capacity */ = -1) {
    return 0;  // change as needed
  }

  cutlass::Status initialize(
      Arguments const& args, void* workspace = nullptr, sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue()) {
    // Initialize the workspace
    CUTLASS_CHECK(Kernel::initialize_workspace(args, workspace));

    KernelParams kernel_params = Kernel::to_underlying_arguments(args, workspace);
    params_ = Params{kernel_params};

    if (is_initialized()) return Status::kSuccess;

    int smem_size = Kernel::SharedStorageSize;
    if (smem_size >= 0) {
      CUTLASS_TRACE_HOST("  Setting smem size to " << smem_size);
    }

    is_initialized(true);

    return cutlass::Status::kSuccess;
  }

  cutlass::Status update(Arguments const& args, void* workspace = nullptr) {
    size_t workspace_bytes = get_workspace_size(args);
    if (workspace_bytes > 0 && nullptr == workspace) {
      return Status::kErrorWorkspaceNull;
    }

    auto fmla_params = Kernel::to_underlying_arguments(args, workspace);
    params_ = Params{fmla_params};

    return cutlass::Status::kSuccess;
  }

  static cutlass::Status run(Params& params, sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue()) {
    launch<Kernel, 128>(params.fmla_params);

    return cutlass::Status::kSuccess;
  }

  //
  // Non-static launch overloads
  //

  /// Launches the kernel after first constructing Params internal state from supplied arguments.
  cutlass::Status
  run(Arguments const& args, void* workspace = nullptr, sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue()) {
    cutlass::Status status = initialize(args, workspace, queue);
    if (cutlass::Status::kSuccess == status) {
      status = run(params_, queue);
    }
    return status;
  }

  /// Launches the kernel after first constructing Params internal state from supplied arguments.
  cutlass::Status operator()(
      Arguments const& args, void* workspace = nullptr, sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue()) {
    return run(args, workspace, queue);
  }

  /// Overload that allows a user to re-launch the same kernel without updating internal params struct.
  cutlass::Status run(sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue()) {
    return run(params_, queue);
  }

  /// Overload that allows a user to re-launch the same kernel without updating internal params struct.
  cutlass::Status operator()(sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue()) {
    return run(params_, queue);
  }
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::flash_attention::device
