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

#include <cmath>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/util/sycl_event_manager.hpp"
#include "sycl/comm/common.h"
#include "sycl/kernels/mla/kernel/xe_mla_kernel.hpp"
#include "sycl/kernels/mla/kernel/xe_mla_reduce_split_kv.hpp"

namespace cutlass::flash_attention::device {
using namespace cute;

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

  using ProblemShape = typename Kernel::ProblemShape;
  using ReductionScheduler = cutlass::flash_attention::kernel::XeMlaReduceSplitKScheduler;
  using ReductionKernel =
      cutlass::flash_attention::kernel::XeMlaReduceSplitKV<ProblemShape, ReductionScheduler, Kernel>;
  using ReductionArguments = typename ReductionKernel::Arguments;
  using ReductionParams = typename ReductionKernel::Params;

  struct Params {
    KernelParams fmla_params;
    ReductionParams reduction_params;
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

  static cutlass::Status can_implement(Arguments const& args) {
    if (!Kernel_::can_implement(args)) return cutlass::Status::kErrorInvalidProblem;
    if constexpr (Kernel::is_split_kv) {
      ReductionArguments reduce_args{};
      reduce_args.kernel.shape = args.kernel.shape;
      reduce_args.num_kv_splits = args.split_kv;
      if (!ReductionKernel::can_implement(reduce_args)) return cutlass::Status::kErrorInvalidProblem;
    }
    return cutlass::Status::kSuccess;
  }

  static size_t get_workspace_size(Arguments const& args) {
    size_t workspace_bytes = 0;
    workspace_bytes += Kernel::get_workspace_size(args);
    if constexpr (Kernel::is_split_kv) {
      ReductionArguments reduce_args{};
      reduce_args.num_kv_splits = args.split_kv;
      workspace_bytes += ReductionKernel::get_workspace_size(reduce_args);
    }
    return workspace_bytes;
  }

  static int maximum_active_blocks(int /* smem_capacity */ = -1) {
    return 0;  // change as needed
  }

  cutlass::Status initialize(
      Arguments const& args, void* workspace = nullptr, sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue()) {
    int split_kv = args.split_kv;
    if constexpr (!Kernel::is_split_kv) {
      // Non-split path
      CUTLASS_CHECK(Kernel::initialize_workspace(args, workspace));
      params_.fmla_params = Kernel::to_underlying_arguments(args, workspace);
    } else {
      // Split-KV path: compute workspace layout and set up both kernel params
      auto const& s = args.kernel.shape;
      // Construct split-KV main kernel arguments
      params_.fmla_params = Kernel::to_underlying_arguments(args, workspace);

      using StrideO = typename Kernel::StrideO;

      typename ReductionKernel::KernelArguments reduce_kernel_args{};
      reduce_kernel_args.shape = s;
      reduce_kernel_args.O = args.kernel.O;
      reduce_kernel_args.dO = args.kernel.dO;
      reduce_kernel_args.O_accum = args.kernel.O_accum;
      reduce_kernel_args.dO_accum = args.kernel.dO_accum;
      reduce_kernel_args.exp_sums = args.kernel.exp_sums;
      reduce_kernel_args.max_logits = args.kernel.max_logits;
      reduce_kernel_args.dLSE = args.kernel.dLSE;

      ReductionArguments reduce_args{reduce_kernel_args, args.hw_info, split_kv};
      params_.reduction_params = ReductionKernel::to_underlying_arguments(reduce_args, workspace);
    }

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
    return initialize(args, workspace);
  }

  static cutlass::Status run(Params& params, sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue()) {
    if constexpr (!Kernel::is_split_kv) {
      // Non-split: launch main kernel only
      launch<Kernel, 128>(params.fmla_params);
    } else {
      // Split-KV: launch split attention kernel + reduction kernel
      launch<Kernel, 128>(params.fmla_params);
      launch<ReductionKernel, 128>(params.reduction_params);
    }

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
