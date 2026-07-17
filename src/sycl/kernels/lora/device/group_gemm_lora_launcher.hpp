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
  \brief Shared CUTLASS-lifecycle launcher for the LoRA pointer-array grouped GEMM.

  GroupGemmLoraFwd<Kernel_> is a thin wrapper around
  cutlass::gemm::device::GemmUniversalAdapter: it takes an already-built
  GemmKernel::Arguments and drives the lifecycle
  (can_implement / get_workspace_size / initialize / run). It holds NO knowledge
  of how those Arguments were built -- that lives in args_from_options() in
  common/group_gemm_types.hpp -- so this class is kernel-agnostic and reused
  verbatim across the LoRA forward grouped GEMMs (A-fwd, B-fwd, ...): a caller
  reuses it by handing it its own GemmKernel (via GroupGemmTypes::GemmKernel)
  and its own Arguments (with the alpha/beta and residual C pointers that its
  epilogue needs -- e.g. beta=0 for A-fwd, beta=1 in-place residual for B-fwd).

  Kernel_ is the fully-configured cutlass::gemm::kernel::GemmUniversal type
  produced by GroupGemmTypes<...>::GemmKernel.
*/

#pragma once

#include <c10/xpu/XPUStream.h>

#include <cstddef>
#include <sycl/sycl.hpp>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

namespace at::native::xpu {

////////////////////////////////////////////////////////////////////////////////
/// Thin lifecycle wrapper around cutlass::gemm::device::GemmUniversalAdapter.
///
/// Public surface (mirrors class MLA<> in mla_runner.hpp):
///   - can_implement(args)          -- static; kernel-level feasibility check
///   - get_workspace_size(args)     -- static; bytes to allocate for workspace
///   - initialize(args, ws, queue)  -- build Params + workspace layout
///   - run(queue)                   -- launch the kernel (initialize first)
///   - run(args, ws, queue)         -- initialize + run in one shot
///   - operator()(args, ws, queue)  -- alias for run(args, ws, queue)
template <class Kernel_>
class GroupGemmLoraFwd {
 public:
  //
  // Type Aliases
  //
  using Kernel = Kernel_;
  using Adapter = cutlass::gemm::device::GemmUniversalAdapter<Kernel_>;
  using Arguments = typename Adapter::Arguments;
  using Params = typename Adapter::Params;

 private:
  //
  // data members
  //
  Adapter adapter_;
  bool initialized_ = false;

 public:
  //
  // Default constructor
  //
  GroupGemmLoraFwd() = default;

  //
  // methods
  //
  Params const& params() const {
    return adapter_.params();
  }

  /// Static feasibility check (delegates to the underlying grouped GEMM kernel).
  static cutlass::Status can_implement(Arguments const& args) {
    return Adapter::can_implement(args);
  }

  /// Bytes required for the CUTLASS-owned workspace (grouped scheduler + any
  /// internal buffers). The caller must allocate this many bytes on-device
  /// before initialize().
  static size_t get_workspace_size(Arguments const& args) {
    return Adapter::get_workspace_size(args);
  }

  /// Build the Params structure and initialize the workspace. On success the
  /// instance is armed to be launched via run(queue).
  cutlass::Status initialize(
      Arguments const& args, void* workspace = nullptr, sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue()) {
    cutlass::Status status = adapter_.initialize(args, workspace, &queue);
    if (status == cutlass::Status::kSuccess) {
      initialized_ = true;
    }
    return status;
  }

  /// Re-initialize with new arguments while keeping the same workspace
  /// allocation. Used to relaunch with a different (M_s, ptr_A/B/D) group
  /// batch without re-allocating the CUTLASS workspace tensor.
  cutlass::Status update(Arguments const& args, void* workspace = nullptr) {
    return adapter_.update(args, workspace);
  }

  /// Launch the kernel on `queue`. Requires initialize() to have run first.
  /// PyTorch XPU streams are in-order (c10 asserts "External SYCL queue must
  /// be in-order"), which is what the grouped scheduler assumes.
  cutlass::Status run(sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue()) {
    return adapter_.run(&queue);
  }

  /// One-shot initialize + run (matches class MLA<>::run(args, ...)).
  cutlass::Status
  run(Arguments const& args, void* workspace = nullptr, sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue()) {
    cutlass::Status status = initialize(args, workspace, queue);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }
    return run(queue);
  }

  /// Callable alias for run(args, workspace, queue).
  cutlass::Status operator()(
      Arguments const& args, void* workspace = nullptr, sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue()) {
    return run(args, workspace, queue);
  }
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace at::native::xpu
