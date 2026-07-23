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
  \brief Device layer for XE Sparse MLA kernels interface
*/

#pragma once

#include <c10/xpu/XPUStream.h>

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/util/sycl_event_manager.hpp"
#include "sycl/comm/common.h"
#include "sycl/kernels/mla_sparse/kernel/xe_mla_sparse_kernel.hpp"

namespace cutlass::flash_attention::device {
using namespace cute;

namespace detail {
// Stage-1 gather companion detection, structurally analogous to MLA's
// detail::ReductionTraits (mla/device/mla_runner.hpp): a kernel opts into a
// secondary "gather+dequant" pass by exposing a nested `GatherKernel` type; the
// runner then launches that companion before the main kernel. Kernels without it
// (the fused sparse MLA decode path, XeMlaSparseFwdKernel) fall through to the
// primary template and MLASparse launches ONLY the main kernel — behavior
// unchanged. This differs from MLA, which keys off a `Kernel::is_split_kv` member
// that is ALWAYS present on its kernels; MLASparse serves both the fused and
// 2-stage kernel families, so the companion must be optional (SFINAE-detected).
struct DummyGatherKernel {
  using Params = int;  // placeholder; never launched (has_gather == false)
};

template <class Kernel, class = void>
struct GatherTraits {
  static constexpr bool has_gather = false;
  using type = DummyGatherKernel;
};

template <class Kernel>
struct GatherTraits<Kernel, cute::void_t<typename Kernel::GatherKernel>> {
  static constexpr bool has_gather = true;
  using type = typename Kernel::GatherKernel;
};
}  // namespace detail

////////////////////////////////////////////////////////////////////////////////
// GrfSize selects the register-file mode passed to launch<Kernel, GrfSize>.
// Defaults to 128 (the fused sparse MLA decode path); the 2-stage dense decode
// kernel is fragment-heavy and instantiates this with GrfSize = 256 to avoid
// register spills (see device/mla_sparse_decode_2stage_types.hpp).
template <class Kernel_, int GrfSize = 128>
class MLASparse {
 public:
  //
  // Type Aliases
  //
  using Kernel = Kernel_;
  using KernelArguments = typename Kernel::KernelArguments;
  using Arguments = typename Kernel::Arguments;
  using KernelParams = typename Kernel::Params;

  // Optional Stage-1 gather+dequant companion kernel (2-stage path). Resolves to
  // DummyGatherKernel with has_gather == false for kernels that don't declare one.
  using GatherKernel = typename detail::GatherTraits<Kernel>::type;
  static constexpr bool has_gather = detail::GatherTraits<Kernel>::has_gather;
  // GRF mode for the gather companion: normal mode (128) for more parallel
  // subgroups; XE3P uses 256 (the shared launch<> helper caps at {128,256}). This
  // matches the gather stage's prior manual launch in the 2-stage launcher.
#if defined(XPU_ENABLED_XE3P)
  static constexpr int kGatherGrfSize = 256;
#else
  static constexpr int kGatherGrfSize = 128;
#endif
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
  MLASparse() = default;

  //
  // methods
  //
  Params const& params() const {
    return params_;
  }

  static cutlass::Status can_implement(Arguments const& args) {
    return Kernel_::can_implement(args) ? cutlass::Status::kSuccess : cutlass::Status::kErrorInvalidProblem;
  }

  static size_t get_workspace_size(Arguments const& args) {
    size_t workspace_bytes = 0;
    workspace_bytes += Kernel::get_workspace_size(args);
    return workspace_bytes;
  }

  static int maximum_active_blocks(int /* smem_capacity */ = -1) {
    return 0;
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
    // Stage 1: gather+dequant companion (2-stage path only). Launched first so its
    // dense gathered-KV HBM tile is materialized before the main kernel reads it;
    // the in-order XPU queue serializes the two launches. Mirrors MLA::run's
    // main-then-reduction dual launch (mla/device/mla_runner.hpp:202-213), inverted
    // (companion-then-main) because the gather feeds the attention. The gather
    // kernel's Params is the same flat SparseAttnDecodeParams as fmla_params, so it
    // reuses that params block. Compiled away entirely when has_gather == false.
    if constexpr (has_gather) {
      launch<GatherKernel, kGatherGrfSize>(params.fmla_params);
    }
    launch<Kernel, GrfSize>(params.fmla_params);

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
