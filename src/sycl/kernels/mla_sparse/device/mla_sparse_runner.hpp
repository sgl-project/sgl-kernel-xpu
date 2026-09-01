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
// Placeholder Stage-1 companion for kernels that have none (the fused sparse MLA
// decode path, XeMlaSparseFwdKernel): has_gather is false, so it is never launched
// and only its empty Arguments / Params types are ever instantiated. Structural
// analog of MLA's DummyReductionKernel (mla/kernel/xe_mla_reduce_split_kv.hpp).
struct DummyGatherKernel {
  struct Arguments {};
  struct Params {};
  static bool can_implement(Arguments const&) {
    return true;
  }
  static int get_workspace_size(Arguments const&) {
    return 0;
  }
  static Params to_underlying_arguments(Arguments const&, void*) {
    return {};
  }
};

// Placeholder split-K reduction companion for kernels that have none (the fused path and
// the non-split two-stage path): has_reduce is false, so it is never launched and only its
// empty Arguments / Params types are ever instantiated. Kept here rather than in
// kernel/xe_mla_sparse_2stage_reduce_split_kv.hpp so this runner -- which the fused path
// also uses -- does not have to include the 2-stage split-K header just to name "no
// companion". Same role as MLA's DummyReductionKernel.
struct DummyReduceKernel {
  struct Arguments {};
  struct Params {};
  static bool can_implement(Arguments const&) {
    return true;
  }
  static int get_workspace_size(Arguments const&) {
    return 0;
  }
  static Params to_underlying_arguments(Arguments const&, void*) {
    return {};
  }
  static cutlass::Status initialize_workspace(Arguments const&, void* = nullptr) {
    return cutlass::Status::kSuccess;
  }
};

// User-facing Arguments for a main kernel that HAS a Stage-1 gather companion. The
// two stages are independent kernels with independent Params (neither type derives
// from the other), so the runner carries one argument object per stage.
//
// This is the one place MLASparse must diverge from MLA: MLA's split-KV reduction
// arguments are a subset of the main kernel's Arguments (O / O_accum / exp_sums /
// max_logits), so MLA::initialize derives them from `args.kernel`. The gather stage
// instead reads tensors the dense kernel never sees (the paged fp8 KV pools, the
// index tensors), so its arguments cannot be derived from the dense ones and travel
// alongside them here.
template <class DenseArgs, class GatherArgs>
struct TwoStageArguments {
  DenseArgs dense{};
  GatherArgs gather{};
};
}  // namespace detail

////////////////////////////////////////////////////////////////////////////////
// GrfSize selects the register-file mode passed to launch<Kernel, GrfSize>.
// Defaults to 128 (the fused sparse MLA decode path); the 2-stage dense decode
// kernel is fragment-heavy and instantiates this with GrfSize = 256 to avoid
// register spills (see device/mla_sparse_decode_2stage_types.hpp).
//
// GatherKernel_ is the optional Stage-1 gather+dequant companion, wired exactly the
// way MLA wires its split-KV reduction companion (mla/device/mla_runner.hpp): the
// runner owns one Params member per stage, builds both in initialize(), and issues
// both launches from run() on the in-order XPU queue. Two differences, both forced
// by the fact that the gather feeds the attention rather than consuming it:
//   - order is companion-then-main (MLA is main-then-companion),
//   - the companion is a template parameter instead of being derived from the main
//     kernel (MLA's ReductionTraits keys off Kernel::is_split_kv and reads the
//     reduction's tensors out of the main kernel's arguments; the gather's tensors --
//     paged fp8 pools, index arrays -- are ones the dense kernel never sees, so the
//     two kernels are fully independent and the config struct names both).
// Left at the default, has_gather is false and this launches only the main kernel,
// which is what the fused path (XeMlaSparseFwdKernel) needs.
//
// ReduceKernel_ is the optional split-K reduction companion, wired the same way as the
// gather but on the *other* side of the main kernel: the launch order becomes
// gather -> dense -> reduce. It is only meaningful for a Stage-2 kernel built with the
// split-K epilogue, which publishes per-split partials instead of the finished row (see
// kernel/xe_mla_sparse_2stage_reduce_split_kv.hpp). Unlike MLA -- whose reduction
// arguments are a strict subset of the main kernel's and are copied field by field in
// MLA::initialize -- the sparse reduction's Arguments simply *are* the dense kernel's
// (every tensor it needs is already in SparseAttn2StageParams), so it is derived from
// dense_args() and adds nothing to this runner's Arguments type. Left at the default,
// has_reduce is false and nothing about the existing paths changes.
template <
    class Kernel_,
    int GrfSize = 128,
    class GatherKernel_ = detail::DummyGatherKernel,
    class ReduceKernel_ = detail::DummyReduceKernel>
class MLASparse {
 public:
  //
  // Type Aliases
  //
  using Kernel = Kernel_;
  using KernelArguments = typename Kernel::KernelArguments;
  using KernelParams = typename Kernel::Params;
  using DenseArguments = typename Kernel::Arguments;

  using GatherKernel = GatherKernel_;
  static constexpr bool has_gather = !cute::is_same_v<GatherKernel, detail::DummyGatherKernel>;
  using GatherArguments = typename GatherKernel::Arguments;
  using GatherParams = typename GatherKernel::Params;

  using ReduceKernel = ReduceKernel_;
  static constexpr bool has_reduce = !cute::is_same_v<ReduceKernel, detail::DummyReduceKernel>;
  using ReduceParams = typename ReduceKernel::Params;

  // Fused path: Arguments are just the main kernel's, unchanged. 2-stage path: the
  // two stages' independent argument objects side by side.
  using Arguments =
      cute::conditional_t<has_gather, detail::TwoStageArguments<DenseArguments, GatherArguments>, DenseArguments>;

  // GRF mode for the gather companion: normal mode (128) for more parallel
  // subgroups; XE3P uses 256 (the shared launch<> helper caps at {128,256}).
#if defined(XPU_ENABLED_XE3P)
  static constexpr int kGatherGrfSize = 256;
#else
  static constexpr int kGatherGrfSize = 128;
#endif
  //
  // Params: one per stage, mirroring MLA::Params{fmla_params, reduction_params}.
  //
  struct Params {
    KernelParams fmla_params;
    GatherParams gather_params;
    ReduceParams reduce_params;
  };

  static constexpr int kReduceGrfSize = 128;

  static DenseArguments const& dense_args(Arguments const& args) {
    if constexpr (has_gather) {
      return args.dense;
    } else {
      return args;
    }
  }

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
    if (!Kernel::can_implement(dense_args(args))) return cutlass::Status::kErrorInvalidProblem;
    if constexpr (has_gather) {
      if (!GatherKernel::can_implement(args.gather)) return cutlass::Status::kErrorInvalidProblem;
    }
    if constexpr (has_reduce) {
      if (!ReduceKernel::can_implement(dense_args(args))) return cutlass::Status::kErrorInvalidProblem;
    }
    return cutlass::Status::kSuccess;
  }

  static size_t get_workspace_size(Arguments const& args) {
    size_t workspace_bytes = 0;
    workspace_bytes += Kernel::get_workspace_size(dense_args(args));
    if constexpr (has_gather) {
      workspace_bytes += GatherKernel::get_workspace_size(args.gather);
    }
    if constexpr (has_reduce) {
      workspace_bytes += ReduceKernel::get_workspace_size(dense_args(args));
    }
    return workspace_bytes;
  }

  static int maximum_active_blocks(int /* smem_capacity */ = -1) {
    return 0;
  }

  cutlass::Status initialize(
      Arguments const& args, void* workspace = nullptr, sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue()) {
    // Initialize the workspace
    CUTLASS_CHECK(Kernel::initialize_workspace(dense_args(args), workspace));

    params_.fmla_params = Kernel::to_underlying_arguments(dense_args(args), workspace);
    if constexpr (has_gather) {
      params_.gather_params = GatherKernel::to_underlying_arguments(args.gather, workspace);
      CUTLASS_CHECK(GatherKernel::initialize_workspace(args.gather, workspace));
    }
    if constexpr (has_reduce) {
      params_.reduce_params = ReduceKernel::to_underlying_arguments(dense_args(args), workspace);
      CUTLASS_CHECK(ReduceKernel::initialize_workspace(dense_args(args), workspace));
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
    if constexpr (!has_gather) {
      launch<Kernel, GrfSize>(params.fmla_params);
    } else {
      launch<GatherKernel, kGatherGrfSize>(params.gather_params);
      launch<Kernel, GrfSize>(params.fmla_params);
    }

    if constexpr (has_reduce) {
      launch<ReduceKernel, kReduceGrfSize>(params.reduce_params);
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
