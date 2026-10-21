/* Copyright (c) 2025 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Header-only form of the vllm-xpu-kernels Xe3 grouped-GEMM launcher
 * (csrc/xpu/grouped_gemm/xe_3/grouped_gemm_kernel.cpp @ 48313ef). The upstream
 * file is a .cpp with an explicit INSTANTIATE_KERNEL list for all 38 dtype
 * policies because the dispatch lives in a separate translation unit; here the
 * dispatch and the launcher share one TU, so kernel_functor_xe35<> is instantiated
 * implicitly and only the policies actually referenced are code-generated.
 *
 * Behavioural changes from upstream: a cutlass error raises a torch exception
 * instead of calling exit(EXIT_FAILURE), and a policy that asks for a scheduler
 * workspace is rejected instead of silently getting an unsafe one (see run()).
 */
#pragma once

#include <ATen/ATen.h>

#include "sycl/xe35/moe/collective/moe_dtype_policy.hpp"

namespace sgl_kernel::vllm_grouped_gemm_xe35 {

using namespace cute;

#define SGL_GG_CUTLASS_CHECK(status)                                                                                 \
  {                                                                                                                  \
    cutlass::Status error = (status);                                                                                \
    TORCH_CHECK(error == cutlass::Status::kSuccess, "cutlass grouped GEMM failed: ", cutlassGetStatusString(error)); \
  }

template <typename Gemm, bool NeedScale>
struct ElementScaleSelector_xe35 {
  using A = void;
  using B = void;
};

template <typename Gemm>
struct ElementScaleSelector_xe35<Gemm, true> {
  using A = typename Gemm::CollectiveMainloop::ElementScaleA;
  using B = typename Gemm::CollectiveMainloop::ElementScaleB;
};

template <class Gemm, bool NeedScale>
struct GroupedGemmRunner_xe35 {
  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementC = typename Gemm::ElementC;

  using CollectiveMainloop = typename Gemm::CollectiveMainloop;
  using CollectiveEpilogue = typename Gemm::CollectiveEpilogue;
  using ElementOutput = typename CollectiveEpilogue::ElementOutput;
  using ElementAccumulator = typename Gemm::ElementAccumulator;

  using ElementScaleA = typename ElementScaleSelector_xe35<Gemm, NeedScale>::A;
  using ElementScaleB = typename ElementScaleSelector_xe35<Gemm, NeedScale>::B;

  typename Gemm::Arguments args_from_mainloop_options(
      const cutlass::KernelHardwareInfo& hw_info,
      int const* rows_per_expert,
      typename CollectiveMainloop::Arguments const& mainloop_args,
      const ElementC* ptr_C,
      ElementOutput* ptr_D,
      int64_t N,
      int64_t K,
      int64_t groups) {
    typename Gemm::Arguments arguments;
    decltype(arguments.epilogue.thread) fusion_args;

    fusion_args.alpha = 1;
    fusion_args.beta = ptr_C ? 1 : 0;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr = nullptr;
    fusion_args.alpha_ptr_array = nullptr;
    fusion_args.beta_ptr_array = nullptr;
    // One alpha and beta per each group
    fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
    fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};
    using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerMoE_xe35::RasterOrderOptions;

    // Per-GEMM problem shape info may only exist on the device.
    arguments = typename Gemm::Arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        mainloop_args,
        {fusion_args, ptr_C, ptr_D},
        rows_per_expert,
        N,
        K,
        groups,
        hw_info,
        {1, RasterOrderOptions::AlongN}};
    return arguments;
  }

  typename Gemm::Arguments args_from_options(
      const cutlass::KernelHardwareInfo& hw_info,
      int const* rows_per_expert,
      const ElementA* ptr_A,
      const ElementScaleA* ptr_A_scale,
      const ElementB* ptr_B,
      const ElementScaleB* ptr_B_scale,
      const ElementC* ptr_C,
      ElementOutput* ptr_D,
      int64_t N,
      int64_t K,
      int64_t groups,
      int block_size) {
    if constexpr (!NeedScale) {
      return args_from_mainloop_options(
          hw_info, rows_per_expert, typename CollectiveMainloop::Arguments{ptr_A, ptr_B}, ptr_C, ptr_D, N, K, groups);
    } else {
      return args_from_mainloop_options(
          hw_info,
          rows_per_expert,
          typename CollectiveMainloop::Arguments{ptr_A, ptr_B, ptr_A_scale, ptr_B_scale, block_size},
          ptr_C,
          ptr_D,
          N,
          K,
          groups);
    }
  }

  cutlass::Status run_with_mainloop(
      sycl::queue& stream,
      const cutlass::KernelHardwareInfo& hw_info,
      int const* rows_per_expert,
      typename CollectiveMainloop::Arguments const& mainloop_args,
      const ElementC* ptr_C,
      ElementOutput* ptr_D,
      int64_t N,
      int64_t K,
      int64_t groups) {
    Gemm gemm_op;

    auto arguments = args_from_mainloop_options(hw_info, rows_per_expert, mainloop_args, ptr_C, ptr_D, N, K, groups);

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    // PersistentTileSchedulerMoE_xe35 asks for no workspace, and this launcher depends on
    // that: `allocation` comes from syclcompat's default queue rather than the caller's
    // torch stream, and its RAII free runs as soon as `run()` returns -- i.e. while the
    // kernel it belongs to may still be executing. A non-zero workspace here would be
    // a use-after-free plus a cross-queue allocation, which is the hazard the FMHA path
    // was fixed for (see flash_attentionXe35_common.hpp). Wire it to the torch stream
    // allocator before enabling any policy that needs one.
    TORCH_CHECK(
        workspace_size == 0,
        "moe grouped GEMM: this policy wants a ",
        workspace_size,
        "-byte workspace, which this launcher cannot allocate safely on the caller's stream");
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    SGL_GG_CUTLASS_CHECK(gemm_op.can_implement(arguments));
    SGL_GG_CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get(), &stream));
    SGL_GG_CUTLASS_CHECK(gemm_op.run(&stream));
    return cutlass::Status::kSuccess;
  }

  cutlass::Status
  run(sycl::queue& stream,
      const cutlass::KernelHardwareInfo& hw_info,
      int const* rows_per_expert,
      const ElementA* ptr_A,
      const ElementScaleA* ptr_A_scale,
      const ElementB* ptr_B,
      const ElementScaleB* ptr_B_scale,
      const ElementC* ptr_C,
      ElementOutput* ptr_D,
      int64_t N,
      int64_t K,
      int64_t groups,
      int block_size) {
    if constexpr (!NeedScale) {
      return run_with_mainloop(
          stream,
          hw_info,
          rows_per_expert,
          typename CollectiveMainloop::Arguments{ptr_A, ptr_B},
          ptr_C,
          ptr_D,
          N,
          K,
          groups);
    } else {
      return run_with_mainloop(
          stream,
          hw_info,
          rows_per_expert,
          typename CollectiveMainloop::Arguments{ptr_A, ptr_B, ptr_A_scale, ptr_B_scale, block_size},
          ptr_C,
          ptr_D,
          N,
          K,
          groups);
    }
  }
};

template <class moe_policy>
void kernel_functor_xe35(
    sycl::queue& stream,
    void* ptr_A,
    void* ptr_A_scale,
    void* ptr_B,
    void* ptr_B_scale,
    void* ptr_bias,
    void* ptr_D,
    void* rows_per_expert,
    int64_t N,
    int64_t K,
    int64_t groups) {
  // Number of Xe cores; the tile scheduler sizes its persistent grid from it.
  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  using Gemm = typename moe_policy::Gemm;
  GroupedGemmRunner_xe35<Gemm, moe_policy::NeedScale> runner;

  runner.run(
      stream,
      hw_info,
      reinterpret_cast<const int*>(rows_per_expert),
      reinterpret_cast<const typename moe_policy::ElementA*>(ptr_A),
      reinterpret_cast<const typename moe_policy::ElementScaleA*>(ptr_A_scale),
      reinterpret_cast<const typename moe_policy::ElementB*>(ptr_B),
      reinterpret_cast<const typename moe_policy::ElementScaleB*>(ptr_B_scale),
      reinterpret_cast<const typename moe_policy::ElementAccumulator*>(ptr_bias),
      reinterpret_cast<typename moe_policy::ElementOutput*>(ptr_D),
      N,
      K,
      groups,
      moe_policy::BlockSize);
}

}  // namespace sgl_kernel::vllm_grouped_gemm_xe35
