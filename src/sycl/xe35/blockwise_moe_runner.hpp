/***************************************************************************************************
 * Copyright 2025 SGLang Team. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 **************************************************************************************************/
/*! \file
    \brief Shared grouped-GEMM runner template for MXFP4/MXFP8 blockwise MoE on Intel XPU (xe35).

    Header-only so that each including TU controls its own preprocessor state
    (notably CUTE_ENABLE_XE_BLOCK_2D_ASSERT, which the MXFP4 path requires as a
    compiler barrier and the MXFP8 path must NOT define because its D-store
    triggers spurious x%4 asserts that the hardware silently handles).
*/

#pragma once

// clang-format off
#include <utility>

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/extension.h>

#include <cute/tensor.hpp>
#include <cute/arch/mma_xe.hpp>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/float_subbyte.h"
#include "cutlass/float8.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

namespace at::native::xpu {

using GroupedProblemShape = cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;
using UnderlyingProblemShapeType = typename GroupedProblemShape::UnderlyingProblemShape;

template <typename Types>
class BlockScaledGroupedGemmRunner {
 public:
  using Gemm          = typename Types::Gemm;
  using GemmKernel    = typename Gemm::GemmKernel;
  using ElementInputA = typename Types::ElementInputA;
  using ElementInputB = typename Types::ElementInputB;
  using ElementScale  = typename Types::ElementScale;
  using ElementOutput = typename Types::ElementOutput;
  using StrideA       = typename Types::StrideA;
  using StrideB       = typename Types::StrideB;
  using StrideC       = typename Types::StrideC;
  using StrideD       = typename Types::StrideD;
  using StrideScaleA  = typename Types::StrideScaleA;
  using StrideScaleB  = typename Types::StrideScaleB;

  static void run(
      torch::Tensor& output,
      torch::Tensor& a_ptrs,
      torch::Tensor& b_ptrs,
      torch::Tensor& out_ptrs,
      torch::Tensor& a_scales_ptrs,
      torch::Tensor& b_scales_ptrs,
      const torch::Tensor& a,
      const torch::Tensor& b,
      const torch::Tensor& scales_a,
      const torch::Tensor& scales_b,
      const torch::Tensor& problem_sizes,
      const torch::Tensor& expert_offsets,
      const torch::Tensor& workspace) {

    TORCH_CHECK(problem_sizes.dim() == 2 && problem_sizes.size(1) == 3,
                "problem_sizes must be (num_experts, 3)");
    TORCH_CHECK(problem_sizes.size(0) == expert_offsets.size(0),
                "Expert count mismatch");
    TORCH_CHECK(problem_sizes.scalar_type() == torch::kInt32 &&
                expert_offsets.scalar_type() == torch::kInt32,
                "Indices must be int32");
    TORCH_CHECK(output.scalar_type() == torch::kFloat32,
                "Output must be float32");

    int num_groups = static_cast<int>(expert_offsets.size(0));
    TORCH_CHECK(num_groups > 0,
                "Number of experts must be positive, got ", num_groups);

    TORCH_CHECK(a.dim() == 3,
                "Input tensor A must be 3-dimensional, got ", a.dim(), " dimensions");
    TORCH_CHECK(b.dim() == 3,
                "Input tensor B must be 3-dimensional, got ", b.dim(), " dimensions");
    TORCH_CHECK(scales_a.dim() == 3,
                "Scales tensor A must be 3-dimensional, got ", scales_a.dim(), " dimensions");
    TORCH_CHECK(scales_b.dim() == 3,
                "Scales tensor B must be 3-dimensional, got ", scales_b.dim(), " dimensions");
    TORCH_CHECK(output.dim() == 3,
                "Output tensor must be 3-dimensional, got ", output.dim(), " dimensions");

    TORCH_CHECK(a.size(0) == num_groups,
                "Tensor A batch size must match num_experts: expected ", num_groups, " got ", a.size(0));
    TORCH_CHECK(b.size(0) == num_groups,
                "Tensor B batch size must match num_experts: expected ", num_groups, " got ", b.size(0));
    TORCH_CHECK(scales_a.size(0) == num_groups,
                "Scales A batch size must match num_experts: expected ", num_groups, " got ", scales_a.size(0));
    TORCH_CHECK(scales_b.size(0) == num_groups,
                "Scales B batch size must match num_experts: expected ", num_groups, " got ", scales_b.size(0));
    TORCH_CHECK(output.size(0) == num_groups,
                "Output batch size must match num_experts: expected ", num_groups, " got ", output.size(0));

    TORCH_CHECK(a.is_contiguous(), "Input tensor A must be contiguous.");
    TORCH_CHECK(b.is_contiguous(), "Input tensor B must be contiguous.");
    TORCH_CHECK(scales_a.is_contiguous(), "Scales tensor A must be contiguous.");
    TORCH_CHECK(scales_b.is_contiguous(), "Scales tensor B must be contiguous.");
    TORCH_CHECK(output.is_contiguous(), "Output tensor must be contiguous.");
    TORCH_CHECK(a_ptrs.is_contiguous(), "Pointer array a_ptrs must be contiguous");
    TORCH_CHECK(b_ptrs.is_contiguous(), "Pointer array b_ptrs must be contiguous");
    TORCH_CHECK(out_ptrs.is_contiguous(), "Pointer array out_ptrs must be contiguous");
    TORCH_CHECK(a_scales_ptrs.is_contiguous(), "Pointer array a_scales_ptrs must be contiguous");
    TORCH_CHECK(b_scales_ptrs.is_contiguous(), "Pointer array b_scales_ptrs must be contiguous");
    TORCH_CHECK(problem_sizes.is_contiguous(), "problem_sizes must be contiguous");

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = static_cast<int>(a.device().index());
    hw_info.sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    auto stream = at::xpu::getCurrentXPUStream(a.device().index());
    sycl::queue& queue = stream.queue();

    auto device  = problem_sizes.device();
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(device);

    auto M_col = problem_sizes.select(1, 0);
    auto N_col = problem_sizes.select(1, 1);
    auto K_col = problem_sizes.select(1, 2);

    torch::Tensor stride_A_dev  = K_col.to(torch::kInt64).contiguous();
    torch::Tensor stride_B_dev  = K_col.to(torch::kInt64).contiguous();
    torch::Tensor stride_CD_dev = N_col.to(torch::kInt64).contiguous();

    auto stride_SFA_dev = build_scale_stride_A(M_col, N_col, K_col, num_groups, opts_i64);
    auto stride_SFB_dev = build_scale_stride_B(M_col, N_col, K_col, num_groups, opts_i64);

    auto* problem_sizes_ptr = reinterpret_cast<UnderlyingProblemShapeType*>(
        problem_sizes.data_ptr<int32_t>());
    auto* stride_A_ptr   = reinterpret_cast<StrideA*>(stride_A_dev.data_ptr<int64_t>());
    auto* stride_B_ptr   = reinterpret_cast<StrideB*>(stride_B_dev.data_ptr<int64_t>());
    auto* stride_C_ptr   = reinterpret_cast<StrideC*>(stride_CD_dev.data_ptr<int64_t>());
    auto* stride_D_ptr   = reinterpret_cast<StrideD*>(stride_CD_dev.data_ptr<int64_t>());
    auto* stride_SFA_ptr = reinterpret_cast<StrideScaleA*>(stride_SFA_dev.data_ptr<int64_t>());
    auto* stride_SFB_ptr = reinterpret_cast<StrideScaleB*>(stride_SFB_dev.data_ptr<int64_t>());

    typename Gemm::Arguments arguments;
    decltype(arguments.epilogue.thread) fusion_args;
    fusion_args.alpha = 1.0f;
    fusion_args.beta  = 0.0f;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr  = nullptr;
    fusion_args.alpha_ptr_array = nullptr;
    fusion_args.beta_ptr_array  = nullptr;
    fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
    fusion_args.dBeta  = {cute::_0{}, cute::_0{}, 0};

    using RasterOrderOptions = typename cutlass::gemm::kernel::detail::
        PersistentTileSchedulerXeGroup<GroupedProblemShape>::RasterOrderOptions;

    typename GemmKernel::Arguments gemm_args{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        typename GemmKernel::ProblemShape{num_groups, problem_sizes_ptr, nullptr},
        typename GemmKernel::MainloopArguments{
            reinterpret_cast<ElementInputA const**>(a_ptrs.data_ptr()),
            stride_A_ptr,
            reinterpret_cast<ElementInputB const**>(b_ptrs.data_ptr()),
            stride_B_ptr,
            reinterpret_cast<ElementScale const**>(a_scales_ptrs.data_ptr()),
            stride_SFA_ptr,
            reinterpret_cast<ElementScale const**>(b_scales_ptrs.data_ptr()),
            stride_SFB_ptr},
        typename GemmKernel::EpilogueArguments{
            fusion_args,
            epilogue_ptr_C(out_ptrs),
            stride_C_ptr,
            reinterpret_cast<ElementOutput**>(out_ptrs.data_ptr()),
            stride_D_ptr},
        hw_info,
        typename GemmKernel::TileSchedulerArguments{1, RasterOrderOptions::AlongN}};

    Gemm gemm_op;
    TORCH_CHECK(gemm_op.can_implement(gemm_args) == cutlass::Status::kSuccess,
                "CUTLASS cannot implement this configuration");

    size_t workspace_size = Gemm::get_workspace_size(gemm_args);
    TORCH_CHECK(static_cast<size_t>(workspace.numel()) >= workspace_size,
                "Workspace insufficient: need ", workspace_size, " bytes");

    TORCH_CHECK(gemm_op.initialize(gemm_args, workspace.data_ptr()) == cutlass::Status::kSuccess,
                "Failed to initialize");
    TORCH_CHECK(gemm_op.run(&queue) == cutlass::Status::kSuccess,
                "Failed to run");
  }

 private:
  // We pass out_ptrs as the source pointer array on both paths. For MXFP8 with
  // ElementC=void the source load is elided (is_source_supported=false), but
  // the generic-group epilogue's to_base_arguments still dereferences
  // args.ptr_C[idx] to build per-group sub-arguments, so the array itself must
  // be a valid device pointer — nullptr hangs on some XPU targets. For MXFP4
  // (ElementC=float) the read is mathematically inert because alpha=1, beta=0
  // applied to the zero-initialized output has no effect.
  using EpilogueArguments = typename GemmKernel::EpilogueArguments;
  using PtrCType = decltype(std::declval<EpilogueArguments>().ptr_C);

  static PtrCType epilogue_ptr_C(torch::Tensor& out_ptrs) {
    return reinterpret_cast<PtrCType>(out_ptrs.data_ptr());
  }

  // A scales: Stride<_1, M, 0> — same layout for both MXFP4 and MXFP8
  static torch::Tensor build_scale_stride_A(
      const torch::Tensor& M_col,
      const torch::Tensor& /*N_col*/,
      const torch::Tensor& /*K_col*/,
      int num_groups,
      const torch::TensorOptions& opts_i64) {
    auto zeros = torch::zeros({num_groups}, opts_i64);
    return torch::stack({M_col.to(torch::kInt64), zeros}, /*dim=*/1).contiguous();
  }

  // B scales: dispatched on StrideScaleB layout (MXFP4=MN-major, MXFP8=row-major)
  static torch::Tensor build_scale_stride_B(
      const torch::Tensor& /*M_col*/,
      const torch::Tensor& N_col,
      const torch::Tensor& K_col,
      int num_groups,
      const torch::TensorOptions& opts_i64) {
    return build_scale_stride_B_impl(N_col, K_col, num_groups, opts_i64,
                                     StrideScaleB{});
  }

  // MXFP4: Stride<_1, N, 0> (MN-major)
  static torch::Tensor build_scale_stride_B_impl(
      const torch::Tensor& N_col,
      const torch::Tensor& /*K_col*/,
      int num_groups,
      const torch::TensorOptions& opts_i64,
      cute::Stride<cute::_1, int64_t, int64_t> /*tag*/) {
    auto zeros = torch::zeros({num_groups}, opts_i64);
    return torch::stack({N_col.to(torch::kInt64), zeros}, /*dim=*/1).contiguous();
  }

  // MXFP8: Stride<K/BS, _1, 0> (row-major)
  static torch::Tensor build_scale_stride_B_impl(
      const torch::Tensor& /*N_col*/,
      const torch::Tensor& K_col,
      int num_groups,
      const torch::TensorOptions& opts_i64,
      cute::Stride<int64_t, cute::_1, int64_t> /*tag*/) {
    constexpr int BS = Types::BlockSize;
    auto scale_k = torch::div(K_col.to(torch::kInt64) + (BS - 1), BS, "trunc");
    auto zeros = torch::zeros({num_groups}, opts_i64);
    return torch::stack({scale_k, zeros}, /*dim=*/1).contiguous();
  }
};

}  // namespace at::native::xpu
