/***************************************************************************************************
 * Copyright 2025 SGLang Team. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 **************************************************************************************************/
/*! \file
    \brief Block-wise Scaled Grouped GEMM for MoE on Intel XPU (xe35)
    Supports MXFP4 (E2M1) and FP8 (E4M3) with configurable block sizes
*/

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/extension.h>

#include <cute/arch/mma_xe.hpp>
#include <cute/tensor.hpp>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/float_subbyte.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;
using namespace cutlass::gemm;

namespace at::native::xpu {

template <typename ElementType_, int BlockSize_, int TileK_>
struct BlockScaledGemmConfig {
  using ElementType = ElementType_;
  using MmaType = typename ElementType::DataType;
  using ElementInputA = typename ElementType::DataType;
  using ElementInputB = typename ElementType::DataType;
  using ElementScale = typename ElementType::ScaleFactorType;

  static constexpr int BlockSize = BlockSize_;
  static constexpr int TileK = TileK_;

  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ElementOutput = float;
  using ElementC = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  using StrideScale = cute::Stride<_1, int64_t, int64_t>;

  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;
  using GmemTiledCopyScaleA = void;
  using GmemTiledCopyScaleB = void;
};

template <typename Config>
class BlockScaledGroupedGemmKernel {
 public:
  using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
  using GroupScheduler = cutlass::gemm::GroupScheduler;
  using ElementInputA = typename Config::ElementInputA;
  using ElementInputB = typename Config::ElementInputB;
  using ElementScale = typename Config::ElementScale;
  using ElementOutput = typename Config::ElementOutput;
  using ElementAccumulator = typename Config::ElementAccumulator;
  using ElementComputeEpilogue = typename Config::ElementComputeEpilogue;
  using LayoutA = typename Config::LayoutA;
  using LayoutB = typename Config::LayoutB;
  using LayoutC = typename Config::LayoutC;
  using LayoutD = typename Config::LayoutD;
  using StrideScale = typename Config::StrideScale;

  static constexpr int BlockSize = Config::BlockSize;
  static constexpr int TileK = Config::TileK;
  static constexpr int PipelineStages = 2;

  using TileShape = Shape<_512, _512, Int<TileK>>;

  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_BDPAS_TT<8, float, ElementInputA>>,
      cute::Layout<TileShape>,
      cute::Layout<Shape<_8, _4, _1>, cute::Stride<_4, _1, _0>>>::TiledMMA;

  using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16BlockScaledGroup<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16Group;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementOutput,
      ElementComputeEpilogue,
      ElementAccumulator,
      ElementAccumulator,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using FusionCallBacks = cutlass::epilogue::fusion::
      FusionCallbacks<EpilogueDispatchPolicy, EpilogueOp, TileShape, decltype(tile_shape(TiledMma()))>;

  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
      EpilogueDispatchPolicy,
      TileShape,
      ElementAccumulator,
      cutlass::gemm::TagToStrideC_t<LayoutC*>,
      ElementOutput,
      cutlass::gemm::TagToStrideC_t<LayoutD*>,
      FusionCallBacks,
      XE_2D_U32x8x16_LD_N,
      void,
      void,
      XE_2D_U32x8x16_ST_N,
      void,
      void>;

  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      GEMMDispatchPolicy,
      TileShape,
      cute::tuple<ElementInputA, ElementScale>,
      cute::tuple<cutlass::gemm::TagToStrideA_t<LayoutA*>, StrideScale*>,
      cute::tuple<ElementInputB, ElementScale>,
      cute::tuple<cutlass::gemm::TagToStrideB_t<LayoutB*>, StrideScale*>,
      TiledMma,
      cute::tuple<typename Config::GmemTiledCopyA, typename Config::GmemTiledCopyScaleA>,
      void,
      void,
      cute::identity,
      cute::tuple<typename Config::GmemTiledCopyB, typename Config::GmemTiledCopyScaleB>,
      void,
      void,
      cute::identity>;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue, GroupScheduler>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;
  using StrideScaleA = StrideScale;
  using StrideScaleB = StrideScale;
  using UnderlyingProblemShape = typename ProblemShape::UnderlyingProblemShape;

  static void
  run(torch::Tensor& output,
      torch::Tensor& a_ptrs,
      torch::Tensor& b_ptrs,
      torch::Tensor& out_ptrs,
      torch::Tensor& a_scales_ptrs,
      torch::Tensor& b_scales_ptrs,
      const torch::Tensor& a,
      const torch::Tensor& b,
      const torch::Tensor& scales_a,
      const torch::Tensor& scales_b,
      const torch::Tensor& stride_a,
      const torch::Tensor& stride_b,
      const torch::Tensor& stride_c,
      const torch::Tensor& layout_sfa,
      const torch::Tensor& layout_sfb,
      const torch::Tensor& problem_sizes,
      const torch::Tensor& expert_offsets,
      const torch::Tensor& workspace) {
    TORCH_CHECK(problem_sizes.dim() == 2 && problem_sizes.size(1) == 3, "problem_sizes must be (num_experts, 3)");
    TORCH_CHECK(problem_sizes.size(0) == expert_offsets.size(0), "Expert count mismatch");
    TORCH_CHECK(
        problem_sizes.scalar_type() == torch::kInt32 && expert_offsets.scalar_type() == torch::kInt32,
        "Indices must be int32");
    TORCH_CHECK(output.scalar_type() == torch::kFloat32, "Output must be float32");

    int num_groups = static_cast<int>(expert_offsets.size(0));

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = static_cast<int>(a.device().index());
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    torch::Tensor problem_sizes_cpu = problem_sizes.to(torch::kCPU).contiguous();
    int32_t* problem_sizes_ptr = problem_sizes_cpu.data_ptr<int32_t>();

    std::vector<UnderlyingProblemShape> problem_sizes_host;
    std::vector<StrideA> stride_A_host;
    std::vector<StrideB> stride_B_host;
    std::vector<StrideC> stride_C_host;
    std::vector<StrideD> stride_D_host;
    std::vector<StrideScaleA> stride_SFA_host;
    std::vector<StrideScaleB> stride_SFB_host;

    problem_sizes_host.reserve(num_groups);
    stride_A_host.reserve(num_groups);
    stride_B_host.reserve(num_groups);
    stride_C_host.reserve(num_groups);
    stride_D_host.reserve(num_groups);
    stride_SFA_host.reserve(num_groups);
    stride_SFB_host.reserve(num_groups);

    for (int i = 0; i < num_groups; ++i) {
      int M = problem_sizes_ptr[i * 3 + 0];
      int N = problem_sizes_ptr[i * 3 + 1];
      int K = problem_sizes_ptr[i * 3 + 2];
      problem_sizes_host.push_back({M, N, K});

      const int scale_k = cute::ceil_div(K, BlockSize);
      auto shape_A = cute::make_shape(M, K, 1);
      auto shape_B = cute::make_shape(N, K, 1);
      auto shape_CD = cute::make_shape(M, N, 1);
      auto shape_scale_A = cute::make_shape(M, scale_k, 1);
      auto shape_scale_B = cute::make_shape(N, scale_k, 1);

      stride_A_host.push_back(cutlass::make_cute_packed_stride(StrideA{}, shape_A));
      stride_B_host.push_back(cutlass::make_cute_packed_stride(StrideB{}, shape_B));
      stride_C_host.push_back(cutlass::make_cute_packed_stride(StrideC{}, shape_CD));
      stride_D_host.push_back(cutlass::make_cute_packed_stride(StrideD{}, shape_CD));
      stride_SFA_host.push_back(cutlass::make_cute_packed_stride(StrideScaleA{}, shape_scale_A));
      stride_SFB_host.push_back(cutlass::make_cute_packed_stride(StrideScaleB{}, shape_scale_B));
    }

    cutlass::DeviceAllocation<UnderlyingProblemShape> problem_sizes_device;
    cutlass::DeviceAllocation<StrideA> stride_A_device;
    cutlass::DeviceAllocation<StrideB> stride_B_device;
    cutlass::DeviceAllocation<StrideC> stride_C_device;
    cutlass::DeviceAllocation<StrideD> stride_D_device;
    cutlass::DeviceAllocation<StrideScaleA> stride_SFA_device;
    cutlass::DeviceAllocation<StrideScaleB> stride_SFB_device;

    problem_sizes_device.reset(num_groups);
    problem_sizes_device.copy_from_host(problem_sizes_host.data());
    stride_A_device.reset(num_groups);
    stride_A_device.copy_from_host(stride_A_host.data());
    stride_B_device.reset(num_groups);
    stride_B_device.copy_from_host(stride_B_host.data());
    stride_C_device.reset(num_groups);
    stride_C_device.copy_from_host(stride_C_host.data());
    stride_D_device.reset(num_groups);
    stride_D_device.copy_from_host(stride_D_host.data());
    stride_SFA_device.reset(num_groups);
    stride_SFA_device.copy_from_host(stride_SFA_host.data());
    stride_SFB_device.reset(num_groups);
    stride_SFB_device.copy_from_host(stride_SFB_host.data());

    typename Gemm::Arguments arguments;
    decltype(arguments.epilogue.thread) fusion_args;
    fusion_args.alpha = 1.0f;
    fusion_args.beta = 0.0f;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr = nullptr;
    fusion_args.alpha_ptr_array = nullptr;
    fusion_args.beta_ptr_array = nullptr;
    fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
    fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};

    using RasterOrderOptions =
        typename cutlass::gemm::kernel::detail::PersistentTileSchedulerXeGroup<ProblemShape>::RasterOrderOptions;

    typename Gemm::GemmKernel::Arguments gemm_args{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        typename Gemm::GemmKernel::ProblemShape{num_groups, problem_sizes_device.get(), problem_sizes_host.data()},
        typename Gemm::GemmKernel::MainloopArguments{
            reinterpret_cast<ElementInputA const**>(a_ptrs.data_ptr()),
            stride_A_device.get(),
            reinterpret_cast<ElementInputB const**>(b_ptrs.data_ptr()),
            stride_B_device.get(),
            reinterpret_cast<ElementScale const**>(a_scales_ptrs.data_ptr()),
            stride_SFA_device.get(),
            reinterpret_cast<ElementScale const**>(b_scales_ptrs.data_ptr()),
            stride_SFB_device.get(),
            BlockSize},
        typename Gemm::GemmKernel::EpilogueArguments{
            fusion_args,
            nullptr,
            stride_C_device.get(),
            reinterpret_cast<ElementOutput**>(out_ptrs.data_ptr()),
            stride_D_device.get()},
        hw_info,
        typename Gemm::GemmKernel::TileSchedulerArguments{1, RasterOrderOptions::AlongN}};

    Gemm gemm_op;
    TORCH_CHECK(
        gemm_op.can_implement(gemm_args) == cutlass::Status::kSuccess, "CUTLASS cannot implement this configuration");

    size_t workspace_size = Gemm::get_workspace_size(gemm_args);
    TORCH_CHECK(
        static_cast<size_t>(workspace.numel()) >= workspace_size,
        "Workspace insufficient: need ",
        workspace_size,
        " bytes");

    TORCH_CHECK(
        gemm_op.initialize(gemm_args, workspace.data_ptr()) == cutlass::Status::kSuccess, "Failed to initialize");
    TORCH_CHECK(gemm_op.run() == cutlass::Status::kSuccess, "Failed to run");
    compat::wait();
  }
};

// MXFP4 configuration: E2M1 with 32-element blocks and TileK=64
using MXFP4Config = BlockScaledGemmConfig<cutlass::mx_float4_t<float_e2m1_t>, 32, 64>;
using MXFP4Kernel = BlockScaledGroupedGemmKernel<MXFP4Config>;

void mxfp4_blockwise_scaled_grouped_mm(
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
    const torch::Tensor& stride_a,
    const torch::Tensor& stride_b,
    const torch::Tensor& stride_c,
    const torch::Tensor& layout_sfa,
    const torch::Tensor& layout_sfb,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& workspace) {
  TORCH_CHECK(
      a.scalar_type() == torch::kUInt8 && b.scalar_type() == torch::kUInt8, "Inputs must be uint8 (packed MXFP4)");
  TORCH_CHECK(
      scales_a.scalar_type() == torch::kUInt8 && scales_b.scalar_type() == torch::kUInt8,
      "Scales must be uint8 (UE8M0)");

  MXFP4Kernel::run(
      output,
      a_ptrs,
      b_ptrs,
      out_ptrs,
      a_scales_ptrs,
      b_scales_ptrs,
      a,
      b,
      scales_a,
      scales_b,
      stride_a,
      stride_b,
      stride_c,
      layout_sfa,
      layout_sfb,
      problem_sizes,
      expert_offsets,
      workspace);
}

}  // namespace at::native::xpu
