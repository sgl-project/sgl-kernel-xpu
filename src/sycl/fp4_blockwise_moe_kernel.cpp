/***************************************************************************************************
 * Copyright 2025 SGLang Team. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 **************************************************************************************************/
/*! \file
    \brief MXFP4 (E2M1) Block-wise Scaled Grouped GEMM for MoE on Intel XPU (xe35)

    Requirements:
      - Group scaled k size must be 32 (MXFP4 OpenCompute standard)
      - scales must be MN-major
      - scales are in UE8M0 format (unsigned 8-bit exponent-only)
*/

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/extension.h>

#include <cute/tensor.hpp>
#include <cute/arch/mma_xe.hpp>

#include "cutlass/cutlass.h"
#include "cutlass/float_subbyte.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

namespace at::native::xpu {

template <typename ElementType>
struct MXFP4BlockwiseScaledGroupGeMMRunner {
  using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;  // <M,N,K> per group
  using GroupScheduler = cutlass::gemm::GroupScheduler;

  // Type definitions from 13_xe35_block_scaled_grouped_gemm_e2m1.cpp
  using MmaType = typename ElementType::DataType;
  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ElementInputA = typename ElementType::DataType;
  using ElementInputB = typename ElementType::DataType;
  using ElementOutput = float;
  using ElementScale = typename ElementType::ScaleFactorType;  // UE8M0 for MXFP4

  // Layouts - A is RowMajor, B is ColumnMajor as per the CUTLASS example
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  // Stride for scales
  using StrideScale = cute::Stride<_1, int64_t, int64_t>;

  // Gmem tiled copy policies
  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;
  using GmemTiledCopyScaleA = void;
  using GmemTiledCopyScaleB = void;

  // Tile configuration - using larger K tile for FP4 as in the example
  using TileShape = Shape<_512, _512, _64>;

  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_BDPAS_TT<8, float, ElementInputA>>,
      cute::Layout<TileShape>,
      cute::Layout<cute::Shape<_8, _4, _1>, cute::Stride<_4, _1, _0>>>::TiledMMA;

  static constexpr int PipelineStages = 2;
  static constexpr int GROUP_SIZE = 32;  // MXFP4 block scaled k size must be 32

  // Dispatch policies
  using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16BlockScaledGroup<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16Group;

  // Epilogue fusion
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementOutput,
      ElementComputeEpilogue,
      ElementAccumulator,
      ElementAccumulator,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<
      EpilogueDispatchPolicy,
      EpilogueOp,
      TileShape,
      decltype(tile_shape(TiledMma()))>;

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
      cute::tuple<GmemTiledCopyA, GmemTiledCopyScaleA>,
      void,
      void,
      cute::identity,
      cute::tuple<GmemTiledCopyB, GmemTiledCopyScaleB>,
      void,
      void,
      cute::identity>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape,
      CollectiveMainloop,
      CollectiveEpilogue,
      GroupScheduler>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  // Type aliases for strides
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;
  using UnderlyingProblemShape = ProblemShape::UnderlyingProblemShape;

  template <typename GemmType>
  auto args_from_options(
      const cutlass::KernelHardwareInfo& hw_info,
      UnderlyingProblemShape* problem_sizes_ptr,
      const int num_experts) {
    typename GemmType::Arguments arguments;

    // Setup fusion arguments for epilogue (alpha=1.0, beta=0.0)
    decltype(arguments.epilogue.thread) fusion_args;
    fusion_args.alpha = 1.0f;
    fusion_args.beta = 0.0f;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr = nullptr;
    fusion_args.alpha_ptr_array = nullptr;
    fusion_args.beta_ptr_array = nullptr;
    // Single alpha and beta for all groups
    fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
    fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};

    // Setup tile scheduler arguments
    using RasterOrderOptions =
        typename cutlass::gemm::kernel::detail::PersistentTileSchedulerXeGroup<ProblemShape>::RasterOrderOptions;

    // Per-GEMM problem shape info may only exist on the device.
    return cute::make_tuple(
        cutlass::gemm::GemmUniversalMode::kGrouped,
        typename GemmType::GemmKernel::ProblemShape{num_experts, problem_sizes_ptr, nullptr},
        fusion_args,
        hw_info,
        typename GemmType::GemmKernel::TileSchedulerArguments{1, RasterOrderOptions::AlongN});
  }

  int init(
      int device_id,
      const void* a_ptrs,
      const void* stride_a,
      const void* b_ptrs,
      const void* stride_b,
      const void* a_scales_ptrs,
      const void* layout_sfa,
      const void* b_scales_ptrs,
      const void* layout_sfb,
      const void* stride_c,
      void* out_ptrs,
      const void* stride_d,
      const void* problem_sizes,
      const int num_experts) {
    hw_info.device_id = device_id;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    auto args_tuple = args_from_options<Gemm>(
        hw_info,
        static_cast<UnderlyingProblemShape*>(const_cast<void*>(problem_sizes)),
        num_experts);

    // Assemble full arguments from tuple and mainloop/epilogue arguments
    // Note: Grouped GEMM expects pointer-to-pointer for element arrays and
    // non-const pointers for stride arrays (one entry per group)
    gemm_args = typename Gemm::GemmKernel::Arguments{
        get<0>(args_tuple),  // mode
        get<1>(args_tuple),  // problem shape
        typename Gemm::GemmKernel::MainloopArguments{
            static_cast<const ElementInputA**>(const_cast<void*>(a_ptrs)),
            static_cast<StrideA*>(const_cast<void*>(stride_a)),
            static_cast<const ElementInputB**>(const_cast<void*>(b_ptrs)),
            static_cast<StrideB*>(const_cast<void*>(stride_b)),
            static_cast<const ElementScale**>(const_cast<void*>(a_scales_ptrs)),
            static_cast<StrideScale*>(const_cast<void*>(layout_sfa)),
            static_cast<const ElementScale**>(const_cast<void*>(b_scales_ptrs)),
            static_cast<StrideScale*>(const_cast<void*>(layout_sfb)),
            GROUP_SIZE},
        typename Gemm::GemmKernel::EpilogueArguments{
            get<2>(args_tuple),  // fusion args
            nullptr,             // C pointer (not used, beta=0)
            static_cast<StrideC*>(const_cast<void*>(stride_c)),
            static_cast<ElementOutput**>(out_ptrs),
            static_cast<StrideD*>(const_cast<void*>(stride_d))},
        get<3>(args_tuple),  // hw_info
        get<4>(args_tuple)   // tile scheduler args
    };

    TORCH_CHECK(
        gemm_op.can_implement(gemm_args) == cutlass::Status::kSuccess,
        "CUTLASS cannot implement this MXFP4 GEMM configuration");

    return Gemm::get_workspace_size(gemm_args);
  }

  void run(sycl::queue& queue, void* workspace) {
    TORCH_CHECK(
        gemm_op.initialize(gemm_args, workspace, &queue) == cutlass::Status::kSuccess,
        "Failed to initialize MXFP4 GEMM");

    // Run the GEMM
    TORCH_CHECK(gemm_op.run(&queue) == cutlass::Status::kSuccess, "Failed to run MXFP4 GEMM");
  }

 public:
  Gemm gemm_op;
  typename Gemm::Arguments gemm_args;
  cutlass::KernelHardwareInfo hw_info;
};

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
  // Input validation
  TORCH_CHECK(problem_sizes.dim() == 2, "problem_sizes must be 2D tensor");
  TORCH_CHECK(problem_sizes.size(1) == 3, "problem_sizes must have shape (num_experts, 3)");
  TORCH_CHECK(
      problem_sizes.size(0) == expert_offsets.size(0),
      "Number of experts in problem_sizes must match expert_offsets");
  TORCH_CHECK(problem_sizes.scalar_type() == torch::kInt32, "problem_sizes must be int32");

  // MXFP4 data is packed as uint8 (two 4-bit values per byte)
  TORCH_CHECK(a.scalar_type() == torch::kUInt8, "a must be uint8 (packed MXFP4 E2M1)");
  TORCH_CHECK(b.scalar_type() == torch::kUInt8, "b must be uint8 (packed MXFP4 E2M1)");
  TORCH_CHECK(output.scalar_type() == torch::kFloat32, "output must be float32");

  // Scales are UE8M0 format (unsigned 8-bit exponent-only), stored as uint8
  TORCH_CHECK(scales_a.scalar_type() == torch::kUInt8, "scales_a must be uint8 (UE8M0)");
  TORCH_CHECK(scales_b.scalar_type() == torch::kUInt8, "scales_b must be uint8 (UE8M0)");

  TORCH_CHECK(stride_a.scalar_type() == torch::kInt64, "stride_a must be int64");
  TORCH_CHECK(stride_b.scalar_type() == torch::kInt64, "stride_b must be int64");
  TORCH_CHECK(stride_c.scalar_type() == torch::kInt64, "stride_c must be int64");
  TORCH_CHECK(expert_offsets.scalar_type() == torch::kInt32, "expert_offsets must be int32");

  int num_experts = static_cast<int>(expert_offsets.size(0));

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  // Use mx_float4_t<float_e2m1_t> for MXFP4 with UE8M0 scales
  using ElementType = cutlass::mx_float4_t<float_e2m1_t>;
  using Kernel = MXFP4BlockwiseScaledGroupGeMMRunner<ElementType>;
  Kernel kernel;

  auto workspace_size = kernel.init(
      a_ptrs.device().index(),
      a_ptrs.data_ptr(),
      stride_a.data_ptr(),
      b_ptrs.data_ptr(),
      stride_b.data_ptr(),
      a_scales_ptrs.data_ptr(),
      layout_sfa.data_ptr(),
      b_scales_ptrs.data_ptr(),
      layout_sfb.data_ptr(),
      stride_c.data_ptr(),
      out_ptrs.data_ptr(),
      stride_c.data_ptr(),  // stride_d same as stride_c
      problem_sizes.data_ptr(),
      num_experts);

  // Verify workspace is large enough
  TORCH_CHECK(
      workspace.numel() >= workspace_size,
      "Workspace size insufficient: need ",
      workspace_size,
      " bytes, got ",
      workspace.numel());

  kernel.run(queue, workspace.data_ptr());
}

}  // namespace at::native::xpu
