/***************************************************************************************************
 * Copyright 2025 SGLang Team. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 **************************************************************************************************/
/*! \file
    \brief MXFP4 (E2M1 + UE8M0 scales, block=32) blockwise grouped GEMM for MoE on Intel XPU (xe35).
           HW-accelerated via XE_BDPAS.

    NOTE: CUTE_ENABLE_XE_BLOCK_2D_ASSERT MUST be defined before any CUTE header
    is included. It acts as a compiler barrier that prevents misoptimization of
    the block-2D payload setup in Xe2DTraitsBase::device_init() (without it the
    MXFP4 mainloop emits invalid surface descriptors). Keep this define at the
    top of this TU and out of the MXFP8 TU (where it triggers spurious x%4
    asserts on a D-store the hardware silently handles).
*/

// clang-format off
#define CUTE_ENABLE_XE_BLOCK_2D_ASSERT

#include "blockwise_moe_runner.hpp"

using namespace cute;
using namespace cutlass::gemm;

namespace at::native::xpu {

// MXFP4: E2M1 + UE8M0 scales, block=32, A=RowMajor, B=ColumnMajor, symmetric MN-major scale strides
struct MXFP4Types {

  using ElementType     = cutlass::mx_float4_t<float_e2m1_t>;
  using ElementInputA   = typename ElementType::DataType;    // float_e2m1_t
  using ElementInputB   = typename ElementType::DataType;    // float_e2m1_t
  using ElementScale    = typename ElementType::ScaleFactorType;  // float_ue8m0_t


  using ElementAccumulator       = float;
  using ElementComputeEpilogue   = float;
  using ElementOutput            = float;


  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  // Scale strides (both MN-major)
  using StrideScaleA = cute::Stride<_1, int64_t, int64_t>;
  using StrideScaleB = cute::Stride<_1, int64_t, int64_t>;


  static constexpr int BlockSize = 32;
  static constexpr int TileK     = 64;

  // Gmem copy atoms (void = auto-select)
  using GmemTiledCopyA      = void;
  using GmemTiledCopyB      = void;
  using GmemTiledCopyScaleA = void;
  using GmemTiledCopyScaleB = void;

  using TileShape = Shape<_512, _512, Int<TileK>>;

  // Thread layout (8×4 SG tiling, n-major)
  using ThreadLayout = cute::Layout<Shape<_8, _4, _1>, cute::Stride<_4, _1, _0>>;


  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_BDPAS_TT<8, float, ElementInputA>>,
      cute::Layout<TileShape>,
      ThreadLayout>::TiledMMA;

  // Mainloop dispatch (integer GroupSize → MXFP specialization)
  static constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy    = cutlass::gemm::MainloopIntelXeXMX16BlockScaledGroup<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16Group;


  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementOutput, ElementComputeEpilogue, ElementAccumulator, ElementAccumulator,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<
      EpilogueDispatchPolicy, EpilogueOp, TileShape, decltype(tile_shape(TiledMma()))>;

  // Collective epilogue (legacy path, explicit copy atoms)
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
      cute::tuple<cutlass::gemm::TagToStrideA_t<LayoutA*>, StrideScaleA*>,
      cute::tuple<ElementInputB, ElementScale>,
      cute::tuple<cutlass::gemm::TagToStrideB_t<LayoutB*>, StrideScaleB*>,
      TiledMma,
      cute::tuple<GmemTiledCopyA, GmemTiledCopyScaleA>,
      void, void, cute::identity,
      cute::tuple<GmemTiledCopyB, GmemTiledCopyScaleB>,
      void, void, cute::identity>;


  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      GroupedProblemShape, CollectiveMainloop, CollectiveEpilogue, cutlass::gemm::GroupScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;


  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;
};

using MXFP4Runner = BlockScaledGroupedGemmRunner<MXFP4Types>;

}  // namespace at::native::xpu


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
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& workspace) {
  TORCH_CHECK(a.device().is_xpu(), "Input tensor A must be on XPU device");
  TORCH_CHECK(b.device().is_xpu(), "Input tensor B must be on XPU device");
  TORCH_CHECK(scales_a.device().is_xpu(), "Scales tensor A must be on XPU device");
  TORCH_CHECK(scales_b.device().is_xpu(), "Scales tensor B must be on XPU device");
  TORCH_CHECK(output.device().is_xpu(), "Output tensor must be on XPU device");
  TORCH_CHECK(workspace.device().is_xpu(), "Workspace tensor must be on XPU device");

  TORCH_CHECK(
      a.scalar_type() == torch::kUInt8 && b.scalar_type() == torch::kUInt8,
      "Inputs must be uint8 (packed MXFP4)");
  TORCH_CHECK(
      scales_a.scalar_type() == torch::kUInt8 && scales_b.scalar_type() == torch::kUInt8,
      "Scales must be uint8 (UE8M0)");

  at::native::xpu::MXFP4Runner::run(
      output, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs,
      a, b, scales_a, scales_b, problem_sizes, expert_offsets, workspace);
}
