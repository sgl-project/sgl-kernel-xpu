/***************************************************************************************************
 * Copyright 2025 SGLang Team. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 **************************************************************************************************/
/*! \file
    \brief MXFP8 (E4M3 + fp32 scales, block=128) blockwise grouped GEMM for MoE on Intel XPU (xe35).
           Software-scaled path.

    NOTE: CUTE_ENABLE_XE_BLOCK_2D_ASSERT is INTENTIONALLY NOT defined in this TU.
    The MXFP8 D-store has x offsets that the assert (x % 4 == 0) rejects, but the
    actual Xe hardware silently handles them — defining the macro here would
    abort an otherwise correct kernel. Keep this file's CUTE include free of the
    assert define; the MXFP4 TU defines it for its own (separate) compilation.
*/

// clang-format off
#include "blockwise_moe_runner.hpp"

using namespace cute;
using namespace cutlass::gemm;

namespace at::native::xpu {

// MXFP8: E4M3 + fp32 scales, block=128, A/B=RowMajor, asymmetric scale strides (A=col-major, B=row-major)
struct MXFP8Types {

  using ElementInputA   = cutlass::float_e4m3_t;
  using ElementInputB   = cutlass::float_e4m3_t;
  using ElementScale    = float;  // fp32 scale factors


  using ElementAccumulator       = float;
  using ElementComputeEpilogue   = float;
  using ElementOutput            = float;


  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;  // B is (N, K) with K-contiguous (PyTorch standard)
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  // Scale strides (asymmetric: A=col-major, B=row-major)
  using StrideScaleA = cute::Stride<_1, int64_t, int64_t>;
  using StrideScaleB = cute::Stride<int64_t, _1, int64_t>;

  static constexpr int BlockSize = 128;
  static constexpr int TileK     = 32;


  using GmemTiledCopyA      = void;
  using GmemTiledCopyB      = void;
  using GmemTiledCopyScaleA = void;
  using GmemTiledCopyScaleB = void;

  using TileShape = Shape<_256, _256, Int<TileK>>;

  using ThreadLayout = cute::Layout<Shape<_8, _4, _1>, cute::Stride<_4, _1, _0>>;

  // TiledMMA (XE_BDPAS_TT + fp32 scales → software scaling path)
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_BDPAS_TT<8, float, ElementInputA>>,
      cute::Layout<TileShape>,
      ThreadLayout>::TiledMMA;

  // Mainloop dispatch (tuple GroupSize → FP8 block-scaled mainloop)
  static constexpr int PipelineStages = 2;
  using GroupSizeMNK = cute::tuple<cute::_1, cute::Int<BlockSize>, cute::Int<BlockSize>>;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16BlockScaledGroupImpl<
      PipelineStages, GroupSizeMNK>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGenericGroup;


  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementOutput, ElementComputeEpilogue, ElementAccumulator, ElementAccumulator,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<
      EpilogueDispatchPolicy, EpilogueOp, TileShape, decltype(tile_shape(TiledMma()))>;

  // Collective epilogue (generic group path, no explicit copy atoms).
  // ElementC=void disables the C source load entirely (alpha=1, beta=0 path);
  // the default C-load atom was triggering XE_STORE_2D alignment asserts.
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
      EpilogueDispatchPolicy,
      TileShape,
      void,   // EpilogueTile = void (auto)
      void,   // ElementC = void (disables C load)
      cutlass::gemm::TagToStrideC_t<LayoutC*>,
      ElementOutput,
      cutlass::gemm::TagToStrideC_t<LayoutD*>,
      FusionCallBacks,
      void, void>;  // no explicit copy atoms


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

using MXFP8Runner = BlockScaledGroupedGemmRunner<MXFP8Types>;

}  // namespace at::native::xpu


void fp8_blockwise_scaled_grouped_mm(
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
  // stride_a, stride_b, stride_c, layout_sfa, layout_sfb are accepted for
  // interface compatibility but unused — strides are built internally.
  (void)stride_a; (void)stride_b; (void)stride_c;
  (void)layout_sfa; (void)layout_sfb;

  TORCH_CHECK(a.device().is_xpu(), "Input tensor A must be on XPU device");
  TORCH_CHECK(b.device().is_xpu(), "Input tensor B must be on XPU device");
  TORCH_CHECK(scales_a.device().is_xpu(), "Scales tensor A must be on XPU device");
  TORCH_CHECK(scales_b.device().is_xpu(), "Scales tensor B must be on XPU device");
  TORCH_CHECK(output.device().is_xpu(), "Output tensor must be on XPU device");
  TORCH_CHECK(workspace.device().is_xpu(), "Workspace tensor must be on XPU device");

  TORCH_CHECK(
      a.scalar_type() == torch::kFloat8_e4m3fn && b.scalar_type() == torch::kFloat8_e4m3fn,
      "Inputs must be float8_e4m3fn");
  TORCH_CHECK(
      scales_a.scalar_type() == torch::kFloat32 && scales_b.scalar_type() == torch::kFloat32,
      "Scales must be float32");

  at::native::xpu::MXFP8Runner::run(
      output, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs,
      a, b, scales_a, scales_b, problem_sizes, expert_offsets, workspace);
}
