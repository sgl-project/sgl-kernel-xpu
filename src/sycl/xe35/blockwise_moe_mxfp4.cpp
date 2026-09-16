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
#include "sgl_kernel_export.h"
#include "moe_group_gemm_helper.hpp"

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

  // Void selects CUTLASS's block-2D auto-detection path.
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
      // cute::type_list — see equivalent comment in blockwise_moe_mxfp8.cpp.
      cute::type_list<GmemTiledCopyA, GmemTiledCopyScaleA>,
      void, void, cute::identity,
      cute::type_list<GmemTiledCopyB, GmemTiledCopyScaleB>,
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


SGL_KERNEL_EXPORT void mxfp4_blockwise_scaled_grouped_mm(
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

  // Empty ptr-array tensors -> on-device prep (flat-2D layout). Filled
  // ptrs -> legacy path (caller pre-built ptr table + scale transposes).
  torch::Tensor ptr_table_keep_alive;
  torch::Tensor scales_a_t_keep_alive;
  torch::Tensor scales_b_t_keep_alive;

  const bool need_prep = (a_ptrs.numel() == 0);

  if (need_prep) {
    TORCH_CHECK(a.dim() == 2,
                "On-device prep requires flat 2D A (sum_m_i, K/2), got ", a.dim(), " dimensions");
    TORCH_CHECK(scales_a.dim() == 2,
                "On-device prep requires flat 2D scales_a (sum_m_i, K/BS), got ", scales_a.dim());
    TORCH_CHECK(output.dim() == 2,
                "On-device prep requires flat 2D output (sum_m_i, N), got ", output.dim());
    TORCH_CHECK(b.dim() == 3, "On-device prep requires 3D B (E, N, K/2)");
    TORCH_CHECK(scales_b.dim() == 3,
                "On-device prep requires 3D scales_b (E, N, K/BS) row-major un-transposed");
    TORCH_CHECK(b_ptrs.numel() == 0 && out_ptrs.numel() == 0 &&
                    a_scales_ptrs.numel() == 0 && b_scales_ptrs.numel() == 0,
                "On-device prep requires all ptr-array tensors to be empty");

    constexpr int BS = at::native::xpu::MXFP4Types::BlockSize;
    const int E = static_cast<int>(expert_offsets.size(0));
    const int packed_K = static_cast<int>(a.size(1));   // K/2
    const int K = packed_K * 2;
    const int N = static_cast<int>(output.size(1));
    TORCH_CHECK(K % BS == 0, "K must be a multiple of ", BS);
    const int scale_cols = K / BS;
    TORCH_CHECK(scales_a.size(1) == scale_cols,
                "scales_a flat shape mismatch; expected (*, K/BS) row-major");
    TORCH_CHECK(scales_b.size(0) == E && scales_b.size(1) == N &&
                    scales_b.size(2) == scale_cols,
                "scales_b shape mismatch; expected (E, N, K/BS) row-major un-transposed");

    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(a.device());
    auto opts_u8  = torch::TensorOptions().dtype(torch::kUInt8).device(a.device());
    ptr_table_keep_alive = torch::empty({5, E}, opts_i64);

    auto& q = at::xpu::getCurrentXPUStream(a.device().index()).queue();

    const int64_t b_elem  = b.element_size();
    const int64_t o_elem  = output.element_size();
    const int64_t sa_elem = scales_a.element_size();
    const int64_t sb_elem = scales_b.element_size();

    // max_m_i drives the padded A-scales scratch.
    auto problem_sizes_cpu = problem_sizes.to(torch::kCPU);
    const int32_t* psz = problem_sizes_cpu.data_ptr<int32_t>();
    int max_m = 0;
    for (int e = 0; e < E; ++e) {
      if (psz[e * 3] > max_m) max_m = psz[e * 3];
    }
    TORCH_CHECK(max_m > 0, "max_m must be positive across experts");

    scales_a_t_keep_alive = torch::empty({E, scale_cols, max_m}, opts_u8);
    scales_b_t_keep_alive = torch::empty({E, scale_cols, N}, opts_u8);

    launch_u8_scale_build_pointers_and_transpose_scales_flat(
        q, E, max_m, scale_cols, N, /*a_row_stride_bytes=*/packed_K, static_cast<int>(o_elem),
        problem_sizes.data_ptr<int32_t>(),
        expert_offsets.data_ptr<int32_t>(),
        scales_a.data_ptr<uint8_t>(),
        scales_a_t_keep_alive.data_ptr<uint8_t>(),
        ptr_table_keep_alive.data_ptr<int64_t>(),
        reinterpret_cast<int64_t>(a.data_ptr()),
        reinterpret_cast<int64_t>(b.data_ptr()),     b.stride(0) * b_elem,
        reinterpret_cast<int64_t>(output.data_ptr()),
        reinterpret_cast<int64_t>(scales_a_t_keep_alive.data_ptr()),
            scales_a_t_keep_alive.stride(0) * sa_elem,
        reinterpret_cast<int64_t>(scales_b_t_keep_alive.data_ptr()),
            scales_b_t_keep_alive.stride(0) * sb_elem,
        scales_a.stride(0),
        scales_a_t_keep_alive.stride(0));

    launch_u8_transpose_b_scales(
        q, E, N, scale_cols,
        scales_b.data_ptr<uint8_t>(),
        scales_b_t_keep_alive.data_ptr<uint8_t>());

    a_ptrs        = ptr_table_keep_alive[0];
    b_ptrs        = ptr_table_keep_alive[1];
    out_ptrs      = ptr_table_keep_alive[2];
    a_scales_ptrs = ptr_table_keep_alive[3];
    b_scales_ptrs = ptr_table_keep_alive[4];
  }

  // Padded A-scales -> override M-stride to max_m; legacy path uses m_i.
  const int sa_m_stride_override =
      need_prep ? static_cast<int>(scales_a_t_keep_alive.stride(1)) : 0;
  at::native::xpu::MXFP4Runner::run(
      output, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs,
      a, b,
      need_prep ? scales_a_t_keep_alive : scales_a,
      need_prep ? scales_b_t_keep_alive : scales_b,
      problem_sizes, expert_offsets, workspace,
      sa_m_stride_override);
}
