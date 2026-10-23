/***************************************************************************************************
 * Copyright 2025 SGLang Team. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 **************************************************************************************************/
/*! \file
    \brief FP8 (DSV3-style, BS=128, fp32 scales, SW-scaled) and MXFP8 (OCP,
           BS=32, UE8M0 uint8 scales, HW-scaled) blockwise grouped GEMM for MoE
           on Intel XPU (xe35). Dispatched on scales_a.scalar_type().

    NOTE: CUTE_ENABLE_XE_BLOCK_2D_ASSERT is intentionally NOT defined here —
    the FP8 D-store's x-offset trips the assert but the hardware handles it.
*/

// clang-format off
#include "blockwise_moe_runner.hpp"
#include "moe_group_gemm_helper.hpp"
#include "sgl_kernel_export.h"

using namespace cute;
using namespace cutlass::gemm;

namespace at::native::xpu {

// Shared type traits for the two blockwise MoE grouped-GEMM configs below.
// Fixed: E4M3 A/B, row-major A/C/D, column-major B, XE_BDPAS_TT, ElementC=void.
// Variants provide (ElementScale, StrideScaleB, GroupSize, TileShape, BlockSize);
// GroupSize as cute::tuple → SW-scaled mainloop, cute::Int → HW-scaled mainloop.
template <
    typename ElementScale_,
    typename StrideScaleB_,
    typename GroupSize_,
    typename TileShape_,
    int BlockSize_>
struct BlockScaledMoETypes {
  using ElementInputA = cutlass::float_e4m3_t;
  using ElementInputB = cutlass::float_e4m3_t;
  using ElementScale  = ElementScale_;

  using ElementAccumulator     = float;
  using ElementComputeEpilogue = float;
  using ElementOutput          = float;

  using LayoutA = cutlass::layout::RowMajor;
  // B is (N, K) K-contiguous (PyTorch); ColumnMajor gives the same physical
  // access pattern as the CUTLASS example's RowMajor (K, N) with no transpose.
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using StrideScaleA = cute::Stride<cute::_1, int64_t, int64_t>;
  using StrideScaleB = StrideScaleB_;

  static constexpr int BlockSize = BlockSize_;

  // Void selects CUTLASS's block-2D auto-detection path (get_block_2d_copy_*
  // takes the is_void branch and picks the right atom based on TiledMma).
  using GmemTiledCopyA      = void;
  using GmemTiledCopyB      = void;
  using GmemTiledCopyScaleA = void;
  using GmemTiledCopyScaleB = void;

  using TileShape    = TileShape_;
  using ThreadLayout = cute::Layout<cute::Shape<cute::_8, cute::_4, cute::_1>,
                                    cute::Stride<cute::_4, cute::_1, cute::_0>>;

  using TiledMma = typename cute::TiledMMAHelper<
      cute::MMA_Atom<cute::XE_BDPAS_TT<8, float, ElementInputA>>,
      cute::Layout<TileShape>,
      ThreadLayout>::TiledMMA;

  static constexpr int PipelineStages = 2;
  // ...GroupImpl (not the ...Group alias): the alias takes an `int` GroupSize,
  // while the variants here pass a type (cute::Int<32> or a cute::tuple).
  using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16BlockScaledGroupImpl<
      PipelineStages, GroupSize_>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGenericGroup;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementOutput, ElementComputeEpilogue, ElementAccumulator, ElementAccumulator,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<
      EpilogueDispatchPolicy, EpilogueOp, TileShape, decltype(cute::tile_shape(TiledMma()))>;

  // ElementC=void (alpha=1, beta=0) — avoids XE_STORE_2D alignment asserts.
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
      EpilogueDispatchPolicy,
      TileShape,
      void,   // EpilogueTile = void (auto)
      void,   // ElementC = void (disables C load)
      cutlass::gemm::TagToStrideC_t<LayoutC*>,
      ElementOutput,
      cutlass::gemm::TagToStrideC_t<LayoutD*>,
      FusionCallBacks,
      void, void>;

  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      GEMMDispatchPolicy,
      TileShape,
      cute::tuple<ElementInputA, ElementScale>,
      cute::tuple<cutlass::gemm::TagToStrideA_t<LayoutA*>, StrideScaleA*>,
      cute::tuple<ElementInputB, ElementScale>,
      cute::tuple<cutlass::gemm::TagToStrideB_t<LayoutB*>, StrideScaleB*>,
      TiledMma,
      // cute::type_list (not cute::tuple / std::tuple) because newer DPCPP's
      // SYCL kernel-name registrar tries to instantiate this template arg, and
      // both {cute,std}::tuple<void, void> fail (can't hold void members).
      // cute::type_list is an empty struct; its std::tuple_element
      // specialization gives CollectiveMma what it needs at metafunction time.
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

// FP8: fp32 scales, row-major B-scales, tuple GroupSize → SW-scaled mainloop.
using FP8Types = BlockScaledMoETypes<
    /* ElementScale = */ float,
    /* StrideScaleB = */ cute::Stride<int64_t, cute::_1, int64_t>,
    /* GroupSize    = */ cute::tuple<cute::_1, cute::Int<128>, cute::Int<128>>,
    /* TileShape    = */ Shape<_256, _256, _32>,
    /* BlockSize    = */ 128>;

using FP8Runner = BlockScaledGroupedGemmRunner<FP8Types>;

// MXFP8: UE8M0 scales, MN-major B-scales, integer GroupSize → HW-scaled mainloop.
using MXFP8Types = BlockScaledMoETypes<
    /* ElementScale = */ typename cutlass::mx_float8_t<cutlass::float_e4m3_t>::ScaleFactorType,
    /* StrideScaleB = */ cute::Stride<cute::_1, int64_t, int64_t>,
    /* GroupSize    = */ cute::Int<32>,
    /* TileShape    = */ Shape<_512, _256, _64>,
    /* BlockSize    = */ 32>;

using MXFP8Runner = BlockScaledGroupedGemmRunner<MXFP8Types>;

}  // namespace at::native::xpu


// Entry point. Dispatches on scales_a dtype: fp32 → FP8Runner, uint8 → MXFP8Runner.
// Both share the on-device prep path (empty ptr sentinels build the {5, E}
// ptr table + transpose A-scales); MXFP8 additionally transposes B-scales.
SGL_KERNEL_EXPORT void fp8_blockwise_scaled_grouped_mm(
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

  // Both FP8 and MXFP8 require float8_e4m3fn inputs — check once here.
  TORCH_CHECK(
      a.scalar_type() == torch::kFloat8_e4m3fn && b.scalar_type() == torch::kFloat8_e4m3fn,
      "Inputs must be float8_e4m3fn");

  // Dispatch: fp32 scales → FP8, uint8 UE8M0 scales → MXFP8.
  const bool is_mxfp8 = (scales_a.scalar_type() == torch::kUInt8);
  if (is_mxfp8) {
    TORCH_CHECK(
        scales_b.scalar_type() == torch::kUInt8,
        "MXFP8 (uint8 UE8M0 A-scales) requires uint8 B-scales, got ",
        scales_b.scalar_type());
  } else {
    TORCH_CHECK(
        scales_a.scalar_type() == torch::kFloat32 && scales_b.scalar_type() == torch::kFloat32,
        "FP8 blockwise grouped GEMM requires float32 A/B scales, got scales_a=",
        scales_a.scalar_type(),
        " scales_b=",
        scales_b.scalar_type());
  }

  // -----------------------------------------------------------------------
  // Empty ptr sentinels ⇒ on-device prep of {5, E} ptr table and A-scale
  // transpose (MXFP8 also transposes B-scales to MN-major). Filled ptrs ⇒
  // legacy caller-managed path (FP8 only — MXFP8 requires the on-device
  // B-scale transpose so filled-ptr callers would run with un-transposed
  // scales and produce silently-wrong output).
  torch::Tensor ptr_table_keep_alive;
  torch::Tensor scales_a_t_keep_alive;
  torch::Tensor scales_b_t_keep_alive;

  const bool need_prep = (a_ptrs.numel() == 0);
  TORCH_CHECK(
      !is_mxfp8 || need_prep,
      "MXFP8 requires the on-device prep path (empty int64 ptr-array sentinels). "
      "The legacy filled-ptrs path does not transpose B-scales to MN-major and "
      "would produce incorrect results.");

  if (need_prep) {
    TORCH_CHECK(a.dim() == 2,
                "On-device prep requires flat 2D A (sum_m_i, K), got ", a.dim(), " dimensions");
    TORCH_CHECK(scales_a.dim() == 2,
                "On-device prep requires flat 2D scales_a (sum_m_i, K/BS), got ", scales_a.dim());
    TORCH_CHECK(output.dim() == 2,
                "On-device prep requires flat 2D output (sum_m_i, N), got ", output.dim());
    TORCH_CHECK(b.dim() == 3, "On-device prep requires 3D B (E, N, K)");
    TORCH_CHECK(scales_b.dim() == 3, "On-device prep requires 3D scales_b");
    TORCH_CHECK(b_ptrs.numel() == 0 && out_ptrs.numel() == 0 &&
                    a_scales_ptrs.numel() == 0 && b_scales_ptrs.numel() == 0,
                "On-device prep requires all ptr-array tensors to be empty");

    const int BS = is_mxfp8 ? at::native::xpu::MXFP8Types::BlockSize
                            : at::native::xpu::FP8Types::BlockSize;
    const int E = static_cast<int>(expert_offsets.size(0));
    const int K = static_cast<int>(a.size(1));
    const int N = static_cast<int>(output.size(1));
    TORCH_CHECK(K % BS == 0, "K must be a multiple of ", BS);
    const int scale_cols = K / BS;
    TORCH_CHECK(scales_a.size(1) == scale_cols,
                "scales_a flat shape mismatch; expected (*, K/BS) row-major");

    if (is_mxfp8) {
      TORCH_CHECK(
          scales_b.size(0) == E && scales_b.size(1) == N &&
              scales_b.size(2) == scale_cols,
          "MXFP8 scales_b shape mismatch; expected (E, N, K/BS) row-major un-transposed");
    }

    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(a.device());
    ptr_table_keep_alive = torch::empty({5, E}, opts_i64);

    auto& q = at::xpu::getCurrentXPUStream(a.device().index()).queue();

    // Element sizes are compile-time known: A/B are always fp8_e4m3 (1 B),
    // output is always fp32 (4 B; enforced in BlockScaledGroupedGemmRunner),
    // A/B scales are uint8 for MXFP8 or fp32 for FP8.
    constexpr int64_t a_elem  = sizeof(cutlass::float_e4m3_t);   // 1
    constexpr int64_t b_elem  = sizeof(cutlass::float_e4m3_t);   // 1
    constexpr int64_t o_elem  = sizeof(float);                   // 4
    const int64_t sa_elem = is_mxfp8 ? sizeof(uint8_t) : sizeof(float);
    const int64_t sb_elem = sa_elem;

    // Upper-bound the padded ragged-M A-scales scratch by total flat A rows.
    // sum(M_i) == a.size(0) by construction (on-device prep requires flat
    // 2D A), so no single expert's M can exceed a.size(0). Using this bound
    // instead of the exact max_m avoids a D->H reduction over problem_sizes
    // and keeps dispatch fully async — matches vLLM's SM100 shape. Trades a
    // small amount of scratch memory (up to E*scale_cols*(a.size(0) - true
    // max_m) bytes) for one fewer host sync per fused-experts call.
    const int max_m = static_cast<int>(a.size(0));
    TORCH_CHECK(max_m > 0, "flat A must have at least one row");

    // Xe block-2D scale loads require 4-byte aligned width/pitch: pad the
    // per-column M stride up to ScaleAlignElems = ceil_div(4, sizeof(scale)).
    // For u8 UE8M0 that's 4; for fp32 it's 1 (already aligned). Also
    // zero-init so the extra rows don't feed garbage into the accumulator.
    const int scale_align = is_mxfp8 ? 4 : 1;
    const int padded_max_m = (max_m + scale_align - 1) & ~(scale_align - 1);

    if (is_mxfp8) {
      // uint8 A-scales + uint8 B-scales (both transposed on device).
      auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(a.device());
      scales_a_t_keep_alive = torch::zeros({E, scale_cols, padded_max_m}, opts_u8);
      scales_b_t_keep_alive = torch::empty({E, scale_cols, N}, opts_u8);

      launch_u8_scale_build_pointers_and_transpose_scales_flat(
          q, E, padded_max_m, scale_cols, N,
          /*a_row_stride_bytes=*/K, static_cast<int>(o_elem),
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
    } else {
      // fp32 A-scales only (B-scales stay untransposed for FP8 row-major).
      // padded_max_m == max_m here (scale_align=1) but keep the name for
      // symmetry with the MXFP8 branch above.
      auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(a.device());
      scales_a_t_keep_alive = torch::zeros({E, scale_cols, padded_max_m}, opts_f32);

      launch_mxfp8_build_pointers_and_transpose_scales_flat(
          q, E, padded_max_m, scale_cols, N, K,
          static_cast<int>(a_elem), static_cast<int>(o_elem),
          problem_sizes.data_ptr<int32_t>(),
          expert_offsets.data_ptr<int32_t>(),
          scales_a.data_ptr<float>(),
          scales_a_t_keep_alive.data_ptr<float>(),
          ptr_table_keep_alive.data_ptr<int64_t>(),
          reinterpret_cast<int64_t>(a.data_ptr()),
          reinterpret_cast<int64_t>(b.data_ptr()),     b.stride(0) * b_elem,
          reinterpret_cast<int64_t>(output.data_ptr()),
          reinterpret_cast<int64_t>(scales_a_t_keep_alive.data_ptr()),
              scales_a_t_keep_alive.stride(0) * sa_elem,
          reinterpret_cast<int64_t>(scales_b.data_ptr()), scales_b.stride(0) * sb_elem,
          scales_a.stride(0),
          scales_a_t_keep_alive.stride(0));
    }

    a_ptrs        = ptr_table_keep_alive[0];
    b_ptrs        = ptr_table_keep_alive[1];
    out_ptrs      = ptr_table_keep_alive[2];
    a_scales_ptrs = ptr_table_keep_alive[3];
    b_scales_ptrs = ptr_table_keep_alive[4];
  }

  // On-device-prep: A-scales packed with M-stride max_m ⇒ pass override.
  // Legacy filled-ptrs path uses per-expert m_i (override = 0).
  const int sa_m_stride_override =
      need_prep ? static_cast<int>(scales_a_t_keep_alive.stride(1)) : 0;

  if (is_mxfp8) {
    at::native::xpu::MXFP8Runner::run(
        output, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs,
        a, b,
        need_prep ? scales_a_t_keep_alive : scales_a,
        need_prep ? scales_b_t_keep_alive : scales_b,
        problem_sizes, expert_offsets, workspace,
        sa_m_stride_override);
  } else {
    at::native::xpu::FP8Runner::run(
        output, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs,
        a, b,
        need_prep ? scales_a_t_keep_alive : scales_a,
        scales_b, problem_sizes, expert_offsets, workspace,
        sa_m_stride_override);
  }
}
