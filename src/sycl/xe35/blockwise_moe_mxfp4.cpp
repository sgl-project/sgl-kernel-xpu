/***************************************************************************************************
 * Copyright 2025 SGLang Team. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 **************************************************************************************************/
/*! \file
    \brief MXFP4 (E2M1 + UE8M0 scales, block=32) blockwise grouped GEMM for MoE on Intel XPU (xe35).
           HW-accelerated via XE_BDPAS.

    CUTE_ENABLE_XE_BLOCK_2D_ASSERT is deliberately not defined here, matching MXFP8.
    It gates only assert(), and FindSYCL.cmake builds these TUs without -DNDEBUG, so
    defining it makes the asserts live even in a Release build. The A-scale width one
    then reads each expert's real token count rather than our padded allocation and
    aborts on any count not divisible by 4 — most decode traffic. That load is safe
    regardless: CUTLASS rounds the width up to 4 right after checking it and the
    epilogue drops the extra rows.
*/

// clang-format off

#include "blockwise_moe_runner.hpp"
#include "sgl_kernel_export.h"
#include "moe_group_gemm_helper.hpp"

using namespace cute;
using namespace cutlass::gemm;

namespace at::native::xpu {

// MXFP4: E2M1 + UE8M0 scales, block=32, A=RowMajor, B=ColumnMajor, symmetric MN-major scale strides.
template <class TileShape_, class ElementOutput_>
struct MXFP4TypesT {

  using ElementType     = cutlass::mx_float4_t<float_e2m1_t>;
  using ElementInputA   = typename ElementType::DataType;    // float_e2m1_t
  using ElementInputB   = typename ElementType::DataType;    // float_e2m1_t
  using ElementScale    = typename ElementType::ScaleFactorType;  // float_ue8m0_t


  using ElementAccumulator       = float;
  using ElementComputeEpilogue   = float;
  using ElementOutput            = ElementOutput_;


  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  // Scale strides (both MN-major)
  using StrideScaleA = cute::Stride<_1, int64_t, int64_t>;
  using StrideScaleB = cute::Stride<_1, int64_t, int64_t>;


  static constexpr int BlockSize = 32;

  // Void selects CUTLASS's block-2D auto-detection path.
  using GmemTiledCopyA      = void;
  using GmemTiledCopyB      = void;
  using GmemTiledCopyScaleA = void;
  using GmemTiledCopyScaleB = void;

  using TileShape = TileShape_;
  using MmaAtom   = MMA_Atom<XE_BDPAS_TT<8, float, ElementInputA>>;

  using TiledMma = typename TiledMMAHelper<
      MmaAtom,
      cute::Layout<TileShape>,
      typename MoeSubgroupLayout<TileShape, MmaAtom>::type>::TiledMMA;

  // Mainloop dispatch (integer GroupSize → MXFP specialization)
  static constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy    = cutlass::gemm::MainloopIntelXeXMX16BlockScaledGroup<PipelineStages>;
  // Same generic group epilogue as MXFP8. The legacy XMX16 one builds its C
  // tensor from a raw ElementC, so it cannot take a void C, and it pins the
  // fusion op's output type to the accumulator, so it cannot store bf16.
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGenericGroup;


  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementOutput, ElementComputeEpilogue, ElementAccumulator, ElementAccumulator,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<
      EpilogueDispatchPolicy, EpilogueOp, TileShape, decltype(tile_shape(TiledMma()))>;

  // ElementC=void disables the C load, so the caller no longer has to zero the
  // output buffer. Void copy atoms let CUTLASS size the D store from ElementOutput.
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

// Shapes mirror MXFP8, including the 256x256 ceiling from the accumulator GRF
// budget — 4-bit packing halves the operand footprint but the accumulator is fp32
// either way, so the wider N_TILE=512 this used to run at step spilled 12 KB.
// Prefill has no bigger tile left to reach for. K_TILE=64 fixed (SG_K>=32,
// mainloop needs 2 K-slices).
//
// Templated on the output element so the dtype fan-out is one branch at the entry
// point and the bucket shapes stay written once.
template <typename ElementOutput_>
struct MXFP4TilePolicy {
  using Decode  = MXFP4TypesT<Shape<_16, _512, _64>, ElementOutput_>;
  using Step    = MXFP4TypesT<Shape<_256, _256, _64>, ElementOutput_>;
  using Prefill = Step;

  // Bucket-independent, so any member answers for it.
  static constexpr int BlockSize = Step::BlockSize;
};

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
  // bf16 halves the D store and lets the caller skip a separate fp32->bf16 pass;
  // fp32 stays for callers that want the accumulator verbatim.
  TORCH_CHECK(output.scalar_type() == torch::kFloat32 ||
                  output.scalar_type() == torch::kBFloat16,
              "Output must be float32 or bfloat16, got ", output.scalar_type());
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

  // Validate before the raw int32 reads below, which otherwise fault or throw
  // an opaque dtype error rather than naming the offending argument.
  TORCH_CHECK(problem_sizes.dim() == 2 && problem_sizes.size(1) == 3 &&
                  problem_sizes.scalar_type() == torch::kInt32 &&
                  problem_sizes.is_contiguous(),
              "problem_sizes must be a contiguous int32 (num_experts, 3) tensor");

  // Threaded into the runner so it skips its own D->H sync of problem_sizes.
  torch::Tensor problem_sizes_host = problem_sizes.to(torch::kCPU);
  const int32_t* psz = problem_sizes_host.data_ptr<int32_t>();
  const int E = static_cast<int>(problem_sizes_host.size(0));
  int max_m = 0;
  for (int e = 0; e < E; ++e) {
    if (psz[e * 3] > max_m) max_m = psz[e * 3];
  }

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

    // BlockSize is output-independent; float just picks an arbitrary instantiation.
    constexpr int BS = at::native::xpu::MXFP4TilePolicy<float>::BlockSize;
    TORCH_CHECK(expert_offsets.size(0) == E,
                "expert_offsets and problem_sizes disagree on num_experts");
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

    // Keep each expert's A-scale base 64-byte aligned for the scale loader;
    // zero-init because the transpose only fills the first m_i rows.
    const int padded_max_m =
        at::native::xpu::moe_pad_scale_m(max_m, scale_cols, static_cast<int>(sa_elem));

    scales_a_t_keep_alive = torch::zeros({E, scale_cols, padded_max_m}, opts_u8);
    scales_b_t_keep_alive = torch::empty({E, scale_cols, N}, opts_u8);

    launch_u8_scale_build_pointers_and_transpose_scales_flat(
        q, E, padded_max_m, scale_cols, N, /*a_row_stride_bytes=*/packed_K, static_cast<int>(o_elem),
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

  TORCH_CHECK(max_m > 0, "problem_sizes[:, 0] must contain at least one positive M");

  const torch::Tensor& scales_a_eff = need_prep ? scales_a_t_keep_alive : scales_a;
  const torch::Tensor& scales_b_eff = need_prep ? scales_b_t_keep_alive : scales_b;
  const auto bucket = at::native::xpu::moe_tile_bucket(max_m);

#define SGL_RUN_MXFP4(ELEMENT_OUTPUT)                                          \
  at::native::xpu::moe_run_bucketed<at::native::xpu::MXFP4TilePolicy<ELEMENT_OUTPUT>>( \
      bucket, output, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs,  \
      a, b, scales_a_eff, scales_b_eff, problem_sizes, expert_offsets,         \
      workspace, sa_m_stride_override, &problem_sizes_host)

  if (output.scalar_type() == torch::kFloat32) {
    SGL_RUN_MXFP4(float);
  } else {
    SGL_RUN_MXFP4(cutlass::bfloat16_t);
  }
#undef SGL_RUN_MXFP4
}
