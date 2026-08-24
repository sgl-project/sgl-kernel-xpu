/***************************************************************************************************
 * Copyright 2025 SGLang Team. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 **************************************************************************************************/
/*! \file
    \brief NVFP4 weight-only-quantized MXFP8 grouped GEMM for MoE on Intel XPU (xe35).

    B arrives as NVFP4 (packed E2M1 payload) and is dequantized to BF16 and
    requantized to MXFP8 inside the mainloop; A is already MXFP8.  See
    nvfp4_woq_mxfp8_mainloop.hpp for the numerics and the register-level layout
    reasoning, and src/sycl/nvfp4_woq.cpp for the one-time scale prep that
    produces the (c, sfmx) pair this kernel consumes.

    NOTE: unlike blockwise_moe_mxfp4.cpp this TU does NOT define
    CUTE_ENABLE_XE_BLOCK_2D_ASSERT.  Those asserts are active in device code
    even in Release builds, and two of them fire on inputs this kernel handles
    correctly by construction: the A-scale surface width is the per-expert token
    count, which MoE routing makes arbitrary (the traits code rounds it up, and
    the scratch buffer is padded to match), and the base-pointer check flags
    scale strides the host side now guarantees explicitly.  Leaving them on
    floods stderr with hundreds of failures per launch on ordinary ragged-M
    shapes.  The epilogue is still the MXFP4-style one (ElementC=float with
    explicit copy atoms) since that pairing is known good for a 4-bit B operand.
*/

// clang-format off
#include <utility>

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <torch/extension.h>

#include <cute/tensor.hpp>
#include <cute/arch/mma_xe.hpp>
#include <cute/util/compat.hpp>

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/float8.h"
#include "cutlass/float_subbyte.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "moe_group_gemm_helper.hpp"
#include "nvfp4_woq_bf16_mainloop.hpp"
#include "nvfp4_woq_mxfp8_mainloop.hpp"

using namespace cute;
using namespace cutlass::gemm;

namespace at::native::xpu {

using NvFp4GroupedProblemShape = cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;
using NvFp4UnderlyingProblemShapeType = typename NvFp4GroupedProblemShape::UnderlyingProblemShape;

// NVFP4-B x MXFP8-A: E2M1 payload for B, E4M3 for A, UE8M0 block scales for
// both (B's are the requantization exponents produced by nvfp4_woq_prep).
struct NvFp4WoqTypes {
  using ElementInputA = cutlass::float_e4m3_t;
  using ElementInputB = cutlass::float_e2m1_t;
  using ElementScale  = typename cutlass::mx_float8_t<cutlass::float_e4m3_t>::ScaleFactorType;
  using ElementMult   = cutlass::bfloat16_t;

  using ElementAccumulator     = float;
  using ElementComputeEpilogue = float;
  using ElementOutput          = float;

  using LayoutA = cutlass::layout::RowMajor;
  // B is (N, K) K-contiguous; ColumnMajor reproduces the CUTLASS example's
  // RowMajor (K, N) access pattern without a transpose.
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  // Both scale tensors are MN-major, matching the Xe block-2D scale atoms.
  using StrideScaleA = cute::Stride<cute::_1, int64_t, int64_t>;
  using StrideScaleB = cute::Stride<cute::_1, int64_t, int64_t>;

  static constexpr int BlockSize     = 32;  // MXFP8 / bdpas block
  static constexpr int MultBlockSize = 16;  // NVFP4 block

  // Void selects CUTLASS's block-2D auto-detection path.
  using GmemTiledCopyA      = void;
  using GmemTiledCopyB      = void;
  using GmemTiledCopyScaleA = void;
  using GmemTiledCopyScaleB = void;

  // 256x256x64 with a 4x4 subgroup layout gives SG_M=64, SG_N=64, SG_K=64.
  //
  // SG_N must stay at 64. With SG_N=32 the block-scaled scale copy narrows to
  // MXFP_SCALE_LOAD_2D<8, H, 32, 32> (scale_traits_num = 1), a configuration the
  // working MXFP8 kernel never selects, and (E=4, ragged M=16, N=256, K=64)
  // then produced a non-deterministic all-zero 32-column block — a different
  // (expert, subgroup) on each launch. SG_N=64 is bit-exact and deterministic on
  // the same shape. Do not narrow N without re-running that case.
  //
  // GRF budget. The grf_size<512> in run_large_grf below is a no-op: that SYCL
  // property is dropped before SPIR-V, so it never reaches IGC (the sibling
  // sub_group_size<16> does survive, as !intel_reqd_sub_group_size). The thread
  // register file is selected by the build instead — this TU is listed in
  // XE35_GEMM_KERNEL_PATTERNS (src/BuildOnLinux.cmake), which compiles it with
  // -cl-intel-512-GRF-per-thread; IGC's auto-large-GRF heuristic used elsewhere
  // only ever picks 128 or 256.
  //
  // At 512 GRF this mainloop no longer spills (verified in the IGC asm dump:
  // no `.spill size`, RA drops from GRAPH_COLORING_SPILL_FF_BC_RA to
  // GRAPH_COLORING_FF_BC_RA, and sends in the k-tile body fall 651 -> 114).
  // Measured effect on the grouped GEMM, device time, E=2 N=1536 K=2048 M=1024:
  // 4669 us -> 3720 us.
  //
  // SG_M=64 is what 512 GRF buys. The B operand is dequantized and requantized
  // per subgroup, so that cost is paid once per SG_M rows of A: doubling SG_M
  // halves the conversion work per MAC. At 256 GRF the 64x64 FP32 accumulator
  // alone (256 GRFs) filled the budget, which is why every earlier attempt at
  // SG_M=64 spilled and lost. At 512 GRF none of these variants spill, and
  // SG_M=64 via 4x4 wins. Grouped-GEMM device time, N=1536 K=2048 unless noted:
  //
  //                              8x4 <256,..>   8x4 <512,..>   4x4 <256,..>
  //                                 SG_M=32       SG_M=64        SG_M=64
  //   E=2 m/g=512                    3719 us       3895 us      2972 us  -20%
  //   E=2 m/g=1750                  11103        [ 7742 ]       8866     -20%
  //   E=2 m/g=512 N=2048 K=768       1434         1597          1228     -14%
  //   E=8 m/g=64                     7420         7711          5913     -20%
  //   E=8 m/g=8                      7420         7707          5912     -20%
  //
  // 4x4 keeps BLK_M at 256; the 8x4 <512,...> column doubles it, which only pays
  // off at large per-expert M (bracketed) and wastes most of the tile at decode
  // sizes, where B conversion is charged for rows that do not exist.
  //
  // Superseded numbers, all taken at 256 GRF while spilling -- kept only so they
  // are not re-derived as evidence against SG_M=64:
  //   CRI simulator, E=2, N=256, K=64 (CUTLASS GPU_Clock)
  //     SG_M=32: decode 0.0485 ms, prefill 0.0504 ms
  //     SG_M=64: decode 0.1004 ms, prefill 0.1016 ms
  //   IGC output: SG_M=32 12,928-byte spill / 8,171 instr
  //               SG_M=64 19,840-byte spill / 10,434 instr
  //   <256,256,64> 8x4  SG_M=32              29.9
  //   <256,256,64> 4x4  SG_M=64  308 spills  33.0
  //   <512,256,64> 8x4  SG_M=64  309 spills  41.1
  //   <256,256,64> 8x4  + Stages=3           30.0   (within noise)
  using TileShape    = Shape<_256, _256, _64>;
  using ThreadLayout = cute::Layout<cute::Shape<cute::_4, cute::_4, cute::_1>,
                                    cute::Stride<cute::_4, cute::_1, cute::_0>>;

  using TiledMma = typename cute::TiledMMAHelper<
      cute::MMA_Atom<cute::XE_BDPAS_TT<8, float, ElementInputA>>,
      cute::Layout<TileShape>,
      ThreadLayout>::TiledMMA;

  static constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy     = sgl_xpu::MainloopXeNvFp4WoqMxFp8Group<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16Group;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementOutput, ElementComputeEpilogue, ElementAccumulator, ElementAccumulator,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<
      EpilogueDispatchPolicy, EpilogueOp, TileShape, decltype(cute::tile_shape(TiledMma()))>;

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
      // cute::type_list (not cute/std::tuple) — see blockwise_moe_mxfp8.cpp.
      cute::type_list<GmemTiledCopyA, GmemTiledCopyScaleA>,
      void, void, cute::identity,
      cute::type_list<GmemTiledCopyB, GmemTiledCopyScaleB>,
      void, void, cute::identity>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      NvFp4GroupedProblemShape, CollectiveMainloop, CollectiveEpilogue, cutlass::gemm::GroupScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;
};

using Gemm = NvFp4WoqTypes::Gemm;
using GemmKernel = Gemm::GemmKernel;

template <class GemmAdapter>
cutlass::Status run_large_grf(
    GemmAdapter const& gemm_op,
    sycl::queue& queue) {
  using Kernel = typename GemmAdapter::GemmKernel;
  auto params = gemm_op.params();
  compat::dim3 const block = Kernel::get_block_shape();
  compat::dim3 const grid = Kernel::get_grid_shape(params);
  constexpr int smem_size = Kernel::SharedStorageSize;

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  compat::experimental::launch_properties launch_props{
      syclex::work_group_scratch_size(smem_size)};
  compat::experimental::kernel_properties kernel_props{
      syclex::sub_group_size<16>, intelex::grf_size<512>};
  compat::experimental::launch_policy policy{
      grid, block, launch_props, kernel_props};

#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
  auto event = compat::experimental::launch<
      cutlass::device_kernel<Kernel>,
      Kernel>(policy, queue, params);
  EventManager::getInstance().addEvent(event);
#else
  compat::experimental::launch<
      cutlass::device_kernel<Kernel>,
      Kernel,
      false>(policy, queue, params);
#endif
  return cutlass::Status::kSuccess;
}

/// Launch the grouped GEMM. All pointer arrays are device-resident.
void run_nvfp4_woq_gemm(
    int num_groups,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& ptr_table,   // {5, E} int64: a, b, out, sfa, sfb
    const torch::Tensor& cb_ptrs,     // {E} int64
    const torch::Tensor& cb_ld,       // {E} int64
    int sfa_m_stride,
    const torch::Tensor& workspace,
    at::Device device) {
  auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(device);

  auto M_col = problem_sizes.select(1, 0);
  auto N_col = problem_sizes.select(1, 1);
  auto K_col = problem_sizes.select(1, 2);

  torch::Tensor stride_A_dev = K_col.to(torch::kInt64).contiguous();
  torch::Tensor stride_B_dev = K_col.to(torch::kInt64).contiguous();
  torch::Tensor stride_CD_dev = N_col.to(torch::kInt64).contiguous();

  auto zeros = torch::zeros({num_groups}, opts_i64);
  // A scales: Stride<_1, padded_max_m, 0>; B scales: Stride<_1, N, 0>.
  auto stride_SFA_dev = torch::stack(
      {torch::full({num_groups}, static_cast<int64_t>(sfa_m_stride), opts_i64), zeros}, 1).contiguous();
  auto stride_SFB_dev = torch::stack({N_col.to(torch::kInt64), zeros}, 1).contiguous();

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = static_cast<int>(device.index());
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  auto stream = at::xpu::getCurrentXPUStream(device.index());
  sycl::queue& queue = stream.queue();

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

  using RasterOrderOptions = typename cutlass::gemm::kernel::detail::
      PersistentTileSchedulerXeGroup<NvFp4GroupedProblemShape>::RasterOrderOptions;

  using ElementInputA = NvFp4WoqTypes::ElementInputA;
  using ElementInputB = NvFp4WoqTypes::ElementInputB;
  using ElementScale = NvFp4WoqTypes::ElementScale;
  using ElementMult = NvFp4WoqTypes::ElementMult;
  using ElementOutput = NvFp4WoqTypes::ElementOutput;

  auto row = [&](int i) { return ptr_table[i].data_ptr(); };

  typename GemmKernel::Arguments gemm_args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      typename GemmKernel::ProblemShape{
          num_groups,
          reinterpret_cast<NvFp4UnderlyingProblemShapeType*>(problem_sizes.data_ptr<int32_t>()),
          nullptr},
      typename GemmKernel::MainloopArguments{
          reinterpret_cast<ElementInputA const**>(row(0)),
          reinterpret_cast<NvFp4WoqTypes::StrideA*>(stride_A_dev.data_ptr<int64_t>()),
          reinterpret_cast<ElementInputB const**>(row(1)),
          reinterpret_cast<NvFp4WoqTypes::StrideB*>(stride_B_dev.data_ptr<int64_t>()),
          reinterpret_cast<ElementScale const**>(row(3)),
          reinterpret_cast<NvFp4WoqTypes::StrideScaleA*>(stride_SFA_dev.data_ptr<int64_t>()),
          reinterpret_cast<ElementScale const**>(row(4)),
          reinterpret_cast<NvFp4WoqTypes::StrideScaleB*>(stride_SFB_dev.data_ptr<int64_t>()),
          reinterpret_cast<ElementMult const**>(cb_ptrs.data_ptr()),
          cb_ld.data_ptr<int64_t>()},
      typename GemmKernel::EpilogueArguments{
          fusion_args,
          reinterpret_cast<ElementOutput const**>(row(2)),
          reinterpret_cast<NvFp4WoqTypes::StrideC*>(stride_CD_dev.data_ptr<int64_t>()),
          reinterpret_cast<ElementOutput**>(row(2)),
          reinterpret_cast<NvFp4WoqTypes::StrideD*>(stride_CD_dev.data_ptr<int64_t>())},
      hw_info,
      typename GemmKernel::TileSchedulerArguments{1, RasterOrderOptions::AlongN}};

  Gemm gemm_op;
  // can_implement is intentionally skipped: it dereferences the host
  // problem-shape pointer (we pass nullptr to keep dispatch async) and rejects
  // M == 0 groups, which MoE decode produces constantly and the Xe group tile
  // scheduler already skips. Same rationale as blockwise_moe_runner.hpp.
  size_t workspace_size = Gemm::get_workspace_size(gemm_args);
  TORCH_CHECK(static_cast<size_t>(workspace.numel()) >= workspace_size,
              "nvfp4_woq_mxfp8_scaled_grouped_mm: workspace insufficient, need ", workspace_size, " bytes");
  TORCH_CHECK(gemm_op.initialize(gemm_args, workspace.data_ptr()) == cutlass::Status::kSuccess,
              "nvfp4_woq_mxfp8_scaled_grouped_mm: failed to initialize");
  TORCH_CHECK(run_large_grf(gemm_op, queue) == cutlass::Status::kSuccess,
              "nvfp4_woq_mxfp8_scaled_grouped_mm: failed to run");
}

// NVFP4-B x BF16-A: B is converted and scaled to BF16 in registers, then a
// regular BF16 DPAS performs the matrix multiply with FP32 accumulation.
struct NvFp4WoqBf16Types {
  using ElementInputA = cutlass::bfloat16_t;
  using ElementInputB = cutlass::float_e2m1_t;
  using ElementScale = uint8_t;
  using ElementMult = cutlass::bfloat16_t;
  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ElementOutput = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  // K=32 holds two NVFP4 scale groups and keeps the wider BF16 B fragment
  // within the same register footprint as the K=64 MXFP8 path.
  //
  // This path keeps SG_M=32 (8x4) even though the MXFP8 WoQ path above moved to
  // SG_M=64. SG_M=64 was measured both ways here at 512 GRF and lost every time:
  //   4x4 <256,256,32>  SG_M=64 : +36% (gemm1 M=1024 1225 -> 1667 us)
  //   8x4 <512,256,32>  SG_M=64 : +47% (                  1225 -> 1802 us)
  // Neither variant spills, so this is not register pressure. With BLK_K=32 this
  // mainloop runs twice the k-iterations of the MXFP8 path for half the work
  // each, which leaves it latency- rather than ALU-bound; halving the subgroups
  // per workgroup (4x4) or doubling the tile (BLK_M=512) both cost more in
  // latency hiding than the extra SG_M saves in B-conversion work.
  using TileShape = Shape<_256, _256, _32>;
  using ThreadLayout = cute::Layout<cute::Shape<cute::_8, cute::_4, cute::_1>,
                                    cute::Stride<cute::_4, cute::_1, cute::_0>>;
  using TiledMma = typename cute::TiledMMAHelper<
      cute::MMA_Atom<cute::XE_DPAS_TT<8, float, ElementInputA>>,
      cute::Layout<TileShape>,
      ThreadLayout>::TiledMMA;

  static constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy =
      sgl_xpu::MainloopXeNvFp4WoqBf16Group<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16Group;

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
      decltype(cute::tile_shape(TiledMma()))>;
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
      ElementInputA,
      cutlass::gemm::TagToStrideA_t<LayoutA*>,
      ElementInputB,
      cutlass::gemm::TagToStrideB_t<LayoutB*>,
      TiledMma,
      void,
      void,
      void,
      cute::identity,
      void,
      void,
      void,
      cute::identity>;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      NvFp4GroupedProblemShape,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::GroupScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename GemmKernel::InternalStrideA;
  using StrideB = typename GemmKernel::InternalStrideB;
  using StrideC = typename GemmKernel::InternalStrideC;
  using StrideD = typename GemmKernel::InternalStrideD;
};

void run_nvfp4_woq_bf16_gemm(
    int num_groups,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& ptr_table,  // {4, E}: a, b, out, sfmx
    const torch::Tensor& cb_ptrs,
    const torch::Tensor& cb_ld,
    const torch::Tensor& workspace,
    at::Device device) {
  using Types = NvFp4WoqBf16Types;
  using Bf16Gemm = Types::Gemm;
  using Bf16GemmKernel = Types::GemmKernel;

  auto N_col = problem_sizes.select(1, 1);
  auto K_col = problem_sizes.select(1, 2);
  auto stride_A_dev = K_col.to(torch::kInt64).contiguous();
  auto stride_B_dev = K_col.to(torch::kInt64).contiguous();
  auto stride_CD_dev = N_col.to(torch::kInt64).contiguous();

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = static_cast<int>(device.index());
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  auto stream = at::xpu::getCurrentXPUStream(device.index());
  sycl::queue& queue = stream.queue();

  typename Bf16Gemm::Arguments arguments;
  decltype(arguments.epilogue.thread) fusion_args;
  fusion_args.alpha = 1.0f;
  fusion_args.beta = 0.0f;
  fusion_args.alpha_ptr = nullptr;
  fusion_args.beta_ptr = nullptr;
  fusion_args.alpha_ptr_array = nullptr;
  fusion_args.beta_ptr_array = nullptr;
  fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
  fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};

  using RasterOrderOptions = typename cutlass::gemm::kernel::detail::
      PersistentTileSchedulerXeGroup<NvFp4GroupedProblemShape>::RasterOrderOptions;
  auto row = [&](int i) { return ptr_table[i].data_ptr(); };

  typename Bf16GemmKernel::Arguments gemm_args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      typename Bf16GemmKernel::ProblemShape{
          num_groups,
          reinterpret_cast<NvFp4UnderlyingProblemShapeType*>(
              problem_sizes.data_ptr<int32_t>()),
          nullptr},
      typename Bf16GemmKernel::MainloopArguments{
          reinterpret_cast<Types::ElementInputA const**>(row(0)),
          reinterpret_cast<Types::StrideA*>(stride_A_dev.data_ptr<int64_t>()),
          reinterpret_cast<Types::ElementInputB const**>(row(1)),
          reinterpret_cast<Types::StrideB*>(stride_B_dev.data_ptr<int64_t>()),
          reinterpret_cast<Types::ElementScale const**>(row(3)),
          reinterpret_cast<Types::ElementMult const**>(cb_ptrs.data_ptr()),
          cb_ld.data_ptr<int64_t>()},
      typename Bf16GemmKernel::EpilogueArguments{
          fusion_args,
          reinterpret_cast<Types::ElementOutput const**>(row(2)),
          reinterpret_cast<Types::StrideC*>(stride_CD_dev.data_ptr<int64_t>()),
          reinterpret_cast<Types::ElementOutput**>(row(2)),
          reinterpret_cast<Types::StrideD*>(stride_CD_dev.data_ptr<int64_t>())},
      hw_info,
      typename Bf16GemmKernel::TileSchedulerArguments{
          1, RasterOrderOptions::AlongN}};

  Bf16Gemm gemm_op;
  const size_t workspace_size = Bf16Gemm::get_workspace_size(gemm_args);
  TORCH_CHECK(
      static_cast<size_t>(workspace.numel()) >= workspace_size,
      "nvfp4_woq_bf16_grouped_mm: workspace insufficient, need ",
      workspace_size,
      " bytes");
  TORCH_CHECK(
      gemm_op.initialize(gemm_args, workspace.data_ptr()) ==
          cutlass::Status::kSuccess,
      "nvfp4_woq_bf16_grouped_mm: failed to initialize");
  TORCH_CHECK(
      gemm_op.run(&queue) == cutlass::Status::kSuccess,
      "nvfp4_woq_bf16_grouped_mm: failed to run");
}

}  // namespace at::native::xpu

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
//
// Flat-2D ragged-M layout, matching fp8_blockwise_scaled_grouped_mm's
// on-device-prep path: A / scales_a / output are 2D with per-expert row
// offsets in expert_offsets, weights are 3D per expert.
void nvfp4_woq_mxfp8_scaled_grouped_mm(
    torch::Tensor& output,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& sfmx_b,
    const torch::Tensor& c_b,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& workspace) {
  using Types = at::native::xpu::NvFp4WoqTypes;
  constexpr int BS = Types::BlockSize;
  constexpr int MBS = Types::MultBlockSize;
  constexpr int BLK_K = cute::get<2>(Types::TileShape{});

  TORCH_CHECK(a.device().is_xpu(), "nvfp4_woq: A must be on XPU");
  TORCH_CHECK(b.device().is_xpu(), "nvfp4_woq: B must be on XPU");
  TORCH_CHECK(output.device().is_xpu(), "nvfp4_woq: output must be on XPU");
  TORCH_CHECK(workspace.device().is_xpu(), "nvfp4_woq: workspace must be on XPU");

  TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn, "nvfp4_woq: A must be float8_e4m3fn (MXFP8)");
  TORCH_CHECK(b.scalar_type() == torch::kUInt8, "nvfp4_woq: B must be uint8 (packed E2M1)");
  TORCH_CHECK(scales_a.scalar_type() == torch::kUInt8, "nvfp4_woq: scales_a must be uint8 (UE8M0)");
  TORCH_CHECK(sfmx_b.scalar_type() == torch::kUInt8, "nvfp4_woq: sfmx_b must be uint8 (UE8M0)");
  TORCH_CHECK(c_b.scalar_type() == torch::kBFloat16, "nvfp4_woq: c_b must be bfloat16");
  TORCH_CHECK(output.scalar_type() == torch::kFloat32, "nvfp4_woq: output must be float32");

  TORCH_CHECK(a.dim() == 2, "nvfp4_woq: A must be flat 2D (sum_m, K)");
  TORCH_CHECK(scales_a.dim() == 2, "nvfp4_woq: scales_a must be flat 2D (sum_m, K/32)");
  TORCH_CHECK(output.dim() == 2, "nvfp4_woq: output must be flat 2D (sum_m, N)");
  TORCH_CHECK(b.dim() == 3, "nvfp4_woq: B must be 3D (E, N, K/2)");
  TORCH_CHECK(sfmx_b.dim() == 3, "nvfp4_woq: sfmx_b must be 3D (E, K/32, N)");
  TORCH_CHECK(c_b.dim() == 3, "nvfp4_woq: c_b must be 3D (E, N, K/16)");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && scales_a.is_contiguous() &&
                  sfmx_b.is_contiguous() && c_b.is_contiguous() && output.is_contiguous(),
              "nvfp4_woq: all tensors must be contiguous");
  TORCH_CHECK(problem_sizes.dim() == 2 && problem_sizes.size(1) == 3,
              "nvfp4_woq: problem_sizes must be (E, 3) int32");
  TORCH_CHECK(problem_sizes.scalar_type() == torch::kInt32 && expert_offsets.scalar_type() == torch::kInt32,
              "nvfp4_woq: problem_sizes / expert_offsets must be int32");

  const int E = static_cast<int>(expert_offsets.size(0));
  const int K = static_cast<int>(a.size(1));
  const int N = static_cast<int>(output.size(1));
  TORCH_CHECK(E > 0, "nvfp4_woq: expert count must be positive");
  TORCH_CHECK(problem_sizes.size(0) == E, "nvfp4_woq: problem_sizes / expert_offsets expert mismatch");
  TORCH_CHECK(K % BLK_K == 0, "nvfp4_woq: K must be a multiple of ", BLK_K, " (got ", K, ")");
  TORCH_CHECK(N % 4 == 0, "nvfp4_woq: N must be 4-byte aligned for the MN-major scale load (got ", N, ")");

  const int scale_k = K / BS;
  TORCH_CHECK(scales_a.size(1) == scale_k, "nvfp4_woq: scales_a must be (*, K/32)");
  TORCH_CHECK(b.size(0) == E && b.size(1) == N && b.size(2) == K / 2,
              "nvfp4_woq: B must be (E, N, K/2)");
  TORCH_CHECK(sfmx_b.size(0) == E && sfmx_b.size(1) == scale_k && sfmx_b.size(2) == N,
              "nvfp4_woq: sfmx_b must be (E, K/32, N) MN-major");
  TORCH_CHECK(c_b.size(0) == E && c_b.size(1) == N && c_b.size(2) == K / MBS,
              "nvfp4_woq: c_b must be (E, N, K/16) row-major");

  // Every per-expert B-scale slice becomes the base address of an Xe block-2D
  // load, and those must be 64-byte aligned. Torch's allocator aligns the
  // tensor itself, so the requirement lands on the per-expert stride.
  TORCH_CHECK(sfmx_b.stride(0) % 64 == 0,
              "nvfp4_woq: N * (K/32) must be a multiple of 64 so each expert's B-scale slice is "
              "64-byte aligned for the Xe block-2D load; got N=", N, " K=", K,
              " (per-expert stride ", sfmx_b.stride(0), " bytes)");

  auto device = a.device();
  auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(device);
  auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(device);

  // Bound the padded ragged-M A-scale scratch by the total flat row count so
  // dispatch stays async (no D->H reduction over problem_sizes).
  const int max_m = static_cast<int>(a.size(0));
  TORCH_CHECK(max_m > 0, "nvfp4_woq: flat A must have at least one row");
  // Pad the M extent to 64 rather than the 4 the surface width alone needs, so
  // that scale_k * padded_max_m — the per-expert stride, and therefore every
  // per-expert A-scale base address — is 64-byte aligned for any scale_k.
  constexpr int scale_align = 64;
  const int padded_max_m = (max_m + scale_align - 1) & ~(scale_align - 1);

  auto ptr_table = torch::empty({5, E}, opts_i64);
  auto scales_a_t = torch::zeros({E, scale_k, padded_max_m}, opts_u8);

  auto& q = at::xpu::getCurrentXPUStream(device.index()).queue();
  launch_u8_scale_build_pointers_and_transpose_scales_flat(
      q, E, padded_max_m, scale_k, N,
      /*a_row_stride_bytes=*/K, /*out_elem_bytes=*/static_cast<int>(sizeof(float)),
      problem_sizes.data_ptr<int32_t>(),
      expert_offsets.data_ptr<int32_t>(),
      scales_a.data_ptr<uint8_t>(),
      scales_a_t.data_ptr<uint8_t>(),
      ptr_table.data_ptr<int64_t>(),
      reinterpret_cast<int64_t>(a.data_ptr()),
      reinterpret_cast<int64_t>(b.data_ptr()), b.stride(0) * static_cast<int64_t>(b.element_size()),
      reinterpret_cast<int64_t>(output.data_ptr()),
      reinterpret_cast<int64_t>(scales_a_t.data_ptr()), scales_a_t.stride(0),
      reinterpret_cast<int64_t>(sfmx_b.data_ptr()), sfmx_b.stride(0),
      scales_a.stride(0),
      scales_a_t.stride(0));

  // Per-expert c pointers and row stride; plain arithmetic on device tensors
  // keeps this out of the SYCL prep kernel.
  auto cb_ptrs = torch::arange(E, opts_i64) * (c_b.stride(0) * c_b.element_size()) +
                 reinterpret_cast<int64_t>(c_b.data_ptr());
  auto cb_ld = torch::full({E}, static_cast<int64_t>(c_b.stride(1)), opts_i64);

  at::native::xpu::run_nvfp4_woq_gemm(
      E, problem_sizes, ptr_table, cb_ptrs.contiguous(), cb_ld,
      static_cast<int>(scales_a_t.stride(1)), workspace, device);
}

class NvFp4WoqBf16BuildPointers;

void nvfp4_woq_bf16_grouped_mm(
    torch::Tensor& output,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& sfmx_b,
    const torch::Tensor& c_b,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& workspace) {
  using Types = at::native::xpu::NvFp4WoqBf16Types;
  constexpr int BLK_K = cute::get<2>(Types::TileShape{});

  TORCH_CHECK(a.device().is_xpu(), "nvfp4_woq_bf16: A must be on XPU");
  TORCH_CHECK(b.device().is_xpu(), "nvfp4_woq_bf16: B must be on XPU");
  TORCH_CHECK(output.device().is_xpu(), "nvfp4_woq_bf16: output must be on XPU");
  TORCH_CHECK(workspace.device().is_xpu(), "nvfp4_woq_bf16: workspace must be on XPU");
  TORCH_CHECK(a.scalar_type() == torch::kBFloat16, "nvfp4_woq_bf16: A must be bfloat16");
  TORCH_CHECK(b.scalar_type() == torch::kUInt8, "nvfp4_woq_bf16: B must be uint8 (packed E2M1)");
  TORCH_CHECK(sfmx_b.scalar_type() == torch::kUInt8, "nvfp4_woq_bf16: sfmx_b must be uint8 (UE8M0)");
  TORCH_CHECK(c_b.scalar_type() == torch::kBFloat16, "nvfp4_woq_bf16: c_b must be bfloat16");
  TORCH_CHECK(output.scalar_type() == torch::kFloat32, "nvfp4_woq_bf16: output must be float32");
  TORCH_CHECK(a.dim() == 2, "nvfp4_woq_bf16: A must be flat 2D (sum_m, K)");
  TORCH_CHECK(output.dim() == 2, "nvfp4_woq_bf16: output must be flat 2D (sum_m, N)");
  TORCH_CHECK(b.dim() == 3, "nvfp4_woq_bf16: B must be 3D (E, N, K/2)");
  TORCH_CHECK(sfmx_b.dim() == 3, "nvfp4_woq_bf16: sfmx_b must be 3D (E, K/32, N)");
  TORCH_CHECK(c_b.dim() == 3, "nvfp4_woq_bf16: c_b must be 3D (E, N, K/16)");
  TORCH_CHECK(
      a.is_contiguous() && b.is_contiguous() && sfmx_b.is_contiguous() &&
          c_b.is_contiguous() && output.is_contiguous(),
      "nvfp4_woq_bf16: all tensors must be contiguous");
  TORCH_CHECK(
      problem_sizes.dim() == 2 && problem_sizes.size(1) == 3,
      "nvfp4_woq_bf16: problem_sizes must be (E, 3) int32");
  TORCH_CHECK(
      problem_sizes.scalar_type() == torch::kInt32 &&
          expert_offsets.scalar_type() == torch::kInt32,
      "nvfp4_woq_bf16: problem_sizes / expert_offsets must be int32");

  const int E = static_cast<int>(expert_offsets.size(0));
  const int K = static_cast<int>(a.size(1));
  const int N = static_cast<int>(output.size(1));
  TORCH_CHECK(E > 0, "nvfp4_woq_bf16: expert count must be positive");
  TORCH_CHECK(
      problem_sizes.size(0) == E,
      "nvfp4_woq_bf16: problem_sizes / expert_offsets expert mismatch");
  TORCH_CHECK(
      output.size(0) == a.size(0),
      "nvfp4_woq_bf16: A and output flat row counts must match");
  TORCH_CHECK(
      K % BLK_K == 0,
      "nvfp4_woq_bf16: K must be a multiple of ",
      BLK_K,
      " (got ",
      K,
      ")");
  TORCH_CHECK(
      b.size(0) == E && b.size(1) == N && b.size(2) == K / 2,
      "nvfp4_woq_bf16: B must be (E, N, K/2)");
  TORCH_CHECK(
      sfmx_b.size(0) == E && sfmx_b.size(1) == K / 32 &&
          sfmx_b.size(2) == N,
      "nvfp4_woq_bf16: sfmx_b must be (E, K/32, N)");
  TORCH_CHECK(
      c_b.size(0) == E && c_b.size(1) == N && c_b.size(2) == K / 16,
      "nvfp4_woq_bf16: c_b must be (E, N, K/16)");

  auto device = a.device();
  auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(device);
  auto ptr_table = torch::empty({4, E}, opts_i64);

  const int32_t* offsets_ptr = expert_offsets.data_ptr<int32_t>();
  int64_t* ptrs = ptr_table.data_ptr<int64_t>();
  const int64_t a_base = reinterpret_cast<int64_t>(a.data_ptr());
  const int64_t b_base = reinterpret_cast<int64_t>(b.data_ptr());
  const int64_t out_base = reinterpret_cast<int64_t>(output.data_ptr());
  const int64_t sfmx_base = reinterpret_cast<int64_t>(sfmx_b.data_ptr());
  const int64_t a_row_bytes = static_cast<int64_t>(K) * a.element_size();
  const int64_t out_row_bytes = static_cast<int64_t>(N) * output.element_size();
  const int64_t b_expert_bytes = b.stride(0) * b.element_size();
  const int64_t sfmx_expert_bytes = sfmx_b.stride(0) * sfmx_b.element_size();

  auto& queue = at::xpu::getCurrentXPUStream(device.index()).queue();
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<NvFp4WoqBf16BuildPointers>(
        sycl::range<1>(static_cast<size_t>(E)), [=](sycl::id<1> id) {
          const int e = static_cast<int>(id[0]);
          const int64_t row = static_cast<int64_t>(offsets_ptr[e]);
          ptrs[0 * E + e] = a_base + row * a_row_bytes;
          ptrs[1 * E + e] = b_base + static_cast<int64_t>(e) * b_expert_bytes;
          ptrs[2 * E + e] = out_base + row * out_row_bytes;
          ptrs[3 * E + e] =
              sfmx_base + static_cast<int64_t>(e) * sfmx_expert_bytes;
        });
  });

  auto cb_ptrs =
      torch::arange(E, opts_i64) * (c_b.stride(0) * c_b.element_size()) +
      reinterpret_cast<int64_t>(c_b.data_ptr());
  auto cb_ld =
      torch::full({E}, static_cast<int64_t>(c_b.stride(1)), opts_i64);
  at::native::xpu::run_nvfp4_woq_bf16_gemm(
      E,
      problem_sizes,
      ptr_table,
      cb_ptrs.contiguous(),
      cb_ld,
      workspace,
      device);
}
