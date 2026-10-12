/***************************************************************************************************
 * Copyright 2026 Intel corporation. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
/*! \file
    \brief FP8 (E4M3) blockwise-scaled GEMM on Intel XPU (xe35).

           Software-scaled path using CUTLASS
           MainloopIntelXeXMX16BlockScaledImpl<Stages, tuple<_1, _128, _128>>.
           Matches the semantics of the CUDA fp8_blockwise_scaled_mm kernel in
           the upstream sglang sgl-kernel repo:

             A  : [M, K] float8_e4m3fn, row-major
             B  : [K, N] float8_e4m3fn, column-major (mat_b.stride(0) == 1)
             SA : [M, K/128] float32,   M-major (scale_a.stride(0) == 1)
             SB : [K/128, N/128] float32, K-major (scale_b.stride(0) == 1)
             out: [M, N] in bfloat16 or float16
*/

#define SYCL_INTEL_TARGET 35

// clang-format off
#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <algorithm>
#include <cute/tensor.hpp>
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/float8.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

#include "../Utils.h"
#include "sgl_kernel_export.h"

using namespace cute;

namespace {

// Config for the software block-scaled FP8 GEMM on xe35.
//
// Block sizes are fixed at (1, 128, 128): per-row A scaling, 128x128 tiles for B.
// A: [M, K] row-major, B: [N, K] row-major (i.e., mat_b viewed as its transpose).
template <typename OutputT, class TileShape_, class WarpLayout>
struct Fp8BlockwiseGemmConfig {
  using ElementInputA        = cutlass::float_e4m3_t;
  using ElementInputB        = cutlass::float_e4m3_t;
  using ElementScale         = float;                     // fp32 scale factors
  using ElementAccumulator   = float;
  using ElementComputeEpilogue = float;
  using ElementOutput        = OutputT;

  using LayoutA = cutlass::layout::RowMajor;
  // B is fed as an [N, K] view whose K axis is contiguous (stride == 1). In
  // cutlass tag terms that maps to ColumnMajor for B (modes are (N, K, L),
  // TagToStrideB<ColumnMajor> = Stride<int64_t, _1, int64_t>).
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  // Scale strides in cute (mode 0 is contiguous):
  //   A: M-major -> Stride<_1, M, L>
  //   B: K-major -> Stride<K/BS, _1, L>
  using StrideScaleA = cute::Stride<cute::_1, int64_t, int64_t>;
  using StrideScaleB = cute::Stride<int64_t, cute::_1, int64_t>;

  using TileShape = TileShape_;

  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_BDPAS_TT<8, float, ElementInputA>>,
      cute::Layout<TileShape>,
      WarpLayout>::TiledMMA;

  // Software block-scaled path: fp8 DPAS, apply fp32 scaleA*scaleB per group.
  static constexpr int PipelineStages = 2;
  using GroupSizeMNK = cute::tuple<cute::_1, cute::Int<128>, cute::Int<128>>;
  // ...BlockScaledImpl (not the ...BlockScaled alias): the alias takes an `int`
  // GroupSize, while the per-MNK group shape here is a type.
  using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16BlockScaledImpl<PipelineStages, GroupSizeMNK>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGeneric;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementOutput,
      ElementComputeEpilogue,
      ElementAccumulator,
      ElementAccumulator,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using FusionCallbacks = cutlass::epilogue::fusion::FusionCallbacks<
      EpilogueDispatchPolicy,
      EpilogueOp,
      TileShape,
      decltype(tile_shape(TiledMma()))>;

  // ElementC = void: no source load; alpha=1, beta=0.
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
      EpilogueDispatchPolicy,
      TileShape,
      void,                                               // Epilogue tile (auto)
      void,                                               // ElementC = void
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      ElementOutput,
      cutlass::gemm::TagToStrideC_t<LayoutD>,
      FusionCallbacks,
      void, void>;

  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      GEMMDispatchPolicy,
      TileShape,
      cute::tuple<ElementInputA, ElementScale>,
      cute::tuple<cutlass::gemm::TagToStrideA_t<LayoutA>, StrideScaleA>,
      cute::tuple<ElementInputB, ElementScale>,
      cute::tuple<cutlass::gemm::TagToStrideB_t<LayoutB>, StrideScaleB>,
      TiledMma,
      // cute::type_list (not cute::tuple / std::tuple) because newer DPCPP's
      // SYCL kernel-name registrar tries to instantiate this template arg
      // and both {cute,std}::tuple<void, void> fail (can't hold void members).
      // cute::type_list is an empty struct; its std::tuple_element
      // specialization gives CollectiveMma what it needs at metafunction time.
      cute::type_list<void, void>, void, void, cute::identity,
      cute::type_list<void, void>, void, void, cute::identity>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename Config>
cutlass::Status run_fp8_blockwise_gemm(
    const at::Tensor& A_rm,       // [M, K] row-major, float8_e4m3fn
    const at::Tensor& B_rm,       // [N, K] row-major, float8_e4m3fn (transpose of the caller's B)
    const at::Tensor& SA,         // [M, K/128] float32, M-major (stride(0)==1)
    const at::Tensor& SB,         // [K/128, N/128] float32, K-major (stride(0)==1)
    at::Tensor& out,              // [M, N] OutputT
    const cutlass::KernelHardwareInfo& hw_info,
    sycl::queue* queue) {
  using Gemm = typename Config::Gemm;
  using GemmKernel = typename Gemm::GemmKernel;
  using ElementInputA = typename Config::ElementInputA;
  using ElementInputB = typename Config::ElementInputB;
  using ElementScale = typename Config::ElementScale;
  using ElementOutput = typename Config::ElementOutput;
  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;
  using StrideScaleA = typename Config::StrideScaleA;
  using StrideScaleB = typename Config::StrideScaleB;

  const int M = static_cast<int>(A_rm.size(0));
  const int K = static_cast<int>(A_rm.size(1));
  const int N = static_cast<int>(B_rm.size(0));
  const int L = 1;
  const int scale_k = K / 128;
  const int scale_n = cute::ceil_div(N, 128);

  auto problem_shape = cute::make_shape(M, N, K, L);
  auto shape_A       = cute::make_shape(M, K, L);
  auto shape_B       = cute::make_shape(N, K, L);
  auto shape_CD      = cute::make_shape(M, N, L);
  auto shape_scale_A = cute::make_shape(M, scale_k, L);
  auto shape_scale_B = cute::make_shape(scale_n, scale_k, L);

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, shape_A);
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, shape_B);
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, shape_CD);
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, shape_CD);
  StrideScaleA stride_SA = cutlass::make_cute_packed_stride(StrideScaleA{}, shape_scale_A);
  StrideScaleB stride_SB = cutlass::make_cute_packed_stride(StrideScaleB{}, shape_scale_B);

  typename Gemm::GemmKernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_shape,
      typename GemmKernel::MainloopArguments{
          static_cast<ElementInputA const*>(A_rm.data_ptr()), stride_A,
          static_cast<ElementInputB const*>(B_rm.data_ptr()), stride_B,
          static_cast<ElementScale const*>(SA.data_ptr()),   stride_SA,
          static_cast<ElementScale const*>(SB.data_ptr()),   stride_SB},
      typename GemmKernel::EpilogueArguments{
          {/*alpha=*/1.0f, /*beta=*/0.0f},
          /*ptr_C=*/nullptr, stride_C,
          static_cast<ElementOutput*>(out.data_ptr()), stride_D},
      hw_info};

  Gemm gemm_op;
  auto s = gemm_op.can_implement(arguments);
  if (s != cutlass::Status::kSuccess) return s;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  s = gemm_op.initialize(arguments, workspace.get());
  if (s != cutlass::Status::kSuccess) return s;

  return gemm_op.run(queue);
}

// Tile shapes and warp layouts for the software block-scaled path.
// Constraint from the mainloop: SG_N (= BLK_N / warps_N) <= GroupN (=128),
// and BLK_M/BLK_N/BLK_K must tile nicely with the MMA atom.
using TileShapeSmall = Shape<_128, _128, _32>;
using TileShapeLarge = Shape<_256, _256, _32>;
using WarpLayout44   = Layout<Shape<_4, _4, _1>, Stride<_4, _1, _0>>;
using WarpLayout84   = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;

template <typename OutputT>
cutlass::Status dispatch_fp8_blockwise(
    const at::Tensor& A_rm,
    const at::Tensor& B_rm,
    const at::Tensor& SA,
    const at::Tensor& SB,
    at::Tensor& out,
    const cutlass::KernelHardwareInfo& hw_info,
    sycl::queue* queue) {
  const int64_t M = A_rm.size(0);
  const int64_t N = B_rm.size(0);
  if (M <= 128 || N <= 128) {
    using Cfg = Fp8BlockwiseGemmConfig<OutputT, TileShapeSmall, WarpLayout44>;
    return run_fp8_blockwise_gemm<Cfg>(A_rm, B_rm, SA, SB, out, hw_info, queue);
  }
  using Cfg = Fp8BlockwiseGemmConfig<OutputT, TileShapeLarge, WarpLayout84>;
  return run_fp8_blockwise_gemm<Cfg>(A_rm, B_rm, SA, SB, out, hw_info, queue);
}

}  // namespace

// Public entry point. Registered in torch_extension_sycl.cc as
// "fp8_blockwise_scaled_mm" for the XPU dispatch key.
//
// Contract (matches the CUDA reference in the upstream sglang repo):
//   mat_a   : [M, K] float8_e4m3fn, row-major (mat_a.stride(1) == 1)
//   mat_b   : [K, N] float8_e4m3fn, column-major (mat_b.stride(0) == 1)
//   scales_a: [M, K/128] float32, M-major (scales_a.stride(0) == 1) or contiguous vector
//   scales_b: [K/128, N/128] float32, K-major (scales_b.stride(0) == 1) or contiguous vector
//   out_dtype: torch.bfloat16 or torch.float16
SGL_KERNEL_EXPORT torch::Tensor fp8_blockwise_scaled_mm(
    const torch::Tensor& mat_a,
    const torch::Tensor& mat_b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Dtype& out_dtype) {
  CHECK_DEVICE(mat_a);
  CHECK_DEVICE(mat_b);
  CHECK_DEVICE(scales_a);
  CHECK_DEVICE(scales_b);
  TORCH_CHECK(mat_a.dim() == 2 && mat_b.dim() == 2,
              "mat_a/mat_b must be 2D");
  TORCH_CHECK(mat_a.stride(1) == 1, "mat_a must be row-major");
  TORCH_CHECK(mat_b.stride(0) == 1, "mat_b must be column-major");
  TORCH_CHECK(mat_a.size(1) == mat_b.size(0), "K dimensions of mat_a and mat_b must match");
  TORCH_CHECK(mat_a.scalar_type() == torch::kFloat8_e4m3fn, "mat_a must be Float8_e4m3fn");
  TORCH_CHECK(mat_b.scalar_type() == torch::kFloat8_e4m3fn, "mat_b must be Float8_e4m3fn");
  TORCH_CHECK(scales_a.scalar_type() == torch::kFloat32, "scales_a must be Float32");
  TORCH_CHECK(scales_b.scalar_type() == torch::kFloat32, "scales_b must be Float32");
  TORCH_CHECK(out_dtype == torch::kBFloat16 || out_dtype == torch::kHalf,
              "out_dtype must be BFloat16 or Half");

  const int64_t M = mat_a.size(0);
  const int64_t K = mat_a.size(1);
  const int64_t N = mat_b.size(1);
  TORCH_CHECK(K % 128 == 0, "K must be a multiple of 128, got K=", K);
  TORCH_CHECK(N % 128 == 0, "N must be a multiple of 128, got N=", N);
  TORCH_CHECK((K * mat_a.element_size()) % 16 == 0,
              "K must be a multiple of 16 bytes for memory alignment");

  auto is_contiguous_vector = [](const torch::Tensor& t) {
    auto s = t.sizes();
    return t.is_contiguous() &&
           (t.dim() == 1 || (t.dim() == 2 && *std::min_element(s.begin(), s.end()) == 1));
  };

  TORCH_CHECK(scales_a.dim() == 2 && scales_a.size(0) == M && scales_a.size(1) == K / 128,
              "scales_a must have shape [M, K/128], got ", scales_a.sizes());
  TORCH_CHECK(scales_a.stride(0) == 1 || is_contiguous_vector(scales_a),
              "scales_a must be M-major (stride(0) == 1)");
  TORCH_CHECK(scales_b.dim() == 2 && scales_b.size(0) == K / 128 && scales_b.size(1) == N / 128,
              "scales_b must have shape [K/128, N/128], got ", scales_b.sizes());
  TORCH_CHECK(scales_b.stride(0) == 1 || is_contiguous_vector(scales_b),
              "scales_b must be K-major (stride(0) == 1)");

  auto out_options = mat_a.options().dtype(out_dtype);
  torch::Tensor out = torch::empty({M, N}, out_options);

  // Row-major A: contiguous if not already.
  torch::Tensor A_rm = mat_a.is_contiguous() ? mat_a : mat_a.contiguous();
  // B is column-major [K, N]; we feed it as row-major [N, K] via a transpose view
  // (no copy — .t() flips the strides and the buffer is already the right shape).
  torch::Tensor B_rm = mat_b.t();
  TORCH_CHECK(B_rm.is_contiguous(),
              "internal error: transposed B expected to be row-major contiguous");

  c10::DeviceGuard guard(mat_a.device());
  auto stream = at::xpu::getCurrentXPUStream(mat_a.device().index());
  sycl::queue& queue = stream.queue();

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = mat_a.device().index();
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  cutlass::Status status;
  if (out_dtype == torch::kBFloat16) {
    status = dispatch_fp8_blockwise<cutlass::bfloat16_t>(A_rm, B_rm, scales_a, scales_b, out, hw_info, &queue);
  } else {
    status = dispatch_fp8_blockwise<cutlass::half_t>(A_rm, B_rm, scales_a, scales_b, out, hw_info, &queue);
  }
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "fp8_blockwise_scaled_mm failed: ", cutlassGetStatusString(status));
  return out;
}
