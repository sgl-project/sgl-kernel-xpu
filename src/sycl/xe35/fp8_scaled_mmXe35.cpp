/***************************************************************************************************
 * Copyright 2026 Intel corporation. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#define SYCL_INTEL_TARGET 35

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

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

#define CUTLASS_CHECK(status)              \
  {                                        \
    auto s_ = (status);                    \
    if (s_ != cutlass::Status::kSuccess) { \
      return s_;                           \
    }                                      \
  }

using namespace cute;

// Epilogue/store traits based on output type
template <typename OutputT>
struct EpilogueTraits {
  using GmemTiledCopyC = XE_2D_U32x8x16_LD_N;
  using GmemTiledCopyD = XE_2D_U32x8x16_ST_N;  // 32-bit store for float
};

template <>
struct EpilogueTraits<cutlass::bfloat16_t> {
  using GmemTiledCopyC = XE_2D_U32x8x16_LD_N;
  using GmemTiledCopyD = XE_2D_U16x8x16_ST_N;  // 16-bit store for bf16
};

template <>
struct EpilogueTraits<cutlass::half_t> {
  using GmemTiledCopyC = XE_2D_U32x8x16_LD_N;
  using GmemTiledCopyD = XE_2D_U16x8x16_ST_N;  // 16-bit store for f16
};

// Base config parametrized by TileShape, WarpLayout, and Output type.
//
// Memory layout convention:
//   A:  [M, K] row-major contiguous - LayoutA = RowMajor
//   B:  [K, N] row-major contiguous - LayoutB = RowMajor
//       (The public API receives B as [N, K] and transposes it internally.)
//   C/D: [M, N] row-major contiguous
//
// The XE_2D_U8x32x32_LD_V copy atom for B requires this specific physical layout.
template <typename ElementFp8, class TileShape, class WarpLayout, typename OutputT>
struct Fp8ScaledGemmConfigT {
  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ElementInputA = ElementFp8;
  using ElementInputB = ElementFp8;
  using ElementOutput = OutputT;
  using ElementScale = cutlass::half_t;  // FP16 scales
  using ElementBias = ElementOutput;     // Bias matches output dtype

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using GmemTiledCopyA = XE_2D_U8x32x32_LD_N;
  using GmemTiledCopyB = XE_2D_U8x32x32_LD_V;
  using GmemTiledCopyC = typename EpilogueTraits<OutputT>::GmemTiledCopyC;
  using GmemTiledCopyD = typename EpilogueTraits<OutputT>::GmemTiledCopyD;

  using TiledMma =
      typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32F16F16F32_TT>, Layout<TileShape>, WarpLayout>::TiledMMA;

  static constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16FP8Scaling<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16;

  using EpilogueOp = cutlass::epilogue::fusion::LinCombPerRowBias<
      ElementOutput,
      ElementComputeEpilogue,
      ElementBias,
      ElementAccumulator,
      ElementAccumulator,
      128 / sizeof_bits_v<ElementBias>,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using FusionCallbacks = cutlass::epilogue::fusion::
      FusionCallbacks<EpilogueDispatchPolicy, EpilogueOp, TileShape, decltype(tile_shape(TiledMma()))>;

  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
      EpilogueDispatchPolicy,
      TileShape,
      ElementAccumulator,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      ElementOutput,
      cutlass::gemm::TagToStrideC_t<LayoutD>,
      FusionCallbacks,
      GmemTiledCopyC,
      void,
      void,
      GmemTiledCopyD,
      void,
      void>;

  // Scale stride: MN-major with runtime dims for K-groups and batch.
  using StrideScale = cute::Stride<_1, int64_t, int64_t>;

  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      GEMMDispatchPolicy,
      TileShape,
      cute::tuple<ElementInputA, ElementScale, StrideScale>,
      cutlass::gemm::TagToStrideA_t<LayoutA>,
      cute::tuple<ElementInputB, ElementScale, StrideScale>,
      cutlass::gemm::TagToStrideB_t<LayoutB>,
      TiledMma,
      GmemTiledCopyA,
      void,
      void,
      cute::identity,
      GmemTiledCopyB,
      void,
      void,
      cute::identity>;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// Runner
// mat_b_t is the transposed weight buffer: [K, N] contiguous (row-major).
// The GEMM computes D[m,n] = sum_k A[m,k] * B_t[k,n], which is equivalent
// to A @ B^T where B is the original [N, K] weight matrix.
template <typename Gemm>
struct Fp8ScaledGemmRunner {
  using CollectiveMainloop = typename Gemm::CollectiveMainloop;
  using CollectiveEpilogue = typename Gemm::CollectiveEpilogue;

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementOutput = typename CollectiveEpilogue::ElementOutput;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  using ElementScaleA = typename CollectiveMainloop::NonVoidElementScaleA;
  using ElementScaleB = typename CollectiveMainloop::NonVoidElementScaleB;
  using StrideScaleA = typename CollectiveMainloop::NonVoidStrideScaleA;
  using StrideScaleB = typename CollectiveMainloop::NonVoidStrideScaleB;

  using StrideBias = Stride<_1, _0, int64_t>;  // per-row bias layout

  cutlass::Status
  run(const at::Tensor& mat_a,    // [M, K] contiguous
      const at::Tensor& mat_b_t,  // [K, N] contiguous (transposed B)
      const at::Tensor& scale_a,  // [M] in FP16
      const at::Tensor& scale_b,  // [N] in FP16
      const int64_t N,            // original N dimension
      const c10::optional<at::Tensor>& bias_opt,
      at::Tensor& out,
      const cutlass::KernelHardwareInfo& hw_info,
      sycl::queue* queue) {
    const int64_t M = mat_a.size(0);
    const int64_t K = mat_a.size(1);
    const int64_t L = 1;

    auto problem_shape = cute::make_shape(int(M), int(N), int(K), int(L));
    auto shape_A = cute::make_shape(int(M), int(K), int(L));
    auto shape_B = cute::make_shape(int(N), int(K), int(L));
    auto shape_CD = cute::make_shape(int(M), int(N), int(L));

    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, shape_A);
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, shape_B);
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, shape_CD);
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, shape_CD);

    // Per-row scaling: one scale group covers the entire K dimension.
    // scale_k = ceil_div(K, K) = 1, so shape is (M_or_N, 1, L).
    const int scale_k = 1;
    auto shape_scaleA = cute::make_shape(int(M), scale_k, int(L));
    auto shape_scaleB = cute::make_shape(int(N), scale_k, int(L));
    StrideScaleA stride_SA = cutlass::make_cute_packed_stride(StrideScaleA{}, shape_scaleA);
    StrideScaleB stride_SB = cutlass::make_cute_packed_stride(StrideScaleB{}, shape_scaleB);

    StrideBias dBias{};
    get<2>(dBias) = int64_t(0);

    float alpha = 1.0f;
    float beta = 0.0f;

    using EpilogueArgs = typename Gemm::GemmKernel::EpilogueArguments;
    EpilogueArgs epilogue_args{
        {alpha, beta},
        /*C*/ nullptr,
        stride_C,
        static_cast<ElementOutput*>(out.data_ptr()),
        stride_D};

    if (bias_opt.has_value()) {
      auto bias = bias_opt.value();
      epilogue_args.thread.bias_ptr = static_cast<ElementOutput*>(bias.data_ptr());
      epilogue_args.thread.dBias = dBias;
    } else {
      epilogue_args.thread.bias_ptr = nullptr;
      epilogue_args.thread.dBias = StrideBias{};
    }

    typename Gemm::GemmKernel::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_shape,
        {static_cast<ElementA*>(mat_a.data_ptr()),
         stride_A,
         static_cast<ElementB*>(mat_b_t.data_ptr()),
         stride_B,
         static_cast<ElementScaleA*>(scale_a.data_ptr()),
         stride_SA,
         static_cast<ElementScaleB*>(scale_b.data_ptr()),
         stride_SB,
         /*zeroA*/ nullptr,
         stride_SA,
         /*zeroB*/ nullptr,
         stride_SB,
         /*g (group size K means per-row scaling)*/ int(K)},
        epilogue_args,
        hw_info};

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    CUTLASS_CHECK(gemm_op.can_implement(arguments));
    CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));
    CUTLASS_CHECK(gemm_op.run(queue));

    return cutlass::Status::kSuccess;
  }
};

// Tile shapes and warp layouts
using TileShapeWide = Shape<_256, _128, _32>;
using TileShapeSquare = Shape<_256, _256, _32>;
using TileShapeSmall = Shape<_128, _128, _32>;

using WarpLayoutLarge = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;
using WarpLayoutSmall = Layout<Shape<_4, _4, _1>, Stride<_4, _1, _0>>;

// Dispatcher over the validated shapes
template <typename ElementFp8, typename OutputT>
cutlass::Status run_fp8_scaled_gemm_dispatch(
    const at::Tensor& A,
    const at::Tensor& B_t,  // [K, N] contiguous
    const at::Tensor& SA,
    const at::Tensor& SB,
    const int64_t N,
    const c10::optional<at::Tensor>& Bias,
    at::Tensor& out,
    const cutlass::KernelHardwareInfo& hw_info,
    sycl::queue* queue) {
  const int64_t M = A.size(0);

  if (M <= 128 && N <= 128) {
    using Config = Fp8ScaledGemmConfigT<ElementFp8, TileShapeSmall, WarpLayoutSmall, OutputT>;
    return Fp8ScaledGemmRunner<typename Config::Gemm>{}.run(A, B_t, SA, SB, N, Bias, out, hw_info, queue);
  } else if (N <= 128) {
    using Config = Fp8ScaledGemmConfigT<ElementFp8, TileShapeWide, WarpLayoutLarge, OutputT>;
    return Fp8ScaledGemmRunner<typename Config::Gemm>{}.run(A, B_t, SA, SB, N, Bias, out, hw_info, queue);
  } else {
    using Config = Fp8ScaledGemmConfigT<ElementFp8, TileShapeSquare, WarpLayoutLarge, OutputT>;
    return Fp8ScaledGemmRunner<typename Config::Gemm>{}.run(A, B_t, SA, SB, N, Bias, out, hw_info, queue);
  }
}

// Public API
// mat_a: [M, K] FP8 e4m3 activations
// mat_b: [N, K] FP8 e4m3 weights (same layout as PyTorch Linear.weight)
// scale_a: [M] FP32 per-row activation scales
// scale_b: [N] FP32 per-row weight scales (per-column in output space)
// out_dtype: float32, bfloat16, or float16
// bias: optional [M] in out_dtype per-row bias
// Returns: [M, N] tensor in out_dtype = diag(scale_a) @ (A @ B^T) @ diag(scale_b) + bias
torch::Tensor fp8_scaled_mm_xpu(
    const torch::Tensor& mat_a,
    const torch::Tensor& mat_b,
    const torch::Tensor& scale_a,
    const torch::Tensor& scale_b,
    const torch::Dtype& out_dtype,
    const c10::optional<torch::Tensor>& bias_opt) {
  TORCH_CHECK(mat_a.dim() == 2 && mat_b.dim() == 2, "mat_a/mat_b must be 2D");
  TORCH_CHECK(mat_a.size(1) == mat_b.size(1), "Inner dimension K must match");
  TORCH_CHECK(scale_a.dim() == 1 && scale_a.size(0) == mat_a.size(0), "scale_a must be [M]");
  TORCH_CHECK(scale_b.dim() == 1 && scale_b.size(0) == mat_b.size(0), "scale_b must be [N]");

  TORCH_CHECK(mat_a.is_xpu(), "mat_a must be on XPU. Got device=", mat_a.device());
  TORCH_CHECK(mat_b.is_xpu(), "mat_b must be on XPU. Got device=", mat_b.device());
  TORCH_CHECK(scale_a.is_xpu(), "scale_a must be on XPU. Got device=", scale_a.device());
  TORCH_CHECK(scale_b.is_xpu(), "scale_b must be on XPU. Got device=", scale_b.device());
  TORCH_CHECK(mat_b.device() == mat_a.device(), "mat_b must be on the same device as mat_a");
  TORCH_CHECK(scale_a.device() == mat_a.device(), "scale_a must be on the same device as mat_a");
  TORCH_CHECK(scale_b.device() == mat_a.device(), "scale_b must be on the same device as mat_a");

  if (bias_opt.has_value()) {
    TORCH_CHECK(bias_opt->dim() == 1 && bias_opt->size(0) == mat_a.size(0), "bias must be [M]");
    TORCH_CHECK(
        bias_opt->is_xpu() && bias_opt->device() == mat_a.device(), "bias must be on the same XPU device as mat_a");
    TORCH_CHECK(
        bias_opt->scalar_type() == out_dtype,
        "bias dtype must match out_dtype. bias=",
        bias_opt->scalar_type(),
        " out_dtype=",
        out_dtype);
  }

  TORCH_CHECK(
      mat_a.scalar_type() == at::ScalarType::Float8_e4m3fn && mat_b.scalar_type() == at::ScalarType::Float8_e4m3fn,
      "mat_a/mat_b must both be float8_e4m3fn");

  TORCH_CHECK(
      out_dtype == at::kFloat || out_dtype == at::kBFloat16 || out_dtype == at::kHalf,
      "out_dtype must be float, bfloat16, or float16");

  auto is_aligned_16 = [](const at::Tensor& t) { return (reinterpret_cast<uintptr_t>(t.data_ptr()) % 16) == 0; };
  TORCH_CHECK(is_aligned_16(mat_a), "mat_a must be 16-byte aligned");
  TORCH_CHECK(is_aligned_16(mat_b), "mat_b must be 16-byte aligned");

  const int64_t M = mat_a.size(0);
  const int64_t N = mat_b.size(0);

  at::Tensor A = mat_a.contiguous();

  // The CUTLASS FP8 mainloop with XE_2D_U8x32x32_LD_V expects B as a [K, N]
  // row-major buffer. Our input mat_b is [N, K] contiguous, so we transpose
  // to get the required physical layout.
  at::Tensor B_t = mat_b.t().contiguous();  // [K, N] contiguous

  // Per-row scales converted to FP16 (matches kernel ElementScale = half_t)
  at::Tensor SA = scale_a.to(at::kHalf).contiguous();  // [M]
  at::Tensor SB = scale_b.to(at::kHalf).contiguous();  // [N]

  c10::optional<at::Tensor> Bias;
  if (bias_opt.has_value()) {
    Bias = bias_opt->contiguous();
  }

  auto out_options = A.options().dtype(out_dtype);
  at::Tensor out = at::empty({M, N}, out_options);

  c10::DeviceGuard guard(mat_a.device());
  auto stream = at::xpu::getCurrentXPUStream(mat_a.device().index());
  sycl::queue& queue = stream.queue();

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = mat_a.device().index();
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  cutlass::Status status;
  if (out_dtype == at::kFloat) {
    status = run_fp8_scaled_gemm_dispatch<cutlass::float_e4m3_t, float>(A, B_t, SA, SB, N, Bias, out, hw_info, &queue);
  } else if (out_dtype == at::kBFloat16) {
    status = run_fp8_scaled_gemm_dispatch<cutlass::float_e4m3_t, cutlass::bfloat16_t>(
        A, B_t, SA, SB, N, Bias, out, hw_info, &queue);
  } else {  // at::kHalf
    status = run_fp8_scaled_gemm_dispatch<cutlass::float_e4m3_t, cutlass::half_t>(
        A, B_t, SA, SB, N, Bias, out, hw_info, &queue);
  }

  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "fp8_scaled_mm_xpu failed with status: " + std::string(cutlassGetStatusString(status)));

  return out;
}

#undef SYCL_INTEL_TARGET
