/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
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

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cute/tensor.hpp>

#include "Utils.h"
#include "comm/common.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/sycl_event_manager.hpp"

using namespace cute;

template <typename Kernel>
class Fp8ScaledGemmKernel {};

// Kernel runner template
template <typename Gemm, typename ElementOutput>
struct Fp8GemmRunner {
  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementC = typename Gemm::ElementC;
  using LayoutA = typename Gemm::LayoutA;
  using LayoutB = typename Gemm::LayoutB;
  using LayoutC = typename Gemm::LayoutC;
  using LayoutD = typename Gemm::LayoutD;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  using CollectiveMainloop = typename Gemm::CollectiveMainloop;
  using ElementScale = typename CollectiveMainloop::NonVoidElementScaleA;
  using StrideScale = typename CollectiveMainloop::NonVoidStrideScaleA;

  cutlass::Status
  run(const at::Tensor& mat_a,
      const at::Tensor& mat_b,
      const at::Tensor& scales_a,
      const at::Tensor& scales_b,
      at::Tensor& out,
      const cutlass::KernelHardwareInfo& hw_info) {
    int M = mat_a.size(0);
    int N = mat_b.size(1);
    int K = mat_a.size(1);

    // Setup problem shape
    auto problem_shape = cute::make_shape(M, N, K, 1);

    // Setup strides
    auto shape_A = cute::make_shape(M, K, 1);
    auto shape_B = cute::make_shape(N, K, 1);
    auto shape_CD = cute::make_shape(M, N, 1);
    auto shape_scale_A = cute::make_shape(M, 1, 1);
    auto shape_scale_B = cute::make_shape(N, 1, 1);

    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, shape_A);
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, shape_B);
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, shape_CD);
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, shape_CD);
    StrideScale stride_SA = cutlass::make_cute_packed_stride(StrideScale{}, shape_scale_A);
    StrideScale stride_SB = cutlass::make_cute_packed_stride(StrideScale{}, shape_scale_B);

    float alpha = 1.0f;
    float beta = 0.0f;

    // Create a dummy C tensor
    cutlass::device_memory::allocation<ElementC> dummy_C(M * N);

    // Prepare arguments
    typename Gemm::GemmKernel::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_shape,
        {static_cast<ElementA*>(mat_a.data_ptr()),
         stride_A,
         static_cast<ElementB*>(mat_b.data_ptr()),
         stride_B,
         static_cast<ElementScale*>(scales_a.data_ptr()),
         stride_SA,
         static_cast<ElementScale*>(scales_b.data_ptr()),
         stride_SB,
         nullptr,
         stride_SA,  // No zero point for A
         nullptr,
         stride_SB,  // No zero point for B
         K},         // group_size = K for per-row/col scaling
        {{alpha, beta}, dummy_C.get(), stride_C, static_cast<ElementOutput*>(out.data_ptr()), stride_D},
        hw_info};

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }

    status = gemm_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
      return status;
    }

    status = gemm_op.run();
    return status;
  }
};

// Configure GEMM based on output dtype and input FP8 type
template <typename ElementInputFp8, typename ElementOutput>
struct Fp8GemmConfig {
  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ElementInputA = ElementInputFp8;
  using ElementInputB = ElementInputFp8;
  using ElementScale = cutlass::half_t;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using StrideScale = cute::Stride<_1, int64_t, int64_t>;

  using GmemTiledCopyA = XE_2D_U8x32x32_LD_N;
  using GmemTiledCopyB = XE_2D_U8x32x32_LD_V;

  using TileShape = Shape<_256, _256, _32>;

  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_8x16x16_F32F16F16F32_TT>,
      Layout<TileShape>,
      Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

  static constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16FP8Scaling<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementOutput,
      ElementComputeEpilogue,
      ElementAccumulator,
      ElementAccumulator,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using FusionCallBacks = cutlass::epilogue::fusion::
      FusionCallbacks<EpilogueDispatchPolicy, EpilogueOp, TileShape, decltype(tile_shape(TiledMma()))>;

  // Use U16 store for FP16/BF16
  using GmemTiledCopyStore = XE_2D_U16x8x16_ST_N;

  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
      EpilogueDispatchPolicy,
      TileShape,
      ElementAccumulator,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      ElementOutput,
      cutlass::gemm::TagToStrideC_t<LayoutD>,
      FusionCallBacks,
      XE_2D_U32x8x16_LD_N,
      void,
      void,
      GmemTiledCopyStore,
      void,
      void>;

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

// Helper to check if output is FP8
static inline bool is_fp8_dtype(at::ScalarType dtype) {
  return dtype == at::ScalarType::Float8_e4m3fn || dtype == at::ScalarType::Float8_e5m2;
}

// Helper function to get FP8 min/max values
static inline std::pair<float, float> get_fp8_range(at::ScalarType dtype) {
  if (dtype == at::ScalarType::Float8_e4m3fn) {
    // E4M3FN: max = 448, min = -448
    return {-448.0f, 448.0f};
  } else {
    // Float8_e5m2
    // E5M2: max = 57344, min = -57344
    return {-57344.0f, 57344.0f};
  }
}

// Helper function to dispatch based on input FP8 type and output dtype
template <typename ElementInputFp8>
static at::Tensor fp8_scaled_mm_impl(
    const at::Tensor& mat_a,
    const at::Tensor& mat_b,
    const at::Tensor& scales_a_half,
    const at::Tensor& scales_b_half,
    const at::ScalarType out_dtype,
    at::Tensor& out,
    const cutlass::KernelHardwareInfo& hw_info) {
  at::Tensor mat_a_contig = mat_a.contiguous();
  at::Tensor mat_b_contig = mat_b.contiguous();

  cutlass::Status status;

  if (out_dtype == at::ScalarType::BFloat16) {
    using Config = Fp8GemmConfig<ElementInputFp8, cutlass::bfloat16_t>;
    Fp8GemmRunner<typename Config::Gemm, cutlass::bfloat16_t> runner;
    status = runner.run(mat_a_contig, mat_b_contig, scales_a_half, scales_b_half, out, hw_info);
  } else {  // Half - used for both FP16 output and FP8 intermediate
    using Config = Fp8GemmConfig<ElementInputFp8, cutlass::half_t>;
    Fp8GemmRunner<typename Config::Gemm, cutlass::half_t> runner;
    status = runner.run(mat_a_contig, mat_b_contig, scales_a_half, scales_b_half, out, hw_info);
  }

  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "FP8 GEMM failed with status: " + std::string(cutlassGetStatusString(status)));

  return out;
}

// Main entry point
at::Tensor fp8_scaled_mm(
    const at::Tensor& mat_a,
    const at::Tensor& mat_b,
    const at::Tensor& scales_a,
    const at::Tensor& scales_b,
    const at::ScalarType out_dtype,
    const c10::optional<at::Tensor>& bias) {
  // Input validation
  auto input_dtype = mat_a.scalar_type();
  TORCH_CHECK(
      input_dtype == at::ScalarType::Float8_e4m3fn || input_dtype == at::ScalarType::Float8_e5m2,
      "mat_a must be Float8_e4m3fn or Float8_e5m2");
  TORCH_CHECK(mat_b.scalar_type() == input_dtype, "mat_a and mat_b must have the same dtype");
  TORCH_CHECK(scales_a.scalar_type() == at::ScalarType::Float, "scales_a must be Float32");
  TORCH_CHECK(scales_b.scalar_type() == at::ScalarType::Float, "scales_b must be Float32");
  TORCH_CHECK(
      out_dtype == at::ScalarType::BFloat16 || out_dtype == at::ScalarType::Half ||
          out_dtype == at::ScalarType::Float8_e4m3fn || out_dtype == at::ScalarType::Float8_e5m2,
      "out_dtype must be BFloat16, Float16, Float8_e4m3fn, or Float8_e5m2");

  CHECK_DEVICE(mat_a);
  CHECK_DEVICE(mat_b);
  CHECK_DEVICE(scales_a);
  CHECK_DEVICE(scales_b);

  TORCH_CHECK(mat_a.dim() == 2, "mat_a must be 2D");
  TORCH_CHECK(mat_b.dim() == 2, "mat_b must be 2D");

  int M = mat_a.size(0);
  int K = mat_a.size(1);
  int K_b = mat_b.size(0);
  int N = mat_b.size(1);

  TORCH_CHECK(K == K_b, "Inner dimensions must match");
  TORCH_CHECK(scales_a.size(0) == M, "scales_a must have size M");
  TORCH_CHECK(scales_b.size(0) == N, "scales_b must have size N");
  TORCH_CHECK(scales_a.is_contiguous(), "scales_a must be contiguous");
  TORCH_CHECK(scales_b.is_contiguous(), "scales_b must be contiguous");

  // Convert scales to half precision for GEMM
  at::Tensor scales_a_half = scales_a.to(at::ScalarType::Half).contiguous();
  at::Tensor scales_b_half = scales_b.to(at::ScalarType::Half).contiguous();

  // Convert back to FP32 for precise calculations
  at::Tensor scales_a_for_unscale = scales_a_half.to(at::ScalarType::Float);
  at::Tensor scales_b_for_unscale = scales_b_half.to(at::ScalarType::Float);

  // For FP8 output, use FP16 intermediate or requested out dtype
  at::ScalarType intermediate_dtype;
  if (is_fp8_dtype(out_dtype)) {
    intermediate_dtype = at::ScalarType::Half;
  } else {
    intermediate_dtype = out_dtype;
  }

  auto opts = mat_a.options().dtype(intermediate_dtype);
  at::Tensor out_intermediate = torch::empty({M, N}, opts);

  c10::DeviceGuard device_guard(mat_a.device());

  // Get hardware info
  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  // Dispatch based on input FP8 type
  if (input_dtype == at::ScalarType::Float8_e4m3fn) {
    fp8_scaled_mm_impl<cutlass::float_e4m3_t>(
        mat_a, mat_b, scales_a_half, scales_b_half, intermediate_dtype, out_intermediate, hw_info);
  } else {
    fp8_scaled_mm_impl<cutlass::float_e5m2_t>(
        mat_a, mat_b, scales_a_half, scales_b_half, intermediate_dtype, out_intermediate, hw_info);
  }

  at::Tensor out = out_intermediate;

  // Add bias if present (before FP8 quantization)
  if (bias.has_value()) {
    at::Tensor bias_tensor = bias.value();
    CHECK_DEVICE(bias_tensor);
    TORCH_CHECK(bias_tensor.size(0) == N, "bias must have size N");
    TORCH_CHECK(bias_tensor.is_contiguous(), "bias must be contiguous");

    if (is_fp8_dtype(out_dtype)) {
      // Convert bias to intermediate dtype
      at::Tensor bias_converted = bias_tensor.to(intermediate_dtype);
      out.add_(bias_converted.view({1, N}));
    } else {
      TORCH_CHECK(bias_tensor.scalar_type() == out_dtype, "bias must have same dtype as output");
      out.add_(bias_tensor.view({1, N}));
    }
  }

  // Quantize to FP8 if needed
  if (is_fp8_dtype(out_dtype)) {
    // Get FP8 range based on output dtype
    auto [fp8_min, fp8_max] = get_fp8_range(out_dtype);

    // Convert to FP32 for quantization operations
    out = out.to(at::ScalarType::Float);

    // Per-element scaling: reverse the input scaling
    at::Tensor scale_a_safe = scales_a_for_unscale.abs().clamp_min(1e-10f);
    at::Tensor scale_b_safe = scales_b_for_unscale.abs().clamp_min(1e-10f);

    at::Tensor scale_matrix = scale_a_safe.view({-1, 1}) * scale_b_safe.view({1, -1});

    // Reverse the per-element scaling
    at::Tensor out_unscaled = out / scale_matrix;

    // Replace any NaN/Inf with 0
    out_unscaled = torch::where(torch::isfinite(out_unscaled), out_unscaled, torch::zeros_like(out_unscaled));

    // Compute global quantization scale from unscaled values
    float amax = out_unscaled.abs().max().item<float>();

    // Check for invalid amax
    if (amax < 1e-10f || std::isnan(amax) || std::isinf(amax)) {
      amax = 1e-10f;
    }

    // Compute quantization scale
    float quant_scale = amax / fp8_max;

    // Quantize: scale down, clamp, and convert to FP8
    out = out_unscaled.div_(quant_scale).clamp_(fp8_min, fp8_max).to(out_dtype);
  }

  return out;
}
