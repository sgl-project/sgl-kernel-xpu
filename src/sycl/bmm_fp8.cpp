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
class BmmFP8Kernel {};

// Kernel runner template
template <typename Gemm, typename ElementOutput>
struct BmmFP8Runner {
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
    int L = mat_a.size(0);
    int N = mat_b.size(2);
    int M = mat_a.size(1);
    int K = mat_a.size(2);

    // Setup problem shape
    auto problem_shape = cute::make_shape(M, N, K, L);

    // Setup strides
    auto shape_A = cute::make_shape(M, K, L);
    auto shape_B = cute::make_shape(N, K, L);
    auto shape_CD = cute::make_shape(M, N, L);
    auto shape_scale_A = cute::make_shape(1, 1, 1);
    auto shape_scale_B = cute::make_shape(1, 1, 1);

    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, shape_A);
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, shape_B);
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, shape_CD);
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, shape_CD);

    float scale_value = scales_a.item<float>();
    auto scale_tensora =
        torch::full({M}, static_cast<float>(scale_value), torch::dtype(torch::kFloat16).device(torch::kXPU));
    cutlass::half_t* ptr_scale_A = reinterpret_cast<cutlass::half_t*>(scale_tensora.data_ptr<at::Half>());
    scale_value = scales_b.item<float>();
    auto scale_tensorb =
        torch::full({N}, static_cast<float>(scale_value), torch::dtype(torch::kFloat16).device(torch::kXPU));
    cutlass::half_t* ptr_scale_B = reinterpret_cast<cutlass::half_t*>(scale_tensorb.data_ptr<at::Half>());
    StrideScale stride_SA = cute::make_stride(Int<1>{}, 0L, 0L);
    StrideScale stride_SB = cute::make_stride(Int<1>{}, 0L, 0L);

    float alpha = 1.0f;
    float beta = 0.0f;

    // Create a dummy C tensor
    cutlass::device_memory::allocation<ElementC> dummy_C(M * N * L);

    // Prepare arguments
    typename Gemm::GemmKernel::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_shape,
        {static_cast<ElementA*>(mat_a.data_ptr()),
         stride_A,
         static_cast<ElementB*>(mat_b.data_ptr()),
         stride_B,
         static_cast<ElementScale*>(ptr_scale_A),
         stride_SA,
         static_cast<ElementScale*>(ptr_scale_B),
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
struct BmmFP8Config {
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

// Helper function to dispatch based on input FP8 type and output dtype
template <typename ElementInputFp8>
static at::Tensor bmm_fp8_impl(
    const at::Tensor& mat_a,
    const at::Tensor& mat_b,
    const at::Tensor& scales_a_half,
    const at::Tensor& scales_b_half,
    const at::ScalarType out_dtype,
    at::Tensor& out,
    const cutlass::KernelHardwareInfo& hw_info) {
  at::Tensor mat_a_contig = mat_a.is_contiguous() ? mat_a : mat_a.contiguous();
  at::Tensor mat_b_contig = mat_b.is_contiguous() ? mat_b : mat_b.contiguous();

  cutlass::Status status;

  if (out_dtype == at::ScalarType::BFloat16) {
    using Config = BmmFP8Config<ElementInputFp8, cutlass::bfloat16_t>;
    BmmFP8Runner<typename Config::Gemm, cutlass::bfloat16_t> runner;
    status = runner.run(mat_a_contig, mat_b_contig, scales_a_half, scales_b_half, out, hw_info);
  } else {  // Half - used for both FP16 output and FP8 intermediate
    using Config = BmmFP8Config<ElementInputFp8, cutlass::half_t>;
    BmmFP8Runner<typename Config::Gemm, cutlass::half_t> runner;
    status = runner.run(mat_a_contig, mat_b_contig, scales_a_half, scales_b_half, out, hw_info);
  }

  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "FP8 GEMM failed with status: " + std::string(cutlassGetStatusString(status)));

  return out;
}
void bmm_fp8(
    at::Tensor mat_a,
    at::Tensor mat_b,
    at::Tensor mat_d,
    at::Tensor scales_a,
    at::Tensor scales_b,
    at::Tensor workspace_buffer,
    int64_t cublas_handle,
    int64_t cuda_stream) {
  // Main entry point

  // TODO: Check workspace_buffer and empty it
  //  Input validation
  auto input_dtype = mat_a.scalar_type();
  auto out_dtype = mat_d.scalar_type();
  TORCH_CHECK(
      input_dtype == at::ScalarType::Float8_e4m3fn || input_dtype == at::ScalarType::Float8_e5m2,
      "mat_a must be Float8_e4m3fn or Float8_e5m2");
  TORCH_CHECK(mat_b.scalar_type() == input_dtype, "mat_a and mat_b must have the same dtype");
  TORCH_CHECK(scales_a.scalar_type() == at::ScalarType::Float, "scales_a must be Float32");
  TORCH_CHECK(scales_b.scalar_type() == at::ScalarType::Float, "scales_b must be Float32");
  TORCH_CHECK(
      out_dtype == at::ScalarType::BFloat16 || out_dtype == at::ScalarType::Half,
      "out_dtype must be BFloat16 or Float16");

  CHECK_DEVICE(mat_a);
  CHECK_DEVICE(mat_b);
  CHECK_DEVICE(mat_d);
  CHECK_DEVICE(scales_a);
  CHECK_DEVICE(scales_b);

  int M = mat_a.size(1);
  int K = mat_a.size(2);
  int L = mat_a.size(0);
  int K_b = mat_b.size(1);
  int N = mat_b.size(2);
  int L_b = mat_b.size(0);

  TORCH_CHECK(K == K_b, "Inner dimensions must match");
  TORCH_CHECK(L == L_b, "Batch dimension must match");

  // Convert scales to half precision for GEMM
  at::Tensor scales_a_half = scales_a.to(at::ScalarType::Half).contiguous();
  at::Tensor scales_b_half = scales_b.to(at::ScalarType::Half).contiguous();

  // For FP8 output, use FP16 intermediate or requested out dtype
  at::ScalarType intermediate_dtype;
  if (is_fp8_dtype(out_dtype)) {
    intermediate_dtype = at::ScalarType::Half;
  } else {
    intermediate_dtype = out_dtype;
  }

  c10::DeviceGuard device_guard(mat_a.device());

  // Get hardware info
  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  // Dispatch based on input FP8 type
  if (input_dtype == at::ScalarType::Float8_e4m3fn) {
    bmm_fp8_impl<cutlass::float_e4m3_t>(mat_a, mat_b, scales_a_half, scales_b_half, intermediate_dtype, mat_d, hw_info);
  } else {
    bmm_fp8_impl<cutlass::float_e5m2_t>(mat_a, mat_b, scales_a_half, scales_b_half, intermediate_dtype, mat_d, hw_info);
  }
}
