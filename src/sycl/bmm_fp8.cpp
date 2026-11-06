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
#include "sgl_kernel_export.h"

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
      float* ptr_scale_A,
      float* ptr_scale_B,
      at::Tensor& out,
      uint8_t* gemm_workspace,
      size_t gemm_workspace_capacity,
      const cutlass::KernelHardwareInfo& hw_info,
      sycl::queue& xpu_q) {
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

    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, shape_A);
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, shape_B);
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, shape_CD);
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, shape_CD);

    StrideScale stride_SA{};
    StrideScale stride_SB{};

    float alpha = 1.0f;
    float beta = 0.0f;

    // beta = 0 -> the Xe epilogue's Sm90LinearCombination short-circuits the C load
    // (Sm90ScalarBroadcast::is_zero() folds Sm90SrcFetch::is_C_load_needed() to false).
    // Our patched Sm90SrcFetch::previsit() zero-fills the tCrC register fragment in that
    // case, so the EVT compute step sees defined zeros instead of uninitialized register
    // bits (which were poisoning the result with NaN bit patterns). Pass nullptr for C.

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
        {{alpha, beta}, nullptr, stride_C, static_cast<ElementOutput*>(out.data_ptr()), stride_D},
        hw_info};

    Gemm gemm_op;

    // GemmKernel::get_workspace_size returns 0 for this kernel; skip the check.
    // can_implement is expensive on the hot path; rely on validation in bmm_fp8().

    cutlass::Status status = gemm_op.initialize(arguments, gemm_workspace, &xpu_q);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }

    return gemm_op.run(&xpu_q);
  }
};

// Tile traits: large tile for big problems, small tile for low-occupancy shapes.
struct BmmFP8TileLarge {
  using TileShape = Shape<_256, _256, _32>;
  using WarpLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;
};

struct BmmFP8TileSmall {
  using TileShape = Shape<_128, _128, _32>;
  using WarpLayout = Layout<Shape<_4, _4, _1>, Stride<_4, _1, _0>>;
};

// Tiny tile for small-M shapes (e.g. M<=32 with N=128): keeps full N coverage
// per WG to avoid splitting the N dim, but shrinks M to 32 so each WG does only
// useful work (no M-dim padding) and uses just 4 subgroups (vs 16 in Small),
// cutting per-WG launch + sync cost.
struct BmmFP8TileTiny {
  using TileShape = Shape<_32, _128, _32>;
  using WarpLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
};

// Configure GEMM based on output dtype, per-operand FP8 input types, and tile traits
template <typename ElementInputFp8A, typename ElementInputFp8B, typename ElementOutput, typename TileTraits>
struct BmmFP8Config {
  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ElementInputA = ElementInputFp8A;
  using ElementInputB = ElementInputFp8B;
  // Scales are consumed directly in FP32 (their native PyTorch dtype). The collective
  // converts FP8 -> FP16 internally, so passing FP32 scales costs an extra fp32->fp16
  // conversion per scale-tile but avoids a separate scale-conversion kernel launch on
  // the host -> a large win for small problem sizes.
  using ElementScale = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using StrideScale = cute::Stride<_0, _0, _0>;

  using GmemTiledCopyA = XE_2D_U8x32x32_LD_N;
  using GmemTiledCopyB = XE_2D_U8x32x32_LD_V;

  using TileShape = typename TileTraits::TileShape;

  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_8x16x16_F32F16F16F32_TT>,
      Layout<TileShape>,
      typename TileTraits::WarpLayout>::TiledMMA;

  static constexpr int PipelineStages = 1;
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

// Helper function to dispatch based on input FP8 types and output dtype
template <typename ElementInputFp8A, typename ElementInputFp8B, typename TileTraits>
static at::Tensor bmm_fp8_impl(
    const at::Tensor& mat_a,
    const at::Tensor& mat_b,
    float* ptr_scale_A,
    float* ptr_scale_B,
    const at::ScalarType out_dtype,
    at::Tensor& out,
    uint8_t* gemm_workspace,
    size_t gemm_workspace_capacity,
    const cutlass::KernelHardwareInfo& hw_info,
    sycl::queue& xpu_q) {
  cutlass::Status status;

  if (out_dtype == at::ScalarType::BFloat16) {
    using Config = BmmFP8Config<ElementInputFp8A, ElementInputFp8B, cutlass::bfloat16_t, TileTraits>;
    BmmFP8Runner<typename Config::Gemm, cutlass::bfloat16_t> runner;
    status = runner.run(
        mat_a, mat_b, ptr_scale_A, ptr_scale_B, out, gemm_workspace, gemm_workspace_capacity, hw_info, xpu_q);
  } else {  // Half - used for both FP16 output and FP8 intermediate
    using Config = BmmFP8Config<ElementInputFp8A, ElementInputFp8B, cutlass::half_t, TileTraits>;
    BmmFP8Runner<typename Config::Gemm, cutlass::half_t> runner;
    status = runner.run(
        mat_a, mat_b, ptr_scale_A, ptr_scale_B, out, gemm_workspace, gemm_workspace_capacity, hw_info, xpu_q);
  }

  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "FP8 GEMM failed with status: " + std::string(cutlassGetStatusString(status)));

  return out;
}

// Dispatch on FP8 input types only (tile traits already chosen).
template <typename TileTraits>
static void bmm_fp8_dispatch_fp8(
    bool a_e4m3,
    bool b_e4m3,
    const at::Tensor& mat_a,
    const at::Tensor& mat_b,
    float* ptr_scale_A,
    float* ptr_scale_B,
    const at::ScalarType out_dtype,
    at::Tensor& out,
    uint8_t* gemm_workspace,
    size_t gemm_workspace_capacity,
    const cutlass::KernelHardwareInfo& hw_info,
    sycl::queue& xpu_q) {
  using e4m3 = cutlass::float_e4m3_t;
  using e5m2 = cutlass::float_e5m2_t;
  if (a_e4m3 && b_e4m3) {
    bmm_fp8_impl<e4m3, e4m3, TileTraits>(
        mat_a,
        mat_b,
        ptr_scale_A,
        ptr_scale_B,
        out_dtype,
        out,
        gemm_workspace,
        gemm_workspace_capacity,
        hw_info,
        xpu_q);
  } else if (a_e4m3 && !b_e4m3) {
    bmm_fp8_impl<e4m3, e5m2, TileTraits>(
        mat_a,
        mat_b,
        ptr_scale_A,
        ptr_scale_B,
        out_dtype,
        out,
        gemm_workspace,
        gemm_workspace_capacity,
        hw_info,
        xpu_q);
  } else if (!a_e4m3 && b_e4m3) {
    bmm_fp8_impl<e5m2, e4m3, TileTraits>(
        mat_a,
        mat_b,
        ptr_scale_A,
        ptr_scale_B,
        out_dtype,
        out,
        gemm_workspace,
        gemm_workspace_capacity,
        hw_info,
        xpu_q);
  } else {
    bmm_fp8_impl<e5m2, e5m2, TileTraits>(
        mat_a,
        mat_b,
        ptr_scale_A,
        ptr_scale_B,
        out_dtype,
        out,
        gemm_workspace,
        gemm_workspace_capacity,
        hw_info,
        xpu_q);
  }
}

SGL_KERNEL_EXPORT void bmm_fp8(
    at::Tensor mat_a,
    at::Tensor mat_b,
    at::Tensor mat_d,
    at::Tensor scales_a,
    at::Tensor scales_b,
    at::Tensor workspace_buffer,
    int64_t cublas_handle,
    int64_t cuda_stream) {
  // Main entry point

  //  Input validation
  auto input_dtype = mat_a.scalar_type();
  auto b_dtype = mat_b.scalar_type();
  auto out_dtype = mat_d.scalar_type();
  TORCH_CHECK(
      input_dtype == at::ScalarType::Float8_e4m3fn || input_dtype == at::ScalarType::Float8_e5m2,
      "mat_a must be Float8_e4m3fn or Float8_e5m2");
  TORCH_CHECK(
      b_dtype == at::ScalarType::Float8_e4m3fn || b_dtype == at::ScalarType::Float8_e5m2,
      "mat_b must be Float8_e4m3fn or Float8_e5m2");
  TORCH_CHECK(scales_a.scalar_type() == at::ScalarType::Float, "scales_a must be Float32");
  TORCH_CHECK(scales_b.scalar_type() == at::ScalarType::Float, "scales_b must be Float32");
  TORCH_CHECK(
      out_dtype == at::ScalarType::BFloat16 || out_dtype == at::ScalarType::Half,
      "out_dtype must be BFloat16 or Float16");
  TORCH_CHECK(
      workspace_buffer.defined() && workspace_buffer.scalar_type() == at::ScalarType::Byte &&
          workspace_buffer.is_contiguous(),
      "workspace_buffer must be a contiguous uint8 tensor");

  CHECK_DEVICE(mat_a);
  CHECK_DEVICE(mat_b);
  CHECK_DEVICE(mat_d);
  CHECK_DEVICE(scales_a);
  CHECK_DEVICE(scales_b);
  CHECK_DEVICE(workspace_buffer);

  int M = mat_a.size(1);
  int K = mat_a.size(2);
  int L = mat_a.size(0);
  int K_b = mat_b.size(1);
  int N = mat_b.size(2);
  int L_b = mat_b.size(0);

  TORCH_CHECK(K == K_b, "Inner dimensions must match");
  TORCH_CHECK(L == L_b, "Batch dimension must match");

  // Make A/B contiguous once here (was duplicated inside bmm_fp8_impl).
  at::Tensor mat_a_contig = mat_a.is_contiguous() ? mat_a : mat_a.contiguous();
  at::Tensor mat_b_contig = mat_b.is_contiguous() ? mat_b : mat_b.contiguous();

  // For FP8 output, use FP16 intermediate or requested out dtype
  at::ScalarType intermediate_dtype = is_fp8_dtype(out_dtype) ? at::ScalarType::Half : out_dtype;

  c10::DeviceGuard device_guard(mat_a.device());

  // Cache the SM count once per process - querying it on every call costs several us.
  static const int cached_sm_count = []() {
    cutlass::KernelHardwareInfo tmp;
    return cutlass::KernelHardwareInfo::query_device_multiprocessor_count(tmp.device_id);
  }();
  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cached_sm_count;

  // Same XPU stream as PyTorch -> proper ordering with surrounding ops.
  sycl::queue& xpu_q = c10::xpu::getCurrentXPUStream().queue();

  // Pass the FP32 scale tensors straight into the GEMM. The collective accepts FP32
  // scales (specialized scale_zero_copy_traits<float,...>) and casts them to the MMA
  // dtype internally. This removes the separate fp32->fp16 scale-conversion submit
  // that was on the hot path for small shapes.
  float* ptr_scale_A = scales_a.data_ptr<float>();
  float* ptr_scale_B = scales_b.data_ptr<float>();

  uint8_t* gemm_workspace = workspace_buffer.data_ptr<uint8_t>();
  size_t gemm_workspace_capacity = static_cast<size_t>(workspace_buffer.numel());

  const bool a_e4m3 = (input_dtype == at::ScalarType::Float8_e4m3fn);
  const bool b_e4m3 = (b_dtype == at::ScalarType::Float8_e4m3fn);

  // ---- Tile-shape heuristic ----
  // Three tiers:
  //   Tiny  (32x128x32, 4 SGs):  small/medium-M shapes where the 128x128 Small
  //                              tile launches few WGs that each finish too fast,
  //                              leaving per-WG launch+sync cost dominant. The
  //                              4-SG Tiny WG produces 4x more, smaller WGs that
  //                              pipeline better on Xe.
  //   Small (128x128x32, 16 SGs): large-M shapes that benefit from amortizing
  //                              per-WG overhead over a 128-row strip.
  //   Large (256x256x32, 32 SGs): big problems that benefit from wide tiles.
  // Tiny vs Small crossover is empirically governed by total Tiny-WG count.
  // Tiny WGs along M = L * ceil(M/32). When that exceeds ~64 the device is
  // already saturated and Small's heavier WGs (16 SGs vs 4) amortize per-WG
  // cost better. Use the proxy L*M <= 2048 (= 64 * 32) as the threshold.
  const bool use_tiny_tile = (M <= 256) && (N <= 256) && (L * M <= 2048);
  const bool use_small_tile = !use_tiny_tile && ((M <= 128) || (N <= 128) || (M * N * L <= 256 * 256 * 4));

  if (use_tiny_tile) {
    bmm_fp8_dispatch_fp8<BmmFP8TileTiny>(
        a_e4m3,
        b_e4m3,
        mat_a_contig,
        mat_b_contig,
        ptr_scale_A,
        ptr_scale_B,
        intermediate_dtype,
        mat_d,
        gemm_workspace,
        gemm_workspace_capacity,
        hw_info,
        xpu_q);
  } else if (use_small_tile) {
    bmm_fp8_dispatch_fp8<BmmFP8TileSmall>(
        a_e4m3,
        b_e4m3,
        mat_a_contig,
        mat_b_contig,
        ptr_scale_A,
        ptr_scale_B,
        intermediate_dtype,
        mat_d,
        gemm_workspace,
        gemm_workspace_capacity,
        hw_info,
        xpu_q);
  } else {
    bmm_fp8_dispatch_fp8<BmmFP8TileLarge>(
        a_e4m3,
        b_e4m3,
        mat_a_contig,
        mat_b_contig,
        ptr_scale_A,
        ptr_scale_B,
        intermediate_dtype,
        mat_d,
        gemm_workspace,
        gemm_workspace_capacity,
        hw_info,
        xpu_q);
  }
}
