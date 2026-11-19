/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Batched GEMM with 16-bit float inputs (no scaling). A and B must share the
 * same dtype (bf16 x bf16 or fp16 x fp16).
 *   A: (L, M, K) bf16/fp16, row-major
 *   B: (L, K, N) bf16/fp16, row-major
 *   D: (L, M, N) bf16 or fp16, row-major
 *
 * Built on the standard cutlass-sycl Xe XMX16 mainloop (no FP8 scaling path),
 * but otherwise mirrors src/sycl/bmm_fp8.cpp (same queue handling, beta=0,
 * RowMajor everywhere).
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

namespace {

template <typename Gemm, typename ElementOutput>
struct BmmBF16Runner {
  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  cutlass::Status
  run(const at::Tensor& mat_a, const at::Tensor& mat_b, at::Tensor& out, const cutlass::KernelHardwareInfo& hw_info) {
    int L = mat_a.size(0);
    int M = mat_a.size(1);
    int K = mat_a.size(2);
    int N = mat_b.size(2);

    auto problem_shape = cute::make_shape(M, N, K, L);

    auto shape_A = cute::make_shape(M, K, L);
    auto shape_B = cute::make_shape(N, K, L);
    auto shape_CD = cute::make_shape(M, N, L);

    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, shape_A);
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, shape_B);
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, shape_CD);
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, shape_CD);

    float alpha = 1.0f;
    float beta = 0.0f;

    // beta = 0 -> C load is short-circuited (see notes in bmm_fp8.cpp).
    // Pass nullptr for C.
    typename Gemm::GemmKernel::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_shape,
        {static_cast<ElementA*>(mat_a.data_ptr()), stride_A, static_cast<ElementB*>(mat_b.data_ptr()), stride_B},
        {{alpha, beta}, nullptr, stride_C, static_cast<ElementOutput*>(out.data_ptr()), stride_D},
        hw_info};

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }

    // Run on PyTorch's current XPU stream queue (see xpu-cutlass note in bmm_fp8.cpp).
    sycl::queue& xpu_q = c10::xpu::getCurrentXPUStream().queue();

    status = gemm_op.initialize(arguments, workspace.get(), &xpu_q);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }
    return gemm_op.run(&xpu_q);
  }
};

// Pick the right XMX16 MMA atom for the input element type.
// Both bf16 and fp16 inputs are 16-bit and share the U16 2D-block load atoms.
template <typename ElementInput>
struct XmxMmaAtom;

template <>
struct XmxMmaAtom<cutlass::bfloat16_t> {
  using type = XE_8x16x16_F32BF16BF16F32_TT;
};

template <>
struct XmxMmaAtom<cutlass::half_t> {
  using type = XE_8x16x16_F32F16F16F32_TT;
};

template <typename ElementInput, typename ElementOutput>
struct BmmBF16Config {
  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ElementInputA = ElementInput;
  using ElementInputB = ElementInput;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using GmemTiledCopyA = XE_2D_U16x32x32_LD_N;
  using GmemTiledCopyB = XE_2D_U16x32x32_LD_V;

  using TileShape = Shape<_256, _256, _32>;

  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<typename XmxMmaAtom<ElementInput>::type>,
      Layout<TileShape>,
      Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

  static constexpr int PipelineStages = 3;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementOutput,
      ElementComputeEpilogue,
      ElementAccumulator,
      ElementAccumulator,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using FusionCallBacks = cutlass::epilogue::fusion::
      FusionCallbacks<EpilogueDispatchPolicy, EpilogueOp, TileShape, decltype(tile_shape(TiledMma()))>;

  // 16-bit (bf16/fp16) output store.
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
      ElementInputA,
      cutlass::gemm::TagToStrideA_t<LayoutA>,
      ElementInputB,
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

}  // namespace

template <typename ElementInput>
static cutlass::Status bmm_bf16_dispatch_out(
    const at::Tensor& mat_a, const at::Tensor& mat_b, at::Tensor& mat_d, const cutlass::KernelHardwareInfo& hw_info) {
  if (mat_d.scalar_type() == at::ScalarType::BFloat16) {
    using Config = BmmBF16Config<ElementInput, cutlass::bfloat16_t>;
    BmmBF16Runner<typename Config::Gemm, cutlass::bfloat16_t> runner;
    return runner.run(mat_a, mat_b, mat_d, hw_info);
  } else {
    using Config = BmmBF16Config<ElementInput, cutlass::half_t>;
    BmmBF16Runner<typename Config::Gemm, cutlass::half_t> runner;
    return runner.run(mat_a, mat_b, mat_d, hw_info);
  }
}

SGL_KERNEL_EXPORT void bmm_bf16(at::Tensor mat_a, at::Tensor mat_b, at::Tensor mat_d) {
  auto in_dtype = mat_a.scalar_type();
  TORCH_CHECK(
      in_dtype == at::ScalarType::BFloat16 || in_dtype == at::ScalarType::Half, "mat_a must be BFloat16 or Float16");
  TORCH_CHECK(mat_b.scalar_type() == in_dtype, "mat_a and mat_b must have the same dtype");
  auto out_dtype = mat_d.scalar_type();
  TORCH_CHECK(
      out_dtype == at::ScalarType::BFloat16 || out_dtype == at::ScalarType::Half,
      "out dtype must be BFloat16 or Float16");

  CHECK_DEVICE(mat_a);
  CHECK_DEVICE(mat_b);
  CHECK_DEVICE(mat_d);

  TORCH_CHECK(mat_a.dim() == 3 && mat_b.dim() == 3 && mat_d.dim() == 3, "all tensors must be 3D");

  int L = mat_a.size(0);
  int M = mat_a.size(1);
  int K = mat_a.size(2);
  int L_b = mat_b.size(0);
  int K_b = mat_b.size(1);
  int N = mat_b.size(2);
  TORCH_CHECK(K == K_b, "Inner dimensions must match");
  TORCH_CHECK(L == L_b, "Batch dimension must match");
  TORCH_CHECK(mat_d.size(0) == L && mat_d.size(1) == M && mat_d.size(2) == N, "output shape mismatch");

  at::Tensor mat_a_contig = mat_a.is_contiguous() ? mat_a : mat_a.contiguous();
  at::Tensor mat_b_contig = mat_b.is_contiguous() ? mat_b : mat_b.contiguous();

  c10::DeviceGuard device_guard(mat_a.device());

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  cutlass::Status status;
  if (in_dtype == at::ScalarType::BFloat16) {
    status = bmm_bf16_dispatch_out<cutlass::bfloat16_t>(mat_a_contig, mat_b_contig, mat_d, hw_info);
  } else {
    status = bmm_bf16_dispatch_out<cutlass::half_t>(mat_a_contig, mat_b_contig, mat_d, hw_info);
  }

  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "BMM (bf16/fp16) failed with status: " + std::string(cutlassGetStatusString(status)));
}
