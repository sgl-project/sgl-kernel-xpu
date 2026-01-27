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

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cute/tensor.hpp>
#include <type_traits>

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

// Fixed router shapes (specialization)
constexpr int DSV3_ROUTER_K = 7168;
constexpr int DSV3_ROUTER_N_256 = 256;
constexpr int DSV3_ROUTER_N_384 = 384;
constexpr int DSV3_ROUTER_M_MIN = 1;
constexpr int DSV3_ROUTER_M_MAX = 16;

// Generic GEMM runner (B is [N, K] column-major)
template <typename Gemm>
struct Dsv3GemmRunner {
  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementOutput = typename Gemm::CollectiveEpilogue::ElementOutput;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  cutlass::Status
  run(const at::Tensor& mat_a,
      const at::Tensor& mat_b,
      at::Tensor& out,
      const cutlass::KernelHardwareInfo& hw_info,
      sycl::queue* queue) {
    int M = mat_a.size(0);
    int N = mat_b.size(0);
    int K = mat_a.size(1);
    int L = 1;

    auto problem_shape = cute::make_shape(M, N, K, L);

    auto shape_A = cute::make_shape(M, K, L);
    auto shape_B = cute::make_shape(N, K, L);  // B is [N, K] column-major
    auto shape_CD = cute::make_shape(M, N, L);

    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, shape_A);
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, shape_B);
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, shape_CD);
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, shape_CD);

    float alpha = 1.0f;
    float beta = 0.0f;

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
      TORCH_CHECK(
          false,
          "dsv3_router_gemm can_implement failed: status=",
          cutlassGetStatusString(status),
          " M=",
          M,
          " N=",
          N,
          " K=",
          K,
          " L=",
          L);
    }

    status = gemm_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) return status;

    status = gemm_op.run(queue);
    return status;
  }
};

// Epilogue selector: FP32 uses IntelXeGeneric + XE_STORE_2D; BF16/FP16 use IntelXeXMX16 + 16-bit store.
template <typename OutputT>
struct EpilogueTraits {
  using DispatchPolicy = cutlass::epilogue::IntelXeGeneric;
  using GmemTiledCopyC = XE_LOAD_2D<32, 8, 16>;
  using GmemTiledCopyD = XE_STORE_2D<32, 8, 16>;
};

template <>
struct EpilogueTraits<cutlass::bfloat16_t> {
  using DispatchPolicy = cutlass::epilogue::IntelXeXMX16;
  using GmemTiledCopyC = XE_2D_U32x8x16_LD_N;
  using GmemTiledCopyD = XE_2D_U16x8x16_ST_N;
};

template <>
struct EpilogueTraits<cutlass::half_t> {
  using DispatchPolicy = cutlass::epilogue::IntelXeXMX16;
  using GmemTiledCopyC = XE_2D_U32x8x16_LD_N;
  using GmemTiledCopyD = XE_2D_U16x8x16_ST_N;
};

// Router specializations: N = 256 or 384, K = 7168, M in [1,16]
// Output type is templated to allow FP32, BF16 or FP16 outputs.
template <int NExperts, typename OutputT>
struct Dsv3RouterSpecializedConfig {
  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ElementInputA = cutlass::bfloat16_t;
  using ElementInputB = cutlass::bfloat16_t;
  using ElementOutput = OutputT;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;  // B is [N, K], column-major
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using TileShape = cute::conditional_t<
      (NExperts == 256),
      cute::Shape<cute::_128, cute::_256, cute::_32>,
      cute::Shape<cute::_128, cute::_384, cute::_32>>;

  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;
  using GmemTiledCopyC = typename EpilogueTraits<OutputT>::GmemTiledCopyC;
  using GmemTiledCopyD = typename EpilogueTraits<OutputT>::GmemTiledCopyD;

  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>,
      Layout<TileShape>,
      Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

  static constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopXeL1Staged<PipelineStages>;
  using EpilogueDispatchPolicy = typename EpilogueTraits<OutputT>::DispatchPolicy;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementOutput,
      ElementComputeEpilogue,
      ElementAccumulator,
      ElementAccumulator,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using FusionCallBacks = cutlass::epilogue::fusion::
      FusionCallbacks<EpilogueDispatchPolicy, EpilogueOp, TileShape, decltype(tile_shape(TiledMma()))>;

  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
      EpilogueDispatchPolicy,
      TileShape,
      ElementAccumulator,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      ElementOutput,
      cutlass::gemm::TagToStrideC_t<LayoutD>,
      FusionCallBacks,
      GmemTiledCopyC,
      void,
      void,
      GmemTiledCopyD,
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

template <typename Gemm>
static cutlass::Status run_gemm(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& O,
    const cutlass::KernelHardwareInfo& hw_info,
    sycl::queue* q) {
  Dsv3GemmRunner<Gemm> runner;
  return runner.run(A, B, O, hw_info, q);
}

// dsv3_router_gemm with hardcoded specialization (K=7168, N in {256,384}, 1<=M<=16)
void dsv3_router_gemm_xpu(at::Tensor& output, const at::Tensor& mat_a, const at::Tensor& mat_b) {
  const int M = mat_a.size(0);  // num_tokens
  const int K = mat_a.size(1);  // hidden_dim
  const int N = mat_b.size(0);  // num_experts

  TORCH_CHECK(
      mat_a.dim() == 2 && mat_b.dim() == 2 && output.dim() == 2,
      "mat_a, mat_b, output must be 2D. Got dims: mat_a=",
      mat_a.dim(),
      " mat_b=",
      mat_b.dim(),
      " output=",
      output.dim());
  TORCH_CHECK(
      mat_a.size(1) == mat_b.size(1), "Inner dimension K must match. Got K_a=", mat_a.size(1), " K_b=", mat_b.size(1));

  // Device checks
  TORCH_CHECK(mat_a.is_xpu(), "mat_a must be on XPU. Got device=", mat_a.device());
  TORCH_CHECK(mat_b.is_xpu(), "mat_b must be on XPU. Got device=", mat_b.device());
  TORCH_CHECK(output.is_xpu(), "output must be on XPU. Got device=", output.device());
  TORCH_CHECK(
      mat_b.device() == mat_a.device(),
      "mat_b must be on the same device as mat_a. mat_b=",
      mat_b.device(),
      " mat_a=",
      mat_a.device());
  TORCH_CHECK(
      output.device() == mat_a.device(),
      "output must be on the same device as mat_a. output=",
      output.device(),
      " mat_a=",
      mat_a.device());

  // Hard contract (match CUDA)
  TORCH_CHECK(M >= DSV3_ROUTER_M_MIN && M <= DSV3_ROUTER_M_MAX, "num_tokens must be in [1,16] (got ", M, ")");
  TORCH_CHECK(K == DSV3_ROUTER_K, "hidden_dim must be 7168 (got ", K, ")");
  TORCH_CHECK(N == DSV3_ROUTER_N_256 || N == DSV3_ROUTER_N_384, "num_experts must be 256 or 384 (got ", N, ")");

  // Dtypes
  TORCH_CHECK(mat_a.scalar_type() == at::ScalarType::BFloat16, "mat_a must be BFloat16. Got ", mat_a.scalar_type());
  TORCH_CHECK(mat_b.scalar_type() == at::ScalarType::BFloat16, "mat_b must be BFloat16. Got ", mat_b.scalar_type());
  TORCH_CHECK(
      output.scalar_type() == at::ScalarType::Float || output.scalar_type() == at::ScalarType::BFloat16 ||
          output.scalar_type() == at::ScalarType::Half,
      "output must be Float, BFloat16, or Float16. Got ",
      output.scalar_type());

  // Output shape
  TORCH_CHECK(output.size(0) == M && output.size(1) == N, "output shape mismatch");

  at::Tensor A = mat_a.contiguous();
  at::Tensor B = mat_b.contiguous();

  // Write directly in the requested dtype; only make a temporary if non-contiguous.
  const bool needs_contig = !output.is_contiguous();
  at::Tensor out_contig = needs_contig ? output.contiguous() : output;

  c10::DeviceGuard guard(mat_a.device());
  auto stream = at::xpu::getCurrentXPUStream(mat_a.device().index());
  sycl::queue& queue = stream.queue();

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = mat_a.device().index();
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  cutlass::Status status;

  if (output.scalar_type() == at::ScalarType::Float) {
    if (N == DSV3_ROUTER_N_256) {
      using Spec = Dsv3RouterSpecializedConfig<DSV3_ROUTER_N_256, float>;
      status = run_gemm<typename Spec::Gemm>(A, B, out_contig, hw_info, &queue);
    } else {
      using Spec = Dsv3RouterSpecializedConfig<DSV3_ROUTER_N_384, float>;
      status = run_gemm<typename Spec::Gemm>(A, B, out_contig, hw_info, &queue);
    }
  } else if (output.scalar_type() == at::ScalarType::BFloat16) {
    if (N == DSV3_ROUTER_N_256) {
      using Spec = Dsv3RouterSpecializedConfig<DSV3_ROUTER_N_256, cutlass::bfloat16_t>;
      status = run_gemm<typename Spec::Gemm>(A, B, out_contig, hw_info, &queue);
    } else {
      using Spec = Dsv3RouterSpecializedConfig<DSV3_ROUTER_N_384, cutlass::bfloat16_t>;
      status = run_gemm<typename Spec::Gemm>(A, B, out_contig, hw_info, &queue);
    }
  } else {  // at::ScalarType::Half
    if (N == DSV3_ROUTER_N_256) {
      using Spec = Dsv3RouterSpecializedConfig<DSV3_ROUTER_N_256, cutlass::half_t>;
      status = run_gemm<typename Spec::Gemm>(A, B, out_contig, hw_info, &queue);
    } else {
      using Spec = Dsv3RouterSpecializedConfig<DSV3_ROUTER_N_384, cutlass::half_t>;
      status = run_gemm<typename Spec::Gemm>(A, B, out_contig, hw_info, &queue);
    }
  }

  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "dsv3_router_gemm failed with status: " + std::string(cutlassGetStatusString(status)));

  // If we had to make a contiguous temporary, copy it back.
  if (needs_contig) {
    output.copy_(out_contig);
  }
}
