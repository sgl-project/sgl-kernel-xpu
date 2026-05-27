/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#define SYCL_INTEL_TARGET 20

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cute/tensor.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"

#include "cutlass/epilogue/collective/collective_epilogue.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

// HC Pre GEMM: C[T, N] = A[T, K] @ B[K, N]
// A: bfloat16, row-major
// B: float32, column-major
// C: float32, row-major
//
// Based on CUTLASS example: examples/00_bmg_gemm/00_bmg_gemm.cpp
// Configuration from mhc_guide.md: TileShape<64, 32, 128>

void hc_pre_gemm(
    const at::Tensor& A,  // [T, K] bfloat16
    const at::Tensor& B,  // [K, N] float32, column-major
    at::Tensor& C) {      // [T, N] float32

  CHECK_INPUT(A);
  CHECK_INPUT(B);
  CHECK_INPUT(C);

  TORCH_CHECK(A.scalar_type() == at::kBFloat16, "A must be bfloat16");
  TORCH_CHECK(B.scalar_type() == at::kFloat, "B must be float32");
  TORCH_CHECK(C.scalar_type() == at::kFloat, "C must be float32");

  const int64_t T = A.size(0);
  const int64_t K_A = A.size(1);
  const int64_t K_B = B.size(0);
  const int64_t N = B.size(1);

  TORCH_CHECK(K_A == K_B, "K dimension mismatch: A has ", K_A, ", B has ", K_B);
  TORCH_CHECK(C.size(0) == T && C.size(1) == N, "C dimension mismatch: expected [",
              T, ", ", N, "], got [", C.size(0), ", ", C.size(1), "]");

  // CUTLASS GEMM configuration
  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ElementInputA = cutlass::bfloat16_t;
  using ElementInputB = float;
  using ElementOutput = float;

  using LayoutA = cutlass::layout::RowMajor;     // A: row-major
  using LayoutB = cutlass::layout::ColumnMajor;  // B: column-major
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using GmemTiledCopyA = void;  // Auto-select
  using GmemTiledCopyB = void;  // Auto-select

  // Tile configuration from mhc_guide.md: M=64, N=32, K=128
  using TileShape = Shape<_64, _32, _128>;

  // TiledMMA: 8x2 subgroups (16 total), XE DPAS with mixed precision (bf16 × fp32 → fp32)
  // For TileShape<64, 32, 128>: DPAS atom is 8x16x8, so we need 8 SGs in M, 2 SGs in N
  // For mixed types, we use the wider type (float) for the MMA
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_DPAS_TT<8, float, cutlass::bfloat16_t>>,
      Layout<TileShape>,
      Layout<Shape<_8, _2, _1>, Stride<_2, _1, _0>>>::TiledMMA;

  constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopXeL1Staged<PipelineStages>;
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

  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
      EpilogueDispatchPolicy,
      TileShape,
      void,  // EpilogueTile (auto)
      ElementAccumulator,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      ElementOutput,
      cutlass::gemm::TagToStrideC_t<LayoutD>,
      FusionCallbacks,
      void,  // Copy atom for C (auto)
      void   // Copy atom for D (auto)
      >;

  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      GEMMDispatchPolicy,
      TileShape,
      ElementInputA,
      cutlass::gemm::TagToStrideA_t<LayoutA>,
      ElementInputB,
      cutlass::gemm::TagToStrideB_t<LayoutB>,
      TiledMma,
      GmemTiledCopyA, void, void, cute::identity,  // A copy ops
      GmemTiledCopyB, void, void, cute::identity   // B copy ops
      >;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,  // Problem shape deferred to runtime
      CollectiveMainloop,
      CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  // Prepare arguments
  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;

  int M = static_cast<int>(T);
  int N_val = static_cast<int>(N);
  int K = static_cast<int>(K_A);
  int L = 1;  // batch size

  // Strides
  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N_val, K, L));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N_val, L));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N_val, L));

  typename GemmKernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N_val, K, L},  // Problem size
      {reinterpret_cast<ElementInputA*>(const_cast<at::BFloat16*>(A.data_ptr<at::BFloat16>())),
       stride_A,
       reinterpret_cast<ElementInputB*>(const_cast<float*>(B.data_ptr<float>())),
       stride_B},
      {{1.0f, 0.0f},                                               // alpha=1, beta=0
       nullptr,                                                    // C ptr (nullptr since beta=0)
       stride_C,
       reinterpret_cast<ElementOutput*>(C.data_ptr<float>()),     // D ptr
       stride_D},
      cutlass::KernelHardwareInfo{
          0,  // device_id
          cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0),  // sm_count
          0   // smem_capacity (auto)
      }};

  Gemm gemm_op;

  // Get workspace size
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  at::Tensor workspace;
  if (workspace_size > 0) {
    workspace = at::empty({static_cast<int64_t>(workspace_size)},
                          at::TensorOptions().dtype(at::kByte).device(A.device()));
  }

  // Check if kernel can implement the problem
  cutlass::Status status = gemm_op.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS GEMM cannot implement the problem: M=", M, ", N=", N_val, ", K=", K);

  // Initialize
  status = gemm_op.initialize(arguments, workspace_size > 0 ? workspace.data_ptr() : nullptr);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "GEMM initialization failed");

  // Run
  status = gemm_op.run();
  TORCH_CHECK(status == cutlass::Status::kSuccess, "GEMM execution failed");

  // Synchronize
  c10::xpu::syncStreamsOnDevice(A.device().index());
}
