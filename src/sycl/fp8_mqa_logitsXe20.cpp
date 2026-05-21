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

// SYCL-TLA optimized FP8 GEMM for MQA logits on Intel BMG (Xe20).
// Computes: D(M,N) = A_fp8(M,K) @ B_fp8(K,N) in ConvertOnly mode.
// B must be pre-transposed to (K,N) row-major by the caller.

#define SYCL_INTEL_TARGET 20

#include <cute/tensor.hpp>
#include <sycl/sycl.hpp>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

namespace {

template <typename, typename>
class Fp8MqaGemmKernelName;

using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using MmaType = cutlass::half_t;
using ElementAccumulator = float;
using ElementOutput = float;

using LayoutA = cutlass::layout::RowMajor;
// B must be in (K,N) row-major layout (N contiguous).
// Caller must transpose B from (N,K) to (K,N) before calling.
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

using StrideA = cutlass::gemm::TagToStrideA_t<LayoutA>;
using StrideB = cutlass::gemm::TagToStrideB_t<LayoutB>;
using StrideC = cutlass::gemm::TagToStrideC_t<LayoutC>;
using StrideD = cutlass::gemm::TagToStrideC_t<LayoutD>;

using GmemTiledCopyA = XE_2D_U8x32x32_LD_N;
using GmemTiledCopyB = XE_2D_U8x32x32_LD_V;

constexpr int PipelineStages = 2;
using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16FP8Scaling<PipelineStages>;
using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16;

// Tile 32x128x32 — smaller tile for MQA logits workloads.
using TileShape = Shape<_32, _128, _32>;
using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;

using TiledMma = typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32F16F16F32_TT>, Layout<TileShape>, SGLayout>::TiledMMA;

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementOutput,
    float,
    ElementAccumulator,
    ElementAccumulator,
    cutlass::FloatRoundStyle::round_to_nearest>;

using FusionCallBacks = cutlass::epilogue::fusion::
    FusionCallbacks<EpilogueDispatchPolicy, EpilogueOp, TileShape, decltype(tile_shape(TiledMma()))>;

using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
    EpilogueDispatchPolicy,
    TileShape,
    ElementAccumulator,
    StrideC,
    ElementOutput,
    StrideD,
    FusionCallBacks,
    XE_2D_U32x8x16_LD_N,
    void,
    void,
    XE_2D_U32x8x16_ST_N,
    void,
    void>;

// ConvertOnly: FP8 data is loaded and converted to fp16 for XMX compute
using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
    GEMMDispatchPolicy,
    TileShape,
    cute::tuple<ElementA>,
    StrideA,
    cute::tuple<ElementB>,
    StrideB,
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

}  // namespace

// Returns 0 on success, non-zero if can_implement fails.
int fp8_mqa_gemm_xe20(sycl::queue* queue_ptr, const void* A_fp8, const void* B_fp8, void* D_f32, int M, int N, int K) {
  using ProblemShapeType = Shape<int, int, int, int>;
  ProblemShapeType problem_size{M, N, K, 1};

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  typename Gemm::GemmKernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      {static_cast<const ElementA*>(A_fp8), stride_A, static_cast<const ElementB*>(B_fp8), stride_B},
      {{1.0f, 0.0f}, nullptr, stride_D, static_cast<ElementOutput*>(D_f32), stride_D},
      hw_info};

  Gemm gemm_op;

  auto status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    return 1;
  }

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  uint8_t* workspace_ptr = nullptr;
  if (workspace_size > 0) {
    workspace_ptr = sycl::malloc_device<uint8_t>(workspace_size, *queue_ptr);
  }

  gemm_op.initialize(arguments, workspace_ptr);
  gemm_op.run(queue_ptr);

  if (workspace_ptr) {
    queue_ptr->wait();
    sycl::free(workspace_ptr, *queue_ptr);
  }
  return 0;
}

#undef SYCL_INTEL_TARGET
