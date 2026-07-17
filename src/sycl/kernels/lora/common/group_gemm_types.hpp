/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
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
/*! \file
    \brief Shared CUTLASS type bundle for the LoRA pointer-array grouped GEMM.

    GroupGemmTypes<> collects every CUTLASS type the grouped-GEMM launcher needs
    (element types, layouts, XMX DPAS MMA atom, tiled MMA, dispatch policies,
    gmem copy atoms, epilogue op, collective mainloop/epilogue, GemmKernel and
    the device adapter) into one traits struct parameterized by the handful of
    perf knobs that actually vary per kernel/tile:

      TensorDType     -- torch element type (at::Half / at::BFloat16)
      TileShape_      -- CUTE tile shape          (e.g. 256 x 256 x 32)
      ThreadLayout_   -- subgroup thread layout   (e.g. 8 x 4 x 1)
      LayoutB_        -- B layout                  (ColumnMajor for LoRA weights)
      PipelineStages_ -- mainloop pipeline depth

    The concrete instantiation is composed in the per-kernel *_types.hpp (via a
    tile option tag) and handed to the GroupGemmLoraFwd<Types> launcher in
    group_gemm_lora_launcher.hpp, which only builds runtime arguments and drives
    the lifecycle. Common across the LoRA forward kernels (A-fwd, B-fwd, ...).
*/

#pragma once

#include <ATen/ATen.h>
#include <torch/all.h>

#include <cute/tensor.hpp>
#include <utility>

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/xe_array_epilogue.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/half.h"
#include "cutlass/layout/matrix.h"
#include "sycl/kernels/lora/device/group_gemm_lora_launcher.hpp"

namespace at::native::xpu {

//----------------- torch dtype -> CUTLASS element type ----------------------//
template <typename T>
struct ToCutlassElementType {
  using type = T;  // float -> float (identity)
};

template <>
struct ToCutlassElementType<at::Half> {
  using type = cutlass::half_t;
};

template <>
struct ToCutlassElementType<at::BFloat16> {
  using type = cutlass::bfloat16_t;
};

//----------------- Group problem shape (shared across all instantiations) ----//
using GroupedProblemShape = cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;
using UnderlyingProblemShapeType = typename GroupedProblemShape::UnderlyingProblemShape;

//----------------- Common CUTLASS type bundle -------------------------------//
// Every type the grouped-GEMM launcher would otherwise hardcode, parameterized
// by the per-kernel/tile perf knobs. LoRA weights match the input dtype, XMX
// accumulates in fp32, and the gmem copy atoms are left `void` so CUTLASS
// auto-selects the correct 2D load/store/prefetch ops per dtype and layout.
template <typename TensorDType, typename TileShape_, typename ThreadLayout_, typename LayoutB_, int PipelineStages_>
struct GroupGemmTypes {
  //--------- Element types ---------//
  using ElementA = typename ToCutlassElementType<TensorDType>::type;
  using ElementB = ElementA;             // same storage dtype as A (LoRA weights match input dtype)
  using ElementOutput = ElementA;        // output matches input dtype (bf16/fp16)
  using ElementAccumulator = float;      // XMX accumulates in fp32 -- required by every atom we support
  using ElementComputeEpilogue = float;  // alpha/beta arithmetic in fp32

  using ElementMma = ElementA;
  using ElementMmaB = ElementMma;

  //--------- Layouts ---------//
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = LayoutB_;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  //--------- Tile / thread layout (host-supplied perf knobs) ---------//
  using TileShape = TileShape_;
  using ThreadLayout = ThreadLayout_;
  static constexpr int PipelineStages = PipelineStages_;

  //--------- TiledMma (which XMX MMA atom + how many subgroups tile it) ---//
  using MmaAtom = cute::XE_DPAS_TT<8, ElementAccumulator, ElementMma>;
  using TiledMma =
      typename cute::TiledMMAHelper<cute::MMA_Atom<MmaAtom>, cute::Layout<TileShape>, ThreadLayout>::TiledMMA;

  //--------- Dispatch policies ---------//
  // MainloopXeL1StagedGroup builds
  // prefetch through make_block_2d_prefetch()/XE_PREFETCH_2D
  using GEMMDispatchPolicy = cutlass::gemm::MainloopXeL1StagedGroup<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGenericGroup;

  //--------- Gmem copy atoms ---------//
  // void => CUTLASS auto-selects the correct 2D load/store/prefetch ops per
  // dtype and layout (see get_block_2d_copy_A/B and make_block_2d_prefetch).
  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;
  using GmemTiledCopyC = void;
  using GmemTiledCopyD = void;

  //--------- Epilogue: D = alpha * acc + beta * C ---------//
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementAccumulator,  // <- must be ElementAccumulator for Grouped GEMM
      ElementComputeEpilogue,
      ElementAccumulator,
      ElementAccumulator,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using FusionCallbacks = cutlass::epilogue::fusion::
      FusionCallbacks<EpilogueDispatchPolicy, EpilogueOp, TileShape, decltype(cute::tile_shape(TiledMma()))>;

  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
      EpilogueDispatchPolicy,
      TileShape,
      void,  // EpilogueTile (void = automatic)
      ElementAccumulator,
      cutlass::gemm::TagToStrideC_t<LayoutC*>,
      ElementOutput,  // bf16/fp16 narrow here
      cutlass::gemm::TagToStrideC_t<LayoutD*>,
      FusionCallbacks,
      GmemTiledCopyC,   // load C (void = automatic)
      GmemTiledCopyD>;  // store D (void = automatic)

  //--------- Mainloop ---------//
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      GEMMDispatchPolicy,
      TileShape,
      ElementMma,
      cutlass::gemm::TagToStrideA_t<LayoutA*>,
      ElementMmaB,
      cutlass::gemm::TagToStrideB_t<LayoutB*>,
      TiledMma,
      GmemTiledCopyA,
      void,
      void,
      cute::identity,
      GmemTiledCopyB,
      void,
      void,
      cute::identity>;

  //--------- GemmKernel + launcher ---------//
  using GemmKernel = cutlass::gemm::kernel::
      GemmUniversal<GroupedProblemShape, CollectiveMainloop, CollectiveEpilogue, cutlass::gemm::GroupScheduler>;

  // Kernel-agnostic CUTLASS-lifecycle launcher (analog of MlaXe::Fmla). The
  // same class is reused across the LoRA forward grouped GEMMs (A-fwd, B-fwd).
  using Gemm = GroupGemmLoraFwd<GemmKernel>;

  // Stride aliases used by args_from_options() and callers.
  using StrideA = typename GemmKernel::InternalStrideA;
  using StrideB = typename GemmKernel::InternalStrideB;
  using StrideC = typename GemmKernel::InternalStrideC;
  using StrideD = typename GemmKernel::InternalStrideD;
};

//----------------- args_from_options (build Arguments) ----------------------//
// Wire the runtime pointer arrays / stride arrays / problem-sizes tensor into a
// GemmKernel::Arguments with epilogue thread-args (alpha, beta) set. This is the
// single place argument construction lives, kept separate from the launcher so
// it is reused across the LoRA forward grouped GEMMs (A-fwd passes beta=0 with
// c_ptrs aliased to d_ptrs; B-fwd passes beta=1 with a real residual C source).
// All device pointers come from make_device_ptrs() and the meta tensors from
// build_grouped_gemm_meta(); the caller keeps them alive across the kernel run.
template <typename Types>
inline typename Types::Gemm::Arguments args_from_options(
    const torch::Tensor& problem_sizes,  // int32  [num_segments, 3]  (M_s, N, K)
    const torch::Tensor& a_ptrs,         // int64  [num_segments]     ptrs into A
    const torch::Tensor& b_ptrs,         // int64  [num_segments]     ptrs into B
    const torch::Tensor& c_ptrs,         // int64  [num_segments]     ptrs into C (aliased to d for beta=0)
    const torch::Tensor& d_ptrs,         // int64  [num_segments]     ptrs into D
    const torch::Tensor& stride_A,       // int64  [num_segments]     leading dim of A = K
    const torch::Tensor& stride_B,       // int64  [num_segments]     leading dim of B = K
    const torch::Tensor& stride_C,       // int64  [num_segments]     must equal stride_D
    const torch::Tensor& stride_D,       // int64  [num_segments]     leading dim of D = N
    int num_segments,
    float alpha,   // epilogue scalar: D = alpha * (A @ B) + beta * C
    float beta) {  //   A-fwd uses (1.0, 0.0); B-fwd / QKV-B-fwd use (1.0, 1.0) for in-place residual.
  using GemmKernel = typename Types::GemmKernel;
  using ElementMma = typename Types::ElementMma;
  using ElementMmaB = typename Types::ElementMmaB;
  using ElementOutput = typename Types::ElementOutput;
  using StrideA = typename Types::StrideA;
  using StrideB = typename Types::StrideB;
  using StrideC = typename Types::StrideC;
  using StrideD = typename Types::StrideD;

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = static_cast<int>(d_ptrs.device().index());
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  auto* problem_sizes_ptr = reinterpret_cast<UnderlyingProblemShapeType*>(problem_sizes.data_ptr<int32_t>());
  auto* a_ptrs_dev = reinterpret_cast<ElementMma const**>(a_ptrs.data_ptr());
  auto* b_ptrs_dev = reinterpret_cast<ElementMmaB const**>(b_ptrs.data_ptr());
  auto* d_ptrs_dev = reinterpret_cast<ElementOutput**>(d_ptrs.data_ptr());
  auto* stride_A_ptr = reinterpret_cast<StrideA*>(stride_A.data_ptr<int64_t>());
  auto* stride_B_ptr = reinterpret_cast<StrideB*>(stride_B.data_ptr<int64_t>());
  auto* stride_C_ptr = reinterpret_cast<StrideC*>(stride_C.data_ptr<int64_t>());
  auto* stride_D_ptr = reinterpret_cast<StrideD*>(stride_D.data_ptr<int64_t>());

  using RasterOrderOptions =
      typename cutlass::gemm::kernel::detail::PersistentTileSchedulerXeGroup<GroupedProblemShape>::RasterOrderOptions;

  using PtrCType = decltype(std::declval<typename GemmKernel::EpilogueArguments>().ptr_C);
  auto c_ptrs_dev = reinterpret_cast<PtrCType>(c_ptrs.data_ptr());

  typename Types::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      typename GemmKernel::ProblemShape{num_segments, problem_sizes_ptr, nullptr},
      typename GemmKernel::MainloopArguments{a_ptrs_dev, stride_A_ptr, b_ptrs_dev, stride_B_ptr},
      typename GemmKernel::EpilogueArguments{
          {},  // fusion_args placeholder; alpha/beta filled in below
          c_ptrs_dev,  // ptr_C (caller-supplied; aliased to d_ptrs for beta=0 paths)
          stride_C_ptr,
          d_ptrs_dev,  // ptr_D
          stride_D_ptr},
      hw_info,
      typename GemmKernel::TileSchedulerArguments{1, RasterOrderOptions::AlongN}};

  // Epilogue thread arguments: D = alpha * acc + beta * C.
  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.alpha = alpha;
  fusion_args.beta = beta;
  fusion_args.alpha_ptr = nullptr;
  fusion_args.beta_ptr = nullptr;
  fusion_args.alpha_ptr_array = nullptr;
  fusion_args.beta_ptr_array = nullptr;
  fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
  fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};

  return arguments;
}

}  // namespace at::native::xpu
