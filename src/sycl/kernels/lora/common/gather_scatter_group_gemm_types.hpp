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
    \brief Type bundle of the chunked-SGMV LoRA gather / scatter grouped GEMM on sycl-tla's
           GemmUniversal: the decode-path sibling of GroupGemmTypes (group_gemm_types.hpp).

    Same layers as the prefill kernels -- GroupGemmLoraFwd -> GemmUniversalAdapter ->
    GemmUniversal (grouped Xe kernel, PersistentTileSchedulerXeGroup) -- with custom pieces
    where the decode path differs:
      ProblemShape       SegmentSliceProblemShape (group shapes read on the device)
      TileScheduler      PersistentTileSchedulerXeGroup<SegmentSliceProblemShape> (specialized)
      CollectiveMainloop GatherAGroupMma<MainloopXeGatherAGroup<SGK, BLOCK_B>, ...> (gathered A)
      CollectiveEpilogue scale / residual scatter, or split-K SLM reduction + scatter
*/

#pragma once

#include <ATen/ATen.h>

#include <cute/tensor.hpp>

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/half.h"
#include "sycl/kernels/lora/collective/xe_gather_a_group_mma.hpp"
#include "sycl/kernels/lora/collective/xe_scatter_group_epilogue.hpp"
#include "sycl/kernels/lora/collective/xe_segment_problem_shape.hpp"
#include "sycl/kernels/lora/collective/xe_segment_tile_scheduler.hpp"
#include "sycl/kernels/lora/common/lora_element_type.hpp"
#include "sycl/kernels/lora/device/group_gemm_lora_launcher.hpp"

namespace at::native::xpu::lora {

// BM x (SG_N * NSG_N) x BK work-group tile; NSG_N sub-groups along N share the TiledMma and
// SGK copies of it split K.
template <
    typename TensorDType,
    int BM,
    int SG_N,
    int BK,
    int NSG_N,
    int SGK,
    template <class, int, int, int, int>
    class Epilogue_,
    bool BLOCK_B>
struct GatherScatterGroupGemmTypes {
  using Element = typename at::native::xpu::ToCutlassElementType<TensorDType>::type;
  using TileShape = cute::Shape<cute::Int<BM>, cute::Int<SG_N * NSG_N>, cute::Int<BK>>;
  using ThreadLayout = cute::Layout<cute::Shape<cute::_1, cute::Int<NSG_N>, cute::_1>>;
  using MmaAtom = cute::XE_DPAS_TT<8, float, Element>;
  using TiledMma =
      typename cute::TiledMMAHelper<cute::MMA_Atom<MmaAtom>, cute::Layout<TileShape>, ThreadLayout>::TiledMMA;

  using ProblemShape = cutlass::lora::collective::SegmentSliceProblemShape;
  using Stride = cute::Stride<int64_t, cute::_1, cute::_0>;
  using DispatchPolicy = cutlass::lora::collective::MainloopXeGatherAGroup<SGK, BLOCK_B>;
  using CollectiveMainloop =
      cutlass::lora::collective::GatherAGroupMma<DispatchPolicy, TileShape, Element, Stride, Element, Stride, TiledMma>;
  using CollectiveEpilogue = Epilogue_<Element, BM, SG_N, NSG_N, SGK>;

  using GemmKernel = cutlass::gemm::kernel::
      GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue, cutlass::gemm::GroupScheduler>;
  static_assert(
      GemmKernel::TileScheduler::kIsLoraSegmentScheduler, "GemmUniversal must use the LoRA segment scheduler");
  using Gemm = at::native::xpu::GroupGemmLoraFwd<GemmKernel>;  // shared with the prefill kernels
};

}  // namespace at::native::xpu::lora
