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
    \brief LoRA-A forward launch wrapper (perf knobs + entry point) over the
    shared grouped-GEMM launcher core.
*/

#pragma once

#include "sycl/kernels/lora/device/group_gemm_lora_launcher.hpp"

namespace at::native::xpu {

//----------------- Compile-time perf knobs (LoRA-A forward) -----------------//
// Pinned once here for every LoRA-A entry point (fp16/bf16 single GEMM) so the
// tile / subgroup / layout mix lives in a single place.
//   TileShape    = 256 x 256 x 32   -- canonical tile from 04_bmg_grouped_gemm.
//   ThreadLayout = 8 x 4 x 1        -- 32 subgroups per workgroup.
//   LayoutB      = ColumnMajor      -- free-transpose B (auto copy atom transposes).
//   PipelineStages = 2              -- matches the upstream BMG reference.
using LoraAFwdTileShape = cute::Shape<cute::_256, cute::_256, cute::_32>;
using LoraAFwdThreadLayout =
    cute::Layout<cute::Shape<cute::_8, cute::_4, cute::_1>, cute::Stride<cute::_4, cute::_1, cute::_0>>;
using LoraAFwdLayoutB = cutlass::layout::ColumnMajor;
constexpr int kLoraAFwdPipelineStages = 2;

//----------------- fp16 / bf16 launch (single grouped GEMM) ------------------//
template <typename TensorDType>
void launch_sgemm_lora_a_fwd(
    const torch::Tensor& input_x,
    const torch::Tensor& weights,
    const torch::Tensor& seg_indptr_i32,
    const torch::Tensor& weight_indices_i32,
    torch::Tensor& output,
    const int stack_num,
    const int max_rank,
    const int num_segments,
    sycl::queue& queue) {
  const int K = static_cast<int>(input_x.size(1));  // input_dim
  const int N = stack_num * max_rank;               // output columns
  const int64_t elem_bytes = static_cast<int64_t>(sizeof(TensorDType));
  const auto device = input_x.device();

  auto meta =
      build_grouped_gemm_meta(seg_indptr_i32, weight_indices_i32, N, K, num_segments, elem_bytes, device, queue);

  auto a_ptrs = make_device_ptrs(input_x, meta.a_off, queue);
  auto b_ptrs = make_device_ptrs(weights, meta.b_off, queue);
  auto d_ptrs = make_device_ptrs(output, meta.d_off, queue);

  // The launcher uses the modern (non-legacy) grouped path
  // (MainloopXeL1StagedGroup), which auto-selects the XMX DPAS MMA atom and all
  // gmem load/store/prefetch copy atoms per dtype.
  launch_group_gemm_lora_fwd<
      TensorDType,
      LoraAFwdTileShape,
      LoraAFwdThreadLayout,
      LoraAFwdLayoutB,
      kLoraAFwdPipelineStages>(
      queue,
      meta.problem_sizes,
      a_ptrs,
      b_ptrs,
      /*c_ptrs   =*/d_ptrs,
      d_ptrs,
      meta.stride_A,
      meta.stride_B,
      /*stride_C =*/meta.stride_D,
      meta.stride_D,
      num_segments,
      /*alpha    =*/1.0f,
      /*beta     =*/0.0f);
}

}  // namespace at::native::xpu
