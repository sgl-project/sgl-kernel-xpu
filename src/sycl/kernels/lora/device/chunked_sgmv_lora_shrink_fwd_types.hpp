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
/*!
  \file
  \brief Tile-configuration option tag for the chunked-SGMV LoRA "shrink" GEMM.

  This is a dedicated, small-N grouped-GEMM tile for the LoRA-A "shrink"
  projection. It reuses the shared grouped-GEMM core (GroupGemmTypes<> +
  group_gemm_lora_launcher.hpp).

  Adding another tile is a two-step change (mirrors the A-fwd convention):
    1) Define a new option tag here.
    2) Register (tag name, C++ type) in ChunkedSgmvLoraShrinkFwdXe20.cmake.
  The dtype/tile dispatch in ChunkedSgmvLoraShrinkFwd.cpp then picks a tag per call.
*/

#pragma once

#include <cute/layout.hpp>

#include "cutlass/layout/matrix.h"
#include "sycl/kernels/lora/common/group_gemm_types.hpp"

namespace chunked_sgmv_lora_shrink_fwd_impl {

//----------------- Tile / thread / staging option tag -----------------------//
// LayoutB is ColumnMajor: the LoRA weight tensor is [num_loras, N, K] row-major,
// which is ColumnMajor when viewed as B in the A @ B^T grouped GEMM (the
// auto-selected copy atom free-transposes it) -- identical to the A-fwd contract.
//
//   TileShape       = 128 x 64 x 32   (small N + short M for the skinny shrink)
//   ThreadLayout    = 8 x 4 x 1        (32 subgroups / workgroup; validated)
//   PipelineStages  = 2
struct ChunkedShrinkTileSmall {
  using TileShape = cute::Shape<cute::_128, cute::_64, cute::_32>;
  using ThreadLayout =
      cute::Layout<cute::Shape<cute::_8, cute::_4, cute::_1>, cute::Stride<cute::_4, cute::_1, cute::_0>>;
  using LayoutB = cutlass::layout::ColumnMajor;
  static constexpr int PipelineStages = 2;

  template <typename T>
  using Types = at::native::xpu::GroupGemmTypes<T, TileShape, ThreadLayout, LayoutB, PipelineStages>;
};

}  // namespace chunked_sgmv_lora_shrink_fwd_impl
