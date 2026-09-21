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
  group_gemm_lora_launcher.hpp), differing from the merged sgemm_lora_a_fwd
  kernel ONLY in the tile shape -- so the merged A-fwd kernel is left untouched.

  Why a separate small tile: the LoRA-A shrink output is skinny in N
  (N = num_slices * rank, typically 16..48) and per-segment short in M (decode
  chunk sizes 16..128). The canonical 256 x 256 x 32 tile that sgemm_lora_a_fwd
  uses computes a 256-wide N block for a 16-wide output -- wasting >90% of the N
  dimension -- and a 256-row M block for a <=128-row segment. Across the many
  small segments of a decode batch that waste dominates, and the Triton chunked
  shrink (which sizes BLOCK_N = next_pow2(N)) beats it badly.

  The tile below cuts the CTA tile to 128 x 64 x 32 while keeping the *validated*
  8 x 4 x 1 subgroup layout used by every merged BMG grouped GEMM (so it is a
  guaranteed-legal TiledMMA: 128/(8*8)=2 M-iters, 64/(16*4)=1 N-iter). Relative
  to 256 x 256 that is 4x less wasted N compute and 2x less wasted M compute for
  the shrink shape space, without touching the subgroup geometry.

  Adding another tile is a two-step change (mirrors the A-fwd convention):
    1) Define a new option tag here.
    2) Register (tag name, C++ type) in ChunkedSgmvLoraShrinkXe20.cmake.
  The dtype/tile dispatch in ChunkedSgmvLoraShrink.cpp then picks a tag per call.
*/

#pragma once

#include <cute/layout.hpp>

#include "cutlass/layout/matrix.h"
#include "sycl/kernels/lora/common/group_gemm_types.hpp"

namespace chunked_sgmv_lora_shrink_impl {

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

}  // namespace chunked_sgmv_lora_shrink_impl
