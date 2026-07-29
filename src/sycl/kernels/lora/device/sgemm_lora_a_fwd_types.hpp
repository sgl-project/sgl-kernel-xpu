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
  \brief Tile-configuration option tags for LoRA-A forward.

  This is the single place where a tile variant is registered: each option tag
  (e.g. LoraAFwdTileLarge) bundles the CUTE tile shape, the subgroup thread
  layout, the B layout, and the pipeline-stages count that the pointer-array
  grouped GEMM consumes, and exposes a Types<T> alias that binds those knobs
  into the shared GroupGemmTypes<> traits bundle (common/group_gemm_types.hpp).
  The runSgemmLoraAFwd<T, TileOpt>() one-shot entry that consumes these tags
  lives in sgemm_lora_a_fwd_runner.hpp, and the shared, reusable grouped-GEMM
  core (lifecycle + device-side metadata build) lives in
  group_gemm_lora_launcher.hpp -- kept intact so the other LoRA kernels (e.g.
  B-fwd) can reuse it.

  Adding a new tile is a two-step change:
    1) Define a new option tag here.
    2) Register (tag name, C++ type) in SGEMMLoraAFwdXe20.cmake.
  The dtype dispatch in sgemm_lora_a_fwd.cpp then picks a tag per call.
*/

#pragma once

#include <cute/layout.hpp>

#include "cutlass/layout/matrix.h"
#include "sycl/kernels/lora/common/group_gemm_types.hpp"

namespace sgemm_lora_a_fwd_impl {

//----------------- Tile / thread / staging option tags ----------------------//
// LayoutB is ColumnMajor for all tiles: the LoRA weight tensor is
// [num_loras, N, K] row-major, which is ColumnMajor when viewed as B in the
// A @ B^T grouped GEMM contract used here (the auto-selected copy atom
// free-transposes it).
//
// Each tag also exposes Types<T> -- the fully-assembled GroupGemmTypes<> bundle
// for element type T -- which is what the launcher template is instantiated on.

// Canonical BMG grouped-GEMM tile (upstream 04_bmg_grouped_gemm):
//   TileShape       = 256 x 256 x 32
//   ThreadLayout    = 8 x 4 x 1   (32 subgroups / workgroup)
//   PipelineStages  = 2
struct LoraAFwdTileLarge {
  using TileShape = cute::Shape<cute::_256, cute::_256, cute::_32>;
  using ThreadLayout =
      cute::Layout<cute::Shape<cute::_8, cute::_4, cute::_1>, cute::Stride<cute::_4, cute::_1, cute::_0>>;
  using LayoutB = cutlass::layout::ColumnMajor;
  static constexpr int PipelineStages = 2;

  template <typename T>
  using Types = at::native::xpu::GroupGemmTypes<T, TileShape, ThreadLayout, LayoutB, PipelineStages>;
};

}  // namespace sgemm_lora_a_fwd_impl
