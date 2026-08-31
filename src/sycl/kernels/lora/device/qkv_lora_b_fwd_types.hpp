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
  \brief Tile-configuration option tags for the fused QKV LoRA-B forward.

  This is the single place where a tile variant is registered: each option tag
  (e.g. QKVLoraBFwdTileTall) bundles the CUTE tile shape, the subgroup thread
  layout, the B layout, and the pipeline-stages count that the pointer-array
  grouped GEMM consumes, and exposes a Types<T> alias binding those knobs into
  the shared GroupGemmTypes<> traits bundle (common/group_gemm_types.hpp). The
  runQKVLoraBFwd<T, TileOpt>() one-shot entry that consumes these tags lives in
  qkv_lora_b_fwd_runner.hpp, and the shared, reusable grouped-GEMM core
  (lifecycle + pointer-array launcher) lives in group_gemm_lora_launcher.hpp --
  shared verbatim with the A-fwd / B-fwd kernels.

  The QKV-B weight tensor is [num_loras, N_total, K] row-major -- the same
  [num_loras, N, K] shape the plain LoRA-B weights have -- so B is ColumnMajor
  when viewed as B in the A @ B^T grouped-GEMM contract. The QKV specifics (3
  groups per segment, variable N, sliced A/D) are just arguments to the shared
  sliced build in common/grouped_gemm_meta.hpp (n_slices = 3 + output_offset).

  Adding a new tile is a two-step change:
    1) Define a new option tag here.
    2) Register (tag name, C++ type) in QKVLoraBFwdXe20.cmake.
  The dtype dispatch in QKVLoraBFwd.cpp then picks a tag per call.
*/

#pragma once

#include <cute/layout.hpp>

#include "cutlass/layout/matrix.h"
#include "sycl/kernels/lora/common/group_gemm_types.hpp"

namespace qkv_lora_b_fwd_impl {

//----------------- Tile / thread / staging option tag -----------------------//
// QKV LoRA-B is a K-thin (rank K = 16..64), memory-bandwidth-bound grouped GEMM:
// its dominant cost is streaming the M x N output, not the tiny reduction. The
// tall/thin 32 x 512 tile (upstream 05_bmg_gemm_with_epilogues) is the fastest
// config across the shape space -- a full tile sweep (benchmark/
// bench_qkv_lora_b_fwd.py) measured it +8..27% over the canonical 256 x 256 tile
// on non-residual shapes, even at large per-group M. The reason is physical: the
// tiny 32-row M tile maximizes the workgroup count, which saturates DRAM
// bandwidth and hides the memory-bound epilogue latency, beating amortization of
// the near-absent MMA over a big tile. The (TileShape, ThreadLayout) pair is
// lifted verbatim from that validated upstream kernel (same XE_DPAS_TT<8, float,
// T> atom).
//   TileShape 32 x 512 x 32, ThreadLayout 2 x 16 x 1 (32 subgroups / workgroup)
struct QKVLoraBFwdTileTall {
  using TileShape = cute::Shape<cute::_32, cute::_512, cute::_32>;
  using ThreadLayout =
      cute::Layout<cute::Shape<cute::_2, cute::_16, cute::_1>, cute::Stride<cute::_16, cute::_1, cute::_0>>;
  using LayoutB = cutlass::layout::ColumnMajor;
  static constexpr int PipelineStages = 2;

  template <typename T>
  using Types = at::native::xpu::GroupGemmTypes<T, TileShape, ThreadLayout, LayoutB, PipelineStages>;
};

}  // namespace qkv_lora_b_fwd_impl
