/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
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
#pragma once

#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"

namespace moe_w4a16 {
using namespace cute;

class xe_gemm_policy_base {
 public:
  using WGTile = Shape<_256, _256, _32>;
  using SGLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;

  // Copy can be tuned for better performance. void => use make_block_2d_copy_*.
  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;
  using GmemTiledCopyD = void;
};

// Policy menu. Every policy keeps the *per-subgroup* tile at 32x32: the 4-bit
// mainloop needs its dequantised B fragment live alongside the accumulators, and
// a wider per-subgroup tile overruns the 256-GRF budget (a 256x256 / 8x4 variant
// compiles with ~190 spilled registers and runs at a third of the speed). So the
// work-group tile is scaled by adding or removing subgroups, not by widening
// them. The subgroup count per dimension must be a power of two -- cute's tile
// division rejects 3 and 6.
//
// Which one runs is decided per call by rows-per-expert; see
// select_w4a16_tile_m() in GroupGemmW4A16Xe20.cpp.

// avg_m <= 4
class w4a16_policy_m_8_n_64 : public xe_gemm_policy_base {
 public:
  using WGTile = Shape<_8, _64, _32>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
};

// avg_m <= 8
class w4a16_policy_m_16_n_64 : public xe_gemm_policy_base {
 public:
  using WGTile = Shape<_16, _64, _32>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
};

// Small avg_m: the 64-wide N tile beats a 256-wide one at this M (the wide-N
// variant measures ~35 TFLOP/s against this one's ~49).
class w4a16_policy_m_32_n_64 : public xe_gemm_policy_base {
 public:
  using WGTile = Shape<_32, _64, _32>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
};

// Mid range. Covers the rows-per-expert values where an M=128 tile would
// compute up to twice the rows an expert actually has.
class w4a16_policy_m_64_n_128 : public xe_gemm_policy_base {
 public:
  using WGTile = Shape<_64, _128, _32>;
  using SGLayout = Layout<Shape<_2, _4, _1>, Stride<_4, _1, _0>>;
};

// Large avg_m. Replaces the previous <_128,_256,_32> / 4x8 policy: at equal M
// the 128-wide N tile is never slower and is faster wherever N is not a
// multiple of 256 (GPT-OSS N=5760 and 2880 both leave a half tile idle),
// measuring 68.2 vs 66.5 TFLOP/s on GPT-OSS gemm1 at avg_m=512 and 68.3 vs
// 68.1 on DeepSeek-V4 gemm1 at avg_m=256.
class w4a16_policy_m_128_n_128 : public xe_gemm_policy_base {
 public:
  using WGTile = Shape<_128, _128, _32>;
  using SGLayout = Layout<Shape<_4, _4, _1>, Stride<_4, _1, _0>>;
};

}  // namespace moe_w4a16
