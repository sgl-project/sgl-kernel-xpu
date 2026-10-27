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
  \brief Tile-configuration option tags for the chunked-SGMV LoRA expand kernel.

  Adding another tile is a two-step change (mirrors the prefill LoRA kernels):
    1) Define a new option tag here.
    2) Register (tag name, C++ type) in ChunkedSgmvLoraExpandFwd.cmake.
  The entrypoint (ChunkedSgmvLoraExpandFwd.cpp) then picks a tag per call.
*/

#pragma once

namespace chunked_sgmv_lora_expand_fwd_impl {

// Sub-group tile (8 * MB) rows x (16 * NT) columns, K = rank in KB-wide steps
// (KB = 16: the rank-tail path assumes 16-wide steps), SGN sub-groups side by
// side along N. Tuned on BMG (B580) over the decode shapes in
// benchmark/bench_chunked_sgmv_lora_expand_fwd.py.
struct ChunkedExpandTileWide {
  static constexpr int MB = 4;
  static constexpr int NT = 4;
  static constexpr int KB = 16;
  static constexpr int SGN = 4;
};

}  // namespace chunked_sgmv_lora_expand_fwd_impl
