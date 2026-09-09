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
#pragma once

#include "cutlass/cutlass.h"

namespace cutlass::fmha::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////
// Relative attention: the sheared bias surface, shared by the prefill and decode mainloops.
//
// A relative bias is logically bias[token, head, rel] with rel = row_kv - col_kv, defined for
// rel in [0, rel_extent) and zero outside -- a band, not a rectangle. The producing kernel writes
// it sheared: for each query token it right-aligns that token's band into a k_tile-aligned column
// window, so a mainloop reading a rectangle of the surface sees the band as a diagonal and needs
// no gather. Columns outside the band hold zero, which is also the correct bias for them.
//
// Padding and shearing are the producer's job. A consumer needs only the window origin and the
// surface's row stride, and never recomputes the column count -- rel_bias_padded_cols exists so
// that the producer and whoever allocates the surface agree on it.
//
// Surface layout: [total_q, heads_q, rel_bias_padded_cols], bf16, row-major. The band is widened
// to a whole number of K tiles, making the surface unconditionally legal for the block-2D atom.
CUTLASS_HOST_DEVICE constexpr int rel_bias_band_cols(int rel_extent, int k_tile) {
  return (rel_extent + k_tile - 1) / k_tile * k_tile;
}

// Padding covers the band's drift across one M tile plus the alignment slack of the K tile.
// `m_drift` must be a multiple of k_tile so the column count stays K-tile aligned.
CUTLASS_HOST_DEVICE constexpr int rel_bias_padded_cols(int rel_extent, int m_drift, int k_tile) {
  return rel_bias_band_cols(rel_extent, k_tile) + m_drift + k_tile;
}

// Column of the sheared bias that holds KV column 0 for an M tile whose first row sits at
// `row_kv_first`. Floor division keeps every in-band column within the allocated surface.
CUTLASS_HOST_DEVICE constexpr int rel_bias_col_origin(int row_kv_first, int rel_extent, int k_tile) {
  int const left = row_kv_first - rel_bias_band_cols(rel_extent, k_tile) + 1;
  int const q = left / k_tile;
  return ((left % k_tile != 0 && left < 0) ? q - 1 : q) * k_tile;
}

}  // namespace cutlass::fmha::collective
