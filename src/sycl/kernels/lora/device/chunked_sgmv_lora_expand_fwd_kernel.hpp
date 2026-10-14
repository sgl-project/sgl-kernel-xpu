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
/*!
  \file
  \brief Gather-fused DPAS kernel of the chunked-SGMV LoRA expand (LoRA-B) forward:
         out[perm[r], n] = scale[l] * x[perm[r], band(n)] @ W[l][n]^T (+ base),
         K = rank small, N large; per-slice column tiles, block-write epilogue.
*/

#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

#include "sycl/SYCLHelpers.h"
#include "sycl/kernels/lora/common/lora_dpas_primitives.hpp"

namespace chunked_sgmv_lora_expand_fwd_impl {

using lora_dpas::float8_t;
using lora_dpas::load_k_block_fast;
using lora_dpas::load_k_block_masked;
using lora_dpas::locate_row_tile;
using lora_dpas::mma_k_block;
using lora_dpas::sg_block_read;
using lora_dpas::sg_block_write;

// Sub-group tile: (8 * MB) rows x (16 * NT) columns, K = rank in KB chunks.
// Work-group: SGN sub-groups side by side along N. dim 0 = row tile, dim 1 =
// column group, enumerated slice by slice so a tile never straddles a slice.
template <typename Elem, int MB, int NT, int KB, bool ALIGNED>
struct ChunkedExpandDpasKernel {
  static constexpr int BM = 8 * MB;

  const uint16_t* x;     // [T, n_slices * R]
  const uint16_t* w;     // [L, N_total, R]
  uint16_t* out;         // [T, N_total]
  const uint16_t* base;  // [T, N_total] or nullptr
  const int32_t* seg_indptr;
  const int32_t* weight_indices;
  const int64_t* perm;  // logical -> physical, or nullptr (identity)
  const int32_t* slice_offsets;
  const float* scalings;
  int num_segments;
  int R;
  int N_total;
  int n_slices;
  int sgn;
  bool io_aligned;  // out (and base, if any) 16-byte aligned

  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<2> item) const {
    auto sg = item.get_sub_group();
    const int lane = sg.get_local_linear_id();
    const int sgid = sg.get_group_linear_id();

    // Column group -> (slice p, column band [col_lo, col_hi)).
    const int wg_cols = sgn * NT * 16;
    int g = item.get_group(1);
    int p = 0, col_lo = 0, col_hi = 0;
    for (; p < n_slices; ++p) {
      const int lo = slice_offsets[p];
      const int hi = slice_offsets[p + 1];
      const int ng = (hi - lo + wg_cols - 1) / wg_cols;
      if (g < ng) {
        col_lo = lo + g * wg_cols;
        col_hi = hi;
        break;
      }
      g -= ng;
    }
    if (p == n_slices) {
      return;
    }
    const int n0 = col_lo + sgid * NT * 16;
    if (n0 >= col_hi) {
      return;  // no work-group barriers below
    }

    int seg = 0, seg_tile = 0;
    if (!locate_row_tile<BM>(sg, seg_indptr, num_segments, item.get_group(0), seg, seg_tile)) {
      return;
    }
    const int row0 = seg_indptr[seg] + seg_tile * BM;
    const int rows = sycl::min(BM, seg_indptr[seg + 1] - row0);
    const int64_t lora = weight_indices[seg];
    const int64_t x_ld = static_cast<int64_t>(n_slices) * R;

    const uint16_t* arow[BM];
    int64_t prow[BM];
#pragma unroll
    for (int r = 0; r < BM; ++r) {
      const int lr = row0 + sycl::min(r, rows - 1);
      prow[r] = perm ? perm[lr] : static_cast<int64_t>(lr);
      arow[r] = x + prow[r] * x_ld + static_cast<int64_t>(p) * R;
    }
    const uint16_t* brow[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      const int n = sycl::min(n0 + t * 16 + lane, col_hi - 1);
      brow[t] = w + (lora * N_total + n) * R;
    }

    float8_t acc[MB][NT];
#pragma unroll
    for (int m = 0; m < MB; ++m) {
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        acc[m][t] = 0.f;
      }
    }
    // Full blocks with fast loads, then the rank tail (or every block when rows
    // are unaligned) with masked loads -- separate loops, see the shrink kernel.
    const int r_fast = ALIGNED ? R / KB * KB : 0;
    for (int k0 = 0; k0 < r_fast; k0 += KB) {
      sycl::vec<uint16_t, KB / 16> a[BM];
      uint32_t b[NT][KB / 2];
      load_k_block_fast<BM, NT, KB>(arow, brow, k0, a, b);
      mma_k_block<Elem, MB, NT, KB>(a, b, acc);
    }
    for (int k0 = r_fast; k0 < R; k0 += KB) {
      sycl::vec<uint16_t, KB / 16> a[BM];
      uint32_t b[NT][KB / 2];
      load_k_block_masked<BM, NT, KB>(arow, brow, k0, R, lane, a, b);
      mma_k_block<Elem, MB, NT, KB>(a, b, acc);
    }

    // Epilogue: scale (+ residual), then one block write per full output row.
    const float scale = scalings[lora];
    const bool full_cols = n0 + NT * 16 <= col_hi;
#pragma unroll
    for (int m = 0; m < MB; ++m) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int r = m * 8 + i;
        if (r >= rows) {
          continue;
        }
        const int64_t off = prow[r] * N_total + n0;
        // Block read / write needs a 16-byte aligned row address. With 16-byte
        // aligned out / base pointers (io_aligned, host-checked) that is exactly
        // off % 8 == 0; slice starts and N_total are arbitrary, so test per row.
        // (A direct per-row pointer test here costs ~30% expand throughput.)
        if (full_cols && io_aligned && (off & 7) == 0) {
          sycl::vec<float, NT> v;
#pragma unroll
          for (int t = 0; t < NT; ++t) {
            v[t] = acc[m][t][i] * scale;
          }
          if (base) {
            const auto bv = sg_block_read<NT>(base + off);
#pragma unroll
            for (int t = 0; t < NT; ++t) {
              v[t] += Elem::to_float(bv[t]);
            }
          }
          sycl::vec<uint16_t, NT> o;
#pragma unroll
          for (int t = 0; t < NT; ++t) {
            o[t] = Elem::from_float(v[t]);
          }
          sg_block_write<NT>(out + off, o);
        } else {
#pragma unroll
          for (int t = 0; t < NT; ++t) {
            const int n = n0 + t * 16 + lane;
            if (n < col_hi) {
              float v = acc[m][t][i] * scale;
              if (base) {
                v += Elem::to_float(base[off + t * 16 + lane]);
              }
              out[off + t * 16 + lane] = Elem::from_float(v);
            }
          }
        }
      }
    }
  }
};

}  // namespace chunked_sgmv_lora_expand_fwd_impl
