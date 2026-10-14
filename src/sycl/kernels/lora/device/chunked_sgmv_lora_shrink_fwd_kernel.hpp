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
  \brief Gather-fused DPAS kernel of the chunked-SGMV LoRA shrink (LoRA-A) forward:
         out[perm[r], :] = x[perm[r], :] @ W[l]^T, K = input_dim large, N small. An
         intra-work-group split-K (SGK sub-groups, SLM reduction) fills the GPU.
*/

#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

#include "sycl/SYCLHelpers.h"
#include "sycl/kernels/lora/common/lora_dpas_primitives.hpp"

namespace chunked_sgmv_lora_shrink_fwd_impl {

using lora_dpas::float8_t;
using lora_dpas::load_k_block_fast;
using lora_dpas::load_k_block_masked;
using lora_dpas::locate_row_tile;
using lora_dpas::mma_k_block;

// Sub-group tile: (8 * MB) rows x (16 * NT) columns. Work-group: SGK sub-groups
// splitting K of the same tile; dim 0 = row tile * n_groups + column group.
template <typename Elem, int MB, int NT, int KB, bool ALIGNED>
struct ChunkedShrinkDpasKernel : public __SYCL_KER_CONFIG_CONVENTION__ {
  static constexpr int BM = 8 * MB;
  static constexpr int TILE_ELEMS = MB * NT * 8 * 16;

  const uint16_t* x;  // [T, K]
  const uint16_t* w;  // [L, N, K]
  uint16_t* out;      // [T, N]
  const int32_t* seg_indptr;
  const int32_t* weight_indices;
  const int64_t* perm;  // logical -> physical, or nullptr (identity)
  int num_segments;
  int K;
  int N;
  int sgk;
  int n_groups;
  sycl::local_accessor<float, 1> slm;

  ChunkedShrinkDpasKernel(
      const uint16_t* x,
      const uint16_t* w,
      uint16_t* out,
      const int32_t* seg_indptr,
      const int32_t* weight_indices,
      const int64_t* perm,
      int num_segments,
      int K,
      int N,
      int sgk,
      int n_groups)
      : x(x),
        w(w),
        out(out),
        seg_indptr(seg_indptr),
        weight_indices(weight_indices),
        perm(perm),
        num_segments(num_segments),
        K(K),
        N(N),
        sgk(sgk),
        n_groups(n_groups) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {
    slm = sycl::local_accessor<float, 1>(sycl::range<1>(sgk > 1 ? sgk * TILE_ELEMS : 1), cgh);
  }

  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> item) const {
    auto sg = item.get_sub_group();
    const int lane = sg.get_local_linear_id();
    const int sgid = sg.get_group_linear_id();
    const int wg = item.get_group(0);
    const int ng = wg % n_groups;

    int seg = 0, seg_tile = 0;
    if (!locate_row_tile<BM>(sg, seg_indptr, num_segments, wg / n_groups, seg, seg_tile)) {
      return;  // uniform across the work-group: every sub-group sees the same tile
    }
    const int row0 = seg_indptr[seg] + seg_tile * BM;
    const int rows = sycl::min(BM, seg_indptr[seg + 1] - row0);
    const int64_t lora = weight_indices[seg];

    // Gathered row pointers (rows past the segment clamp onto its last row).
    const uint16_t* arow[BM];
#pragma unroll
    for (int r = 0; r < BM; ++r) {
      const int lr = row0 + sycl::min(r, rows - 1);
      arow[r] = x + (perm ? perm[lr] : static_cast<int64_t>(lr)) * K;
    }
    const int n0 = ng * NT * 16;
    const uint16_t* brow[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      const int n = sycl::min(n0 + t * 16 + lane, N - 1);
      brow[t] = w + (lora * N + n) * K;
    }

    float8_t acc[MB][NT];
#pragma unroll
    for (int m = 0; m < MB; ++m) {
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        acc[m][t] = 0.f;
      }
    }

    // Blocks [0, num_fast) use the fast loads; the rest (the K tail, or every
    // block when rows are unaligned) the masked loads. Both loops keep the
    // round-robin assignment c % sgk == sgid. Keeping the paths in separate
    // loops matters: a per-block branch between them costs ~45% throughput.
    const int num_k_blocks = (K + KB - 1) / KB;
    const int num_fast = ALIGNED ? K / KB : 0;
    for (int c = sgid; c < num_fast; c += sgk) {
      sycl::vec<uint16_t, KB / 16> a[BM];
      uint32_t b[NT][KB / 2];
      load_k_block_fast<BM, NT, KB>(arow, brow, c * KB, a, b);
      mma_k_block<Elem, MB, NT, KB>(a, b, acc);
    }
    for (int c = num_fast + (sgid - num_fast % sgk + sgk) % sgk; c < num_k_blocks; c += sgk) {
      sycl::vec<uint16_t, KB / 16> a[BM];
      uint32_t b[NT][KB / 2];
      load_k_block_masked<BM, NT, KB>(arow, brow, c * KB, K, lane, a, b);
      mma_k_block<Elem, MB, NT, KB>(a, b, acc);
    }

    if (sgk > 1) {
      // SLM layout [sg][m][t][i][lane]; all work-items then reduce + store a slice.
      float* mine = &slm[0] + sgid * TILE_ELEMS;
#pragma unroll
      for (int m = 0; m < MB; ++m) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            mine[((m * NT + t) * 8 + i) * 16 + lane] = acc[m][t][i];
          }
        }
      }
      sycl::group_barrier(item.get_group());
      const int nl = item.get_local_range(0);
      for (int e = item.get_local_id(0); e < TILE_ELEMS; e += nl) {
        float s = 0.f;
        for (int g = 0; g < sgk; ++g) {
          s += slm[g * TILE_ELEMS + e];
        }
        const int l = e % 16;
        const int i = (e / 16) % 8;
        const int t = (e / 128) % NT;
        const int m = e / (128 * NT);
        const int r = m * 8 + i;
        const int n = n0 + t * 16 + l;
        if (r < rows && n < N) {
          const int64_t pr = perm ? perm[row0 + r] : static_cast<int64_t>(row0 + r);
          out[pr * N + n] = Elem::from_float(s);
        }
      }
    } else {
#pragma unroll
      for (int m = 0; m < MB; ++m) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int r = m * 8 + i;
          if (r < rows) {
            const int64_t pr = perm ? perm[row0 + r] : static_cast<int64_t>(row0 + r);
#pragma unroll
            for (int t = 0; t < NT; ++t) {
              const int n = n0 + t * 16 + lane;
              if (n < N) {
                out[pr * N + n] = Elem::from_float(acc[m][t][i]);
              }
            }
          }
        }
      }
    }
  }
};

}  // namespace chunked_sgmv_lora_shrink_fwd_impl
