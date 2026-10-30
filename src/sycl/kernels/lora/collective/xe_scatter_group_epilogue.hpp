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
  \brief Scatter epilogues of the chunked-SGMV LoRA kernels for sycl-tla's grouped GemmUniversal
         (xe_gemm_array_cooperative.hpp): write the tile's rows to out[perm[r]].

  Same contract as the stock grouped Xe epilogue: global Arguments / Params, a per-group Base
  built on the device by to_base_arguments(), and TensorStorage, which the kernel places in
  the work-group's SLM.

  Both read the C fragment, whose element t * BM + i is (row i, column sg_n0 + 16 t + lane), so
  one output row of a sub-group is NT = SG_N / 16 elements per lane: one sub-group block write.

    * ScaleResidualScatterGroupEpilogue : out = scale[adapter] * acc (+ residual), one block
                                          write per row (expand / LoRA-B). No split-K.
    * SplitKScatterGroupEpilogue        : sums the SGK partial tiles through SLM, then stores
                                          (shrink / LoRA-A). One sub-group along N.
*/

#pragma once

#include <cstdint>
#include <cute/tensor.hpp>
#include <sycl/sycl.hpp>

#include "cutlass/cutlass.h"
#include "sycl/kernels/lora/collective/xe_gather_scatter_copy.hpp"
#include "sycl/kernels/lora/collective/xe_segment_problem_shape.hpp"

namespace cutlass::lora::collective {

// Types the grouped kernel reads from every epilogue (strides only feed its rank-3 checks).
template <class Element_>
struct ScatterGroupEpilogueTypes {
  using ElementC = Element_;
  using ElementD = Element_;
  using StrideC = cute::Stride<int64_t, cute::_1, cute::_0>;
  using InternalStrideC = StrideC;
  using StrideD = StrideC;
  using InternalStrideD = StrideC;
  // Descriptive types GemmUniversalAdapter reads; the epilogues do their own stores.
  using GmemTiledCopyC = void;
  using GmemTiledCopyD = void;
  using ThreadEpilogueOp = void;
};

template <class Element_, int BM_, int SG_N_, int NSG_N_, int SGK_>
struct ScaleResidualScatterGroupEpilogue : ScatterGroupEpilogueTypes<Element_> {
  using Element = Element_;
  static constexpr int BM = BM_, SG_N = SG_N_, NSG_N = NSG_N_, NT = SG_N_ / 16;
  static_assert(SGK_ == 1, "the scale-residual epilogue does not reduce split-K partials");

  struct Arguments {
    uint16_t* out;          // [num_rows, N]
    const uint16_t* base;   // [num_rows, N] residual, or nullptr
    const float* scalings;  // [num_adapters]
    const int64_t* perm;    // logical -> physical row, or nullptr (identity)
    bool io_aligned;        // out (and base, if any) 16-byte aligned: block writes per row
    SegmentSliceProblemShape groups;
  };
  using Params = Arguments;
  struct TensorStorage {};

  struct Base {
    struct Arguments {
      uint16_t* out;
      const uint16_t* base;
      // Loaded in operator(), after the mainloop: reading it while the kernel builds these
      // per-group params would stall the mainloop's start on a dependent DRAM load.
      const float* scale;
      const int64_t* perm;
      int N;     // output row length (all slices)
      int n_lo;  // first output column of the slice
      int row0;  // first logical row of the segment
      bool io_aligned;
    };
    using Params = Arguments;
    Params p;

    template <class ProblemShape>
    static constexpr Params to_underlying_arguments(ProblemShape const&, Arguments const& args, void*) {
      return args;
    }

    CUTLASS_DEVICE Base(Params const& params, TensorStorage&) : p(params) {}

    template <class ProblemShapeMNKL, class SubgroupShape, class TileCoord, class Accumulator, class TiledMma>
    CUTLASS_DEVICE void operator()(
        ProblemShapeMNKL const& problem_shape_MNKL,
        SubgroupShape,
        TileCoord const& tile_coord,
        Accumulator const& accum,
        TiledMma const&,
        int thread_idx) {
      using namespace cute;
      const int lane = thread_idx % 16;
      const int sg_n = (thread_idx / 16) % NSG_N;
      const int m0 = get<0>(tile_coord) * BM;
      const int rows = sycl::min(BM, static_cast<int>(get<0>(problem_shape_MNKL)) - m0);
      const int n_end = p.n_lo + static_cast<int>(get<1>(problem_shape_MNKL));
      const int n0 = p.n_lo + get<1>(tile_coord) * (SG_N * NSG_N) + sg_n * SG_N;
      if (n0 >= n_end) {
        return;
      }
      const float scale = *p.scale;
      const bool full_cols = n0 + SG_N <= n_end;
      CUTE_UNROLL
      for (int i = 0; i < BM; ++i) {
        if (i >= rows) {
          continue;
        }
        const int lr = p.row0 + m0 + i;
        const int64_t off = (p.perm ? p.perm[lr] : static_cast<int64_t>(lr)) * p.N + n0;
        // Block read / write needs a 16-byte aligned row address; with aligned out / base
        // pointers (io_aligned) that is exactly off % 8 == 0.
        if (full_cols && p.io_aligned && (off & 7) == 0) {
          sycl::vec<float, NT> v;
          CUTE_UNROLL
          for (int t = 0; t < NT; ++t) {
            v[t] = accum(t * BM + i) * scale;
          }
          if (p.base) {
            const auto bv = sg_block_read<NT>(p.base + off);
            CUTE_UNROLL
            for (int t = 0; t < NT; ++t) {
              v[t] += float(Element::bitcast(bv[t]));
            }
          }
          sycl::vec<uint16_t, NT> o;
          CUTE_UNROLL
          for (int t = 0; t < NT; ++t) {
            o[t] = Element(v[t]).raw();
          }
          sg_block_write<NT>(p.out + off, o);
        } else {
          CUTE_UNROLL
          for (int t = 0; t < NT; ++t) {
            const int n = n0 + t * 16 + lane;
            if (n < n_end) {
              float v = accum(t * BM + i) * scale;
              if (p.base) {
                v += float(Element::bitcast(p.base[off + t * 16 + lane]));
              }
              p.out[off + t * 16 + lane] = Element(v).raw();
            }
          }
        }
      }
    }
  };

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const&, Arguments const& args, void*) {
    return args;
  }

  template <class ProblemShape>
  static bool can_implement(ProblemShape const&, Arguments const&) {
    return true;
  }

  CUTLASS_DEVICE static typename Base::Arguments to_base_arguments(Arguments const& args, int g) {
    auto const& gr = args.groups;
    return {
        args.out, args.base, args.scalings + gr.adapter(g), args.perm, gr.N, gr.n_lo(g), gr.row0(g), args.io_aligned};
  }
};

template <class Element_, int BM_, int SG_N_, int NSG_N_, int SGK_>
struct SplitKScatterGroupEpilogue : ScatterGroupEpilogueTypes<Element_> {
  using Element = Element_;
  static constexpr int BM = BM_, SG_N = SG_N_, SGK = SGK_;
  static_assert(NSG_N_ == 1, "the split-K epilogue expects one sub-group along N");
  static constexpr int TILE = BM * SG_N;  // floats in one sub-group's partial tile
  static constexpr int NumThreads = 16 * SGK;

  struct Arguments {
    uint16_t* out;  // [num_rows, N]
    const int64_t* perm;
    SegmentSliceProblemShape groups;
  };
  using Params = Arguments;
  struct TensorStorage {
    float partials[SGK > 1 ? SGK* TILE : 1];  // [sg_k][idx][lane]
  };

  struct Base {
    struct Arguments {
      uint16_t* out;
      const int64_t* perm;
      int N;
      int row0;
    };
    using Params = Arguments;
    Params p;
    TensorStorage& slm;

    template <class ProblemShape>
    static constexpr Params to_underlying_arguments(ProblemShape const&, Arguments const& args, void*) {
      return args;
    }

    CUTLASS_DEVICE Base(Params const& params, TensorStorage& storage) : p(params), slm(storage) {}

    template <class ProblemShapeMNKL, class SubgroupShape, class TileCoord, class Accumulator, class TiledMma>
    CUTLASS_DEVICE void operator()(
        ProblemShapeMNKL const& problem_shape_MNKL,
        SubgroupShape,
        TileCoord const& tile_coord,
        Accumulator const& accum,
        TiledMma const&,
        int thread_idx) {
      using namespace cute;
      const int lane = thread_idx % 16;
      const int sg_k = thread_idx / 16;
      const int m0 = get<0>(tile_coord) * BM;
      const int rows = sycl::min(BM, static_cast<int>(get<0>(problem_shape_MNKL)) - m0);
      const int n_end = static_cast<int>(get<1>(problem_shape_MNKL));
      const int n0 = get<1>(tile_coord) * SG_N;
      // Rows are indexed at runtime here, so they are read from perm, not kept in an array.
      auto store = [&](int r, int n, float v) {
        if (r < rows && n < n_end) {
          const int lr = p.row0 + m0 + r;
          p.out[(p.perm ? p.perm[lr] : static_cast<int64_t>(lr)) * p.N + n] = Element(v).raw();
        }
      };
      if constexpr (SGK == 1) {
        CUTE_UNROLL
        for (int idx = 0; idx < size(accum); ++idx) {
          store(idx % BM, n0 + (idx / BM) * 16 + lane, accum(idx));
        }
      } else {
        float* mine = slm.partials + sg_k * TILE;
        CUTE_UNROLL
        for (int i = 0; i < size(accum); ++i) {
          mine[i * 16 + lane] = accum(i);
        }
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_work_group<3>());
        for (int e = thread_idx; e < TILE; e += NumThreads) {
          float s = 0.f;
          for (int g = 0; g < SGK; ++g) {
            s += slm.partials[g * TILE + e];
          }
          const int idx = e / 16;
          store(idx % BM, n0 + (idx / BM) * 16 + e % 16, s);
        }
        // No trailing barrier: the segment scheduler gives every work-group one tile, so the SLM
        // is never reused (xe_segment_tile_scheduler.hpp, fetch_next_work()).
      }
    }
  };

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const&, Arguments const& args, void*) {
    return args;
  }

  template <class ProblemShape>
  static bool can_implement(ProblemShape const&, Arguments const&) {
    return true;
  }

  CUTLASS_DEVICE static typename Base::Arguments to_base_arguments(Arguments const& args, int g) {
    auto const& gr = args.groups;
    return {args.out, args.perm, gr.N, gr.row0(g)};
  }
};

}  // namespace cutlass::lora::collective
