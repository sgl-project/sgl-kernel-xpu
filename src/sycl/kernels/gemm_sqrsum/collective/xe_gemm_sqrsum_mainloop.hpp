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
/*! \file
    \brief XPU GEMM with Row-wise Square Sum Mainloop

    Computes:
      C = A @ B
      sqrsum[i] = sum(A[i,:]^2)  for each row i
*/

#pragma once

#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/algorithm/subgroup_algorithms.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"

namespace cutlass::gemm_sqrsum {

template <int Stages>
class XeDefault {};

}  // namespace cutlass::gemm_sqrsum

namespace cutlass::gemm_sqrsum::collective {
using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    class DispatchPolicy_,
    class TiledMMA_,            // Tiling for A*B GEMM
    class TensorA_,             // Global A tensor
    class TensorB_,             // Global B tensor
    class TiledCopyA_ = void,   // Optional TiledCopy for loading A
    class TiledCopyB_ = void>   // Optional TiledCopy for loading B
struct XeGemmSqrSumMainloop {
  static_assert(cutlass::detail::dependent_false<DispatchPolicy_>, "Could not find a mainloop specialization.");
};

/////////////////////////////////////////////////////////////////////////////////////////////////
template <
    int Stages,
    class TiledMMA_,
    class TensorA_,
    class TensorB_,
    class TiledCopyA_,
    class TiledCopyB_>
struct XeGemmSqrSumMainloop<
    XeDefault<Stages>,
    TiledMMA_,
    TensorA_,
    TensorB_,
    TiledCopyA_,
    TiledCopyB_> {
  //
  // Type Aliases
  //
  using TiledMMA = TiledMMA_;
  using TileShape = decltype(TiledMMA{}.tile_mnk());

  static constexpr auto BLK_M = get<0>(TileShape{});
  static constexpr auto BLK_N = get<1>(TileShape{});
  static constexpr auto BLK_K = get<2>(TileShape{});

  using SubgroupLayout = decltype(TiledMMA{}.get_atom_layout_mnk());
  using SGPerWG = decltype(product(take<1, 4>(shape(typename TiledMMA::ThrLayoutVMNK{}))));

  static constexpr int NumSubgroups = SGPerWG::value;

  using TensorA = TensorA_;
  using TensorB = TensorB_;

  using ElementA = typename TensorA_::value_type;
  using ElementB = typename TensorB_::value_type;

  using TensorA2D = decltype(TensorA_{}(append<rank_v<TensorA_>>(make_coord(_, _), 0)));
  using TensorB2D = decltype(TensorB_{}(append<rank_v<TensorB_>>(make_coord(_, _), 0)));

  using TiledCopyA =
      conditional_t<is_void_v<TiledCopyA_>, decltype(make_block_2d_copy_A(TiledMMA{}, TensorA2D{})), TiledCopyA_>;
  using TiledCopyB =
      conditional_t<is_void_v<TiledCopyB_>, decltype(make_block_2d_copy_B(TiledMMA{}, TensorB2D{})), TiledCopyB_>;

  //
  // Accumulator types
  //
  // GEMM accumulator: the plain MMA C fragment (FrgTypeC == float), exactly as
  // in the canonical Xe GEMM tutorial. This is the valid source for the Xe
  // block-2D store in the epilogue (a SubgroupTensor from partition_sg_fragment_C
  // would NOT be a valid copy source).
  using FragGemm = decltype(partition_fragment_C(TiledMMA{}, select<0, 1>(TileShape{})));
  using ElementAccum = typename TiledMMA::ValTypeD;     // Accumulator type (float)
  using ElementC = ElementA;                             // Output type (same as input)

  // Square sum accumulator: one accumulator per row in the BLK_M tile.
  // BLK_M is a CuTe Int<> instance, so its compile-time extent is decltype(BLK_M)::value.
  using FragSqrSum = cute::array<float, decltype(BLK_M)::value>;
  using ElementSqrSum = float;  // Square sum output type (always float for atomic support)

  //
  // Arguments
  //
  struct Arguments {};

  //
  // Params
  //
  using Params = Arguments;

  //
  // Shared memory storage
  //
  struct SharedStorage {};

  Params params;
  SharedStorage& shared;

  //
  // Methods
  //
  XeGemmSqrSumMainloop(Params const& params_, SharedStorage& shared_) : params(params_), shared(shared_) {}

  static constexpr Params to_underlying_arguments(Arguments const& args, void* /* workspace */) {
    return Params{};
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) {
    return true;
  }

  //
  // GEMM + Square Sum Mainloop Operator
  //
  // Computes:
  //   C = A @ B
  //   sqrsum[i] = sum(A[i,:]^2) for each row i
  //
  template <typename TensorA2D_runtime, typename TensorB2D_runtime, typename MNCoord>
  CUTLASS_DEVICE void operator()(
      TensorA2D_runtime const& A_2D,      // A (m, k) - runtime shaped
      TensorB2D_runtime const& B_2D,      // B (k, n) - runtime shaped
      FragGemm& tC,               // GEMM accumulator output (m,n)
      FragSqrSum& tSqrSum,        // Square sum accumulator output (m)
      MNCoord blk_mn,             // WG tile indices: (M,N)
      int thr_id) {
    using namespace sycl::ext::oneapi::this_work_item;

    /* Create proxy coordinate tensors */
    Tensor cA = make_identity_tensor(A_2D.shape());  // (m,k)
    Tensor cB = make_identity_tensor(B_2D.shape());  // (n,k)  [B is the (N,K) view]

    /* Partition global tensors into workgroup tiles.
     * Step<_1,X,_1> selects (M,K) of TileShape -> A tile (BLK_M,BLK_K,k).
     * Step<X,_1,_1> selects (N,K) of TileShape -> B tile (BLK_N,BLK_K,k),
     * matching cB's (n,k) layout and the canonical Xe GEMM convention. */
    Tensor gA = local_tile(cA, TileShape{}, append(blk_mn, _), Step<_1, X, _1>{});  // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(cB, TileShape{}, append(blk_mn, _), Step<X, _1, _1>{});  // (BLK_N,BLK_K,k)

    /* Create global -> register copies */
    TiledCopyA copy_a{A_2D};
    TiledCopyB copy_b{B_2D};

    /* Create MMA */
    TiledMMA mma{};

    /* Slice TiledCopy/TiledMMA operations down to work-item level */
    auto thr_copy_a = copy_a.get_slice(thr_id);
    auto thr_copy_b = copy_b.get_slice(thr_id);
    auto thr_mma = mma.get_slice(thr_id);

    /* Recover this work-item's N-subgroup coordinate (ThrN).
     * The A block-2D copy replicates A across the N subgroup dimension
     * (drop_n uses stride 0 on ThrN in make_block_2d_copy_A), so every value of
     * ThrN loads the SAME A data. To compute the square sum exactly once we let
     * only the ThrN == 0 column of subgroups accumulate; together they cover the
     * entire (BLK_M, BLK_K) A tile with no duplication. */
    auto thr_vmnk = mma.get_thr_layout_vmnk().get_flat_coord(thr_id);  // (ThrV,ThrM,ThrN,ThrK)
    int thr_n = int(get<2>(thr_vmnk));

    /* Partition coordinate tensors for copy */
    auto tAgA = thr_copy_a.partition_S(gA);  // (atom_val,m',k',K)
    auto tBgB = thr_copy_b.partition_S(gB);  // (atom_val,k',n',K)

    /* Create register fragments for MMA and copies */
    auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
    auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));

    auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));
    auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

    /* Create TiledCopy objects for prefetches */
    auto prefetch_a = make_block_2d_prefetch(copy_a);
    auto prefetch_b = make_block_2d_prefetch(copy_b);

    /* Partition global tensors for prefetch */
    auto pAgA = prefetch_a.get_slice(thr_id).partition_S(gA);
    auto pBgB = prefetch_b.get_slice(thr_id).partition_S(gB);

    // Initialize accumulators
    clear(tC);
    clear(tSqrSum);

    /* Prefetch first tile */
    prefetch(prefetch_a, pAgA(_, _, _, 0));
    prefetch(prefetch_b, pBgB(_, _, _, 0));

    /* Main loop over K dimension */
    int num_k_tiles = size<2>(gA);
    for (int K = 0; K < num_k_tiles; K++) {
      /* Prefetch next K tile */
      if (K + 1 < num_k_tiles) {
        prefetch(prefetch_a, pAgA(_, _, _, K + 1));
        prefetch(prefetch_b, pBgB(_, _, _, K + 1));
      }

      /* Load A and B tiles */
      copy(copy_a, tAgA(_, _, _, K), tArA);
      copy(copy_b, tBgB(_, _, _, K), tBrB);

      /* Reorder for MMA */
      reorder(tArA, tCrA);
      reorder(tBrB, tCrB);

      /* Compute C += A @ B; tC is the plain MMA accumulator (== tutorial's tCrC) */
      cute::gemm(mma, tCrA, tCrB, tC);

      /* Accumulate square sum using the SAME A values */
      // Reuse tArA (the A fragment already loaded for the GEMM) so A is read once.
      // tArA(i) and tAgA(_,_,_,K)(i) are co-indexed by linear index i: the copy()
      // above moves tAgA -> tArA element-for-element, so the i-th register value
      // corresponds to the i-th coordinate in the partitioned proxy tensor.
      //
      // cA is an identity tensor over the *global* A shape, so the coordinate's
      // M component is the GLOBAL row. Subtract the tile base (blk_m * BLK_M) to
      // index the per-tile accumulator array.
      // Only the ThrN == 0 subgroup column accumulates; other columns hold
      // duplicate copies of A and would over-count.
      {
        // tArA is a SubgroupTensor; .tensor() exposes this work-item's plain
        // register fragment so size()/element access resolve normally.
        auto a_frag = tArA.tensor();
        auto a_coord = tAgA(_, _, _, K);  // co-indexed coordinate proxy for this K-tile
        int m_base = int(get<0>(blk_mn)) * int(decltype(BLK_M)::value);
#ifdef GEMM_SQRSUM_DEBUG
        // One-shot instrumentation: dump real shapes + sample coords for a few
        // work-items in the first workgroup, first K-tile.
        if (K == 0 && get<0>(blk_mn) == 0 && get<1>(blk_mn) == 0 &&
            (thr_id == 0 || thr_id == 1 || thr_id == 16 || thr_id == 128)) {
          sycl::ext::oneapi::experimental::printf(
              "[DBG] thr=%d thr_n=%d a_frag_size=%d coords: i0=(%d,%d) i1=(%d,%d) iL=(%d,%d)\n",
              thr_id, thr_n, int(size(a_frag)),
              int(get<0>(a_coord(0))), int(get<1>(a_coord(0))),
              int(get<0>(a_coord(1))), int(get<1>(a_coord(1))),
              int(get<0>(a_coord(int(size(a_frag)) - 1))),
              int(get<1>(a_coord(int(size(a_frag)) - 1))));
        }
#endif
        // Only the ThrN == 0 subgroup column accumulates; other columns hold
        // duplicate copies of A and would over-count.
        if (thr_n == 0) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(a_frag); i++) {
            ElementA a_val = a_frag(i);
            ElementSqrSum sq_val = static_cast<ElementSqrSum>(a_val) * static_cast<ElementSqrSum>(a_val);

            // Global row of this element, then convert to local [0, BLK_M).
            int m_global = int(get<0>(a_coord(i)));
            int m_local = m_global - m_base;

            if (m_local >= 0 && m_local < int(decltype(BLK_M)::value)) {
              tSqrSum[m_local] += sq_val;
            }
          }
        }
      }
    }
    // tC already holds the final accumulator; the kernel epilogue stores it.
  }
};

}  // namespace cutlass::gemm_sqrsum::collective
