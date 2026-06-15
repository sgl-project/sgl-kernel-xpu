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

  // Global LOAD types (the gmem tensor value_types). ElementA is bf16 (A is
  // loaded narrow and converted in-register); ElementB is tf32 (fp32 bits
  // reinterpreted). These drive the block-2D copies, NOT the MMA math.
  using ElementA = typename TensorA_::value_type;
  using ElementB = typename TensorB_::value_type;

  // MMA input type (tf32). The square-sum reuses the MMA fragments (tCrA etc.),
  // which hold this type after reorder() converts the bf16 A load. The square
  // cast below MUST target this, not ElementA (bf16) — otherwise A^2 would be
  // re-truncated to bf16 and lose the tf32 precision the MMA path provides.
  using ElementMMA = typename TiledMMA::ValTypeA;

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
  // C output is fp32 == the accumulator type. The mhc_pre pipeline consumes
  // gemm_out_mul as FP32, and keeping C == accumulator lets the epilogue store
  // the MMA accumulator straight through the block-2D copy with no convert
  // (the convert was only needed when C was a narrower 16-bit type).
  using ElementC = ElementAccum;                         // Output type (fp32)

  // Square sum is computed as a SECOND GEMM: (A elementwise-squared) @ ones[K,N].
  // Every column of that (M,N) product equals sqrsum[row], so the accumulator is
  // a full C-fragment (identical type/layout to FragGemm) and the epilogue reuses
  // the proven C-write coordinate mapping to extract one column per row.
  using FragSqrSum = decltype(partition_fragment_C(TiledMMA{}, select<0, 1>(TileShape{})));
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
  // The K loop runs over the tile range [k_tile_begin, k_tile_end) so the kernel
  // can partition K across workgroups (split-K). k_tile_begin == 0 and
  // k_tile_end == ceil(K/BLK_K) is the full single-pass loop (split_k == 1).
  // An empty range (begin == end) clears the accumulators and returns a zero
  // partial, which the host sums in harmlessly.
  template <typename TensorA2D_runtime, typename TensorB2D_runtime, typename MNCoord>
  CUTLASS_DEVICE void operator()(
      TensorA2D_runtime const& A_2D,      // A (m, k) - runtime shaped
      TensorB2D_runtime const& B_2D,      // B (k, n) - runtime shaped
      FragGemm& tC,               // GEMM accumulator output (m,n)
      FragSqrSum& tSqrSum,        // Square sum accumulator output (m)
      MNCoord blk_mn,             // WG tile indices: (M,N)
      int thr_id,
      int k_tile_begin,           // first K-tile this split computes
      int k_tile_end) {           // one past last K-tile this split computes
    using namespace sycl::ext::oneapi::this_work_item;

    TiledMMA mma{};
    auto wg_tile = mma.tile_mnk();
    int wg_m = int(get<0>(blk_mn));
    int wg_n = int(get<1>(blk_mn));

    /* Proxy coordinate tensors. A is (M,K); B is the (N,K) view. */
    Tensor cA = make_identity_tensor(A_2D.shape());  // (M,K)
    Tensor cB = make_identity_tensor(B_2D.shape());  // (N,K)

    /* Workgroup tiles — transcribed from the canonical Xe GEMM tutorial:
     *   gA: (BLK_M,BLK_K,k) via select<0,2>(wg_tile) at (wg_m,_)
     *   gB: (BLK_N,BLK_K,k) via select<1,2>(wg_tile) at (wg_n,_)
     * Using the 2-mode tilers (not the full 3-mode Step form) is what lets the
     * MMA fragments span the full WG tile across all subgroups. */
    Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(wg_m, _));  // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(wg_n, _));  // (BLK_N,BLK_K,k)

    /* Block-2D copies */
    auto copy_a = make_block_2d_copy_A(mma, A_2D);
    auto copy_b = make_block_2d_copy_B(mma, B_2D);

    /* Slice to work-item level */
    auto thr_mma = mma.get_slice(thr_id);
    auto thr_copy_a = copy_a.get_slice(thr_id);
    auto thr_copy_b = copy_b.get_slice(thr_id);

    /* MMA register fragments */
    auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
    auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

    /* Copy register fragments */
    auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
    auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

    /* Partition global proxies for copies */
    Tensor tAgA = thr_copy_a.partition_S(gA);
    Tensor tBgB = thr_copy_b.partition_S(gB);

    /* Square-sum-as-GEMM operands (same MMA layouts as tCrA/tCrB) */
    auto tCrAsq = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
    auto tCrBones = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

    /* Prefetch instances */
    auto prefetch_a = make_block_2d_prefetch(copy_a);
    auto prefetch_b = make_block_2d_prefetch(copy_b);
    auto pAgA = prefetch_a.get_slice(thr_id).partition_S(gA);
    auto pBgB = prefetch_b.get_slice(thr_id).partition_S(gB);

    // Prefetch depth = the dispatch policy's pipeline Stages (repo convention,
    // matches the bf16 MoE mainloop's `prefetch_dist = Stages`). K-tiles kept in
    // flight ahead of compute: trades L1 pressure vs. latency hiding. Set at the
    // config site via XeDefault<Stages> (see gemm_sqrsum_types.hpp).
    constexpr int prefetch_dist = Stages;
    constexpr int barrier_scope = 2;
    // This split owns global K-tiles [k_tile_begin, k_tile_end). Prefetch and
    // load index pAgA/tAgA with the GLOBAL tile index (they span all
    // k_tiles_total tiles), so a split simply offsets its window into them.
    int k_tile_prefetch = k_tile_begin;

    /* Clear accumulators */
    clear(tC);
    clear(tSqrSum);

    /* Warm up prefetch to L1: prefetch this split's first `prefetch_dist`
     * tiles. Hints that run off the end of this split's window fall into the
     * next split's (still-valid) tiles, or are clamped OOB by the block-2D
     * prefetch past k_tiles_total — both harmless. */
    CUTE_UNROLL
    for (int p = 0; p < prefetch_dist; p++, k_tile_prefetch++) {
      prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
      prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
    }

    /* Main loop over this split's K-tile sub-range */
    for (int k_tile = k_tile_begin; k_tile < k_tile_end; k_tile++, k_tile_prefetch++) {
      barrier_arrive(barrier_scope);

      /* Load A/B to registers */
      copy(copy_a, tAgA(_, _, _, k_tile), tArA);
      copy(copy_b, tBgB(_, _, _, k_tile), tBrB);

      /* Prefetch ahead */
      prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
      prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));

      /* Shuffle copy fragments into MMA fragments */
      reorder(tArA, tCrA);
      reorder(tBrB, tCrB);

      /* C += A * B */
      cute::gemm(mma, tCrA, tCrB, tC);

      /* Square sum as a second GEMM: tSqrSum += (A^2) @ ones.
       * Reuses the already-loaded A values (tCrA); every column of tSqrSum then
       * holds sum_k A[m,k]^2 for row m. Write through the SubgroupTensors'
       * operator() (their .tensor() view is const). */
      // Square in the MMA type (tf32), reusing the already-loaded+converted A
      // fragment tCrA. tCrAsq/tCrBones are MMA fragments (ElementMMA == tf32),
      // so cast to ElementMMA — casting to ElementA (bf16) would re-truncate.
      // Static trip counts: size() of a register fragment with a static layout
      // is a compile-time Int<N>. Pull the value from the TYPE (no tensor
      // construction, no runtime int cast) so the unroll fully eliminates the
      // loop (these are tiny, ~8-16 elems/thread).
      constexpr int n_a = decltype(size(tCrAsq.tensor()))::value;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < n_a; i++) {
        ElementMMA a_in = tCrA(i);
        tCrAsq(i) = static_cast<ElementMMA>(a_in * a_in);
      }
      constexpr int n_b = decltype(size(tCrBones.tensor()))::value;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < n_b; i++) {
        tCrBones(i) = static_cast<ElementMMA>(1);
      }
      cute::gemm(mma, tCrAsq, tCrBones, tSqrSum);

      barrier_wait(barrier_scope);
    }
    // tC and tSqrSum hold the final accumulators; the kernel epilogue stores them.
  }
};

}  // namespace cutlass::gemm_sqrsum::collective
