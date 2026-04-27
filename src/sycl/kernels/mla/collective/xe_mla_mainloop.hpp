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
    \brief XPU MLA Mainloop
*/

#pragma once

#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/algorithm/subgroup_algorithms.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
namespace cutlass::flash_attention {

template <int Stages>
class XeDefault {};

};  // namespace cutlass::flash_attention

namespace cutlass::flash_attention::collective {
using namespace cute;
////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    class DispatchPolicy_,
    bool CausalMask_,
    class TiledMMAQK_,         // Tiling for Q*K GEMM
    class TiledMMAPV_,         // Tiling for P*V GEMM
    int VTiles_,               // # of tiles in V dimension
    class TensorQ_,            // Global Q tensors
    class TensorK_,            // Global KV tensor
    class TensorV_,            // Global V tensor
    class TiledCopyQ_ = void,  // Optional TiledCopy for loading Q
    class TiledCopyK_ = void,  // Optional TiledCopy for loading KV
    class TiledCopyV_ = void>  // Optional TiledCopy for loading V
struct XeMlaMainloop {
  static_assert(cutlass::detail::dependent_false<DispatchPolicy_>, "Could not find a mainloop specialization.");
};

/////////////////////////////////////////////////////////////////////////////////////////////////
template <
    int Stages,
    bool CausalMask_,
    class TiledMMAQK_,
    class TiledMMAPV_,
    int VTiles_,
    class TensorQ_,
    class TensorK_,
    class TensorV_,
    class TiledCopyQ_,
    class TiledCopyK_,
    class TiledCopyV_>
struct XeMlaMainloop<
    XeDefault<Stages>,
    CausalMask_,
    TiledMMAQK_,
    TiledMMAPV_,
    VTiles_,
    TensorQ_,
    TensorK_,
    TensorV_,
    TiledCopyQ_,
    TiledCopyK_,
    TiledCopyV_> {
  //
  // Type Aliases
  //
  using TiledMMAQK = TiledMMAQK_;
  using TiledMMAPV = TiledMMAPV_;
  using TileShapeQK = decltype(TiledMMAQK{}.tile_mnk());
  using TileShapePV = decltype(TiledMMAPV{}.tile_mnk());

  static constexpr auto QK_BLK_M = get<0>(TileShapeQK{});
  static constexpr auto QK_BLK_N = get<1>(TileShapeQK{});
  static constexpr auto QK_BLK_K = get<2>(TileShapeQK{});

  static constexpr int VTiles = VTiles_;
  using SubgroupLayoutQK = decltype(TiledMMAQK{}.get_atom_layout_mnk());
  using SGPerWG = decltype(product(take<1, 4>(shape(typename TiledMMAQK::ThrLayoutVMNK{}))));

  // Number of subgroups for cross-SG reduction
  static constexpr int NumSubgroups = SGPerWG::value;

  using TensorQ = TensorQ_;
  using TensorK = TensorK_;  // KV tensor used for both K and V
  using TensorV = TensorV_;

  using ElementQ = typename TensorQ_::value_type;

  using TensorQ2D = decltype(TensorQ_{}(append<rank_v<TensorQ_>>(make_coord(_, _), 0)));

  using TensorK2D = decltype(TensorK_{}(append<rank_v<TensorK_>>(make_coord(_, _), 0)));
  using TensorK3D = decltype(TensorK_{}(append<rank_v<TensorK_>>(make_coord(_, _, _), 0)));

  using TensorV2D = decltype(TensorV_{}(append<rank_v<TensorV_>>(make_coord(_, _), 0)));
  using TensorV3D = decltype(TensorV_{}(append<rank_v<TensorV_>>(make_coord(_, _, _), 0)));

  using TiledCopyQ =
      conditional_t<is_void_v<TiledCopyQ_>, decltype(make_block_2d_copy_A(TiledMMAQK{}, TensorQ2D{})), TiledCopyQ_>;
  using TiledCopyK =
      conditional_t<is_void_v<TiledCopyK_>, decltype(make_block_2d_copy_B(TiledMMAQK{}, TensorK2D{})), TiledCopyK_>;
  using TiledCopyV =
      conditional_t<is_void_v<TiledCopyV_>, decltype(make_block_2d_copy_B(TiledMMAPV{}, TensorV2D{})), TiledCopyV_>;

  //
  // Accumulator types
  //
  // FragS:    accumulator for Q*K MMA (score matrix)
  // FragA:    accumulator for P*V MMAs (output)
  //           Note: v mode may be split into multiple pieces to reduce register pressure.
  // Frag*Row types are reductions of the corresponding Frag* types over rows.
  //
  template <typename TiledMMA>
  using FragC = decltype(TiledMMA{}.get_slice(0).partition_sg_fragment_C(
      make_identity_tensor(select<0, 1>(TiledMMA{}.tile_mnk()))));

  using FragS = FragC<TiledMMAQK>;
  using FragSRow = decltype(reduce<1>(FragS{}, sycl::plus<void>{}));
  using ElementS = typename TiledMMAQK::ValTypeD;

  using SingleFragA = FragC<TiledMMAPV>;                       // (atom val,q',v')
  using FragA = expand_sg_fragment_t<SingleFragA, 1, VTiles>;  // (atom val,q',v',VV)
  using FragARow = decltype(reduce<1>(FragA{}, sycl::plus<void>{}));
  using ElementA = typename TiledMMAPV::ValTypeD;

  static constexpr bool CausalMask = CausalMask_;

  //
  // Arguments
  //
  // User-facing arguments
  struct Arguments {
    ElementS scale;
    int const* ptr_page_table = nullptr;
    int page_size = 0;
    int total_page = 0;
    int num_pages_per_batch = 0;
  };

  //
  // Params
  //
  // Kernel-facing parameters
  using Params = Arguments;

  //
  // Shared memory storage for cross-subgroup softmax reduction
  //
  struct SharedStorage {};

  Params params;
  SharedStorage& shared;

  //
  // Methods
  //
  XeMlaMainloop(Params const& params_, SharedStorage& shared_) : params(params_), shared(shared_) {}

  static constexpr Params to_underlying_arguments(Arguments const& args, void* /* workspace */) {
    // exp(x) = exp2(x * log2(e))
    constexpr double kLog2e = 1.4426950408889634074;  // log_2(e)
    ElementS val = args.scale * static_cast<ElementS>(kLog2e);
    return Params{val, args.ptr_page_table, args.page_size, args.total_page, args.num_pages_per_batch};
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) {
    return true;
  }

  // Get physical block index and intra-page tile index for paged KV cache
  CUTLASS_DEVICE
  cute::tuple<int, int> get_physical_k_tile(int K, int seq_len_kv_cache, int batch_coord) {
    int kv_start_coord = K * get<1>(TileShapeQK{});
    int logical_page_idx = kv_start_coord / params.page_size;
    int intra_page_tile_idx = (kv_start_coord % params.page_size) / get<1>(TileShapeQK{});
    // Use batch_coord to index into the correct row of the page table
    int page_table_idx = batch_coord * params.num_pages_per_batch + logical_page_idx;
    int physical_block_idx = params.ptr_page_table[page_table_idx];
    return cute::make_tuple(physical_block_idx, intra_page_tile_idx);
  }

  //
  // MLA Mainloop Operator
  //
  // -------------------------------------------
  // Score = Q_nope @ KV^T + Q_pe @ K_pe^T
  // P = softmax(Score * scale)
  // O = P @ V
  //
  template <typename QVCoord>
  CUTLASS_DEVICE void operator()(
      TensorQ2D const& Qnope_2D,  // Q_nope (q, d_latent)
      TensorQ2D const& Qpe_2D,    // Q_pe (q, d_rope)
      TensorK3D const& K_3D,      // K (k, d_latent, num_blocks) - paged cache
      TensorK3D const& Kpe_3D,    // K_pe (k, d_rope, num_blocks) - paged cache
      TensorV3D const& V_3D,      // V (d_latent, k, num_blocks) - paged cache
      FragA& tArA,                // Output accumulator (q,v)
      FragARow& tA_max,           // Softmax row-wise max accumulator
      FragARow& tA_sum,           // Softmax row-wise sum accumulator
      QVCoord blk_qv,             // WG tile indices: (Q,V)
      int blk_k0,                 // K block range start
      int blk_k1,                 // K block range end
      int total_blk,              // Total # of K blocks
      int thr_id,
      int seq_len_kv,
      int batch_coord) {  // Batch index for page table lookup
    using namespace sycl::ext::oneapi::this_work_item;

    // Short dimension names:
    //    q = sequence len dimension for Q
    //    k = sequence len dimension for K
    //    d = head size dimension for K/Q
    //    v = head size dimension for V (subset of d, first v_head_dim columns of KV)
    //   VV = MMA tile indices for V
    // Capital letters (Q, K, ...) refer to WG block indices.

    /* Create proxy coordinate tensors for Q/K/P */
    Tensor cQnope = make_identity_tensor(Qnope_2D.shape());       // (q,d_latent)
    Tensor cQpe = make_identity_tensor(Qpe_2D.shape());           // (q,d_rope)
    Tensor cK = make_identity_tensor(K_3D.shape());               // (k,d_latent,num_blocks)
    Tensor cKpe = make_identity_tensor(Kpe_3D.shape());           // (k,d_rope,num_blocks)
    Tensor cV = make_identity_tensor(V_3D.shape());               // (d_latent,k, num_blocks)
    Tensor cP = make_identity_tensor(take<0, 2>(TileShapeQK{}));  // (q,k)

    /* Partition global tensors into workgroup tiles */
    Tensor gQnope = local_tile(cQnope, TileShapeQK{}, append(blk_qv, _), Step<_1, X, _1>{});  // (q,d,D)
    Tensor gQpe = local_tile(cQpe, TileShapeQK{}, append(blk_qv, _), Step<_1, X, _1>{});      // (q,d,D)
    Tensor gK = local_tile(cK, TileShapeQK{}, make_coord(_, _, _), Step<X, _1, _1>{});        // (k,d,K,D,num_blocks)
    Tensor gKpe = local_tile(cKpe, TileShapeQK{}, make_coord(_, _, _), Step<X, _1, _1>{});    // (k,d,K,D,num_blocks)

    auto tile_shape_v = make_shape(get<1>(TileShapePV{}) * C<VTiles>{}, get<2>(TileShapePV{}));  // (_512, _16)
    Tensor gV = local_tile(cV, tile_shape_v, make_coord(get<1>(blk_qv), _, _));                  // (v,k,K,num_blocks)
    Tensor gV_split = local_tile(gV, TileShapePV{}, make_coord(_, _, 0), Step<X, _1, _1>{});     // (v,k,VV,K)

    /* Create global -> register copies */
    TiledCopyQ copy_qnope{Qnope_2D};
    TiledCopyQ copy_qpe{Qpe_2D};
    TiledCopyK copy_k{K_3D(_, _, 0)};      // Slice 3D to 2D for copy traits construction
    TiledCopyK copy_kpe{Kpe_3D(_, _, 0)};  // Slice 3D to 2D for copy traits construction
    TiledCopyV copy_v{V_3D(_, _, 0)};      // Reuse KV tensor for V copy, slice 3D to 2D

    /* Create MMAs */
    TiledMMAQK mma_qk{};
    TiledMMAPV mma_pv{};

    /* Slice TiledCopy/TiledMMA operations down to work-item level */
    auto thr_copy_qnope = copy_qnope.get_slice(thr_id);
    auto thr_copy_qpe = copy_qpe.get_slice(thr_id);
    auto thr_copy_k = copy_k.get_slice(thr_id);
    auto thr_copy_kpe = copy_kpe.get_slice(thr_id);
    auto thr_copy_v = copy_v.get_slice(thr_id);
    auto thr_mma_qk = mma_qk.get_slice(thr_id);
    auto thr_mma_pv = mma_pv.get_slice(thr_id);

    /* Partition coordinate tensors for copy */
    auto tQnopegQ = thr_copy_qnope.partition_S(gQnope);  // (atom_val,q',d',D)
    auto tQpegQ = thr_copy_qpe.partition_S(gQpe);        // (atom_val,q',d',D)
    auto tKgK = thr_copy_k.partition_S(gK);              // (atom_val,k',d',K,D,num_blocks)
    auto tKpegK = thr_copy_kpe.partition_S(gKpe);        // (atom_val,k',d',K,D,num_blocks)
    auto tVgV = thr_copy_v.partition_S(gV_split);        // (atom_val,v',k',VV,K,num_blocks) - reuses KV

    /* Create register fragments for MMA and copies */
    // Q_nope fragments
    auto tQnoperQ = thr_copy_qnope.partition_sg_fragment_D(gQnope(_, _, 0));
    auto tSrQnope = thr_mma_qk.partition_sg_fragment_A(gQnope(_, _, 0));

    // Q_pe fragments
    auto tQperQ = thr_copy_qpe.partition_sg_fragment_D(gQpe(_, _, 0));
    auto tSrQpe = thr_mma_qk.partition_sg_fragment_A(gQpe(_, _, 0));

    // KV fragments (used for both Q*K^T and P*V)
    auto tKrK = thr_copy_k.partition_sg_fragment_D(gK(_, _, 0, 0, 0));
    auto tSrK = thr_mma_qk.partition_sg_fragment_B(gK(_, _, 0, 0, 0));

    // K_pe fragments
    auto tKperK = thr_copy_kpe.partition_sg_fragment_D(gKpe(_, _, 0, 0, 0));
    auto tSrKpe = thr_mma_qk.partition_sg_fragment_B(gKpe(_, _, 0, 0, 0));

    // Score/P fragments
    auto tSrS = thr_mma_qk.partition_sg_fragment_C(cP);
    auto tArP = thr_mma_pv.partition_sg_fragment_A(cP);

    // V fragments (reuses KV data - first v_head_dim columns)
    auto tVrV = thr_copy_v.partition_sg_fragment_D(gV_split(_, _, 0, 0, 0));
    auto tArV = thr_mma_pv.partition_sg_fragment_B(gV_split(_, _, 0, 0, 0));

    /* Create TiledCopy objects for prefetches */
    auto prefetch_qnope = make_block_2d_prefetch(copy_qnope);
    auto prefetch_qpe = make_block_2d_prefetch(copy_qpe);
    auto prefetch_k = make_block_2d_prefetch(copy_k);
    auto prefetch_kpe = make_block_2d_prefetch(copy_kpe);
    // // V prefetch reuses KV tensor (no separate prefetch needed since KV is already prefetched)
    auto prefetch_v = make_block_2d_prefetch<SGPerWG::value>(tile_shape_v, V_3D);

    /* Partition global tensors for prefetch */
    auto pQnopegQ = prefetch_qnope.get_slice(thr_id).partition_S(gQnope);
    auto pQpegQ = prefetch_qpe.get_slice(thr_id).partition_S(gQpe);
    auto pKgK = prefetch_k.get_slice(thr_id).partition_S(gK);
    auto pKpegK = prefetch_kpe.get_slice(thr_id).partition_S(gKpe);
    auto pVgV = prefetch_v.get_slice(thr_id).partition_S(gV);  // V uses same KV data

    // ------
    // Kernel
    // ------

    /* Check if we have remainder tiles */
    bool check_remainder_k = (seq_len_kv % get<1>(TileShapeQK{}) != 0);

    auto [physical_block_idx, intra_page_tile_idx] = get_physical_k_tile(blk_k0, seq_len_kv, batch_coord);

    /* Initialization steps for first block: Q prefetch, O init */
    if (blk_k0 == 0) {
      for (int D = 0; D < size<3>(pQnopegQ); D++) {
        prefetch(prefetch_qnope, pQnopegQ(_, _, _, D));
        prefetch(prefetch_qpe, pQpegQ(_, _, _, D));
      }
      for (int D = 0; D < size<4>(pKgK); D++) {
        prefetch(prefetch_k, pKgK(_, _, _, intra_page_tile_idx, D, physical_block_idx));
        prefetch(prefetch_kpe, pKpegK(_, _, _, intra_page_tile_idx, D, physical_block_idx));
      }
      prefetch(prefetch_v, pVgV(_, _, 0, intra_page_tile_idx, physical_block_idx));

      // Initialize accumulators
      clear(tArA);
      fill(tA_max, cutlass::platform::numeric_limits<ElementA>::lowest());
      clear(tA_sum);
    }

    /* Main loop, blocked in k. */
    for (int K = blk_k0; K < blk_k1; K++) {
      /* Prefetch next K tile and save its physical block info for next iteration */
      int next_physical_block_idx = 0, next_intra_page_tile_idx = 0;
      int K_prefetch = K + 1;
      if (K_prefetch < blk_k1) {
        auto [pf_block_idx, pf_tile_idx] = get_physical_k_tile(K_prefetch, seq_len_kv, batch_coord);
        next_physical_block_idx = pf_block_idx;
        next_intra_page_tile_idx = pf_tile_idx;
        for (int D = 0; D < size<4>(pKgK); D++) {
          prefetch(prefetch_k, pKgK(_, _, _, pf_tile_idx, D, pf_block_idx));
          prefetch(prefetch_kpe, pKpegK(_, _, _, pf_tile_idx, D, pf_block_idx));
        }
        prefetch(prefetch_v, pVgV(_, _, 0, pf_tile_idx, pf_block_idx));
      }

      // TODO: need to remove copy_kv1, copy_kpe1, copy_v1, as this is unoptimized approach
      // as for every block K we are creating new TiledCopy objects which is expensive.
      // Instead we should create these TiledCopy objects once outside the loop and reuse them by
      // just changing the block index they point to. but currently TiledCopy objects are immutable
      // and we cannot change the underlying pointer after creation. We should optimize this by
      // making TiledCopy objects mutable or by creating a new type of copy object that allows
      // changing the underlying pointer without creating a new object.
      TiledCopyK copy_kv1{K_3D(_, _, physical_block_idx)};
      TiledCopyK copy_kpe1{Kpe_3D(_, _, physical_block_idx)};
      TiledCopyV copy_v1{V_3D(_, _, physical_block_idx)};

      /* =================================================================
       * GEMM 1: S = Q_nope @ KV_c^T + Q_pe @ K_pe^T
       * MLA Score Computation - Two-part accumulation
       * ================================================================= */
      clear(tSrS);

      // Part 1: Accumulate Q_nope @ KV_c^T
      CUTLASS_PRAGMA_UNROLL
      for (int D = 0; D < size<4>(tKgK); D++) {
        copy(copy_qnope, tQnopegQ(_, _, _, D), tQnoperQ);
        copy(copy_kv1, tKgK(_, _, _, intra_page_tile_idx, D, physical_block_idx), tKrK);
        reorder(tQnoperQ, tSrQnope);
        reorder(tKrK, tSrK);
        cute::gemm(mma_qk, tSrQnope, tSrK, tSrS);
      }

      // Part 2: Accumulate Q_pe @ K_pe^T
      CUTLASS_PRAGMA_UNROLL
      for (int D = 0; D < size<4>(tKpegK); D++) {
        copy(copy_qpe, tQpegQ(_, _, _, D), tQperQ);
        copy(copy_kpe1, tKpegK(_, _, _, intra_page_tile_idx, D, physical_block_idx), tKperK);
        reorder(tQperQ, tSrQpe);
        reorder(tKperK, tSrKpe);
        cute::gemm(mma_qk, tSrQpe, tSrKpe, tSrS);
      }

      /* PagedKV masking - mask out invalid positions */
      if (check_remainder_k && K == total_blk - 1) {
        FragSRow k_rem_mask;
        int k_intra_page = get<0>(tKgK(0, 0, 0, intra_page_tile_idx, 0, physical_block_idx)) + K * params.page_size;
        int k = k_intra_page + get_sub_group().get_local_id()[0];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < k_rem_mask.size(); i++, k += intel::sg_size) {
          k_rem_mask(i) = (k < seq_len_kv) ? ElementS(INFINITY) : ElementS(-INFINITY);
        }

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); i++) {
          tSrS(i) = sycl::fmin(tSrS(i), broadcast<1>(k_rem_mask, tSrS, i));
        }
      }

      /* =================================================================
       * Apply softmax and scaling (online softmax algorithm)
       * ================================================================= */
      softmax(K == blk_k0, tSrS, tA_max, tA_sum, tArA);
      reorder(tSrS, tArP);
      /* =================================================================
       * GEMM 2: O += P @ V
       * ================================================================= */
      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        copy(copy_v1, tVgV(_, _, _, VV, intra_page_tile_idx, physical_block_idx), tVrV);
        reorder(tVrV, tArV);
        cute::gemm(mma_pv, tArP, tArV, tArA(_, _, _, VV));
      }

      /* use prefetched block info for next iteration's computation */
      physical_block_idx = next_physical_block_idx;
      intra_page_tile_idx = next_intra_page_tile_idx;
    }
  }

  //
  // Online Softmax Algorithm for Each Tile:
  //   1. Apply scale: tS_scaled = tS * scale
  //   2. Compute current tile max: m_curr = max(tS_scaled)
  //   3. Compute new global max: m_new = max(m_prev, m_curr)
  //   4. Compute correction factor: correction = exp2(m_prev - m_new)
  //   5. Compute P_tile (unnormalized): P_tile = exp2(tS_scaled - m_new)
  //   6. Compute sum for current tile: l_curr = sum(P_tile)
  //   7. Update running sum: l_new = l_prev * correction + l_curr
  //   8. Update output accumulator: O_new = O_prev * correction
  //   9. Update state: m_prev <- m_new, l_prev <- l_new
  //   10. Final output after all tiles:  O_final = O_new / l_new [epilogue step, not shown here]
  CUTLASS_DEVICE
  void softmax(
      bool first_block,  // First softmax block?
      FragS& tS,         // Softmax src/dst block
      FragSRow& tS_max,  // Softmax row-wise max accumulator
      FragSRow& tS_sum,  // Softmax row-wise sum accumulator
      FragA& tA) {       // O accumulator (for rescaling)

    /* Compute row-wise maxima for this block */
    auto tS_bmax = reduce<1>(tS, sycl::maximum{});

    /* Update (scaled) maxima */
    auto tS_prev_max = tS_max;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_max.size(); i++) {
      tS_max(i) = sycl::max(tS_max(i), params.scale * tS_bmax(i));
    }

    /* Scale S and subtract maxima, then exponentiate */
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS.size(); i++)
      tS(i) = sycl::native::exp2(params.scale * tS(i) - broadcast<0>(tS_max, tS, i));

    /* Rescale existing S sums and O accumulator */
    if (!first_block) {
      FragSRow rescale;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tS_max.size(); i++) {
        rescale(i) = sycl::native::exp2(tS_prev_max(i) - tS_max(i));
        tS_sum(i) *= rescale(i);
      }

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tA.size(); i++)
        tA(i) *= broadcast<0>(rescale, tA, i);
    }

    /* Update sums */
    auto tS_bsum = reduce<1>(tS, sycl::plus<void>{});
    for (int i = 0; i < tS_sum.size(); i++)
      tS_sum(i) += tS_bsum(i);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
template <typename SGLayoutQK>
CUTLASS_HOST_DEVICE constexpr auto get_sg_layout_pv(SGLayoutQK const&) {
  return make_layout(get<0>(SGLayoutQK{}), Layout<_1, _0>{}, get<1>(SGLayoutQK{}));
}

}  // namespace cutlass::flash_attention::collective
