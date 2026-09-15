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
    \brief MLA Epilogue
*/
#pragma once

#include <sycl/sycl.hpp>

#include "cute/algorithm/subgroup_algorithms.hpp"
#include "cute/algorithm/tensor_algorithms.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/layout.hpp"
#include "cutlass/epilogue/collective/collective_epilogue.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "sycl/comm/copy_block_slm.hpp"

namespace cutlass::flash_attention::collective {
/////////////////////////////////////////////////////////////////////////////////////////////////
template <
    class CollectiveMainloop,
    class TileShapeO_,
    class TensorO_,
    class TiledCopyO_,  // Optional TiledCopy for storing O (void => default)
    class TensorLSE_>   // Global softmax-LSE tensor: (q, head, batch)
class XeMlaEpilogue {
 public:
  //
  // Type Aliases
  //
  using TiledMMAPV = typename CollectiveMainloop::TiledMMAPV;
  using TileShapePV = decltype(TiledMMAPV{}.tile_mnk());
  using TileShapeO = TileShapeO_;
  using SGPerWG = decltype(product(take<1, 4>(shape(typename TiledMMAPV::ThrLayoutVMNK{}))));

  using TensorO = TensorO_;
  using TensorO2D = decltype(TensorO_{}(append<rank_v<TensorO_>>(make_coord(_, _), 0)));
  using ElementO = typename TensorO_::value_type;

  // LSE is sliced down to the (q) mode for one (head, batch) pair, exactly like
  // O is sliced down to (q,v) -- see TensorO2D.
  using TensorLSE = TensorLSE_;
  using TensorLSE1D = decltype(TensorLSE_{}(make_coord(_, 0, 0)));
  using ElementLSE = typename TensorLSE_::value_type;

  using FragA = typename CollectiveMainloop::FragA;
  using FragARow = typename CollectiveMainloop::FragARow;
  using ElementA = typename FragA::value_type;
  using ElementAcc = float;  // Accumulator type for exp_sums/max_logits in split-KV path

  using ReduceK = decltype(size<3>(typename TiledMMAPV::ThrLayoutVMNK{}));

  static auto reduce_sg_v_helper() {
    constexpr auto v_total_sg = get<1>(SGTileShapeA{}) / intel::_SGSize{};
    constexpr auto v_avail_sg = ReduceK{} / ReduceSGQ{};
    return Int < (v_total_sg > v_avail_sg) ? cute::gcd(v_total_sg, v_avail_sg) : v_total_sg > {};
  }

  using SGTileShapeA = decltype(atuple_coshape(FragA{}.tv_layout()));
  using ReduceSGQ = decltype(cute::gcd(get<0>(SGTileShapeA{}), ReduceK{}));
  using ReduceSGV = decltype(reduce_sg_v_helper());
  using ReduceSGLayout = decltype(make_identity_layout(Shape<ReduceSGQ, ReduceSGV>{}));

  using SGTileShapeO = decltype(shape_div(take<0, 2>(SGTileShapeA{}), shape(ReduceSGLayout{})));

  using ReduceFragA =
      decltype(make_subgroup_tensor<ElementA>(make_layout(select<1, 0>(SGTileShapeO{}), Stride<E<1>, E<0>>{})));
  using ReduceFragARow = decltype(reduce<1>(ReduceFragA{}, sycl::plus<void>{}));

  static auto default_tiled_copy_O_helper() {
    if constexpr (ReduceK{} == _1{})
      return make_block_2d_copy_D(TiledMMAPV{}, TensorO2D{});
    else
      return make_block_2d_copy_D_subtiled(TiledMMAPV{}, ReduceFragA{}.tv_layout(), ReduceSGLayout{}, TensorO2D{});
  }

  // Default TiledCopy for writing output
  using DefaultTiledCopyO = decltype(default_tiled_copy_O_helper());
  using TiledCopyO = conditional_t<is_void_v<TiledCopyO_>, DefaultTiledCopyO, TiledCopyO_>;

  //
  // Shared memory storage
  //
  // Note sum/max tiles are padded to 16 elements, due to limitations in CuTe block load infrastructure.
  using AlignedSGTileA_Q = C<((size<0>(SGTileShapeA{}) + intel::sg_size - 1) / intel::sg_size) * intel::sg_size>;

  struct SharedStorageNone {};
  struct SharedStorageReduceK {
    cute::array<ElementA, size(SGTileShapeA{}) * SGPerWG{}> a_data;
    cute::array<ElementA, AlignedSGTileA_Q{} * SGPerWG{}> a_sum_data, a_max_data;
  };

  using SharedStorage = conditional_t<(ReduceK{} > _1{}), SharedStorageReduceK, SharedStorageNone>;

 private:
  SharedStorage& shared;

 public:
  //
  // Arguments
  //
  struct Arguments {};

  //
  // Params
  //
  struct Params {};

  //
  // methods
  //

  CUTLASS_HOST_DEVICE
  XeMlaEpilogue(Params const& params_, SharedStorage& shared_) : shared(shared_) {}

  static constexpr Params to_underlying_arguments(Arguments const& args, void* /* workspace */) {
    return {};
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) {
    return true;
  }

  template <typename QVCoord>
  CUTLASS_DEVICE void operator()(
      TensorO2D const& O,       // Global O tensor:   (q,v)
      FragA& tArA,              // O accumulator:     (q,v)
      FragARow& tA_max,         // Softmax row-wise max accumulator
      FragARow& tA_sum,         // Softmax row-wise sum accumulator
      QVCoord blk_qv,           // WG tile indices: (Q,V)
      int thr_id,               // Work-item ID
      TensorLSE1D const& LSE) {  // Global LSE tensor for this (head,batch): (q)
    using namespace cute;
    using ElementA = typename FragA::element_type;

    // Reduce k-blocks of A and A_sum across WG, if needed.
    auto [rA, rA_sum, rA_max, active] = reduce_A(tArA, tA_max, tA_sum, thr_id);

    /* Some subgroups may not have any work to do; if so, quit early. */
    if (!active) return;

    /* Tile output. cO/gO are identity tensors, so tOgO exposes the (q,v)
       coordinate of each output fragment element. */
    Tensor cO = make_identity_tensor(O.shape());       // (q,v)
    Tensor gO = local_tile(cO, TileShapeO{}, blk_qv);  // (q,v)

    /* Prepare slices */
    TiledCopyO copy_o{O};
    auto thr_copy_o = copy_o.get_slice(thr_id);

    auto tOrO = thr_copy_o.partition_sg_fragment_S(gO);
    auto tOgO = thr_copy_o.partition_D(gO);

    /* Emit LSE while the softmax sums are still the raw denominators. */
    write_lse(rA, rA_sum, rA_max, tOrO, tOgO, blk_qv, LSE);

    /* Complete softmax, dividing out sums. */
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < rA_sum.size(); i++)
      rA_sum(i) = ElementA(1) / rA_sum(i);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < rA.size(); i++) {
      auto val = broadcast<0>(rA_sum, rA, i);
      rA(i) *= val;
    }

    /* Reorder tile and write out */
    reorder(rA, tOrO);
    copy(copy_o, tOrO, tOgO);
  }

  /// Write the per-row log2-domain log-sum-exp of the attention scores.
  ///
  /// The mainloop keeps softmax state in the log2 domain with sm_scale already
  /// folded into the logits, so rA_max is log2(e) * (scaled row max) and
  /// rA_sum(row) = sum_j exp2(logit_j - rA_max(row)). The LSE is emitted in that
  /// same log2 domain, so no base conversion is applied:
  ///     lse = log2(sum_j exp2(logit_j)) = rA_max + log2(rA_sum)
  /// (Multiply by ln(2) to get the natural-log log-sum-exp.)
  /// Rows attending to no unmasked key have rA_sum == 0 and emit -INFINITY.
  ///
  /// The store happens in output-element space at the v == 0 column so each
  /// query row is written exactly once. Rows past the tensor's Q extent (the
  /// padding rows of a partial Q tile) are skipped: unlike the 2D block store
  /// used for O, these scalar stores are not bounds-clamped, and in the ragged
  /// prefill layout they would land on the next request's rows.
  ///
  /// A null LSE data pointer (caller does not want LSE) turns this into a no-op;
  /// the softmax statistics are computed by the mainloop either way.
  template <class RedFragA, class RedFragARow, class FragO, class CoordO, class QVCoord>
  CUTLASS_DEVICE void write_lse(
      RedFragA const& rA,         // Reduced O accumulator: (q,v)
      RedFragARow const& rA_sum,  // Reduced softmax row-wise sum
      RedFragARow const& rA_max,  // Reduced softmax row-wise max (log2 domain)
      FragO const& tOrO,          // Output fragment (layout donor)
      CoordO const& tOgO,         // Output coordinates: (q,v) per element
      QVCoord blk_qv,             // WG tile indices: (Q,V)
      TensorLSE1D const& LSE) {   // Global LSE tensor for this (head,batch): (q)
    using namespace cute;

    if (raw_pointer_cast(LSE.data()) == nullptr) return;
    /* Only the first V tile owns the LSE row (V is not split for MLA, but keep
       the guard so a future V-split cannot double-write). */
    if (int(get<1>(blk_qv)) != 0) return;

    /* Broadcast the row-wise statistics to every element of the A fragment,
       then reorder them into the output fragment layout, where tOgO gives each
       element its (q,v) coordinate. Keep them in ElementA (float): tOrO may be
       a narrower output type. */
    auto sum_e = rA;
    auto max_e = rA;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < rA.size(); i++) {
      sum_e(i) = broadcast<0>(rA_sum, rA, i);
      max_e(i) = broadcast<0>(rA_max, rA, i);
    }

    auto tv = tOrO.tv_layout();
    auto tO_sum = make_subgroup_tensor(make_fragment_like<ElementA>(tOrO.layout()), tv);
    auto tO_max = make_subgroup_tensor(make_fragment_like<ElementA>(tOrO.layout()), tv);
    reorder(sum_e, tO_sum);
    reorder(max_e, tO_max);

    int num_rows = int(size<0>(LSE));

    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < int(tO_sum.size()); j++) {
      if (int(get<1>(tOgO(j))) != 0) continue;
      int row = int(get<0>(tOgO(j)));
      if (row >= num_rows) continue;
      float d = float(tO_sum(j));
      float lse = (d > 0.f) ? (float(tO_max(j)) + sycl::log2(d)) : -INFINITY;
      LSE(row) = static_cast<ElementLSE>(lse);
    }
  }

  /// Split-KV epilogue operator.
  ///
  /// Stores unnormalized partial O to O_accum and writes exp_sum / max_logit
  /// for this (head, batch, kv_split) to global memory.
  ///
  template <typename QVCoord>
  CUTLASS_DEVICE void operator()(
      TensorO2D const& O_accum,  // 2D slice of partial output buffer: (q, v)
      FragA& tArA,               // O accumulator fragment from mainloop
      FragARow& tA_max,          // Softmax row-wise max accumulator
      FragARow& tA_sum,          // Softmax row-wise sum accumulator
      QVCoord blk_qv,            // WG tile indices: (Q, V)
      int thr_id,                // Work-item ID
      ElementAcc& exp_sum,       // Reference to this split's exp_sum element
      ElementAcc& max_logit,     // Reference to this split's max_logit element
      int num_kv_splits) {       // Total number of KV splits
    using namespace cute;

    // Step 1: Cross-subgroup reduction of accumulators
    auto [rA, rA_sum, rA_max, active] = reduce_A(tArA, tA_max, tA_sum, thr_id);

    // Step 2: Store LSE statistics for this split.
    // Thread 0 (subgroup 0, lane 0) is always active after reduce_A.
    if (thr_id == 0) {
      exp_sum = static_cast<ElementAcc>(rA_sum(0));
      max_logit = static_cast<ElementAcc>(rA_max(0));
    }

    // Inactive subgroups (when ReduceK > 1) have no work to do.
    if (!active) return;

    // Step 3: Write O_accum.
    // For num_kv_splits > 1, O_accum is stored UNNORMALIZED (raw numerator):
    //   O_accum_s = sum_i exp2(S_si - max_s) * V_si
    // The reduction kernel merges via:
    //   O = [sum_s O_accum_s * exp2(max_s - global_max)]
    //     / [sum_s exp_sum_s * exp2(max_s - global_max)]
    //
    // For num_kv_splits == 1, normalize in-place (no reduction needed).
    if (num_kv_splits == 1) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < rA_sum.size(); i++)
        rA_sum(i) = ElementA(1) / rA_sum(i);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < rA.size(); i++) {
        rA(i) *= broadcast<0>(rA_sum, rA, i);
      }
    }

    /* Tile output */
    Tensor cO = make_identity_tensor(O_accum.shape());  // (q,v)
    Tensor gO = local_tile(cO, TileShapeO{}, blk_qv);   // (q,v)

    /* Prepare slices */
    TiledCopyO copy_o{O_accum};
    auto thr_copy_o = copy_o.get_slice(thr_id);

    auto tOrO = thr_copy_o.partition_sg_fragment_S(gO);
    auto tOgO = thr_copy_o.partition_D(gO);

    /* Reorder tile and write out */
    reorder(rA, tOrO);
    copy(copy_o, tOrO, tOgO);
  }

  template <typename FragA, typename FragARow>
  CUTLASS_DEVICE decltype(auto) reduce_A(
      FragA& tArA,       // O accumulator:   (q,v)
      FragARow& tA_max,  // Softmax row-wise max accumulator
      FragARow& tA_sum,  // Softmax row-wise sum accumulator
      int thr_id) {      // Work-item ID

    using namespace sycl::ext::oneapi::this_work_item;
    if constexpr (ReduceK{} == _1{}) {
      return std::make_tuple(tArA, tA_sum, tA_max, true);
    } else {
      /* Identify A tile ID and k block for this subgroup. */
      auto thr_vak = group<1, 3>(TiledMMAPV{}.get_thr_layout_vmnk()).get_flat_coord(assert_uniform(thr_id));
      auto a_tile = get<1>(thr_vak);
      auto k_blk = get<2>(thr_vak);

      /* Set up SLM tensors and partition A tiles among participating subgroups */
      auto shape_A = append(append(SGTileShapeA{}, ReduceK{}), SGPerWG{} / ReduceK{});
      auto shape_A_row = make_shape(get<0>(SGTileShapeO{}), shape(ReduceSGLayout{}), ReduceK{}, SGPerWG{} / ReduceK{});

      /* Physical layouts, with sub-tile modes broken out */
      auto sA_layout = group<2, 4>(flat_divide(make_ordered_layout(shape_A, Step<_1, _0, _2, _3>{}), SGTileShapeO{}));
      auto sA_row_stride =
          make_stride(_1{}, make_stride(get<0>(shape_A_row), _0{}), AlignedSGTileA_Q{}, AlignedSGTileA_Q{} * ReduceK{});
      auto sA_row_layout = make_layout(shape_A_row, sA_row_stride);

      /* Coordinate layouts, with sub-tile modes broken out */
      auto basis2 = make_basis_like(SGTileShapeO{});
      auto sA_coords = make_layout(
          append(SGTileShapeO{}, shape(ReduceSGLayout{})), append(basis2, product_each(zip(SGTileShapeO{}, basis2))));

      auto sA = make_tensor(make_smem_ptr<ElementA>(&shared.a_data), sA_layout);  // (q,v,rblk_dst,rblk_src,a_tile)
      auto sA_max =
          make_tensor(make_smem_ptr<ElementA>(&shared.a_max_data), sA_row_layout);  // (q,rblk_dst,rblk_src,a_tile)
      auto sA_sum =
          make_tensor(make_smem_ptr<ElementA>(&shared.a_sum_data), sA_row_layout);  // (q,rblk_dst,rblk_src,a_tile)

      /* Write my contributions to SLM. */
      copy_block_r2s(tA_max, sA_max(_, _, k_blk, a_tile));
      barrier_arrive(ScopeWorkgroup, SemanticsRelease | SemanticsWGMemory);
      copy_block_r2s(tA_sum, sA_sum(_, _, k_blk, a_tile));
      copy_block_r2s(tArA, sA(_, _, _, k_blk, a_tile), sA_coords);

      bool active = (k_blk < size(ReduceSGLayout{})) || (ReduceK{} == size(ReduceSGLayout{}));  // help compiler out

      /* Wait for maxima to be available, signal other data available */
      barrier_wait(ScopeWorkgroup, SemanticsAcquire | SemanticsWGMemory);
      barrier_arrive(ScopeWorkgroup, SemanticsRelease | SemanticsWGMemory);

      ReduceFragA rA;
      ReduceFragARow rA_sum, rA_max, rA_kmax[ReduceK{}];

      if (active) {
        /* Read A_max back from SLM and reduce. */
        CUTLASS_PRAGMA_UNROLL
        for (int kr = 0; kr < ReduceK{}; kr++) {
          copy_block_s2r(sA_max(_, k_blk, kr, a_tile), rA_kmax[kr]);
        }

        rA_max = rA_kmax[0];
        for (int kr = 1; kr < ReduceK{}; kr++) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < rA_max.size(); i++) {
            rA_max(i) = cute::max(rA_max(i), rA_kmax[kr](i));
          }
        }

        /* Calculate scale factors for aligning per-block maxima. */
        for (int kr = 0; kr < ReduceK{}; kr++) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < rA_max.size(); i++) {
            rA_kmax[kr](i) = sycl::native::exp2(rA_kmax[kr](i) - rA_max(i));
          }
        }
      }

      /* Wait for A/A_sum data to be available */
      barrier_wait(ScopeWorkgroup, SemanticsAcquire | SemanticsWGMemory);

      if (active) {
        /* Read A/A_sum back from SLM, align scaling to new maxima, and reduce. */
        clear(rA_sum);

        CUTLASS_PRAGMA_UNROLL
        for (int kr = 0; kr < ReduceK{}; kr++) {
          ReduceFragARow rA_sum_read;
          copy_block_s2r(sA_sum(_, k_blk, kr, a_tile), rA_sum_read);

          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < rA_sum_read.size(); i++) {
            rA_sum(i) += rA_sum_read(i) * rA_kmax[kr](i);
          }
        }

        clear(rA);

        CUTLASS_PRAGMA_UNROLL
        for (int kr = 0; kr < ReduceK{}; kr++) {
          ReduceFragA rA_read;
          copy_block_s2r(sA(_, _, k_blk, kr, a_tile), sA_coords(_, _, 0), rA_read);

          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < rA_read.size(); i++) {
            rA(i) += rA_read(i) * broadcast<0>(rA_kmax[kr], rA, i);
          }
        }
      }
      return std::make_tuple(rA, rA_sum, rA_max, active);
    }
  }
};
/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::flash_attention::collective
