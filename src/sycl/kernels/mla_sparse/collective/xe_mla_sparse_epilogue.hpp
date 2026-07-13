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
    \brief Sparse MLA Epilogue with LSE output and attn_sink merge
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
template <class CollectiveMainloop, class TileShapeO_, class TensorO_, class TiledCopyO_ = void>
class XeMlaSparseEpilogue {
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

  using FragA = typename CollectiveMainloop::FragA;
  using FragARow = typename CollectiveMainloop::FragARow;
  using ElementA = typename FragA::value_type;

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
  using AlignedSGTileA_Q = C<((size<0>(SGTileShapeA{}) + intel::sg_size - 1) / intel::sg_size) * intel::sg_size>;

  struct SharedStorageNone {};
  struct SharedStorageReduceK {
    cute::array<ElementA, size(SGTileShapeA{}) * SGPerWG{}> a_data;
    cute::array<ElementA, AlignedSGTileA_Q{} * SGPerWG{}> a_sum_data, a_max_data;
  };

  using SharedStorage = conditional_t<(ReduceK{} > _1{}), SharedStorageReduceK, SharedStorageNone>;

 private:
  SharedStorage& shared;
  float const* ptr_attn_sink;
  float* ptr_lse_out;
  int num_heads_q;

 public:
  //
  // Arguments
  //
  struct Arguments {
    float const* ptr_attn_sink = nullptr;  // [H_Q] per-head attention sink values
    float* ptr_lse_out = nullptr;          // [B, H_Q, 1] output LSE values
    int num_heads_q = 0;
  };

  //
  // Params
  //
  struct Params {
    float const* ptr_attn_sink = nullptr;
    float* ptr_lse_out = nullptr;
    int num_heads_q = 0;
  };

  //
  // methods
  //

  CUTLASS_HOST_DEVICE
  XeMlaSparseEpilogue(Params const& params_, SharedStorage& shared_)
      : shared(shared_),
        ptr_attn_sink(params_.ptr_attn_sink),
        ptr_lse_out(params_.ptr_lse_out),
        num_heads_q(params_.num_heads_q) {}

  static constexpr Params to_underlying_arguments(Arguments const& args, void* /* workspace */) {
    return Params{args.ptr_attn_sink, args.ptr_lse_out, args.num_heads_q};
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) {
    return true;
  }

  template <typename QVCoord>
  CUTLASS_DEVICE void operator()(
      TensorO2D const& O,  // Global O tensor: (q,v)
      FragA& tArA,         // O accumulator:   (q,v)
      FragARow& tA_max,    // Softmax row-wise max accumulator
      FragARow& tA_sum,    // Softmax row-wise sum accumulator
      QVCoord blk_qv,      // WG tile indices: (Q,V)
      int thr_id,
      int head_coord,     // Head index for attn_sink lookup
      int batch_coord) {  // Batch index for LSE output
    using namespace cute;
    using ElementA = typename FragA::element_type;

    // Reduce k-blocks of A and A_sum across WG, if needed.
    // Returns: reduced output accumulator, reduced max, reduced sum, active flag
    auto [rA, rA_max, rA_sum, active] = reduce_A(tArA, tA_max, tA_sum, thr_id);

    /* Some subgroups may not have any work to do; if so, quit early. */
    if (!active) return;

    // -----------------------------------------------------------------------
    // Compute LSE: lse = m / log2(e) + ln(sum)
    // where m = rA_max (in log2-scaled space), sum = rA_sum
    // -----------------------------------------------------------------------
    constexpr float kRcpLog2e = 0.6931471805599453f;  // 1/log2(e) = ln(2)

    // Complete softmax: divide output by sum, then apply attn_sink merge
    float attn_sink_val = ptr_attn_sink[head_coord];

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < rA_sum.size(); i++) {
      // Compute LSE for this row
      // rA_max is m (in log2-scaled space): m_log2 = scale * max(scores)
      // where scale = sm_scale * log2(e)
      // Actual max in natural log space: m_nat = m_log2 / log2(e) = m_log2 * ln(2)
      // lse = m_nat + ln(sum) = m_log2 * ln(2) + ln(sum)
      float m_log2 = static_cast<float>(rA_max(i));
      float sum_val = static_cast<float>(rA_sum(i));
      float lse = m_log2 * kRcpLog2e + sycl::log(sum_val);

      // Store LSE output: [B, H, 1]
      // Only the first thread in the row writes LSE (avoid duplicates)
      if (thr_id == 0 && i == 0) {
        int lse_idx = batch_coord * num_heads_q + head_coord;
        ptr_lse_out[lse_idx] = lse;
      }

      // Compute attn_sink merge weight:
      // w = exp(lse) / (exp(lse) + exp(attn_sink[h]))
      // = sigmoid(lse - attn_sink[h])
      // = 1 / (1 + exp(attn_sink[h] - lse))
      float w = 1.0f / (1.0f + sycl::exp(attn_sink_val - lse));

      // Apply normalization and attn_sink weighting:
      // out = w * (O_acc / sum)
      float inv_sum_w = w / sum_val;
      rA_sum(i) = static_cast<ElementA>(inv_sum_w);
    }

    // Apply combined normalization + attn_sink weight
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < rA.size(); i++) {
      auto val = broadcast<0>(rA_sum, rA, i);
      rA(i) *= val;
    }

    /* Tile output */
    Tensor cO = make_identity_tensor(O.shape());
    Tensor gO = local_tile(cO, TileShapeO{}, blk_qv);

    /* Prepare slices */
    TiledCopyO copy_o{O};
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
      return std::make_tuple(tArA, tA_max, tA_sum, true);
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

      auto sA = make_tensor(make_smem_ptr<ElementA>(&shared.a_data), sA_layout);
      auto sA_max = make_tensor(make_smem_ptr<ElementA>(&shared.a_max_data), sA_row_layout);
      auto sA_sum = make_tensor(make_smem_ptr<ElementA>(&shared.a_sum_data), sA_row_layout);

      /* Write my contributions to SLM. */
      copy_block_r2s(tA_max, sA_max(_, _, k_blk, a_tile));
      barrier_arrive(ScopeWorkgroup, SemanticsRelease | SemanticsWGMemory);
      copy_block_r2s(tA_sum, sA_sum(_, _, k_blk, a_tile));
      copy_block_r2s(tArA, sA(_, _, _, k_blk, a_tile), sA_coords);

      bool active = (k_blk < size(ReduceSGLayout{})) || (ReduceK{} == size(ReduceSGLayout{}));

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
        for (int kr = 1; kr < ReduceK{}; kr++)
          cute::transform(rA_max, rA_kmax[kr], rA_max, cute::max_fn{});

        /* Calculate scale factors for aligning per-block maxima. */
        for (int kr = 0; kr < ReduceK{}; kr++) {
          cute::transform(
              rA_max, rA_kmax[kr], rA_kmax[kr], [](auto gmax, auto kmax) { return sycl::native::exp2(kmax - gmax); });
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
      return std::make_tuple(rA, rA_max, rA_sum, active);
    }
  }
};
/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::flash_attention::collective
