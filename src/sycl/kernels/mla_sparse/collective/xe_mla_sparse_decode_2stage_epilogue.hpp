/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
/*! \file
    \brief Two-stage sparse MLA decode Stage 2 epilogue collective for DeepSeek V4.

    Cross-subgroup reduction of the per-split O accumulator + softmax row stats
    (reduce_L), pre-sink LSE emission, optional attn_sink merge, final softmax
    normalization, and the coalesced store of O. Consumes the tArA / tA_max /
    tA_sum produced by XeMlaSparseDecode2StageMainloop.

    Structural analog of collective/xe_mla_sparse_epilogue.hpp (the fused path):
    owns the reduction/output type aliases, the ReduceK SharedStorage, its Params,
    ctor (Params const&, SharedStorage&), and operator(). IMPORTANT: the softmax
    normalization / attn_sink / LSE math here is the 2-stage kernel's own
    formulation (exp2-based sink merge, pre-sink LSE).

    reference: tests/test_flash_mla_with_kvcache.py _sm120_sparse_decode_fwd.
*/

#pragma once

#include "sycl/kernels/mla_sparse/kernel/xe_mla_sparse_decode_2stage_common.hpp"

namespace cutlass::flash_attention::collective {

using cutlass::flash_attention::kernel::LOG_2_E;
using cutlass::flash_attention::kernel::LOG_E_2;
using cutlass::flash_attention::kernel::SparseAttnDecodeParams;

/////////////////////////////////////////////////////////////////////////////////////////////////
template <class CollectiveMainloop_, bool HAS_ATTN_SINK_>
class XeMlaSparseDecode2StageEpilogue {
 public:
  //
  // Type Aliases
  //
  using CollectiveMainloop = CollectiveMainloop_;
  static constexpr bool HAS_ATTN_SINK = HAS_ATTN_SINK_;

  using Traits = typename CollectiveMainloop::Traits;
  using ElementO = typename CollectiveMainloop::ElementO;
  using TiledMMAPV = typename CollectiveMainloop::TiledMMAPV;
  using TileShapePV = typename CollectiveMainloop::TileShapePV;
  using TileShapeOut = typename CollectiveMainloop::TileShapeOut;
  using ElementA = typename CollectiveMainloop::ElementA;

  using ReduceK = typename CollectiveMainloop::ReduceK;
  using FragA = typename CollectiveMainloop::FragA;
  using FragARow = typename CollectiveMainloop::FragARow;

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
  using SGPerWG = decltype(product(take<1, 4>(shape(typename TiledMMAPV::ThrLayoutVMNK{}))));
  using GmemLayoutO = Layout<Shape<int, Int<Traits::D_V_PER_SPLIT>>, Stride<Int<Traits::D_V>, _1>>;
  using TensorO2D = decltype(make_tensor(make_gmem_ptr((ElementO*)nullptr), GmemLayoutO{}));

  static auto default_tiled_copy_O_helper() {
    if constexpr (ReduceK{} == _1{})
      return make_block_2d_copy_D(TiledMMAPV{}, TensorO2D{});
    else
      return make_block_2d_copy_D_subtiled(TiledMMAPV{}, ReduceFragA{}.tv_layout(), ReduceSGLayout{}, TensorO2D{});
  }

  using TiledCopyO = decltype(default_tiled_copy_O_helper());

  struct SharedStorageNonReduceK {};
  using AlignedSGTileA_Q = C<((size<0>(SGTileShapeA{}) + intel::sg_size - 1) / intel::sg_size) * intel::sg_size>;

  struct SharedStorageReduceK {
    cute::array<ElementA, size(SGTileShapeA{}) * SGPerWG{}> a_data;
    cute::array<ElementA, AlignedSGTileA_Q{} * SGPerWG{}> a_sum_data, a_max_data;
  };

  using SharedStorage = conditional_t<(ReduceK{} > 1), SharedStorageReduceK, SharedStorageNonReduceK>;

  //
  // Arguments / Params (flat SparseAttnDecodeParams, matching the launcher wiring)
  //
  using Arguments = SparseAttnDecodeParams;
  using Params = SparseAttnDecodeParams;

 private:
  Params params;
  SharedStorage& shared_storage;

 public:
  //
  // methods
  //
  CUTLASS_DEVICE
  XeMlaSparseDecode2StageEpilogue(Params const& params_, SharedStorage& shared)
      : params(params_), shared_storage(shared) {}

  static constexpr Params to_underlying_arguments(Arguments const& args, void* /* workspace */) {
    return args;
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) {
    return true;
  }

  // Reduce O / softmax stats across the ReduceK subgroups that share a query row,
  // then normalize (with optional attn_sink merge), emit pre-sink LSE, and store O.
  template <class TensorO>
  CUTLASS_DEVICE void operator()(
      TensorO const& O,  // [h_q, D_V] gmem, offset to (batch, seq)
      FragA& tArA,
      FragARow& tA_max,
      FragARow& tA_sum,
      int thr_id,
      int sg_id,
      int tid_in_sg,
      int v_split_idx,
      int head_bid,
      int cur_head_start_idx,
      int batch_idx,
      int seq_idx) {
    float* lse = params.lse;

    TiledMMAPV mma_pv{};

    Tensor proxyO = make_identity_tensor(O.shape());
    Tensor gO = local_tile(proxyO, TileShapeOut{}, make_coord(head_bid, v_split_idx));

    TiledCopyO tiled_copy_O{O};
    auto thr_copy_o = tiled_copy_O.get_slice(thr_id);
    auto tOrO = thr_copy_o.partition_sg_fragment_S(gO);
    auto tOgO = thr_copy_o.partition_D(gO);

    auto reduce_L = [&]() {
      if constexpr (ReduceK{} == _1{}) {
        return std::make_tuple(tArA, tA_max, tA_sum, true);
      } else {
        auto thr_vak = group<1, 3>(mma_pv.get_thr_layout_vmnk()).get_flat_coord(assert_uniform(thr_id));
        auto a_tile = get<1>(thr_vak);
        auto k_blk = get<2>(thr_vak);

        auto shape_A = append(append(SGTileShapeA{}, ReduceK{}), SGPerWG{} / ReduceK{});
        auto shape_A_row =
            make_shape(get<0>(SGTileShapeO{}), shape(ReduceSGLayout{}), ReduceK{}, SGPerWG{} / ReduceK{});
        auto sA_layout = group<2, 4>(flat_divide(make_ordered_layout(shape_A, Step<_1, _0, _2, _3>{}), SGTileShapeO{}));
        auto sA_row_stride = make_stride(
            _1{}, make_stride(get<0>(shape_A_row), _0{}), AlignedSGTileA_Q{}, AlignedSGTileA_Q{} * ReduceK{});
        auto sA_row_layout = make_layout(shape_A_row, sA_row_stride);
        auto basis2 = make_basis_like(SGTileShapeO{});
        auto sA_coords = make_layout(
            append(SGTileShapeO{}, shape(ReduceSGLayout{})), append(basis2, product_each(zip(SGTileShapeO{}, basis2))));

        auto sA = make_tensor(make_smem_ptr<ElementA>(&shared_storage.a_data), sA_layout);
        auto sA_max = make_tensor(make_smem_ptr<ElementA>(&shared_storage.a_max_data), sA_row_layout);
        auto sA_sum = make_tensor(make_smem_ptr<ElementA>(&shared_storage.a_sum_data), sA_row_layout);

        copy_block_r2s(tA_max, sA_max(_, _, k_blk, a_tile));
        barrier_arrive(ScopeWorkgroup, SemanticsRelease | SemanticsWGMemory);
        copy_block_r2s(tA_sum, sA_sum(_, _, k_blk, a_tile));
        copy_block_r2s(tArA, sA(_, _, _, k_blk, a_tile), sA_coords);

        bool active = (k_blk < size(ReduceSGLayout{})) || (ReduceK{} == size(ReduceSGLayout{}));
        barrier_wait(ScopeWorkgroup, SemanticsAcquire | SemanticsWGMemory);
        barrier_arrive(ScopeWorkgroup, SemanticsRelease | SemanticsWGMemory);

        ReduceFragA rA;
        ReduceFragARow rA_sum, rA_max, rA_kmax[ReduceK{}];

        if (active) {
          CUTLASS_PRAGMA_UNROLL
          for (int kr = 0; kr < ReduceK{}; kr++) {
            copy_block_s2r(sA_max(_, k_blk, kr, a_tile), rA_kmax[kr]);
          }

          rA_max = rA_kmax[0];
          for (int kr = 1; kr < ReduceK{}; kr++) {
            cute::transform(rA_max, rA_kmax[kr], rA_max, cute::max_fn{});
          }

          for (int kr = 0; kr < ReduceK{}; kr++) {
            cute::transform(
                rA_max, rA_kmax[kr], rA_kmax[kr], [](auto gmax, auto kmax) { return sycl::native::exp2(kmax - gmax); });
          }
        }

        barrier_wait(ScopeWorkgroup, SemanticsAcquire | SemanticsWGMemory);

        if (active) {
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
    };

    auto [rA, rA_max, rA_sum, active] = reduce_L();
    if (!active) return;

    constexpr int valid_tid_per_sg = Traits::B_H / get<0>(typename Traits::SubgroupLayoutQK{}.shape());
    static_assert(
        valid_tid_per_sg <= Traits::SUBGROUP_SIZE, "valid_tid_per_sg must be less than or equal to SUBGROUP_SIZE");
    if (v_split_idx == 0 && tid_in_sg < valid_tid_per_sg && sg_id * valid_tid_per_sg < Traits::B_H) {
      int local_head_idx = sg_id * valid_tid_per_sg + tid_in_sg;
      int head_idx = cur_head_start_idx + local_head_idx;
      if (head_idx < params.shape.h_q) {
        int lse_idx = batch_idx * params.stride_lse_b + seq_idx * params.stride_lse_s_q + head_idx;

        float row_max = -INFINITY;
        if (rA_max(0) != params.sm_scale_div_log2 * cutlass::platform::numeric_limits<ElementA>::lowest()) {
          row_max = rA_max(0) * LOG_E_2;
        }

        float row_lse = INFINITY;
        if (rA_sum(0) > 0.f) {
          row_lse = row_max + sycl::native::log2(rA_sum(0)) * LOG_E_2;
        }
        lse[lse_idx] = row_lse;
      }
    }

    if constexpr (HAS_ATTN_SINK) {
      if constexpr (ReduceK{} > 1) {
        auto thr_vak = group<1, 3>(mma_pv.get_thr_layout_vmnk()).get_flat_coord(assert_uniform(thr_id));
        int reduce_head_offset = (get<2>(thr_vak) / ReduceSGV{}) * get<0>(SGTileShapeO{});

        CUTE_UNROLL
        for (int i = 0; i < rA.size(); ++i) {
          auto global_exp_sum = broadcast<0>(rA_sum, rA, i);
          ElementA final_rescale = global_exp_sum != 0 ? ElementA(1) / global_exp_sum : ElementA(0);
          int local_head_idx = get<0>(rA.tv_layout()(0, i));
          int head_idx = cur_head_start_idx + reduce_head_offset + local_head_idx;
          if (head_idx < params.shape.h_q) {
            auto global_max = broadcast<0>(rA_max, rA, i);
            float attn_sink_val = params.attn_sink[head_idx];
            ElementA sink_exp_sum = sycl::native::exp2(static_cast<ElementA>(attn_sink_val * LOG_2_E) - global_max);
            ElementA global_exp_sum_with_sink = global_exp_sum + sink_exp_sum;
            final_rescale = global_exp_sum_with_sink != 0 ? ElementA(1) / global_exp_sum_with_sink : ElementA(0);
          }
          rA(i) *= final_rescale;
        }
      } else {
        if (tid_in_sg < valid_tid_per_sg && sg_id * valid_tid_per_sg < Traits::B_H) {
          int local_head_idx = sg_id * valid_tid_per_sg + tid_in_sg;
          int head_idx = cur_head_start_idx + local_head_idx;
          if (head_idx < params.shape.h_q) {
            float attn_sink_val = params.attn_sink[head_idx];
            CUTE_UNROLL
            for (int i = 0; i < rA_sum.size(); ++i) {
              rA_sum(i) += sycl::native::exp2(static_cast<ElementA>(attn_sink_val * LOG_2_E) - rA_max(i));
            }
          }
        }

        CUTE_UNROLL
        for (int i = 0; i < rA_sum.size(); ++i) {
          if (rA_sum(i) != 0) {
            rA_sum(i) = ElementA(1) / rA_sum(i);
          } else {
            rA_sum(i) = 0;
          }
        }

        CUTE_UNROLL
        for (int i = 0; i < rA.size(); ++i) {
          rA(i) *= broadcast<0>(rA_sum, rA, i);
        }
      }
    } else {
      CUTE_UNROLL
      for (int i = 0; i < rA_sum.size(); ++i) {
        if (rA_sum(i) != 0) {
          rA_sum(i) = ElementA(1) / rA_sum(i);
        } else {
          rA_sum(i) = 0;
        }
      }

      CUTE_UNROLL
      for (int i = 0; i < rA.size(); ++i) {
        rA(i) *= broadcast<0>(rA_sum, rA, i);
      }
    }

    reorder(rA, tOrO);
    copy(tiled_copy_O, tOrO, tOgO);
  }
};
/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::flash_attention::collective
