/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
/*! \file
    \brief Two-stage sparse MLA decode Stage 2 mainloop collective for DeepSeek V4.

    QK/PV DPAS GEMM engine + online (log2) softmax over the Stage 1 gathered tile.
    Consumes the per-(batch, seq, v-split) gmem Q/K/V tiles built by the kernel
    wrapper and produces the O accumulator + softmax max/sum row stats, which the
    epilogue collective then reduces, normalizes, and writes out.

    Structural analog of collective/xe_mla_sparse_mainloop.hpp (the fused path): a
    compute collective owning the MMA/tile/fragment type aliases, its Params, its
    SharedStorage, ctor (Params const&, SharedStorage&), and operator(). Shared
    declarations (SparseAttnDecodeParams, LOG_* constants, the copy_block_*
    helpers) come from the kernel/ common header; the DPAS/tile config it reads
    off its Traits template param is MlaSparseDecode2StageXe (host types header).

    Correctness reference: tests/test_flash_mla_with_kvcache.py _sm120_sparse_decode_fwd.
*/

#pragma once

#include "sycl/kernels/mla_sparse/kernel/xe_mla_sparse_decode_2stage_common.hpp"

namespace cutlass::flash_attention::collective {

using cutlass::flash_attention::kernel::LOG_2_E;
using cutlass::flash_attention::kernel::LOG_E_2;
using cutlass::flash_attention::kernel::SparseAttnDecodeParams;

/////////////////////////////////////////////////////////////////////////////////////////////////
template <int D_QK_, bool IS_FP8_QUERY_, typename Traits_>
class XeMlaSparseDecode2StageMainloop {
 public:
  //
  // Type Aliases
  //
  using Traits = Traits_;
  static constexpr int D_QK = D_QK_;
  static constexpr bool IS_FP8_QUERY = IS_FP8_QUERY_;

  using ElementQ = typename Traits::ElementQ;
  using ElementKV = typename Traits::ElementKV;
  using ElementO = typename Traits::ElementO;
  using TiledMMAQK = typename Traits::TiledMMAQK;
  using TiledMMAPV = typename Traits::TiledMMAPV;
  using TileShapeQK = typename Traits::TileShapeQK;
  using TileShapePV = typename Traits::TileShapePV;
  using TileShapeOut = typename Traits::TileShapeOut;
  using ElementS = typename TiledMMAQK::ValTypeD;
  using ElementA = typename TiledMMAPV::ValTypeD;

  using ReduceK = decltype(size<3>(typename TiledMMAPV::ThrLayoutVMNK{}));
  using SingleFragA = decltype(TiledMMAPV{}.get_slice(0).partition_sg_fragment_C(
      make_identity_tensor(select<0, 1>(TiledMMAPV{}.tile_mnk()))));
  static_assert(
      Traits::D_V_PER_SPLIT % get<1>(TileShapePV{}) == 0, "D_V_PER_SPLIT must be divisible by TileShapePV N dimension");
  using FragA = expand_sg_fragment_t<SingleFragA, 1, Traits::D_V_PER_SPLIT / get<1>(TileShapePV{})>;
  using FragARow = decltype(reduce<1>(FragA{}, sycl::plus<void>{}));

  // Mainloop uses no SLM; the cross-subgroup reduction storage lives in the epilogue.
  struct SharedStorage {};

  //
  // Arguments / Params (flat SparseAttnDecodeParams, matching the launcher wiring)
  //
  using Arguments = SparseAttnDecodeParams;
  using Params = SparseAttnDecodeParams;

 private:
  Params params;

 public:
  //
  // methods
  //
  CUTLASS_HOST_DEVICE
  XeMlaSparseDecode2StageMainloop(Params const& params_, SharedStorage& /* shared */) : params(params_) {}

  static constexpr Params to_underlying_arguments(Arguments const& args, void* /* workspace */) {
    return args;
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) {
    return true;
  }

  // Runs the topk-block QK/softmax/PV loop for one (batch, seq, head-block, v-split)
  // tile, accumulating into tArA and the softmax row stats tA_max / tA_sum.
  template <class TensorQ, class TensorK, class TensorV>
  CUTLASS_DEVICE void operator()(
      TensorQ const& Q,  // [h_q, D_QK] gmem, offset to (batch, seq)
      TensorK const& K,  // [gathered_topk, D_QK] gmem, offset to (batch, seq)
      TensorV const& V,  // [D_V_PER_SPLIT, gathered_topk] gmem, offset to (batch, seq, v-split)
      FragA& tArA,       // O accumulator: (q, v)
      FragARow& tA_max,  // Softmax row-wise max accumulator
      FragARow& tA_sum,  // Softmax row-wise sum accumulator
      int thr_id,
      int batch_idx,
      int seq_idx,
      int head_bid) {
    const cutlass::float_e4m3_t* q_fp8 = reinterpret_cast<const cutlass::float_e4m3_t*>(params.q);
    const int* gathered_valid_mask = params.gathered_valid_mask;
    const int cur_head_start_idx = head_bid * Traits::B_H;

    Tensor proxyQ = make_identity_tensor(Q.shape());
    Tensor proxyK = make_identity_tensor(K.shape());
    Tensor proxyP = make_identity_tensor(select<0, 1>(TileShapeQK{}));
    Tensor proxyV = make_identity_tensor(V.shape());

    constexpr int V_TILE_PER_SPLIT = Traits::D_V_PER_SPLIT / get<1>(TileShapePV{});
    auto tile_shape_v = make_shape(Int<Traits::D_V_PER_SPLIT>{}, get<2>(TileShapePV{}));

    Tensor gQ = local_tile(proxyQ, TileShapeQK{}, make_coord(head_bid, _, _), Step<_1, X, _1>{});
    Tensor gK = local_tile(proxyK, TileShapeQK{}, make_coord(_, _, _), Step<X, _1, _1>{});
    Tensor gV = local_tile(proxyV, tile_shape_v, make_coord(0, _));
    Tensor gV_split = local_tile(gV, TileShapePV{}, make_coord(_, _, 0), Step<X, _1, _1>{});

    TiledMMAQK mma_qk{};
    TiledMMAPV mma_pv{};

    auto tiled_copy_Q = make_block_2d_copy_A(mma_qk, Q);
    auto thr_copy_q = tiled_copy_Q.get_slice(thr_id);
    auto tiled_copy_k = make_block_2d_copy_B(mma_qk, K);
    auto tiled_copy_v = make_block_2d_copy_B(mma_pv, V);
    auto thr_copy_k = tiled_copy_k.get_slice(thr_id);
    auto thr_copy_v = tiled_copy_v.get_slice(thr_id);

    auto thr_mma_qk = mma_qk.get_slice(thr_id);
    auto thr_mma_pv = mma_pv.get_slice(thr_id);

    auto tQgQ = thr_copy_q.partition_S(gQ);
    auto tKgK = thr_copy_k.partition_S(gK);
    auto tVgV = thr_copy_v.partition_S(gV_split);

    auto tSrQ = thr_mma_qk.partition_sg_fragment_A(gQ(_, _, 0));
    Tensor coord_Q = make_identity_tensor(select<0, 2>(TileShapeQK{}));
    auto tQcQ = thr_mma_qk.partition_A(coord_Q);
    auto tQrQ = thr_copy_q.partition_sg_fragment_D(gQ(_, _, 0));

    auto tKrK = thr_copy_k.partition_sg_fragment_D(gK(_, _, 0, 0));
    auto tSrK = thr_mma_qk.partition_sg_fragment_B(gK(_, _, 0, 0));
    auto tSrS = thr_mma_qk.partition_sg_fragment_C(proxyP);

    auto qk_gemm_one_tile = [&](int block_idx, int tile_idx) {
      if constexpr (IS_FP8_QUERY) {
        CUTE_UNROLL
        for (int i = 0; i < tSrQ.size(); ++i) {
          int row_idx = get<0>(tQcQ(i));
          int col_idx = get<1>(tQcQ(i)) + tile_idx * get<2>(TileShapeQK{});
          int global_head_idx = cur_head_start_idx + row_idx;
          if (global_head_idx < params.shape.h_q) {
            const auto* q_tile_ptr = q_fp8 + batch_idx * params.stride_q_b + seq_idx * params.stride_q_s_q +
                                     global_head_idx * params.stride_q_h_q;
            float scale = params.q_scale_numel == 1 ? params.q_scale[0] : params.q_scale[global_head_idx];
            tSrQ(i) = ElementQ(static_cast<float>(q_tile_ptr[col_idx]) * scale);
          } else {
            tSrQ(i) = ElementQ(0.0f);
          }
        }
      } else {
        copy(tiled_copy_Q, tQgQ(_, _, _, tile_idx), tQrQ);
        reorder(tQrQ, tSrQ);
      }
      copy(tiled_copy_k, tKgK(_, _, _, block_idx, tile_idx), tKrK);
      reorder(tKrK, tSrK);
      cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
    };

    const int* gathered_valid_mask_ptr =
        gathered_valid_mask + batch_idx * params.stride_gathered_mask_b + seq_idx * params.stride_gathered_mask_s_q;
    auto mask_rS = [&](int block_idx) {
      Tensor coord_S = make_identity_tensor(select<0, 1>(TileShapeQK{}));
      Tensor tCgC = thr_mma_qk.partition_C(coord_S);
      CUTE_UNROLL
      for (int i = 0; i < tSrS.size(); ++i) {
        int col_idx = get<1>(tCgC(i));
        int topk_idx = block_idx * Traits::B_TOPK + col_idx;
        bool is_valid = topk_idx < params.shape.gathered_topk && gathered_valid_mask_ptr[topk_idx];
        if (!is_valid) {
          tSrS(i) = cutlass::platform::numeric_limits<ElementS>::lowest();
        }
      }
    };

    cute::clear(tArA);
    cute::fill(tA_max, cutlass::platform::numeric_limits<ElementA>::lowest());
    cute::fill(tA_sum, 0.0f);

    auto online_softmax_and_rescale_o = [&](int block_idx) {
      auto tA_bmax = reduce<1>(tSrS, sycl::maximum{});
      auto tA_prev_max = tA_max;
      CUTE_UNROLL
      for (int i = 0; i < tA_max.size(); i++) {
        tA_max(i) = sycl::max(tA_max(i), params.sm_scale_div_log2 * tA_bmax(i));
      }

      CUTE_UNROLL
      for (int i = 0; i < tSrS.size(); i++) {
        tSrS(i) = sycl::native::exp2(params.sm_scale_div_log2 * tSrS(i) - broadcast<0>(tA_max, tSrS, i));
      }

      FragARow rescale;
      if (block_idx > 0) {
        CUTE_UNROLL
        for (int i = 0; i < tA_max.size(); i++) {
          rescale(i) = sycl::native::exp2(tA_prev_max(i) - tA_max(i));
          tA_sum(i) *= rescale(i);
        }
      }

      auto tA_bsum = reduce<1>(tSrS, sycl::plus<void>{});
      CUTE_UNROLL
      for (int i = 0; i < tA_sum.size(); i++) {
        tA_sum(i) += tA_bsum(i);
      }
      return rescale;
    };

    auto tArP = thr_mma_pv.partition_sg_fragment_A(proxyP);
    auto tVrV = thr_copy_v.partition_sg_fragment_D(gV_split(_, _, 0, 0));
    auto tArV = thr_mma_pv.partition_sg_fragment_B(gV_split(_, _, 0, 0));

    auto pv_gemm_one_tile = [&](int block_idx, int local_v_tile_idx) {
      copy(tiled_copy_v, tVgV(_, _, _, local_v_tile_idx, block_idx), tVrV);
      reorder(tVrV, tArV);
      cute::gemm(mma_pv, tArP, tArV, tArA(_, _, _, local_v_tile_idx));
    };

    const int num_topk_blocks = ceil_div(params.shape.gathered_topk, Traits::B_TOPK);
    CUTE_NO_UNROLL
    for (int topk_idx = 0; topk_idx < num_topk_blocks; ++topk_idx) {
      clear(tSrS);
      CUTE_UNROLL
      for (int i = 0; i < (D_QK / get<2>(TileShapeQK{})); ++i) {
        qk_gemm_one_tile(topk_idx, i);
      }

      mask_rS(topk_idx);
      auto rescale = online_softmax_and_rescale_o(topk_idx);
      reorder(tSrS, tArP);

      if (topk_idx > 0) {
        CUTE_UNROLL
        for (int i = 0; i < tArA.size(); i++) {
          tArA(i) *= broadcast<0>(rescale, tArA, i);
        }
      }

      CUTE_UNROLL
      for (int i = 0; i < V_TILE_PER_SPLIT; ++i) {
        pv_gemm_one_tile(topk_idx, i);
      }
    }
  }
};
/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::flash_attention::collective
