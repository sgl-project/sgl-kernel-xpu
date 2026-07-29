/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
/*! \file
    \brief Two-stage sparse MLA tile schedulers for DeepSeek V4 (Stage 1 + Stage 2).

    One scheduler per stage, each mapping its own launch grid to work-tiles so that
    neither kernel body touches BlockIdx directly. Both mirror the sycl-tla
    convention used by kernel/mla_sparse_tile_scheduler.hpp and are stateless
    single-tile decoders (no persistent-CTA loop): constructed once, yield one tile,
    then terminate — the launcher owns grid computation and passes the flat params
    straight through.

      - XeMlaSparseGather2StageTileScheduler<B_TOPK> (Stage 1, gather). Grid is
        (b * s_q, ceil_div(gathered_topk, B_TOPK), 1): BlockIdxX enumerates the
        (batch, seq) pairs row-major and BlockIdxY the topk-block, decoded into a
        (batch_idx, seq_idx, topk_block_idx) coordinate. Unlike the Stage-2 scheduler
        it also owns get_grid_shape, since the gather kernel's grid is derivable from
        its params alone. Its Params slice is the *base* Gather2StageParams, so the
        one scheduler serves both the decode and prefill param children.

      - XeMlaSparse2StageIndividualTileScheduler<B_H> (Stage 2, dense flash).
        Grid is (ceil_div(h_q, B_H) * s_q * b, V_SPLIT, 1) (see
        launch_sparse_mla_decode_fp8_fwd_kernel_policy): BlockIdxX enumerates the
        (batch, seq, head-block) tuples row-major and BlockIdxY the V-split, decoded
        into a (batch_idx, seq_idx, head_bid, v_split_idx) coordinate — the exact
        index math the Stage-2 kernel carried before it was decomposed into
        collectives + scheduler, factored out here.

    Neither name carries "Decode": both stages' schedulers, like the Stage-2 kernel and
    its collectives, are shared verbatim by the decode and prefill paths.
*/

#pragma once

#include "sycl/kernels/mla_sparse/device/xe_mla_sparse_2stage_common.hpp"

namespace cutlass::flash_attention::kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////
// Stage 1 (gather).
/////////////////////////////////////////////////////////////////////////////////////////////////

// A decoded Stage 1 work-tile coordinate.
struct SparseGather2StageWorkTile {
  int batch_idx;
  int seq_idx;
  int topk_block_idx;
};

template <int B_TOPK_>
class XeMlaSparseGather2StageTileScheduler {
 public:
  static constexpr int B_TOPK = B_TOPK_;

  // The scheduler's own param slice: the dims that enumerate the grid. This is the
  // *base* gather params, so one scheduler serves both the decode and prefill children.
  using Params = Gather2StageParams;

  // One work-group per (batch*seq, topk-block); B_TOPK topk columns per work-group.
  // The gather grid follows from its params alone, so the scheduler owns it (the
  // Stage-2 scheduler cannot: its grid also depends on the config's V_SPLIT).
  static dim3 get_grid_shape(Params const& params) {
    return dim3(params.b * params.s_q, ceil_div(params.gathered_topk, B_TOPK), 1);
  }

  CUTLASS_DEVICE
  XeMlaSparseGather2StageTileScheduler(Params const& params) : valid_(true) {
    const int seq_linear_idx = int(BlockIdxX());
    tile_.batch_idx = seq_linear_idx / params.s_q;
    tile_.seq_idx = seq_linear_idx - tile_.batch_idx * params.s_q;
    tile_.topk_block_idx = int(BlockIdxY());
  }

  CUTLASS_DEVICE
  bool is_valid() const {
    return valid_;
  }

  CUTLASS_DEVICE
  SparseGather2StageWorkTile get_block_coord() const {
    return tile_;
  }

  CUTLASS_DEVICE
  XeMlaSparseGather2StageTileScheduler& operator++() {
    valid_ = false;
    return *this;
  }

 private:
  SparseGather2StageWorkTile tile_;
  bool valid_;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
// Stage 2 (dense flash decode).
/////////////////////////////////////////////////////////////////////////////////////////////////

// A decoded Stage 2 work-tile coordinate.
struct Sparse2StageWorkTile {
  int batch_idx;
  int seq_idx;
  int head_bid;
  int v_split_idx;
};

template <int B_H_>
class XeMlaSparse2StageIndividualTileScheduler {
 public:
  static constexpr int B_H = B_H_;

  // The scheduler's own param slice: the two dims needed to enumerate head-blocks
  // per query tile. Built by the host adapter as the composite's `scheduler` member
  // and forwarded by the dense kernel (params.scheduler).
  using Params = TileScheduler2StageParams;

  CUTLASS_DEVICE
  XeMlaSparse2StageIndividualTileScheduler(Params const& params) : valid_(true) {
    const int num_head_blocks = ceil_div(params.h_q, B_H);
    const int wg_id = int(BlockIdxX());
    const int q_tile_idx = wg_id / num_head_blocks;

    tile_.batch_idx = q_tile_idx / params.s_q;
    tile_.seq_idx = q_tile_idx - tile_.batch_idx * params.s_q;
    tile_.head_bid = wg_id % num_head_blocks;
    tile_.v_split_idx = int(BlockIdxY());
  }

  CUTLASS_DEVICE
  bool is_valid() const {
    return valid_;
  }

  CUTLASS_DEVICE
  Sparse2StageWorkTile get_block_coord() const {
    return tile_;
  }

  CUTLASS_DEVICE
  XeMlaSparse2StageIndividualTileScheduler& operator++() {
    valid_ = false;
    return *this;
  }

 private:
  Sparse2StageWorkTile tile_;
  bool valid_;
};

}  // namespace cutlass::flash_attention::kernel
