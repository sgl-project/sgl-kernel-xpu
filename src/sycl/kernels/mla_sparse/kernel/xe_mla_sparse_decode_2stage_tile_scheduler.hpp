/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
/*! \file
    \brief Two-stage sparse MLA decode Stage 2 tile scheduler for DeepSeek V4.

    Maps the launch grid to work-tiles. The grid is
      (ceil_div(h_q, B_H) * s_q * b, V_SPLIT, 1)
    (see launch_sparse_mla_decode_fp8_fwd_kernel_policy). BlockIdxX enumerates the
    (batch, seq, head-block) tuples row-major and BlockIdxY the V-split; this
    scheduler decodes those linear indices into a (batch_idx, seq_idx, head_bid,
    v_split_idx) work-tile coordinate — the exact index math from the monolithic
    DenseDecodeFwdKernel, factored out to mirror the sycl-tla convention used by
    kernel/mla_sparse_tile_scheduler.hpp.

    Since the launcher owns grid computation and passes the flat SparseAttnDecodeParams
    straight through, this scheduler is a stateless single-tile decoder (no
    persistent-CTA loop): it is constructed once, yields one tile, and terminates.
*/

#pragma once

#include "sycl/kernels/mla_sparse/kernel/xe_mla_sparse_decode_2stage_common.hpp"

namespace cutlass::flash_attention::kernel {

// A decoded Stage 2 work-tile coordinate.
struct SparseDecode2StageWorkTile {
  int batch_idx;
  int seq_idx;
  int head_bid;
  int v_split_idx;
};

template <int B_H_>
class XeMlaSparseDecode2StageIndividualTileScheduler {
 public:
  static constexpr int B_H = B_H_;

  CUTLASS_DEVICE
  XeMlaSparseDecode2StageIndividualTileScheduler(int h_q, int s_q) : valid_(true) {
    const int num_head_blocks = ceil_div(h_q, B_H);
    const int wg_id = int(BlockIdxX());
    const int q_tile_idx = wg_id / num_head_blocks;

    tile_.batch_idx = q_tile_idx / s_q;
    tile_.seq_idx = q_tile_idx - tile_.batch_idx * s_q;
    tile_.head_bid = wg_id % num_head_blocks;
    tile_.v_split_idx = int(BlockIdxY());
  }

  CUTLASS_DEVICE
  bool is_valid() const {
    return valid_;
  }

  CUTLASS_DEVICE
  SparseDecode2StageWorkTile get_block_coord() const {
    return tile_;
  }

  CUTLASS_DEVICE
  XeMlaSparseDecode2StageIndividualTileScheduler& operator++() {
    valid_ = false;
    return *this;
  }

 private:
  SparseDecode2StageWorkTile tile_;
  bool valid_;
};

}  // namespace cutlass::flash_attention::kernel
