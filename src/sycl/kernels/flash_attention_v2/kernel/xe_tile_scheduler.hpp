/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
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

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/kernel_hardware_info.h"

namespace cutlass::fmha::kernel {

struct XeFHMAIndividualTileScheduler {
  struct Params {
    dim3 grid;
    FastDivmod divmod_num_heads;
    FastDivmod divmod_batch;
    int num_kv_splits_ = -1;
    int const* cu_seqlens_q = nullptr;
    int batch = 0;
    int tile_q = 0;
    bool compact_varlen = false;
  };

  bool valid_ = true;
  Params params;
  int idx_b_ = 0;
  int blk_q_ = 0;

  CUTLASS_DEVICE
  XeFHMAIndividualTileScheduler(Params const& params) : params(params) {
    if (params.compact_varlen) {
      auto subgroup = sycl::ext::oneapi::this_work_item::get_sub_group();
      int mapped_batch = -1;
      int mapped_q_tile = 0;
      if (subgroup.get_local_linear_id() == 0) {
        int tile_ordinal = BlockIdxY();
        for (int batch = 0; batch < params.batch; ++batch) {
          int const seq_len = params.cu_seqlens_q[batch + 1] - params.cu_seqlens_q[batch];
          int const num_tiles = cute::ceil_div(seq_len, params.tile_q);
          if (tile_ordinal < num_tiles) {
            mapped_batch = batch;
            mapped_q_tile = tile_ordinal;
            break;
          }
          tile_ordinal -= num_tiles;
        }
      }
      idx_b_ = sycl::group_broadcast(subgroup, mapped_batch, 0);
      blk_q_ = sycl::group_broadcast(subgroup, mapped_q_tile, 0);
      valid_ = idx_b_ >= 0;
    }
  }

  template <class ProblemShape, class TileShape>
  static Params to_underlying_arguments(
      ProblemShape const& shape,
      KernelHardwareInfo hw_info,
      TileShape const& tile_shape,
      const int& num_kv_splits = -1) {
    using namespace cute;

    dim3 grid(
        size(ceil_div(shape.head_size_vo, get<1>(tile_shape))),  // V
        size(ceil_div(shape.seq_len_qo, get<0>(tile_shape))),    // Q
        size(shape.batch * shape.num_heads_q));                  // (h,b) -- split later
    int num_head = shape.num_heads_q;
    if (num_kv_splits >= 1) {
      // for splitKV, each wg handles group query heads
      grid.z = size(shape.batch * shape.num_heads_kv);
      grid.z *= num_kv_splits;
      num_head = shape.num_heads_kv;
    }
    Params params{grid, {num_head}, {shape.batch * num_head}, num_kv_splits};
    if constexpr (ProblemShape::is_var_len) {
      bool const is_ragged =
          shape.seq_len_qo.cumulative_length != nullptr &&
          shape.seq_len_qo.total_length != shape.batch * shape.seq_len_qo.max_length;
      bool const is_sparse_ragged =
          is_ragged &&
          static_cast<int64_t>(shape.seq_len_qo.total_length) * 5 <=
              static_cast<int64_t>(shape.batch) * shape.seq_len_qo.max_length;
      if (num_kv_splits < 1 && is_sparse_ragged) {
        int const tile_q = get<0>(tile_shape);
        // sum(ceil(q_i / tile_q)) <= ceil(total_q / tile_q) + batch - 1.
        grid.y = ceil_div(shape.seq_len_qo.total_length, tile_q) + shape.batch - 1;
        grid.z = shape.num_heads_q;
        params.grid = grid;
        params.cu_seqlens_q = shape.seq_len_qo.cumulative_length;
        params.batch = shape.batch;
        params.tile_q = tile_q;
        params.compact_varlen = true;
      }
    }
    return params;
  }

  template <int Num_SGs>
  static dim3 get_grid_shape(Params const& params) {
    return params.grid;
  }

  CUTLASS_DEVICE
  bool is_valid() {
    return valid_;
  }

  CUTLASS_DEVICE
  auto get_block_coord() {
    using namespace cute;
    if (params.compact_varlen) {
      return make_coord(unsigned(blk_q_), BlockIdxX(), int(BlockIdxZ()), idx_b_, (int)-1);
    }

    int idx_kv_split = BlockIdxZ();
    int head, idx_b;

    if (params.num_kv_splits_ >= 1) {
      params.divmod_batch(idx_kv_split, idx_b, idx_kv_split);
      params.divmod_num_heads(idx_b, head, idx_b);
      return make_coord(BlockIdxY(), BlockIdxX(), head, idx_b, idx_kv_split);
    }

    idx_b = idx_kv_split;
    params.divmod_num_heads(idx_b, head, idx_b);
    return make_coord(BlockIdxY(), BlockIdxX(), head, idx_b, (int)-1);
  }

  CUTLASS_DEVICE
  XeFHMAIndividualTileScheduler& operator++() {
    valid_ = false;
    return *this;
  }
};

struct XeFHMAIndividualPersistentTileScheduler {
  struct Params {
    dim3 grid;
    FastDivmod divmod_num_heads;
  };

  bool valid_ = true;
  Params params;
  int kv_tile_size_;
  // num of kv blocks for each head
  int local_num_kv_blocks_;
  int num_batch_heads_;

  CUTLASS_DEVICE
  XeFHMAIndividualPersistentTileScheduler(
      Params const& params, int kv_tile_size, int local_num_kv_blocks, int num_batch_heads)
      : params(params),
        kv_tile_size_(kv_tile_size),
        local_num_kv_blocks_(local_num_kv_blocks),
        num_batch_heads_(num_batch_heads) {}

  template <class ProblemShape, class TileShape>
  static Params
  to_underlying_arguments(ProblemShape const& shape, KernelHardwareInfo hw_info, TileShape const& tile_shape) {
    using namespace cute;

    dim3 grid(
        size(ceil_div(shape.head_size_vo, get<1>(tile_shape))),  // V
        size(ceil_div(shape.seq_len_qo, get<0>(tile_shape))),    // Q
        size(shape.batch * shape.num_heads_q));                  // (h,b) -- split later
    int num_heads = shape.num_heads_q;
    grid.z = hw_info.sm_count;

    return Params{grid, {num_heads}};
  }

  template <int Num_SGs>
  static dim3 get_grid_shape(Params const& params) {
    return params.grid;
  }

  CUTLASS_DEVICE
  bool is_valid() {
    return valid_;
  }

  CUTLASS_DEVICE
  auto get_block_coord() {
    using namespace cute;
    int wg_id = BlockIdxZ();

    // total number of blocks need to be processed across all wgs
    int total_num_kv_blocks = local_num_kv_blocks_ * num_batch_heads_;
    // guarantee all wg process similar number of blocks of KV (load balance)
    int num_blocks_per_wg = cute::ceil_div(total_num_kv_blocks, GridDimZ());

    // compute start batch head id for current wg
    int start_batch_head_id = wg_id * num_blocks_per_wg / local_num_kv_blocks_;

    return make_coord(BlockIdxY(), BlockIdxX(), start_batch_head_id);
  }

  CUTLASS_DEVICE
  XeFHMAIndividualPersistentTileScheduler& operator++() {
    valid_ = false;
    return *this;
  }
};

struct XeReduceSplitKTileScheduler {
  struct Params {
    dim3 grid;
    FastDivmod divmod_num_heads;
    int num_kv_splits = -1;
  };

  bool valid_ = true;
  Params params;

  CUTLASS_DEVICE
  XeReduceSplitKTileScheduler(Params const& params) : params(params) {}

  template <class ProblemShape, class TileShape>
  static Params to_underlying_arguments(
      ProblemShape const& shape,
      KernelHardwareInfo hw_info,
      TileShape const& tile_shape,
      const int& num_kv_splits = -1) {
    using namespace cute;

    dim3 grid(shape.seq_len_qo, shape.num_heads_q, shape.batch);
    return Params{grid, {shape.num_heads_q}, num_kv_splits};
  }

  template <int Num_SGs>
  static dim3 get_grid_shape(Params const& params) {
    return params.grid;
  }

  CUTLASS_DEVICE
  bool is_valid() {
    return valid_;
  }

  CUTLASS_DEVICE
  auto get_block_coord() {
    using namespace cute;

    return make_coord(BlockIdxX(), BlockIdxY(), BlockIdxZ());
  }

  CUTLASS_DEVICE
  XeReduceSplitKTileScheduler& operator++() {
    valid_ = false;
    return *this;
  }
};
}  // namespace cutlass::fmha::kernel
