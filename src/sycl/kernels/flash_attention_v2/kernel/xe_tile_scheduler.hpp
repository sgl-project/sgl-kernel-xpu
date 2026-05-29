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
  };

  bool valid_ = true;
  Params params;

  CUTLASS_DEVICE
  XeFHMAIndividualTileScheduler(Params const& params) : params(params) {}

  template <int Num_SGs = 16, class ProblemShape, class TileShape>
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
    return Params{grid, {num_head}, {shape.batch * num_head}, num_kv_splits};
  }

  template <int Num_SGs>
  static dim3 get_grid_shape(Params const& params) {
    return params.grid;
  }

  CUTLASS_DEVICE
  bool is_valid() {
    return valid_;
  }

  template <class Varlen = cutlass::fmha::collective::VariableLength>
  CUTLASS_DEVICE auto get_block_coord() {
    using namespace cute;
    int idx_b = BlockIdxZ();
    int head;
    params.divmod_num_heads(idx_b, head, idx_b);
    return make_coord(BlockIdxY(), BlockIdxX(), head, idx_b);
  }

  CUTLASS_DEVICE
  auto get_block_coord() {
    using namespace cute;
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

  template <int Num_SGs = 16, class ProblemShape, class TileShape>
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

  template <class Varlen = cutlass::fmha::collective::VariableLength>
  CUTLASS_DEVICE auto get_block_coord() {
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

template <class Varlen, class Scheduler>
CUTLASS_DEVICE void xe_fmha_persistent_decompose_work_idx(
    Scheduler const& scheduler, int work_idx, int& blk_q, int& blk_v, int& head, int& idx_b) {
  auto const& params = scheduler.params;

  if constexpr (cutlass::fmha::collective::is_variable_length_v<Varlen>) {
    int qh_idx = params.divmod_num_v_blocks.divmod(blk_v, work_idx);
    idx_b = 0;
    int q_blocks = 0;

    if (params.cumulative_q_blocks) {
      int lo = 0;
      int hi = params.batch;
      while (lo + 1 < hi) {
        int mid = (lo + hi) / 2;
        int mid_work_tiles = params.cumulative_q_blocks[mid] * params.num_heads_q;
        if (qh_idx >= mid_work_tiles) {
          lo = mid;
        } else {
          hi = mid;
        }
      }

      idx_b = lo;
      qh_idx -= params.cumulative_q_blocks[idx_b] * params.num_heads_q;
      q_blocks = params.cumulative_q_blocks[idx_b + 1] - params.cumulative_q_blocks[idx_b];
    } else {
      q_blocks = scheduler.template q_blocks_for_batch<Varlen>(idx_b);
      int batch_work_tiles = q_blocks * params.num_heads_q;

      CUTLASS_PRAGMA_NO_UNROLL
      while (qh_idx >= batch_work_tiles && idx_b + 1 < params.batch) {
        qh_idx -= batch_work_tiles;
        ++idx_b;
        q_blocks = scheduler.template q_blocks_for_batch<Varlen>(idx_b);
        batch_work_tiles = q_blocks * params.num_heads_q;
      }
    }

    head = qh_idx / q_blocks;
    blk_q = qh_idx - head * q_blocks;
  } else {
    int qhb_idx = params.divmod_num_v_blocks.divmod(blk_v, work_idx);
    int hb_idx = params.divmod_max_q_blocks.divmod(blk_q, qhb_idx);
    params.divmod_num_heads(idx_b, head, hb_idx);
  }
}

template <class Varlen, class Scheduler>
CUTLASS_DEVICE void xe_fmha_split_kv_persistent_decompose_work_idx(
    Scheduler const& scheduler, int work_idx, int& blk_q, int& blk_v, int& head, int& idx_b, int& blk_k) {
  auto const& params = scheduler.params;

  if constexpr (cutlass::fmha::collective::is_variable_length_v<Varlen>) {
    if (params.cumulative_split_kv_blocks && params.cumulative_q_blocks && params.cumulative_k_blocks) {
      int qk_head_idx = params.divmod_num_v_blocks.divmod(blk_v, work_idx);

      int lo = 0;
      int hi = params.batch;
      while (lo + 1 < hi) {
        int mid = (lo + hi) / 2;
        int mid_work_tiles = params.cumulative_split_kv_blocks[mid] * params.num_heads_q;
        if (qk_head_idx >= mid_work_tiles) {
          lo = mid;
        } else {
          hi = mid;
        }
      }

      idx_b = lo;
      qk_head_idx -= params.cumulative_split_kv_blocks[idx_b] * params.num_heads_q;
      int q_blocks = params.cumulative_q_blocks[idx_b + 1] - params.cumulative_q_blocks[idx_b];
      int k_splits = params.cumulative_k_blocks[idx_b + 1] - params.cumulative_k_blocks[idx_b];
      int qk_blocks = q_blocks * k_splits;

      head = qk_head_idx / qk_blocks;
      int qk_idx = qk_head_idx - head * qk_blocks;
      blk_q = qk_idx / k_splits;
      blk_k = qk_idx - blk_q * k_splits;
      return;
    }
  }

  work_idx = params.divmod_num_kv_splits.divmod(blk_k, work_idx);
  xe_fmha_persistent_decompose_work_idx<Varlen>(scheduler, work_idx, blk_q, blk_v, head, idx_b);
}

struct XeFHMAStaticPresistentTileScheduler {
  struct Params {
    dim3 grid;
    int batch;
    int num_heads_q;
    int num_v_blocks;
    int max_q_blocks;
    int q_tile_size;
    int const* cumulative_seqlen_q;
    int const* cumulative_q_blocks;
    int total_q_blocks;
    FastDivmod divmod_num_v_blocks;
    FastDivmod divmod_max_q_blocks;
    FastDivmod divmod_num_heads;
  };

  Params params;
  uint64_t current_work_linear_idx_ = 0;
  uint64_t total_grid_size_ = 1;

  CUTLASS_DEVICE
  XeFHMAStaticPresistentTileScheduler(Params const& params) : params(params) {
    current_work_linear_idx_ = uint64_t(BlockIdxX()) + uint64_t(BlockIdxY()) * uint64_t(GridDimX()) +
                               uint64_t(BlockIdxZ()) * uint64_t(GridDimX()) * uint64_t(GridDimY());
    total_grid_size_ = uint64_t(GridDimX()) * uint64_t(GridDimY()) * uint64_t(GridDimZ());
  }

  template <int Num_SGs = 16, class ProblemShape, class TileShape>
  static Params
  to_underlying_arguments(ProblemShape const& shape, KernelHardwareInfo hw_info, TileShape const& tile_shape) {
    using namespace cute;
    using SeqLenQ = remove_cvref_t<decltype(shape.seq_len_qo)>;

    int const num_v_blocks = size(ceil_div(shape.head_size_vo, get<1>(tile_shape)));
    int const q_tile_size = get<0>(tile_shape);
    int max_seq_len_q = 0;
    int const* cumulative_seqlen_q = nullptr;
    int const* cumulative_q_blocks = nullptr;
    int total_q_blocks = 0;

    if constexpr (cutlass::fmha::collective::is_variable_length_v<SeqLenQ>) {
      max_seq_len_q = shape.seq_len_qo.max_length;
      cumulative_seqlen_q = shape.seq_len_qo.cumulative_length;
      cumulative_q_blocks = shape.seq_len_qo.cumulative_blocks;
      total_q_blocks = shape.seq_len_qo.total_blocks;
    } else {
      max_seq_len_q = shape.seq_len_qo;
      total_q_blocks = cute::ceil_div(max_seq_len_q, q_tile_size) * shape.batch;
    }

    int const max_q_blocks = size(ceil_div(max_seq_len_q, q_tile_size));
    int const work_q_blocks = total_q_blocks > 0 ? total_q_blocks : shape.batch * max_q_blocks;
    int const max_work_tiles = shape.num_heads_q * num_v_blocks * work_q_blocks;
    int const grid_x = cute::min(max_work_tiles, hw_info.sm_count * 8);
    dim3 grid(grid_x, 1, 1);

    total_q_blocks = total_q_blocks > 0 ? total_q_blocks : shape.batch * max_q_blocks;

    return Params{
        grid,
        shape.batch,
        shape.num_heads_q,
        num_v_blocks,
        max_q_blocks,
        q_tile_size,
        cumulative_seqlen_q,
        cumulative_q_blocks,
        total_q_blocks,
        {num_v_blocks},
        {max_q_blocks},
        {shape.num_heads_q}};
  }

  template <int Num_SGs>
  static dim3 get_grid_shape(Params const& params) {
    return params.grid;
  }

  template <class Varlen = cutlass::fmha::collective::VariableLength>
  CUTLASS_DEVICE int q_blocks_for_batch(int batch_idx) const {
    if constexpr (cutlass::fmha::collective::is_variable_length_v<Varlen>) {
      int seq_len_q = params.cumulative_seqlen_q[batch_idx + 1] - params.cumulative_seqlen_q[batch_idx];
      return cute::ceil_div(seq_len_q, params.q_tile_size);
    }
    return params.max_q_blocks;
  }

  CUTLASS_DEVICE
  int total_work_tiles() const {
    return params.total_q_blocks * params.num_heads_q * params.num_v_blocks;
  }

  CUTLASS_DEVICE
  bool is_valid() {
    return current_work_linear_idx_ < uint64_t(total_work_tiles());
  }

  template <class Varlen = cutlass::fmha::collective::VariableLength>
  CUTLASS_DEVICE auto get_block_coord() {
    using namespace cute;
    int work_idx = static_cast<int>(current_work_linear_idx_);
    int blk_q;
    int blk_v;
    int head;
    int idx_b;
    xe_fmha_persistent_decompose_work_idx<Varlen>(*this, work_idx, blk_q, blk_v, head, idx_b);

    return make_coord(blk_q, blk_v, head, idx_b);
  }

  CUTLASS_DEVICE
  XeFHMAStaticPresistentTileScheduler& operator++() {
    current_work_linear_idx_ += total_grid_size_;
    return *this;
  }
};

struct XeFHMADynamicPresistentTileScheduler {
  struct Params {
    dim3 grid;
    int batch;
    int num_heads_q;
    int num_v_blocks;
    int max_q_blocks;
    int q_tile_size;
    int const* cumulative_seqlen_q;
    int const* cumulative_q_blocks;
    int total_q_blocks;
    int* tile_counter;
    FastDivmod divmod_num_v_blocks;
    FastDivmod divmod_max_q_blocks;
    FastDivmod divmod_num_heads;
  };

  Params params;
  uint64_t current_work_linear_idx_ = 0;
  int total_work_tiles_ = 0;

  CUTLASS_DEVICE
  XeFHMADynamicPresistentTileScheduler(Params const& params) : params(params) {
    // Derive the exact total number of q blocks on device from the cumulative
    // prefix-sum buffer (cumulative_q_blocks[batch]). This avoids a per-call
    // blocking device-to-host copy on the host side just to read this scalar.
    int total_q_blocks =
        params.cumulative_q_blocks != nullptr ? params.cumulative_q_blocks[params.batch] : params.total_q_blocks;
    total_work_tiles_ = total_q_blocks * params.num_heads_q * params.num_v_blocks;
    current_work_linear_idx_ = fetch_next_work_tile();
  }

  template <int Num_SGs = 16, class ProblemShape, class TileShape>
  static Params
  to_underlying_arguments(ProblemShape const& shape, KernelHardwareInfo hw_info, TileShape const& tile_shape) {
    using namespace cute;
    using SeqLenQ = remove_cvref_t<decltype(shape.seq_len_qo)>;

    int const num_v_blocks = size(ceil_div(shape.head_size_vo, get<1>(tile_shape)));
    int const q_tile_size = get<0>(tile_shape);
    int max_seq_len_q = 0;
    int const* cumulative_seqlen_q = nullptr;
    int const* cumulative_q_blocks = nullptr;
    int total_q_blocks = 0;

    if constexpr (cutlass::fmha::collective::is_variable_length_v<SeqLenQ>) {
      max_seq_len_q = shape.seq_len_qo.max_length;
      cumulative_seqlen_q = shape.seq_len_qo.cumulative_length;
      cumulative_q_blocks = shape.seq_len_qo.cumulative_blocks;
      total_q_blocks = shape.seq_len_qo.total_blocks;
    } else {
      max_seq_len_q = shape.seq_len_qo;
      total_q_blocks = cute::ceil_div(max_seq_len_q, q_tile_size) * shape.batch;
    }

    int const max_q_blocks = size(ceil_div(max_seq_len_q, q_tile_size));
    int const work_q_blocks = total_q_blocks > 0 ? total_q_blocks : shape.batch * max_q_blocks;
    int const max_work_tiles = shape.num_heads_q * num_v_blocks * work_q_blocks;
    int const grid_x = cute::min(max_work_tiles, hw_info.sm_count * 8);
    dim3 grid(grid_x, 1, 1);

    total_q_blocks = total_q_blocks > 0 ? total_q_blocks : shape.batch * max_q_blocks;

    return Params{
        grid,
        shape.batch,
        shape.num_heads_q,
        num_v_blocks,
        max_q_blocks,
        q_tile_size,
        cumulative_seqlen_q,
        cumulative_q_blocks,
        total_q_blocks,
        nullptr,
        {num_v_blocks},
        {max_q_blocks},
        {shape.num_heads_q}};
  }

  template <int Num_SGs>
  static dim3 get_grid_shape(Params const& params) {
    return params.grid;
  }

  template <class Varlen = cutlass::fmha::collective::VariableLength>
  CUTLASS_DEVICE int q_blocks_for_batch(int batch_idx) const {
    if constexpr (cutlass::fmha::collective::is_variable_length_v<Varlen>) {
      int seq_len_q = params.cumulative_seqlen_q[batch_idx + 1] - params.cumulative_seqlen_q[batch_idx];
      return cute::ceil_div(seq_len_q, params.q_tile_size);
    }
    return params.max_q_blocks;
  }

  CUTLASS_DEVICE
  int total_work_tiles() const {
    return total_work_tiles_;
  }

  CUTLASS_DEVICE
  bool is_valid() {
    return current_work_linear_idx_ < uint64_t(total_work_tiles());
  }

  template <class Varlen = cutlass::fmha::collective::VariableLength>
  CUTLASS_DEVICE auto get_block_coord() {
    using namespace cute;
    int work_idx = static_cast<int>(current_work_linear_idx_);
    int blk_q;
    int blk_v;
    int head;
    int idx_b;
    xe_fmha_persistent_decompose_work_idx<Varlen>(*this, work_idx, blk_q, blk_v, head, idx_b);

    return make_coord(blk_q, blk_v, head, idx_b);
  }

  CUTLASS_DEVICE
  uint64_t fetch_next_work_tile() const {
    int work_idx = 0;
    if (ThreadIdxX() == 0) {
      work_idx = atomicAdd(params.tile_counter, 1);
    }
    work_idx = sycl::group_broadcast(sycl::ext::oneapi::this_work_item::get_work_group<3>(), work_idx, 0);
    return uint64_t(work_idx);
  }

  CUTLASS_DEVICE
  XeFHMADynamicPresistentTileScheduler& operator++() {
    current_work_linear_idx_ = fetch_next_work_tile();
    return *this;
  }
};

struct XeFMHASplitKVStaticPersistentTileScheduler {
  struct Params {
    dim3 grid;
    int batch;
    int num_heads_q;
    int num_v_blocks;
    int max_q_blocks;
    int q_tile_size;
    int const* cumulative_seqlen_q;
    int const* cumulative_q_blocks;
    int total_q_blocks;
    int const* cumulative_k_blocks;
    int total_k_blocks;
    int const* cumulative_split_kv_blocks;
    int total_split_kv_blocks;
    int num_kv_splits_;
    FastDivmod divmod_num_v_blocks;
    FastDivmod divmod_max_q_blocks;
    FastDivmod divmod_num_heads;
    FastDivmod divmod_num_kv_splits;
  };

  Params params;
  uint64_t current_work_linear_idx_ = 0;
  uint64_t total_grid_size_ = 1;

  CUTLASS_DEVICE
  XeFMHASplitKVStaticPersistentTileScheduler(Params const& params) : params(params) {
    current_work_linear_idx_ = uint64_t(BlockIdxX()) + uint64_t(BlockIdxY()) * uint64_t(GridDimX()) +
                               uint64_t(BlockIdxZ()) * uint64_t(GridDimX()) * uint64_t(GridDimY());
    total_grid_size_ = uint64_t(GridDimX()) * uint64_t(GridDimY()) * uint64_t(GridDimZ());
  }

  template <int Num_SGs = 16, class ProblemShape, class TileShape>
  static Params to_underlying_arguments(
      ProblemShape const& shape, KernelHardwareInfo hw_info, TileShape const& tile_shape, int num_kv_splits = 1) {
    using namespace cute;
    using SeqLenQ = remove_cvref_t<decltype(shape.seq_len_qo)>;
    using SeqLenKV = remove_cvref_t<decltype(shape.seq_len_kv)>;

    int const num_v_blocks = size(ceil_div(shape.head_size_vo, get<1>(tile_shape)));
    int const q_tile_size = get<0>(tile_shape);
    int max_seq_len_q = 0;
    int const* cumulative_seqlen_q = nullptr;
    int const* cumulative_q_blocks = nullptr;
    int total_q_blocks = 0;
    int const* cumulative_k_blocks = nullptr;
    int total_k_blocks = 0;
    int const* cumulative_split_kv_blocks = nullptr;
    int total_split_kv_blocks = 0;

    if constexpr (cutlass::fmha::collective::is_variable_length_v<SeqLenQ>) {
      max_seq_len_q = shape.seq_len_qo.max_length;
      cumulative_seqlen_q = shape.seq_len_qo.cumulative_length;
      cumulative_q_blocks = shape.seq_len_qo.cumulative_blocks;
      total_q_blocks = shape.seq_len_qo.total_blocks;
    } else {
      max_seq_len_q = shape.seq_len_qo;
      total_q_blocks = cute::ceil_div(max_seq_len_q, q_tile_size) * shape.batch;
    }

    if constexpr (cutlass::fmha::collective::is_variable_length_v<SeqLenKV>) {
      cumulative_k_blocks = shape.seq_len_kv.cumulative_blocks;
      total_k_blocks = shape.seq_len_kv.total_blocks;
      cumulative_split_kv_blocks = shape.seq_len_kv.cumulative_split_kv_blocks;
      total_split_kv_blocks = shape.seq_len_kv.total_split_kv_blocks;
    } else {
      total_k_blocks = shape.batch * num_kv_splits;
    }

    int const max_q_blocks = size(ceil_div(max_seq_len_q, q_tile_size));
    int const work_q_blocks = total_q_blocks > 0 ? total_q_blocks : shape.batch * max_q_blocks;
    int const work_split_kv_blocks = total_split_kv_blocks > 0 ? total_split_kv_blocks : work_q_blocks * num_kv_splits;
    int const max_work_tiles = shape.num_heads_q * num_v_blocks * work_split_kv_blocks;
    int const grid_x = cute::min(max_work_tiles, hw_info.sm_count * 8);
    dim3 grid(grid_x, 1, 1);

    total_q_blocks = work_q_blocks;

    return Params{
        grid,
        shape.batch,
        shape.num_heads_q,
        num_v_blocks,
        max_q_blocks,
        q_tile_size,
        cumulative_seqlen_q,
        cumulative_q_blocks,
        total_q_blocks,
        cumulative_k_blocks,
        total_k_blocks,
        cumulative_split_kv_blocks,
        total_split_kv_blocks,
        num_kv_splits,
        {num_v_blocks},
        {max_q_blocks},
        {shape.num_heads_q},
        {num_kv_splits}};
  }

  template <int Num_SGs>
  static dim3 get_grid_shape(Params const& params) {
    return params.grid;
  }

  template <class Varlen = cutlass::fmha::collective::VariableLength>
  CUTLASS_DEVICE int q_blocks_for_batch(int batch_idx) const {
    if constexpr (cutlass::fmha::collective::is_variable_length_v<Varlen>) {
      int seq_len_q = params.cumulative_seqlen_q[batch_idx + 1] - params.cumulative_seqlen_q[batch_idx];
      return cute::ceil_div(seq_len_q, params.q_tile_size);
    }
    return params.max_q_blocks;
  }

  CUTLASS_DEVICE
  int total_work_tiles() const {
    int split_kv_blocks =
        params.total_split_kv_blocks > 0 ? params.total_split_kv_blocks : params.total_q_blocks * params.num_kv_splits_;
    return split_kv_blocks * params.num_heads_q * params.num_v_blocks;
  }

  CUTLASS_DEVICE
  bool is_valid() {
    return current_work_linear_idx_ < uint64_t(total_work_tiles());
  }

  template <class Varlen = cutlass::fmha::collective::VariableLength>
  CUTLASS_DEVICE auto get_block_coord() {
    using namespace cute;
    int work_idx = static_cast<int>(current_work_linear_idx_);
    int blk_q;
    int blk_v;
    int head;
    int idx_b;
    int blk_k;
    xe_fmha_split_kv_persistent_decompose_work_idx<Varlen>(*this, work_idx, blk_q, blk_v, head, idx_b, blk_k);

    return make_coord(blk_q, blk_v, head, idx_b, blk_k);
  }

  CUTLASS_DEVICE
  XeFMHASplitKVStaticPersistentTileScheduler& operator++() {
    current_work_linear_idx_ += total_grid_size_;
    return *this;
  }
};

struct XeFMHASplitKVDynamicPersistentTileScheduler {
  struct Params {
    dim3 grid;
    int batch;
    int num_heads_q;
    int num_v_blocks;
    int max_q_blocks;
    int q_tile_size;
    int const* cumulative_seqlen_q;
    int const* cumulative_q_blocks;
    int total_q_blocks;
    int const* cumulative_k_blocks;
    int total_k_blocks;
    int const* cumulative_split_kv_blocks;
    int total_split_kv_blocks;
    int num_kv_splits_;
    int* tile_counter;
    FastDivmod divmod_num_v_blocks;
    FastDivmod divmod_max_q_blocks;
    FastDivmod divmod_num_heads;
    FastDivmod divmod_num_kv_splits;
  };

  Params params;
  uint64_t current_work_linear_idx_ = 0;

  CUTLASS_DEVICE
  XeFMHASplitKVDynamicPersistentTileScheduler(Params const& params) : params(params) {
    current_work_linear_idx_ = fetch_next_work_tile();
  }

  template <int Num_SGs = 16, class ProblemShape, class TileShape>
  static Params to_underlying_arguments(
      ProblemShape const& shape, KernelHardwareInfo hw_info, TileShape const& tile_shape, int num_kv_splits = 1) {
    using namespace cute;
    using SeqLenQ = remove_cvref_t<decltype(shape.seq_len_qo)>;
    using SeqLenKV = remove_cvref_t<decltype(shape.seq_len_kv)>;

    int const num_v_blocks = size(ceil_div(shape.head_size_vo, get<1>(tile_shape)));
    int const q_tile_size = get<0>(tile_shape);
    int max_seq_len_q = 0;
    int const* cumulative_seqlen_q = nullptr;
    int const* cumulative_q_blocks = nullptr;
    int total_q_blocks = 0;
    int const* cumulative_k_blocks = nullptr;
    int total_k_blocks = 0;
    int const* cumulative_split_kv_blocks = nullptr;
    int total_split_kv_blocks = 0;

    if constexpr (cutlass::fmha::collective::is_variable_length_v<SeqLenQ>) {
      max_seq_len_q = shape.seq_len_qo.max_length;
      cumulative_seqlen_q = shape.seq_len_qo.cumulative_length;
      cumulative_q_blocks = shape.seq_len_qo.cumulative_blocks;
      total_q_blocks = shape.seq_len_qo.total_blocks;
    } else {
      max_seq_len_q = shape.seq_len_qo;
      total_q_blocks = cute::ceil_div(max_seq_len_q, q_tile_size) * shape.batch;
    }

    if constexpr (cutlass::fmha::collective::is_variable_length_v<SeqLenKV>) {
      cumulative_k_blocks = shape.seq_len_kv.cumulative_blocks;
      total_k_blocks = shape.seq_len_kv.total_blocks;
      cumulative_split_kv_blocks = shape.seq_len_kv.cumulative_split_kv_blocks;
      total_split_kv_blocks = shape.seq_len_kv.total_split_kv_blocks;
    } else {
      total_k_blocks = shape.batch * num_kv_splits;
    }

    int const max_q_blocks = size(ceil_div(max_seq_len_q, q_tile_size));
    int const work_q_blocks = total_q_blocks > 0 ? total_q_blocks : shape.batch * max_q_blocks;
    int const work_split_kv_blocks = total_split_kv_blocks > 0 ? total_split_kv_blocks : work_q_blocks * num_kv_splits;
    int const max_work_tiles = shape.num_heads_q * num_v_blocks * work_split_kv_blocks;
    int const grid_x = cute::min(max_work_tiles, hw_info.sm_count * 8);
    dim3 grid(grid_x, 1, 1);

    total_q_blocks = work_q_blocks;

    return Params{
        grid,
        shape.batch,
        shape.num_heads_q,
        num_v_blocks,
        max_q_blocks,
        q_tile_size,
        cumulative_seqlen_q,
        cumulative_q_blocks,
        total_q_blocks,
        cumulative_k_blocks,
        total_k_blocks,
        cumulative_split_kv_blocks,
        total_split_kv_blocks,
        num_kv_splits,
        nullptr,
        {num_v_blocks},
        {max_q_blocks},
        {shape.num_heads_q},
        {num_kv_splits}};
  }

  template <int Num_SGs>
  static dim3 get_grid_shape(Params const& params) {
    return params.grid;
  }

  template <class Varlen = cutlass::fmha::collective::VariableLength>
  CUTLASS_DEVICE int q_blocks_for_batch(int batch_idx) const {
    if constexpr (cutlass::fmha::collective::is_variable_length_v<Varlen>) {
      int seq_len_q = params.cumulative_seqlen_q[batch_idx + 1] - params.cumulative_seqlen_q[batch_idx];
      return cute::ceil_div(seq_len_q, params.q_tile_size);
    }
    return params.max_q_blocks;
  }

  CUTLASS_DEVICE
  int total_work_tiles() const {
    int split_kv_blocks =
        params.total_split_kv_blocks > 0 ? params.total_split_kv_blocks : params.total_q_blocks * params.num_kv_splits_;
    return split_kv_blocks * params.num_heads_q * params.num_v_blocks;
  }

  CUTLASS_DEVICE
  bool is_valid() {
    return current_work_linear_idx_ < uint64_t(total_work_tiles());
  }

  template <class Varlen = cutlass::fmha::collective::VariableLength>
  CUTLASS_DEVICE auto get_block_coord() {
    using namespace cute;
    int work_idx = static_cast<int>(current_work_linear_idx_);
    int blk_q;
    int blk_v;
    int head;
    int idx_b;
    int blk_k;
    xe_fmha_split_kv_persistent_decompose_work_idx<Varlen>(*this, work_idx, blk_q, blk_v, head, idx_b, blk_k);

    return make_coord(blk_q, blk_v, head, idx_b, blk_k);
  }

  CUTLASS_DEVICE
  uint64_t fetch_next_work_tile() const {
    int work_idx = 0;
    if (ThreadIdxX() == 0) {
      work_idx = atomicAdd(params.tile_counter, 1);
    }
    work_idx = sycl::group_broadcast(sycl::ext::oneapi::this_work_item::get_work_group<3>(), work_idx, 0);
    return uint64_t(work_idx);
  }

  CUTLASS_DEVICE
  XeFMHASplitKVDynamicPersistentTileScheduler& operator++() {
    current_work_linear_idx_ = fetch_next_work_tile();
    return *this;
  }
};

struct XeReduceSplitKTileScheduler {
  struct Params {
    dim3 grid;
    FastDivmod divmod_num_heads;
    int num_kv_splits = 1;
  };

  bool valid_ = true;
  Params params;

  CUTLASS_DEVICE
  XeReduceSplitKTileScheduler(Params const& params) : params(params) {}

  template <int Num_SGs = 16, class ProblemShape, class TileShape>
  static Params to_underlying_arguments(
      ProblemShape const& shape, KernelHardwareInfo hw_info, TileShape const& tile_shape, int num_kv_splits = 1) {
    using namespace cute;
    int max_seq_len_q = 0;
    if constexpr (cutlass::fmha::collective::is_variable_length_v<remove_cvref_t<decltype(shape.seq_len_qo)>>) {
      max_seq_len_q = shape.seq_len_qo.max_length;
    } else {
      max_seq_len_q = shape.seq_len_qo;
    }
    dim3 grid(max_seq_len_q, shape.num_heads_q, shape.batch);
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

  template <class Varlen = cutlass::fmha::collective::VariableLength>
  CUTLASS_DEVICE auto get_block_coord() {
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
