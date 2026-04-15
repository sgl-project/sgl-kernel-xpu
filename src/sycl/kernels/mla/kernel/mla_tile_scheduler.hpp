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
    \brief Tile scheduler for MLA decode attention
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/kernel_hardware_info.h"

namespace cutlass::flash_attention::kernel {
////////////////////////////////////////////////////////////////////////////////

struct XeMlaIndividualTileScheduler {
  //
  // Params
  //
  struct Params {
    dim3 grid;
    FastDivmod divmod_num_heads;
    FastDivmod divmod_batch_heads;
    int num_kv_splits_ = -1;
  };

  //
  // data members
  //
  bool valid_ = true;
  Params params;
  //
  // methods
  //

  CUTLASS_DEVICE
  XeMlaIndividualTileScheduler(Params const& params) : params(params) {}

  template <class ProblemShape, class TileShape>
  static Params to_underlying_arguments(
      ProblemShape const& shape, KernelHardwareInfo hw_info, TileShape const& tile_shape, int num_kv_splits = -1) {
    using namespace cute;

    // Ensure num_heads_q is at least 1 to avoid division by zero
    int num_heads = shape.num_heads_q > 0 ? shape.num_heads_q : 1;

    dim3 grid(
        size(ceil_div(shape.head_size_o, get<1>(tile_shape))),  // V tiles
        size(ceil_div(shape.seq_len_qo, get<0>(tile_shape))),   // Q tiles
        size(shape.batch * num_heads));                         // (h,b) combined

    if (num_kv_splits > 1) {
      grid.z *= num_kv_splits;
    }

    return Params{grid, FastDivmod{num_heads}, FastDivmod{shape.batch * num_heads}, num_kv_splits};
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
    int idx_kv_split = BlockIdxZ();
    int head, idx_b;

    if (params.num_kv_splits_ > 1) {
      // First extract kv_split_idx, then (head, batch)
      params.divmod_batch_heads(idx_kv_split, idx_b, idx_kv_split);
      params.divmod_num_heads(idx_b, head, idx_b);
      return make_coord(BlockIdxY(), BlockIdxX(), head, idx_b, idx_kv_split);
    }

    idx_b = idx_kv_split;
    params.divmod_num_heads(idx_b, head, idx_b);
    return make_coord(BlockIdxY(), BlockIdxX(), head, idx_b, int(1));
  }

  CUTLASS_DEVICE
  XeMlaIndividualTileScheduler& operator++() {
    valid_ = false;
    return *this;
  }
};

////////////////////////////////////////////////////////////////////////////////

struct XeMlaReduceSplitKScheduler {
  //
  // Params
  //
  struct Params {
    dim3 grid;
    FastDivmod divmod_num_heads;
    int num_kv_splits = -1;
  };

  //
  // data members
  //
  bool valid_ = true;
  Params params;

  CUTLASS_DEVICE
  XeMlaReduceSplitKScheduler(Params const& params) : params(params) {}

  template <class ProblemShape, class TileShape>
  static Params to_underlying_arguments(
      ProblemShape const& shape, KernelHardwareInfo hw_info, TileShape const& tile_shape, int num_kv_splits = -1) {
    using namespace cute;
    // Grid: (seq_len_qo, num_heads_q, batch)
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
    // (seq_q, head_q, batch)
    return make_coord(BlockIdxX(), BlockIdxY(), BlockIdxZ());
  }

  CUTLASS_DEVICE
  XeMlaReduceSplitKScheduler& operator++() {
    valid_ = false;
    return *this;
  }
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::flash_attention::kernel
