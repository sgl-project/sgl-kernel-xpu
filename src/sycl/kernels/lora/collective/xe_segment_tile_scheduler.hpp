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
/*!
  \file
  \brief Tile scheduler of the chunked-SGMV LoRA kernels: sycl-tla's grouped-GEMM scheduler
         PersistentTileSchedulerXeGroup, specialized for SegmentSliceProblemShape.

  The grouped GemmUniversal (xe_gemm_array_cooperative.hpp) always instantiates
  PersistentTileSchedulerXeGroup<ProblemShape>. The generic one walks the groups one by one in
  every work-group, reading each group's shape from global memory; with tens of
  segment x slice groups that walk alone cost more than the whole GEMM (measured 2-4x slower
  kernels on BMG). This specialization keeps its interface but maps a work-group to its tile
  directly:

    * grid    : x = column tile (enumerated slice by slice), y = row tile, both upper bounds:
                ceil(N / BN) + n_slices and ceil(num_rows / BM) + num_segments.
    * columns : column tiles never straddle a slice of slice_offsets.
    * rows    : a sub-group prefix scan over the segments' BM-row tile counts, 16 segments per
                step, finds the segment; surplus work-groups get an invalid tile and exit.

  Each work-group computes at most one tile (fetch_next_work() returns an invalid tile), so the
  host never needs the group shapes and never syncs.
*/

#pragma once

#include <cstdint>
#include <cute/tensor.hpp>
#include <sycl/sycl.hpp>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/gemm/kernel/xe_tile_scheduler_group.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "sycl/kernels/lora/collective/xe_segment_problem_shape.hpp"

namespace cutlass::gemm::kernel::detail {

template <>
class PersistentTileSchedulerXeGroup<cutlass::lora::collective::SegmentSliceProblemShape> {
 public:
  // Checked by GatherScatterGroupGemmTypes: the kernel uses this specialization.
  static constexpr bool kIsLoraSegmentScheduler = true;
  using GroupProblemShape = cutlass::lora::collective::SegmentSliceProblemShape;
  using ProblemShape = typename GroupProblemShape::UnderlyingProblemShape;
  using RasterOrder = cutlass::gemm::kernel::detail::RasterOrder;
  using RasterOrderOptions = cutlass::gemm::kernel::detail::RasterOrderOptions;

  struct WorkTileInfo {
    int32_t M_idx = 0;  // row tile inside the segment
    int32_t N_idx = 0;  // column tile inside the slice
    int32_t L_idx = 0;  // group id, SegmentSliceProblemShape::group_id(segment, slice)
    bool is_valid_tile = false;

    CUTLASS_HOST_DEVICE bool is_valid() const {
      return is_valid_tile;
    }
    CUTLASS_HOST_DEVICE static WorkTileInfo invalid_work_tile() {
      return {-1, -1, -1, false};
    }
    CUTLASS_HOST_DEVICE bool is_final_split(uint32_t) const {
      return true;
    }
    CUTLASS_HOST_DEVICE int32_t reduction_subtile_idx() const {
      return -1;
    }
  };

  struct Arguments {
    int max_swizzle_size = 1;
    RasterOrderOptions raster_order = RasterOrderOptions::AlongM;
  };

  struct Params {
    GroupProblemShape problem_shapes_;
    int bm = 1;  // tile rows / columns
    int bn = 1;
    uint32_t row_tiles = 0;  // grid upper bounds
    uint32_t col_tiles = 0;
    RasterOrder raster_order_ = RasterOrder::AlongM;  // read by the kernel's get_grid_shape()
  };

 private:
  WorkTileInfo work_;

 public:
  template <class TileShape, class ClusterShape>
  static Params to_underlying_arguments(
      GroupProblemShape problem_shapes,
      TileShape,
      ClusterShape,
      KernelHardwareInfo const&,
      Arguments const&,
      [[maybe_unused]] void* workspace = nullptr,
      [[maybe_unused]] const uint32_t epilogue_subtile = 1,
      [[maybe_unused]] uint32_t ktile_start_alignment_count = 1u) {
    static_assert(cute::is_static<TileShape>::value);
    constexpr int BM = cute::size<0>(TileShape{});
    constexpr int BN = cute::size<1>(TileShape{});
    Params p;
    p.problem_shapes_ = problem_shapes;
    p.bm = BM;
    p.bn = BN;
    p.row_tiles = static_cast<uint32_t>((problem_shapes.num_rows + BM - 1) / BM + problem_shapes.num_segments);
    p.col_tiles = static_cast<uint32_t>(
        (problem_shapes.N + BN - 1) / BN + (problem_shapes.n_slices > 1 ? problem_shapes.n_slices : 0));
    return p;
  }

  template <class TileShape, class ClusterShape>
  CUTLASS_HOST_DEVICE static dim3 get_grid_shape(
      Params const& params, GroupProblemShape, TileShape, ClusterShape, KernelHardwareInfo, Arguments, bool = true) {
    return dim3(params.col_tiles, params.row_tiles, 1);
  }

  static bool can_implement(Arguments const&) {
    return true;
  }

  PersistentTileSchedulerXeGroup() = default;

  CUTLASS_DEVICE explicit PersistentTileSchedulerXeGroup(Params const& params) {
#if defined __SYCL_DEVICE_ONLY__
    auto const& gr = params.problem_shapes_;
    // Column tile -> (slice, column tile inside the slice).
    int col = BlockIdxX(), slice = 0;
    if (gr.slice_offsets) {
      for (; slice < gr.n_slices; ++slice) {
        const int ng = (gr.slice_offsets[slice + 1] - gr.slice_offsets[slice] + params.bn - 1) / params.bn;
        if (col < ng) {
          break;
        }
        col -= ng;
      }
      if (slice == gr.n_slices) {
        return;
      }
    } else if (col * params.bn >= gr.N) {
      return;
    }
    // Row tile -> (segment, row tile inside the segment).
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    const int lane = sg.get_local_linear_id();
    const int num_segments = gr.num_segments;
    int t = BlockIdxY();
    for (int s0 = 0; s0 < num_segments; s0 += 16) {
      const int s = s0 + lane;
      const int len = s < num_segments ? gr.seg_indptr[s + 1] - gr.seg_indptr[s] : 0;
      const int nt = (len + params.bm - 1) / params.bm;
      const int incl = sycl::inclusive_scan_over_group(sg, nt, sycl::plus<int>());
      const int total = sycl::group_broadcast(sg, incl, 15);
      if (t < total) {
        const bool hit = t < incl && t >= incl - nt;
        const int hit_lane = sycl::reduce_over_group(sg, hit ? lane : 16, sycl::minimum<int>());
        work_ = {
            t - sycl::group_broadcast(sg, incl - nt, hit_lane),
            col,
            GroupProblemShape::group_id(s0 + hit_lane, slice),
            true};
        return;
      }
      t -= total;
    }
#else
    (void)params;
#endif
  }

  template <class ClusterShape>
  CUTLASS_DEVICE WorkTileInfo initial_work_tile_info(ClusterShape) {
    return work_;
  }

  // One tile per work-group.
  CUTLASS_DEVICE auto fetch_next_work(WorkTileInfo) {
    return cute::make_tuple(WorkTileInfo::invalid_work_tile(), true);
  }

  CUTLASS_HOST_DEVICE static bool compute_epilogue(WorkTileInfo const&, Params const&) {
    return true;
  }

  template <class FrgTensorC>
  CUTLASS_DEVICE static void fixup(Params const&, WorkTileInfo const&, FrgTensorC&, uint32_t, uint32_t) {}

  template <class ProblemShape_, class ElementAccumulator>
  static size_t get_workspace_size(
      Arguments const&, ProblemShape_, KernelHardwareInfo const&, uint32_t, const uint32_t = 1, uint32_t = 1) {
    return 0;
  }

  template <class ProblemShape_, class ElementAccumulator>
  static cutlass::Status initialize_workspace(
      Arguments const&,
      void*,
      cudaStream_t,
      ProblemShape_,
      KernelHardwareInfo const&,
      uint32_t,
      const uint32_t = 1,
      uint32_t = 1,
      CudaHostAdapter* = nullptr) {
    return cutlass::Status::kSuccess;
  }

  template <class ProblemShape_MNKL, class TileShape>
  CUTLASS_HOST_DEVICE static int
  get_work_k_tile_count(WorkTileInfo const&, ProblemShape_MNKL problem_shape, TileShape tile_shape) {
    return cute::size(cute::ceil_div(cute::get<2>(problem_shape), cute::get<2>(tile_shape)));
  }

  CUTLASS_HOST_DEVICE static uint32_t get_work_k_tile_start(WorkTileInfo const&) {
    return 0u;
  }
};

}  // namespace cutlass::gemm::kernel::detail
