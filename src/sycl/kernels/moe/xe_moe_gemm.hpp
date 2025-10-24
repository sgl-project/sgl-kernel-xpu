/***************************************************************************************************
 * Copyright 2025 Intel corporation. All rights reserved.
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
#include "cutlass/workspace.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/device_kernel.h"
#include "cutlass/detail/layout.hpp"
#include "cutlass/detail/mma.hpp"
#include "cutlass/cuda_host_adapter.hpp"

#include "cutlass/kernel_launch.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/sycl_event_manager.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cute/tensor.hpp"
#include "dispatch_policy.hpp"

namespace cutlass::gemm::kernel::detail {

///////////////////////////////////////////////////////////////////////////////

// Persistent Thread Block (TB) scheduler for MoE GEMM
template <class GroupProblemShape>
class PersistentTileSchedulerXeMoE {
  //
  // Data members
  //

 private:
  uint64_t current_work_linear_idx_ = 0;
  uint64_t total_grid_size_ = 0;
  int32_t* num_rows_per_expert_ = nullptr;
  int32_t K_ = 0;
  int32_t N_ = 0;
  int32_t num_experts_ = 0;

  // Tracking current group, its starting linear idx and total tiles
  struct GroupInfo {
    int group_idx = 0;
    uint64_t start_linear_idx = 0;
    uint64_t total_tiles = 0;
  } current_group_info_;

 public:
  struct WorkTileInfo {
    int32_t M_idx = 0;
    int32_t N_idx = 0;
    int32_t L_idx = 0;
    bool is_valid_tile = false;

    CUTLASS_HOST_DEVICE
    bool is_valid() const {
      return is_valid_tile;
    }

    CUTLASS_HOST_DEVICE
    static WorkTileInfo invalid_work_tile() {
      return {-1, -1, -1, false};
    }

    CUTLASS_HOST_DEVICE
    bool is_final_split(uint32_t k_tiles_per_output_tile) const {
      return true;
    }

    CUTLASS_HOST_DEVICE
    int32_t reduction_subtile_idx() const {
      return -1;
    }
  };

  using ProblemShape = typename GroupProblemShape::UnderlyingProblemShape;
  using Params = PersistentTileSchedulerSm90GroupParams<GroupProblemShape>;
  using RasterOrder = typename Params::RasterOrder;
  using RasterOrderOptions = typename Params::RasterOrderOptions;

  struct Arguments {
    int max_swizzle_size = 1;
    // Not applying Heuristics for Grouped problems, since largest dimension can change per group
    RasterOrderOptions raster_order = RasterOrderOptions::AlongM;
  };

  // Sink scheduler params as a member
  Params scheduler_params;

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE void configure(int32_t* num_rows_per_expert, int32_t N, int32_t K, int32_t num_experts) {
    num_rows_per_expert_ = num_rows_per_expert;
    N_ = N;
    K_ = K;
    num_experts_ = num_experts;
  }

  // Given the inputs, computes the total number of output blocks this problem will compute over
  // Note that this is only the logical size of our grid, not the physical grid we will actually launch.
  template <class ClusterShape>
  CUTLASS_HOST_DEVICE static dim3
  get_tiled_cta_shape_mnl(const KernelHardwareInfo& hw_info, ClusterShape cluster_shape) {
    uint32_t total_ctas = 0;
    uint32_t cta_in_N_dim = 1;  // We linearize the blocks across all the problems here

    total_ctas = hw_info.sm_count;

    return Params::get_tiled_cta_shape_mnl(to_gemm_coord(cluster_shape), total_ctas, cta_in_N_dim);
  }

  template <class TileShape, class ClusterShape>
  static Params to_underlying_arguments(
      GroupProblemShape problem_shapes,
      TileShape tile_shape,
      ClusterShape cluster_shape,
      KernelHardwareInfo const& hw_info,
      Arguments const& arguments,
      [[maybe_unused]] void* workspace = nullptr,
      [[maybe_unused]] const uint32_t epilogue_subtile = 1,
      [[maybe_unused]] uint32_t ktile_start_alignment_count = 1u) {
    // We only need the tile and cluster shape during scheduler setup, so let FTAD do the magic
    static_assert(cute::is_static<TileShape>::value);
    static_assert(cute::is_static<ClusterShape>::value);

    dim3 problem_blocks = get_tiled_cta_shape_mnl(hw_info, cluster_shape);

    Params params;
    params.initialize(
        problem_blocks,
        problem_shapes,
        to_gemm_coord(tile_shape),
        to_gemm_coord(cluster_shape),
        hw_info,
        arguments.max_swizzle_size,
        arguments.raster_order);

    return params;
  }

  // Given the inputs, computes the physical grid we should launch.
  template <class TileShape, class ClusterShape>
  CUTLASS_HOST_DEVICE static dim3 get_grid_shape(
      [[maybe_unused]] Params const& params,
      GroupProblemShape problem_shapes,
      TileShape tile_shape,
      ClusterShape cluster_shape,
      KernelHardwareInfo hw_info,
      Arguments arguments,
      bool truncate_by_problem_size = true) {
    dim3 problem_blocks = get_tiled_cta_shape_mnl(hw_info, cluster_shape);

    return Params::get_grid_shape(
        problem_blocks,
        to_gemm_coord(cluster_shape),
        hw_info,
        arguments.max_swizzle_size,
        arguments.raster_order,
        /* truncate_by_problem_size = */ true);
  }

  static bool can_implement(Arguments const& args) {
    return true;
  }

  PersistentTileSchedulerXeMoE() = default;

  CUTLASS_DEVICE explicit PersistentTileSchedulerXeMoE(Params const& params_) : scheduler_params(params_) {
    // MSVC requires protecting use of CUDA-specific nonstandard syntax,
    // like blockIdx and gridDim, with __CUDA_ARCH__.
#if defined(__CUDA_ARCH__) || defined __SYCL_DEVICE_ONLY__
    if (scheduler_params.raster_order_ == RasterOrder::AlongN) {
      current_work_linear_idx_ = uint64_t(BlockIdxX()) + uint64_t(BlockIdxY()) * uint64_t(GridDimX());
    } else {
      current_work_linear_idx_ = uint64_t(BlockIdxX()) * uint64_t(GridDimY()) + uint64_t(BlockIdxY());
    }

    total_grid_size_ = uint64_t(GridDimX()) * uint64_t(GridDimY()) * uint64_t(GridDimZ());

#else
    CUTLASS_ASSERT(false && "This line should never be reached");
#endif
  }

  CUTLASS_DEVICE
  WorkTileInfo get_current_work() {
    return get_current_work_for_linear_idx(current_work_linear_idx_);
  }

  CUTLASS_DEVICE
  WorkTileInfo get_current_work_for_linear_idx(uint64_t linear_idx) {
    if (scheduler_params.pre_processed_problem_shapes && linear_idx >= scheduler_params.blocks_across_problem_) {
      return WorkTileInfo::invalid_work_tile();
    }

    return get_work_idx_m_and_n(
        linear_idx,
        current_group_info_,
        scheduler_params.problem_shapes_,
        scheduler_params.cta_shape_,
        scheduler_params.cluster_shape_,
        scheduler_params.divmod_cluster_shape_major_,
        scheduler_params.divmod_cluster_shape_minor_,
        scheduler_params.divmod_cta_shape_m_,
        scheduler_params.divmod_cta_shape_n_,
        scheduler_params.log_swizzle_size_,
        scheduler_params.raster_order_);
  }

  CUTLASS_DEVICE
  void advance_to_next_work(uint32_t advance_count = 1) {
    current_work_linear_idx_ += total_grid_size_ * uint64_t(advance_count);
  }

  // get work_idx_m, work_idx_n from linear_idx while applying swizzle
  CUTLASS_DEVICE
  WorkTileInfo get_work_idx_m_and_n(
      uint64_t linear_idx,
      struct GroupInfo& group_info,
      GroupProblemShape& problem_shapes,
      GemmCoord cta_shape,
      GemmCoord cluster_shape,
      FastDivmodU64Pow2 const& divmod_cluster_shape_major,
      FastDivmodU64Pow2 const& divmod_cluster_shape_minor,
      FastDivmodU64 const& divmod_cta_shape_m,
      FastDivmodU64 const& divmod_cta_shape_n,
      int32_t log_swizzle_size,
      RasterOrder raster_order) {
    bool valid_tile = true;
    uint64_t ctas_along_m, ctas_along_n;
    int total_problem_groups = num_experts_;
    ctas_along_m = divmod_cta_shape_m.divide(
        cute::shape<0>(ProblemShape(num_rows_per_expert_[group_info.group_idx], N_, K_)) + divmod_cta_shape_m.divisor -
        1);
    ctas_along_n = divmod_cta_shape_n.divide(
        cute::shape<1>(ProblemShape(num_rows_per_expert_[group_info.group_idx], N_, K_)) + divmod_cta_shape_n.divisor -
        1);

    auto problem_blocks_m = round_up(ctas_along_m, (1 << log_swizzle_size) * cluster_shape.m());
    auto problem_blocks_n = round_up(ctas_along_n, (1 << log_swizzle_size) * cluster_shape.n());
    group_info.total_tiles = problem_blocks_m * problem_blocks_n;

    while (group_info.start_linear_idx + group_info.total_tiles <= linear_idx) {
      group_info.group_idx++;

      if (group_info.group_idx >= total_problem_groups) return WorkTileInfo::invalid_work_tile();

      group_info.start_linear_idx += group_info.total_tiles;
      ctas_along_m = divmod_cta_shape_m.divide(
          cute::shape<0>(ProblemShape(num_rows_per_expert_[group_info.group_idx], N_, K_)) +
          divmod_cta_shape_m.divisor - 1);
      ctas_along_n = divmod_cta_shape_n.divide(
          cute::shape<1>(ProblemShape(num_rows_per_expert_[group_info.group_idx], N_, K_)) +
          divmod_cta_shape_n.divisor - 1);

      problem_blocks_m = round_up(ctas_along_m, (1 << log_swizzle_size) * cluster_shape.m());
      problem_blocks_n = round_up(ctas_along_n, (1 << log_swizzle_size) * cluster_shape.n());
      group_info.total_tiles = problem_blocks_m * problem_blocks_n;
    }

    uint64_t cluster_id, cluster_major_offset = 0, cluster_minor_offset = 0;
    uint64_t blk_per_grid_dim = divmod_cluster_shape_minor.divide(linear_idx - group_info.start_linear_idx);
    divmod_cluster_shape_major(cluster_id, cluster_major_offset, blk_per_grid_dim);

    // With static schedulers, we launch grid such that all cluster are linear (1-D) order, i.e.,
    // there can only be one cluster in the minor dimension. get_grid_shape() in scheduler params
    // put cluster_shape.m/n() as the minor dimension based on raster order AlongN/M resp.
    // Therefore, the offset of a CTA (inside a cluster) in the minor dimension can be directly be
    // inferred by the blockIdx along the minor dimension.
    if (raster_order == RasterOrder::AlongN) {
      cluster_minor_offset = BlockIdxX();
    } else {
      cluster_minor_offset = BlockIdxY();
    }

    uint64_t cluster_idx_minor, cluster_idx_major;

    uint64_t cluster_idx_minor_div_swizzle, extra, offset;

    offset = cluster_id & ((1 << log_swizzle_size) - 1);
    extra = cluster_id >> log_swizzle_size;

    uint64_t curr_group_cluster_blk_major;
    if (raster_order == RasterOrder::AlongN) {
      curr_group_cluster_blk_major = divmod_cluster_shape_major.divide(problem_blocks_n);
    } else {
      curr_group_cluster_blk_major = divmod_cluster_shape_major.divide(problem_blocks_m);
    }
    cluster_idx_minor_div_swizzle = extra / curr_group_cluster_blk_major;
    cluster_idx_major = extra % curr_group_cluster_blk_major;

    cluster_idx_minor = cluster_idx_minor_div_swizzle * (1 << log_swizzle_size) + offset;

    auto minor_work_idx =
        static_cast<int32_t>(cluster_idx_minor * divmod_cluster_shape_minor.divisor + cluster_minor_offset);
    auto major_work_idx =
        static_cast<int32_t>(cluster_idx_major * divmod_cluster_shape_major.divisor + cluster_major_offset);

    if (raster_order == RasterOrder::AlongN) {
      return {minor_work_idx, major_work_idx, group_info.group_idx, valid_tile};
    } else {
      return {major_work_idx, minor_work_idx, group_info.group_idx, valid_tile};
    }
  }

  // Returns whether the block assigned this work should compute the epilogue for the corresponding
  // output tile. For the basic tile scheduler, this is always true.
  CUTLASS_HOST_DEVICE
  static bool compute_epilogue(WorkTileInfo const&, Params const&) {
    return true;
  }

  // Performs the reduction across splits for a given output tile. Since this scheduler does
  // not split output tiles, no reduction is needed.
  template <class FrgTensorC>
  CUTLASS_DEVICE static void fixup(Params const&, WorkTileInfo const&, FrgTensorC&, uint32_t, uint32_t) {}

  // Returns whether the current WorkTileInfo passed in should continue to be used. Since
  // this scheduler only schedules work in units of single, full output tiles, the WorkTileInfo
  // passed in should not be used after having been processed.
  CUTLASS_DEVICE
  static bool continue_current_work(WorkTileInfo&) {
    return false;
  }

  // The basic tile scheduler does not require any additional workspace
  template <class ProblemShape, class ElementAccumulator>
  static size_t get_workspace_size(
      Arguments const&, ProblemShape, KernelHardwareInfo const&, uint32_t, const uint32_t = 1, uint32_t = 1) {
    return 0;
  }

  template <class ProblemShape, class ElementAccumulator>
  static cutlass::Status initialize_workspace(
      Arguments const&,
      void*,
      cudaStream_t,
      ProblemShape,
      KernelHardwareInfo const&,
      uint32_t,
      const uint32_t = 1,
      uint32_t = 1,
      CudaHostAdapter* cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  template <class ProblemShape_MNKL, class TileShape>
  CUTLASS_HOST_DEVICE static int
  get_work_k_tile_count(WorkTileInfo const& work_tile_info, ProblemShape_MNKL problem_shape, TileShape tile_shape) {
    // All work units returned by this scheduler cover the entire K iteration
    // space of the output tile assigned to the work unit.
    return cute::size(cute::ceil_div(cute::get<2>(problem_shape), cute::get<2>(tile_shape)));
  }

  CUTLASS_HOST_DEVICE
  static uint32_t get_work_k_tile_start(WorkTileInfo const&) {
    // All work units returned by this scheduler start from K tile 0
    return 0u;
  }

  CUTLASS_DEVICE
  static bool need_separate_reduction(Params const& params) {
    return false;
  }

  CUTLASS_DEVICE
  bool is_work_tile_for_reduction(WorkTileInfo const& work_tile_info, Params const& params) {
    return false;
  }

  CUTLASS_DEVICE
  uint32_t epilgoue_subtile_idx(WorkTileInfo const& work_tile_info, Params const& params) const {
    return 0;
  }

  template <class FrgTensorC>
  CUTLASS_DEVICE void separate_reduction(
      Params const& params,
      WorkTileInfo const& work_tile_info,
      FrgTensorC& accumulators,
      uint32_t num_barriers,
      uint32_t barrier_idx) {}

  // Shares the accumulator set with peers in the global workspace
  template <class FrgTensorC>
  CUTLASS_DEVICE static void share(
      Params const& params,
      WorkTileInfo const& work_tile_info,
      FrgTensorC& accumulators,
      uint32_t num_barriers,
      uint32_t barrier_idx) {}

  CUTLASS_DEVICE
  static bool valid_warpgroup_in_work_tile(WorkTileInfo const& work_tile_info) {
    return true;
  }

  CUTLASS_DEVICE
  static bool requires_separate_reduction(Params const& params) {
    return false;
  }

  // Kernel helper function to get next work tile
  CUTLASS_DEVICE
  auto fetch_next_work(WorkTileInfo work_tile_info) {
    if (continue_current_work(work_tile_info)) {
      return cute::make_tuple(work_tile_info, true);
    }

    advance_to_next_work();
    return cute::make_tuple(get_current_work(), true);
  }

  // Returns the initial work tile info that will be computed over
  template <class ClusterShape>
  CUTLASS_DEVICE WorkTileInfo initial_work_tile_info(ClusterShape) {
    return get_current_work();
  }
};

}  // namespace cutlass::gemm::kernel::detail

///////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::kernel {

///////////////////////////////////////////////////////////////////////////////

template <
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileScheduler_
>
class GemmMoEUniversal{
public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;
  static_assert(cute::rank(typename ProblemShape::UnderlyingProblemShape{}) == 3 or cute::rank(typename ProblemShape::UnderlyingProblemShape{}) == 4,
    "ProblemShape{} should be <M,N,K> or <M,N,K,L>");

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::WorkgroupTileShape;
  using WorkgroupTileShape = TileShape;
  using TiledMma = typename CollectiveMainloop::TiledMma;
  using ArchTag = typename CollectiveMainloop::ArchTag;
  using ElementA = typename CollectiveMainloop::ElementA;
  using StrideA = typename CollectiveMainloop::StrideA;
  using InternalStrideA = typename CollectiveMainloop::InternalStrideA;
  using ElementB = typename CollectiveMainloop::ElementB;
  using StrideB = typename CollectiveMainloop::StrideB;
  using InternalStrideB = typename CollectiveMainloop::InternalStrideB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC = typename CollectiveEpilogue::StrideC;
  using InternalStrideC = typename CollectiveEpilogue::InternalStrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD = typename CollectiveEpilogue::StrideD;
  using InternalStrideD = typename CollectiveEpilogue::InternalStrideD;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  static_assert(cute::is_same_v<TileScheduler_, GroupScheduler>,
                "Only Group Scheduler is supported with this code.");
  using TileSchedulerTag = TileScheduler_;
  using TileScheduler =
      typename detail::PersistentTileSchedulerXeMoE<ProblemShape>;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;

  static constexpr int SubgroupSize =
      CollectiveMainloop::SubgroupSize; // sub_group size
  static constexpr uint32_t MaxThreadsPerBlock =
      CollectiveMainloop::MaxThreadsPerBlock;
  using MmaAtomShape = typename CollectiveMainloop::MmaAtomShape;
  using SubgroupTileShape = typename CollectiveMainloop::SubgroupTileShape;

  using MainloopTensors = typename CollectiveMainloop::MainloopTensors;
  using EpilogueTensors = typename CollectiveEpilogue::EpilogueTensors;

  // Kernel level shared memory storage
  struct SharedStorage {
    using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;
    EpilogueTensorStorage epilogue;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  static_assert(cute::is_same_v<ClusterShape, cute::Shape<_1, _1, _1>>);

  // Device side arguments
  struct Arguments {
    GemmUniversalMode mode{};
    const ElementA** A_ptr;
    const ElementB** B_ptr;
    const ElementC** C_ptr;
    ElementD** D_ptr;
    decltype(EpilogueArguments{}.thread) fusion_args;
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
    const int *M_per_group{nullptr};
    int num_experts;
    int N;
    int K;
  };

  // Kernel entry point API
  struct Params {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopParams mainloop{};
    EpilogueParams epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerParams scheduler{};
    void *workspace{nullptr};
    const int *M_per_group{nullptr};
    int num_experts;
    int N;
    int K;
  };

  //
  // Methods
  //

  // Convert to underlying arguments. In this case, a simple copy for the aliased type.
  static
  Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    CUTLASS_TRACE_HOST("to_underlying_arguments():");
    auto dummy_problem_shape = cute::Shape<int, int, int>{256, args.N, args.K};
    auto dummy_group_problem_shape = ProblemShape{1, &dummy_problem_shape, nullptr};

    // Get SM count if needed, otherwise use user supplied SM count
    int sm_count = args.hw_info.sm_count;
    if (sm_count <= 0) {
      CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
      sm_count = KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }

    CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);

    KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count};

    // Calculate workspace pointers
    uint8_t *workspace_ptr = reinterpret_cast<uint8_t *>(workspace);

    TileSchedulerParams scheduler = TileScheduler::to_underlying_arguments(
        dummy_group_problem_shape, TileShape{}, ClusterShape{}, hw_info, args.scheduler,
        workspace_ptr);

    return {args.mode,
            dummy_group_problem_shape,
            CollectiveMainloop::to_underlying_arguments(
                dummy_group_problem_shape,
                MainloopArguments{
                  args.A_ptr,
                  nullptr,
                  args.B_ptr,
                  nullptr
                },
                workspace_ptr
            ),
            CollectiveEpilogue::to_underlying_arguments(
                dummy_group_problem_shape,
                EpilogueArguments{
                  args.fusion_args,
                  args.C_ptr,
                  nullptr,
                  args.D_ptr,
                  nullptr
                },
                workspace_ptr
            ),
            hw_info,
            scheduler,
            workspace,
            args.M_per_group,
            args.num_experts,
            args.N,
            args.K};
  }

  static bool
  can_implement(Arguments const& args) {
    bool implementable = true;

    implementable = implementable && (args.mode == GemmUniversalMode::kGrouped ||
          (args.mode == GemmUniversalMode::kBatched && rank(typename ProblemShape::UnderlyingProblemShape{}) == 3));

    implementable = implementable && TileScheduler::can_implement(args.scheduler);
    auto dummy_problem_shape = cute::Shape<int, int, int>{256, args.N, args.K};
    auto dummy_group_problem_shape = ProblemShape{1, &dummy_problem_shape, nullptr};
    implementable &= CollectiveMainloop::can_implement(dummy_group_problem_shape,
                MainloopArguments{
                  args.A_ptr,
                  nullptr,
                  args.B_ptr,
                  nullptr
                });
    implementable &= CollectiveEpilogue::can_implement(dummy_group_problem_shape,
              EpilogueArguments{
                  args.fusion_args,
                  args.C_ptr,
                  nullptr,
                  args.D_ptr,
                  nullptr
                });

    return implementable;
  }

  static size_t
  get_workspace_size(Arguments const& args) {
    size_t workspace_size = 0;
    workspace_size += TileScheduler::template get_workspace_size<typename ProblemShape::UnderlyingProblemShape, ElementAccumulator>(
      args.scheduler, typename ProblemShape::UnderlyingProblemShape{}, args.hw_info, -1);
    return workspace_size;
  }

  static cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr,
    CudaHostAdapter* cuda_adapter = nullptr) {
    Status status = Status::kSuccess;
    uint8_t *workspace_ptr = reinterpret_cast<uint8_t *>(workspace);

    status = TileScheduler::template initialize_workspace<typename ProblemShape::UnderlyingProblemShape, ElementAccumulator>(
      args.scheduler, workspace_ptr, stream, typename ProblemShape::UnderlyingProblemShape{}, args.hw_info, -1);

    return status;
  }

  // Computes the kernel launch grid shape based on runtime parameters
  static dim3
  get_grid_shape(Params const& params) {
    // Given device SM count, set grid size s.t. we do not launch more thread blocks than we can run concurrently
    TileSchedulerArguments args{};
    args.raster_order = params.scheduler.raster_order_ == TileScheduler::RasterOrder::AlongN ? TileScheduler::RasterOrderOptions::AlongN : TileScheduler::RasterOrderOptions::AlongM;
    return TileScheduler::get_grid_shape(params.scheduler, params.problem_shape, TileShape{}, ClusterShape{}, params.hw_info, args);
  }

  static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  CUTLASS_DEVICE
  void
  operator()(Params const& params, char* smem_buf) {
    // Preconditions
    CUTE_STATIC_ASSERT(is_static<WorkgroupTileShape>::value);

    static_assert(cute::rank(InternalStrideA{}) == 3, "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(InternalStrideB{}) == 3, "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(InternalStrideC{}) == 3, "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(InternalStrideD{}) == 3, "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

    // Kernel level shared memory storage
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    TileScheduler scheduler{params.scheduler};
    const int32_t N = params.N;
    const int32_t K = params.K;
    scheduler.configure(
        const_cast<int32_t *>(params.M_per_group), params.N, params.K, params.num_experts);
    auto work_tile_info = scheduler.initial_work_tile_info(ClusterShape{});
    constexpr auto workgroup_shape = WorkgroupTileShape{};                                                  // (BLK_M,BLK_N,BLK_K)

    int thread_idx = int(ThreadIdxX());
    constexpr auto subgroup_shape = SubgroupTileShape{}; // (SUB_M,SUB_N,SUB_K)
    bool did_group_change = true;
    int32_t curr_group = -1;
    using ProblemShapeMNKL = Shape<int, int, int, int>;
    ProblemShapeMNKL problem_shape_MNKL;
    MainloopTensors AB_tensors;
    EpilogueTensors CD_tensors;

    if (work_tile_info.is_valid()) {
      curr_group = work_tile_info.L_idx;
      problem_shape_MNKL = append<4>(Shape<int, int, int>{params.M_per_group[curr_group], N, K}, 1);
    }

    while (work_tile_info.is_valid()) {
      auto M = get<0>(problem_shape_MNKL);
      auto L = get<3>(problem_shape_MNKL);

      Tensor mA_mkl = cute::get_xe_tensor(make_shape(M, K, L)); //(m,k,l)
      Tensor mB_nkl = cute::get_xe_tensor(make_shape(N, K, L)); //(n,k,l)

      auto m_coord = work_tile_info.M_idx;
      auto n_coord = work_tile_info.N_idx;

      auto gA_mkl = local_tile(mA_mkl, select<0,2>(workgroup_shape), make_coord(m_coord, _, 0));
      auto gB_nkl = local_tile(mB_nkl, select<1,2>(workgroup_shape), make_coord(n_coord, _, 0));

      CollectiveMainloop collective_mma;
      if (did_group_change) {
        AB_tensors = collective_mma.update_tensor_shape_stride(
            params.mainloop, curr_group, problem_shape_MNKL,
            params.M_per_group);
      }
      auto tile_coord = make_coord(m_coord, n_coord, _, 0);

      // Get the number of K tiles to compute for this work as well as the starting K tile offset of the work.
      int work_k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, workgroup_shape);
      int work_k_tile_start = TileScheduler::get_work_k_tile_start(work_tile_info);
      auto k_tile_iter = cute::make_coord_iterator(idx2crd(work_k_tile_start, make_shape(K)), make_shape(K));

      TiledMma tiled_mma;
      Tensor accumulators = partition_fragment_C(tiled_mma, take<0,2>(workgroup_shape));

      // Perform the collective scoped MMA
      collective_mma(
        accumulators,
        gA_mkl,
        gB_nkl,
        accumulators,
        k_tile_iter, work_k_tile_count,
        tile_coord,
        K,
        thread_idx,
        params.mainloop,
        AB_tensors
      );

      TileScheduler::fixup(
        params.scheduler, work_tile_info, accumulators, -1, -1);

      if (TileScheduler::compute_epilogue(work_tile_info, params.scheduler)) {
        CollectiveEpilogue epilogue{params.epilogue, shared_storage.epilogue};

        if (did_group_change) {
          CD_tensors = epilogue.update_tensor_shape_stride(
              curr_group, problem_shape_MNKL, params.M_per_group);
          did_group_change = false;
        }

        epilogue(
          problem_shape_MNKL,
          subgroup_shape,
          tile_coord,
          accumulators,
          tiled_mma,
          thread_idx,
          CD_tensors
        );
      }

      // Get next work tile
      auto [next_work_tile_info, temp] = scheduler.fetch_next_work(work_tile_info);
      work_tile_info = next_work_tile_info;

      did_group_change = curr_group != work_tile_info.L_idx;

      if (did_group_change && work_tile_info.is_valid()) {
        curr_group = work_tile_info.L_idx;
        problem_shape_MNKL = append<4>(Shape<int, int, int>{params.M_per_group[curr_group], N, K}, 1);
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel


namespace cutlass::gemm::device {

template <class GemmKernel_>
class GemmMoEUniversalAdapter
{
public:
  using GemmKernel = GetUnderlyingKernel_t<GemmKernel_>;
  using TileShape = typename GemmKernel::TileShape;
  using ElementA = typename GemmKernel::ElementA;
  using ElementB = typename GemmKernel::ElementB;
  using ElementC = typename GemmKernel::ElementC;
  using ElementD = typename GemmKernel::ElementD;
  using ElementAccumulator = typename GemmKernel::ElementAccumulator;
  using DispatchPolicy = typename GemmKernel::DispatchPolicy;
  using CollectiveMainloop = typename GemmKernel::CollectiveMainloop;
  using CollectiveEpilogue = typename GemmKernel::CollectiveEpilogue;

  // Map back to 2.x type as best as possible
  using LayoutA = gemm::detail::StrideToLayoutTagA_t<typename GemmKernel::StrideA>;
  using LayoutB = gemm::detail::StrideToLayoutTagB_t<typename GemmKernel::StrideB>;
  using LayoutC = gemm::detail::StrideToLayoutTagC_t<typename GemmKernel::StrideC>;
  using LayoutD = gemm::detail::StrideToLayoutTagC_t<typename GemmKernel::StrideD>;

  static bool const kEnableCudaHostAdapter = CUTLASS_ENABLE_CUDA_HOST_ADAPTER;

  static ComplexTransform const kTransformA = cute::is_same_v<typename GemmKernel::CollectiveMainloop::TransformA, cute::conjugate> ?
                                              ComplexTransform::kConjugate : ComplexTransform::kNone;
  static ComplexTransform const kTransformB = cute::is_same_v<typename GemmKernel::CollectiveMainloop::TransformB, cute::conjugate> ?
                                              ComplexTransform::kConjugate : ComplexTransform::kNone;

  // Legacy: Assume MultiplyAdd only since we do not use this tag type in 3.0
  using MathOperator = cutlass::arch::OpMultiplyAdd;

  using OperatorClass = cutlass::detail::get_operator_class_t<typename CollectiveMainloop::TiledMma>;

  using ArchTag = typename GemmKernel::ArchTag;

  // NOTE: Assume identity swizzle for now
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  // Assume TiledMma's ShapeMNK is the same as 2.x's ThreadblockShape
  using ThreadblockShape = cutlass::gemm::GemmShape<
      cute::size<0>(TileShape{}),
      cute::size<1>(TileShape{}),
      cute::size<2>(TileShape{})>;

  using ClusterShape = cutlass::gemm::GemmShape<
      cute::size<0>(typename GemmKernel::DispatchPolicy::ClusterShape{}),
      cute::size<1>(typename GemmKernel::DispatchPolicy::ClusterShape{}),
      cute::size<2>(typename GemmKernel::DispatchPolicy::ClusterShape{})>;

  // Instruction shape is easy too, since we get that directly from our TiledMma's atom shape
  using InstructionShape = cutlass::gemm::GemmShape<
      cute::size<0>(typename CollectiveMainloop::TiledMma::AtomShape_MNK{}),
      cute::size<1>(typename CollectiveMainloop::TiledMma::AtomShape_MNK{}),
      cute::size<2>(typename CollectiveMainloop::TiledMma::AtomShape_MNK{})>;

  // Legacy: provide a correct warp count, but no reliable warp shape
  static int const kThreadCount = GemmKernel::MaxThreadsPerBlock;

  // Warp shape is not a primary API type in 3.x
  // But we can best approximate it by inspecting the TiledMma
  // For this, we make the assumption that we always have 4 warps along M, and rest along N, none along K
  // We also always round up the warp count to 4 if the tiled mma is smaller than 128 threads
  static constexpr int WarpsInMma = cute::max(4, CUTE_STATIC_V(cute::size(typename GemmKernel::TiledMma{})) / 32);
  static constexpr int WarpsInMmaM = 4;
  static constexpr int WarpsInMmaN = cute::ceil_div(WarpsInMma, WarpsInMmaM);
  using WarpCount = cutlass::gemm::GemmShape<WarpsInMmaM, WarpsInMmaN, 1>;
  using WarpShape = cutlass::gemm::GemmShape<
      CUTE_STATIC_V(cute::tile_size<0>(typename CollectiveMainloop::TiledMma{})) / WarpsInMmaM,
      CUTE_STATIC_V(cute::tile_size<1>(typename CollectiveMainloop::TiledMma{})) / WarpsInMmaN,
      CUTE_STATIC_V(cute::tile_size<2>(typename CollectiveMainloop::TiledMma{}))>;

  static int constexpr kStages = detail::stages_member(typename CollectiveMainloop::DispatchPolicy{});

  // Inspect TiledCopy for A and B to compute the alignment size
  static int constexpr kAlignmentA = cutlass::detail::get_alignment_count_from_gmem_tiled_copy<
      typename CollectiveMainloop::GmemTiledCopyA, ElementA, typename CollectiveMainloop::TiledMma::ValTypeA>();
  static int constexpr kAlignmentB = cutlass::detail::get_alignment_count_from_gmem_tiled_copy<
      typename CollectiveMainloop::GmemTiledCopyB, ElementB, typename CollectiveMainloop::TiledMma::ValTypeB>();
  static int constexpr kAlignmentC = cutlass::detail::get_alignment_count_from_gmem_tiled_copy<
      typename CollectiveEpilogue::GmemTiledCopyC, ElementC>();
  static int constexpr kAlignmentD = cutlass::detail::get_alignment_count_from_gmem_tiled_copy<
      typename CollectiveEpilogue::GmemTiledCopyD, ElementD>();

  using EpilogueOutputOp = typename CollectiveEpilogue::ThreadEpilogueOp;

  // Split-K preserves splits that are 128b aligned
  static int constexpr kSplitKAlignment = cute::max(
      128 / sizeof_bits<ElementA>::value, 128 / sizeof_bits<ElementB>::value);

  /// Argument structure: User API
  using Arguments = typename GemmKernel::Arguments;
  /// Argument structure: Kernel API
  using Params = typename GemmKernel::Params;

private:

  /// Kernel API parameters object
  Params params_;

public:

  /// Access the Params structure
  Params const& params() const {
    return params_;
  }

  /// Determines whether the GEMM can execute the given problem.
  static Status
  can_implement(Arguments const& args) {
    if (GemmKernel::can_implement(args)) {
      return Status::kSuccess;
    }
    else {
      return Status::kInvalid;
    }
  }

  /// Gets the workspace size
  static size_t
  get_workspace_size(Arguments const& args) {
    size_t workspace_bytes = 0;
    if (args.mode == GemmUniversalMode::kGemmSplitKParallel) {
      workspace_bytes += sizeof(int) * size_t(cute::size<0>(TileShape{})) * size_t(cute::size<1>(TileShape{}));
    }

    workspace_bytes += GemmKernel::get_workspace_size(args);

    CUTLASS_TRACE_HOST("  workspace_bytes: " << workspace_bytes);

    return workspace_bytes;
  }

  /// Computes the grid shape
  static dim3
  get_grid_shape(Arguments const& args, void* workspace = nullptr) {
    auto tmp_params = GemmKernel::to_underlying_arguments(args, workspace);
    return GemmKernel::get_grid_shape(tmp_params);
  }

  /// Computes the grid shape
  static dim3
  get_grid_shape(Params const& params) {
    return GemmKernel::get_grid_shape(params);
  }

  /// Computes the maximum number of active blocks per multiprocessor
  static int maximum_active_blocks(int /* smem_capacity */ = -1) {
    CUTLASS_TRACE_HOST("GemmUniversal::maximum_active_blocks()");
    int max_active_blocks = -1;
    int smem_size = GemmKernel::SharedStorageSize;

    // first, account for dynamic smem capacity if needed
    cudaError_t result;
    if (smem_size >= (48 << 10)) {
      CUTLASS_TRACE_HOST("  Setting smem size to " << smem_size);
      result = cudaFuncSetAttribute(
          device_kernel<GemmKernel>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          smem_size);
      if (cudaSuccess != result) {
        result = cudaGetLastError(); // to clear the error bit
        CUTLASS_TRACE_HOST(
          "  cudaFuncSetAttribute() returned error: "
          << cudaGetErrorString(result));
        return -1;
      }
    }

    // query occupancy after setting smem size
    result = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks,
        device_kernel<GemmKernel>,
        GemmKernel::MaxThreadsPerBlock,
        smem_size);

    if (cudaSuccess != result) {
      result = cudaGetLastError(); // to clear the error bit
      CUTLASS_TRACE_HOST(
        "  cudaOccupancyMaxActiveBlocksPerMultiprocessor() returned error: "
        << cudaGetErrorString(result));
      return -1;
    }

    CUTLASS_TRACE_HOST("  max_active_blocks: " << max_active_blocks);
    return max_active_blocks;
  }

  /// Initializes GEMM state from arguments.
  Status
  initialize(
    Arguments const& args,
    void* workspace = nullptr,
    cudaStream_t stream = nullptr,
    CudaHostAdapter* cuda_adapter = nullptr) {

    CUTLASS_TRACE_HOST("GemmUniversal::initialize() - workspace "
      << workspace << ", stream: " << (stream ? "non-null" : "null"));

    // Initialize the workspace
    Status status = GemmKernel::initialize_workspace(args, workspace, stream, cuda_adapter);
    if (status != Status::kSuccess) {
      return status;
    }
    // Initialize the Params structure
    params_ = GemmKernel::to_underlying_arguments(args, workspace);
    // Don't set the function attributes - require the CudaHostAdapter to set it.
    if constexpr (kEnableCudaHostAdapter) {
      CUTLASS_ASSERT(cuda_adapter);
      return Status::kSuccess;
    }
    else {
      //
      // Account for dynamic smem capacity if needed
      //
      int smem_size = GemmKernel::SharedStorageSize;

      CUTLASS_ASSERT(cuda_adapter == nullptr);
    }
    return Status::kSuccess;
  }

  /// Primary run() entry point API that is static allowing users to create and manage their own params.
  /// Supplied params struct must be construct by calling GemmKernel::to_underlying_arguments()
  static Status
  run(Params& params,
      cudaStream_t stream,
      CudaHostAdapter *cuda_adapter = nullptr,
      bool launch_with_pdl = false) {
    CUTLASS_TRACE_HOST("GemmUniversal::run()");
    dim3 const block = GemmKernel::get_block_shape();
    dim3 const grid = get_grid_shape(params);

    const compat::dim3 sycl_block(block.x, block.y, block.z);
    const compat::dim3 sycl_grid(grid.x, grid.y, grid.z);

    // configure smem size and carveout
    int smem_size = GemmKernel::SharedStorageSize;

    Status launch_result{ Status::kSuccess };
    cutlass::arch::synclog_setup();

    if constexpr (kEnableCudaHostAdapter) {
      CUTLASS_ASSERT(cuda_adapter);
      if (cuda_adapter) {
        void* kernel_params[] = {&params};
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
        CUTLASS_TRACE_HOST("GemmUniversal::run: Launching kernel with CUDA host adapter");
#endif
        launch_result = cuda_adapter->launch(
          grid, block, smem_size, stream, kernel_params, 0
        );

      }
      else {
        CUTLASS_TRACE_HOST("GemmUniversal::run: CUDA host adapter is null");
        return Status::kErrorInternal;
      }
    }
    else {
      CUTLASS_ASSERT(cuda_adapter == nullptr);
      sycl::queue q = *stream;
#if !defined(SYCL_EXT_ONEAPI_WORK_GROUP_SCRATCH_MEMORY)
      using namespace compat::experimental;
      if constexpr (cute::is_same_v<DispatchPolicy, MainloopDeviceAgnostic>) {
        auto event = launch<device_kernel<GemmKernel>>(launch_policy{
          sycl_grid, sycl_block, local_mem_size{static_cast<std::size_t>(smem_size)}
        }, q, params);
        EventManager::getInstance().addEvent(event);
      } else {
        auto event = launch<device_kernel<GemmKernel>>(launch_policy{
          sycl_grid, sycl_block, local_mem_size{static_cast<std::size_t>(smem_size)}
          , kernel_properties{sycl_exp::sub_group_size<DispatchPolicy::SubgroupSize>}
        }, q, params);
        EventManager::getInstance().addEvent(event);
      }
#else
#if defined (SYCL_INTEL_TARGET)
        constexpr bool allow_subgroup_size_prop = true;
#else
        constexpr bool allow_subgroup_size_prop = false;
#endif
        auto kernel_props = [] {
          constexpr bool is_device_agnostic =
            cute::is_same_v<DispatchPolicy, MainloopDeviceAgnostic>;
          if constexpr (!allow_subgroup_size_prop or is_device_agnostic) {
            using EmptyProperties = decltype(sycl::ext::oneapi::experimental::properties());
            return compat::experimental::kernel_properties<EmptyProperties>{};
          } else {
            return compat::experimental::kernel_properties{
              sycl::ext::oneapi::experimental::sub_group_size<DispatchPolicy::SubgroupSize>
            };
          }
        }();
        compat::experimental::launch_properties launch_props {
          sycl::ext::oneapi::experimental::work_group_scratch_size(smem_size),
        };
        compat::experimental::launch_policy policy{
          sycl_grid, sycl_block, launch_props, kernel_props
        };
        auto event = compat::experimental::launch<device_kernel<GemmKernel>, GemmKernel>(policy, q, params);
        EventManager::getInstance().addEvent(event);
#endif // !defined(SYCL_EXT_ONEAPI_WORK_GROUP_SCRATCH_MEMORY)
    }
    cudaError_t result = cudaGetLastError();
    if (cudaSuccess == result && Status::kSuccess == launch_result) {
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
      CUTLASS_TRACE_HOST("GemmUniversal::run: cudaGetLastError reports success");
#endif
      return Status::kSuccess;
    }
    else {
      CUTLASS_TRACE_HOST("  Kernel launch failed. Reason: " << result);
      return Status::kErrorInternal;
    }
  }

  /// Overload that allows a user to re-launch the same kernel without updating internal params struct.
  Status
  run(sycl::queue* stream) {
    return run(params_, stream);
  }

};

} // namespace cutlass::gemm::device
