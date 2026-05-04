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
    \brief XPU MLA Kernel
*/

#pragma once
#include "cute/util/type_traits.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "sycl/kernels/mla/collective/xe_mla_epilogue.hpp"
#include "sycl/kernels/mla/collective/xe_mla_mainloop.hpp"
#include "sycl/kernels/mla/kernel/mla_tile_scheduler.hpp"
namespace cutlass::flash_attention::kernel {
using namespace cute;

struct SplitKVWorkspaceLayout {
  size_t o_accum_offset;
  size_t exp_sums_offset;
  size_t max_logits_offset;
  size_t total_bytes;

  SplitKVWorkspaceLayout(int batch, int num_heads, int num_splits, int head_size_o, size_t elem_o_size) {
    size_t o_accum_bytes = size_t(batch) * num_heads * num_splits * head_size_o * elem_o_size;
    size_t lse_bytes = size_t(batch) * num_heads * num_splits * sizeof(float);

    size_t o_accum_aligned = (o_accum_bytes + 255) & ~size_t(255);
    size_t lse_aligned = (lse_bytes + 255) & ~size_t(255);

    o_accum_offset = 0;
    exp_sums_offset = o_accum_aligned;
    max_logits_offset = o_accum_aligned + lse_aligned;
    total_bytes = o_accum_aligned + lse_aligned + lse_aligned;
  }
};

///////////////////////////////////////////////////////////////////////////////
template <class ProblemShape_, class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_>
class XeMlaFwdKernel {
 public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  using TileShapeQK = typename CollectiveMainloop::TileShapeQK;
  using TileShapePV = typename CollectiveMainloop::TileShapePV;

  using ElementQ = typename CollectiveMainloop::TensorQ::element_type;
  using ElementK = typename CollectiveMainloop::TensorK::element_type;
  using ElementV = typename CollectiveMainloop::TensorV::element_type;

  using StrideQ = decltype(stride(typename CollectiveMainloop::TensorQ{}));
  using StrideK = decltype(stride(typename CollectiveMainloop::TensorK{}));
  using StrideV = decltype(stride(typename CollectiveMainloop::TensorV{}));
  using SGPerWG = typename CollectiveMainloop::SGPerWG;

  using FragA = typename CollectiveMainloop::FragA;
  using FragARow = typename CollectiveMainloop::FragARow;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  using TileShapeO = typename CollectiveEpilogue::TileShapeO;
  using ElementO = typename CollectiveEpilogue::TensorO::element_type;
  using StrideO = decltype(stride(typename CollectiveEpilogue::TensorO{}));

  // Tile scheduler derived types
  using TileScheduler = TileScheduler_;
  using TileSchedulerParams = typename TileScheduler::Params;

  //
  // Kernel level shared memory storage
  //
  using MainloopSharedStorage = typename CollectiveMainloop::SharedStorage;
  using EpilogueSharedStorage = typename CollectiveEpilogue::SharedStorage;
  union SharedStorage {
    MainloopSharedStorage mainloop;
    EpilogueSharedStorage epilogue;
  };
  static constexpr int SharedStorageSize = is_empty_v<SharedStorage> ? size_t(0) : sizeof(SharedStorage);

  static constexpr bool is_split_kv = false;

  //
  // KernelArguments
  //
  struct KernelArguments {
    ProblemShape shape{};
    // Q tensors
    const ElementQ* Q_nope = nullptr;
    StrideQ dQ_nope{};
    const ElementQ* Q_pe = nullptr;
    StrideQ dQ_pe{};

    // K/V tensors (base pointer for paged KV cache)
    const ElementK* K = nullptr;
    StrideK dK{};
    const ElementK* K_pe = nullptr;
    StrideK dK_pe{};

    const ElementV* V = nullptr;
    StrideV dV{};

    // output tensor
    ElementO* O = nullptr;
    StrideO dO{};

    // Sequence lengths per batch (for computing total_blk)
    const int* seq_lens = nullptr;

    // Default constructor
    KernelArguments() = default;
  };

  //
  // KernelParams
  //
  using KernelParams = KernelArguments;

  //
  // Arguments
  //
  struct Arguments {
    KernelArguments kernel{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    int split_kv = -1;
    int* ptr_split_kv = nullptr;
  };

  //
  // Params
  //
  struct Params {
    KernelParams kernel{};
    MainloopParams mainloop{};
    EpilogueParams epilogue{};
    TileSchedulerParams scheduler{};
    int split_kv = -1;
    int* ptr_split_kv = nullptr;
  };

  //
  // Methods
  //
  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    return {
        args.kernel,
        CollectiveMainloop::to_underlying_arguments(args.mainloop, workspace),
        CollectiveEpilogue::to_underlying_arguments(args.epilogue, workspace),
        TileScheduler::to_underlying_arguments(args.kernel.shape, args.hw_info, TileShapeO{}, args.split_kv)};
  }

  static bool can_implement(Arguments const& args) {
    return CollectiveMainloop::can_implement(args.mainloop) && CollectiveEpilogue::can_implement(args.epilogue);
  }

  static int get_workspace_size(Arguments const& args) {
    if (args.split_kv > 1) {
      assert(false && "Split-K workspace size is calculated in different path");
    }
    return 0;
  }

  static cutlass::Status initialize_workspace(Arguments const& args, void* workspace) {
    return Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::template get_grid_shape<SGPerWG::value>(params.scheduler);
  }

  static dim3 get_block_shape() {
    return dim3(SGPerWG::value * intel::sg_size, 1, 1);
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    using namespace sycl::ext::oneapi::this_work_item;

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    auto& p = params.kernel;
    ProblemShape const& s = p.shape;

    int thr_id = int(ThreadIdxX());

    TileScheduler tile_scheduler{params.scheduler};

    CUTLASS_PRAGMA_NO_UNROLL
    for (; tile_scheduler.is_valid(); ++tile_scheduler) {
      auto [blk_q, blk_v, head_coord, batch_coord, unused_split] = tile_scheduler.get_block_coord();
      auto blk_qv = make_coord(blk_q, blk_v);

      // Get actual kv sequence length for this batch
      int seq_len_kv = p.seq_lens[batch_coord];
      int total_blk = cute::ceil_div(seq_len_kv, get<1>(TileShapeQK{}));

      auto shape_Q_nope = make_shape(s.seq_len_qo, s.head_size_q_nope, s.num_heads_q, s.batch);
      auto shape_Q_pe = make_shape(s.seq_len_qo, s.head_size_q_pe, s.num_heads_q, s.batch);
      auto shape_O = make_shape(s.seq_len_qo, s.head_size_o, s.num_heads_q, s.batch);

      auto dcQ_nope = const_cast<ElementQ*>(p.Q_nope);
      auto dcQ_pe = const_cast<ElementQ*>(p.Q_pe);

      Tensor Q_nope = make_tensor(make_gmem_ptr(dcQ_nope), make_layout(shape_Q_nope, p.dQ_nope));
      Tensor Q_pe = make_tensor(make_gmem_ptr(dcQ_pe), make_layout(shape_Q_pe, p.dQ_pe));
      Tensor O = make_tensor(make_gmem_ptr(p.O), make_layout(shape_O, p.dO));

      // O accumulator types
      FragA tArA;
      FragARow tA_max, tA_sum;

      // Create KV tensors for paged KV cache
      // Shape: (page_size, head_size, num_heads_kv, num_pages)
      // The mainloop will handle page table lookup internally
      // Note: page_size comes from mainloop params
      int page_size = params.mainloop.page_size;
      auto shape_K = make_shape(page_size, s.head_size_kv, s.total_page, _1{});
      auto shape_K_pe = make_shape(page_size, s.head_size_k_pe, s.total_page, _1{});

      auto shape_V = make_shape(s.head_size_kv, page_size, s.total_page, _1{});

      auto dcK = const_cast<ElementK*>(p.K);
      auto dcK_pe = const_cast<ElementK*>(p.K_pe);

      Tensor K = make_tensor(make_gmem_ptr(dcK), make_layout(shape_K, p.dK));              // (k,d,h,page)
      Tensor K_pe = make_tensor(make_gmem_ptr(dcK_pe), make_layout(shape_K_pe, p.dK_pe));  // (k,d,h,page)
      Tensor V = make_tensor(make_gmem_ptr(dcK), make_layout(shape_V, p.dV));

      CollectiveMainloop mainloop(params.mainloop, shared_storage.mainloop);
      mainloop(
          Q_nope(_, _, head_coord, batch_coord),
          Q_pe(_, _, head_coord, batch_coord),
          K(_, _, _, _0{}),
          K_pe(_, _, _, _0{}),
          V(_, _, _, _0{}),
          tArA,
          tA_max,
          tA_sum,
          blk_qv,
          0,          // blk_k0: start from first K block
          total_blk,  // blk_k1: end at last K block
          total_blk,  // total_blk: total number of K blocks
          thr_id,
          seq_len_kv,
          batch_coord);  // batch index for page table lookup

      if constexpr (!is_empty_v<MainloopSharedStorage> && !is_empty_v<EpilogueSharedStorage>) {
        sycl::group_barrier(get_work_group<3>());
      }

      CollectiveEpilogue epilogue(params.epilogue, shared_storage.epilogue);
      epilogue(O(_, _, head_coord, batch_coord), tArA, tA_max, tA_sum, blk_qv, thr_id);
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
template <class ProblemShape_, class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_>
class XeMlaSplitKVKernel {
 public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  using TileShapeQK = typename CollectiveMainloop::TileShapeQK;
  using TileShapePV = typename CollectiveMainloop::TileShapePV;

  using ElementQ = typename CollectiveMainloop::TensorQ::element_type;
  using ElementK = typename CollectiveMainloop::TensorK::element_type;
  using ElementV = typename CollectiveMainloop::TensorV::element_type;

  using StrideQ = decltype(stride(typename CollectiveMainloop::TensorQ{}));
  using StrideK = decltype(stride(typename CollectiveMainloop::TensorK{}));
  using StrideV = decltype(stride(typename CollectiveMainloop::TensorV{}));
  using SGPerWG = typename CollectiveMainloop::SGPerWG;

  using FragA = typename CollectiveMainloop::FragA;
  using FragARow = typename CollectiveMainloop::FragARow;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  using TileShapeO = typename CollectiveEpilogue::TileShapeO;
  using ElementO = typename CollectiveEpilogue::TensorO::element_type;
  using StrideO = decltype(stride(typename CollectiveEpilogue::TensorO{}));

  // Tile scheduler derived types
  using TileScheduler = TileScheduler_;
  using TileSchedulerParams = typename TileScheduler::Params;

  //
  // Kernel level shared memory storage
  //
  using MainloopSharedStorage = typename CollectiveMainloop::SharedStorage;
  using EpilogueSharedStorage = typename CollectiveEpilogue::SharedStorage;
  union SharedStorage {
    MainloopSharedStorage mainloop;
    EpilogueSharedStorage epilogue;
  };
  static constexpr int SharedStorageSize = is_empty_v<SharedStorage> ? size_t(0) : sizeof(SharedStorage);

  static constexpr bool is_split_kv = true;
  // TODO: change the max_num_kv_split to `SGPerWG::value * intel::sg_size`
  static constexpr int kvMaxSplits = 128;

  // Accumulator element type for exp_sums/max_logits
  using ElementAcc = float;

  //
  // KernelArguments
  //
  struct KernelArguments {
    ProblemShape shape{};
    // Q tensors
    const ElementQ* Q_nope = nullptr;
    StrideQ dQ_nope{};
    const ElementQ* Q_pe = nullptr;
    StrideQ dQ_pe{};

    // K/V tensors (base pointer for paged KV cache)
    const ElementK* K = nullptr;
    StrideK dK{};
    const ElementK* K_pe = nullptr;
    StrideK dK_pe{};
    const ElementV* V = nullptr;
    StrideV dV{};

    // Split-KV partial output
    ElementO* O_accum = nullptr;  // (batch, num_heads, num_kv_splits, head_size_o)
    StrideO dO_accum{};

    // Softmax statistics
    ElementAcc* exp_sums = nullptr;    // (batch, num_heads, num_kv_splits)
    ElementAcc* max_logits = nullptr;  // (batch, num_heads, num_kv_splits)
    StrideO dLSE{};

    // Final output
    ElementO* O = nullptr;
    StrideO dO{};

    // Sequence lengths per batch (for computing total_blk)
    const int* seq_lens = nullptr;

    KernelArguments() = default;
  };

  //
  // KernelParams
  //
  using KernelParams = KernelArguments;

  //
  // Arguments
  //
  struct Arguments {
    KernelArguments kernel{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    int split_kv = -1;
  };

  struct Params {
    KernelParams kernel{};
    MainloopParams mainloop{};
    EpilogueParams epilogue{};
    TileSchedulerParams scheduler{};
  };

  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    return {
        args.kernel,
        CollectiveMainloop::to_underlying_arguments(args.mainloop, workspace),
        CollectiveEpilogue::to_underlying_arguments(args.epilogue, workspace),
        TileScheduler::to_underlying_arguments(args.kernel.shape, args.hw_info, TileShapeO{}, args.split_kv)};
  }

  static bool can_implement(Arguments const& args) {
    return CollectiveMainloop::can_implement(args.mainloop) && CollectiveEpilogue::can_implement(args.epilogue);
  }

  static int get_workspace_size(Arguments const& args) {
    int splits = args.split_kv;
    if (splits <= 1) {
      assert(false && "non Split-K workspace size is calculated in different path");
    }
    auto const& s = args.kernel.shape;
    SplitKVWorkspaceLayout ws(s.batch, s.num_heads_q, splits, s.head_size_o, sizeof(ElementO));
    return ws.total_bytes;
  }

  static cutlass::Status initialize_workspace(Arguments const& args, void* workspace) {
    return Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::template get_grid_shape<SGPerWG::value>(params.scheduler);
  }

  static dim3 get_block_shape() {
    return dim3(SGPerWG::value * intel::sg_size, 1, 1);
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    using namespace sycl::ext::oneapi::this_work_item;

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    auto& p = params.kernel;
    ProblemShape const& s = p.shape;
    int num_kv_splits = params.scheduler.num_kv_splits_;

    int thr_id = int(ThreadIdxX());

    TileScheduler tile_scheduler{params.scheduler};

    // LSE statistics: (seq_len_qo, num_kv_splits, num_heads_q, batch)
    auto shape_lse = make_shape(s.seq_len_qo, num_kv_splits, s.num_heads_q, s.batch);
    Tensor gExpSums = make_tensor(make_gmem_ptr(p.exp_sums), make_layout(shape_lse, p.dLSE));
    Tensor gMaxLogits = make_tensor(make_gmem_ptr(p.max_logits), make_layout(shape_lse, p.dLSE));

    CUTLASS_PRAGMA_NO_UNROLL
    for (; tile_scheduler.is_valid(); ++tile_scheduler) {
      auto [blk_q, blk_v, head_coord, batch_coord, idx_kv_split] = tile_scheduler.get_block_coord();
      auto blk_qv = make_coord(blk_q, blk_v);

      // Get actual kv sequence length for this batch
      int seq_len_kv = p.seq_lens[batch_coord];
      int total_blk = cute::ceil_div(seq_len_kv, get<1>(TileShapeQK{}));

      // Compute this split's K-block range
      int num_blocks_per_split = cute::ceil_div(total_blk, num_kv_splits);
      int start_blk = idx_kv_split * num_blocks_per_split;
      int end_blk = cute::min(start_blk + num_blocks_per_split, total_blk);

      // Empty split: write zero-contribution sentinels and skip.
      // exp_sums=0 gates any garbage in O_accum for this split.
      // max_logits=lowest() ensures empty splits don't pollute global_max.
      if (start_blk >= total_blk) {
        if (thr_id == 0) {
          gExpSums(0, idx_kv_split, head_coord, batch_coord) = ElementAcc(0);
          gMaxLogits(0, idx_kv_split, head_coord, batch_coord) = std::numeric_limits<ElementAcc>::lowest();
        }
        continue;
      }

      auto shape_Q_nope = make_shape(s.seq_len_qo, s.head_size_q_nope, s.num_heads_q, s.batch);
      auto shape_Q_pe = make_shape(s.seq_len_qo, s.head_size_q_pe, s.num_heads_q, s.batch);

      auto dcQ_nope = const_cast<ElementQ*>(p.Q_nope);
      auto dcQ_pe = const_cast<ElementQ*>(p.Q_pe);

      Tensor Q_nope = make_tensor(make_gmem_ptr(dcQ_nope), make_layout(shape_Q_nope, p.dQ_nope));
      Tensor Q_pe = make_tensor(make_gmem_ptr(dcQ_pe), make_layout(shape_Q_pe, p.dQ_pe));

      // Initialize accumulators for this split (needed for non-first splits
      // since the mainloop only initializes when blk_k0 == 0)
      FragA tArA;
      FragARow tA_max, tA_sum;
      clear(tArA);
      fill(tA_max, cutlass::platform::numeric_limits<typename FragARow::element_type>::lowest());
      clear(tA_sum);

      int page_size = params.mainloop.page_size;
      auto shape_K = make_shape(page_size, s.head_size_kv, s.total_page, _1{});
      auto shape_K_pe = make_shape(page_size, s.head_size_k_pe, s.total_page, _1{});
      auto shape_V = make_shape(s.head_size_kv, page_size, s.total_page, _1{});

      auto dcK = const_cast<ElementK*>(p.K);
      auto dcK_pe = const_cast<ElementK*>(p.K_pe);

      Tensor K = make_tensor(make_gmem_ptr(dcK), make_layout(shape_K, p.dK));
      Tensor K_pe = make_tensor(make_gmem_ptr(dcK_pe), make_layout(shape_K_pe, p.dK_pe));
      Tensor V = make_tensor(make_gmem_ptr(dcK), make_layout(shape_V, p.dV));

      CollectiveMainloop mainloop(params.mainloop, shared_storage.mainloop);
      mainloop(
          Q_nope(_, _, head_coord, batch_coord),
          Q_pe(_, _, head_coord, batch_coord),
          K(_, _, _, _0{}),
          K_pe(_, _, _, _0{}),
          V(_, _, _, _0{}),
          tArA,
          tA_max,
          tA_sum,
          blk_qv,
          start_blk,
          end_blk,
          total_blk,
          thr_id,
          seq_len_kv,
          batch_coord);

      if constexpr (!is_empty_v<MainloopSharedStorage> && !is_empty_v<EpilogueSharedStorage>) {
        sycl::group_barrier(get_work_group<3>());
      }

      // Write partial O and LSE statistics via split-KV epilogue
      auto shape_Oaccum = make_shape(s.seq_len_qo, s.head_size_o, s.num_heads_q * num_kv_splits, s.batch);
      Tensor Oaccum = make_tensor(make_gmem_ptr(p.O_accum), make_layout(shape_Oaccum, p.dO_accum));
      int head_split_coord = head_coord * num_kv_splits + idx_kv_split;

      CollectiveEpilogue epilogue(params.epilogue, shared_storage.epilogue);
      epilogue(
          Oaccum(_, _, head_split_coord, batch_coord),
          tArA,
          tA_max,
          tA_sum,
          blk_qv,
          thr_id,
          gExpSums(0, idx_kv_split, head_coord, batch_coord),
          gMaxLogits(0, idx_kv_split, head_coord, batch_coord),
          num_kv_splits);
    }
  }
};

}  // namespace cutlass::flash_attention::kernel
