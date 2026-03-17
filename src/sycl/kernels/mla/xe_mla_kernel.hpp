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
#include "mla_tile_scheduler.hpp"
#include "xe_mla_epilogue.hpp"
#include "xe_mla_mainloop.hpp"

namespace cutlass::flash_attention::kernel {
using namespace cute;

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

  static constexpr int QK_BLK_M = CollectiveMainloop::QK_BLK_M;
  static constexpr int QK_BLK_N = CollectiveMainloop::QK_BLK_N;
  static constexpr int QK_BLK_K = CollectiveMainloop::QK_BLK_K;

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
        TileScheduler::to_underlying_arguments(args.kernel.shape, args.hw_info, TileShapeO{})};
  }

  static bool can_implement(Arguments const& args) {
    return CollectiveMainloop::can_implement(args.mainloop) && CollectiveEpilogue::can_implement(args.epilogue);
    // return true; // change as needed
  }

  static int get_workspace_size(Arguments const& args) {
    // If no split-K, no workspace needed
    if (args.split_kv <= 1) {
      return 0;
    }

    // TODO: For split-K calculate and return workspace size
    assert(false && "Split-K workspace size calculation not implemented yet.");
    return -1;
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
    int head_group_q = s.num_heads_q / s.num_heads_kv;

    int thr_id = int(ThreadIdxX());

    TileScheduler tile_scheduler{params.scheduler};

    CUTLASS_PRAGMA_NO_UNROLL
    for (; tile_scheduler.is_valid(); ++tile_scheduler) {
      auto [blk_q, blk_v, head_coord, batch_coord] = tile_scheduler.get_block_coord();
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
}  // namespace cutlass::flash_attention::kernel
