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
    \brief XPU Sparse MLA Kernel
*/

#pragma once
#include "cute/util/type_traits.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "sycl/kernels/mla_sparse/collective/xe_mla_sparse_epilogue.hpp"
#include "sycl/kernels/mla_sparse/collective/xe_mla_sparse_mainloop.hpp"
#include "sycl/kernels/mla_sparse/kernel/mla_sparse_tile_scheduler.hpp"

namespace cutlass::flash_attention::kernel {
using namespace cute;

///////////////////////////////////////////////////////////////////////////////
template <class ProblemShape_, class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_>
class XeMlaSparseFwdKernel {
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
  using StrideKV = decltype(stride(typename CollectiveMainloop::TensorK{}));
  using StrideKV_V = decltype(stride(typename CollectiveMainloop::TensorV{}));

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

    // Q nope tensor
    const ElementQ* Q = nullptr;
    StrideQ dQ{};

    // Q rope tensor
    const ElementQ* Q_pe = nullptr;
    StrideQ dQ_pe{};

    // Primary K nope (FP8)
    const ElementK* K_nope = nullptr;
    StrideKV dK_nope{};

    // Primary K rope (bf16)
    const ElementQ* K_pe = nullptr;
    StrideKV dK_pe{};

    // Primary KV scale (uint8)
    const uint8_t* KV_scale = nullptr;
    StrideKV dKV_scale{};

    // Primary V nope (FP8)
    const ElementK* V_nope = nullptr;
    StrideKV_V dV_nope{};

    // Primary V rope (bf16)
    const ElementQ* V_pe = nullptr;
    StrideKV_V dV_pe{};

    // Extra K nope (FP8)
    const ElementK* K_nope_extra = nullptr;
    StrideKV dK_nope_extra{};

    // Extra K rope (bf16)
    const ElementQ* K_pe_extra = nullptr;
    StrideKV dK_pe_extra{};

    // Extra KV scale (uint8)
    const uint8_t* KV_scale_extra = nullptr;
    StrideKV dKV_scale_extra{};

    // Extra V nope (FP8)
    const ElementK* V_nope_extra = nullptr;
    StrideKV_V dV_nope_extra{};

    // Extra V rope (bf16)
    const ElementQ* V_pe_extra = nullptr;
    StrideKV_V dV_pe_extra{};

    // Output tensor
    ElementO* O = nullptr;
    StrideO dO{};

    // LSE output
    float* lse_out = nullptr;

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
  };

  //
  // Params
  //
  struct Params {
    KernelParams kernel{};
    MainloopParams mainloop{};
    EpilogueParams epilogue{};
    TileSchedulerParams scheduler{};
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
  }

  static int get_workspace_size(Arguments const& args) {
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

  static constexpr int HEADS_PER_WG = CollectiveMainloop::HEADS_PER_WG;

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
      auto [blk_q, blk_v, head_group, batch_coord] = tile_scheduler.get_block_coord();
      auto blk_qv = make_coord(blk_q, blk_v);

      // head_group is the group index; actual heads are [head_group*N .. head_group*N + N-1]
      int head_base = head_group * HEADS_PER_WG;

      // Q tensor: use full 512-dim (448 nope + 64 pe are contiguous in memory).
      // The combined D_SLICE=128 last iteration reads Q[384:512] = Q_nope_tail || Q_pe.
      int q_full_dim = s.head_size_q_nope + s.head_size_q_pe;  // 448 + 64 = 512
      auto shape_Q = make_shape(s.seq_len_qo, q_full_dim, s.num_heads_q, s.batch);
      auto shape_Q_pe = make_shape(s.seq_len_qo, s.head_size_q_pe, s.num_heads_q, s.batch);
      auto shape_O = make_shape(s.seq_len_qo, s.head_size_o, s.num_heads_q, s.batch);

      auto dcQ = const_cast<ElementQ*>(p.Q);
      auto dcQ_pe = const_cast<ElementQ*>(p.Q_pe);

      Tensor Q = make_tensor(make_gmem_ptr(dcQ), make_layout(shape_Q, p.dQ));
      Tensor Q_pe = make_tensor(make_gmem_ptr(dcQ_pe), make_layout(shape_Q_pe, p.dQ_pe));
      Tensor O = make_tensor(make_gmem_ptr(p.O), make_layout(shape_O, p.dO));

      int page_size = s.page_size;
      int extra_page_size = s.extra_page_size;

      auto shape_K_nope = make_shape(page_size, s.head_size_k, s.total_page, _1{});
      Tensor K_nope = make_tensor(make_gmem_ptr(const_cast<ElementK*>(p.K_nope)), make_layout(shape_K_nope, p.dK_nope));

      auto shape_K_pe = make_shape(page_size, s.head_size_q_pe, s.total_page, _1{});
      Tensor K_pe = make_tensor(make_gmem_ptr(const_cast<ElementQ*>(p.K_pe)), make_layout(shape_K_pe, p.dK_pe));

      auto shape_KV_scale = make_shape(page_size, 8, s.total_page, _1{});
      Tensor KV_scale =
          make_tensor(make_gmem_ptr(const_cast<uint8_t*>(p.KV_scale)), make_layout(shape_KV_scale, p.dKV_scale));

      auto shape_V_nope = make_shape(s.head_size_k, page_size, s.total_page, _1{});
      Tensor V_nope = make_tensor(make_gmem_ptr(const_cast<ElementK*>(p.V_nope)), make_layout(shape_V_nope, p.dV_nope));

      auto shape_V_pe = make_shape(s.head_size_q_pe, page_size, s.total_page, _1{});
      Tensor V_pe = make_tensor(make_gmem_ptr(const_cast<ElementQ*>(p.V_pe)), make_layout(shape_V_pe, p.dV_pe));

      auto shape_K_nope_extra = make_shape(extra_page_size, s.head_size_k, s.total_extra_page, _1{});
      Tensor K_nope_extra = make_tensor(
          make_gmem_ptr(const_cast<ElementK*>(p.K_nope_extra)), make_layout(shape_K_nope_extra, p.dK_nope_extra));

      auto shape_K_pe_extra = make_shape(extra_page_size, s.head_size_q_pe, s.total_extra_page, _1{});
      Tensor K_pe_extra =
          make_tensor(make_gmem_ptr(const_cast<ElementQ*>(p.K_pe_extra)), make_layout(shape_K_pe_extra, p.dK_pe_extra));

      auto shape_KV_scale_extra = make_shape(extra_page_size, 8, s.total_extra_page, _1{});
      Tensor KV_scale_extra = make_tensor(
          make_gmem_ptr(const_cast<uint8_t*>(p.KV_scale_extra)), make_layout(shape_KV_scale_extra, p.dKV_scale_extra));

      auto shape_V_nope_extra = make_shape(s.head_size_k, extra_page_size, s.total_extra_page, _1{});
      Tensor V_nope_extra = make_tensor(
          make_gmem_ptr(const_cast<ElementK*>(p.V_nope_extra)), make_layout(shape_V_nope_extra, p.dV_nope_extra));

      auto shape_V_pe_extra = make_shape(s.head_size_q_pe, extra_page_size, s.total_extra_page, _1{});
      Tensor V_pe_extra =
          make_tensor(make_gmem_ptr(const_cast<ElementQ*>(p.V_pe_extra)), make_layout(shape_V_pe_extra, p.dV_pe_extra));

      // Build per-head Q arrays and accumulators
      using TensorQ2D = typename CollectiveMainloop::TensorQ2D;
      TensorQ2D Q_arr[HEADS_PER_WG];
      TensorQ2D Q_pe_arr[HEADS_PER_WG];
      CUTLASS_PRAGMA_UNROLL
      for (int h = 0; h < HEADS_PER_WG; h++) {
        Q_arr[h] = Q(_, _, head_base + h, batch_coord);
        Q_pe_arr[h] = Q_pe(_, _, head_base + h, batch_coord);
      }

      FragA tArA[HEADS_PER_WG];
      FragARow tA_max[HEADS_PER_WG], tA_sum[HEADS_PER_WG];

      CollectiveMainloop mainloop(params.mainloop, shared_storage.mainloop);
      mainloop(
          Q_arr,
          Q_pe_arr,
          K_nope(_, _, _, _0{}),
          K_pe(_, _, _, _0{}),
          KV_scale(_, _, _, _0{}),
          V_nope(_, _, _, _0{}),
          V_pe(_, _, _, _0{}),
          K_nope_extra(_, _, _, _0{}),
          K_pe_extra(_, _, _, _0{}),
          KV_scale_extra(_, _, _, _0{}),
          V_nope_extra(_, _, _, _0{}),
          V_pe_extra(_, _, _, _0{}),
          tArA,
          tA_max,
          tA_sum,
          blk_qv,
          thr_id,
          batch_coord);

      // Epilogue: write output for each head
      CUTLASS_PRAGMA_UNROLL
      for (int h = 0; h < HEADS_PER_WG; h++) {
        if constexpr (!is_empty_v<MainloopSharedStorage> && !is_empty_v<EpilogueSharedStorage>) {
          sycl::group_barrier(get_work_group<3>());
        }
        CollectiveEpilogue epilogue(params.epilogue, shared_storage.epilogue);
        epilogue(
            O(_, _, head_base + h, batch_coord),
            tArA[h],
            tA_max[h],
            tA_sum[h],
            blk_qv,
            thr_id,
            head_base + h,
            batch_coord);
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
}  // namespace cutlass::flash_attention::kernel
