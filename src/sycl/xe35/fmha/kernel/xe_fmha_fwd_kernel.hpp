/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
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

#include "../collective/fmha_fusion.hpp"
#include "../collective/xe_fmha_fwd_epilogue.hpp"
#include "../collective/xe_fmha_fwd_mainloop.hpp"
#include "cute/util/type_traits.hpp"
#include "cute/util/xe_split_barrier.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "xe_tile_scheduler.hpp"

namespace cutlass::fmha_xe35::kernel {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
template <bool IsVarLen_ = false>
struct FMHAProblemShape_xe35 {
  using SeqLenType = cute::conditional_t<IsVarLen_, cutlass::fmha_xe35::collective::VariableLength_xe35, int>;
  int batch;
  int num_heads_q, num_heads_kv;
  SeqLenType seq_len_qo, seq_len_kv, seq_len_kv_cache;
  int head_size_qk, head_size_vo;
};

///////////////////////////////////////////////////////////////////////////////

// Same result as T(_,_,i2,i3), but the head/batch offset is accumulated in 64bits. The packed
// strides are 32-bit, so i3 * stride<3> wraps once a K/V/O tensor exceeds 2^31 elements.
template <class Tensor>
CUTLASS_HOST_DEVICE auto slice_2d_wide_xe35(Tensor const& T, int i2, int i3) {
  using namespace cute;
  auto layout = T.layout();
  int64_t const offset = static_cast<int64_t>(i2) * static_cast<int64_t>(stride<2>(layout)) +
                         static_cast<int64_t>(i3) * static_cast<int64_t>(stride<3>(layout));
  auto sliced =
      make_layout(make_shape(shape<0>(layout), shape<1>(layout)), make_stride(stride<0>(layout), stride<1>(layout)));
  return make_tensor(T.data() + offset, sliced);
}

template <
    class ProblemShape_,
    class CollectiveMainloop_,
    class CollectiveEpilogue_,
    class TileScheduler_,
    bool SplitKV_ = false>
class XeFMHAFwdKernel_xe35 {
 public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;
  using VariableLength_xe35 = cutlass::fmha_xe35::collective::VariableLength_xe35;
  static constexpr bool is_var_len = cutlass::fmha_xe35::collective::is_variable_length_v<typename ProblemShape::SeqLenType>;
  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  using TiledMMAQK = typename CollectiveMainloop::TiledMMAQK;
  using TiledMMAPV = typename CollectiveMainloop::TiledMMAPV;
  using TileShapeQK = typename CollectiveMainloop::TileShapeQK;
  using TileShapePV = typename CollectiveMainloop::TileShapePV;
  using SubgroupLayoutQK = typename CollectiveMainloop::SubgroupLayoutQK;
  using ElementQ = typename CollectiveMainloop::TensorQ::element_type;
  using ElementK = typename CollectiveMainloop::TensorK::element_type;
  using ElementV = typename CollectiveMainloop::TensorV::element_type;
  using ElementScale = typename CollectiveMainloop::TensorScaleQ::element_type;
  using StrideScaleQ = decltype(stride(typename CollectiveMainloop::TensorScaleQ{}));
  using StrideScaleK = decltype(stride(typename CollectiveMainloop::TensorScaleK{}));
  using StrideScaleV = decltype(stride(typename CollectiveMainloop::TensorScaleV{}));
  using StrideQ = decltype(stride(typename CollectiveMainloop::TensorQ{}));
  using StrideK = decltype(stride(typename CollectiveMainloop::TensorK{}));
  using StrideV = decltype(stride(typename CollectiveMainloop::TensorV{}));
  static constexpr bool BlockScale = CollectiveMainloop::BlockScale;

  using SGPerWG = typename CollectiveMainloop::SGPerWG;

  using FragA = typename CollectiveMainloop::FragA;
  using FragARow = typename CollectiveMainloop::FragARow;
  using FragSPartialRow = typename CollectiveMainloop::FragSPartialRow;

  // Tile scheduler derived types
  using TileScheduler = TileScheduler_;
  using TileSchedulerParams = typename TileScheduler::Params;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  using TileShapeO = typename CollectiveEpilogue::TileShapeO;
  using ElementO = typename CollectiveEpilogue::TensorO::element_type;
  using StrideO = decltype(stride(typename CollectiveEpilogue::TensorO{}));
  static constexpr int max_num_kv_splits = SGPerWG::value * intel::sg_size;

  // Kernel level shared memory storage
  using MainloopSharedStorage = typename CollectiveMainloop::SharedStorage;
  using EpilogueSharedStorage = typename CollectiveEpilogue::SharedStorage;
  union SharedStorage {
    MainloopSharedStorage mainloop;
    EpilogueSharedStorage epilogue;
  };

  static constexpr int SharedStorageSize = is_empty_v<SharedStorage> ? size_t(0) : sizeof(SharedStorage);

  // Device side arguments
  struct KernelArguments {
    ProblemShape shape;
    const ElementQ* Q;
    StrideQ dQ;
    const ElementK* K;
    StrideK dK;
    const ElementV* V;
    StrideV dV;
    ElementO* O;
    StrideO dO;
    const ElementScale* scaleQ = nullptr;
    StrideScaleQ dScaleQ{};
    const ElementScale* scaleK = nullptr;
    StrideScaleK dScaleK{};
    const ElementScale* scaleV = nullptr;
    StrideScaleV dScaleV{};
    float scale_q = 1.f;
    float scale_k = 1.f;
    float scale_v = 1.f;
    int group_size = 32;
    const ElementK* K_cache;
    StrideK dK_cache{};
    const ElementV* V_cache;
    StrideV dV_cache{};
    const ElementQ* sm_sink = nullptr;
    ElementO* Oaccum = nullptr;
    StrideO dOaccum{};
    float* exp_sums = nullptr;
    StrideO dExp_sums{};
    float* max_logits = nullptr;
    StrideO dMax_logits{};
  };
  using KernelParams = KernelArguments;

  struct Arguments {
    KernelArguments kernel{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    int num_kv_splits = -1;
  };

  // Kernel entry point API
  struct Params {
    KernelParams kernel;
    MainloopParams mainloop;
    EpilogueParams epilogue;
    TileSchedulerParams scheduler;
  };

  //
  // Methods
  //

  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    return {
        args.kernel,
        CollectiveMainloop::to_underlying_arguments(args.mainloop, workspace),
        CollectiveEpilogue::to_underlying_arguments(args.epilogue, workspace),
        TileScheduler::to_underlying_arguments(
            args.kernel.shape, args.hw_info, TileShapeO{}, SplitKV_ ? args.num_kv_splits : -1)};
  }

  static bool can_implement(Arguments const& args) {
    if constexpr (SplitKV_) {
      if (args.num_kv_splits < 1 || args.num_kv_splits > max_num_kv_splits) return false;
    }
    return CollectiveMainloop::can_implement(args.mainloop) && CollectiveEpilogue::can_implement(args.epilogue);
  }

  static int get_workspace_size(Arguments const& args) {
    return 0;
  }

  static cutlass::Status initialize_workspace(
      Arguments const& args,
      void* workspace = nullptr,
      cudaStream_t stream = nullptr,
      CudaHostAdapter* cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::template get_grid_shape<SGPerWG::value>(params.scheduler);
  }

  static dim3 get_block_shape() {
    return dim3(SGPerWG::value * intel::sg_size, 1, 1);
  }

  CUTLASS_DEVICE
  Shape<int, int, int> get_sequence_length_shape(ProblemShape const& problem_shape, int const& batch) {
    if constexpr (is_var_len) {
      int seq_len_q =
          problem_shape.seq_len_qo.cumulative_length[batch + 1] - problem_shape.seq_len_qo.cumulative_length[batch];
      int seq_len_k_new = 0;
      // Match Xe20 paged-KV handling: cu_seqlens_k carries per-batch cache lengths
      // directly in paged mode, while non-paged mode uses a cumulative prefix sum.
      int seq_len_k_cache = 0;
      if constexpr (CollectiveMainloop::CachedKV) {
        if constexpr (CollectiveMainloop::PagedKV) {
          seq_len_k_cache = problem_shape.seq_len_kv_cache.cumulative_length[batch];
        } else {
          seq_len_k_cache = problem_shape.seq_len_kv_cache.cumulative_length[batch + 1] -
                            problem_shape.seq_len_kv_cache.cumulative_length[batch];
        }
      } else {
        // Plain attention uses the new-KV buffer for the contiguous ragged KV.
        // The cache slot stays empty so the local-window / offset math remains in
        // the same coordinates as the upstream FMHA configuration.
        // cu_seqlens_k is cumulative (size batch+1), so per-batch length is the diff.
        seq_len_k_new =
            problem_shape.seq_len_kv.cumulative_length[batch + 1] - problem_shape.seq_len_kv.cumulative_length[batch];
        seq_len_k_cache = 0;
      }
      return cute::make_tuple<int, int, int>(seq_len_q, seq_len_k_new, seq_len_k_cache);
    } else {
      return Shape<int, int, int>{problem_shape.seq_len_qo, problem_shape.seq_len_kv, problem_shape.seq_len_kv_cache};
    }
  }

  CUTLASS_DEVICE void split_kv_operator(Params const& params, char* smem_buf) {
    using namespace sycl::ext::oneapi::this_work_item;
    auto& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    auto& p = params.kernel;
    const int head_group_q = p.shape.num_heads_q / p.shape.num_heads_kv;
    const int num_kv_splits = params.scheduler.num_kv_splits;
    const int thr_id = int(ThreadIdxX());
    TileScheduler tile_scheduler{params.scheduler};

    for (; tile_scheduler.is_valid(); ++tile_scheduler) {
      auto [blk_q, blk_v, head, idx_b, idx_kv_split] = tile_scheduler.get_block_coord();
      if (blk_q != 0) continue;
      auto sequence_lengths = get_sequence_length_shape(p.shape, idx_b);
      const int seq_len_qo = get<0>(sequence_lengths);
      const int seq_len_kv = CollectiveMainloop::CachedKV ? get<2>(sequence_lengths) : get<1>(sequence_lengths);
      if (seq_len_qo <= 0 || seq_len_kv <= 0) continue;

      const int tile_k = get<1>(TileShapeQK{});
      const int k_blocks = cute::ceil_div(seq_len_kv, tile_k);
      const int k_block0 =
          CollectiveMainloop::LocalMask ? cute::max(seq_len_kv - 1 - params.mainloop.window_size_left, 0) / tile_k : 0;
      const int windowed_k_blocks = k_blocks - k_block0;
      constexpr int kMinBlocksForSplit = 16;
      const bool is_single_split = num_kv_splits > 1 && windowed_k_blocks < kMinBlocksForSplit;
      if (is_single_split && idx_kv_split > 0) continue;
      const int blocks_per_split = cute::ceil_div(windowed_k_blocks, num_kv_splits);
      const int split_start = is_single_split ? k_block0 : k_block0 + idx_kv_split * blocks_per_split;
      const int split_blocks = is_single_split
                                   ? windowed_k_blocks
                                   : cute::min(windowed_k_blocks - idx_kv_split * blocks_per_split, blocks_per_split);

      int q_offset = 0;
      int output_offset = 0;
      int lse_offset = 0;
      int k_offset = 0;
      int v_offset = 0;
      if constexpr (is_var_len) {
        auto q_offsets = p.shape.seq_len_qo.cumulative_length;
        q_offset =
            p.shape.num_heads_q * p.shape.head_size_qk * q_offsets[idx_b] + head * head_group_q * p.shape.head_size_qk;
        output_offset = p.shape.num_heads_q * p.shape.head_size_vo * num_kv_splits * q_offsets[idx_b];
        lse_offset = p.shape.num_heads_q * num_kv_splits * q_offsets[idx_b];
        if constexpr (!CollectiveMainloop::PagedKV) {
          auto kv_offsets = p.shape.seq_len_kv.cumulative_length;
          if (kv_offsets != nullptr) {
            k_offset = p.shape.num_heads_kv * p.shape.head_size_qk * kv_offsets[idx_b];
            v_offset = p.shape.num_heads_kv * p.shape.head_size_vo * kv_offsets[idx_b];
          }
        }
      }

      const int batch_dim = is_var_len ? 1 : p.shape.batch;
      const int l_coord = is_var_len ? 0 : idx_b;
      auto shape_q = make_shape(head_group_q, p.shape.head_size_qk, p.shape.num_heads_kv, batch_dim);
      const int kv_len_new = CollectiveMainloop::CachedKV ? 0 : seq_len_kv;
      const int pool_seqlen_kv = CollectiveMainloop::CachedKV ? params.mainloop.total_seqlen_kv : 0;
      auto shape_k = make_shape(kv_len_new, p.shape.head_size_qk, p.shape.num_heads_kv, batch_dim);
      auto shape_v = make_shape(p.shape.head_size_vo, kv_len_new, p.shape.num_heads_kv, batch_dim);
      auto shape_k_cache = make_shape(pool_seqlen_kv, p.shape.head_size_qk, p.shape.num_heads_kv, batch_dim);
      auto shape_v_cache = make_shape(p.shape.head_size_vo, pool_seqlen_kv, p.shape.num_heads_kv, batch_dim);
      auto shape_o = make_shape(head_group_q, p.shape.head_size_vo, p.shape.num_heads_kv, num_kv_splits, batch_dim);
      auto shape_lse = make_shape(head_group_q, num_kv_splits, p.shape.num_heads_kv, batch_dim);

      auto layout_q = make_ordered_layout(shape_q, Step<_1, _0, _2, _3>{});
      auto layout_k = make_layout(shape_k, p.dK);
      auto layout_v = make_layout(shape_v, p.dV);
      auto layout_k_cache = make_layout(shape_k_cache, p.dK_cache);
      auto layout_v_cache = make_layout(shape_v_cache, p.dV_cache);
      auto layout_o = make_ordered_layout(shape_o, Step<_1, _0, _2, _3, _4>{});
      auto layout_lse = make_ordered_layout(shape_lse, Step<_1, _0, _2, _3>{});

      Tensor Q = make_tensor(make_gmem_ptr(const_cast<ElementQ*>(p.Q + q_offset)), layout_q);
      Tensor K = make_tensor(make_gmem_ptr(const_cast<ElementK*>(p.K + k_offset)), layout_k);
      Tensor V = make_tensor(make_gmem_ptr(const_cast<ElementV*>(p.V + v_offset)), layout_v);
      Tensor K_cache = make_tensor(make_gmem_ptr(const_cast<ElementK*>(p.K_cache)), layout_k_cache);
      Tensor V_cache = make_tensor(make_gmem_ptr(const_cast<ElementV*>(p.V_cache)), layout_v_cache);
      Tensor O = make_tensor(make_gmem_ptr(p.Oaccum + output_offset), layout_o);
      Tensor exp_sums = make_tensor(make_gmem_ptr(p.exp_sums + lse_offset), layout_lse);
      Tensor max_logits = make_tensor(make_gmem_ptr(p.max_logits + lse_offset), layout_lse);

      if (split_blocks <= 0) {
        if (thr_id < head_group_q) {
          exp_sums(thr_id, idx_kv_split, head, l_coord) = 0.0f;
        }
        continue;
      }

      FragA tArA;
      FragARow tA_max;
      FragSPartialRow tA_sum;
      auto blk_qv = make_coord(blk_q, blk_v);
      int gqa_fusion_q_per_head = 0;
      if constexpr (CollectiveMainloop::CausalMask || CollectiveMainloop::LocalMask) {
        gqa_fusion_q_per_head = seq_len_qo;
      }
      CollectiveMainloop mainloop(params.mainloop, shared_storage.mainloop);
      mainloop.template operator()<true>(
          Q(_, _, 0, l_coord),
          K(_, _, head, l_coord),
          V(_, _, head, l_coord),
          tArA,
          tA_max,
          tA_sum,
          blk_qv,
          split_start,
          split_start + split_blocks,
          k_blocks,
          thr_id,
          seq_len_kv,
          CollectiveMainloop::CachedKV ? seq_len_kv : 0,
          idx_b,
          seq_len_kv - seq_len_qo,
          0,
          0,
          0,
          gqa_fusion_q_per_head,
          K_cache(_, _, head, l_coord),
          V_cache(_, _, head, l_coord),
          p.scale_q,
          p.scale_k,
          p.scale_v);
      if constexpr (!is_empty_v<MainloopSharedStorage> && !is_empty_v<EpilogueSharedStorage>) {
        sycl::group_barrier(get_work_group<3>());
      }
      CollectiveEpilogue epilogue(params.epilogue, shared_storage.epilogue);
      epilogue(
          O(_, _, head, idx_kv_split, l_coord),
          tArA,
          tA_max,
          tA_sum,
          blk_qv,
          thr_id,
          exp_sums(_, _, head, l_coord),
          max_logits(_, _, head, l_coord),
          idx_kv_split,
          head_group_q,
          num_kv_splits,
          is_single_split);
    }
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    if constexpr (SplitKV_) {
      split_kv_operator(params, smem_buf);
      return;
    }
    using namespace sycl::ext::oneapi::this_work_item;

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    auto& p = params.kernel;
    ProblemShape const& s = p.shape;
    int head_group_q = s.num_heads_q / s.num_heads_kv;

    int thr_id = int(ThreadIdxX());

    TileScheduler tile_scheduler{params.scheduler};

    CUTLASS_PRAGMA_NO_UNROLL
    for (; tile_scheduler.is_valid(); ++tile_scheduler) {
      auto [blk_q, blk_v, head_q, idx_b, unused_split] = tile_scheduler.get_block_coord();  // (Q,V,h,b,split)
      auto blk_qv = make_coord(blk_q, blk_v);
      int head = tile_scheduler.divide_head_group(head_q);

      auto sequence_length_shape = get_sequence_length_shape(s, idx_b);
      auto [seq_len_qo, seq_len_kv, seq_len_kv_cache] = sequence_length_shape;
      if (blk_q * get<0>(TileShapeQK{}) >= seq_len_qo) continue;

      int discard_seq_coord = 0;
      int full_tile_offset = 0;
      int seq_len_new = seq_len_kv;
      int seq_coord = 0;
      const int seq_len_kv_total = seq_len_kv_cache + seq_len_kv;
      int kv_col_lo = 0;
      int kv_cols = seq_len_kv_total;
      if constexpr (CollectiveMainloop::CausalMask || CollectiveMainloop::LocalMask) {
        int q_sg_tile = get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})));
        auto cS = make_identity_tensor(take<0, 2>(TiledMMAQK{}.tile_mnk()));
        auto tScS = TiledMMAQK{}.get_slice(thr_id).partition_C(cS);
        auto q_offset_wi = get<0>(tScS(0));
        auto q_offset_sg = group_broadcast(sycl::ext::oneapi::this_work_item::get_sub_group(), q_offset_wi, 0);

        int offset = CollectiveMainloop::LocalMask ? seq_len_qo : cute::min(seq_len_qo, seq_len_kv_total);
        discard_seq_coord = seq_len_qo - offset;
        full_tile_offset = seq_len_kv_total - offset;
        seq_coord = cute::min(seq_len_qo, (blk_q * get<0>(TileShapeQK{}) + q_offset_sg));

        int cols = full_tile_offset + (seq_coord - discard_seq_coord) + q_sg_tile;
        if constexpr (CollectiveMainloop::LocalMask && !CollectiveMainloop::CausalMask) {
          cols = params.mainloop.window_size_right >= 0 ? cols + params.mainloop.window_size_right : seq_len_kv_total;
        }
        kv_cols = cute::max(0, cute::min(seq_len_kv_total, cols));

        if constexpr (CollectiveMainloop::LocalMask) {
          if (params.mainloop.window_size_left >= 0) {
            kv_col_lo =
                cute::max(0, blk_q * get<0>(TileShapeQK{}) + full_tile_offset - params.mainloop.window_size_left);
          }
        }
      }
      if constexpr (CollectiveMainloop::CachedKV) {
        seq_len_new = cute::max(0, kv_cols - seq_len_kv_cache);
      } else {
        seq_len_new = kv_cols;
      }
      const int seq_len = seq_len_new + seq_len_kv_cache;
      // Compute k_blocks as sum of cache tiles + new tiles to avoid losing new data
      // when seq_len_kv_cache is not a multiple of the tile size.
      int k_blocks;
      if constexpr (CollectiveMainloop::CausalMask || CollectiveMainloop::LocalMask || CollectiveMainloop::CachedKV) {
        const int kblocks_cache =
            CollectiveMainloop::CachedKV ? cute::ceil_div(seq_len_kv_cache, get<1>(TileShapeQK{})) : 0;
        const int cache_cols = cute::min(kv_cols, seq_len_kv_cache);
        const int kblocks_cache_used =
            CollectiveMainloop::CachedKV ? cute::ceil_div(cache_cols, get<1>(TileShapeQK{})) : 0;
        const int kblocks_new = cute::ceil_div(kv_cols - cache_cols, get<1>(TileShapeQK{}));
        k_blocks = kblocks_cache_used + kblocks_new;
      } else {
        // ceil (not floor) so a ragged non-paged seqlen that isn't a multiple of
        // the tile still processes its partial last block (remainder-masked below).
        k_blocks = cute::ceil_div(seq_len, get<1>(TileShapeQK{}));
      }
      int causal_k_block_start = 0;
      if constexpr (CollectiveMainloop::CausalMask && CollectiveMainloop::CachedKV) {
        causal_k_block_start = (seq_coord + full_tile_offset) / get<1>(TileShapeQK{});
      }

      int blk_k0 = 0;
      int blk_k1 = k_blocks;
      if (kv_col_lo > 0) {
        const int tile_k = get<1>(TileShapeQK{});
        const int kblocks_cache = CollectiveMainloop::CachedKV ? cute::ceil_div(seq_len_kv_cache, tile_k) : 0;
        blk_k0 = kv_col_lo >= seq_len_kv_cache ? kblocks_cache + (kv_col_lo - seq_len_kv_cache) / tile_k
                                               : (CollectiveMainloop::CachedKV ? kv_col_lo / tile_k : 0);
        if (blk_k0 >= blk_k1) {
          blk_k0 = 0;
          blk_k1 = 0;
        }
      }

      int offset_q = 0, offset_k = 0, offset_v = 0, offset_o = 0;
      int offset_k_cache = 0, offset_v_cache = 0;
      int batch_dim = s.batch;
      int l_coord = idx_b;
      if constexpr (is_var_len) {
        batch_dim = 1;
        l_coord = 0;
        auto qo_cumulative = s.seq_len_qo.cumulative_length;
        auto kv_cumulative = s.seq_len_kv.cumulative_length;
        offset_q = s.num_heads_q * s.head_size_qk * qo_cumulative[idx_b];
        if constexpr (CollectiveMainloop::PagedKV) {
          // In paged-KV mode, all K/V pointers are indexed via page_table.
          // Keep base pointers at pool start for both new and cache tensors.
          offset_k = 0;
          offset_v = 0;
          offset_k_cache = 0;
          offset_v_cache = 0;
        } else {
          // "New" K/V may be absent (cache-only non-paged, e.g. varlen prefill):
          // its cumulative pointer is then null, so keep the base offset at 0.
          if (kv_cumulative) {
            offset_k = s.num_heads_kv * s.head_size_qk * kv_cumulative[idx_b];
            offset_v = s.num_heads_kv * s.head_size_vo * kv_cumulative[idx_b];
          }
          if (s.seq_len_kv_cache.cumulative_length) {
            auto kv_cumulative_cache = s.seq_len_kv_cache.cumulative_length;
            offset_k_cache = s.num_heads_kv * s.head_size_qk * kv_cumulative_cache[idx_b];
            offset_v_cache = s.num_heads_kv * s.head_size_vo * kv_cumulative_cache[idx_b];
          }
        }
        offset_o = s.num_heads_q * s.head_size_vo * qo_cumulative[idx_b];
      }

      auto shape_Q = make_shape(seq_len_qo, s.head_size_qk, s.num_heads_q, batch_dim);
      auto shape_K = make_shape(seq_len_kv, s.head_size_qk, s.num_heads_kv, batch_dim);
      auto shape_V = make_shape(s.head_size_vo, seq_len_kv, s.num_heads_kv, batch_dim);
      auto shape_O = make_shape(seq_len_qo, s.head_size_vo, s.num_heads_q, batch_dim);

      // Under PagedKV the physical tile index (via page_table) can address any
      // page in the pool, so the cache tensor must span the whole pool rather
      // than just seq_len_kv_cache; otherwise scattered pages get clamped to 0.
      auto pool_seqlen_kv_cache = CollectiveMainloop::PagedKV ? params.mainloop.total_seqlen_kv : seq_len_kv_cache;
      auto shape_K_cache = make_shape(pool_seqlen_kv_cache, s.head_size_qk, s.num_heads_kv, batch_dim);
      auto shape_V_cache = make_shape(s.head_size_vo, pool_seqlen_kv_cache, s.num_heads_kv, batch_dim);

      auto dcQ = const_cast<ElementQ*>(p.Q + offset_q);
      auto dcK = const_cast<ElementK*>(p.K + offset_k);
      auto dcV = const_cast<ElementV*>(p.V + offset_v);
      auto dcK_cache = const_cast<ElementK*>(p.K_cache + offset_k_cache);
      auto dcV_cache = const_cast<ElementV*>(p.V_cache + offset_v_cache);
      auto ptrO = p.O + offset_o;

      StrideQ stride_q = p.dQ;
      StrideK stride_k = p.dK;
      StrideV stride_v = p.dV;
      StrideO stride_o = p.dO;
      StrideK stride_k_cache = p.dK_cache;
      StrideV stride_v_cache = p.dV_cache;
      // Note: do NOT recompute strides with make_cute_packed_stride here. That
      // helper assumes a [num_heads, seq, dim] (head-outermost) packed layout,
      // whereas the host passes standard [total_seq, num_heads, dim] tensors and
      // supplies matching strides in p.dQ/dK/dV/dO. The two only coincide when
      // num_heads == 1; overriding them corrupts multi-head results. The var-len
      // per-batch base offsets (offset_q = num_heads * head_size * cumulative)
      // are already computed for the [total_seq, num_heads, dim] layout, so the
      // host strides are the correct choice for both the new and cache tensors.

      Tensor Q = make_tensor(make_gmem_ptr(dcQ), make_layout(shape_Q, stride_q));
      Tensor K = make_tensor(make_gmem_ptr(dcK), make_layout(shape_K, stride_k));
      Tensor V = make_tensor(make_gmem_ptr(dcV), make_layout(shape_V, stride_v));
      Tensor K_cache = make_tensor(make_gmem_ptr(dcK_cache), make_layout(shape_K_cache, stride_k_cache));
      Tensor V_cache = make_tensor(make_gmem_ptr(dcV_cache), make_layout(shape_V_cache, stride_v_cache));
      Tensor O = make_tensor(make_gmem_ptr(ptrO), make_layout(shape_O, stride_o));

      // O accumulator types
      FragA tArA;
      FragARow tA_max;
      FragSPartialRow tA_sum;

      // Main loop
      CollectiveMainloop mainloop(params.mainloop, shared_storage.mainloop);
      CollectiveEpilogue epilogue{params.epilogue, shared_storage.epilogue};
      static constexpr int QK_BLK_M = decltype(get<0>(TileShapeQK{}))::value;
      constexpr bool kGqaFusion = TileScheduler::kGqaFusion;

      if constexpr (kGqaFusion) {
        const int head_kv = head;
        const int g = tile_scheduler.get_gqa_group_size();

        const int q_len = seq_len_qo;
        const int total_rows = g * q_len;
        const int num_q_blocks = (total_rows + int(QK_BLK_M) - 1) / int(QK_BLK_M);

        // GQA fusion is selected only by the decode dispatch (q_len == 1).
        // Preserve the paged-cache bounds computed above: replacing them with
        // seq_len_kv drops the entire cache because decode has no new K/V.
        const int gqa_fusion_q_per_head = q_len;

        const int idx_b_l = idx_b;
        const auto q_group_off = tile_scheduler.get_gqa_q_head_start(head_q) * stride<2>(Q.layout());
        const auto o_group_off = tile_scheduler.get_gqa_q_head_start(head_q) * stride<2>(O.layout());
        for (int qb = 0; qb < num_q_blocks; ++qb) {
          const int row_start = qb * int(QK_BLK_M);
          const int valid_rows = cute::min(int(QK_BLK_M), total_rows - row_start);

          auto make_gqa_view_q = [&, idx_b_l, row_start, valid_rows]() {
            auto offset = idx_b_l * stride<3>(Q.layout()) + q_group_off + row_start * stride<2>(Q.layout());
            return make_tensor(
                Q.data() + offset,
                make_layout(
                    make_shape(valid_rows, int(s.head_size_qk)),
                    make_stride(int(stride<2>(Q.layout())), stride<1>(Q.layout()))));
          };
          auto make_gqa_view_o = [&, idx_b_l, row_start, valid_rows]() {
            auto offset = idx_b_l * stride<3>(O.layout()) + o_group_off + row_start * stride<2>(O.layout());
            return make_tensor(
                O.data() + offset,
                make_layout(
                    make_shape(valid_rows, int(s.head_size_vo)),
                    make_stride(int(stride<2>(O.layout())), stride<1>(O.layout()))));
          };

          if (qb > 0) {
            sycl::group_barrier(get_work_group<3>());
          }

          FragA tArA;
          FragARow tA_max;
          FragSPartialRow tA_sum;

          mainloop.template operator()<true, TileScheduler::kCacheOnly>(
              make_gqa_view_q(),
              K(_, _, head_kv, idx_b),
              V(_, _, head_kv, idx_b),
              tArA,
              tA_max,
              tA_sum,
              blk_qv,
              blk_k0,
              blk_k1,
              k_blocks,
              thr_id,
              seq_len,
              seq_len_kv_cache,
              idx_b,
              full_tile_offset,
              discard_seq_coord,
              causal_k_block_start,
              row_start,
              gqa_fusion_q_per_head,
              slice_2d_wide_xe35(K_cache, head, l_coord),
              slice_2d_wide_xe35(V_cache, head, l_coord),
              p.scale_q,
              p.scale_k,
              p.scale_v);

          if constexpr (!is_empty_v<MainloopSharedStorage> && !is_empty_v<EpilogueSharedStorage>) {
            sycl::group_barrier(get_work_group<3>());
          }

          epilogue(
              make_gqa_view_o(),
              tArA,
              tA_max,
              tA_sum,
              blk_qv,
              thr_id,
              p.scale_v,
              p.sm_sink == nullptr ? ElementQ{} : p.sm_sink[head_q]);
        }
        return;
      }
      if constexpr (BlockScale) {
        auto scale_q = cute::ceil_div(s.head_size_qk, p.group_size);
        auto scale_k = cute::ceil_div(s.head_size_qk, p.group_size);
        int scale_v = cute::ceil_div(seq_len_kv, p.group_size);

        auto shape_scale_Q = make_shape(seq_len_qo, scale_q, s.num_heads_q, batch_dim);
        auto shape_scale_K = make_shape(seq_len_kv, scale_k, s.num_heads_kv, batch_dim);
        auto shape_scale_V = make_shape(s.head_size_vo, scale_v, s.num_heads_kv, batch_dim);
        int offset_scaleQ = 0;
        int offset_scaleK = 0;
        int offset_scaleV = 0;
        StrideScaleQ stride_scaleQ = p.dScaleQ;
        StrideScaleK stride_scaleK = p.dScaleK;
        StrideScaleV stride_scaleV = p.dScaleV;
        if constexpr (is_var_len) {
          auto qo_cumulative = s.seq_len_qo.cumulative_length;
          auto kv_cumulative = s.seq_len_kv.cumulative_length;
          auto kv_scale_cumulative = s.seq_len_kv.cumulative_scale_length;
          offset_scaleQ = s.num_heads_q * scale_q * qo_cumulative[idx_b];
          offset_scaleK = s.num_heads_kv * scale_k * kv_cumulative[idx_b];
          offset_scaleV = s.num_heads_kv * kv_scale_cumulative[idx_b];
          stride_scaleQ = cutlass::make_cute_packed_stride(StrideScaleQ{}, shape_scale_Q);
          stride_scaleK = cutlass::make_cute_packed_stride(StrideScaleK{}, shape_scale_K);
          stride_scaleV = cutlass::make_cute_packed_stride(StrideScaleV{}, shape_scale_V);
        }

        auto dcScaleQ = const_cast<ElementScale*>(p.scaleQ + offset_scaleQ);
        auto dcScaleK = const_cast<ElementScale*>(p.scaleK + offset_scaleK);
        auto dcScaleV = const_cast<ElementScale*>(p.scaleV + offset_scaleV);

        Tensor ScaleQ = make_tensor(make_gmem_ptr(dcScaleQ), make_layout(shape_scale_Q, stride_scaleQ));
        Tensor ScaleK = make_tensor(make_gmem_ptr(dcScaleK), make_layout(shape_scale_K, stride_scaleK));
        Tensor ScaleV = make_tensor(make_gmem_ptr(dcScaleV), make_layout(shape_scale_V, stride_scaleV));

        auto ScaleQ_head = ScaleQ(_, _, head_q, l_coord);
        auto ScaleK_head = ScaleK(_, _, head, l_coord);
        auto ScaleV_head = ScaleV(_, _, head, l_coord);

        mainloop.template operator()<false, TileScheduler::kCacheOnly>(
            Q(_, _, head_q, l_coord),
            K(_, _, head, l_coord),
            V(_, _, head, l_coord),
            tArA,
            tA_max,
            tA_sum,
            blk_qv,
            blk_k0,
            blk_k1,
            k_blocks,
            thr_id,
            seq_len,
            0,
            l_coord,
            full_tile_offset,
            discard_seq_coord,
            causal_k_block_start,
            0,
            0,
            slice_2d_wide_xe35(K_cache, head, l_coord),
            slice_2d_wide_xe35(V_cache, head, l_coord),
            p.scale_q,
            p.scale_k,
            p.scale_v,
            ScaleQ_head,
            ScaleK_head,
            ScaleV_head);
      } else {
        mainloop.template operator()<false, TileScheduler::kCacheOnly>(
            Q(_, _, head_q, l_coord),
            K(_, _, head, l_coord),
            V(_, _, head, l_coord),
            tArA,
            tA_max,
            tA_sum,
            blk_qv,
            blk_k0,
            blk_k1,
            k_blocks,
            thr_id,
            seq_len,
            seq_len_kv_cache,
            idx_b,
            full_tile_offset,
            discard_seq_coord,
            causal_k_block_start,
            0,
            0,
            slice_2d_wide_xe35(K_cache, head, l_coord),
            slice_2d_wide_xe35(V_cache, head, l_coord),
            p.scale_q,
            p.scale_k,
            p.scale_v);
      }
      if constexpr (!is_empty_v<MainloopSharedStorage> && !is_empty_v<EpilogueSharedStorage>) {
        sycl::group_barrier(get_work_group<3>());
      }

      // Epilogue
      epilogue(
          O(_, _, head_q, l_coord),
          tArA,
          tA_max,
          tA_sum,
          blk_qv,
          thr_id,
          p.scale_v,
          p.sm_sink == nullptr ? ElementQ{} : p.sm_sink[head_q]);
    }
  }
};

// Compute the maximum number of partitions (WGs) that can contribute to a single
// batch_head. This depends on sm_count and num_batch_heads at runtime.
// Free function (template-independent) so it can be unit-tested without
// instantiating the full kernel type.
inline int compute_max_num_partitions_xe35(int num_partition_wgs, int num_batch_heads) {
  return cute::ceil_div(num_partition_wgs, cute::max(1, num_batch_heads)) + 1;
}

template <class ProblemShape_, class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_>
class XeFMHAFwdDynamicSplitKernel_xe35 {
 public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  using TiledMMAQK = typename CollectiveMainloop::TiledMMAQK;
  using TiledMMAPV = typename CollectiveMainloop::TiledMMAPV;
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
  using SingleFragA = typename CollectiveMainloop::SingleFragA;
  using FragARow = typename CollectiveMainloop::FragARow;
  using FragSPartialRow = typename CollectiveMainloop::FragSPartialRow;
  // element dtype for MmaPV results
  using ElementA = typename CollectiveMainloop::ElementA;

  // Tile scheduler derived types
  static_assert(is_same_v<TileScheduler_, XeFHMAIndividualPersistentTileScheduler_xe35>);
  using TileScheduler = TileScheduler_;
  using TileSchedulerParams = typename TileScheduler::Params;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  using TileShapeO = typename CollectiveEpilogue::TileShapeO;
  using ElementO = typename CollectiveEpilogue::TensorO::element_type;
  using StrideO = decltype(stride(typename CollectiveEpilogue::TensorO{}));

  // Kernel level shared memory storage
  using MainloopSharedStorage = typename CollectiveMainloop::SharedStorage;
  using EpilogueSharedStorage = typename CollectiveEpilogue::SharedStorage;
  union SharedStorage {
    MainloopSharedStorage mainloop;
    EpilogueSharedStorage epilogue;
  };

  static constexpr bool BlockScale = false;

  static constexpr int SharedStorageSize = is_empty_v<SharedStorage> ? size_t(0) : sizeof(SharedStorage);

  // Important: make sure multiple of 16 element for each copy
  // this is for storing partial results from different KV partitions
  static constexpr int num_elem_per_thread = (size(FragA{}.shape()) + 2 * size(FragARow{}.shape()) + 15) / 16 * 16;

  // Device side arguments
  struct KernelArguments {
    ProblemShape shape;
    const ElementQ* Q;
    StrideQ dQ;
    const ElementK* K;
    StrideK dK;
    const ElementV* V;
    StrideV dV;
    ElementO* O;
    StrideO dO;
    const ElementK* K_cache = nullptr;
    StrideK dK_cache{};
    const ElementV* V_cache = nullptr;
    StrideV dV_cache{};
  };
  using KernelParams = KernelArguments;

  struct Arguments {
    KernelArguments kernel{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
  };

  // Kernel entry point API
  struct Params {
    KernelParams kernel;
    MainloopParams mainloop;
    EpilogueParams epilogue;
    TileSchedulerParams scheduler;
    // workspace for storing partial results of different KV partitions
    ElementA* partial_results_ptr = nullptr;
    // for atomic add
    int32_t* atomic_reduce_cnt_ptr = nullptr;
    // max partitions per batch_head (computed from sm_count/2)
    int max_num_partitions = 0;
  };

  //
  // Methods
  //

  static size_t partial_results_byte_offset(int num_batch_heads) {
    return ((num_batch_heads * sizeof(int32_t) + size_t(15)) / 16) * 16;
  }

  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    int num_batch_heads = args.kernel.shape.batch * args.kernel.shape.num_heads_kv;
    int max_parts = compute_max_num_partitions_xe35(args.hw_info.sm_count / 2, num_batch_heads);
    int32_t* atomic_reduce_cnt_ptr = reinterpret_cast<int32_t*>(workspace);
    ElementA* partial_results_ptr =
        reinterpret_cast<ElementA*>(reinterpret_cast<char*>(workspace) + partial_results_byte_offset(num_batch_heads));
    return {
        args.kernel,
        CollectiveMainloop::to_underlying_arguments(args.mainloop, workspace),
        CollectiveEpilogue::to_underlying_arguments(args.epilogue, workspace),
        TileScheduler::to_underlying_arguments(args.kernel.shape, args.hw_info, TileShapeO{}),
        partial_results_ptr,
        atomic_reduce_cnt_ptr,
        max_parts};
  }

  static bool can_implement(Arguments const& args) {
    return CollectiveMainloop::can_implement(args.mainloop) && CollectiveEpilogue::can_implement(args.epilogue);
  }

  static int get_workspace_size(Arguments const& args) {
    int ws_size = 0;
    int num_batch_heads = args.kernel.shape.batch * args.kernel.shape.num_heads_kv;
    int max_parts = compute_max_num_partitions_xe35(args.hw_info.sm_count / 2, num_batch_heads);
    const int wg_size = SGPerWG::value * intel::sg_size;

    // partial attn outputs, exp sum and max logits
    ws_size += (max_parts * num_batch_heads) * wg_size * num_elem_per_thread * sizeof(ElementA);
    // atomic counter
    ws_size += partial_results_byte_offset(num_batch_heads);
    return ws_size;
  }

  static cutlass::Status initialize_workspace(
      Arguments const& args,
      void* workspace = nullptr,
      cudaStream_t stream = nullptr,
      CudaHostAdapter* cuda_adapter = nullptr) {
    int num_batch_heads = args.kernel.shape.batch * args.kernel.shape.num_heads_kv;
    compat::fill(reinterpret_cast<int32_t*>(workspace), (int32_t)0, num_batch_heads);
    size_t cnt_bytes = partial_results_byte_offset(num_batch_heads);
    auto partial_ws_count = (get_workspace_size(args) - cnt_bytes) / sizeof(ElementA);
    auto* partial_results_ptr = reinterpret_cast<ElementA*>(reinterpret_cast<char*>(workspace) + cnt_bytes);
    compat::fill(partial_results_ptr, (ElementA)0, partial_ws_count);
    return Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::template get_grid_shape<SGPerWG::value>(params.scheduler);
  }

  static dim3 get_block_shape() {
    return dim3(SGPerWG::value * intel::sg_size, 1, 1);
  }

  CUTLASS_DEVICE
  int get_partition_id(
      const int cur_wg_id, const int batch_head_id, const int num_blocks_per_wg, const int local_k_blocks) {
    int partition_id = 0;
    if (batch_head_id == 0) {
      return cur_wg_id;
    }
    int start_wg_id = batch_head_id * local_k_blocks / num_blocks_per_wg;
    partition_id = cur_wg_id - start_wg_id;
    return partition_id;
  }

  CUTLASS_DEVICE
  int get_num_partitions(const int batch_head_id, const int num_blocks_per_wg, const int local_k_blocks) {
    int num_partitions = 1;
    int start_wg_id = batch_head_id * local_k_blocks / num_blocks_per_wg;
    int end_wg_id = (batch_head_id + 1) * local_k_blocks / num_blocks_per_wg;
    num_partitions = end_wg_id - start_wg_id + 1;
    // end_wg_id is the starting wg id of next batch head id
    if (((batch_head_id + 1) * local_k_blocks) % num_blocks_per_wg == 0) {
      num_partitions -= 1;
    }
    return num_partitions;
  }

  template <class Params, class FragA, class FragARow>
  CUTLASS_DEVICE void reduce_split2(
      const Params& params,
      FragA& out1,
      FragARow& max_val1,
      FragARow& exp_sum_val1,
      FragA& out2,
      FragARow& max_val2,
      FragARow& exp_sum_val2) {
    // global max value
    FragARow max_prev1 = max_val1;
    FragARow max_prev2 = max_val2;

    auto scale = params.mainloop.scale;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < max_val1.size(); i++) {
      max_val1(i) = sycl::max(max_val1(i), max_val2(i));
    }

    FragARow rescale1, rescale2;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < max_val1.size(); i++) {
      rescale1(i) = sycl::native::exp2(max_prev1(i) - max_val1(i));
      rescale2(i) = sycl::native::exp2(max_prev2(i) - max_val1(i));
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < exp_sum_val1.size(); i++) {
      exp_sum_val1(i) = exp_sum_val1(i) * rescale1(i) + exp_sum_val2(i) * rescale2(i);
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < out1.size(); i++)
      out1(i) = out1(i) * broadcast<0>(rescale1, out1, i) + out2(i) * broadcast<0>(rescale2, out2, i);
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    using namespace sycl::ext::oneapi::this_work_item;

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    auto& p = params.kernel;
    ProblemShape const& s = p.shape;
    int head_group_q = s.num_heads_q / s.num_heads_kv;

    int thr_id = int(ThreadIdxX());
    int wg_id = int(BlockIdxZ());

    int sg_id = thr_id / intel::sg_size;
    int tid_in_sg = thr_id % intel::sg_size;
    int num_batch_heads = s.batch * s.num_heads_kv;

    int local_k_blocks = cute::ceil_div(s.seq_len_kv, get<1>(TileShapeQK{}));
    // total number of blocks need to be processed across all wgs
    int total_k_blocks = local_k_blocks * num_batch_heads;
    // to guarantee all wg process similar number of blocks of KV
    int num_blocks_per_wg = cute::ceil_div(total_k_blocks, GridDimZ());

    TileScheduler tile_scheduler{params.scheduler, get<1>(TileShapeQK{}), local_k_blocks, num_batch_heads};

    CUTLASS_PRAGMA_NO_UNROLL
    for (; tile_scheduler.is_valid(); ++tile_scheduler) {
      auto [blk_q, blk_v, start_batch_head_id] = tile_scheduler.get_block_coord();  // (Q,V, batch_head_idx)
      auto blk_qv = make_coord(blk_q, blk_v);

      auto shape_Q = make_shape(s.seq_len_qo, s.head_size_qk, s.num_heads_q, s.batch);
      auto shape_K = make_shape(s.seq_len_kv, s.head_size_qk, s.num_heads_kv, s.batch);
      auto shape_V = make_shape(s.head_size_vo, s.seq_len_kv, s.num_heads_kv, s.batch);
      auto shape_O = make_shape(s.seq_len_qo, s.head_size_vo, s.num_heads_q, s.batch);

      auto dcQ = const_cast<ElementQ*>(p.Q);  // de-const these for uniformity
      auto dcK = const_cast<ElementK*>(p.K);
      auto dcV = const_cast<ElementV*>(p.V);

      Tensor Q = make_tensor(make_gmem_ptr(dcQ), make_layout(shape_Q, p.dQ));  // (q,d,h,b)
      Tensor K = make_tensor(make_gmem_ptr(dcK), make_layout(shape_K, p.dK));  // (k,d,h,b)
      Tensor V = make_tensor(make_gmem_ptr(dcV), make_layout(shape_V, p.dV));  // (v,k,h,b)
      Tensor O = make_tensor(make_gmem_ptr(p.O), make_layout(shape_O, p.dO));  // (q,v,h,b)

      auto shape_K_cache = make_shape(s.seq_len_kv_cache, s.head_size_qk, s.num_heads_kv, s.batch);
      auto shape_V_cache = make_shape(s.head_size_vo, s.seq_len_kv_cache, s.num_heads_kv, s.batch);
      auto dcK_cache = const_cast<ElementK*>(p.K_cache);
      auto dcV_cache = const_cast<ElementV*>(p.V_cache);
      Tensor K_cache = make_tensor(make_gmem_ptr(dcK_cache), make_layout(shape_K_cache, p.dK_cache));
      Tensor V_cache = make_tensor(make_gmem_ptr(dcV_cache), make_layout(shape_V_cache, p.dV_cache));

      // O accumulator types
      FragA tArA;
      FragARow tA_max, tA_sum;
      FragSPartialRow tA_sum_partial;

      // compute num computed blocks for start batch head id
      int num_computed_blocks = (start_batch_head_id == 0)
                                    ? (wg_id * num_blocks_per_wg)
                                    : (wg_id * num_blocks_per_wg - start_batch_head_id * local_k_blocks);
      int start_blk, end_blk, idx_b, head_kv;
      // leader wg is also responsible for reducing partial results, while other
      // worker wg only to compute partial results
      bool is_leader_wg = wg_id < num_batch_heads;

      if (thr_id == 0 && is_leader_wg) {
        // reset atomic counter before computation
        *(params.atomic_reduce_cnt_ptr + wg_id) = 0;
      }

      // Main loop
      CollectiveMainloop mainloop(params.mainloop, shared_storage.mainloop);

      // compute blocks budget remained for each wg
      int block_budget_remained = num_blocks_per_wg;
      int batch_head_id = start_batch_head_id;
      bool is_update_batch_head_id = false;
      // Skip excess WGs whose start_batch_head_id is already out of range.
      // This happens when total_k_blocks < GridDimZ (more WGs than work).
      if (batch_head_id >= num_batch_heads) {
        block_budget_remained = 0;
      }
      while (block_budget_remained > 0) {
        int num_new_blocks = local_k_blocks - num_computed_blocks;
        if (num_new_blocks <= block_budget_remained) {
          // finished current batch head id
          start_blk = num_computed_blocks;
          end_blk = start_blk + num_new_blocks;

          // update states
          num_computed_blocks = 0;
          block_budget_remained -= num_new_blocks;
          is_update_batch_head_id = true;
        } else {
          // budget cannot afford finishing current batch head id
          start_blk = num_computed_blocks;
          end_blk = start_blk + block_budget_remained;

          block_budget_remained = 0;
          is_update_batch_head_id = false;
        }

        head_kv = batch_head_id % s.num_heads_kv;
        idx_b = batch_head_id / s.num_heads_kv;
        const int total_rows_q = head_group_q * s.seq_len_qo;
        const int64_t q_group_off = int64_t(head_kv) * head_group_q * stride<2>(Q.layout());
        auto make_gqa_view_q = [&]() {
          int64_t offset = int64_t(idx_b) * stride<3>(Q.layout()) + q_group_off;
          return make_tensor(
              Q.data() + offset,
              make_layout(
                  make_shape(total_rows_q, int(s.head_size_qk)),
                  make_stride(int(stride<0>(Q.layout())), stride<1>(Q.layout()))));
        };

        int split_full_tile_offset = 0;
        int split_q_per_head = 0;
        if constexpr (CollectiveMainloop::CausalMask) {
          split_q_per_head = s.seq_len_qo;
          split_full_tile_offset = s.seq_len_kv - cute::min(int(s.seq_len_qo), int(s.seq_len_kv));
        }

        mainloop.template operator()<true>(
            make_gqa_view_q(),
            K(_, _, head_kv, idx_b),
            V(_, _, head_kv, idx_b),
            tArA,
            tA_max,
            tA_sum_partial,
            blk_qv,
            start_blk,
            end_blk,
            local_k_blocks,
            thr_id,
            s.seq_len_kv,
            0,
            idx_b,
            split_full_tile_offset,
            0,
            0,
            0,
            split_q_per_head);

        if constexpr (CollectiveMainloop::DeferSumHReduce) {
          tA_sum = reduce<0, cute::ReduceMode::Horizontal>(tA_sum_partial, sycl::plus<void>{});
        } else {
          tA_sum = tA_sum_partial;
        }

        // partition id of start batch head id in current wg
        int partition_id = get_partition_id(wg_id, batch_head_id, num_blocks_per_wg, local_k_blocks);

        // store partial result: tArA, tA_max and tA_sum
        int offset = batch_head_id * params.max_num_partitions * num_elem_per_thread * SGPerWG::value * intel::sg_size +
                     partition_id * num_elem_per_thread * SGPerWG::value * intel::sg_size +
                     sg_id * intel::sg_size * num_elem_per_thread + tid_in_sg * num_elem_per_thread;
        Tensor tPartial = make_tensor(params.partial_results_ptr + offset, make_shape(Int<num_elem_per_thread>{}));
        Tensor merged_res = make_tensor<ElementA>(Int<num_elem_per_thread>{});

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(FragA{}.shape()); ++i) {
          merged_res(i) = tArA(i);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(FragARow{}.shape()); ++i) {
          merged_res(2 * i + size(FragA{}.shape())) = tA_max(i);
          merged_res(2 * i + 1 + size(FragA{}.shape())) = tA_sum(i);
        }
        copy(merged_res, tPartial);

        // Split barrier: arrive (flush stores) then wait (ensure all WG threads arrived)
        cute::barrier_arrive(ScopeWorkgroup, SemanticsRelease | SemanticsCrossWGMemory);
        cute::barrier_wait(ScopeWorkgroup, SemanticsAcquire | SemanticsCrossWGMemory);

        // after store, set atomic cnt
        if (thr_id == 0) {
          atomicAdd(params.atomic_reduce_cnt_ptr + batch_head_id, 1);
        }

        // advance to next batch head id
        if (is_update_batch_head_id) {
          batch_head_id += 1;
          if (batch_head_id >= num_batch_heads) {
            break;
          }
        }
      }

      if (is_leader_wg) {
        int num_partitions = get_num_partitions(wg_id, num_blocks_per_wg, local_k_blocks);

        // check atomic to wait for partial results ready
        while (atomicLoad(params.atomic_reduce_cnt_ptr + wg_id) != num_partitions) {
        }

        clear(tArA);
        clear(tA_max);
        clear(tA_sum);

        for (int i = 0; i < num_partitions; ++i) {
          int offset = wg_id * params.max_num_partitions * SGPerWG::value * intel::sg_size * num_elem_per_thread +
                       i * SGPerWG::value * intel::sg_size * num_elem_per_thread +
                       sg_id * intel::sg_size * num_elem_per_thread + tid_in_sg * num_elem_per_thread;
          Tensor tPartial = make_tensor(params.partial_results_ptr + offset, make_shape(Int<num_elem_per_thread>{}));
          Tensor merged_res = make_tensor<ElementA>(Int<num_elem_per_thread>{});
          copy(tPartial, merged_res);

          if (i == 0) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(FragA{}.shape()); ++i) {
              tArA(i) = merged_res(i);
            }

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(FragARow{}.shape()); ++i) {
              tA_max(i) = merged_res(2 * i + size(FragA{}.shape()));
              tA_sum(i) = merged_res(2 * i + 1 + size(FragA{}.shape()));
            }

            continue;
          }

          FragA tArA_2;
          FragARow tA_max_2, tA_sum_2;
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(FragA{}.shape()); ++i) {
            tArA_2(i) = merged_res(i);
          }

          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(FragARow{}.shape()); ++i) {
            tA_max_2(i) = merged_res(2 * i + size(FragA{}.shape()));
            tA_sum_2(i) = merged_res(2 * i + 1 + size(FragA{}.shape()));
          }

          reduce_split2(params, tArA, tA_max, tA_sum, tArA_2, tA_max_2, tA_sum_2);
        }

        // require group barrier if using SLM
        if constexpr (!is_empty_v<MainloopSharedStorage> && !is_empty_v<EpilogueSharedStorage>) {
          sycl::group_barrier(get_work_group<3>());
        }

        head_kv = wg_id % s.num_heads_kv;
        idx_b = wg_id / s.num_heads_kv;

        const int total_rows_o = head_group_q * s.seq_len_qo;
        const int64_t o_group_off = int64_t(head_kv) * head_group_q * stride<2>(O.layout());
        auto make_gqa_view_o = [&]() {
          int64_t offset = int64_t(idx_b) * stride<3>(O.layout()) + o_group_off;
          return make_tensor(
              O.data() + offset,
              make_layout(
                  make_shape(total_rows_o, int(s.head_size_vo)),
                  make_stride(int(stride<0>(O.layout())), stride<1>(O.layout()))));
        };

        // Epilogue
        CollectiveEpilogue epilogue{params.epilogue, shared_storage.epilogue};
        epilogue.template operator()<true>(make_gqa_view_o(), tArA, tA_max, tA_sum, blk_qv, thr_id);
      }
    }
  }
};

}  // namespace cutlass::fmha_xe35::kernel
