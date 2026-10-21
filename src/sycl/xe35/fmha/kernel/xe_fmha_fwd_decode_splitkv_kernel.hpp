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

/*
  Paged-KV decode kernel with full GQA fusion and split-KV.

  Decode (one query token per request) is bandwidth bound: the work is reading
  the cached K/V once.  The generic kernel dispatches one workgroup per query
  head (or per pair of heads), so the same K/V tile is fetched by every
  workgroup of the GQA group, and at small batch sizes only a few hundred
  subgroups are in flight.  This kernel instead assigns one workgroup per
  (batch, kv head, kv split):

    * all query heads of the GQA group form the M dimension of the DPAS tile,
      so each cached K/V element is read exactly once per launch;
    * the cached sequence is cut into `num_kv_splits` contiguous tile ranges
      so the grid has enough workgroups to saturate memory;
    * each split writes an unnormalized partial (O, row max, row sum); the
      last split to finish (device-scope atomic counter) merges the partials
      and writes the normalized output.  The counter resets itself, so the
      workspace needs zeroing only when it is allocated.
*/

#pragma once

#include <sycl/sycl.hpp>

#include "../collective/fmha_fusion.hpp"
#include "../collective/xe_fmha_fwd_epilogue.hpp"
#include "../collective/xe_fmha_fwd_mainloop.hpp"
#include "cute/util/type_traits.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "xe_fmha_fwd_kernel.hpp"

namespace cutlass::fmha_xe35::kernel {

using namespace cute;

// One workgroup per (kv split, kv head, batch).
struct XeFMHADecodeSplitKvTileScheduler_xe35 {
  static constexpr bool kGqaFusion = true;
  static constexpr bool kCacheOnly = true;

  struct Params {
    dim3 grid;
    FastDivmod divmod_num_heads_kv;
    int num_splits;
  };

  bool valid_ = true;
  Params params;

  CUTLASS_DEVICE
  XeFMHADecodeSplitKvTileScheduler_xe35(Params const& params) : params(params) {}

  template <class ProblemShape>
  static Params to_underlying_arguments(ProblemShape const& shape, int num_splits) {
    Params p{};
    p.grid = dim3(num_splits, 1, shape.batch * shape.num_heads_kv);
    p.divmod_num_heads_kv = FastDivmod(shape.num_heads_kv);
    p.num_splits = num_splits;
    return p;
  }

  static dim3 get_grid_shape(Params const& params) {
    return params.grid;
  }

  CUTLASS_DEVICE
  bool is_valid() {
    return valid_;
  }

  // Returns (kv split, kv head, batch).
  CUTLASS_DEVICE
  auto get_block_coord() {
    int idx_b = int(BlockIdxZ());
    int head_kv = 0;
    params.divmod_num_heads_kv(idx_b, head_kv, idx_b);
    return make_coord(int(BlockIdxX()), head_kv, idx_b);
  }

  CUTLASS_DEVICE
  XeFMHADecodeSplitKvTileScheduler_xe35& operator++() {
    valid_ = false;
    return *this;
  }
};

// `LaunchTag` only distinguishes otherwise identical instantiations that are
// launched with different kernel properties (e.g. GRF mode), so each gets its
// own SYCL kernel name.
template <
    class ProblemShape_,
    class CollectiveMainloop_,
    class CollectiveEpilogue_,
    class TileScheduler_,
    int LaunchTag = 0>
class XeFMHAFwdDecodeSplitKvKernel_xe35 {
 public:
  using ProblemShape = ProblemShape_;
  static constexpr int kLaunchTag = LaunchTag;
  static_assert(
      cutlass::fmha_xe35::collective::is_variable_length_v<typename ProblemShape::SeqLenType>,
      "The decode split-KV kernel reads per-request cache lengths from cu_seqlens");

  using CollectiveMainloop = CollectiveMainloop_;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;
  using TiledMMAQK = typename CollectiveMainloop::TiledMMAQK;
  using TileShapeQK = typename CollectiveMainloop::TileShapeQK;
  using ElementQ = typename CollectiveMainloop::TensorQ::element_type;
  using ElementK = typename CollectiveMainloop::TensorK::element_type;
  using ElementV = typename CollectiveMainloop::TensorV::element_type;
  using StrideQ = decltype(stride(typename CollectiveMainloop::TensorQ{}));
  using StrideK = decltype(stride(typename CollectiveMainloop::TensorK{}));
  using StrideV = decltype(stride(typename CollectiveMainloop::TensorV{}));
  using SGPerWG = typename CollectiveMainloop::SGPerWG;
  using FragA = typename CollectiveMainloop::FragA;
  using FragARow = typename CollectiveMainloop::FragARow;
  using FragSPartialRow = typename CollectiveMainloop::FragSPartialRow;
  using ElementA = typename CollectiveMainloop::ElementA;

  using TileScheduler = TileScheduler_;
  using TileSchedulerParams = typename TileScheduler::Params;

  using CollectiveEpilogue = CollectiveEpilogue_;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;
  using ElementO = typename CollectiveEpilogue::TensorO::element_type;
  using StrideO = decltype(stride(typename CollectiveEpilogue::TensorO{}));

  static_assert(CollectiveMainloop::CachedKV && CollectiveMainloop::PagedKV, "decode split-KV requires a paged cache");
  static_assert(!CollectiveMainloop::CausalMask, "a single query token sees the whole cache; use the plain mask");
  static_assert(!CollectiveMainloop::BlockScale, "block-scaled inputs are not supported");
  static_assert(SGPerWG::value == 1, "one subgroup per workgroup: each split is an independent subgroup");

  using MainloopSharedStorage = typename CollectiveMainloop::SharedStorage;
  using EpilogueSharedStorage = typename CollectiveEpilogue::SharedStorage;
  union SharedStorage {
    MainloopSharedStorage mainloop;
    EpilogueSharedStorage epilogue;
  };
  static constexpr int SharedStorageSize = is_empty_v<SharedStorage> ? size_t(0) : sizeof(SharedStorage);

  static constexpr int kMaxGqaGroup = get<0>(TileShapeQK{});
  static constexpr int kBlkK = get<1>(TileShapeQK{});
  static constexpr int kFragASize = decltype(size(FragA{}.shape()))::value;
  static constexpr int kFragARowSize = decltype(size(FragARow{}.shape()))::value;
  // Per-lane partial: O accumulator, then interleaved (row max, row sum).
  static constexpr int kPartialElems = kFragASize + 2 * kFragARowSize;

  struct KernelArguments {
    ProblemShape shape;
    const ElementQ* Q;
    StrideQ dQ;
    const ElementK* K_cache;
    StrideK dK_cache;
    const ElementV* V_cache;
    StrideV dV_cache;
    ElementO* O;
    StrideO dO;
  };
  using KernelParams = KernelArguments;

  struct Arguments {
    KernelArguments kernel{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    int num_kv_splits = 1;
    // Only read when num_kv_splits > 1.  `partial_ws` holds
    // batch * num_heads_kv * num_kv_splits partial blocks of
    // partial_bytes_per_split() bytes; `split_counters` holds one int per
    // (batch, kv head) and must be zero when the kernel starts.
    ElementA* partial_ws = nullptr;
    int* split_counters = nullptr;
  };

  struct Params {
    KernelParams kernel;
    MainloopParams mainloop;
    EpilogueParams epilogue;
    TileSchedulerParams scheduler;
    int num_kv_splits;
    ElementA* partial_ws;
    int* split_counters;
  };

  static constexpr size_t partial_bytes_per_split() {
    return size_t(kPartialElems) * intel::sg_size * sizeof(ElementA);
  }

  static Params to_underlying_arguments(Arguments const& args, void* workspace = nullptr) {
    return {
        args.kernel,
        CollectiveMainloop::to_underlying_arguments(args.mainloop, workspace),
        CollectiveEpilogue::to_underlying_arguments(args.epilogue, workspace),
        TileScheduler::to_underlying_arguments(args.kernel.shape, args.num_kv_splits),
        args.num_kv_splits,
        args.partial_ws,
        args.split_counters};
  }

  static bool can_implement(Arguments const& args) {
    auto const& s = args.kernel.shape;
    if (s.num_heads_kv <= 0 || s.num_heads_q % s.num_heads_kv != 0) return false;
    if (s.num_heads_q / s.num_heads_kv > kMaxGqaGroup) return false;
    if (args.num_kv_splits < 1) return false;
    if (args.num_kv_splits > 1 && (args.partial_ws == nullptr || args.split_counters == nullptr)) return false;
    return CollectiveMainloop::can_implement(args.mainloop) && CollectiveEpilogue::can_implement(args.epilogue);
  }

  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::get_grid_shape(params.scheduler);
  }

  static dim3 get_block_shape() {
    return dim3(SGPerWG::value * intel::sg_size, 1, 1);
  }

  // Fold one partial (o2, m2, l2) into the running (o, m, l).
  CUTLASS_DEVICE
  static void
  merge_partial(FragA& o, FragARow& m, FragARow& l, FragA const& o2, FragARow const& m2, FragARow const& l2) {
    FragARow scale1, scale2;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kFragARowSize; i++) {
      ElementA m_new = sycl::max(m(i), m2(i));
      scale1(i) = sycl::native::exp2(m(i) - m_new);
      scale2(i) = sycl::native::exp2(m2(i) - m_new);
      m(i) = m_new;
      l(i) = l(i) * scale1(i) + l2(i) * scale2(i);
    }
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kFragASize; i++) {
      o(i) = o(i) * broadcast<0>(scale1, o, i) + o2(i) * broadcast<0>(scale2, o, i);
    }
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    using namespace sycl::ext::oneapi::this_work_item;

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    auto& p = params.kernel;
    ProblemShape const& s = p.shape;
    const int gqa_group = s.num_heads_q / s.num_heads_kv;
    const int thr_id = int(ThreadIdxX());
    const int num_splits = params.num_kv_splits;

    TileScheduler tile_scheduler{params.scheduler};
    auto [split, head_kv, idx_b] = tile_scheduler.get_block_coord();

    // Decode geometry: one query token per request, no new K/V, the query
    // attends to cache positions [0, seq_len_kv_cache) from position
    // seq_len_kv_cache - 1.
    const int q_row = s.seq_len_qo.cumulative_length[idx_b];
    const int seq_len_kv_cache = s.seq_len_kv_cache.cumulative_length[idx_b];
    const int k_blocks = cute::ceil_div(seq_len_kv_cache, kBlkK);
    const int q_pos = seq_len_kv_cache - 1;

    int blk_k0 = 0;
    if constexpr (CollectiveMainloop::LocalMask) {
      const int win_left = CollectiveMainloop::window_size_left(params.mainloop);
      if (win_left >= 0) {
        blk_k0 = cute::max(0, q_pos - win_left) / kBlkK;
      }
    }
    const int blk_k1 = k_blocks;

    // Contiguous tile range of this split.  Splits past the end are empty and
    // contribute (O = 0, max = lowest, sum = 0).
    const int total_blocks = cute::max(0, blk_k1 - blk_k0);
    const int blocks_per_split = cute::ceil_div(total_blocks, num_splits);
    const int split_k0 = cute::min(blk_k1, blk_k0 + split * blocks_per_split);
    const int split_k1 = cute::min(blk_k1, split_k0 + blocks_per_split);

    // Global views.  The paged cache spans the whole pool; the page table
    // (indexed by idx_b inside the mainloop) selects the physical tiles.
    const int pool_len = params.mainloop.total_seqlen_kv;
    auto dcK_cache = const_cast<ElementK*>(p.K_cache);
    auto dcV_cache = const_cast<ElementV*>(p.V_cache);
    Tensor K_cache = make_tensor(
        make_gmem_ptr(dcK_cache), make_layout(make_shape(pool_len, s.head_size_qk, s.num_heads_kv, 1), p.dK_cache));
    Tensor V_cache = make_tensor(
        make_gmem_ptr(dcV_cache), make_layout(make_shape(s.head_size_vo, pool_len, s.num_heads_kv, 1), p.dV_cache));
    // The mainloop interface still takes new-K/V views; alias the pool with a
    // zero sequence length (CacheOnly never touches them).
    Tensor K_new = make_tensor(
        make_gmem_ptr(dcK_cache), make_layout(make_shape(0, s.head_size_qk, s.num_heads_kv, 1), p.dK_cache));
    Tensor V_new = make_tensor(
        make_gmem_ptr(dcV_cache), make_layout(make_shape(s.head_size_vo, 0, s.num_heads_kv, 1), p.dV_cache));

    // All query heads of the GQA group are consecutive: view them as the rows
    // of one (gqa_group, head_dim) tile.
    const int q_head0 = head_kv * gqa_group;
    auto dcQ = const_cast<ElementQ*>(p.Q) + q_row * get<0>(p.dQ) + q_head0 * get<2>(p.dQ);
    Tensor Q_2d = make_tensor(
        make_gmem_ptr(dcQ),
        make_layout(make_shape(gqa_group, int(s.head_size_qk)), make_stride(int(get<2>(p.dQ)), get<1>(p.dQ))));
    auto ptrO = p.O + q_row * get<0>(p.dO) + q_head0 * get<2>(p.dO);
    Tensor O_2d = make_tensor(
        make_gmem_ptr(ptrO),
        make_layout(make_shape(gqa_group, int(s.head_size_vo)), make_stride(int(get<2>(p.dO)), get<1>(p.dO))));

    FragA tArA;
    FragARow tA_max;
    FragSPartialRow tA_sum;

    CollectiveMainloop mainloop(params.mainloop, shared_storage.mainloop);
    CollectiveEpilogue epilogue{params.epilogue, shared_storage.epilogue};
    auto blk_qv = make_coord(0, 0);

    mainloop.template operator()<true, true>(
        Q_2d,
        K_new(_, _, head_kv, 0),
        V_new(_, _, head_kv, 0),
        tArA,
        tA_max,
        tA_sum,
        blk_qv,
        split_k0,
        split_k1,
        k_blocks,
        thr_id,
        seq_len_kv_cache,
        seq_len_kv_cache,
        idx_b,
        q_pos,
        0,
        0,
        0,
        1,
        K_cache(_, _, head_kv, 0),
        V_cache(_, _, head_kv, 0));

    FragARow tA_sum_row;
    if constexpr (CollectiveMainloop::DeferSumHReduce) {
      tA_sum_row = reduce<0, ReduceMode::Horizontal>(tA_sum, sycl::plus<void>{});
    } else {
      tA_sum_row = tA_sum;
    }

    if constexpr (!is_empty_v<MainloopSharedStorage> && !is_empty_v<EpilogueSharedStorage>) {
      sycl::group_barrier(get_work_group<3>());
    }

    if (num_splits == 1) {
      epilogue.template operator()<true>(O_2d, tArA, tA_max, tA_sum_row, blk_qv, thr_id);
      return;
    }

    // Publish this split's partial.  Lane-interleaved layout: element i of
    // lane t lives at [i * sg_size + t], so every store covers 64 contiguous
    // bytes across the subgroup.
    const int slot = idx_b * s.num_heads_kv + head_kv;
    const int lane = thr_id % intel::sg_size;
    ElementA* ws_base = params.partial_ws + size_t(slot) * num_splits * kPartialElems * intel::sg_size;
    {
      ElementA* ws = ws_base + size_t(split) * kPartialElems * intel::sg_size;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kFragASize; i++) {
        ws[i * intel::sg_size + lane] = tArA(i);
      }
      CUTLASS_PRAGMA_UNROLL
      for (int r = 0; r < kFragARowSize; r++) {
        ws[(kFragASize + 2 * r) * intel::sg_size + lane] = tA_max(r);
        ws[(kFragASize + 2 * r + 1) * intel::sg_size + lane] = tA_sum_row(r);
      }
    }

    auto sg = get_sub_group();
    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
    sycl::group_barrier(sg);
    int prev = 0;
    if (lane == 0) {
      sycl::atomic_ref<
          int,
          sycl::memory_order::acq_rel,
          sycl::memory_scope::device,
          sycl::access::address_space::global_space>
          counter(params.split_counters[slot]);
      prev = counter.fetch_add(1);
    }
    prev = sycl::group_broadcast(sg, prev, 0);
    if (prev != num_splits - 1) return;

    // Last split for this (batch, kv head): every other partial is complete.
    // Reset the counter for the next launch, then merge.
    if (lane == 0) {
      sycl::atomic_ref<
          int,
          sycl::memory_order::relaxed,
          sycl::memory_scope::device,
          sycl::access::address_space::global_space>
          counter(params.split_counters[slot]);
      counter.store(0);
    }
    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::device);

    // Merge in split order from an empty accumulator so the result does not
    // depend on which split happened to finish last.
    FragA acc;
    FragARow acc_max, acc_sum;
    clear(acc);
    fill(acc_max, cutlass::platform::numeric_limits<ElementA>::lowest());
    clear(acc_sum);
    for (int other = 0; other < num_splits; other++) {
      if (other == split) {
        merge_partial(acc, acc_max, acc_sum, tArA, tA_max, tA_sum_row);
        continue;
      }
      ElementA const* ws = ws_base + size_t(other) * kPartialElems * intel::sg_size;
      FragA o2;
      FragARow m2, l2;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kFragASize; i++) {
        o2(i) = ws[i * intel::sg_size + lane];
      }
      CUTLASS_PRAGMA_UNROLL
      for (int r = 0; r < kFragARowSize; r++) {
        m2(r) = ws[(kFragASize + 2 * r) * intel::sg_size + lane];
        l2(r) = ws[(kFragASize + 2 * r + 1) * intel::sg_size + lane];
      }
      merge_partial(acc, acc_max, acc_sum, o2, m2, l2);
    }

    epilogue.template operator()<true>(O_2d, acc, acc_max, acc_sum, blk_qv, thr_id);
  }
};

}  // namespace cutlass::fmha_xe35::kernel
