/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
/*!
  \file
  \brief Two-stage sparse MLA Stage 2 kernel for DeepSeek V4 (decode + prefill).

  XeMlaSparse2StageDenseKernel: dense flash-decode over the Stage 1 gathered tile via
  XMX DPAS QK/PV GEMMs, B_H head packing, V-split, online (log2) softmax, and
  attn_sink merge.

  Shared verbatim by BOTH two-stage paths, which is why neither this kernel nor its
  collectives / scheduler / work-tile carry "Decode" in their names: prefill maps each
  query row to a decode "batch" and reuses all of them unchanged
  (MlaSparsePrefill2StageXe is an alias of MlaSparseDecode2StageXe), so everything
  path-specific lives in Stage 1. The algorithm is still decode-*shaped* -- one query
  position per work-tile -- and that is precisely what lets prefill reuse it.

  Decomposed into the sycl-tla convention used by the fused MLA kernels: this
  file is the thin kernel wrapper that wires together
    - collective mainloop  (collective/xe_mla_sparse_2stage_mainloop.hpp)
        QK/PV GEMM + online softmax producing the O accumulator + row stats,
    - collective epilogue   (collective/xe_mla_sparse_2stage_epilogue.hpp)
        cross-subgroup reduce, normalize, attn_sink merge, LSE, and store,
    - tile scheduler        (kernel/xe_mla_sparse_2stage_tile_scheduler.hpp)
        (batch, seq, head-block, v-split) work-tile decode.

  Like XeMlaSparseFwdKernel (the fused path), the collectives + scheduler are
  template parameters and the kernel exposes the device::MLASparse / launch<>
  contract (Arguments/Params, to_underlying_arguments, can_implement,
  get_workspace_size, initialize_workspace, get_grid_shape, get_block_shape,
  SharedStorageSize). The config struct MlaSparseDecode2StageXe assembles the
  concrete instantiation (see device/mla_sparse_decode_2stage_types.hpp).

  The kernel's Arguments == Params is SparseAttn2StageParams, its own per-layer params
  object -- Stage 1 is a separate kernel with separate params, so there is no gather
  template parameter and no shared/derived params type here. The runner
  (device::MLASparse) pairs the two, the same way device::MLA pairs the split-KV
  attention kernel with its reduction companion. The two stages communicate only through
  the gathered-KV HBM buffers (params.kernel.gathered_k /
  params.mainloop.gathered_valid_mask), which Stage 1 fills before this kernel is
  launched. to_underlying_arguments fans out to each collective's slice
  (params.mainloop / params.epilogue), and the kernel body forwards params.kernel /
  params.scheduler / params.mainloop / params.epilogue to the layers that read them.
  Grid/block, previously computed in the launcher, now live in get_grid_shape /
  get_block_shape.

  Shared declarations (the per-layer params blocks, constants, the copy_block_*
  rmem<->smem helpers) come from xe_mla_sparse_2stage_common.hpp.

  Correctness reference: tests/test_flash_mla_with_kvcache.py _sm120_sparse_decode_fwd.
*/

#pragma once

#include "sycl/kernels/mla_sparse/collective/xe_mla_sparse_2stage_epilogue.hpp"
#include "sycl/kernels/mla_sparse/collective/xe_mla_sparse_2stage_mainloop.hpp"
#include "sycl/kernels/mla_sparse/device/xe_mla_sparse_2stage_common.hpp"
#include "sycl/kernels/mla_sparse/kernel/xe_mla_sparse_2stage_tile_scheduler.hpp"

namespace cutlass::flash_attention::kernel {

// Stage-2 only: this kernel knows nothing about Stage 1. It reads the gathered-KV
// tile that Stage 1 already materialized in HBM (via its own kernel params), so it is
// shaped like an ordinary dense MLA kernel -- same collectives + scheduler + Params
// fan-out, no gather template parameter. The config struct pairs it with a Stage-1
// kernel inside device::MLASparse, which launches gather-then-dense.
template <class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_>
class XeMlaSparse2StageDenseKernel {
 public:
  //
  // Type Aliases
  //
  using CollectiveMainloop = CollectiveMainloop_;
  using CollectiveEpilogue = CollectiveEpilogue_;
  using TileScheduler = TileScheduler_;

  using Traits = typename CollectiveMainloop::Traits;
  static constexpr int D_QK = CollectiveMainloop::D_QK;

  static constexpr bool IS_FP8_QUERY = CollectiveMainloop::IS_FP8_QUERY;
  static constexpr bool HAS_ATTN_SINK = CollectiveEpilogue::HAS_ATTN_SINK;
  static constexpr bool HAS_MAX_LOGITS = CollectiveEpilogue::HAS_MAX_LOGITS;
  static constexpr bool is_split_kv = CollectiveEpilogue::IS_SPLIT_KV;
  static constexpr int kvMaxSplits = 16;

  // Per-layer Params/Arguments slices from the collectives + scheduler.
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;
  using TileSchedulerParams = typename TileScheduler::Params;

  using ElementQ = typename CollectiveMainloop::ElementQ;
  using ElementKV = typename CollectiveMainloop::ElementKV;
  using ElementO = typename CollectiveMainloop::ElementO;

  using FragA = typename CollectiveMainloop::FragA;
  using FragARow = typename CollectiveMainloop::FragARow;

  // The mainloop uses no SLM; only the epilogue's ReduceK path does. Union them so
  // SharedStorageSize reflects whichever collective actually needs shared memory.
  union SharedStorage {
    typename CollectiveMainloop::SharedStorage mainloop;
    typename CollectiveEpilogue::SharedStorage epilogue;
  };
  static constexpr int SharedStorageSize = is_empty_v<SharedStorage> ? size_t(0) : sizeof(SharedStorage);

  //
  // Arguments / Params: this kernel's own per-layer params object, exactly like an
  // ordinary dense MLA kernel. to_underlying_arguments fans out per layer, mirroring
  // the fused path (kernel/xe_mla_sparse_kernel.hpp): each collective / the scheduler
  // builds its own slice. Stage 1 has its own separate Params, so nothing here is
  // shared with (or derived from) the gather kernel; the two stages meet only at the
  // gathered-KV HBM buffers named in params.kernel / params.mainloop.
  //
  using Params = SparseAttn2StageParams;
  using Arguments = Params;
  using KernelArguments = Params;

  //
  // Host-side contract for device::MLASparse / launch<>
  //
  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    Params params = args;
    params.mainloop = CollectiveMainloop::to_underlying_arguments(args.mainloop, workspace);
    params.epilogue = CollectiveEpilogue::to_underlying_arguments(args.epilogue, workspace);
    return params;
  }

  static bool can_implement(Arguments const& args) {
    if constexpr (is_split_kv) {
      if (args.scheduler.num_kv_splits < 1 || args.scheduler.num_kv_splits > kvMaxSplits) return false;
      // The split epilogue publishes only partials, so without this scratch the result
      // would silently be garbage: require it rather than fall back.
      if (args.kernel.o_accum == nullptr) return false;
      if (args.epilogue.split_exp_sums == nullptr || args.epilogue.split_max_logits == nullptr) return false;
    } else {
      // A non-split epilogue writes the final row from one work-group, so a split factor
      // > 1 would have every split overwrite the others with a partial result.
      if (args.scheduler.num_kv_splits > 1) return false;
    }
    return CollectiveMainloop::can_implement(args.mainloop) && CollectiveEpilogue::can_implement(args.epilogue);
  }

  // Split-K HBM scratch (o_accum + the two per-split stat arrays). Reported here so the
  // host's workspace accounting -- which is what bounds the batch-chunk size -- covers
  // Stage 2 as well as Stage 1's gathered-KV tile; as on the gather side, the buffers are
  // allocated by the host orchestrator and reach the kernel through the params pointers
  // rather than through the opaque workspace blob. Zero without split-K.
  static size_t get_workspace_size(Arguments const& args) {
    if constexpr (is_split_kv) {
      auto const& s = args.kernel.shape;
      SparseSplitKV2StageWorkspaceLayout layout(
          s.b, s.s_q, s.h_q, args.scheduler.num_kv_splits, Traits::D_V, sizeof(ElementO));
      return layout.total_bytes;
    } else {
      return 0;
    }
  }

  static cutlass::Status initialize_workspace(Arguments const& /* args */, void* /* workspace */ = nullptr) {
    return cutlass::Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const& params) {
    auto const& s = params.kernel.shape;
    return dim3(
        ceil_div(s.h_q, Traits::B_H) * s.s_q * s.b * Traits::V_SPLIT, 1, cute::max(1, params.scheduler.num_kv_splits));
  }

  static dim3 get_block_shape() {
    return dim3(Traits::NUM_THREADS, 1, 1);
  }

  CUTLASS_DEVICE
  void operator()(const Params& params, char* smem_buf) const {
    using namespace sycl::ext::oneapi::this_work_item;

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    auto const& kp = params.kernel;
    auto const& s = kp.shape;

    const ElementQ* q = reinterpret_cast<const ElementQ*>(kp.q);
    ElementO* out = kp.out;

    const int thr_id = int(ThreadIdxX());
    const int sg_id = thr_id / Traits::SUBGROUP_SIZE;
    const int tid_in_sg = thr_id % Traits::SUBGROUP_SIZE;

    // Grid-uniform. The mainloop divides the topk range by it to get this split's slice.
    const int num_kv_splits = cute::max(1, params.scheduler.num_kv_splits);

    TileScheduler tile_scheduler{params.scheduler};
    CUTLASS_PRAGMA_NO_UNROLL
    for (; tile_scheduler.is_valid(); ++tile_scheduler) {
      auto tile = tile_scheduler.get_block_coord();
      const int batch_idx = tile.batch_idx;
      const int seq_idx = tile.seq_idx;
      const int head_bid = tile.head_bid;
      const int v_split_idx = tile.v_split_idx;
      const int cur_head_start_idx = head_bid * Traits::B_H;
      const int cur_v_start_idx = v_split_idx * Traits::D_V_PER_SPLIT;

      // Q [h_q, D_QK] gmem view, offset to (batch, seq).
      auto* q_ptr = q + batch_idx * kp.stride_q_b + seq_idx * kp.stride_q_s_q;
      auto q_layout = make_layout(make_shape(s.h_q, D_QK), make_stride(kp.stride_q_h_q, _1{}));
      Tensor Q = make_tensor(make_gmem_ptr(q_ptr), q_layout);

      // O [h_q, D_V] gmem view, offset to (batch, seq). Under split-K this is instead the
      // o_accum slice for (batch, seq, kv-split): same shape, so the epilogue's store path
      // is untouched and only the base pointer / row stride change. The final `out` is
      // then written by the reduction companion.
      auto o_layout_for = [&](ElementO* base, int stride_h_q) {
        auto layout = make_layout(make_shape(s.h_q, Traits::D_V), make_stride(stride_h_q, _1{}));
        return make_tensor(make_gmem_ptr(base), layout);
      };
      auto O = [&] {
        if constexpr (is_split_kv) {
          ElementO* base = kp.o_accum + batch_idx * kp.stride_o_accum_b + seq_idx * kp.stride_o_accum_s_q +
                           tile.kv_split_idx * kp.stride_o_accum_split;
          return o_layout_for(base, kp.stride_o_accum_h_q);
        } else {
          return o_layout_for(out + batch_idx * kp.stride_o_b + seq_idx * kp.stride_o_s_q, kp.stride_o_h_q);
        }
      }();

      // K == V == the Stage 1 gathered latent (MLA aliasing). K is the full
      // [gathered_topk, D_QK] view (D_QK is 512 or 576); V is the transposed
      // [D_V_PER_SPLIT, gathered_topk] first-D_V sub-view of the same buffer offset
      // to this V-split (V width stays D_V == 512 even when D_QK == 576).
      const auto* gathered_k_ptr =
          kp.gathered_k + batch_idx * kp.stride_gathered_k_b + seq_idx * kp.stride_gathered_k_s_q;
      const auto* gathered_v_ptr = gathered_k_ptr + cur_v_start_idx;
      auto gathered_k_layout =
          make_layout(make_shape(s.gathered_topk, D_QK), make_stride(kp.stride_gathered_k_topk, _1{}));
      auto gathered_v_layout =
          make_layout(make_shape(Traits::D_V_PER_SPLIT, s.gathered_topk), make_stride(_1{}, kp.stride_gathered_k_topk));
      Tensor K = make_tensor(make_gmem_ptr(const_cast<ElementKV*>(gathered_k_ptr)), gathered_k_layout);
      Tensor V = make_tensor(make_gmem_ptr(const_cast<ElementKV*>(gathered_v_ptr)), gathered_v_layout);

      FragA tArA;
      FragARow tA_max, tA_sum;

      CollectiveMainloop mainloop{params.mainloop, shared_storage.mainloop};
      mainloop(Q, K, V, tArA, tA_max, tA_sum, thr_id, batch_idx, seq_idx, head_bid, tile.kv_split_idx, num_kv_splits);

      // Both collectives use the same SLM union; the epilogue's ReduceK reduction
      // reads/writes it via workgroup barriers internally. The mainloop uses no SLM,
      // so no extra barrier is needed between phases here.
      CollectiveEpilogue epilogue{params.epilogue, shared_storage.epilogue};
      epilogue(
          O,
          tArA,
          tA_max,
          tA_sum,
          thr_id,
          sg_id,
          tid_in_sg,
          v_split_idx,
          head_bid,
          cur_head_start_idx,
          batch_idx,
          seq_idx,
          tile.kv_split_idx);
    }
  }
};

}  // namespace cutlass::flash_attention::kernel
