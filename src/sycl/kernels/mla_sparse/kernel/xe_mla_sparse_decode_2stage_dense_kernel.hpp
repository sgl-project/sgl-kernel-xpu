/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
/*!
  \file
  \brief Two-stage sparse MLA decode Stage 2 kernel for DeepSeek V4.

  DenseDecodeFwdKernel: dense flash-decode over the Stage 1 gathered tile via XMX
  DPAS QK/PV GEMMs, B_H head packing, V-split, online (log2) softmax, and
  attn_sink merge.

  Decomposed into the sycl-tla convention used by the fused MLA kernels: this
  file is the thin kernel wrapper that wires together
    - collective mainloop  (collective/xe_mla_sparse_decode_2stage_mainloop.hpp)
        QK/PV GEMM + online softmax producing the O accumulator + row stats,
    - collective epilogue   (collective/xe_mla_sparse_decode_2stage_epilogue.hpp)
        cross-subgroup reduce, normalize, attn_sink merge, LSE, and store,
    - tile scheduler        (kernel/xe_mla_sparse_decode_2stage_tile_scheduler.hpp)
        (batch, seq, head-block, v-split) work-tile decode.

  Like XeMlaSparseFwdKernel (the fused path), the collectives + scheduler are
  template parameters and the kernel exposes the device::MLASparse / launch<>
  contract (Arguments/Params, to_underlying_arguments, can_implement,
  get_workspace_size, initialize_workspace, get_grid_shape, get_block_shape,
  SharedStorageSize). The config struct MlaSparseDecode2StageXe assembles the
  concrete instantiation (see device/mla_sparse_decode_2stage_types.hpp).

  The kernel keeps the flat SparseAttnDecodeParams as its Arguments == Params (the
  collectives' to_underlying_arguments are identity), so the host adapter
  (args_from_options_2stage) and the cpp.in / dispatch / cmake wiring stay
  unchanged. Grid/block, previously computed in the launcher, now live in
  get_grid_shape / get_block_shape.

  Shared declarations (params, constants, the copy_block_* rmem<->smem helpers)
  come from xe_mla_sparse_decode_2stage_common.hpp.

  Correctness reference: tests/test_flash_mla_with_kvcache.py _sm120_sparse_decode_fwd.
*/

#pragma once

#include "sycl/kernels/mla_sparse/collective/xe_mla_sparse_decode_2stage_epilogue.hpp"
#include "sycl/kernels/mla_sparse/collective/xe_mla_sparse_decode_2stage_mainloop.hpp"
#include "sycl/kernels/mla_sparse/kernel/xe_mla_sparse_decode_2stage_common.hpp"
#include "sycl/kernels/mla_sparse/kernel/xe_mla_sparse_decode_2stage_gather_kernel.hpp"
#include "sycl/kernels/mla_sparse/kernel/xe_mla_sparse_decode_2stage_tile_scheduler.hpp"

namespace cutlass::flash_attention::kernel {

template <class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_>
class DenseDecodeFwdKernel {
 public:
  //
  // Type Aliases
  //
  using CollectiveMainloop = CollectiveMainloop_;
  using CollectiveEpilogue = CollectiveEpilogue_;
  using TileScheduler = TileScheduler_;

  using Traits = typename CollectiveMainloop::Traits;
  static constexpr int D_QK = CollectiveMainloop::D_QK;

  // Stage-1 gather+dequant companion. Declaring this nested type opts the kernel
  // into the device::MLASparse runner's two-launch path (gather then dense); the
  // runner detects it via SFINAE (detail::GatherTraits) and launches it first,
  // sharing this kernel's flat SparseAttnDecodeParams. Kept here (not in the
  // config struct) so the runner can find it off the Kernel type alone.
  using GatherKernel = SparseDecodeGatherDequantKernel<D_QK>;
  static constexpr bool IS_FP8_QUERY = CollectiveMainloop::IS_FP8_QUERY;
  static constexpr bool HAS_ATTN_SINK = CollectiveEpilogue::HAS_ATTN_SINK;

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
  // Arguments / Params: flat SparseAttnDecodeParams (identity to_underlying_arguments),
  // so the launcher / host adapter contract is preserved ("maintain params").
  //
  using Arguments = SparseAttnDecodeParams;
  using KernelArguments = SparseAttnDecodeParams;
  using Params = SparseAttnDecodeParams;

  //
  // Host-side contract for device::MLASparse / launch<>
  //
  static Params to_underlying_arguments(Arguments const& args, void* /* workspace */) {
    return args;
  }

  static bool can_implement(Arguments const& args) {
    return CollectiveMainloop::can_implement(args) && CollectiveEpilogue::can_implement(args);
  }

  static int get_workspace_size(Arguments const& /* args */) {
    return 0;
  }

  static cutlass::Status initialize_workspace(Arguments const& /* args */, void* /* workspace */ = nullptr) {
    return cutlass::Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const& params) {
    return dim3(ceil_div(params.shape.h_q, Traits::B_H) * params.shape.s_q * params.shape.b, Traits::V_SPLIT, 1);
  }

  static dim3 get_block_shape() {
    return dim3(Traits::NUM_THREADS, 1, 1);
  }

  CUTLASS_DEVICE
  void operator()(const Params& params, char* smem_buf) const {
    using namespace sycl::ext::oneapi::this_work_item;

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    const ElementQ* q = reinterpret_cast<const ElementQ*>(params.q);
    ElementO* out = params.out;

    const int thr_id = int(ThreadIdxX());
    const int sg_id = thr_id / Traits::SUBGROUP_SIZE;
    const int tid_in_sg = thr_id % Traits::SUBGROUP_SIZE;

    TileScheduler tile_scheduler{params.shape.h_q, params.shape.s_q};
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
      auto* q_ptr = q + batch_idx * params.stride_q_b + seq_idx * params.stride_q_s_q;
      auto q_layout = make_layout(make_shape(params.shape.h_q, D_QK), make_stride(params.stride_q_h_q, _1{}));
      Tensor Q = make_tensor(make_gmem_ptr(q_ptr), q_layout);

      // O [h_q, D_V] gmem view, offset to (batch, seq).
      auto* out_ptr = out + batch_idx * params.stride_o_b + seq_idx * params.stride_o_s_q;
      auto o_layout = make_layout(make_shape(params.shape.h_q, Traits::D_V), make_stride(params.stride_o_h_q, _1{}));
      Tensor O = make_tensor(make_gmem_ptr(out_ptr), o_layout);

      // K == V == the Stage 1 gathered 512-dim latent (MLA aliasing). K is the full
      // [gathered_topk, D_QK] view; V is the transposed [D_V_PER_SPLIT, gathered_topk]
      // view of the same buffer offset to this V-split.
      const auto* gathered_k_ptr =
          params.gathered_k + batch_idx * params.stride_gathered_k_b + seq_idx * params.stride_gathered_k_s_q;
      const auto* gathered_v_ptr = gathered_k_ptr + cur_v_start_idx;
      auto gathered_k_layout =
          make_layout(make_shape(params.shape.gathered_topk, D_QK), make_stride(params.stride_gathered_k_topk, _1{}));
      auto gathered_v_layout = make_layout(
          make_shape(Traits::D_V_PER_SPLIT, params.shape.gathered_topk),
          make_stride(_1{}, params.stride_gathered_k_topk));
      Tensor K = make_tensor(make_gmem_ptr(const_cast<ElementKV*>(gathered_k_ptr)), gathered_k_layout);
      Tensor V = make_tensor(make_gmem_ptr(const_cast<ElementKV*>(gathered_v_ptr)), gathered_v_layout);

      FragA tArA;
      FragARow tA_max, tA_sum;

      CollectiveMainloop mainloop{params, shared_storage.mainloop};
      mainloop(Q, K, V, tArA, tA_max, tA_sum, thr_id, batch_idx, seq_idx, head_bid);

      // Both collectives use the same SLM union; the epilogue's ReduceK reduction
      // reads/writes it via workgroup barriers internally. The mainloop uses no SLM,
      // so no extra barrier is needed between phases here.
      CollectiveEpilogue epilogue{params, shared_storage.epilogue};
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
          seq_idx);
    }
  }
};

}  // namespace cutlass::flash_attention::kernel
