/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
/*! \file
    \brief Two-stage sparse MLA Stage-2 split-K reduction kernel for DeepSeek V4.

    Third and last kernel of the split-K variant of the two-stage sparse MLA path:

        Stage 1  gather+dequant  ->  Stage 2  dense flash (per kv-split partials)
                                 ->  THIS     combine splits + finish the row

*/

#pragma once

#include "sycl/kernels/mla_sparse/device/xe_mla_sparse_2stage_common.hpp"
#include "sycl/kernels/mla_sparse/kernel/xe_mla_sparse_2stage_tile_scheduler.hpp"

namespace cutlass::flash_attention::kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////
// The no-reduction placeholder lives with the runner (device::detail::DummyReduceKernel),
// next to DummyGatherKernel, so the fused path's runner does not have to include this
// header just to name the "no companion" type.
/////////////////////////////////////////////////////////////////////////////////////////////////
// DenseKernel_ is the split-K Stage-2 kernel this reduces for; the reduction derives its
// geometry (D_V), element type, and behavior flags (HAS_ATTN_SINK / HAS_MAX_LOGITS) from
// it so the two cannot disagree about what the row finish should do.
template <class DenseKernel_>
class XeMlaSparse2StageReduceSplitKV {
 public:
  //
  // Type Aliases
  //
  using DenseKernel = DenseKernel_;
  using Traits = typename DenseKernel::Traits;
  using ElementO = typename DenseKernel::ElementO;
  using ElementAcc = float;  // o_accum combines and normalizes in fp32

  using TileScheduler = XeMlaSparse2StageReduceTileScheduler;

  static constexpr bool HAS_ATTN_SINK = DenseKernel::HAS_ATTN_SINK;
  static constexpr bool HAS_MAX_LOGITS = DenseKernel::HAS_MAX_LOGITS;
  static constexpr int kvMaxSplits = DenseKernel::kvMaxSplits;

  static constexpr int D_V = Traits::D_V;
  static constexpr int SUBGROUP_SIZE = Traits::SUBGROUP_SIZE;
  // 4 subgroups (64 threads at sg_size 16): each step has every subgroup read
  // SUBGROUP_SIZE contiguous o_accum elements, and D_V divides evenly by the work-group
  // size so the inner loop needs no bounds check. Kept small so VALS_PER_THREAD stays
  // large enough for the k-outer loop to amortize the per-split rescale over several
  // columns, while the grid (b * s_q * h_q work-groups) supplies the parallelism.
  static constexpr int NUM_SUBGROUPS = 4;
  static constexpr int WG_SIZE = NUM_SUBGROUPS * SUBGROUP_SIZE;
  static_assert(D_V % WG_SIZE == 0, "D_V must be divisible by the reduction work-group size");
  // Compile-time so acc[] below is statically indexed and stays in registers.
  static constexpr int VALS_PER_THREAD = D_V / WG_SIZE;

  //
  // Arguments / Params: the dense kernel's, verbatim (see the header note). Nothing is
  // derived, so to_underlying_arguments is the identity and the runner can build this
  // from the same argument object it builds the dense params from.
  //
  using Params = SparseAttn2StageParams;
  using Arguments = Params;
  using KernelArguments = Params;

  // No SLM: the per-split stats are re-read from gmem by every thread instead of being
  // staged through shared memory, which also removes the work-group barrier.
  struct SharedStorage {};
  static constexpr int SharedStorageSize = 0;

  //
  // Host-side contract for device::MLASparse / launch<>
  //
  static Params to_underlying_arguments(Arguments const& args, void* /* workspace */) {
    return args;
  }

  static bool can_implement(Arguments const& args) {
    auto const& s = args.kernel.shape;
    if (s.b <= 0 || s.s_q <= 0 || s.h_q <= 0) return false;
    if (args.scheduler.num_kv_splits < 1) return false;
    if (args.scheduler.num_kv_splits > kvMaxSplits) return false;
    if (args.kernel.o_accum == nullptr || args.kernel.out == nullptr) return false;
    if (args.epilogue.split_exp_sums == nullptr || args.epilogue.split_max_logits == nullptr) return false;
    return true;
  }

  static int get_workspace_size(Arguments const& /* args */) {
    // The split-K scratch is reported (and allocated) on the dense side; the reduction
    // only consumes it.
    return 0;
  }

  static cutlass::Status initialize_workspace(Arguments const& /* args */, void* /* workspace */ = nullptr) {
    return cutlass::Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::get_grid_shape(params.kernel.shape);
  }

  static dim3 get_block_shape() {
    return dim3(WG_SIZE, 1, 1);
  }

  CUTLASS_DEVICE
  void operator()(const Params& params, char* /* smem_buf */) const {
    using namespace sycl::ext::oneapi::this_work_item;

    auto const& kp = params.kernel;
    auto const& ep = params.epilogue;
    auto const& s = kp.shape;

    const int num_kv_splits = cute::max(1, params.scheduler.num_kv_splits);
    const int thr_id = int(ThreadIdxX());

    TileScheduler tile_scheduler{s};
    CUTLASS_PRAGMA_NO_UNROLL
    for (; tile_scheduler.is_valid(); ++tile_scheduler) {
      const Sparse2StageReduceWorkTile tile = tile_scheduler.get_block_coord();
      const int batch_idx = tile.batch_idx;
      const int seq_idx = tile.seq_idx;
      const int head_idx = tile.head_idx;

      // Per-split partial O: [b, s_q, num_kv_splits, h_q, D_V] -> this row's [splits, D_V].
      Tensor mOaccum = make_tensor(
          make_gmem_ptr(kp.o_accum),
          make_layout(
              make_shape(s.b, s.s_q, num_kv_splits, s.h_q, D_V),
              make_stride(
                  kp.stride_o_accum_b, kp.stride_o_accum_s_q, kp.stride_o_accum_split, kp.stride_o_accum_h_q, _1{})));
      Tensor rOaccum = mOaccum(batch_idx, seq_idx, _, head_idx, _);

      // Final output row: [b, s_q, h_q, D_V] -> this row's [D_V].
      Tensor mOut = make_tensor(
          make_gmem_ptr(kp.out),
          make_layout(
              make_shape(s.b, s.s_q, s.h_q, D_V), make_stride(kp.stride_o_b, kp.stride_o_s_q, kp.stride_o_h_q, _1{})));
      Tensor rOut = mOut(batch_idx, seq_idx, head_idx, _);

      // Per-split softmax stats: [b, s_q, num_kv_splits, h_q] (h_q contiguous) -> [splits].
      auto stat_row = [&](const float* base) {
        Tensor t = make_tensor(
            make_gmem_ptr(base),
            make_layout(
                make_shape(s.b, s.s_q, num_kv_splits, s.h_q),
                make_stride(ep.stride_split_stats_b, ep.stride_split_stats_s_q, ep.stride_split_stats_split, _1{})));
        return t(batch_idx, seq_idx, _, head_idx);
      };
      Tensor split_exp_sums = stat_row(ep.split_exp_sums);
      Tensor split_max_logits = stat_row(ep.split_max_logits);

      // Pass 1: the row's global (log2-domain, sm_scale-folded) max across splits, then the
      // flash-rescaled total exp-sum. Every thread computes both from the same
      // 2 * num_kv_splits gmem scalars -- one L1 line per split serves the whole
      // work-group, which is cheaper than an SLM stage plus a barrier.
      //
      // An empty or fully-masked split published exp_sum == 0; it
      // is skipped in all three loops below, so neither its max (still the sentinel) nor its
      // o_accum slice (never accumulated into) is read.
      ElementAcc global_max = cutlass::platform::numeric_limits<ElementAcc>::lowest();
      CUTLASS_PRAGMA_NO_UNROLL
      for (int k = 0; k < num_kv_splits; ++k) {
        if (split_exp_sums(k) <= ElementAcc(0)) continue;
        global_max = sycl::max(global_max, split_max_logits(k));
      }

      ElementAcc total_exp_sum = ElementAcc(0);
      CUTLASS_PRAGMA_NO_UNROLL
      for (int k = 0; k < num_kv_splits; ++k) {
        const ElementAcc local_exp_sum = split_exp_sums(k);
        if (local_exp_sum <= ElementAcc(0)) continue;
        const ElementAcc local_max = split_max_logits(k);
        total_exp_sum += local_exp_sum * sycl::native::exp2(local_max - global_max);
      }

      // Pre-sink LSE (and the prefill-only pre-sink row max), keyed off total_exp_sum rather
      // than a max sentinel so a fully-masked row reports (-inf max, +inf lse) exactly as
      // the non-split epilogue does. One writer per row.
      const bool row_has_mass = total_exp_sum > ElementAcc(0);
      if (thr_id == 0) {
        const float row_max = row_has_mass ? global_max * LOG_E_2 : -INFINITY;
        const float row_lse = row_has_mass ? row_max + sycl::native::log2(total_exp_sum) * LOG_E_2 : INFINITY;
        Tensor r_Lse = make_tensor(
            make_gmem_ptr(ep.lse),
            make_layout(make_shape(s.b, s.s_q, s.h_q), make_stride(ep.stride_lse_b, ep.stride_lse_s_q, _1{})));
        r_Lse(batch_idx, seq_idx, head_idx) = row_lse;
        if constexpr (HAS_MAX_LOGITS) {
          Tensor rMaxLogits = make_tensor(
              make_gmem_ptr(ep.max_logits),
              make_layout(
                  make_shape(s.b, s.s_q, s.h_q), make_stride(ep.stride_max_logits_b, ep.stride_max_logits_s_q, _1{})));
          rMaxLogits(batch_idx, seq_idx, head_idx) = row_max;
        }
      }

      // attn_sink joins the denominator only, after LSE (which is pre-sink) -- same order
      // and same exp2 formulation as the non-split epilogue's ReduceK == 1 branch.
      if constexpr (HAS_ATTN_SINK) {
        if (row_has_mass) {
          total_exp_sum += sycl::native::exp2(static_cast<ElementAcc>(ep.attn_sink[head_idx] * LOG_2_E) - global_max);
        }
      }
      const ElementAcc inv_exp_sum =
          total_exp_sum != ElementAcc(0) ? sycl::native::recip(total_exp_sum) : ElementAcc(0);

      // Pass 2: combine the partial O columns this thread owns. k-outer / v-inner so the
      // per-split exp2 rescale is hoisted out of the D_V walk and acc[] stays in registers.
      ElementAcc acc[VALS_PER_THREAD];
      CUTLASS_PRAGMA_UNROLL
      for (int v = 0; v < VALS_PER_THREAD; ++v) {
        acc[v] = ElementAcc(0);
      }

      CUTLASS_PRAGMA_NO_UNROLL
      for (int k = 0; k < num_kv_splits; ++k) {
        const ElementAcc local_exp_sum = split_exp_sums(k);
        if (local_exp_sum <= ElementAcc(0)) continue;
        const ElementAcc rescale = sycl::native::exp2(split_max_logits(k) - global_max);
        Tensor o_split = rOaccum(k, _);  // this split's [D_V] row, stride 1
        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < VALS_PER_THREAD; ++v) {
          acc[v] += static_cast<ElementAcc>(o_split(v * WG_SIZE + thr_id)) * rescale;
        }
      }

      CUTLASS_PRAGMA_UNROLL
      for (int v = 0; v < VALS_PER_THREAD; ++v) {
        rOut(v * WG_SIZE + thr_id) = static_cast<ElementO>(acc[v] * inv_exp_sum);
      }
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::flash_attention::kernel
