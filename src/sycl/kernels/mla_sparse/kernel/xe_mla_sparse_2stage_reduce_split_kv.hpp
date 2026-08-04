/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
/*! \file
    \brief Two-stage sparse MLA Stage-2 split-K reduction kernel for DeepSeek V4.

    Third and last kernel of the split-K variant of the two-stage sparse MLA path:

        Stage 1  gather+dequant  ->  Stage 2  dense flash (per kv-split partials)
                                 ->  THIS     combine splits + finish the row

    When the Stage-2 dense kernel is instantiated with a split-K epilogue
    (XeMlaSparse2StageEpilogue<..., IS_SPLIT_KV = true>), each kv-split work-group only
    covers a slice of the gathered topk dim, so it can produce an UNNORMALIZED partial O
    plus this split's online-softmax row stats but not the finished row. This kernel does
    the cross-split flash rescale-and-combine and everything that depends on the whole
    row: the attn_sink merge, the softmax normalization, the pre-sink LSE, and (prefill)
    max_logits. Without split-K it is never instantiated or launched, and the fused /
    non-split two-stage paths are unchanged.

    Structural analog of the paged path's XeMlaReduceSplitKV
    (mla/kernel/xe_mla_reduce_split_kv.hpp), with three deliberate differences:

      - Params ARE the dense kernel's Params (SparseAttn2StageParams). Every tensor the
        reduction touches -- o_accum / split stats (kernel + epilogue slices), the final
        out / lse / max_logits / attn_sink (kernel + epilogue slices), num_kv_splits
        (scheduler slice) -- already lives there, so there is no separate argument type to
        keep in sync and the runner hands both kernels the same object.

      - No SLM and no work-group barrier. The paged kernel stages the per-split stats
        through SLM (which is what forces its kvMaxSplits-sized arrays); here every thread
        in the work-group needs the same 2 * num_kv_splits scalars, and re-reading them
        from gmem costs one L1 line per split for the whole work-group. That drops the
        barrier, the shared storage, and the kvMaxSplits device-side dependency.

      - One work-group per (batch, seq, head) row rather than a tile scheduler over
        (seq, head, batch): the row is exactly D_V wide, which one work-group covers in
        VALS_PER_THREAD statically-unrolled steps, so the k-outer / v-inner loop keeps the
        partial sums in registers (a v-outer / k-inner loop would need one exp2 per element
        per split, or a runtime-indexed rescale array that would spill).
*/

#pragma once

#include "sycl/kernels/mla_sparse/device/xe_mla_sparse_2stage_common.hpp"

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

  // One work-group per (batch, seq, head) output row. head is the fastest-varying index
  // so neighbouring work-groups walk contiguous o_accum rows (its layout is
  // [b, s_q, num_kv_splits, h_q, d_v]).
  static dim3 get_grid_shape(Params const& params) {
    auto const& s = params.kernel.shape;
    return dim3(s.b * s.s_q * s.h_q, 1, 1);
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

    const int row = int(BlockIdxX());
    const int head_idx = row % s.h_q;
    const int q_tile = row / s.h_q;
    const int seq_idx = q_tile % s.s_q;
    const int batch_idx = q_tile / s.s_q;

    // Per-split stats for this row: [b, s_q, num_kv_splits, h_q], split-strided.
    const int stat_base = batch_idx * ep.stride_split_stats_b + seq_idx * ep.stride_split_stats_s_q + head_idx;
    const float* __restrict__ split_exp_sums = ep.split_exp_sums + stat_base;
    const float* __restrict__ split_max_logits = ep.split_max_logits + stat_base;

    // Pass 1: the row's global (log2-domain, sm_scale-folded) max across splits, then the
    // flash-rescaled total exp-sum. Every thread computes both from the same
    // 2 * num_kv_splits gmem scalars -- one L1 line per split serves the whole
    // work-group, which is cheaper than an SLM stage plus a barrier.
    //
    // An empty or fully-masked split published exp_sum == 0 (see the split epilogue); it
    // is skipped in all three loops below, so neither its max (still the sentinel) nor its
    // o_accum slice (never accumulated into) is read.
    ElementAcc global_max = cutlass::platform::numeric_limits<ElementAcc>::lowest();
    CUTLASS_PRAGMA_NO_UNROLL
    for (int k = 0; k < num_kv_splits; ++k) {
      if (split_exp_sums[k * ep.stride_split_stats_split] <= ElementAcc(0)) continue;
      global_max = sycl::max(global_max, split_max_logits[k * ep.stride_split_stats_split]);
    }

    ElementAcc total_exp_sum = ElementAcc(0);
    CUTLASS_PRAGMA_NO_UNROLL
    for (int k = 0; k < num_kv_splits; ++k) {
      const ElementAcc local_exp_sum = split_exp_sums[k * ep.stride_split_stats_split];
      if (local_exp_sum <= ElementAcc(0)) continue;
      const ElementAcc local_max = split_max_logits[k * ep.stride_split_stats_split];
      total_exp_sum += local_exp_sum * sycl::native::exp2(local_max - global_max);
    }

    // Pre-sink LSE (and the prefill-only pre-sink row max), keyed off total_exp_sum rather
    // than a max sentinel so a fully-masked row reports (-inf max, +inf lse) exactly as
    // the non-split epilogue does. One writer per row.
    const bool row_has_mass = total_exp_sum > ElementAcc(0);
    if (thr_id == 0) {
      const float row_max = row_has_mass ? global_max * LOG_E_2 : -INFINITY;
      const float row_lse = row_has_mass ? row_max + sycl::native::log2(total_exp_sum) * LOG_E_2 : INFINITY;
      ep.lse[batch_idx * ep.stride_lse_b + seq_idx * ep.stride_lse_s_q + head_idx] = row_lse;
      if constexpr (HAS_MAX_LOGITS) {
        ep.max_logits[batch_idx * ep.stride_max_logits_b + seq_idx * ep.stride_max_logits_s_q + head_idx] = row_max;
      }
    }

    // attn_sink joins the denominator only, after LSE (which is pre-sink) -- same order
    // and same exp2 formulation as the non-split epilogue's ReduceK == 1 branch.
    if constexpr (HAS_ATTN_SINK) {
      total_exp_sum += sycl::native::exp2(static_cast<ElementAcc>(ep.attn_sink[head_idx] * LOG_2_E) - global_max);
    }
    const ElementAcc inv_exp_sum = total_exp_sum != ElementAcc(0) ? ElementAcc(1) / total_exp_sum : ElementAcc(0);

    // Pass 2: combine the partial O columns this thread owns. k-outer / v-inner so the
    // per-split exp2 rescale is hoisted out of the D_V walk and acc[] stays in registers.
    const ElementO* __restrict__ o_accum = kp.o_accum + batch_idx * kp.stride_o_accum_b +
                                           seq_idx * kp.stride_o_accum_s_q + head_idx * kp.stride_o_accum_h_q;

    ElementAcc acc[VALS_PER_THREAD];
    CUTLASS_PRAGMA_UNROLL
    for (int v = 0; v < VALS_PER_THREAD; ++v) {
      acc[v] = ElementAcc(0);
    }

    CUTLASS_PRAGMA_NO_UNROLL
    for (int k = 0; k < num_kv_splits; ++k) {
      const ElementAcc local_exp_sum = split_exp_sums[k * ep.stride_split_stats_split];
      if (local_exp_sum <= ElementAcc(0)) continue;
      const ElementAcc rescale = sycl::native::exp2(split_max_logits[k * ep.stride_split_stats_split] - global_max);
      const ElementO* __restrict__ o_split = o_accum + k * kp.stride_o_accum_split;
      CUTLASS_PRAGMA_UNROLL
      for (int v = 0; v < VALS_PER_THREAD; ++v) {
        acc[v] += static_cast<ElementAcc>(o_split[v * WG_SIZE + thr_id]) * rescale;
      }
    }

    ElementO* __restrict__ out =
        kp.out + batch_idx * kp.stride_o_b + seq_idx * kp.stride_o_s_q + head_idx * kp.stride_o_h_q;
    CUTLASS_PRAGMA_UNROLL
    for (int v = 0; v < VALS_PER_THREAD; ++v) {
      out[v * WG_SIZE + thr_id] = static_cast<ElementO>(acc[v] * inv_exp_sum);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::flash_attention::kernel
