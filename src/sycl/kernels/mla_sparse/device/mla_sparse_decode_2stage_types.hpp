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
/*!
  \file
  \brief Two-stage sparse MLA decode Stage-2 config + host orchestrator for DeepSeek V4.

  Config struct and op-facing run hierarchy for the two-stage sparse MLA decode path,
  layered like the fused path's mla_sparse_decode_types.hpp (config MlaSparseXe + the
  run* orchestration in one types header):

    - MlaSparseDecode2StageXe<T, D_QK, HAS_ATTN_SINK, B_H, GatherKernel, V_SPLIT>:
        the DPAS/tile config struct that assembles the collectives + tile scheduler +
        dense kernel wrapper + Stage-1 gather kernel + device::MLASparse runner
        (analog of MlaSparseXe, and of MlaXe wiring its split-KV attention and
        reduction kernels into device::MLA). T is the op's query dtype, resolved to
        ElementQ via SparseMlaToCutlassElementType (IS_FP8_QUERY is deduced from it).
    - args_from_options_2stage<Config>: adapts our tensor arguments to the runner's
        two-stage Arguments ({dense, gather}); no allocation, no launch.
    - runMlaSparse2StageImpl<Element, D_QK, B_H, HAS_ATTN_SINK>: resolves the Stage-2
        Config from the template params, allocates the dense gathered-KV + valid-mask HBM
        workspaces (batch-chunked to bound peak memory) and runs the launch loop against
        Config::Fmla.
    - runMlaSparse2Stage<Element, D_QK, B_H, HAS_ATTN_SINK>: op-facing entry.
        Validates inputs and forwards the dispatched head dim + head-block size +
        attn_sink flag to the Impl (which resolves the Config). Its signature matches the
        generated instantiation stub (mla_sparse_decode_2stage_kernel.cpp.in).

  runMlaSparse2StageImpl / runMlaSparse2Stage are templated on the config-keying params
  (head dim D_QK + head-block size B_H + attn_sink); the Impl resolves the Stage-2
  Config from them, so the heavy Config::Fmla instantiation is keyed by (D_QK, B_H)
  exactly the way the fused decode path is keyed by page size. Each generated launcher
  launch_mla_sparse_decode_2stage_<ELEM>_<D_QK>_<B_H>_<HAS_ATTN_SINK> (from
  mla_sparse_decode_2stage_kernel.cpp.in) instantiates a single (D_QK, B_H, sink) variant
  in its own TU, so the CUTLASS codegen for one variant lands in a separate object file
  (build OOM guard preserved -- one variant per file). The op (mla_sparse_decode.cpp)
  dispatches dtype, then D_QK, then B_H, then the runtime attn_sink flag, mirroring the
  fused path's dtype-then-page-size dispatch. Decode is always D_QK == 512.

  This is an ALTERNATIVE to the fused sparse MLA decode path in
  kernels/mla_sparse/{collective,kernel,device}/. It is selected at compile time via
  the SGLANG_USE_SPARSE_MLA_2STAGE macro (see mla_sparse_decode.cpp) and is decode-only.
*/

#pragma once

#ifndef SYCL_INTEL_TARGET
#define SYCL_INTEL_TARGET 20
#endif

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <sycl/sycl.hpp>

#include "../../../Utils.h"  // CUTLASS_CHECK (used by mla_sparse_runner.hpp)
#include "cutlass/bfloat16.h"
#include "cutlass/float8.h"
// The collective headers pull in the full cute/cutlass sycl-tla stack (defining
// cute::intel, etc.) that mla_sparse_runner.hpp -> comm/common.h references. They
// sort under collective/ (before device/), so the runner always sees cute::intel
// even after include re-alphabetization. (Matches the fused path, which likewise
// includes its collectives before the runner.)
#include "sycl/kernels/mla_sparse/collective/xe_mla_sparse_2stage_epilogue.hpp"
#include "sycl/kernels/mla_sparse/collective/xe_mla_sparse_2stage_mainloop.hpp"
#include "sycl/kernels/mla_sparse/device/mla_sparse_runner.hpp"
// The two stages' kernels, included as peers: the config struct below resolves one of
// each (GatherKernel / DenseKernel) and hands both to the runner, which launches them
// in order. The dense kernel transitively includes the common prologue (the per-layer params blocks /
// LOG_2_E / the V-split knobs / DISPATCH_BOOLEAN_FLAG), the tile scheduler, and both
// collectives; the gather kernel is independent of it.
#include "sycl/kernels/mla_sparse/kernel/xe_mla_sparse_2stage_dense_kernel.hpp"
#include "sycl/kernels/mla_sparse/kernel/xe_mla_sparse_2stage_gather_kernel.hpp"
// Optional third stage, instantiated only when the config's IS_SPLIT_KV is true.
#include "sycl/kernels/mla_sparse/kernel/xe_mla_sparse_2stage_reduce_split_kv.hpp"

namespace cutlass::flash_attention::kernel {

//----------------- Stage-2 dense-decode Xe configuration --------------------//
// Assembly layer: picks the tile geometry, the three collectives + kernel wrapper,
// the Stage-1 gather companion, and the device::MLASparse runner, mirroring the fused
// path's MlaSparseXe (and MlaXe on the dense side).
//
// The DPAS/tile geometry itself lives in MlaSparseDecode2StageTileTraits
// (xe_mla_sparse_2stage_common.hpp, next to the params blocks the same collectives
// read) and is what they receive as their `Traits`. That two-layer split is
// deliberate: this struct used to pass
// *itself* as the collectives' Traits, i.e. name itself as a template argument while
// still incomplete, which worked only because every alias in the chain is lazy -- and
// which let the collectives reach members that are none of their business (Fmla, the
// runner that contains them; GatherKernel, the other stage). The traits type is
// complete before any collective names it, so that cycle is gone. See the header
// comment there for the full rationale.
//
// Both stage kernels keep Arguments == Params, so the device::MLASparse
// Arguments->Params flow is a per-stage identity and the host adapter
// (args_from_options_2stage) fills the params directly. GrfSize is 256: this
// dense-decode kernel is fragment-heavy and would spill at the runner's default 128.
// GatherKernelTmpl_ selects the Stage-1 companion (keyed on D_QK). Defaults to the
// decode gather (packed fp8 -> bf16 dequant); the prefill path reuses this exact
// config struct but supplies SparsePrefillGatherKernel (dense bf16 copy). All other
// template args and the whole DPAS/tile/collective assembly are shared verbatim.
//
// T is the op's query element (sycl::half / sycl::ext::oneapi::bfloat16), forwarded to
// the traits, which resolves it to a cutlass element via SparseMlaToCutlassElementType
// exactly like the fused config MlaSparseXe. This lets the run* entry points
// instantiate the config straight from the dispatched dtype instead of branching on it.
template <
    typename T,
    int D_QK_,
    bool HAS_ATTN_SINK_,
    int B_H_,
    template <int> class GatherKernelTmpl_ = SparseDecodeGatherDequantKernel,
    int V_SPLIT_ = FLASH_MLA_PREFILL_V_SPLIT,
    bool HAS_MAX_LOGITS_ = false,
    bool IS_SPLIT_KV_ = false>
struct MlaSparseDecode2StageXe {
  // Stage-2 DPAS / tile geometry. Complete at this point, so it can be handed to the
  // collectives below without the former self-reference.
  using TileTraits = MlaSparseDecode2StageTileTraits<T, D_QK_, B_H_, V_SPLIT_>;

  static constexpr int D_QK = D_QK_;
  static constexpr bool HAS_ATTN_SINK = HAS_ATTN_SINK_;
  // Prefill returns the pre-sink row max (max_logits) alongside lse; decode does not.
  // Threaded into the epilogue so the extra store is compiled out for decode.
  static constexpr bool HAS_MAX_LOGITS = HAS_MAX_LOGITS_;
  // Re-exported for the host side: the Impl only allocates the split-K scratch and
  // resolves num_kv_splits when this is true.
  static constexpr bool IS_SPLIT_KV = IS_SPLIT_KV_;

  // Re-exported for the host side: the run* Impls TORCH_CHECK the op's d_qk / d_v
  // against these, and D_QK also keys the gather kernel below.
  static constexpr int D_V = TileTraits::D_V;
  static constexpr bool IS_FP8_QUERY = TileTraits::IS_FP8_QUERY;

  // Collective mainloop / epilogue + tile scheduler + kernel wrapper, parameterized on
  // the tile geometry above.
  using CollectiveMainloop =
      cutlass::flash_attention::collective::XeMlaSparse2StageMainloop<D_QK, IS_FP8_QUERY, TileTraits>;
  using CollectiveEpilogue = cutlass::flash_attention::collective::
      XeMlaSparse2StageEpilogue<CollectiveMainloop, HAS_ATTN_SINK, HAS_MAX_LOGITS, IS_SPLIT_KV>;
  using TileScheduler = cutlass::flash_attention::kernel::XeMlaSparse2StageIndividualTileScheduler<B_H_, V_SPLIT_>;

  using DenseKernel = cutlass::flash_attention::kernel::
      XeMlaSparse2StageDenseKernel<CollectiveMainloop, CollectiveEpilogue, TileScheduler>;

  // Stage-3 split-K reduction companion, present only under IS_SPLIT_KV. It derives
  // everything (geometry, element type, HAS_ATTN_SINK / HAS_MAX_LOGITS, kvMaxSplits) from
  // DenseKernel and shares its Params type, so there is nothing to keep in sync here.
  using ReduceKernel = cute::conditional_t<
      IS_SPLIT_KV,
      cutlass::flash_attention::kernel::XeMlaSparse2StageReduceSplitKV<DenseKernel>,
      cutlass::flash_attention::device::detail::DummyReduceKernel>;

  // Largest split factor the host heuristic may pick, re-exported from the kernel.
  static constexpr int kvMaxSplits = DenseKernel::kvMaxSplits;

  // Stage-1 gather kernel: an independent kernel with its own Arguments/Params,
  // selected here (decode dequant vs prefill dense copy). This config struct is the
  // single place that knows about both stages; neither kernel references the other.
  using GatherKernel = GatherKernelTmpl_<D_QK>;

  // Both stages wired into one runner, the way MlaXe wires the split-KV attention +
  // reduction kernels into device::MLA: Fmla::run issues gather-then-dense on the
  // in-order queue, and Fmla::Arguments carries one argument object per stage
  // (.gather / .dense).
  //
  // Dense GrfSize is 256 (see note above); XE3P's 512-GRF mode from the prior manual
  // launch is capped to 256 by the shared launch<> helper's {128,256} constraint. The
  // gather's GRF mode is picked by the runner (MLASparse::kGatherGrfSize).
  // The reduction companion is a no-op placeholder unless IS_SPLIT_KV, in which case
  // Fmla::run issues gather -> dense -> reduce on the in-order queue.
  using Fmla = cutlass::flash_attention::device::MLASparse<DenseKernel, 256, GatherKernel, ReduceKernel>;
};

// ---------------------------------------------------------------------------
// Split-K build switch.
// ---------------------------------------------------------------------------
#ifndef FLASH_MLA_SPARSE_2STAGE_SPLIT_K
#define FLASH_MLA_SPARSE_2STAGE_SPLIT_K 1
#endif

// ---------------------------------------------------------------------------
// Resolves the split-K factor for one call.
// ---------------------------------------------------------------------------
static inline int resolve_sparse_2stage_num_kv_splits(
    int b, int s_q, int h_q, int B_H, int V_SPLIT, int gathered_topk, int B_TOPK, int kv_max_splits) {
  // will remove the case for b == 8 and gathered_topk >= 512 in the future.
  if (b == 8 && gathered_topk >= 512) {
    return 1;
  }
  const int num_topk_blocks = std::max(1, (gathered_topk + B_TOPK - 1) / B_TOPK);
  constexpr int WGS_PER_CORE = 8;
  constexpr int MIN_BLOCKS_PER_SPLIT = 2;

  const int64_t base_wgs = int64_t((h_q + B_H - 1) / B_H) * s_q * b * V_SPLIT;
  if (base_wgs <= 0) return 1;
  const int64_t target_wgs = int64_t(WGS_PER_CORE) * dpcppMaxComputeUnitSize();

  int splits = static_cast<int>(std::max<int64_t>(1, target_wgs / base_wgs));
  splits = std::min(splits, kv_max_splits);
  splits = std::min(splits, std::max(1, num_topk_blocks / MIN_BLOCKS_PER_SPLIT));

  return std::max(1, splits);
}

}  // namespace cutlass::flash_attention::kernel

template <typename T>
inline typename T::Fmla::Arguments args_from_options_2stage(
    at::Tensor& out,                                     // [B, 1, H, head_dim_v]
    at::Tensor& lse_out,                                 // [B, H, 1] (contiguous [B,1,H])
    const at::Tensor& q,                                 // [B, 1, H, D_qk=512]
    const at::Tensor& k_cache,                           // [num_pages, page_size, 1, 584] fp8 packed
    const at::Tensor& indices,                           // [B, 1, topk]
    const std::optional<at::Tensor>& topk_length,        // [B] or nullopt
    const std::optional<at::Tensor>& extra_k_cache,      // [num_ext_pg, ep, 1, 584] or nullopt
    const std::optional<at::Tensor>& extra_indices,      // [B, 1, extra_topk] or nullopt
    const std::optional<at::Tensor>& extra_topk_length,  // [B] or nullopt
    const std::optional<at::Tensor>& attn_sink,          // [H] or nullopt
    const at::Tensor& gathered_k,                        // [chunk_b, 1, gathered_topk, 512] bf16 workspace
    const at::Tensor& gathered_valid_mask,               // [chunk_b, 1, gathered_topk] int workspace
    double sm_scale,
    int64_t head_dim_v,
    const std::optional<at::Tensor>& o_accum = std::nullopt,           // [chunk_b, 1, splits, H, D_V] bf16
    const std::optional<at::Tensor>& split_exp_sums = std::nullopt,    // [chunk_b, 1, splits, H] fp32
    const std::optional<at::Tensor>& split_max_logits = std::nullopt,  // [chunk_b, 1, splits, H] fp32
    int num_kv_splits = 1) {
  namespace F = cutlass::flash_attention::kernel;

  const int b = q.size(0);
  const int s_q = q.size(1);
  const int h_q = q.size(2);
  const int d_qk = q.size(3);
  const int d_v = static_cast<int>(head_dim_v);

  const int num_blocks = k_cache.size(0);
  const int page_block_size = k_cache.size(1);
  const int topk = indices.size(2);

  const bool has_extra = extra_k_cache.has_value() && extra_indices.has_value();
  const int extra_num_blocks = has_extra ? static_cast<int>(extra_k_cache.value().size(0)) : 0;
  const int extra_page_block_size = has_extra ? static_cast<int>(extra_k_cache.value().size(1)) : 0;
  const int extra_topk = has_extra ? static_cast<int>(extra_indices.value().size(2)) : 0;
  const int gathered_topk = topk + extra_topk;

  auto to_int_stride = [](int64_t s) {
    TORCH_CHECK(s <= std::numeric_limits<int>::max(), "Stride exceeds int32 limit: ", s);
    return static_cast<int>(s);
  };

  const float sm_scale_div_log2 = static_cast<float>(sm_scale) * F::LOG_2_E;

  // Problem geometry, built once and carried intact in the kernel slice.
  F::SparseDecode2StageProblemShape shape;
  shape.b = b;
  shape.s_q = s_q;
  shape.h_q = h_q;
  shape.h_kv = 1;
  shape.d_qk = d_qk;
  shape.d_v = d_v;
  shape.num_blocks = num_blocks;
  shape.page_block_size = page_block_size;
  shape.topk = topk;
  shape.gathered_topk = gathered_topk;
  shape.extra_num_blocks = extra_num_blocks;
  shape.extra_page_block_size = extra_page_block_size;
  shape.extra_topk = extra_topk;

  typename T::Fmla::Arguments args{};
  auto& params = args.dense;

  // --- Kernel slice: Q / O / gathered-K tensors + the intact problem shape. ---
  auto& k = params.kernel;
  k.shape = shape;
  k.q = q.data_ptr();
  k.stride_q_b = to_int_stride(q.stride(0));
  k.stride_q_s_q = to_int_stride(q.stride(1));
  k.stride_q_h_q = to_int_stride(q.stride(2));
  k.gathered_k = reinterpret_cast<cutlass::bfloat16_t*>(gathered_k.data_ptr());
  k.stride_gathered_k_b = to_int_stride(gathered_k.stride(0));
  k.stride_gathered_k_s_q = to_int_stride(gathered_k.stride(1));
  k.stride_gathered_k_topk = to_int_stride(gathered_k.stride(2));
  k.out = reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr());
  k.stride_o_b = to_int_stride(out.stride(0));
  k.stride_o_s_q = to_int_stride(out.stride(1));
  k.stride_o_h_q = to_int_stride(out.stride(2));

  // --- Mainloop slice: QK/PV/softmax reads. ---
  auto& ml = params.mainloop;
  ml.h_q = h_q;
  ml.topk = topk;
  ml.extra_topk = extra_topk;
  ml.gathered_topk = gathered_topk;
  ml.sm_scale_div_log2 = sm_scale_div_log2;
  ml.q = q.data_ptr();  // fp8 re-read path (bf16 query leaves it unused)
  ml.stride_q_b = k.stride_q_b;
  ml.stride_q_s_q = k.stride_q_s_q;
  ml.stride_q_h_q = k.stride_q_h_q;
  ml.q_scale = nullptr;
  ml.q_scale_numel = 0;
  ml.gathered_valid_mask = reinterpret_cast<int*>(gathered_valid_mask.data_ptr());
  ml.stride_gathered_mask_b = to_int_stride(gathered_valid_mask.stride(0));
  ml.stride_gathered_mask_s_q = to_int_stride(gathered_valid_mask.stride(1));
  ml.topk_length = topk_length.has_value() ? reinterpret_cast<int*>(topk_length.value().data_ptr()) : nullptr;
  ml.stride_topk_length_b = topk_length.has_value() ? to_int_stride(topk_length.value().stride(0)) : 0;
  ml.extra_topk_length =
      extra_topk_length.has_value() ? reinterpret_cast<int*>(extra_topk_length.value().data_ptr()) : nullptr;
  ml.stride_extra_topk_length_b =
      extra_topk_length.has_value() ? to_int_stride(extra_topk_length.value().stride(0)) : 0;

  // --- Epilogue slice: reduce / normalize / LSE / attn_sink (no max_logits for decode). ---
  auto& ep = params.epilogue;
  ep.h_q = h_q;
  ep.sm_scale_div_log2 = sm_scale_div_log2;
  ep.lse = reinterpret_cast<float*>(lse_out.data_ptr());
  ep.stride_lse_b = to_int_stride(lse_out.stride(0));
  ep.stride_lse_s_q = to_int_stride(lse_out.stride(1));
  ep.attn_sink = attn_sink.has_value() ? static_cast<float*>(attn_sink.value().data_ptr()) : nullptr;

  // --- Stage-1 gather params (decode): dual packed-fp8 paged pools. Independent of
  // the dense params above; the shared gathered_k / mask / topk_length pointers are
  // copied across so each stage's params is self-contained. ---
  auto& g = args.gather;
  g.b = b;
  g.s_q = s_q;
  g.topk = topk;
  g.gathered_topk = gathered_topk;
  g.indices = reinterpret_cast<int*>(indices.data_ptr());
  g.stride_indices_b = to_int_stride(indices.stride(0));
  g.stride_indices_s_q = to_int_stride(indices.stride(1));
  g.topk_length = ml.topk_length;
  g.stride_topk_length_b = ml.stride_topk_length_b;
  g.gathered_k = k.gathered_k;
  g.stride_gathered_k_b = k.stride_gathered_k_b;
  g.stride_gathered_k_s_q = k.stride_gathered_k_s_q;
  g.stride_gathered_k_topk = k.stride_gathered_k_topk;
  g.gathered_valid_mask = ml.gathered_valid_mask;
  g.stride_gathered_mask_b = ml.stride_gathered_mask_b;
  g.stride_gathered_mask_s_q = ml.stride_gathered_mask_s_q;
  g.num_blocks = num_blocks;
  g.page_block_size = page_block_size;
  g.extra_num_blocks = extra_num_blocks;
  g.extra_page_block_size = extra_page_block_size;
  g.extra_topk = extra_topk;
  g.kv = reinterpret_cast<uint8_t*>(k_cache.data_ptr());
  g.stride_kv_block = to_int_stride(k_cache.stride(0));
  g.extra_kv = has_extra ? reinterpret_cast<uint8_t*>(extra_k_cache.value().data_ptr()) : nullptr;
  g.stride_extra_kv_block = has_extra ? to_int_stride(extra_k_cache.value().stride(0)) : 0;
  g.extra_indices = has_extra ? reinterpret_cast<int*>(extra_indices.value().data_ptr()) : nullptr;
  g.stride_extra_indices_b = has_extra ? to_int_stride(extra_indices.value().stride(0)) : 0;
  g.stride_extra_indices_s_q = has_extra ? to_int_stride(extra_indices.value().stride(1)) : 0;
  g.extra_topk_length = ml.extra_topk_length;
  g.stride_extra_topk_length_b = ml.stride_extra_topk_length_b;

  // --- Tile scheduler slice. ---
  params.scheduler.h_q = h_q;
  params.scheduler.s_q = s_q;
  params.scheduler.num_kv_splits = num_kv_splits;

  if constexpr (T::IS_SPLIT_KV) {
    TORCH_CHECK(num_kv_splits >= 1, "num_kv_splits must be >= 1, got ", num_kv_splits);
    TORCH_CHECK(
        o_accum.has_value() && split_exp_sums.has_value() && split_max_logits.has_value(),
        "2-stage sparse MLA split-K requires the o_accum / split_exp_sums / split_max_logits workspaces");

    const F::SparseSplitKV2StageWorkspaceLayout layout(b, s_q, h_q, num_kv_splits, T::D_V, sizeof(cutlass::bfloat16_t));

    k.o_accum = reinterpret_cast<cutlass::bfloat16_t*>(o_accum.value().data_ptr());
    k.stride_o_accum_b = layout.stride_o_accum_b;
    k.stride_o_accum_s_q = layout.stride_o_accum_s_q;
    k.stride_o_accum_split = layout.stride_o_accum_split;
    k.stride_o_accum_h_q = layout.stride_o_accum_h_q;

    ep.split_exp_sums = reinterpret_cast<float*>(split_exp_sums.value().data_ptr());
    ep.split_max_logits = reinterpret_cast<float*>(split_max_logits.value().data_ptr());
    ep.stride_split_stats_b = layout.stride_split_stats_b;
    ep.stride_split_stats_s_q = layout.stride_split_stats_s_q;
    ep.stride_split_stats_split = layout.stride_split_stats_split;
  }

  return args;
}

template <typename Element, int D_QK, int B_H, bool HAS_ATTN_SINK>
inline void runMlaSparse2StageImpl(
    at::Tensor& out,
    at::Tensor& lse_out,
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& indices,
    const std::optional<at::Tensor>& topk_length,
    const std::optional<at::Tensor>& extra_k_cache,
    const std::optional<at::Tensor>& extra_indices,
    const std::optional<at::Tensor>& extra_topk_length,
    const std::optional<at::Tensor>& attn_sink,
    double sm_scale,
    int64_t head_dim_v) {
  namespace F = cutlass::flash_attention::kernel;

  using MlaSparseDecode2StageXeType = F::MlaSparseDecode2StageXe<
      Element,
      D_QK,
      HAS_ATTN_SINK,
      B_H,
      F::SparseDecodeGatherDequantKernel,
      FLASH_MLA_PREFILL_V_SPLIT,
      /* HAS_MAX_LOGITS */ false,
      /* IS_SPLIT_KV */ FLASH_MLA_SPARSE_2STAGE_SPLIT_K != 0>;
  using TileTraits = typename MlaSparseDecode2StageXeType::TileTraits;
  static constexpr bool kSplitK = MlaSparseDecode2StageXeType::IS_SPLIT_KV;

  const int b = q.size(0);
  const int s_q = q.size(1);
  const int h_q = q.size(2);
  const int d_qk = q.size(3);
  const int topk = indices.size(2);
  const bool has_extra = extra_k_cache.has_value() && extra_indices.has_value();
  const int extra_topk = has_extra ? static_cast<int>(extra_indices.value().size(2)) : 0;
  const int gathered_topk = topk + extra_topk;

  auto device = q.device();
  const c10::DeviceGuard device_guard(device);

  // Chunk gathered_k along the batch dim to bound peak device memory. The gather
  // stage materializes a dense [chunk_b, s_q, gathered_topk, d_qk] bf16 workspace;
  // without a cap it grows linearly with b*s_q*(topk+extra_topk) and can OOM / stall
  // the caching allocator at large batch (e.g. bs=512, extra_topk=8256 -> ~4 GiB).
  // Use a loose cap so typical decode shapes stay a single launch and only
  // pathologically large batch*topk get split across launches.
  constexpr int64_t DECODE_GATHERED_K_MAX_BYTES = 512LL * 1024 * 1024;

  const int num_kv_splits = kSplitK ? F::resolve_sparse_2stage_num_kv_splits(
                                          b,
                                          s_q,
                                          h_q,
                                          B_H,
                                          TileTraits::V_SPLIT,
                                          gathered_topk,
                                          TileTraits::B_TOPK,
                                          MlaSparseDecode2StageXeType::kvMaxSplits)
                                    : 1;

  typename MlaSparseDecode2StageXeType::Fmla::Arguments probe_args{};
  probe_args.gather.b = 1;
  probe_args.gather.s_q = s_q;
  probe_args.gather.gathered_topk = gathered_topk;

  probe_args.dense.kernel.shape.b = 1;
  probe_args.dense.kernel.shape.s_q = s_q;
  probe_args.dense.kernel.shape.h_q = h_q;
  probe_args.dense.scheduler.num_kv_splits = num_kv_splits;
  const int64_t per_batch_gathered_bytes =
      static_cast<int64_t>(MlaSparseDecode2StageXeType::Fmla::get_workspace_size(probe_args));
  int chunk_b = per_batch_gathered_bytes > 0
                    ? static_cast<int>(std::max<int64_t>(1, DECODE_GATHERED_K_MAX_BYTES / per_batch_gathered_bytes))
                    : b;
  chunk_b = std::min(chunk_b, b);

  // Dense gathered-KV + valid-mask HBM workspaces (Stage 1 output, Stage 2 input).
  // Sized for one batch chunk and reused across chunks.
  auto bf16_opts = at::TensorOptions().dtype(at::kBFloat16).device(device);
  auto i32_opts = at::TensorOptions().dtype(at::kInt).device(device);
  at::Tensor gathered_k = at::empty({chunk_b, s_q, gathered_topk, d_qk}, bf16_opts);
  at::Tensor gathered_valid_mask = at::empty({chunk_b, s_q, gathered_topk}, i32_opts);

  // Split-K scratch: per-split unnormalized partial O + the two row-stat arrays. Chunk-
  // sized and reused across chunks like the gathered-KV tile above, and left
  // uninitialized: every (chunk-batch, seq, split, head) entry is written by exactly one
  // Stage-2 work-group before the reduction reads it (same coverage argument as the LSE
  // store, which already writes an uninitialized output). Empty when split-K is off, so
  // the non-split path allocates nothing new.
  auto f32_opts = at::TensorOptions().dtype(at::kFloat).device(device);
  std::optional<at::Tensor> o_accum, split_exp_sums, split_max_logits;
  if constexpr (kSplitK) {
    constexpr int D_V = MlaSparseDecode2StageXeType::D_V;
    o_accum = at::empty({chunk_b, s_q, num_kv_splits, h_q, D_V}, bf16_opts);
    split_exp_sums = at::empty({chunk_b, s_q, num_kv_splits, h_q}, f32_opts);
    split_max_logits = at::empty({chunk_b, s_q, num_kv_splits, h_q}, f32_opts);
  }

  auto args = args_from_options_2stage<MlaSparseDecode2StageXeType>(
      out,
      lse_out,
      q,
      k_cache,
      indices,
      topk_length,
      extra_k_cache,
      extra_indices,
      extra_topk_length,
      attn_sink,
      gathered_k,
      gathered_valid_mask,
      sm_scale,
      head_dim_v,
      o_accum,
      split_exp_sums,
      split_max_logits,
      num_kv_splits);
  auto& params = args.dense;
  auto& gather_args = args.gather;

  // Process the batch in chunks of chunk_b so the gather workspace stays bounded.
  //
  // Both stages run through one Fmla::run call, the same way the dense MLA path runs
  // its split-KV attention + reduction pair through device::MLA::run: the runner holds
  // one Params per stage and issues gather-then-dense on the in-order XPU queue, so the
  // gathered-KV tile is complete before the dense kernel reads it. Its
  // to_underlying_arguments fans out per collective; the opaque workspace blob stays
  // null because Stage 1's workspace (the gathered-KV tile + valid mask, sized above
  // via Fmla::get_workspace_size) is passed through the params pointers instead.
  // Base pointer of batch row b0: data_ptr() + b0 * stride(0), in bytes.
  auto batch_base = [](const at::Tensor& t, int b0) -> void* {
    return static_cast<char*>(t.data_ptr()) + static_cast<int64_t>(b0) * t.stride(0) * t.element_size();
  };

  typename MlaSparseDecode2StageXeType::Fmla fmla;
  for (int b0 = 0; b0 < b; b0 += chunk_b) {
    const int cb = std::min(chunk_b, b - b0);
    params.kernel.shape.b = cb;
    gather_args.b = cb;

    void* q_ptr = batch_base(q, b0);
    params.kernel.q = q_ptr;
    params.mainloop.q = q_ptr;
    params.kernel.out = reinterpret_cast<cutlass::bfloat16_t*>(batch_base(out, b0));
    params.epilogue.lse = reinterpret_cast<float*>(batch_base(lse_out, b0));
    gather_args.indices = reinterpret_cast<int*>(batch_base(indices, b0));
    if (topk_length.has_value()) {
      int* tl = reinterpret_cast<int*>(batch_base(topk_length.value(), b0));
      params.mainloop.topk_length = tl;
      gather_args.topk_length = tl;
    }
    if (has_extra) {
      gather_args.extra_indices = reinterpret_cast<int*>(batch_base(extra_indices.value(), b0));
    }
    if (extra_topk_length.has_value()) {
      int* etl = reinterpret_cast<int*>(batch_base(extra_topk_length.value(), b0));
      params.mainloop.extra_topk_length = etl;
      gather_args.extra_topk_length = etl;
    }

    CUTLASS_CHECK(MlaSparseDecode2StageXeType::Fmla::can_implement(args));
    CUTLASS_CHECK(fmla.run(args, /* workspace */ nullptr));
  }
}

template <typename Element, int D_QK, int B_H, bool HAS_ATTN_SINK>
inline void runMlaSparse2Stage(
    at::Tensor& out,                                     // [B, 1, H, head_dim_v]
    at::Tensor& lse_out,                                 // [B, H, 1] (contiguous [B,1,H])
    const at::Tensor& q,                                 // [B, 1, H, D_qk=512]
    const at::Tensor& k_cache,                           // [num_pages, page_size, 1, 584] fp8 packed
    const at::Tensor& indices,                           // [B, 1, topk]
    const std::optional<at::Tensor>& topk_length,        // [B] or nullopt
    const std::optional<at::Tensor>& extra_k_cache,      // [num_ext_pg, ep, 1, 584] or nullopt
    const std::optional<at::Tensor>& extra_indices,      // [B, 1, extra_topk] or nullopt
    const std::optional<at::Tensor>& extra_topk_length,  // [B] or nullopt
    const std::optional<at::Tensor>& attn_sink,          // [H] or nullopt
    double sm_scale,
    int64_t head_dim_v,
    bool is_fp8_kvcache) {
  TORCH_CHECK(is_fp8_kvcache, "2-stage sparse MLA decode requires the FP8 packed KV cache");
  TORCH_CHECK(q.size(3) == D_QK, "2-stage sparse MLA decode q head dim must match the dispatched D_QK");
  TORCH_CHECK(attn_sink.has_value() == HAS_ATTN_SINK, "attn_sink presence must match the dispatched HAS_ATTN_SINK");

  // Delegate to the Impl, forwarding the compile-time config-keying params
  // (D_QK, B_H, HAS_ATTN_SINK). The Impl resolves the Stage-2 Config from them;
  // no runtime dispatch is needed here.
  runMlaSparse2StageImpl<Element, D_QK, B_H, HAS_ATTN_SINK>(
      out,
      lse_out,
      q,
      k_cache,
      indices,
      topk_length,
      extra_k_cache,
      extra_indices,
      extra_topk_length,
      attn_sink,
      sm_scale,
      head_dim_v);
}
