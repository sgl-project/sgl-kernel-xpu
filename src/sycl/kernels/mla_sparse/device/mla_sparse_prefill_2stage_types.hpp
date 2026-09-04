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
  \brief Two-stage sparse MLA prefill Stage-2 config + host orchestrator for DeepSeek V4.

  Config struct and op-facing run hierarchy for the two-stage sparse MLA *prefill*
  path, layered exactly like the decode side (mla_sparse_decode_2stage_types.hpp):
  the config struct and the run* orchestration live together in this one types header.
  It reuses the decode 2-stage device stack verbatim (collectives, dense kernel, tile
  scheduler, device::MLASparse runner), swapping in only the Stage-1 gather companion
  (SparsePrefillGatherKernel, a dense bf16 copy) via the decode config's GatherKernel
  template parameter.

    - MlaSparsePrefill2StageXe<T, D_QK, HAS_ATTN_SINK, B_H>: alias of the decode config
        MlaSparseDecode2StageXe with SparsePrefillGatherKernel as the Stage-1 companion
        (dense bf16 copy) and a prefill-tuned V-split (sparse_mla_prefill_v_split<B_H>).
        T is the query dtype (ElementQ resolved via SparseMlaToCutlassElementType).
    - args_from_options_prefill_2stage<Config>: adapts the op's tensor arguments to the
        runner's two-stage Arguments ({dense, gather}), applying the query-row -> batch
        mapping (no allocation, no launch).
    - runMlaSparsePrefill2StageImpl<Element, D_QK, B_H, HAS_ATTN_SINK>: resolves the
        Stage-2 Config from the template params, allocates the dense gathered-KV +
        valid-mask HBM workspaces (chunked along the mapped batch = query rows to bound
        peak memory), and runs the launch loop against Config::Fmla.
    - runMlaSparsePrefill2Stage<Element, D_QK, B_H, HAS_ATTN_SINK>: op-facing entry.
        Validates inputs and forwards the dispatched head dim {512,576} + head-block size +
        attn_sink flag to the Impl (which resolves the Config). Its signature matches
        the generated instantiation stub (mla_sparse_prefill_2stage_kernel.cpp.in).

  The prefill problem is mapped onto the decode collectives by treating each query
  row as a decode "batch": shape.b = s_q (query rows), shape.s_q = 1. With that
  mapping the grid, tile scheduler, per-row indices/topk_length indexing, and
  gathered-tile layout all line up, so no collective changes are needed. The
  epilogue additionally emits max_logits (params.max_logits non-null) alongside lse.

  runMlaSparsePrefill2StageImpl / runMlaSparsePrefill2Stage are templated on the
  config-keying params (head dim D_QK + head-block size B_H + attn_sink); the Impl
  resolves the Stage-2 Config from them, so the heavy Config::Fmla instantiation is
  keyed by (D_QK, B_H) exactly the way decode is.
  Each generated launcher launch_mla_sparse_prefill_2stage_<ELEM>_<D_QK>_<B_H>_<HAS_ATTN_SINK>
  (from mla_sparse_prefill_2stage_kernel.cpp.in) instantiates a single (D_QK, B_H, sink)
  variant in its own TU, so the CUTLASS codegen for one variant lands in a separate
  object file (build OOM guard preserved -- one variant per file). The op
  (mla_sparse_prefill.cpp) dispatches dtype, then D_QK, then B_H, then the runtime
  attn_sink flag.

  d_qk is 512 (dense latent) or 576 (nope-512 + rope-64) for prefill; both use dense
  bf16 KV. d_v/output stays 512 (V is the first-512 sub-view of each gathered row);
  only the QK contraction widens for 576. Selected at runtime by params.shape.d_qk.

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
#include <limits>
#include <sycl/sycl.hpp>

// The decode 2-stage types header brings in the shared device stack this prefill
// config reuses verbatim: the MlaSparseDecode2StageXe config struct (which prefill
// aliases below), the collectives, dense kernel wrapper, tile scheduler,
// device::MLASparse runner, plus the shared XPUSparseDecodeAttnFwdParams /
// SparseAttnDecodeParams params, the FLASH_MLA_*_V_SPLIT constants, and the
// DISPATCH_BOOLEAN_FLAG macro. We only add the prefill gather companion here.
#include "sycl/kernels/mla_sparse/device/mla_sparse_decode_2stage_types.hpp"
#include "sycl/kernels/mla_sparse/kernel/xe_mla_sparse_2stage_gather_kernel.hpp"

namespace cutlass::flash_attention::kernel {

// Prefill Stage-2 config: the decode 2-stage config with SparsePrefillGatherKernel
// as the Stage-1 companion (dense bf16 copy instead of packed-fp8 dequant) and a
// prefill-tuned V-split. bf16 query only; D_QK is 512
// (dense latent) or 576 (nope-512 + rope-64). The value / output width stays
// D_V == 512 in both cases (V is the first-512 sub-view of each gathered row); only
// the QK contraction widens for 576.
//
// V-split is chosen by B_H, because B_H flips the epilogue's reduction mode:
//   - B_H <= 16: SubgroupLayoutPV splits the PV *K (topk)* dim -> ReduceK > 1 -> the
//       epilogue uses the SharedStorageReduceK SLM path, whose size scales with
//       D_V_PER_SPLIT (= D_V / V_SPLIT). Shrinking V_SPLIT here *doubles* that SLM,
//       cutting occupancy -- measured ~50% slower at h_q=16. So keep the decode-sized
//       V-split (4) for these.
//   - B_H >= 32: SubgroupLayoutPV splits the M (heads) dim -> ReduceK == 1 -> the
//       epilogue's SharedStorageNonReduceK is empty (no SLM). No occupancy penalty
//       from a wider D_V_PER_SPLIT, so the smaller V-split (2) wins by halving the
//       redundant per-split K re-read + QK recompute -- measured ~35% faster.
// The prefill grid is already saturated by the s_q batch dim (each query row is one
// decode "batch"), so a large V-split mostly re-reads K / recomputes QK for nothing;
// the only thing stopping us from always using 2 is the small-B_H SLM cliff above.
template <int B_H>
inline constexpr int sparse_mla_prefill_v_split =
    (B_H <= 16) ? FLASH_MLA_PREFILL_V_SPLIT : FLASH_MLA_SPARSE_PREFILL_V_SPLIT;

template <typename T, int D_QK, bool HAS_ATTN_SINK, int B_H>
using MlaSparsePrefill2StageXe = MlaSparseDecode2StageXe<
    T,
    D_QK,
    HAS_ATTN_SINK,
    B_H,
    SparsePrefillGatherKernel,
    sparse_mla_prefill_v_split<B_H>,
    /* HAS_MAX_LOGITS */ true>;

}  // namespace cutlass::flash_attention::kernel

// ---------------------------------------------------------------------------
// args_from_options_prefill_2stage: adapts the prefill op's tensor arguments into the
// runner's two-stage Arguments ({dense, gather}), fanning each field into the layer
// that reads it and applying the query-row -> batch-dim mapping. T is the resolved
// config struct, so the return type is its runner's Arguments (same convention as
// args_from_options<T> in the dense MLA path). The prefill op tensors are 3D:
//   q       [s_q, h_q, d_qk=512]      bf16
//   kv      [s_kv, h_kv=1, d_qk=512]  bf16 (dense, unpaged)
//   indices [s_q, h_kv=1, topk]       int32
//   out     [s_q, h_q, d_v=512]       bf16
//   max_logits / lse [s_q, h_q]       fp32
// We view the s_q query rows as the decode "batch" (shape.b = s_q, shape.s_q = 1),
// so the shared collectives/gather see decode-shaped indexing. gathered_k /
// gathered_valid_mask HBM workspaces are allocated by the caller and passed in so
// their strides are recorded here (no allocation / launch here). shape.b is set to
// the full row count; the chunk loop in the Impl re-bases pointers per chunk.
// ---------------------------------------------------------------------------
template <typename T>
inline typename T::Fmla::Arguments args_from_options_prefill_2stage(
    at::Tensor& out,                               // [s_q, h_q, d_v]
    at::Tensor& max_logits,                        // [s_q, h_q]
    at::Tensor& lse,                               // [s_q, h_q]
    const at::Tensor& q,                           // [s_q, h_q, d_qk=512]
    const at::Tensor& kv,                          // [s_kv, h_kv=1, d_qk=512]
    const at::Tensor& indices,                     // [s_q, h_kv=1, topk]
    const std::optional<at::Tensor>& topk_length,  // [s_q] or nullopt
    const std::optional<at::Tensor>& attn_sink,    // [h_q] or nullopt
    const at::Tensor& gathered_k,                  // [chunk_rows, 1, topk, 512] bf16 workspace
    const at::Tensor& gathered_valid_mask,         // [chunk_rows, 1, topk] int workspace
    double sm_scale,
    int64_t head_dim_v) {
  namespace F = cutlass::flash_attention::kernel;

  const int s_q = q.size(0);
  const int h_q = q.size(1);
  const int d_qk = q.size(2);
  const int d_v = static_cast<int>(head_dim_v);

  const int s_kv = kv.size(0);
  const int topk = indices.size(2);

  auto to_int_stride = [](int64_t s) {
    TORCH_CHECK(s <= std::numeric_limits<int>::max(), "Stride exceeds int32 limit: ", s);
    return static_cast<int>(s);
  };

  const float sm_scale_div_log2 = static_cast<float>(sm_scale) * F::LOG_2_E;

  // Problem geometry with the query-row -> batch mapping: each of the s_q rows is
  // one decode "batch" of s_q == 1. No paging / no extra pool for dense bf16 prefill.
  F::SparseDecode2StageProblemShape shape;
  shape.b = s_q;
  shape.s_q = 1;
  shape.h_q = h_q;
  shape.h_kv = 1;
  shape.d_qk = d_qk;
  shape.d_v = d_v;
  shape.s_kv = s_kv;
  shape.num_blocks = 0;
  shape.page_block_size = 0;
  shape.topk = topk;
  shape.gathered_topk = topk;  // no extra pool for prefill
  shape.extra_num_blocks = 0;
  shape.extra_page_block_size = 0;
  shape.extra_topk = 0;

  typename T::Fmla::Arguments args{};
  auto& params = args.dense;

  // Strides. The batch (row) strides are the tensors' row-0 strides; s_q strides are
  // 0 (each mapped batch has s_q == 1, one row).
  const int stride_q_b = to_int_stride(q.stride(0));
  const int stride_q_h_q = to_int_stride(q.stride(1));

  // --- Kernel slice. ---
  auto& k = params.kernel;
  k.shape = shape;
  k.q = q.data_ptr();
  k.stride_q_b = stride_q_b;
  k.stride_q_s_q = 0;
  k.stride_q_h_q = stride_q_h_q;
  k.gathered_k = reinterpret_cast<cutlass::bfloat16_t*>(gathered_k.data_ptr());
  k.stride_gathered_k_b = to_int_stride(gathered_k.stride(0));
  k.stride_gathered_k_s_q = to_int_stride(gathered_k.stride(1));
  k.stride_gathered_k_topk = to_int_stride(gathered_k.stride(2));
  k.out = reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr());
  k.stride_o_b = to_int_stride(out.stride(0));
  k.stride_o_s_q = 0;
  k.stride_o_h_q = to_int_stride(out.stride(1));

  // --- Mainloop slice (bf16 query; q re-read path unused, extra pool empty). ---
  auto& ml = params.mainloop;
  ml.h_q = h_q;
  ml.topk = topk;
  ml.extra_topk = 0;
  ml.gathered_topk = topk;
  ml.sm_scale_div_log2 = sm_scale_div_log2;
  ml.q = q.data_ptr();
  ml.stride_q_b = stride_q_b;
  ml.stride_q_s_q = 0;
  ml.stride_q_h_q = stride_q_h_q;
  ml.q_scale = nullptr;
  ml.q_scale_numel = 0;
  ml.gathered_valid_mask = reinterpret_cast<int*>(gathered_valid_mask.data_ptr());
  ml.stride_gathered_mask_b = to_int_stride(gathered_valid_mask.stride(0));
  ml.stride_gathered_mask_s_q = to_int_stride(gathered_valid_mask.stride(1));
  ml.topk_length = topk_length.has_value() ? reinterpret_cast<int*>(topk_length.value().data_ptr()) : nullptr;
  ml.stride_topk_length_b = topk_length.has_value() ? to_int_stride(topk_length.value().stride(0)) : 0;
  ml.extra_topk_length = nullptr;
  ml.stride_extra_topk_length_b = 0;

  // --- Epilogue slice: prefill also emits max_logits (HAS_MAX_LOGITS=true). lse /
  //     max_logits are [s_q, h_q]; the row is the mapped batch, s_q stride is 0. ---
  auto& ep = params.epilogue;
  ep.h_q = h_q;
  ep.sm_scale_div_log2 = sm_scale_div_log2;
  ep.lse = reinterpret_cast<float*>(lse.data_ptr());
  ep.stride_lse_b = to_int_stride(lse.stride(0));
  ep.stride_lse_s_q = 0;
  ep.attn_sink = attn_sink.has_value() ? static_cast<float*>(attn_sink.value().data_ptr()) : nullptr;
  ep.max_logits = reinterpret_cast<float*>(max_logits.data_ptr());
  ep.stride_max_logits_b = to_int_stride(max_logits.stride(0));
  ep.stride_max_logits_s_q = 0;

  // --- Stage-1 gather params (prefill): dense bf16 unpaged source. Independent of the
  // dense params above; the shared gathered_k / mask / topk_length pointers are copied
  // across so each stage's params is self-contained. ---
  auto& g = args.gather;
  g.b = s_q;
  g.s_q = 1;
  g.topk = topk;
  g.gathered_topk = topk;
  g.indices = reinterpret_cast<int*>(indices.data_ptr());
  g.stride_indices_b = to_int_stride(indices.stride(0));
  g.stride_indices_s_q = 0;
  g.topk_length = ml.topk_length;
  g.stride_topk_length_b = ml.stride_topk_length_b;
  g.gathered_k = k.gathered_k;
  g.stride_gathered_k_b = k.stride_gathered_k_b;
  g.stride_gathered_k_s_q = k.stride_gathered_k_s_q;
  g.stride_gathered_k_topk = k.stride_gathered_k_topk;
  g.gathered_valid_mask = ml.gathered_valid_mask;
  g.stride_gathered_mask_b = ml.stride_gathered_mask_b;
  g.stride_gathered_mask_s_q = ml.stride_gathered_mask_s_q;
  g.s_kv = s_kv;
  g.kv_dense = reinterpret_cast<cutlass::bfloat16_t*>(kv.data_ptr());
  g.stride_kv_dense_s = to_int_stride(kv.stride(0));

  // --- Tile scheduler slice. ---
  params.scheduler.h_q = h_q;
  params.scheduler.s_q = 1;

  return args;
}

template <typename Element, int D_QK, int B_H, bool HAS_ATTN_SINK>
inline void runMlaSparsePrefill2StageImpl(
    at::Tensor& out,
    at::Tensor& max_logits,
    at::Tensor& lse,
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices,
    const std::optional<at::Tensor>& topk_length,
    const std::optional<at::Tensor>& attn_sink,
    double sm_scale,
    int64_t head_dim_v) {
  namespace F = cutlass::flash_attention::kernel;

  // The Stage-2 config for this D_QK + B_H + attn_sink flag. All three are compile-time
  // template params dispatched by the op (mla_sparse_prefill.cpp), so no runtime dispatch
  // is needed here. The 2-stage codegen only emits bf16 (MlaSparsePrefillXe20.cmake), so
  // Element is always bf16 and no per-dtype guard is required.
  using MlaSparsePrefill2StageXeType = F::MlaSparsePrefill2StageXe<Element, D_QK, HAS_ATTN_SINK, B_H>;

  const int s_q = q.size(0);
  const int d_qk = q.size(2);
  const int topk = indices.size(2);

  auto device = q.device();
  const c10::DeviceGuard device_guard(device);

  // Chunk gathered_k along the mapped batch (query rows) to bound peak device
  // memory. The gather materializes a dense [chunk_rows, 1, topk, d_qk] bf16
  // workspace; without a cap it grows linearly with s_q*topk and can OOM at long
  // prefill (e.g. s_q=128, topk=512 -> ~64 MiB, fine; large s_q*topk gets split).
  constexpr int64_t PREFILL_GATHERED_K_MAX_BYTES = 256LL * 1024 * 1024;
  // Per-row workspace footprint comes from the runner (which sums its stages'
  // get_workspace_size) rather than being recomputed here: the gathered-KV tile +
  // valid mask are Stage 1's outputs, so their layout is the gather kernel's
  // business. Probe one mapped batch (b == 1, s_q == 1 -- each query row is one
  // decode "batch") with the real topk to get the per-row bytes the cap above is
  // divided by. Same shape as the decode side's probe.
  typename MlaSparsePrefill2StageXeType::Fmla::Arguments probe_args{};
  probe_args.gather.b = 1;
  probe_args.gather.s_q = 1;
  probe_args.gather.gathered_topk = topk;
  const int64_t per_row_gathered_bytes =
      static_cast<int64_t>(MlaSparsePrefill2StageXeType::Fmla::get_workspace_size(probe_args));
  int chunk_rows = per_row_gathered_bytes > 0
                       ? static_cast<int>(std::max<int64_t>(1, PREFILL_GATHERED_K_MAX_BYTES / per_row_gathered_bytes))
                       : s_q;
  chunk_rows = std::min(chunk_rows, s_q);

  auto bf16_opts = at::TensorOptions().dtype(at::kBFloat16).device(device);
  auto i32_opts = at::TensorOptions().dtype(at::kInt).device(device);
  at::Tensor gathered_k = at::empty({chunk_rows, 1, topk, d_qk}, bf16_opts);
  at::Tensor gathered_valid_mask = at::empty({chunk_rows, 1, topk}, i32_opts);

  auto args = args_from_options_prefill_2stage<MlaSparsePrefill2StageXeType>(
      out,
      max_logits,
      lse,
      q,
      kv,
      indices,
      topk_length,
      attn_sink,
      gathered_k,
      gathered_valid_mask,
      sm_scale,
      head_dim_v);
  auto& params = args.dense;
  auto& gather_args = args.gather;

  // Config-level invariants (formerly checked in the deleted launch_..._policy):
  // the resolved config fixes D_QK / D_V, and the gathered tile width must equal
  // topk + extra_topk (extra_topk == 0 for prefill).
  TORCH_CHECK(
      params.kernel.shape.d_qk == MlaSparsePrefill2StageXeType::D_QK, "Invalid d_qk for this kernel instantiation");
  TORCH_CHECK(
      params.kernel.shape.d_v == MlaSparsePrefill2StageXeType::D_V, "d_v must match MlaSparseDecode2StageXe::D_V");
  TORCH_CHECK(
      params.kernel.shape.gathered_topk == params.kernel.shape.topk + params.kernel.shape.extra_topk,
      "gathered_topk must equal topk + extra_topk");

  // Process the query rows in chunks of chunk_rows so the gather workspace stays
  // bounded. Per chunk we re-base the row-indexed input/output pointers and reuse the
  // same gathered_k/gathered_valid_mask workspace.
  //
  // Both stages run through one Fmla::run call: the runner holds one Params per stage
  // and issues gather-then-dense on the in-order XPU queue (Stage 1 fills gathered_k /
  // gathered_valid_mask, Stage 2 consumes them), the same way device::MLA runs the
  // dense path's split-KV attention + reduction pair. to_underlying_arguments fans the
  // dense arguments out per collective; the opaque workspace blob stays null because
  // Stage 1's workspace (the gathered-KV tile + valid mask, sized above via
  // Fmla::get_workspace_size) is passed through the params pointers instead.
  //
  // Per row-chunk we re-base the row-indexed slices: the mapped batch count feeds
  // kernel.shape.b + gather_args.b; q feeds kernel + mainloop; out -> kernel; lse /
  // max_logits -> epilogue; indices -> gather; topk_length -> mainloop + gather.
  auto row_base = [](const at::Tensor& t, int r0) -> void* {
    return static_cast<char*>(t.data_ptr()) + static_cast<int64_t>(r0) * t.stride(0) * t.element_size();
  };

  typename MlaSparsePrefill2StageXeType::Fmla fmla;
  for (int r0 = 0; r0 < s_q; r0 += chunk_rows) {
    const int cr = std::min(chunk_rows, s_q - r0);
    params.kernel.shape.b = cr;
    gather_args.b = cr;

    void* q_ptr = row_base(q, r0);
    params.kernel.q = q_ptr;
    params.mainloop.q = q_ptr;
    params.kernel.out = reinterpret_cast<cutlass::bfloat16_t*>(row_base(out, r0));
    params.epilogue.lse = reinterpret_cast<float*>(row_base(lse, r0));
    params.epilogue.max_logits = reinterpret_cast<float*>(row_base(max_logits, r0));
    gather_args.indices = reinterpret_cast<int*>(row_base(indices, r0));
    if (topk_length.has_value()) {
      int* tl = reinterpret_cast<int*>(row_base(topk_length.value(), r0));
      params.mainloop.topk_length = tl;
      gather_args.topk_length = tl;
    }

    CUTLASS_CHECK(MlaSparsePrefill2StageXeType::Fmla::can_implement(args));
    CUTLASS_CHECK(fmla.run(args, /* workspace */ nullptr));
  }
}

template <typename Element, int D_QK, int B_H, bool HAS_ATTN_SINK>
inline void runMlaSparsePrefill2Stage(
    at::Tensor& out,                               // [s_q, h_q, d_v]
    at::Tensor& max_logits,                        // [s_q, h_q]
    at::Tensor& lse,                               // [s_q, h_q]
    const at::Tensor& q,                           // [s_q, h_q, d_qk=512]
    const at::Tensor& kv,                          // [s_kv, h_kv=1, d_qk=512]
    const at::Tensor& indices,                     // [s_q, h_kv=1, topk]
    const std::optional<at::Tensor>& attn_sink,    // [h_q] or nullopt
    const std::optional<at::Tensor>& topk_length,  // [s_q] or nullopt
    double sm_scale,
    int64_t head_dim_v) {
  TORCH_CHECK(head_dim_v == 512, "head_dim_v must be 512 for DeepSeek V4 MLA prefill");
  TORCH_CHECK(
      (std::is_same<Element, sycl::ext::oneapi::bfloat16>::value),
      "2-stage sparse MLA prefill currently supports only bf16 query");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16, "2-stage sparse MLA prefill query must be bfloat16");
  TORCH_CHECK(kv.scalar_type() == at::kBFloat16, "2-stage sparse MLA prefill kv must be bfloat16");
  TORCH_CHECK(q.dim() == 3, "q must be [s_q, h_q, d_qk]");
  TORCH_CHECK(kv.dim() == 3, "kv must be [s_kv, h_kv, d_qk]");
  TORCH_CHECK(indices.dim() == 3, "indices must be [s_q, h_kv, topk]");
  // d_qk is 512 (dense latent) or 576 (nope-512 + rope-64). d_v/output stays 512 in
  // both (V is the first-512 sub-view of each gathered row); only the QK contraction
  // widens for 576. q and kv must agree on d_qk since kv is the gathered K source.
  // D_QK is a compile-time template param dispatched by the op; the query head dim must
  // match it.
  TORCH_CHECK(
      q.size(2) == D_QK, "2-stage sparse MLA prefill q head dim must match the dispatched D_QK, got ", q.size(2));
  TORCH_CHECK(
      kv.size(2) == q.size(2),
      "2-stage sparse MLA prefill requires kv head dim to match q head dim, got kv ",
      kv.size(2),
      " vs q ",
      q.size(2));
  TORCH_CHECK(kv.size(1) == 1, "2-stage sparse MLA prefill requires h_kv == 1");
  TORCH_CHECK(attn_sink.has_value() == HAS_ATTN_SINK, "attn_sink presence must match the dispatched HAS_ATTN_SINK");

  // Delegate to the Impl, forwarding the compile-time config-keying params
  // (D_QK, B_H, HAS_ATTN_SINK). The Impl resolves the Stage-2 Config from them;
  // no runtime dispatch is needed here.
  runMlaSparsePrefill2StageImpl<Element, D_QK, B_H, HAS_ATTN_SINK>(
      out, max_logits, lse, q, kv, indices, topk_length, attn_sink, sm_scale, head_dim_v);
}
