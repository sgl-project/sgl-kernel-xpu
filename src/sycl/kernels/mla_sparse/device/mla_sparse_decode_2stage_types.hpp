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

    - MlaSparseDecode2StageXe<D_QK, IS_FP8_QUERY, HAS_ATTN_SINK, B_H, GatherKernel, V_SPLIT>:
        the DPAS/tile config struct that assembles the collectives + tile scheduler +
        dense kernel wrapper + device::MLASparse runner (analog of MlaSparseXe).
    - args_from_options_2stage<ElementSycl>: adapts our tensor arguments to
        XPUSparseDecodeAttnFwdParams (no allocation, no launch).
    - runMlaSparse2StageImpl<ElementSycl, Config>: allocates the dense gathered-KV +
        valid-mask HBM workspaces (batch-chunked to bound peak memory) and runs the
        launch loop against Config::Fmla.
    - runMlaSparse2Stage<ElementSycl, D_QK, B_H, HAS_ATTN_SINK>: op-facing entry.
        Validates inputs, resolves the Config for the dispatched head dim + head-block
        size + attn_sink flag, and delegates to the Impl. Its signature matches the
        generated instantiation stub (mla_sparse_decode_2stage_kernel.cpp.in).

  runMlaSparse2StageImpl / runMlaSparse2Stage are templated on the resolved Stage-2
  Config (head dim D_QK + head-block size B_H + attn_sink), so the heavy Config::Fmla
  instantiation is keyed by (D_QK, B_H) exactly the way the fused decode path is keyed
  by page size. Each generated launcher
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
#include <limits>
#include <sycl/sycl.hpp>

#include "../../../Utils.h"  // CUTLASS_CHECK (used by mla_sparse_runner.hpp)
#include "cutlass/bfloat16.h"
#include "cutlass/float8.h"
#include "cutlass/kernel_hardware_info.hpp"
// The collective headers pull in the full cute/cutlass sycl-tla stack (defining
// cute::intel, etc.) that mla_sparse_runner.hpp -> comm/common.h references. They
// sort under collective/ (before device/), so the runner always sees cute::intel
// even after include re-alphabetization. (Matches the fused path, which likewise
// includes its collectives before the runner.)
#include "sycl/kernels/mla_sparse/collective/xe_mla_sparse_decode_2stage_epilogue.hpp"
#include "sycl/kernels/mla_sparse/collective/xe_mla_sparse_decode_2stage_mainloop.hpp"
#include "sycl/kernels/mla_sparse/device/mla_sparse_runner.hpp"
// Pulls in the whole device side of the 2-stage path: the dense kernel transitively
// includes the common prologue (XPUSparseDecodeAttnFwdParams / LOG_2_E / the V-split
// knobs / DISPATCH_BOOLEAN_FLAG), the Stage-1 gather kernel (its nested GatherKernel),
// the tile scheduler, and both collectives.
#include "sycl/kernels/mla_sparse/kernel/xe_mla_sparse_decode_2stage_dense_kernel.hpp"

namespace cutlass::flash_attention::kernel {

//----------------- Stage-2 dense-decode Xe configuration --------------------//
// Assembles the DPAS/tile config (the former KernelTraits body), the three
// collectives + kernel wrapper, and the device::MLASparse runner into one config
// struct, mirroring the fused path's MlaSparseXe. The struct is passed to the
// collectives as their `Traits` (they read ElementQ / TiledMMAQK / B_H / ... off
// it), so the collective/kernel aliases are lazy and only instantiate at the
// launcher use-site where the struct is already complete.
//
// The kernel keeps flat SparseAttnDecodeParams as Arguments == Params, so the
// device::MLASparse Arguments->Params flow is identity and the host adapter
// (args_from_options_2stage) is unchanged. GrfSize is 256: this dense-decode
// kernel is fragment-heavy and would spill at the runner's default 128.
// GatherKernelTmpl_ selects the Stage-1 companion (keyed on D_QK). Defaults to the
// decode gather (packed fp8 -> bf16 dequant); the prefill path reuses this exact
// config struct but supplies SparsePrefillGatherKernel (dense bf16 copy). All other
// template args and the whole DPAS/tile/collective assembly are shared verbatim.
template <
    int D_QK_,
    bool IS_FP8_QUERY_,
    bool HAS_ATTN_SINK_,
    int B_H_,
    template <int> class GatherKernelTmpl_ = SparseDecodeGatherDequantKernel,
    int V_SPLIT_ = FLASH_MLA_PREFILL_V_SPLIT,
    bool HAS_MAX_LOGITS_ = false>
struct MlaSparseDecode2StageXe {
  static constexpr int D_QK = D_QK_;
  static constexpr bool IS_FP8_QUERY = IS_FP8_QUERY_;
  static constexpr bool HAS_ATTN_SINK = HAS_ATTN_SINK_;
  // Prefill returns the pre-sink row max (max_logits) alongside lse; decode does not.
  // Threaded into the epilogue so the extra store is compiled out for decode.
  static constexpr bool HAS_MAX_LOGITS = HAS_MAX_LOGITS_;

  using ElementQ = cutlass::bfloat16_t;
  using ElementKV = cutlass::bfloat16_t;
  using ElementO = cutlass::bfloat16_t;

  using StrideQ = cute::tuple<int, _1, int>;
  using StrideKV = cute::tuple<int, _1, int>;
  using StrideO = cute::tuple<int, _1, int>;

  static constexpr int B_H = B_H_;  // h_q block size
  static constexpr int SUBGROUP_SIZE = intel::sg_size;
  static constexpr int NUM_SUBGROUPS = B_H > 16 ? (B_H > 32 ? 8 : 4) : 4;
  static constexpr int NUM_THREADS = NUM_SUBGROUPS * SUBGROUP_SIZE;
  static constexpr int B_TOPK = 64;  // topk_length block size

  // static constexpr int D_QK = 576;
  static constexpr int D_PE = 64;
  static constexpr int D_V = 512;
  // V-split factor: how many work-groups split the D_V output for one query tile.
  // Defaults (via the template param) to the decode knob; the prefill config passes
  // a smaller value since its grid is already saturated by the s_q batch dim.
  static constexpr int V_SPLIT = V_SPLIT_;
  static_assert(V_SPLIT >= 1, "V_SPLIT must be >= 1");
  static_assert(D_V % V_SPLIT == 0, "D_V must be divisible by V_SPLIT");
  static constexpr int D_V_PER_SPLIT = D_V / V_SPLIT;
  static constexpr int HEAD_DIM_TILE_SIZE = 32;

  static constexpr int stages = 64 / B_TOPK;
  static_assert(stages == 1, "only support single stage for now");

  // 576 / 32 = 18
  // Q head packing size = B_H
  using TileShapeQK = Shape<Int<B_H>, Int<B_TOPK>, Int<HEAD_DIM_TILE_SIZE>>;
  using SubgroupLayoutQK =
      conditional_t<(B_H > 16), Layout<Shape<Int<NUM_SUBGROUPS>, _1, _1>>, Layout<Shape<_1, Int<NUM_SUBGROUPS>, _1>>>;

  using TileShapePV = Shape<Int<B_H>, Int<HEAD_DIM_TILE_SIZE>, Int<B_TOPK>>;
  using SubgroupLayoutPV =
      conditional_t<(B_H > 16), Layout<Shape<Int<NUM_SUBGROUPS>, _1, _1>>, Layout<Shape<_1, _1, Int<NUM_SUBGROUPS>>>>;

  // D_V / 64 = 8 tiles for v_dim
  using TileShapeOut = Shape<Int<B_H>, Int<D_V_PER_SPLIT>>;

  using SmemTileLayoutK = Layout<Shape<Int<B_TOPK>, Int<HEAD_DIM_TILE_SIZE>>, Stride<Int<HEAD_DIM_TILE_SIZE>, _1>>;
  using SmemTileLayoutV = Layout<Shape<Int<HEAD_DIM_TILE_SIZE>, Int<B_TOPK>>, Stride<Int<B_TOPK>, _1>>;

  constexpr static int SGTileQ = get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})))();
  // bf16 dpas m8n16k16
  // (8, 128, 64) / ((8, 16, 16) * (1, 16, 1)) = (1, 1, 4) iterations per subgroup
  constexpr static int MAX_M_DPAS = 8;
  using MMAOperation = XE_DPAS_TT<cute::gcd(SGTileQ, MAX_M_DPAS), float, bfloat16_t>;
  using TiledMMAQK = typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<TileShapeQK>, SubgroupLayoutQK>::TiledMMA;
  using TiledMMAPV = typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<TileShapePV>, SubgroupLayoutPV>::TiledMMA;

  // Collective mainloop / epilogue + tile scheduler + kernel wrapper. The struct
  // passes itself as the collectives' Traits (aliases are lazy — no self-instantiation
  // until the launcher uses DenseKernel below).
  using CollectiveMainloop = cutlass::flash_attention::collective::
      XeMlaSparseDecode2StageMainloop<D_QK, IS_FP8_QUERY, MlaSparseDecode2StageXe>;
  using CollectiveEpilogue = cutlass::flash_attention::collective::
      XeMlaSparseDecode2StageEpilogue<CollectiveMainloop, HAS_ATTN_SINK, HAS_MAX_LOGITS>;
  using TileScheduler = cutlass::flash_attention::kernel::XeMlaSparseDecode2StageIndividualTileScheduler<B_H>;

  using DenseKernel = cutlass::flash_attention::kernel::
      DenseDecodeFwdKernel<CollectiveMainloop, CollectiveEpilogue, TileScheduler, GatherKernelTmpl_>;

  // GrfSize 256 (see note above); XE3P's 512-GRF mode from the prior manual launch
  // is capped to 256 by the shared launch<> helper's {128,256} constraint.
  using Fmla = cutlass::flash_attention::device::MLASparse<DenseKernel, 256>;
};

}  // namespace cutlass::flash_attention::kernel

// ---------------------------------------------------------------------------
// args_from_options_2stage: adapts our op's tensor arguments to the composite
// DecodeSparseAttn2StageParams (the device-launch layer's argument struct), fanning
// each field out into the layer that reads it (kernel / mainloop / epilogue /
// gather / scheduler).
//
// Structural analog of args_from_options_sparse() in the fused
// mla_sparse_decode_types.hpp: it derives problem shapes/strides from the
// tensors and populates the launcher argument struct, but performs no
// allocation and no launch. The gather workspaces (gathered_k /
// gathered_valid_mask) are allocated by the caller (runMlaSparse2StageImpl) and
// passed in so their strides can be recorded here. The batch (shape.b + the per-slice
// b copies) is set to the full batch; the chunk loop in the Impl re-bases the batched
// pointers per chunk via rebase_decode_chunk().
//
// ElementSycl is the query element type. The PR decode kernels are bf16-query
// (KernelTraits::ElementQ = cutlass::bfloat16_t); the FP8-query path
// (is_fp8_query) is not wired through this op yet, so bf16 query is required
// (checked in runMlaSparse2Stage). The gathered-KV / out casts are fixed to
// cutlass::bfloat16_t by the params field types.
// ---------------------------------------------------------------------------
template <typename ElementSycl>
inline cutlass::flash_attention::kernel::DecodeSparseAttn2StageParams args_from_options_2stage(
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
    int64_t head_dim_v) {
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

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = q.device().index();
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

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

  F::DecodeSparseAttn2StageParams params;

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

  // --- Gather slice (decode child): dual packed-fp8 paged pools. ---
  auto& g = params.gather;
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

  (void)hw_info;
  return params;
}

// ---------------------------------------------------------------------------
// runMlaSparse2StageImpl: allocates the dense gathered-KV + valid-mask HBM
// workspaces, builds the launcher arguments, and runs the batch-chunked launch
// loop against Config::Fmla. Structural analog of runMlaSparseImpl() in the fused
// path (which allocates the CUTLASS workspace, calls args_from_options_sparse(),
// and runs the device op) — here the "device op" is Config::Fmla, the
// device::MLASparse runner assembled by the resolved Stage-2 config struct.
//
// Config is a fully-resolved MlaSparseDecode2StageXe<D_QK, IS_FP8_QUERY,
// HAS_ATTN_SINK, B_H>; instantiating this function pulls in the heavy CUTLASS
// kernel for that single B_H, so the generated per-(dtype,B_H) TU keeps its codegen
// confined to one head-block (build OOM guard). ElementSycl is the query element
// type used only for the host-side arg adaptation (the kernel is bf16-internal).
// ---------------------------------------------------------------------------
template <typename ElementSycl, typename Config>
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

  const int b = q.size(0);
  const int s_q = q.size(1);
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
  const int64_t per_batch_gathered_bytes = static_cast<int64_t>(s_q) * gathered_topk * d_qk * 2;  // bf16 = 2 bytes
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

  auto params = args_from_options_2stage<ElementSycl>(
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
      head_dim_v);

  // Config-level invariants (formerly checked in the deleted launch_..._policy):
  // the resolved config fixes D_QK / D_V, and the gathered tile width must equal
  // topk + extra_topk.
  TORCH_CHECK(params.kernel.shape.d_qk == Config::D_QK, "Invalid d_qk for this kernel instantiation");
  TORCH_CHECK(params.kernel.shape.d_v == Config::D_V, "d_v must match MlaSparseDecode2StageXe::D_V");
  TORCH_CHECK(
      params.kernel.shape.gathered_topk == params.kernel.shape.topk + params.kernel.shape.extra_topk,
      "gathered_topk must equal topk + extra_topk");

  // Process the batch in chunks of chunk_b so the gather workspace stays bounded.
  // Per chunk we re-base the batched input/output pointers (slicing preserves
  // strides) into every slice that reads them and reuse the same
  // gathered_k/gathered_valid_mask workspace, whose batch stride was sized for
  // chunk_b. Re-basing is centralized here so no slice's copy of a batched pointer
  // is ever missed (q feeds kernel + mainloop; topk_length / extra_topk_length feed
  // mainloop + gather; the batch count feeds kernel.shape + gather).
  //
  // Both stages run through the device::MLASparse runner: the dense kernel declares
  // the Stage-1 gather+dequant kernel as its nested GatherKernel, so the runner
  // launches gather-then-dense on the in-order XPU queue, sharing one composite
  // Params object. Mirrors the fused path's runMlaSparseImpl (build arguments,
  // can_implement, run); to_underlying_arguments fans out per collective and the
  // workspace is empty (get_workspace_size == 0 -> nullptr).
  typename Config::Fmla fmla;
  for (int b0 = 0; b0 < b; b0 += chunk_b) {
    const int cb = std::min(chunk_b, b - b0);
    params.kernel.shape.b = cb;
    params.gather.b = cb;

    void* q_ptr = q.slice(0, b0, b0 + cb).data_ptr();
    params.kernel.q = q_ptr;
    params.mainloop.q = q_ptr;
    params.kernel.out = reinterpret_cast<cutlass::bfloat16_t*>(out.slice(0, b0, b0 + cb).data_ptr());
    params.epilogue.lse = reinterpret_cast<float*>(lse_out.slice(0, b0, b0 + cb).data_ptr());
    params.gather.indices = reinterpret_cast<int*>(indices.slice(0, b0, b0 + cb).data_ptr());
    if (topk_length.has_value()) {
      int* tl = reinterpret_cast<int*>(topk_length.value().slice(0, b0, b0 + cb).data_ptr());
      params.mainloop.topk_length = tl;
      params.gather.topk_length = tl;
    }
    if (has_extra) {
      params.gather.extra_indices = reinterpret_cast<int*>(extra_indices.value().slice(0, b0, b0 + cb).data_ptr());
    }
    if (extra_topk_length.has_value()) {
      int* etl = reinterpret_cast<int*>(extra_topk_length.value().slice(0, b0, b0 + cb).data_ptr());
      params.mainloop.extra_topk_length = etl;
      params.gather.extra_topk_length = etl;
    }

    CUTLASS_CHECK(Config::Fmla::can_implement(params));
    CUTLASS_CHECK(fmla.run(params, /* workspace */ nullptr));
  }
}

// ---------------------------------------------------------------------------
// runMlaSparse2Stage: op-facing entry point. Structural analog of runMlaSparse()
// in the fused path — it validates the inputs, resolves the Stage-2 Config for the
// dispatched head dim (D_QK), head-block size (B_H), and attn_sink flag
// (HAS_ATTN_SINK), and delegates to the Impl. The op (mla_sparse_decode.cpp) picks
// ElementSycl (dtype), D_QK, B_H, and the attn_sink flag, mirroring the fused path's
// dtype-then-page-size dispatch, and each generated launcher
// launch_mla_sparse_decode_2stage_<ELEM>_<D_QK>_<B_H>_<SINK> instantiates a single
// (D_QK, B_H, HAS_ATTN_SINK) in its own TU (build OOM guard preserved -- one variant
// per file). Decode is always D_QK == 512 (the packed FP8 latent); the dimension is
// still threaded so the codegen shape matches prefill.
//
// The Stage-2 kernel is bf16-internal and IS_FP8_QUERY is false. ElementSycl is the
// query element type; only bf16 query is supported (see the note on
// args_from_options_2stage). The half instantiation compiles but is rejected here at
// runtime.
// ---------------------------------------------------------------------------
template <typename ElementSycl, int D_QK, int B_H, bool HAS_ATTN_SINK>
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
  namespace F = cutlass::flash_attention::kernel;

  TORCH_CHECK(is_fp8_kvcache, "2-stage sparse MLA decode requires the FP8 packed KV cache");
  TORCH_CHECK(head_dim_v == 512, "head_dim_v must be 512 for DeepSeek V4 MLA");
  TORCH_CHECK(
      (std::is_same<ElementSycl, sycl::ext::oneapi::bfloat16>::value),
      "2-stage sparse MLA decode currently supports only bf16 query");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16, "2-stage sparse MLA decode query must be bfloat16");
  TORCH_CHECK(q.size(3) == D_QK, "2-stage sparse MLA decode q head dim must match the dispatched D_QK");
  TORCH_CHECK(attn_sink.has_value() == HAS_ATTN_SINK, "attn_sink presence must match the dispatched HAS_ATTN_SINK");

  // The Stage-2 kernel is bf16-internal; only the bf16-query instantiation carries the
  // heavy Config::Fmla. Guarding the Impl (and thus the CUTLASS instantiation) behind
  // if constexpr keeps the generated half TUs cheap -- half query is rejected by the
  // TORCH_CHECK above at runtime, so its Impl is dead code. (Previously this fell out of
  // the element-agnostic policy TUs; here it must be explicit since the TU is per-dtype.)
  if constexpr (std::is_same_v<ElementSycl, sycl::ext::oneapi::bfloat16>) {
    // The Stage-2 config for this D_QK + B_H + attn_sink flag. All three are compile-time
    // template params dispatched by the op (mla_sparse_decode.cpp), so no runtime
    // dispatch is needed here.
    using Config = F::MlaSparseDecode2StageXe<D_QK, /* IS_FP8_QUERY */ false, HAS_ATTN_SINK, B_H>;
    runMlaSparse2StageImpl<ElementSycl, Config>(
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
}
