/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
/*!
  \file
  \brief Two-stage sparse MLA shared device declarations for DeepSeek V4.

  Shared by BOTH two-stage paths (decode and prefill): the Stage-2 dense kernel,
  its collectives, and its tile geometry are path-agnostic, and the Stage-1 gather
  params keep their common base here with one child per path.

  Contains:
    - LOG_2_E / LOG_E_2 log-base constants + packed FP8 KV layout constants.
    - SparseDecode2StageProblemShape: pure problem geometry.
    - The per-layer param blocks (Kernel2StageParams / Mainloop2StageParams /
      Epilogue2StageParams / TileScheduler2StageParams) bundled into the Stage-2
      dense SparseAttn2StageParams, plus the independent Stage-1 Gather2StageParams
      and its decode / prefill children.
    - DISPATCH_BOOLEAN_FLAG: compile-time boolean dispatch.
    - FLASH_MLA_PREFILL_V_SPLIT: dense-decode V-split knob.
    - MlaSparseDecode2StageTileTraits: the Stage-2 DPAS / tile geometry (element
      types, MMA atoms, tile shapes, subgroup layouts, sizes) that the collectives
      and the dense kernel wrapper receive as their `Traits`. The *assembly* around
      it (which collectives / gather kernel / runner) is MlaSparseDecode2StageXe in
      device/mla_sparse_decode_2stage_types.hpp.

  reference: tests/test_flash_mla_with_kvcache.py
    _gather_and_dequant (Stage 1) + _sm120_sparse_decode_fwd (Stage 2).
*/

#pragma once

#ifndef SYCL_INTEL_TARGET
#define SYCL_INTEL_TARGET 20
#endif

#include <cstdint>
#include <cute/algorithm/subgroup_algorithms.hpp>
#include <cute/atom/copy_traits_xe_2d.hpp>
#include <cute/tensor.hpp>
#include <cute/util/compat/device.hpp>
#include <cute/util/compat/dims.hpp>
#include <cute/util/compat/launch_policy.hpp>
#include <limits>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include "cutlass/bfloat16.h"
#include "cutlass/device_kernel.h"
#include "cutlass/float8.h"
#include "cutlass/half.h"

// rmem<->smem block copies (copy_block_r2s / copy_block_s2r, in namespace cute) used
// by the dense kernel's cross-subgroup softmax reduction (only reached when V_SPLIT
// produces ReduceK > 1). Shared with the rest of the repo.
#include "sycl/comm/copy_block_slm.hpp"

using namespace cute;

namespace cutlass::flash_attention::kernel {

// ---------------------------------------------------------------------------
// Query element mapping (sycl -> cutlass). Local copy of the fused path's
// SparseMlaToCutlassElementType (device/mla_sparse_decode_types.hpp), kept here so
// the 2-stage config can resolve its ElementQ straight from the dispatched dtype
// without pulling in the heavy fused kernel header. Same specializations.
// ---------------------------------------------------------------------------
template <typename T>
struct SparseMlaToCutlassElementType {
  using type = T;
};

template <>
struct SparseMlaToCutlassElementType<sycl::half> {
  using type = cutlass::half_t;
};

template <>
struct SparseMlaToCutlassElementType<sycl::ext::oneapi::bfloat16> {
  using type = cutlass::bfloat16_t;
};

// ---------------------------------------------------------------------------
// log-base constants + packed FP8 KV layout.
// ---------------------------------------------------------------------------
static constexpr float LOG_2_E = 1.4426950408889634f;
static constexpr float LOG_E_2 = 0.6931471805599453f;

// specific for DeepSeek V4 packed fp8 sparse MLA decode KV cache layout.
static constexpr int SPARSE_MLA_FP8_NOPE_BYTES = 448;
static constexpr int SPARSE_MLA_FP8_ROPE_DIM = 64;
static constexpr int SPARSE_MLA_FP8_DATA_BYTES_PER_TOKEN = 576;
static constexpr int SPARSE_MLA_FP8_SCALE_BYTES_PER_TOKEN = 8;
static constexpr int SPARSE_MLA_FP8_HEAD_BYTES = 584;

// ---------------------------------------------------------------------------
// Problem shape for the two-stage sparse MLA decode. Structural analog of the
// fused path's FSparseMlAProblemShape (device/mla_sparse_decode_types.hpp): the
// pure problem geometry (batch/heads/dims/topk/paging), separated from the data
// pointers and strides so it can be reasoned about on its own. The host adapter
// builds one of these, then distributes the dims it needs into the per-layer
// param blocks below (each layer carries only the individual shape scalars it
// reads).
// ---------------------------------------------------------------------------
struct SparseDecode2StageProblemShape {
  int b = 0;                      // batch (prefill: mapped from query rows s_q)
  int s_q = 0;                    // query seqlen (1 for decode; 1 per mapped row for prefill)
  int h_q = 0;                    // number of query heads
  int h_kv = 0;                   // number of KV heads (1 for MLA)
  int d_qk = 0;                   // QK head dim (512 = 448 nope + 64 rope; prefill uses dense 512)
  int d_v = 0;                    // V head dim (512)
  int num_blocks = 0;             // primary KV cache pages
  int page_block_size = 0;        // primary KV cache page size
  int topk = 0;                   // primary sparse top-k
  int gathered_topk = 0;          // topk + extra_topk (dense gathered tile width)
  int extra_num_blocks = 0;       // extra KV cache pages
  int extra_page_block_size = 0;  // extra KV cache page size
  int extra_topk = 0;             // extra pool sparse top-k
  int s_kv = 0;                   // dense KV seqlen (prefill only; decode leaves 0)

  SparseDecode2StageProblemShape() = default;
};

// ===========================================================================
// Per-layer parameter blocks for the two-stage sparse MLA path.
//
// The former monolithic SparseAttnDecodeParams is decomposed into one block per
// consuming layer, mirroring the fused MLA kernel's Params fan-out
// (kernel/xe_mla_sparse_kernel.hpp: KernelParams / MainloopParams /
// EpilogueParams / TileSchedulerParams). Each block carries ONLY the scalars,
// pointers, and strides its own layer actually reads, so the coupling between a
// layer and the fields it touches is explicit. SparseAttn2StageParams below
// assembles the Stage-2 blocks into the dense kernel's Params; the Stage-1 gather
// blocks stay separate and are the gather kernel's own Params (the runner holds one
// Params per stage and shares only the gathered-KV HBM buffers between them).
//
// Unused monolith fields (plain sm_scale, is_fp8_query, h_kv, the SplitKV block)
// are intentionally dropped: they were host-set but never read on device.
// ===========================================================================

// Tile scheduler: decodes the launch grid into (batch, seq, head-block, v-split).
// Reads only the two dims needed to enumerate head-blocks per query tile.
struct TileScheduler2StageParams {
  int h_q = 0;
  int s_q = 0;
  // Split-K factor over the gathered topk dim, mapped onto grid.z. 1 disables split-K
  // (grid.z == 1, kv_split_idx == 0) and is the non-split path unchanged. Runtime
  // rather than compile-time because the useful factor depends on gathered_topk, which
  // is only known per call -- same reason the paged path carries num_kv_splits in its
  // scheduler params (kernel/mla_tile_scheduler.hpp:52).
  int num_kv_splits = 1;
};

// Stage-2 dense kernel wrapper: builds the per-tile Q / O / gathered-K/V gmem
// views and the launch grid. Owns the query, output, and gathered-KV tensors.
// Carries the whole problem shape (like the fused path's KernelParams, which holds
// a ProblemShape member); the kernel reads b / s_q / h_q / gathered_topk off it.
struct Kernel2StageParams {
  SparseDecode2StageProblemShape shape;

  void* __restrict__ q = nullptr;  // [b, s_q, h_q, d_qk], bf16 or fp8_e4m3
  int stride_q_b = 0, stride_q_s_q = 0, stride_q_h_q = 0;

  cutlass::bfloat16_t* __restrict__ gathered_k = nullptr;  // [b, s_q, gathered_topk, d_qk] (Stage-1 output)
  int stride_gathered_k_b = 0, stride_gathered_k_s_q = 0, stride_gathered_k_topk = 0;

  cutlass::bfloat16_t* __restrict__ out = nullptr;  // [b, s_q, h_q, d_v]
  int stride_o_b = 0, stride_o_s_q = 0, stride_o_h_q = 0;

  // --- Split-K over the gathered topk dim (num_kv_splits > 1 only) ---
  //
  // Per-split UNNORMALIZED partial O, written by the split-KV epilogue and consumed by
  // the reduction kernel (kernel/xe_mla_sparse_2stage_reduce_split_kv.hpp), which
  // combines the splits and writes `out` / `epilogue.lse`. Laid out
  // [b, s_q, num_kv_splits, h_q, d_v] so that for a fixed (b, s_q, kv_split) it is a
  // [h_q, d_v] 2D view -- structurally identical to the `out` view above, which lets the
  // split epilogue reuse the non-split block-2D store path verbatim (only the base
  // pointer and the skipped normalization differ).
  //
  // Element type is ElementO (bf16), matching the paged MLA split-KV path
  // (mla/kernel/xe_mla_reduce_split_kv.hpp), so the existing TiledCopyO applies
  // unchanged. The partials are unnormalized, so this does cost precision relative to
  // an fp32 accumulator buffer; the reduction accumulates in fp32.
  cutlass::bfloat16_t* __restrict__ o_accum = nullptr;
  int stride_o_accum_b = 0, stride_o_accum_s_q = 0, stride_o_accum_split = 0, stride_o_accum_h_q = 0;
};

// Stage-2 mainloop collective: QK/PV GEMM + online softmax over the gathered tile.
struct Mainloop2StageParams {
  int h_q = 0, topk = 0, extra_topk = 0, gathered_topk = 0;
  float sm_scale_div_log2 = 0.f;

  void* __restrict__ q = nullptr;  // fp8 re-read path; bf16 path uses the kernel's Q tensor
  int stride_q_b = 0, stride_q_s_q = 0, stride_q_h_q = 0;
  float* __restrict__ q_scale = nullptr;  // scalar or [h_q]; nullptr for bf16 query
  int q_scale_numel = 0;

  int* __restrict__ gathered_valid_mask = nullptr;  // [b, s_q, gathered_topk]
  int stride_gathered_mask_b = 0, stride_gathered_mask_s_q = 0;

  int* __restrict__ topk_length = nullptr;  // [b], may be nullptr
  int stride_topk_length_b = 0;
  int* __restrict__ extra_topk_length = nullptr;  // [b], may be nullptr
  int stride_extra_topk_length_b = 0;
};

// Stage-2 epilogue collective: cross-subgroup reduce, normalize, LSE / max_logits,
// optional attn_sink merge, store. max_logits is populated only by the prefill
// path; the decode epilogue is templated HAS_MAX_LOGITS=false and compiles the
// write out (leaving these fields null/0).
struct Epilogue2StageParams {
  int h_q = 0;
  float sm_scale_div_log2 = 0.f;

  float* __restrict__ lse = nullptr;  // [b, s_q, h_q]
  int stride_lse_b = 0, stride_lse_s_q = 0;

  float* __restrict__ attn_sink = nullptr;  // [h_q], may be nullptr

  float* __restrict__ max_logits = nullptr;  // [b, s_q, h_q], prefill only
  int stride_max_logits_b = 0, stride_max_logits_s_q = 0;

  // --- Split-K over the gathered topk dim (num_kv_splits > 1 only) ---
  //
  // Per-split softmax row stats published by the split-KV epilogue alongside the
  // unnormalized partial O in Kernel2StageParams::o_accum, and consumed by the
  // reduction kernel. Both are [b, s_q, num_kv_splits, h_q]; split_max_logits is in the
  // *log2* domain and already scaled by sm_scale_div_log2 (it is the mainloop's tA_max
  // verbatim), matching what the paged MLA reduction expects of its max_logits.
  //
  // An empty trailing split (blk_start >= num_topk_blocks) publishes
  // split_exp_sums == 0, which is the reduction's "skip this split" signal -- the same
  // contract as the paged path (mla/kernel/xe_mla_kernel.hpp:557).
  //
  // These are *not* epilogue.lse / epilogue.max_logits: those stay the final per-row
  // outputs and, under split-K, are written by the reduction kernel instead of here.
  float* __restrict__ split_exp_sums = nullptr;
  float* __restrict__ split_max_logits = nullptr;
  int stride_split_stats_b = 0, stride_split_stats_s_q = 0, stride_split_stats_split = 0;
};

// Stage-1 gather common params (base). This is the standalone Stage-1 kernel's own
// Params (its decode / prefill child below is what SparseGatherKernel launches with)
// -- independent of the Stage-2 SparseAttn2StageParams. The subgroup-coalesced
// gather grid, the per-(batch, seq) index/gathered base pointers, and the valid-mask
// write are shared by decode and prefill; the path-specific KV *source* fields live
// in the children below.
struct Gather2StageParams {
  int b = 0, s_q = 0, topk = 0, gathered_topk = 0;

  int* __restrict__ indices = nullptr;  // [b, s_q, topk]
  int stride_indices_b = 0, stride_indices_s_q = 0;

  int* __restrict__ topk_length = nullptr;  // [b], may be nullptr
  int stride_topk_length_b = 0;

  cutlass::bfloat16_t* __restrict__ gathered_k = nullptr;  // [b, s_q, gathered_topk, d_qk]
  int stride_gathered_k_b = 0, stride_gathered_k_s_q = 0, stride_gathered_k_topk = 0;

  int* __restrict__ gathered_valid_mask = nullptr;  // [b, s_q, gathered_topk]
  int stride_gathered_mask_b = 0, stride_gathered_mask_s_q = 0;
};

// Decode gather child: dual packed-fp8 *paged* pools (primary + extra), dequantized.
struct DecodeGather2StageParams : Gather2StageParams {
  int num_blocks = 0, page_block_size = 0;
  int extra_num_blocks = 0, extra_page_block_size = 0, extra_topk = 0;

  uint8_t* __restrict__ kv = nullptr;  // packed fp8 KV cache
  int stride_kv_block = 0;
  uint8_t* __restrict__ extra_kv = nullptr;  // packed fp8 extra KV cache, may be nullptr
  int stride_extra_kv_block = 0;

  int* __restrict__ extra_indices = nullptr;  // [b, s_q, extra_topk]
  int stride_extra_indices_b = 0, stride_extra_indices_s_q = 0;
  int* __restrict__ extra_topk_length = nullptr;  // [b], may be nullptr
  int stride_extra_topk_length_b = 0;
};

// Prefill gather child: dense bf16 *unpaged* source, plain D_QK-wide copy.
struct PrefillGather2StageParams : Gather2StageParams {
  int s_kv = 0;
  cutlass::bfloat16_t* __restrict__ kv_dense = nullptr;  // [s_kv, h_kv=1, d_qk]
  int stride_kv_dense_s = 0;
};

// ---------------------------------------------------------------------------
// Stage-2 dense params: the layers the dense flash kernel actually consumes,
// bundled the way a normal (non-sparse) MLA kernel bundles its Params fan-out.
// It carries NO gather slice: Stage 1 is a separate kernel with its own Params
// (the Gather2StageParams children above). The runner (device::MLASparse) holds one
// Params member per stage and launches both, exactly as device::MLA does for the
// split-KV attention + reduction pair. The two stages communicate only through the
// gathered_k / gathered_valid_mask HBM buffers, whose pointers+strides each side
// records in its own params.
//
// Both paths (decode / prefill) share this one type -- the Stage-2 dense kernel,
// collectives, and tile scheduler are path-agnostic; the path-specific bits live
// entirely in the Stage-1 gather params.
// ---------------------------------------------------------------------------
struct SparseAttn2StageParams {
  Kernel2StageParams kernel;
  Mainloop2StageParams mainloop;
  Epilogue2StageParams epilogue;
  TileScheduler2StageParams scheduler;
};

// ---------------------------------------------------------------------------
// Split-K (over the gathered topk dim) HBM scratch sizing + strides.
//
// Host-only helper, the sparse analog of the paged path's SplitKVWorkspaceLayout
// (mla/kernel/xe_mla_kernel.hpp:47). Two differences, both from the fact that Stage 2
// is decode-*shaped* rather than decode-only: it carries an s_q dim (paged MLA decode
// has seq_len_qo == 1 and omits it), and it reports the per-tensor strides directly so
// the caller can drop them straight into Kernel2StageParams / Epilogue2StageParams
// instead of building CuTe strides.
//
// Layouts (all tightly packed, row-major in the listed order):
//   o_accum          [b, s_q, num_kv_splits, h_q, d_v]  ElementO (bf16)
//   split_exp_sums   [b, s_q, num_kv_splits, h_q]       float
//   split_max_logits [b, s_q, num_kv_splits, h_q]       float
//
// Offsets are 256B-aligned like the paged layout so a single blob can back all three,
// but the sparse host path allocates them as separate tensors (the way it already
// allocates Stage 1's gathered_k / gathered_valid_mask) and only uses the byte totals
// for the workspace accounting that bounds the batch-chunk size.
struct SparseSplitKV2StageWorkspaceLayout {
  size_t o_accum_bytes = 0;
  size_t stats_bytes = 0;  // per stats tensor (exp_sums and max_logits are the same size)
  size_t total_bytes = 0;

  // o_accum strides, in elements.
  int stride_o_accum_b = 0, stride_o_accum_s_q = 0, stride_o_accum_split = 0, stride_o_accum_h_q = 0;
  // Shared by split_exp_sums and split_max_logits, in elements.
  int stride_split_stats_b = 0, stride_split_stats_s_q = 0, stride_split_stats_split = 0;

  SparseSplitKV2StageWorkspaceLayout() = default;

  SparseSplitKV2StageWorkspaceLayout(int b, int s_q, int h_q, int num_kv_splits, int d_v, size_t elem_o_size) {
    const size_t rows = size_t(b) * s_q * num_kv_splits * h_q;
    o_accum_bytes = rows * d_v * elem_o_size;
    stats_bytes = rows * sizeof(float);
    auto align256 = [](size_t n) { return (n + 255) & ~size_t(255); };
    total_bytes = align256(o_accum_bytes) + 2 * align256(stats_bytes);

    stride_o_accum_h_q = d_v;
    stride_o_accum_split = h_q * d_v;
    stride_o_accum_s_q = num_kv_splits * h_q * d_v;
    stride_o_accum_b = s_q * num_kv_splits * h_q * d_v;

    stride_split_stats_split = h_q;
    stride_split_stats_s_q = num_kv_splits * h_q;
    stride_split_stats_b = s_q * num_kv_splits * h_q;
  }
};

// ===========================================================================
// Stage-2 dense-decode DPAS/tile configuration knob, consumed by
// MlaSparseDecode2StageTileTraits below. The config struct that assembles the
// collectives, kernels, and device::MLASparse runner around those traits is
// MlaSparseDecode2StageXe in device/mla_sparse_decode_2stage_types.hpp (host side,
// matching the fused path's MlaSparseXe convention), and it forwards this knob.
// ===========================================================================

#ifndef FLASH_MLA_PREFILL_V_SPLIT
#define FLASH_MLA_PREFILL_V_SPLIT 4
#endif

// Prefill maps each query row to a decode "batch" (shape.b = s_q), so the Stage-2
// grid (ceil_div(h_q, B_H) * s_q * b) is already saturated at V_SPLIT=1 for the
// hundreds-to-thousands of query rows. A larger V_SPLIT there is mostly redundant:
// every v-split work-group re-reads the full K tile and recomputes the full QK GEMM
// (the PV split only narrows the output slice). So prefill uses a smaller V-split
// than decode -- 2 roughly halves the redundant K re-reads / QK recompute.
//
// This smaller value is applied ONLY for B_H >= 32 (see sparse_mla_prefill_v_split
// in mla_sparse_prefill_2stage_types.hpp). For B_H <= 16 the PV subgroup layout
// splits the topk dim (ReduceK > 1), so the epilogue's SharedStorageReduceK SLM
// scales with D_V_PER_SPLIT = D_V / V_SPLIT; shrinking V_SPLIT there doubles that
// SLM and drops occupancy (measured ~50% slower at h_q=16), so those configs keep
// the decode-sized V-split of 4. For B_H >= 32 the layout splits heads (ReduceK==1,
// no epilogue SLM), so the smaller split is a pure win (measured ~35% faster).
#ifndef FLASH_MLA_SPARSE_PREFILL_V_SPLIT
#define FLASH_MLA_SPARSE_PREFILL_V_SPLIT 2
#endif

// ===========================================================================
// Stage-2 dense-decode DPAS / tile geometry.
//
// MlaSparseDecode2StageTileTraits is the inner half of what used to be one
// monolithic config struct: the pure Stage-2 *geometry* -- element types, DPAS MMA
// atoms, tile shapes, subgroup layouts, and the derived size constants. It is what
// the collectives and the dense kernel wrapper receive as their `Traits` template
// parameter and read members off (Traits::B_H, Traits::TiledMMAQK, ...), which is
// why it lives here in the shared header alongside the params blocks they also read.
//
// The outer half -- the *assembly* (which collectives, which tile scheduler, which
// Stage-1 gather kernel, which runner) -- stays in the host types header as
// MlaSparseDecode2StageXe (device/mla_sparse_decode_2stage_types.hpp).
//
// Why the split. The config struct previously passed *itself* as its collectives'
// Traits, i.e. it was named as a template argument while still incomplete:
//
//     MlaSparseDecode2StageXe -> CollectiveMainloop<..., MlaSparseDecode2StageXe>
//                             -> CollectiveEpilogue<CollectiveMainloop, ...>
//                             -> DenseKernel -> Fmla
//
// That self-reference compiled only because every alias in the chain is lazy and
// nothing inside them touched the enclosing type eagerly; a single member needing the
// complete type would break it with an error that points nowhere useful. It also let
// the collectives reach members that are none of their business -- including Fmla, the
// runner that contains them, and GatherKernel, the other stage.
//
// With the geometry here, the traits type is COMPLETE before any collective names it,
// the cycle is gone, and a collective can only see geometry. This also matches how the
// dense (non-sparse) MLA path parameterizes XeMlaMainloop with explicit geometry
// (TiledMMAQK / TiledMMAPV / VTiles / Tensor*), just bundled instead of spelled out
// per-parameter -- there are 16 distinct members in use, which is well past the point
// where individual template params are the clearer option.
//
// This is purely a compile-time / coupling concern: the traits type is never
// instantiated (no object, no sizeof, no pass-by-value anywhere), so it costs no
// registers, no SLM, and no kernel-argument bytes. Every use is
// `typename Traits::X` or `Traits::kConstant`.
//
// T is the op's query dtype (sycl::half / sycl::ext::oneapi::bfloat16), resolved to a
// cutlass element via SparseMlaToCutlassElementType exactly like the fused path's
// MlaSparseXe, so the geometry can be instantiated straight from the dispatched dtype
// without branching on it.
//
// Keyed by (T, D_QK, B_H, V_SPLIT) only -- the flags that select *behavior* rather
// than geometry (HAS_ATTN_SINK, HAS_MAX_LOGITS) and the Stage-1 gather choice belong
// to the assembly layer and are deliberately absent here.
// ===========================================================================
template <typename T, int D_QK_, int B_H_, int V_SPLIT_>
struct MlaSparseDecode2StageTileTraits {
  static constexpr int D_QK = D_QK_;

  // Query element resolved from the op's dtype, mirroring the fused MlaSparseXe. K/V
  // are the Stage-1 gathered bf16 latent and the out / gathered_k param slices are
  // bf16, so those stay bf16 (the QK DPAS is bf16; a non-bf16 query is converted on
  // load). IS_FP8_QUERY is deduced from the element -- true only for an fp8 query,
  // which the current codegen never instantiates (half/bf16 only), so it is false in
  // practice; the fp8 dequant path stays compiled behind it for when it is wired up.
  using ElementType = typename SparseMlaToCutlassElementType<T>::type;
  using ElementQ = ElementType;
  using ElementKV = ElementType;
  using ElementO = ElementType;
  static constexpr bool IS_FP8_QUERY = cute::is_same_v<ElementQ, cutlass::float_e4m3_t>;

  using StrideQ = cute::tuple<int, _1, int>;
  using StrideKV = cute::tuple<int, _1, int>;
  using StrideO = cute::tuple<int, _1, int>;

  static constexpr int B_H = B_H_;  // h_q block size
  static constexpr int SUBGROUP_SIZE = intel::sg_size;
  static constexpr int NUM_SUBGROUPS = B_H > 16 ? (B_H > 32 ? 8 : 4) : 4;
  static constexpr int NUM_THREADS = NUM_SUBGROUPS * SUBGROUP_SIZE;
  static constexpr int B_TOPK = 64;  // topk_length block size

  static constexpr int D_PE = 64;
  static constexpr int D_V = 512;
  // V-split factor: how many work-groups split the D_V output for one query tile.
  // Decode and prefill pass different values (prefill's grid is already saturated by
  // its s_q batch dim); see the knob comments just above.
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

  constexpr static int SGTileQ = get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})))();
  // bf16 dpas m8n16k16
  // (8, 128, 64) / ((8, 16, 16) * (1, 16, 1)) = (1, 1, 4) iterations per subgroup
  constexpr static int MAX_M_DPAS = 8;
  using MMAOperation = XE_DPAS_TT<cute::gcd(SGTileQ, MAX_M_DPAS), float, ElementType>;
  using TiledMMAQK = typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<TileShapeQK>, SubgroupLayoutQK>::TiledMMA;
  using TiledMMAPV = typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<TileShapePV>, SubgroupLayoutPV>::TiledMMA;
};

}  // namespace cutlass::flash_attention::kernel
