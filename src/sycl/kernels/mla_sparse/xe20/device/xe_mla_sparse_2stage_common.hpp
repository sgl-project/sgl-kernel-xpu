/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
/*!
  \file
  \brief Two-stage sparse MLA shared device declarations.

  Shared by BOTH two-stage paths (decode and prefill): the Stage-2 dense kernel,
  its collectives, and its tile geometry are path-agnostic, and the Stage-1 gather
  params keep their common base here with one child per path.
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
#include "cutlass/fast_math.h"
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
// log-base constants.
// ---------------------------------------------------------------------------
static constexpr float LOG_2_E = 1.4426950408889634f;
static constexpr float LOG_E_2 = 0.6931471805599453f;

// ---------------------------------------------------------------------------
// Packed FP8 KV cache layout for sparse MLA decode, keyed by the QK head dim.
//
//   D_QK = 512 -- DeepSeek V4 ("MODEL1"), 584 B/token. Page-internal *sections*:
//     a data section of page_block_size records
//         [448 B fp8 NoPE | 128 B bf16 RoPE]     (record stride 576 B)
//     followed by a page-END scale section of page_block_size records
//         [7 UE8M0 scale bytes | 1 pad]          (record stride 8 B, one scale per 64)
//     k_cache is an as_strided view whose stride(1) == 584 is a metadata value, NOT
//     physical token spacing; stride(0) carries the real (576-aligned) page stride.
//
//   D_QK = 576 -- DeepSeek V3.2 / GLM-DSA, 656 B/token. One contiguous, self-contained
//     record per token, no page-end section:
//         [512 B fp8 NoPE | 16 B = 4 fp32 scales | 128 B bf16 RoPE]
//     one scale per 128 NoPE values, inline. The tensor is plain contiguous, so
//     stride(1) == 656 IS real token spacing and the page stride is exactly
//     page_block_size * 656.
//
// ---------------------------------------------------------------------------
template <int D_QK>
struct SparseMlaFp8KvLayout;

template <>
struct SparseMlaFp8KvLayout<512> {
  static constexpr int NOPE_DIM = 448;  // fp8_e4m3 values
  static constexpr int ROPE_DIM = 64;   // bf16 values

  static constexpr int QUANT_GROUP = 64;                     // NoPE values sharing one scale
  static constexpr int NUM_SCALES = NOPE_DIM / QUANT_GROUP;  // 7
  static constexpr int SCALE_BYTES = 8;                      // 7 UE8M0 bytes + 1 pad

  // Spacing between consecutive token records inside the page's data section.
  static constexpr int TOKEN_STRIDE_BYTES = NOPE_DIM + ROPE_DIM * 2;  // 576
  static constexpr int ROPE_BYTE_OFFSET = NOPE_DIM;                   // 448, within the record
  static constexpr int SCALE_BYTE_OFFSET = 0;                         // unused (page-end scales)

  static constexpr bool SCALES_INLINE = false;    // separate page-end scale section
  static constexpr bool SCALES_ARE_FP32 = false;  // UE8M0 exponent bytes

  // Scales live outside the record, so the advertised per-token width adds them on.
  static constexpr int HEAD_BYTES = TOKEN_STRIDE_BYTES + SCALE_BYTES;  // 584

  static_assert(NOPE_DIM + ROPE_DIM == 512, "NoPE + RoPE must equal the D_QK this layout is keyed by");
  static_assert(NOPE_DIM % QUANT_GROUP == 0, "NoPE must tile evenly over the quant group");
  static_assert(NUM_SCALES == SCALE_BYTES - 1, "only the first seven scale bytes are valid for 448 NoPE values");
};

template <>
struct SparseMlaFp8KvLayout<576> {
  static constexpr int NOPE_DIM = 512;  // fp8_e4m3 values
  static constexpr int ROPE_DIM = 64;   // bf16 values

  static constexpr int QUANT_GROUP = 128;                    // NoPE values sharing one scale
  static constexpr int NUM_SCALES = NOPE_DIM / QUANT_GROUP;  // 4
  static constexpr int SCALE_BYTES = NUM_SCALES * 4;         // 16, fp32 scales

  // The whole record is contiguous and includes its own scales, so this is both the
  // record width and the per-token width the host sees.
  static constexpr int TOKEN_STRIDE_BYTES = NOPE_DIM + SCALE_BYTES + ROPE_DIM * 2;  // 656
  static constexpr int SCALE_BYTE_OFFSET = NOPE_DIM;                                // 512
  static constexpr int ROPE_BYTE_OFFSET = NOPE_DIM + SCALE_BYTES;                   // 528

  static constexpr bool SCALES_INLINE = true;    // inline, immediately after NoPE
  static constexpr bool SCALES_ARE_FP32 = true;  // fp32  scales

  static constexpr int HEAD_BYTES = TOKEN_STRIDE_BYTES;  // 656

  static_assert(NOPE_DIM + ROPE_DIM == 576, "NoPE + RoPE must equal the D_QK this layout is keyed by");
  static_assert(NOPE_DIM % QUANT_GROUP == 0, "NoPE must tile evenly over the quant group");
  static_assert(HEAD_BYTES == 656, "DeepSeek V3.2 / GLM-DSA packed fp8 KV cache is 656 bytes per token");
};

// ---------------------------------------------------------------------------
// Problem shape for the two-stage sparse MLA decode.
// ---------------------------------------------------------------------------
struct SparseDecode2StageProblemShape {
  int b = 0;                      // batch (prefill: mapped from query rows s_q)
  int s_q = 0;                    // query seqlen (1 for decode; 1 per mapped row for prefill)
  int h_q = 0;                    // number of query heads
  int h_kv = 0;                   // number of KV heads (1 for MLA)
  int d_qk = 0;                   // QK head dim: 512 (448 nope + 64 rope) or 576 (512 nope + 64 rope)
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

struct TileScheduler2StageParams {
  int h_q = 0;
  int s_q = 0;
  int num_kv_splits = 1;
};

struct Kernel2StageParams {
  SparseDecode2StageProblemShape shape;

  void* __restrict__ q = nullptr;  // [b, s_q, h_q, d_qk], bf16 or fp8_e4m3
  int stride_q_b = 0, stride_q_s_q = 0, stride_q_h_q = 0;

  cutlass::bfloat16_t* __restrict__ gathered_k = nullptr;  // [b, s_q, gathered_topk, d_qk] (Stage-1 output)
  int stride_gathered_k_b = 0, stride_gathered_k_s_q = 0, stride_gathered_k_topk = 0;

  cutlass::bfloat16_t* __restrict__ out = nullptr;  // [b, s_q, h_q, d_v]
  int stride_o_b = 0, stride_o_s_q = 0, stride_o_h_q = 0;

  // --- Split-K over the gathered topk dim (num_kv_splits > 1 only) ---
  cutlass::bfloat16_t* __restrict__ o_accum = nullptr;
  int stride_o_accum_b = 0, stride_o_accum_s_q = 0, stride_o_accum_split = 0, stride_o_accum_h_q = 0;
};

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

struct Epilogue2StageParams {
  int h_q = 0;
  int b = 0, s_q = 0, num_kv_splits = 1;
  float sm_scale_div_log2 = 0.f;

  float* __restrict__ lse = nullptr;  // [b, s_q, h_q]
  int stride_lse_b = 0, stride_lse_s_q = 0;

  float* __restrict__ attn_sink = nullptr;  // [h_q], may be nullptr

  float* __restrict__ max_logits = nullptr;  // [b, s_q, h_q], prefill only
  int stride_max_logits_b = 0, stride_max_logits_s_q = 0;

  // --- Split-K over the gathered topk dim (num_kv_splits > 1 only) ---
  float* __restrict__ split_exp_sums = nullptr;
  float* __restrict__ split_max_logits = nullptr;
  int stride_split_stats_b = 0, stride_split_stats_s_q = 0, stride_split_stats_split = 0;
};

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

struct DecodeGather2StageParams : Gather2StageParams {
  int num_blocks = 0, page_block_size = 0;
  int extra_num_blocks = 0, extra_page_block_size = 0, extra_topk = 0;

  // Precomputed magic-number reciprocals for the two page sizes, so locate_token's
  // token_idx -> (block_idx, rel_idx) split costs a multiply-high plus two shifts instead
  // of an divide.
  cutlass::FastDivmod page_block_divmod;
  cutlass::FastDivmod extra_page_block_divmod;

  uint8_t* __restrict__ kv = nullptr;  // packed fp8 KV cache
  int stride_kv_block = 0;
  uint8_t* __restrict__ extra_kv = nullptr;  // packed fp8 extra KV cache, may be nullptr
  int stride_extra_kv_block = 0;

  int* __restrict__ extra_indices = nullptr;  // [b, s_q, extra_topk]
  int stride_extra_indices_b = 0, stride_extra_indices_s_q = 0;
  int* __restrict__ extra_topk_length = nullptr;  // [b], may be nullptr
  int stride_extra_topk_length_b = 0;
};

struct PrefillGather2StageParams : Gather2StageParams {
  int s_kv = 0;
  cutlass::bfloat16_t* __restrict__ kv_dense = nullptr;  // [s_kv, h_kv=1, d_qk]
  int stride_kv_dense_s = 0;
};

struct SparseAttn2StageParams {
  Kernel2StageParams kernel;
  Mainloop2StageParams mainloop;
  Epilogue2StageParams epilogue;
  TileScheduler2StageParams scheduler;
};

// ---------------------------------------------------------------------------
// Split-K (over the gathered topk dim) HBM scratch sizing + strides.
//
// Layouts (all tightly packed, row-major in the listed order):
//   o_accum          [b, s_q, num_kv_splits, h_q, d_v]  ElementO (bf16)
//   split_exp_sums   [b, s_q, num_kv_splits, h_q]       float
//   split_max_logits [b, s_q, num_kv_splits, h_q]       float
// ---------------------------------------------------------------------------
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
// Stage-2 dense-decode/prefill DPAS/tile configuration knob, consumed by
// MlaSparseDecode2StageTileTraits below.
// ===========================================================================
#ifndef FLASH_MLA_PREFILL_V_SPLIT
#define FLASH_MLA_PREFILL_V_SPLIT 4
#endif

#ifndef FLASH_MLA_SPARSE_PREFILL_V_SPLIT
#define FLASH_MLA_SPARSE_PREFILL_V_SPLIT 2
#endif

// ===========================================================================
// Stage-2 dense-decode DPAS / tile geometry.
// ===========================================================================
template <typename T, int D_QK_, int B_H_, int V_SPLIT_>
struct MlaSparseDecode2StageTileTraits {
  static constexpr int D_QK = D_QK_;
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
  // Decode and prefill pass different values
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
