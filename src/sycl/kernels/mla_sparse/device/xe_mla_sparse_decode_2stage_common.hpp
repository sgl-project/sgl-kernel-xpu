/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
/*!
  \file
  \brief Two-stage sparse MLA decode shared device declarations for DeepSeek V4.

  Contains:
    - LOG_2_E / LOG_E_2 log-base constants + packed FP8 KV layout constants.
    - SparseDecode2StageProblemShape: pure problem geometry.
    - The per-layer param blocks (Kernel2StageParams / Mainloop2StageParams /
      Epilogue2StageParams / TileScheduler2StageParams / Gather2StageParams and its
      decode / prefill children) and the composite {Decode,Prefill}SparseAttn2StageParams.
    - DISPATCH_BOOLEAN_FLAG: compile-time boolean dispatch.
    - FLASH_MLA_PREFILL_V_SPLIT: dense-decode V-split knob (the DPAS/tile config
      struct MlaSparseDecode2StageXe that reads it lives in the host types header).

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

// rmem<->smem block copies (copy_block_r2s / copy_block_s2r, in namespace cute) used
// by the dense kernel's cross-subgroup softmax reduction (only reached when V_SPLIT
// produces ReduceK > 1). Shared with the rest of the repo.
#include "sycl/comm/copy_block_slm.hpp"

using namespace cute;

namespace cutlass::flash_attention::kernel {

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
// layer and the fields it touches is explicit. The composite
// {Decode,Prefill}SparseAttn2StageParams below assembles these into the single
// Params object shared by the Stage-1 gather and Stage-2 dense launches (the
// device::MLASparse runner hands the same object to both).
//
// Unused monolith fields (plain sm_scale, is_fp8_query, h_kv, the SplitKV block)
// are intentionally dropped: they were host-set but never read on device.
// ===========================================================================

// Tile scheduler: decodes the launch grid into (batch, seq, head-block, v-split).
// Reads only the two dims needed to enumerate head-blocks per query tile.
struct TileScheduler2StageParams {
  int h_q = 0;
  int s_q = 0;
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
};

// Stage-1 gather common params (base). The subgroup-coalesced gather grid, the
// per-(batch, seq) index/gathered base pointers, and the valid-mask write are
// shared by decode and prefill; the path-specific KV *source* fields live in the
// children below.
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
// Composite params: one base carrying the layers common to both paths, plus a
// decode / prefill child adding the path-specific gather slice. This is the
// single Params object the device::MLASparse runner shares between the Stage-1
// gather launch and the Stage-2 dense launch (both take typename K::Params, and
// the dense kernel derives its Params from its GatherKernel so the two always
// agree). The Stage-2 dense kernel + collectives + tile scheduler read only the
// base slices, so they are path-agnostic; only the gather kernel touches
// `.gather`.
// ---------------------------------------------------------------------------
struct SparseAttn2StageParamsBase {
  Kernel2StageParams kernel;
  Mainloop2StageParams mainloop;
  Epilogue2StageParams epilogue;
  TileScheduler2StageParams scheduler;
};

struct DecodeSparseAttn2StageParams : SparseAttn2StageParamsBase {
  DecodeGather2StageParams gather;
};

struct PrefillSparseAttn2StageParams : SparseAttn2StageParamsBase {
  PrefillGather2StageParams gather;
};

// ===========================================================================
// Stage-2 dense-decode DPAS/tile configuration knob. The full config struct
// (MlaSparseDecode2StageXe) that assembles the tile shapes, MMAs, collectives,
// and the device::MLASparse runner lives in
// device/mla_sparse_decode_2stage_types.hpp (host side, matching the fused
// path's MlaSparseXe convention); it reads this V-split knob.
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

}  // namespace cutlass::flash_attention::kernel
