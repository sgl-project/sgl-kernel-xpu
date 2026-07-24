/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
/*!
  \file
  \brief Two-stage sparse MLA decode shared device declarations for DeepSeek V4.

  Contains:
    - LOG_2_E / LOG_E_2 log-base constants + packed FP8 KV layout constants.
    - SparseAttnDecodeParams / XPUSparseDecodeAttnFwdParams: decode kernel param block.
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
// compile-time boolean dispatch.
// ---------------------------------------------------------------------------
#define DISPATCH_BOOLEAN_FLAG(FLAG, CONSTEXPR_NAME, ...) \
  [&]() {                                                \
    if (FLAG) {                                          \
      static constexpr bool CONSTEXPR_NAME = true;       \
      return __VA_ARGS__();                              \
    } else {                                             \
      static constexpr bool CONSTEXPR_NAME = false;      \
      return __VA_ARGS__();                              \
    }                                                    \
  }();

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
// pointers and strides so it can be reasoned about on its own and passed to a
// tile scheduler, mirroring the sycl-tla / device::MLA convention. Carried as the
// `shape` member of SparseAttnDecodeParams (below). The collective-template
// widening that would consume this like the fused mainloop is a later step; for
// now the kernels read the fields off params.shape.
// ---------------------------------------------------------------------------
struct SparseDecode2StageProblemShape {
  int b = 0;                      // batch
  int s_q = 0;                    // query seqlen (1 for decode)
  int h_q = 0;                    // number of query heads
  int h_kv = 0;                   // number of KV heads (1 for MLA)
  int d_qk = 0;                   // QK head dim (512 = 448 nope + 64 rope)
  int d_v = 0;                    // V head dim (512)
  int num_blocks = 0;             // primary KV cache pages
  int page_block_size = 0;        // primary KV cache page size
  int topk = 0;                   // primary sparse top-k
  int gathered_topk = 0;          // topk + extra_topk (dense gathered tile width)
  int extra_num_blocks = 0;       // extra KV cache pages
  int extra_page_block_size = 0;  // extra KV cache page size
  int extra_topk = 0;             // extra pool sparse top-k

  SparseDecode2StageProblemShape() = default;
};
// take the base struct (no queue) as their Params; the XPU variant carries the
// sycl::queue for the host-side launch (device-only copy of a queue is not allowed).
//
// The pure problem geometry now lives in the nested `shape` member
// (SparseDecode2StageProblemShape); the remaining fields are data pointers,
// scalars, and strides. Kernels read dims via params.shape.<dim>.
// ---------------------------------------------------------------------------
struct SparseAttnDecodeParams {
  using ProblemShape = SparseDecode2StageProblemShape;
  ProblemShape shape;
  float sm_scale, sm_scale_div_log2;
  bool is_fp8_query;
  float* __restrict__ q_scale;  // scalar or [h_q], may be nullptr for bf16 query
  int q_scale_numel;

  void* __restrict__ q;           // [b, s_q, h_q, d_qk], bf16 or fp8_e4m3
  uint8_t* __restrict__ kv;       // packed fp8 KV cache, [num_blocks, page_block_size, h_kv=1, head_bytes]
  int* __restrict__ indices;      // [b, s_q, topk]
  int* __restrict__ topk_length;  // [b], may be nullptr
  float* __restrict__ attn_sink;  // [h_q], may be nullptr
  cutlass::bfloat16_t* __restrict__ gathered_k;  // [b, s_q, gathered_topk, d_qk]
  int* __restrict__ gathered_valid_mask;         // [b, s_q, gathered_topk]

  float* __restrict__ lse;                // [b, s_q, h_q]
  cutlass::bfloat16_t* __restrict__ out;  // [b, s_q, h_q, d_v]

  int extra_num_blocks, extra_page_block_size, extra_topk;
  uint8_t* __restrict__ extra_kv;       // packed fp8 KV cache, may be nullptr
  int* __restrict__ extra_indices;      // [b, s_q, extra_topk]
  int* __restrict__ extra_topk_length;  // [b], may be nullptr

  int stride_q_b, stride_q_s_q, stride_q_h_q;
  int stride_kv_block, stride_kv_row, stride_kv_head;
  int stride_indices_b, stride_indices_s_q;
  int stride_topk_length_b, stride_topk_length_s_q;  // stride_topk_length_s_q is unused for decode (1D topk_length)
  int stride_gathered_k_b, stride_gathered_k_s_q, stride_gathered_k_topk;
  int stride_gathered_mask_b, stride_gathered_mask_s_q;
  int stride_lse_b, stride_lse_s_q;
  int stride_o_b, stride_o_s_q, stride_o_h_q;
  int stride_extra_kv_block, stride_extra_kv_row, stride_extra_kv_head;
  int stride_extra_indices_b, stride_extra_indices_s_q;
  int stride_extra_topk_length_b,
      stride_extra_topk_length_s_q;  // stride_extra_topk_length_s_q is unused for decode (1D extra_topk_length)

  // SplitKV-related parameters (unused by the no-split sparse decode for now)
  float* __restrict__ lse_accum;  // [num_splits, s_q, h_q]
  float* __restrict__ o_accum;    // [num_splits, s_q, h_q, d_v]
  int stride_lse_accum_split, stride_lse_accum_s_q;
  int stride_o_accum_split, stride_o_accum_s_q, stride_o_accum_h_q;
  void* __restrict__ tile_scheduler_metadata_ptr;
  int* __restrict__ num_splits_ptr;
  int num_sm_parts;
  int num_sm;
};

struct XPUSparseDecodeAttnFwdParams : public SparseAttnDecodeParams {
  sycl::queue queue;
};

template <int D_QK, bool IS_FP8_QUERY>
void launch_sparse_mla_decode_fp8_fwd_kernel(const XPUSparseDecodeAttnFwdParams& params);

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

}  // namespace cutlass::flash_attention::kernel
