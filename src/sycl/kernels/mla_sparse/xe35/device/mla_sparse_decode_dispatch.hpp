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
  \brief Forward declarations for generated Sparse MLA decode kernel launch functions
*/

#pragma once

#include <ATen/ATen.h>

#include <sycl/sycl.hpp>

// Compile-time selector for the sparse MLA decode implementation:
//   1 -> two-stage path (gather+dequant to HBM, then dense flash-decode)
//   0 -> fused path (SLM per-d-slice gather + inline DPAS)
// Set here for convenience; a build-time -DSGLANG_USE_SPARSE_MLA_2STAGE=<0|1>
// override still wins because of the guard. Consumed by an #if in
// mla_sparse_decode.cpp (SGL_DISABLE_PACKGQA-style A/B toggle).
#ifndef SGLANG_USE_SPARSE_MLA_2STAGE
#define SGLANG_USE_SPARSE_MLA_2STAGE 1
#endif

namespace mla_sparse_decode {

// Each function is defined in a separate generated .cpp file from
// mla_sparse_decode_kernel.cpp.in, compiled as its own library.
//
// Naming: launch_mla_sparse_decode_<ELEM_TAG>_<PAGE_SIZE>
// Parameters:
//   ELEM_TAG  in {half, bf16}
//   PAGE_SIZE in {128, 256}

#define DECLARE_MLA_SPARSE_DECODE_LAUNCH(ELEM)            \
  void launch_mla_sparse_decode_##ELEM##_128(             \
      at::Tensor& out,                                    \
      at::Tensor& lse_out,                                \
      const at::Tensor& q,                                \
      const at::Tensor& k_cache,                          \
      const at::Tensor& indices,                          \
      const std::optional<at::Tensor>& topk_length,       \
      const std::optional<at::Tensor>& extra_k_cache,     \
      const std::optional<at::Tensor>& extra_indices,     \
      const std::optional<at::Tensor>& extra_topk_length, \
      const std::optional<at::Tensor>& attn_sink,         \
      double sm_scale,                                    \
      int64_t head_dim_v,                                 \
      bool is_fp8_kvcache);

DECLARE_MLA_SPARSE_DECODE_LAUNCH(half)
DECLARE_MLA_SPARSE_DECODE_LAUNCH(bf16)

#undef DECLARE_MLA_SPARSE_DECODE_LAUNCH

// Two-stage variant (gather+dequant to HBM, then dense flash-decode). Selected at
// compile time via SGLANG_USE_SPARSE_MLA_2STAGE. Generated from
// mla_sparse_decode_2stage_kernel.cpp.in, one TU per (ELEM_TAG, D_QK, B_H,
// HAS_ATTN_SINK) -- D_QK is the QK head dim and B_H the sparse-decode analog of the
// fused path's PAGE_SIZE, together keying the Stage-2 config; HAS_ATTN_SINK selects the
// sink epilogue variant. One variant lands in its own object file (heavy CUTLASS
// instantiation split per file).
//
// D_QK additionally selects the packed fp8 KV cache byte layout, since the two are 1:1
// in production (512 -> DSv4 584 B/token, 576 -> DSv3.2 / GLM-DSA 656 B/token); see
// sparse_mla_decode_fp8_head_bytes below and SparseMlaFp8KvLayout in
// device/xe_mla_sparse_2stage_common.hpp. Same {512, 576} pair as sparse prefill.
//
// Naming: launch_mla_sparse_decode_2stage_<ELEM_TAG>_<D_QK>_<B_H>_<HAS_ATTN_SINK>
//   ELEM_TAG      in {half, bf16}
//   D_QK          in {512, 576}
//   B_H           in {8, 16, 32, 64}
//   HAS_ATTN_SINK in {0, 1}
#define DECLARE_MLA_SPARSE_DECODE_2STAGE_LAUNCH(ELEM, D_QK, B_H, SINK)   \
  void launch_mla_sparse_decode_2stage_##ELEM##_##D_QK##_##B_H##_##SINK( \
      at::Tensor& out,                                                   \
      at::Tensor& lse_out,                                               \
      const at::Tensor& q,                                               \
      const at::Tensor& k_cache,                                         \
      const at::Tensor& indices,                                         \
      const std::optional<at::Tensor>& topk_length,                      \
      const std::optional<at::Tensor>& extra_k_cache,                    \
      const std::optional<at::Tensor>& extra_indices,                    \
      const std::optional<at::Tensor>& extra_topk_length,                \
      const std::optional<at::Tensor>& attn_sink,                        \
      double sm_scale,                                                   \
      int64_t head_dim_v,                                                \
      bool is_fp8_kvcache);

#define DECLARE_MLA_SPARSE_DECODE_2STAGE_ALL_B_H(ELEM, D_QK) \
  DECLARE_MLA_SPARSE_DECODE_2STAGE_LAUNCH(ELEM, D_QK, 8, 0)  \
  DECLARE_MLA_SPARSE_DECODE_2STAGE_LAUNCH(ELEM, D_QK, 8, 1)  \
  DECLARE_MLA_SPARSE_DECODE_2STAGE_LAUNCH(ELEM, D_QK, 16, 0) \
  DECLARE_MLA_SPARSE_DECODE_2STAGE_LAUNCH(ELEM, D_QK, 16, 1) \
  DECLARE_MLA_SPARSE_DECODE_2STAGE_LAUNCH(ELEM, D_QK, 32, 0) \
  DECLARE_MLA_SPARSE_DECODE_2STAGE_LAUNCH(ELEM, D_QK, 32, 1) \
  DECLARE_MLA_SPARSE_DECODE_2STAGE_LAUNCH(ELEM, D_QK, 64, 0) \
  DECLARE_MLA_SPARSE_DECODE_2STAGE_LAUNCH(ELEM, D_QK, 64, 1)

DECLARE_MLA_SPARSE_DECODE_2STAGE_ALL_B_H(bf16, 512)
DECLARE_MLA_SPARSE_DECODE_2STAGE_ALL_B_H(bf16, 576)

#undef DECLARE_MLA_SPARSE_DECODE_2STAGE_LAUNCH
#undef DECLARE_MLA_SPARSE_DECODE_2STAGE_ALL_B_H

// Head-block (B_H) selection rule for the two-stage path. B_H is the sparse-decode
// analog of the fused path's page_size: it keys the per-(ELEM, B_H) launcher. Pure
// host logic (h_q -> B_H) with no CUTLASS dependency, so the op TU can pick the
// launcher without pulling in the heavy Stage-2 config header.
// TODO: currently a simple rule; a smarter heuristic could balance occupancy and
// per-WG workload.
inline int sparse_mla_decode_select_b_h(int h_q) {
  if (h_q <= 8) return 8;
  if (h_q <= 16) return 16;
  if (h_q <= 32) return 32;
  return 64;
}

// Expected packed fp8 KV cache last-dim (bytes per token) for a decode d_qk; 0 for an
// unsupported d_qk. The two layouts are 1:1 with d_qk, so this pairing is what the op
// validates k_cache against before dispatching.
//
//   512 -> 584: DSv4          448 fp8 nope + 128 B bf16 rope + 8 B page-END UE8M0 scales
//   576 -> 656: DSv3.2/GLM-DSA 512 fp8 nope + 16 B INLINE fp32 scales + 128 B bf16 rope
//
// Host-side mirror of SparseMlaFp8KvLayout<D_QK>::HEAD_BYTES, kept here (rather than
// read off the trait) so the op TU can validate without pulling in the heavy Stage-2
// config header. runMlaSparse2Stage static_asserts the two against each other, so the
// mirror cannot drift.
constexpr int sparse_mla_decode_fp8_head_bytes(int d_qk) {
  switch (d_qk) {
    case 512:
      return 584;
    case 576:
      return 656;
    default:
      return 0;
  }
}

}  // namespace mla_sparse_decode
