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
  \brief Forward declarations for generated Sparse MLA prefill kernel launch functions.

  The 2-stage sparse MLA prefill path reuses the decode 2-stage device stack (see
  device/mla_sparse_prefill_2stage_types.hpp); only the Stage-1 gather companion
  differs (dense bf16 copy vs. packed-fp8 dequant). These declarations mirror the
  decode dispatch header, with the prefill op signature (max_logits + lse outputs,
  dense bf16 kv, no dual pool).
*/

#pragma once

#include <ATen/ATen.h>

#include <sycl/sycl.hpp>

namespace mla_sparse_prefill {

// Each function is defined in a separate generated .cpp file from
// mla_sparse_prefill_2stage_kernel.cpp.in, compiled as its own library. D_QK is the QK
// head dim (prefill supports {512, 576}) and B_H the sparse analog of page size:
// together they key the Stage-2 config; HAS_ATTN_SINK selects the sink epilogue
// variant. One variant lands in its own object file (build OOM guard -- one sink
// variant per file instead of both).
//
// Naming: launch_mla_sparse_prefill_2stage_<ELEM_TAG>_<D_QK>_<B_H>_<HAS_ATTN_SINK>
// Parameters:
//   ELEM_TAG      in {half, bf16}
//   D_QK          in {512, 576}
//   B_H           in {8, 16, 32, 64}
//   HAS_ATTN_SINK in {0, 1}
#define DECLARE_MLA_SPARSE_PREFILL_2STAGE_LAUNCH(ELEM, D_QK, B_H, SINK)   \
  void launch_mla_sparse_prefill_2stage_##ELEM##_##D_QK##_##B_H##_##SINK( \
      at::Tensor& out,                                                    \
      at::Tensor& max_logits,                                             \
      at::Tensor& lse,                                                    \
      const at::Tensor& q,                                                \
      const at::Tensor& kv,                                               \
      const at::Tensor& indices,                                          \
      const std::optional<at::Tensor>& attn_sink,                         \
      const std::optional<at::Tensor>& topk_length,                       \
      double sm_scale,                                                    \
      int64_t head_dim_v);

#define DECLARE_MLA_SPARSE_PREFILL_2STAGE_ALL_B_H(ELEM)      \
  DECLARE_MLA_SPARSE_PREFILL_2STAGE_LAUNCH(ELEM, 512, 8, 0)  \
  DECLARE_MLA_SPARSE_PREFILL_2STAGE_LAUNCH(ELEM, 512, 8, 1)  \
  DECLARE_MLA_SPARSE_PREFILL_2STAGE_LAUNCH(ELEM, 512, 16, 0) \
  DECLARE_MLA_SPARSE_PREFILL_2STAGE_LAUNCH(ELEM, 512, 16, 1) \
  DECLARE_MLA_SPARSE_PREFILL_2STAGE_LAUNCH(ELEM, 512, 32, 0) \
  DECLARE_MLA_SPARSE_PREFILL_2STAGE_LAUNCH(ELEM, 512, 32, 1) \
  DECLARE_MLA_SPARSE_PREFILL_2STAGE_LAUNCH(ELEM, 512, 64, 0) \
  DECLARE_MLA_SPARSE_PREFILL_2STAGE_LAUNCH(ELEM, 512, 64, 1) \
  DECLARE_MLA_SPARSE_PREFILL_2STAGE_LAUNCH(ELEM, 576, 8, 0)  \
  DECLARE_MLA_SPARSE_PREFILL_2STAGE_LAUNCH(ELEM, 576, 8, 1)  \
  DECLARE_MLA_SPARSE_PREFILL_2STAGE_LAUNCH(ELEM, 576, 16, 0) \
  DECLARE_MLA_SPARSE_PREFILL_2STAGE_LAUNCH(ELEM, 576, 16, 1) \
  DECLARE_MLA_SPARSE_PREFILL_2STAGE_LAUNCH(ELEM, 576, 32, 0) \
  DECLARE_MLA_SPARSE_PREFILL_2STAGE_LAUNCH(ELEM, 576, 32, 1) \
  DECLARE_MLA_SPARSE_PREFILL_2STAGE_LAUNCH(ELEM, 576, 64, 0) \
  DECLARE_MLA_SPARSE_PREFILL_2STAGE_LAUNCH(ELEM, 576, 64, 1)

DECLARE_MLA_SPARSE_PREFILL_2STAGE_ALL_B_H(half)
DECLARE_MLA_SPARSE_PREFILL_2STAGE_ALL_B_H(bf16)

#undef DECLARE_MLA_SPARSE_PREFILL_2STAGE_LAUNCH
#undef DECLARE_MLA_SPARSE_PREFILL_2STAGE_ALL_B_H

// Head-block (B_H) selection rule for the two-stage prefill path. B_H is the sparse
// analog of page size: it keys the per-(ELEM, B_H) launcher. Pure host logic
// (h_q -> B_H) with no CUTLASS dependency, so the op TU can pick the launcher without
// pulling in the heavy Stage-2 config header. Same rule as decode.
inline int sparse_mla_prefill_select_b_h(int h_q) {
  if (h_q <= 8) return 8;
  if (h_q <= 16) return 16;
  if (h_q <= 32) return 32;
  return 64;
}

}  // namespace mla_sparse_prefill
