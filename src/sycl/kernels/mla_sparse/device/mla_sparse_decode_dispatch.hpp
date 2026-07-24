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
// mla_sparse_decode_2stage_kernel.cpp.in, same per-dtype-TU split.
//
// Naming: launch_mla_sparse_decode_2stage_<ELEM_TAG>_<PAGE_SIZE>
#define DECLARE_MLA_SPARSE_DECODE_2STAGE_LAUNCH(ELEM)     \
  void launch_mla_sparse_decode_2stage_##ELEM##_128(      \
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

DECLARE_MLA_SPARSE_DECODE_2STAGE_LAUNCH(half)
DECLARE_MLA_SPARSE_DECODE_2STAGE_LAUNCH(bf16)

#undef DECLARE_MLA_SPARSE_DECODE_2STAGE_LAUNCH

}  // namespace mla_sparse_decode
