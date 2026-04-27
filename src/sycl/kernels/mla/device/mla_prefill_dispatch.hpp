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
  \brief Forward declarations for generated MLA prefill kernel launch functions
*/

#pragma once

#include <ATen/ATen.h>

#include <sycl/sycl.hpp>

namespace mla_prefill {

// Each function is defined in a separate generated .cpp file from
// mla_prefill_kernel.cpp.in, compiled as its own library.
//
// Naming: launch_mla_prefill_<ELEM_TAG>_<PAGE_SIZE>
// Parameters:
//   ELEM_TAG  in {half, bf16}
//   PAGE_SIZE in {16, 32, 64, 128}

#define DECLARE_MLA_PREFILL_LAUNCH(ELEM, PS)  \
  void launch_mla_prefill_##ELEM##_##PS(      \
      at::Tensor& out,                        \
      const at::Tensor& q_nope,               \
      const at::Tensor& q_pe,                 \
      const at::Tensor& kv_c_and_k_pe_cache,  \
      const at::Tensor& cu_seqlens_q,         \
      const at::Tensor& seq_lens,             \
      int64_t max_seqlen_q,                   \
      const at::Tensor& page_table,           \
      at::Tensor& workspace,                  \
      double sm_scale,                        \
      bool causal,                            \
      int64_t num_kv_splits);

#define DECLARE_MLA_PREFILL_ALL_PAGE_SIZES(ELEM)  \
  DECLARE_MLA_PREFILL_LAUNCH(ELEM, 16)            \
  DECLARE_MLA_PREFILL_LAUNCH(ELEM, 32)            \
  DECLARE_MLA_PREFILL_LAUNCH(ELEM, 64)            \
  DECLARE_MLA_PREFILL_LAUNCH(ELEM, 128)

DECLARE_MLA_PREFILL_ALL_PAGE_SIZES(half)
DECLARE_MLA_PREFILL_ALL_PAGE_SIZES(bf16)

#undef DECLARE_MLA_PREFILL_LAUNCH
#undef DECLARE_MLA_PREFILL_ALL_PAGE_SIZES

}  // namespace mla_prefill
