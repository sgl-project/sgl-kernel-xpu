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
  \brief Forward declarations for generated MLA decode kernel launch functions
*/

#pragma once

#include <ATen/ATen.h>

#include <optional>
#include <sycl/sycl.hpp>

namespace mla_decode {

// Each function is defined in a separate generated .cpp file from
// mla_decode_kernel.cpp.in, compiled as its own library.
//
// Naming: launch_mla_decode_<ELEM_TAG>_<PAGE_SIZE>_<HAS_LSE>
// Parameters:
//   ELEM_TAG  in {half, bf16}
//   PAGE_SIZE in {16, 32, 64, 128}
//   HAS_LSE   in {0, 1} -- 1 emits the softmax log-sum-exp into `lse`, 0 skips
//             it entirely (no LSE registers, no LSE stores, `lse` unread)
//
// The 16 symbols below are exactly the set MlaDecodeXe20.cmake generates and
// flash_mla_decode()'s dispatch ladder calls; the three must stay in lockstep or
// the TU fails to link.

#define DECLARE_MLA_DECODE_LAUNCH(ELEM, PS, HAS_LSE) \
  void launch_mla_decode_##ELEM##_##PS##_##HAS_LSE(  \
      at::Tensor& out,                               \
      const std::optional<at::Tensor>& lse,          \
      const at::Tensor& q_nope,                      \
      const at::Tensor& q_pe,                        \
      const at::Tensor& kv_c_and_k_pe_cache,         \
      const at::Tensor& seq_lens,                    \
      const at::Tensor& page_table,                  \
      at::Tensor& workspace,                         \
      double sm_scale,                               \
      int64_t num_kv_splits);

#define DECLARE_MLA_DECODE_ALL_PAGE_SIZES(ELEM) \
  DECLARE_MLA_DECODE_LAUNCH(ELEM, 16, 0)        \
  DECLARE_MLA_DECODE_LAUNCH(ELEM, 16, 1)        \
  DECLARE_MLA_DECODE_LAUNCH(ELEM, 32, 0)        \
  DECLARE_MLA_DECODE_LAUNCH(ELEM, 32, 1)        \
  DECLARE_MLA_DECODE_LAUNCH(ELEM, 64, 0)        \
  DECLARE_MLA_DECODE_LAUNCH(ELEM, 64, 1)        \
  DECLARE_MLA_DECODE_LAUNCH(ELEM, 128, 0)       \
  DECLARE_MLA_DECODE_LAUNCH(ELEM, 128, 1)

DECLARE_MLA_DECODE_ALL_PAGE_SIZES(half)
DECLARE_MLA_DECODE_ALL_PAGE_SIZES(bf16)

#undef DECLARE_MLA_DECODE_LAUNCH
#undef DECLARE_MLA_DECODE_ALL_PAGE_SIZES

}  // namespace mla_decode
