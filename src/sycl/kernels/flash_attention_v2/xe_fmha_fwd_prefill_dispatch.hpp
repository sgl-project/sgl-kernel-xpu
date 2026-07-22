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
#pragma once

#include "xe_fmha_fwd_prefill_runner.hpp"

namespace prefill {

// Explicit instantiation declarations — tell the compiler these are compiled
// in separate translation units (generated from the .cpp.in templates).
//
// Parameters:
//   HEAD_DIM in {64, 72, 96, 128, 192, 256, 512}

#define EXTERN_FMHA_PREFILL_RUNNER(HD) extern template struct FmhaPrefillRunner<HD>;
#define EXTERN_FMHA_PREFILL_NP_RUNNER(HD) extern template struct FmhaPrefillNpRunner<HD>;

// Paged prefill runners: paged head dims only (64, 96, 128, 192, 256, 512).
// bf16 query only.
EXTERN_FMHA_PREFILL_RUNNER(64)
EXTERN_FMHA_PREFILL_RUNNER(96)
EXTERN_FMHA_PREFILL_RUNNER(128)
EXTERN_FMHA_PREFILL_RUNNER(192)
EXTERN_FMHA_PREFILL_RUNNER(256)
EXTERN_FMHA_PREFILL_RUNNER(512)

// Non-paged (no_page) prefill runners: np head dims only (64, 72, 96, 128, 192).
// bf16 query only.
EXTERN_FMHA_PREFILL_NP_RUNNER(64)
EXTERN_FMHA_PREFILL_NP_RUNNER(72)
EXTERN_FMHA_PREFILL_NP_RUNNER(96)
EXTERN_FMHA_PREFILL_NP_RUNNER(128)
EXTERN_FMHA_PREFILL_NP_RUNNER(192)

#undef EXTERN_FMHA_PREFILL_RUNNER
#undef EXTERN_FMHA_PREFILL_NP_RUNNER

// Dispatch macros following the same pattern as decode.
// Directly call struct operator() - no function pointers.
// Expands inside prefill::mha_fwd where a local `params` is in scope.
//
// Paged prefill supports bf16 query only. The KV layout is selected at runtime:
// fp8 KV cache (FmhaPrefillFp8Runner) vs 16-bit KV (FmhaPrefillRunner). Each is
// a separate translation unit / shared library.

#define DISPATCH_PREFILL_KERNEL(HD)                                             \
  do {                                                                          \
    TORCH_CHECK(params.is_bf16, "Prefill attention only supports bf16 query");  \
    if (params.is_e4m3 || params.is_e5m2) {                                     \
      FmhaPrefillFp8Runner<HD>{}(params);                                       \
    } else {                                                                    \
      FmhaPrefillRunner<HD>{}(params);                                          \
    }                                                                           \
  } while (0)

// Non-paged (no_page) prefill: bf16 query only (no fp8 KV cache).
#define DISPATCH_PREFILL_NOPAGE_KERNEL(HD)                                              \
  do {                                                                                  \
    TORCH_CHECK(params.is_bf16, "Non-paged prefill attention only supports bf16 query"); \
    FmhaPrefillNpRunner<HD>{}(params);                                                   \
  } while (0)

}  // namespace prefill
