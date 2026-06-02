/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
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

#include "xe_fmha_fwd_decode_runner.hpp"

namespace decode {

// Struct functor declarations for FMHA decode kernel launchers.
// Each template specialization is explicitly instantiated in a separate
// generated .cpp file (from xe_fmha_fwd_decode_kernel.cpp.in /
// xe_fmha_fwd_split_decode_kernel.cpp.in).
//
// Naming: launch_fmha_decode_<QG_SZ>_<HEAD_DIM>_<PAGE_SIZE>
// Parameters:
//   QG_SZ    in {1, 2, 4, 8, 16, 32}
//   HEAD_DIM in {64, 72, 96, 128, 192, 256, 512}
//   PAGE_SIZE in {32, 64, 128}  (with NUM_SG = PAGE_SIZE / 16)

// Explicit instantiation declarations — tell the compiler these are compiled
// in separate translation units (generated from the .cpp.in templates).

#define EXTERN_FMHA_DECODE_RUNNER(QG, HD, PS) extern template struct FmhaDecodeRunner<QG, HD, PS>;

#define EXTERN_FMHA_SPLIT_DECODE_RUNNER(QG, HD, PS) extern template struct FmhaSplitDecodeRunner<QG, HD, PS>;

#define EXTERN_FMHA_DECODE_RUNNER_ALL_PAGE_SIZES(QG, HD) \
  EXTERN_FMHA_DECODE_RUNNER(QG, HD, 64)                  \
  EXTERN_FMHA_DECODE_RUNNER(QG, HD, 128)

#define EXTERN_FMHA_SPLIT_DECODE_RUNNER_ALL_PAGE_SIZES(QG, HD) \
  EXTERN_FMHA_SPLIT_DECODE_RUNNER(QG, HD, 64)                  \
  EXTERN_FMHA_SPLIT_DECODE_RUNNER(QG, HD, 128)

#define EXTERN_FMHA_DECODE_RUNNER_ALL_QG(HD)      \
  EXTERN_FMHA_DECODE_RUNNER_ALL_PAGE_SIZES(1, HD) \
  EXTERN_FMHA_DECODE_RUNNER_ALL_PAGE_SIZES(2, HD) \
  EXTERN_FMHA_DECODE_RUNNER_ALL_PAGE_SIZES(4, HD) \
  EXTERN_FMHA_DECODE_RUNNER_ALL_PAGE_SIZES(8, HD) \
  EXTERN_FMHA_DECODE_RUNNER_ALL_PAGE_SIZES(16, HD)

#define EXTERN_FMHA_SPLIT_DECODE_RUNNER_ALL_QG(HD)      \
  EXTERN_FMHA_SPLIT_DECODE_RUNNER_ALL_PAGE_SIZES(1, HD) \
  EXTERN_FMHA_SPLIT_DECODE_RUNNER_ALL_PAGE_SIZES(2, HD) \
  EXTERN_FMHA_SPLIT_DECODE_RUNNER_ALL_PAGE_SIZES(4, HD) \
  EXTERN_FMHA_SPLIT_DECODE_RUNNER_ALL_PAGE_SIZES(8, HD) \
  EXTERN_FMHA_SPLIT_DECODE_RUNNER_ALL_PAGE_SIZES(16, HD)

EXTERN_FMHA_DECODE_RUNNER_ALL_QG(64)
EXTERN_FMHA_DECODE_RUNNER_ALL_QG(72)
EXTERN_FMHA_DECODE_RUNNER_ALL_QG(96)
EXTERN_FMHA_DECODE_RUNNER_ALL_QG(128)
EXTERN_FMHA_DECODE_RUNNER_ALL_QG(192)
EXTERN_FMHA_DECODE_RUNNER_ALL_QG(256)
EXTERN_FMHA_DECODE_RUNNER_ALL_QG(512)

EXTERN_FMHA_SPLIT_DECODE_RUNNER_ALL_QG(64)
EXTERN_FMHA_SPLIT_DECODE_RUNNER_ALL_QG(72)
EXTERN_FMHA_SPLIT_DECODE_RUNNER_ALL_QG(96)
EXTERN_FMHA_SPLIT_DECODE_RUNNER_ALL_QG(128)
EXTERN_FMHA_SPLIT_DECODE_RUNNER_ALL_QG(192)
EXTERN_FMHA_SPLIT_DECODE_RUNNER_ALL_QG(256)
EXTERN_FMHA_SPLIT_DECODE_RUNNER_ALL_QG(512)

#undef EXTERN_FMHA_DECODE_RUNNER
#undef EXTERN_FMHA_SPLIT_DECODE_RUNNER
#undef EXTERN_FMHA_DECODE_RUNNER_ALL_PAGE_SIZES
#undef EXTERN_FMHA_SPLIT_DECODE_RUNNER_ALL_PAGE_SIZES
#undef EXTERN_FMHA_DECODE_RUNNER_ALL_QG
#undef EXTERN_FMHA_SPLIT_DECODE_RUNNER_ALL_QG

// Dispatch macros following the GroupGemmXe20.cpp pattern.
// Directly call struct operator() - no function pointers.
// These expand inside decode::mha_fwd where a local `params` is in scope.

#define DISPATCH_DECODE_KERNEL(QG, HD, PS)         \
  do {                                             \
    if (params.use_split_kv) {                     \
      FmhaSplitDecodeRunner<QG, HD, PS>{}(params); \
    } else {                                       \
      FmhaDecodeRunner<QG, HD, PS>{}(params);      \
    }                                              \
  } while (0)

#define DISPATCH_DECODE_PAGE_SIZE(QG, HD)                                                     \
  do {                                                                                        \
    switch (params.page_size) {                                                               \
      case 64:                                                                                \
        DISPATCH_DECODE_KERNEL(QG, HD, 64);                                                   \
        break;                                                                                \
      case 128:                                                                               \
        DISPATCH_DECODE_KERNEL(QG, HD, 128);                                                  \
        break;                                                                                \
      default:                                                                                \
        TORCH_CHECK(false, "Unsupported page_size for decode attention: ", params.page_size); \
    }                                                                                         \
  } while (0)

#define DISPATCH_DECODE_HEAD_DIM(QG)                                                  \
  do {                                                                                \
    switch (params.d) {                                                               \
      case 64:                                                                        \
        DISPATCH_DECODE_PAGE_SIZE(QG, 64);                                            \
        break;                                                                        \
      case 72:                                                                        \
        DISPATCH_DECODE_PAGE_SIZE(QG, 72);                                            \
        break;                                                                        \
      case 96:                                                                        \
        DISPATCH_DECODE_PAGE_SIZE(QG, 96);                                            \
        break;                                                                        \
      case 128:                                                                       \
        DISPATCH_DECODE_PAGE_SIZE(QG, 128);                                           \
        break;                                                                        \
      case 192:                                                                       \
        DISPATCH_DECODE_PAGE_SIZE(QG, 192);                                           \
        break;                                                                        \
      case 256:                                                                       \
        DISPATCH_DECODE_PAGE_SIZE(QG, 256);                                           \
        break;                                                                        \
      case 512:                                                                       \
        DISPATCH_DECODE_PAGE_SIZE(QG, 512);                                           \
        break;                                                                        \
      default:                                                                        \
        TORCH_CHECK(false, "Unsupported head size for decode attention: ", params.d); \
    }                                                                                 \
  } while (0)

#define DISPATCH_DECODE(qg_sz)                                                                      \
  do {                                                                                              \
    switch (qg_sz) {                                                                                \
      case 1:                                                                                       \
        DISPATCH_DECODE_HEAD_DIM(1);                                                                \
        break;                                                                                      \
      case 2:                                                                                       \
        DISPATCH_DECODE_HEAD_DIM(2);                                                                \
        break;                                                                                      \
      case 4:                                                                                       \
        DISPATCH_DECODE_HEAD_DIM(4);                                                                \
        break;                                                                                      \
      case 8:                                                                                       \
        DISPATCH_DECODE_HEAD_DIM(8);                                                                \
        break;                                                                                      \
      case 16:                                                                                      \
        DISPATCH_DECODE_HEAD_DIM(16);                                                               \
        break;                                                                                      \
      default:                                                                                      \
        TORCH_CHECK(false, "Unsupported q_group_size for decode attention: ", params.q_group_size); \
    }                                                                                               \
  } while (0)

}  // namespace decode
