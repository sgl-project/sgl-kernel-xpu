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

namespace decode {

struct Arguments;

// Declarations for generated FMHA decode kernel launch functions.
// Each function is defined in a separate generated .cpp file from
// xe_fmha_fwd_decode_kernel.cpp.in, compiled as its own library.
//
// Naming: launch_fmha_decode_<QG_SZ>_<HEAD_DIM>_<PAGE_SIZE>
// Parameters:
//   QG_SZ    in {1, 2, 4, 8, 16, 32}
//   HEAD_DIM in {64, 96, 128, 192}
//   PAGE_SIZE in {32, 64, 128}  (with NUM_SG = PAGE_SIZE / 16)

#define DECLARE_LAUNCH_FMHA_DECODE(QG, HD, PS) \
  void launch_fmha_decode_##QG##_##HD##_##PS(bool use_sink, const Arguments& params);

#define DECLARE_LAUNCH_FMHA_DECODE_ALL_PAGE_SIZES(QG, HD) \
  DECLARE_LAUNCH_FMHA_DECODE(QG, HD, 32)                  \
  DECLARE_LAUNCH_FMHA_DECODE(QG, HD, 64)                  \
  DECLARE_LAUNCH_FMHA_DECODE(QG, HD, 128)

#define DECLARE_LAUNCH_FMHA_DECODE_ALL_QG(HD)       \
  DECLARE_LAUNCH_FMHA_DECODE_ALL_PAGE_SIZES(1, HD)  \
  DECLARE_LAUNCH_FMHA_DECODE_ALL_PAGE_SIZES(2, HD)  \
  DECLARE_LAUNCH_FMHA_DECODE_ALL_PAGE_SIZES(4, HD)  \
  DECLARE_LAUNCH_FMHA_DECODE_ALL_PAGE_SIZES(8, HD)  \
  DECLARE_LAUNCH_FMHA_DECODE_ALL_PAGE_SIZES(16, HD) \
  DECLARE_LAUNCH_FMHA_DECODE_ALL_PAGE_SIZES(32, HD)

DECLARE_LAUNCH_FMHA_DECODE_ALL_QG(64)
DECLARE_LAUNCH_FMHA_DECODE_ALL_QG(96)
DECLARE_LAUNCH_FMHA_DECODE_ALL_QG(128)
DECLARE_LAUNCH_FMHA_DECODE_ALL_QG(192)

#undef DECLARE_LAUNCH_FMHA_DECODE
#undef DECLARE_LAUNCH_FMHA_DECODE_ALL_PAGE_SIZES
#undef DECLARE_LAUNCH_FMHA_DECODE_ALL_QG

}  // namespace decode
