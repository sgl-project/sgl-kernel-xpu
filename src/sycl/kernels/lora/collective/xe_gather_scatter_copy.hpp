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
  \brief Sub-group block read / write of 16-bit rows, used by the gather / scatter collectives
         of the chunked-SGMV LoRA grouped GEMM (xe_gather_a_group_mma.hpp, xe_scatter_group_epilogue.hpp).

  The gathered A rows and the scattered output rows have arbitrary (permuted) addresses, so
  they cannot be one block-2D surface; each row is one sub-group block message instead. In the
  DPAS register layout a row is spread one element per lane, which is exactly the lane order of
  a sub-group block read / write: lane l gets elements l, 16 + l, ..., 16 * (V - 1) + l of the
  16 * V contiguous elements at p. p must be 16-byte aligned.
*/

#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

namespace cutlass::lora::collective {

template <int V>
inline sycl::vec<uint16_t, V> sg_block_read(const uint16_t* p) {
  sycl::vec<uint16_t, V> r;
#ifdef __SYCL_DEVICE_ONLY__
  // p always points into a torch XPU device allocation (global memory), so the generic ->
  // global cast is valid; sycl::address_space_cast would add a runtime address-space test.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsycl-strict"
  auto gp = (const __attribute__((opencl_global)) uint16_t*)p;
#pragma clang diagnostic pop
  if constexpr (V == 1) {
    r[0] = __spirv_SubgroupBlockReadINTEL<uint16_t>(gp);
  } else {
    typedef uint16_t VT __attribute__((ext_vector_type(V)));
    VT v = __spirv_SubgroupBlockReadINTEL<VT>(gp);
#pragma unroll
    for (int i = 0; i < V; ++i) {
      r[i] = v[i];
    }
  }
#else
  (void)p;  // host pass: the body is device-only
#endif
  return r;
}

template <int V>
inline void sg_block_write(uint16_t* p, const sycl::vec<uint16_t, V>& r) {
#ifdef __SYCL_DEVICE_ONLY__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsycl-strict"
  auto gp = (__attribute__((opencl_global)) uint16_t*)p;
#pragma clang diagnostic pop
  if constexpr (V == 1) {
    __spirv_SubgroupBlockWriteINTEL(gp, static_cast<uint16_t>(r[0]));
  } else {
    typedef uint16_t VT __attribute__((ext_vector_type(V)));
    VT v;
#pragma unroll
    for (int i = 0; i < V; ++i) {
      v[i] = r[i];
    }
    __spirv_SubgroupBlockWriteINTEL(gp, v);
  }
#else
  (void)p;  // host pass: the body is device-only
  (void)r;
#endif
}

inline bool aligned16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 15) == 0;
}

inline bool aligned64(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 63) == 0;
}

// Whether sycl-tla's 2D block loads can read a stack of row-major [rows, row_len] 16-bit
// matrices (one per adapter) starting at `base`: a 64-byte aligned surface base for every
// matrix and a pitch (row_len * 2 bytes) that is a multiple of 16 bytes. On BMG, pitches that
// are only a multiple of 4 (what cute asserts) return wrong data, e.g. 24 / 60 bytes.
inline bool block2d_rows_ok(const void* base, int64_t rows, int64_t row_len) {
  return aligned64(base) && row_len % 8 == 0 && (rows * row_len * 2) % 64 == 0;
}

}  // namespace cutlass::lora::collective
