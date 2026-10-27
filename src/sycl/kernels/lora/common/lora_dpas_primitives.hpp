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
  \brief Shared building blocks of the gather-fused DPAS LoRA kernels.

  Used by the chunked-SGMV shrink / expand decode kernels
  (device/chunked_sgmv_lora_{shrink,expand}_fwd_kernel.hpp):

    * Bf16Elem / HalfElem   -- 16-bit element traits + the matching DPAS builtin
    * dpas_8x16x16          -- SIMD16 8x16x16 DPAS, fp32 accumulate
    * sg_block_read / write -- sub-group block messages (16-byte aligned addresses)
    * locate_row_tile       -- device-side row tile -> (segment, tile) scheduler
    * load_k_block_* / mma_k_block -- one k-block of A / B operands and its DPAS steps

  Operand layouts (per lane, SIMD16):
    A: element i = row i,  lane l = column k0 + l       (a block read of one row)
    B: element j = packed (k0 + 2j, k0 + 2j + 1) of column n = lane (VNNI)
    D: element i = row i,  lane l = output column n0 + l (a block write of one row)
*/

#pragma once

#include <ATen/ATen.h>

#include <cstdint>
#include <sycl/sycl.hpp>
#include <type_traits>

#ifdef __SYCL_DEVICE_ONLY__
namespace lora_dpas::native {
typedef short short8 __attribute__((ext_vector_type(8)));
typedef int int8 __attribute__((ext_vector_type(8)));
typedef float float8 __attribute__((ext_vector_type(8)));
}  // namespace lora_dpas::native

SYCL_EXTERNAL extern "C" lora_dpas::native::float8 __builtin_IB_sub_group16_fdpas_f_f_bf_bf_8_8(
    lora_dpas::native::float8 acc, lora_dpas::native::short8 a, lora_dpas::native::int8 b);
SYCL_EXTERNAL extern "C" lora_dpas::native::float8 __builtin_IB_sub_group16_fdpas_f_f_hf_hf_8_8(
    lora_dpas::native::float8 acc, lora_dpas::native::short8 a, lora_dpas::native::int8 b);
#endif

namespace lora_dpas {

// Per-lane DPAS operands for one SIMD16 8x16x16 step (16-bit inputs, fp32 accumulate):
//   A: element i = row i, lane l = column k0 + l
//   B: element j = packed (k0 + 2j, k0 + 2j + 1) of column n = lane
//   D: element i = row i, lane l = column n0 + l
using short8_t = sycl::vec<short, 8>;
using int8_t_ = sycl::vec<int, 8>;
using float8_t = sycl::vec<float, 8>;

#ifdef __SYCL_DEVICE_ONLY__
template <bool IsBf16>
inline float8_t dpas_8x16x16(const float8_t& acc, const short8_t& a, const int8_t_& b) {
  native::float8 c;
  native::short8 av;
  native::int8 bv;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    c[i] = acc[i];
    av[i] = a[i];
    bv[i] = b[i];
  }
  native::float8 d;
  if constexpr (IsBf16) {
    d = __builtin_IB_sub_group16_fdpas_f_f_bf_bf_8_8(c, av, bv);
  } else {
    d = __builtin_IB_sub_group16_fdpas_f_f_hf_hf_8_8(c, av, bv);
  }
  float8_t r;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    r[i] = d[i];
  }
  return r;
}
#else
template <bool IsBf16>
inline float8_t dpas_8x16x16(const float8_t& acc, const short8_t&, const int8_t_&) {
  return acc;
}
#endif

//----------------- Element-type traits (raw 16-bit storage) --------------------//

struct Bf16Elem {
  static inline float to_float(uint16_t v) {
    return sycl::bit_cast<float>(static_cast<uint32_t>(v) << 16);
  }
  static inline uint16_t from_float(float f) {
    return sycl::bit_cast<uint16_t>(sycl::ext::oneapi::bfloat16(f));
  }
  static inline float8_t dpas(const float8_t& acc, const short8_t& a, const int8_t_& b) {
    return dpas_8x16x16<true>(acc, a, b);
  }
};

struct HalfElem {
  static inline float to_float(uint16_t v) {
    return static_cast<float>(sycl::bit_cast<sycl::half>(v));
  }
  static inline uint16_t from_float(float f) {
    return sycl::bit_cast<uint16_t>(static_cast<sycl::half>(f));
  }
  static inline float8_t dpas(const float8_t& acc, const short8_t& a, const int8_t_& b) {
    return dpas_8x16x16<false>(acc, a, b);
  }
};

//----------------- Sub-group block read / write of 16-bit elements --------------------//
// Lane l receives / provides elements l, 16 + l, ..., 16 * (V - 1) + l of the
// 16 * V contiguous elements at p. p must be 16-byte aligned.

template <int V>
inline sycl::vec<uint16_t, V> sg_block_read(const uint16_t* p) {
  sycl::vec<uint16_t, V> r;
#ifdef __SYCL_DEVICE_ONLY__
  // p always points into a torch XPU device allocation (global memory), so the
  // generic -> global cast is valid. sycl::address_space_cast would add a runtime
  // address-space test to the innermost loop, so the -Wsycl-strict note is silenced.
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
  // p always points into a torch XPU device allocation (global memory), so the
  // generic -> global cast is valid. sycl::address_space_cast would add a runtime
  // address-space test to the innermost loop, so the -Wsycl-strict note is silenced.
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

//----------------- Shared tile helpers --------------------//

inline bool aligned16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 15) == 0;
}

// Map a flat row-tile index onto (segment, tile-within-segment) by a sub-group
// prefix scan over the BM-row tile counts of the segments. Returns false when
// `tile` lies past the last segment's tiles (surplus work-group). The result is
// uniform across the sub-group.
template <int BM>
inline bool locate_row_tile(
    const sycl::sub_group& sg, const int32_t* seg_indptr, int num_segments, int tile, int& seg, int& seg_tile) {
  const int lane = sg.get_local_linear_id();
  for (int s0 = 0; s0 < num_segments; s0 += 16) {
    const int s = s0 + lane;
    const int len = s < num_segments ? seg_indptr[s + 1] - seg_indptr[s] : 0;
    const int nt = (len + BM - 1) / BM;
    const int incl = sycl::inclusive_scan_over_group(sg, nt, sycl::plus<int>());
    const int total = sycl::group_broadcast(sg, incl, 15);
    if (tile < total) {
      const bool hit = tile < incl && tile >= incl - nt;
      const int hit_lane = sycl::reduce_over_group(sg, hit ? lane : 16, sycl::minimum<int>());
      seg = s0 + hit_lane;
      seg_tile = tile - sycl::group_broadcast(sg, incl - nt, hit_lane);
      return true;
    }
    tile -= total;
  }
  return false;
}

// Load KB k-elements for NT weight-column tiles: lane n of tile t reads KB
// contiguous elements of its weight row (VNNI-packed B operand, KB / 2 dwords).
template <int NT, int KB>
inline void load_b_tiles(const uint16_t* const (&brow)[NT], int k0, uint32_t (&b)[NT][KB / 2]) {
#pragma unroll
  for (int t = 0; t < NT; ++t) {
    const auto* p = reinterpret_cast<const sycl::vec<uint32_t, 4>*>(brow[t] + k0);
#pragma unroll
    for (int q = 0; q < KB / 8; ++q) {
      const auto v = p[q];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        b[t][q * 4 + e] = v[e];
      }
    }
  }
}

// Alignment-free, K-masked variants of the A / B loads: plain per-lane loads
// with zero fill for k >= K. Used for the K-tail block and for the unaligned
// kernel instantiation; zero-padded k columns contribute nothing to the DPAS.
template <int KS>
inline sycl::vec<uint16_t, KS> load_a_row_masked(const uint16_t* row, int k0, int K, int lane) {
  sycl::vec<uint16_t, KS> r;
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) {
    const int k = k0 + ks * 16 + lane;
    r[ks] = k < K ? row[k] : uint16_t(0);
  }
  return r;
}

template <int NT, int KB>
inline void load_b_tiles_masked(const uint16_t* const (&brow)[NT], int k0, int K, uint32_t (&b)[NT][KB / 2]) {
#pragma unroll
  for (int t = 0; t < NT; ++t) {
#pragma unroll
    for (int j = 0; j < KB / 2; ++j) {
      const int k = k0 + 2 * j;
      const uint32_t lo = k < K ? brow[t][k] : 0u;
      const uint32_t hi = k + 1 < K ? brow[t][k + 1] : 0u;
      b[t][j] = lo | (hi << 16);
    }
  }
}

// Fast load of one full KB-wide k-block: block reads of the (16-byte aligned) A
// rows and 16-byte vector loads of the B tiles.
template <int BM, int NT, int KB>
inline void load_k_block_fast(
    const uint16_t* const (&arow)[BM],
    const uint16_t* const (&brow)[NT],
    int k0,
    sycl::vec<uint16_t, KB / 16> (&a)[BM],
    uint32_t (&b)[NT][KB / 2]) {
#pragma unroll
  for (int r = 0; r < BM; ++r) {
    a[r] = sg_block_read<KB / 16>(arow[r] + k0);
  }
  load_b_tiles<NT, KB>(brow, k0, b);
}

// Masked load of one k-block (K tail, or any block when rows are unaligned).
template <int BM, int NT, int KB>
inline void load_k_block_masked(
    const uint16_t* const (&arow)[BM],
    const uint16_t* const (&brow)[NT],
    int k0,
    int K,
    int lane,
    sycl::vec<uint16_t, KB / 16> (&a)[BM],
    uint32_t (&b)[NT][KB / 2]) {
#pragma unroll
  for (int r = 0; r < BM; ++r) {
    a[r] = load_a_row_masked<KB / 16>(arow[r], k0, K, lane);
  }
  load_b_tiles_masked<NT, KB>(brow, k0, K, b);
}

// acc[m][t] += A(rows 8m..8m+7, k-block) x B(tile t, k-block), KB / 16 DPAS steps.
template <typename Elem, int MB, int NT, int KB>
inline void
mma_k_block(const sycl::vec<uint16_t, KB / 16> (&a)[8 * MB], const uint32_t (&b)[NT][KB / 2], float8_t (&acc)[MB][NT]) {
#pragma unroll
  for (int ks = 0; ks < KB / 16; ++ks) {
#pragma unroll
    for (int m = 0; m < MB; ++m) {
      short8_t av;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        av[i] = static_cast<short>(a[m * 8 + i][ks]);
      }
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        int8_t_ bv;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          bv[j] = static_cast<int>(b[t][ks * 8 + j]);
        }
        acc[m][t] = Elem::dpas(acc[m][t], av, bv);
      }
    }
  }
}
// Element traits for a torch storage type (at::BFloat16 / at::Half).
template <typename T>
struct ElemForTorch;
template <>
struct ElemForTorch<at::BFloat16> {
  using type = Bf16Elem;
};
template <>
struct ElemForTorch<at::Half> {
  using type = HalfElem;
};
template <typename T>
using ElemFor = typename ElemForTorch<T>::type;

}  // namespace lora_dpas
