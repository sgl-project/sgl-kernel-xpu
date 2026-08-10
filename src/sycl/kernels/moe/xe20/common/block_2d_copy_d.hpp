/***************************************************************************************************
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

// Explicit D-store atom selection for the Xe2 MoE GEMMs.
//
// Background (CUTLASS9-656).  cute's `make_block_2d_copy_D` sizes the D store
// from a *single MMA-C atom*.  The Xe DPAS N is hardware-fixed at 16, so for a
// 16-bit output that atom is 16 x 2B = 32B -- half of a 64B cache line -- even
// when the subgroup owns a contiguous 32- or 64-element run of N.  Each store
// then moves half a line, doubling the store message count for the same bytes.
//
// This header provides the explicit-op form: the store atom is chosen *here*,
// from the per-subgroup output tile, and handed to `make_block_2d_copy_CD`
// rather than left to the generic per-atom selector.  It keeps the widening
// local to these kernels, so it does not depend on the pinned sycl-tla revision
// having the upstream fix, and it leaves an override hook for tuning.
//
// Correctness prerequisite: a store wider than one MMA-C atom no longer matches
// the DPAS accumulator's register order, so the caller must bridge through the
// *copy's* fragment (`partition_sg_fragment_S` + `reorder`) and store to
// `partition_D`, not to `thr_mma.partition_C`.

#include "cute/tensor.hpp"

namespace moe_xe20 {
using namespace cute;

// Per-subgroup output tile (SG_M, SG_N) of a TiledMMA: the work-group tile
// divided by the (M,N) extent of the subgroup layout.
template <class TiledMMA>
CUTE_HOST_DEVICE constexpr auto sg_output_tile(TiledMMA const& mma) {
  auto thr_vmnk = mma.get_thr_layout_vmnk();  // (ThrV,ThrM,ThrN,ThrK)
  return shape_div(select<0, 1>(mma.tile_mnk()), make_shape(size<1>(shape(thr_vmnk)), size<2>(shape(thr_vmnk))));
}

// Widest block-2D store atom that a subgroup can legally use for a row-major D.
//
//   width  = gcd(64B / sizeof(ElementD), SG_N)   -- cap at one cache line
//   height = gcd(SG_M, 8)                        -- 8 is the store height limit
//
// The gcd keeps this self-limiting: when the subgroup owns only one atom's worth
// of N (e.g. WGTile N=64 over a 4-wide subgroup layout => SG_N=16) the result is
// identical to what the generic selector already picks, so narrow tiles are
// untouched.
template <class ElementD, class TiledMMA>
CUTE_HOST_DEVICE constexpr auto select_block_2d_store_D(TiledMMA const& mma) {
  constexpr int Bits = sizeof_bits_v<ElementD>;
  constexpr int MaxWidth = 64 * 8 / Bits;  // elements in a 64B cache line
  constexpr int MaxHeight = 8;             // hardware store height limit

  auto sg_tile = sg_output_tile(mma);
  constexpr int SgM = size<0>(decltype(sg_tile){});
  constexpr int SgN = size<1>(decltype(sg_tile){});

  return XE_STORE_2D<Bits, cute::gcd(SgM, MaxHeight), cute::gcd(SgN, MaxWidth)>{};
}

// D-store TiledCopy for the MoE GEMMs.
//
//   CopyOp = void  -> pick the widest legal atom for this subgroup tile
//                     (see select_block_2d_store_D)
//   CopyOp = XE_STORE_2D<...> -> use exactly that atom
//
// Both paths go through `make_block_2d_copy_CD`, i.e. the op is always explicit
// at this level; `void` only means "let this header choose" rather than "let the
// generic per-atom selector choose".
template <class CopyOp, class TiledMMA, class DTensor>
CUTE_HOST_DEVICE auto make_moe_block_2d_copy_D(TiledMMA const& mma, DTensor const& d) {
  if constexpr (!cute::is_void_v<CopyOp>) {
    return make_block_2d_copy_CD(CopyOp{}, mma, d);
  } else {
    using ElementD = typename DTensor::element_type;
    return make_block_2d_copy_CD(select_block_2d_store_D<ElementD>(mma), mma, d);
  }
}

}  // namespace moe_xe20
