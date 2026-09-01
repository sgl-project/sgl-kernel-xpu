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

// MXFP4 dequantization fused into the E2M1 -> BF16 subgroup reorder.
//
// cute's E2M1 -> BF16 reorder ends in one multiply per output register, by the
// constant 2^126 that compensates where the 4-bit exponent lands after the bit
// shuffling. Applying the E8M0 group scale afterwards costs much more than that
// multiply: the scale of a value depends on which of two columns it belongs to,
// which alternates between neighbouring values inside a work-item, so the scaling
// pass walks the fragment one *element* at a time -- and because each element is
// a separate variable, each one is a copy out of the fragment, a 16-wide
// multiply, and a copy back.
//
// Folding the scale into the conversion's own multiply removes all of it. The
// alternating columns cost execution size (8 sixteen-wide multiplies per chunk
// instead of 4 thirty-two-wide ones) but nothing else: no extra instructions
// beyond the conversion's own, and no copies, since the multiplies address the
// fragment in place with a stride of two.
//
// The folded multiplier is 2^(126 - kFoldShift) * scale, so that it stays inside
// the f32 range for E8M0 exponents up to 192 instead of only 128; the resulting
// 2^-kFoldShift on every product is undone once on the accumulator after the
// k-loop.

#include <cute/algorithm/reorder.hpp>
#include <cute/tensor.hpp>
#include <cute/util/compat.hpp>
#include <sycl/sycl.hpp>

#include "cutlass/cutlass.h"

namespace moe_w4a16 {

using namespace cute;

// Exponent headroom kept out of the folded multiplier, and undone on the
// accumulator. Covers E8M0 exponents in [0, 192], i.e. scales up to 2^65.
static constexpr int kFoldShift = 64;

// f32 multiplier that converts *and* scales in one step, from an E8M0 byte.
//
// The parameter is uint32_t on purpose. The gather that feeds it is a
// `load.ugm.d8u32`, which already zero-extends every byte into a dword, so a
// uint8_t parameter makes IGC narrow the loaded dword back to a packed byte
// register and re-widen it -- two exec-16 movs per gather that buy nothing, in a
// loop where 19 slots is the whole margin between ALU-bound and DPAS-bound.
CUTE_DEVICE float mxfp4_fold_multiplier(uint32_t e8m0) {
  return sycl::bit_cast<float>((e8m0 + (126u - kFoldShift)) << 23);
}

// Undo kFoldShift on an accumulator fragment.
template <class CTensor>
CUTE_DEVICE void mxfp4_unfold(CTensor& tCrC) {
  constexpr float unfold = kFoldShift == 64 ? 0x1p64f : 0x1p0f;
  static_assert(kFoldShift == 64, "unfold constant must match kFoldShift");
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < tCrC.size(); ++i) {
    tCrC(i) *= unfold;
  }
}

// Tail of the E2M1 -> BF16 sequence with the group scale folded in: identical to
// CUTE_XE_REORDER_E2M1_BF16_SEQ except that the conversion's constant multiplier
// becomes the pair of per-lane multipliers in %3.
//
// A fragment register holds one value index of every work-item before it holds
// the next (element = 16 * value + work_item), and a work-item's neighbouring
// values alternate between the two columns it covers. So the two columns' group
// scales split a register into its two contiguous halves -- which is exactly the
// register layout of a per-lane float pair, and one multiply per register still
// covers everything.
#define MOE_E2M1_BF16_FOLDED_TAIL                                          \
  ".decl OUT_W v_type=G type=W num_elts=128 alias=<%0,0>\n"                \
  ".decl OUT_UD v_type=G type=UD num_elts=64 alias=<%0,0>\n"               \
  ".decl OUT_BF v_type=G type=BF num_elts=128 alias=<%0,0>\n"              \
  ".decl MULS_F v_type=G type=F num_elts=32 alias=<%3,0>\n"                \
  "asr (M1_NM, 32) OUT_W(0,0)<1> OUT_W(0,0)<1;1,0> 6:uw\n"                 \
  "asr (M1_NM, 32) OUT_W(1,0)<1> OUT_W(1,0)<1;1,0> 6:uw\n"                 \
  "asr (M1_NM, 32) OUT_W(2,0)<1> OUT_W(2,0)<1;1,0> 6:uw\n"                 \
  "asr (M1_NM, 32) OUT_W(3,0)<1> OUT_W(3,0)<1;1,0> 6:uw\n"                 \
  "and (M1_NM, 32) OUT_UD(0,0)<1> OUT_UD(0,0)<1;1,0> 0x81C081C0:ud\n"      \
  "and (M1_NM, 32) OUT_UD(2,0)<1> OUT_UD(2,0)<1;1,0> 0x81C081C0:ud\n"      \
  "mul (M1_NM, 32) OUT_BF(0,0)<1> OUT_BF(0,0)<1;1,0> MULS_F(0,0)<1;1,0>\n" \
  "mul (M1_NM, 32) OUT_BF(1,0)<1> OUT_BF(1,0)<1;1,0> MULS_F(0,0)<1;1,0>\n" \
  "mul (M1_NM, 32) OUT_BF(2,0)<1> OUT_BF(2,0)<1;1,0> MULS_F(0,0)<1;1,0>\n" \
  "mul (M1_NM, 32) OUT_BF(3,0)<1> OUT_BF(3,0)<1;1,0> MULS_F(0,0)<1;1,0>\n"

// E2M1 -> BF16 reorder with the group scale folded into the conversion.
template <ReorderKind Kind>
CUTE_DEVICE void
mxfp4_reorder_folded(intel::uchar4 const& src0, intel::ushort8& dst0, intel::vector_t<float, 2> const& muls) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_INTEL_TARGET) && SYCL_INTEL_TARGET == 20
  const uint32_t shifts = 0x0008000C;
  if constexpr (Kind == ReorderKind::UU) {
    asm("{\n"
        ".decl IN_UB v_type=G type=UB num_elts=64 alias=<%1,0>\n"
        ".decl OUT_UW v_type=G type=UW num_elts=128 alias=<%0,0>\n"
        ".decl SHIFTS v_type=G type=UW num_elts=2 alias=<%2,0>\n"
        "shl (M1_NM, 32) OUT_UW(0,0)<1> IN_UB(0,0)<1;2,0>  SHIFTS(0,0)<0;2,1>\n"
        "shl (M1_NM, 32) OUT_UW(1,0)<1> IN_UB(0,16)<1;2,0> SHIFTS(0,0)<0;2,1>\n"
        "shl (M1_NM, 32) OUT_UW(2,0)<1> IN_UB(0,32)<1;2,0> SHIFTS(0,0)<0;2,1>\n"
        "shl (M1_NM, 32) OUT_UW(3,0)<1> IN_UB(0,48)<1;2,0> SHIFTS(0,0)<0;2,1>\n" MOE_E2M1_BF16_FOLDED_TAIL "}\n"
        : "=rw"(dst0)
        : "rw"(src0), "rw.u"(shifts), "rw"(muls));
  } else {
    static_assert(Kind == ReorderKind::VV, "unsupported E2M1 -> BF16 reorder kind for the folded dequant");
    asm("{\n"
        ".decl IN_UB v_type=G type=UB num_elts=64 alias=<%1,0>\n"
        ".decl OUT_UW v_type=G type=UW num_elts=128 alias=<%0,0>\n"
        ".decl SHIFTS v_type=G type=UW num_elts=2 alias=<%2,0>\n"
        "shl (M1_NM, 32) OUT_UW(0,0)<1> IN_UB(0,0)<4;2,0> SHIFTS(0,0)<0;2,1>\n"
        "shl (M1_NM, 32) OUT_UW(1,0)<1> IN_UB(0,1)<4;2,0> SHIFTS(0,0)<0;2,1>\n"
        "shl (M1_NM, 32) OUT_UW(2,0)<1> IN_UB(0,2)<4;2,0> SHIFTS(0,0)<0;2,1>\n"
        "shl (M1_NM, 32) OUT_UW(3,0)<1> IN_UB(0,3)<4;2,0> SHIFTS(0,0)<0;2,1>\n" MOE_E2M1_BF16_FOLDED_TAIL "}\n"
        : "=rw"(dst0)
        : "rw"(src0), "rw.u"(shifts), "rw"(muls));
  }
#elif defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_INTEL_TARGET)
#error "MXFP4 folded reorder is only supported on SYCL_INTEL_TARGET 20 (BMG/Xe20)"
#endif
}

template <class T>
struct reorder_kind_of;
template <ReorderKind K, class S, class D>
struct reorder_kind_of<Xe_Reorder<K, S, D>> {
  static constexpr ReorderKind value = K;
};

// Subgroup-cooperative E2M1 -> BF16 reorder that also applies the group scale.
// This is cute::reorder() / reorder_impl() with a folded-scale atom;
// mul_of(dst_offset) supplies a chunk's two per-lane multipliers.
template <class SEngine, class SLayoutWI, class SLayout, class DEngine, class DLayoutWI, class DLayout, class MulOf>
CUTE_DEVICE void mxfp4_reorder_dequant(
    SubgroupTensor<SEngine, SLayoutWI, SLayout> const& src,
    SubgroupTensor<DEngine, DLayoutWI, DLayout>& dst,
    MulOf const& mul_of) {
  using SType = typename SEngine::element_type;
  using DType = typename DEngine::element_type;
  static_assert(is_same_v<SType, float_e2m1_t> && is_same_v<DType, bfloat16_t>, "folded dequant is E2M1 -> BF16 only");
  static_assert(size(DLayoutWI{}) == size(SLayoutWI{}), "broadcasting reorders are not folded");

  using SL0 = decltype(detail::subbyte_sg_tv_swizzle<SType>(project_strides(SLayout{})));
  using DL0 = decltype(detail::subbyte_sg_tv_swizzle<DType>(project_strides(DLayout{})));
  using Atom = decltype(choose_xe_reorder_impl<SType, DType>(SL0{}, DL0{}));
  using RegTypeSrc = typename remove_extent<typename Atom::SRegisters>::type;
  using RegTypeDst = typename remove_extent<typename Atom::DRegisters>::type;

  static constexpr int values = size(SL0{}) / size<0>(SL0{});
  static constexpr int vchunk = sizeof_bits_v<typename Atom::SRegisters> / sizeof_bits_v<SType>;

  // src index -> dst index, then src value -> dst value (cute::reorder_impl).
  using RLayout = decltype(coalesce(composition(right_inverse(DL0{}), SL0{})));
  using VRLayout = decltype(composition(
      composition(Layout<Shape<intel::_SGSize, Int<values>>, Stride<_0, _1>>{}, RLayout{}),
      Layout<Shape<_1, Int<values>>, Stride<_0, intel::_SGSize>>{}));

  for_each(make_int_sequence<values / vchunk>{}, [&](auto ci) {
    constexpr int sv = decltype(ci)::value * vchunk;
    constexpr int dv = VRLayout{}(sv);
    // The two multipliers land on the chunk's even and odd elements, so a chunk
    // must start on an even value index for them to mean channel 0 and 1.
    static_assert(dv % 2 == 0, "a reorder chunk must start on an even value index");
    auto pS = recast_ptr<RegTypeSrc>(src.data() + sv);
    auto pD = recast_ptr<RegTypeDst>(dst.data() + dv);
    mxfp4_reorder_folded<reorder_kind_of<Atom>::value>(*pS, *pD, mul_of(Int<dv>{}));
  });
}

}  // namespace moe_w4a16
