/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
#pragma once

// MXFP4 dequantization fused into the E2M1 -> BF16 subgroup reorder.
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

CUTE_DEVICE float mxfp4_fold_multiplier(uint8_t e8m0) {
  return sycl::bit_cast<float>((static_cast<uint32_t>(e8m0) + (126u - kFoldShift)) << 23);
}

template <class CTensor>
CUTE_DEVICE void mxfp4_unfold(CTensor& tCrC) {
  constexpr float unfold = kFoldShift == 64 ? 0x1p64f : 0x1p0f;
  static_assert(kFoldShift == 64, "unfold constant must match kFoldShift");
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < tCrC.size(); ++i) {
    tCrC(i) *= unfold;
  }
}

// Tail of CUTE_XE_REORDER_E2M1_BF16_SEQ with its constant conversion
// multiplier replaced by two per-lane, already-scaled multipliers.
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

template <ReorderKind Kind>
CUTE_DEVICE void
mxfp4_reorder_folded(intel::uchar4 const& src0, intel::ushort8& dst0, intel::vector_t<float, 2> const& muls) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_INTEL_TARGET)
  const uint32_t shifts = 0x0008000C;
  if constexpr (Kind == ReorderKind::UU) {
    asm("{\n"
        ".decl IN_UB v_type=G type=UB num_elts=64 alias=<%1,0>\n"
        ".decl OUT_UW v_type=G type=UW num_elts=128 alias=<%0,0>\n"
        ".decl SHIFTS v_type=G type=UW num_elts=2 alias=<%2,0>\n"
        "shl (M1_NM, 32) OUT_UW(0,0)<1> IN_UB(0,0)<1;2,0> SHIFTS(0,0)<0;2,1>\n"
        "shl (M1_NM, 32) OUT_UW(1,0)<1> IN_UB(0,16)<1;2,0> SHIFTS(0,0)<0;2,1>\n"
        "shl (M1_NM, 32) OUT_UW(2,0)<1> IN_UB(0,32)<1;2,0> SHIFTS(0,0)<0;2,1>\n"
        "shl (M1_NM, 32) OUT_UW(3,0)<1> IN_UB(0,48)<1;2,0> SHIFTS(0,0)<0;2,1>\n" MOE_E2M1_BF16_FOLDED_TAIL "}\n"
        : "=rw"(dst0)
        : "rw"(src0), "rw.u"(shifts), "rw"(muls));
  } else {
    static_assert(Kind == ReorderKind::VV, "unsupported E2M1 -> BF16 reorder kind");
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
#endif
}

template <class T>
struct reorder_kind_of;
template <ReorderKind K, class S, class D>
struct reorder_kind_of<Xe_Reorder<K, S, D>> {
  static constexpr ReorderKind value = K;
};

template <class SEngine, class SLayoutWI, class SLayout, class DEngine, class DLayoutWI, class DLayout, class MulOf>
CUTE_DEVICE void mxfp4_reorder_dequant(
    SubgroupTensor<SEngine, SLayoutWI, SLayout> const& src,
    SubgroupTensor<DEngine, DLayoutWI, DLayout>& dst,
    MulOf const& mul_of) {
  using SType = typename SEngine::element_type;
  using DType = typename DEngine::element_type;
  static_assert(is_same_v<SType, float_e2m1_t> && is_same_v<DType, bfloat16_t>, "folded dequant is E2M1 -> BF16 only");
  static_assert(size(DLayoutWI{}) == size(SLayoutWI{}), "broadcasting reorders are not folded");

  using SL0 = decltype(cute::detail::subbyte_sg_tv_swizzle<SType>(project_strides(SLayout{})));
  using DL0 = decltype(cute::detail::subbyte_sg_tv_swizzle<DType>(project_strides(DLayout{})));
  using Atom = decltype(choose_xe_reorder_impl<SType, DType>(SL0{}, DL0{}));
  using RegTypeSrc = typename remove_extent<typename Atom::SRegisters>::type;
  using RegTypeDst = typename remove_extent<typename Atom::DRegisters>::type;
  static constexpr int values = size(SL0{}) / size<0>(SL0{});
  static constexpr int vchunk = sizeof_bits_v<typename Atom::SRegisters> / sizeof_bits_v<SType>;
  using RLayout = decltype(coalesce(composition(right_inverse(DL0{}), SL0{})));
  using VRLayout = decltype(composition(
      composition(Layout<Shape<intel::_SGSize, Int<values>>, Stride<_0, _1>>{}, RLayout{}),
      Layout<Shape<_1, Int<values>>, Stride<_0, intel::_SGSize>>{}));

  for_each(make_int_sequence<values / vchunk>{}, [&](auto ci) {
    constexpr int sv = decltype(ci)::value * vchunk;
    constexpr int dv = VRLayout{}(sv);
    static_assert(dv % 2 == 0, "a reorder chunk must start on an even value index");
    auto pS = recast_ptr<RegTypeSrc>(src.data() + sv);
    auto pD = recast_ptr<RegTypeDst>(dst.data() + dv);
    mxfp4_reorder_folded<reorder_kind_of<Atom>::value>(*pS, *pD, mul_of(Int<dv>{}));
  });
}
}  // namespace moe_w4a16
