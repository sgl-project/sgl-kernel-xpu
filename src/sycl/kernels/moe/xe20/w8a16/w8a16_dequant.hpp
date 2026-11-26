/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
#pragma once

// W8A16 (FP8 E4M3 and INT8/UINT8) dequantization fused into the 8-bit -> BF16
// subgroup reorder on Intel Xe2 (BMG).
//
// In Xe2 DPAS BLayout (BV = 2 for BF16), a 64-element (2-GRF) output chunk holds
// 4 BF16 values per work-item arranged as:
//   GRF 0: [lane 0..15 v=0 (col_even), lane 0..15 v=1 (col_odd)]
//   GRF 1: [lane 0..15 v=2 (col_even), lane 0..15 v=3 (col_odd)]
// Because `intel::vector_t<float, 2>` per lane also places `[lane 0..15 c=0,
// lane 0..15 c=1]` into a 32-float register pair, replacing the conversion's
// constant multiplier (`2^120` for E4M3, `2^8` for INT8/UINT8) with per-lane
// `MULS_F` applies distinct per-column scales in place at zero extra ALU cost
// and eliminates the secondary accumulator (`tCrC_group`).

#include <cute/algorithm/reorder.hpp>
#include <cute/tensor.hpp>
#include <cute/util/compat.hpp>
#include <sycl/sycl.hpp>

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/float8.h"

namespace moe_w8a16 {

using namespace cute;

static constexpr int kShiftUnset = -1000;

CUTE_DEVICE int fp8_required_shift_from_biased_exp(int exp_b) {
  if (exp_b < 10) return -64;
  if (exp_b <= 134) return 0;
  if (exp_b <= 198) return 64;
  return 120;
}

CUTE_DEVICE float fp8_shift_scale_factor(int shift_diff) {
  // Valid shift_diff = old_shift - new_shift (< 0) or final my_shift:
  // Differences between {-64, 0, 64, 120} are {-184, -128, -120, -64, -56, 0, 64, 120}.
  if (shift_diff == 0) return 1.0f;
  if (shift_diff <= -128) {
    // Split into two normal powers of two when shift_diff < -126 (-184 or -128).
    const uint32_t rem_bits = static_cast<uint32_t>(127 + (shift_diff + 64)) << 23;
    return sycl::bit_cast<float>(rem_bits) * 0x1p-64f;
  }
  const uint32_t bits = static_cast<uint32_t>(127 + shift_diff) << 23;
  return sycl::bit_cast<float>(bits);
}

CUTE_DEVICE float mxfp8_fold_multiplier(uint32_t e8m0, int shift) {
  if (e8m0 == 0xFFu) {
    return sycl::bit_cast<float>(0x7FC00000u);
  }
  const int fp32_exp = static_cast<int>(e8m0) + 120 - shift;
  uint32_t bits = 0u;
  if (fp32_exp >= 255) {
    bits = 0x7F800000u;
  } else if (fp32_exp > 0) {
    bits = static_cast<uint32_t>(fp32_exp) << 23;
  } else if (fp32_exp == 0) {
    bits = 1u << 22;
  }
  return sycl::bit_cast<float>(bits);
}

CUTE_DEVICE float fp32_e4m3_fold_multiplier(float scale, int shift) {
  if (shift == 0) {
    return scale * 0x1p120f;
  }
  if (shift == 64) {
    return scale * 0x1p56f;
  }
  if (shift == 120) {
    return scale;
  }
  // shift == -64: scale * 2^184 = (scale * 2^64) * 2^120
  return (scale * 0x1p64f) * 0x1p120f;
}

#define MOE_E4M3_BF16_FOLDED_TAIL                                          \
  ".decl OUT_W v_type=G type=W num_elts=64 alias=<%0,0>\n"                 \
  ".decl OUT_UD v_type=G type=UD num_elts=32 alias=<%0,0>\n"               \
  ".decl OUT_HF v_type=G type=HF num_elts=64 alias=<%0,0>\n"               \
  ".decl OUT_BF v_type=G type=BF num_elts=64 alias=<%0,0>\n"               \
  ".decl MULS_F v_type=G type=F num_elts=32 alias=<%2,0>\n"                \
  ".decl NZ_PRED0 v_type=P num_elts=32\n"                                  \
  ".decl NZ_PRED1 v_type=P num_elts=32\n"                                  \
  "asr (M1_NM, 32) OUT_W(0,0)<1> OUT_W(0,0)<1;1,0> 4:uw\n"                 \
  "asr (M1_NM, 32) OUT_W(1,0)<1> OUT_W(1,0)<1;1,0> 4:uw\n"                 \
  "and (M1_NM, 32) OUT_UD(0,0)<1> OUT_UD(0,0)<1;1,0> 0x87FF87FF:ud\n"      \
  "cmp.ge (M1_NM, 32) NZ_PRED0 (abs)OUT_HF(0,0)<1;1,0> 0x07F0:hf\n"        \
  "cmp.ge (M1_NM, 32) NZ_PRED1 (abs)OUT_HF(1,0)<1;1,0> 0x07F0:hf\n"        \
  "mul (M1_NM, 32) OUT_BF(0,0)<1> OUT_BF(0,0)<1;1,0> MULS_F(0,0)<1;1,0>\n" \
  "mul (M1_NM, 32) OUT_BF(1,0)<1> OUT_BF(1,0)<1;1,0> MULS_F(0,0)<1;1,0>\n" \
  "(NZ_PRED0) mov (M1_NM, 32) OUT_UW(0,0)<1> 0x7FC0:uw\n"                  \
  "(NZ_PRED1) mov (M1_NM, 32) OUT_UW(1,0)<1> 0x7FC0:uw\n"

template <ReorderKind Kind>
CUTE_DEVICE void
e4m3_reorder_folded(intel::uchar4 const& src0, intel::ushort4& dst0, intel::vector_t<float, 2> const& muls) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_INTEL_TARGET) && SYCL_INTEL_TARGET == 20
  if constexpr (Kind == ReorderKind::UU) {
    asm("{\n"
        ".decl IN_UB v_type=G type=UB num_elts=64 alias=<%1,0>\n"
        ".decl OUT_UW v_type=G type=UW num_elts=64 alias=<%0,0>\n"
        "shl (M1_NM, 32) OUT_UW(0,0)<1> IN_UB(0,0)<1;1,0> 8:uw\n"
        "shl (M1_NM, 32) OUT_UW(1,0)<1> IN_UB(0,32)<1;1,0> 8:uw\n" MOE_E4M3_BF16_FOLDED_TAIL "}\n"
        : "=rw"(dst0)
        : "rw"(src0), "rw"(muls));
  } else {
    static_assert(Kind == ReorderKind::VV, "unsupported E4M3 -> BF16 reorder kind for folded dequant");
    asm("{\n"
        ".decl IN_UB v_type=G type=UB num_elts=64 alias=<%1,0>\n"
        ".decl OUT_UW v_type=G type=UW num_elts=64 alias=<%0,0>\n"
        "shl (M1_NM, 32) OUT_UW(0,0)<1> IN_UB(0,0)<4;2,1> 8:uw\n"
        "shl (M1_NM, 32) OUT_UW(1,0)<1> IN_UB(0,2)<4;2,1> 8:uw\n" MOE_E4M3_BF16_FOLDED_TAIL "}\n"
        : "=rw"(dst0)
        : "rw"(src0), "rw"(muls));
  }
#elif defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_INTEL_TARGET)
#error "W8A16 folded reorder is only supported on SYCL_INTEL_TARGET 20 (BMG/Xe20)"
#endif
}

#define MOE_INT8_BF16_SYM_FOLDED_TAIL                                                                 \
  ".decl OUT_BF v_type=G type=BF num_elts=64 alias=<%0,0>\n"                                          \
  ".decl F_7E000000 v_type=G type=F num_elts=1 alias=<%2,0>\n"                                        \
  ".decl F_BF000000 v_type=G type=F num_elts=1 alias=<%3,0>\n"                                        \
  ".decl MULS_F v_type=G type=F num_elts=32 alias=<%4,0>\n"                                           \
  "mad (M1_NM, 32) OUT_BF(0,0)<1> F_7E000000(0,0)<0;1,0> OUT_BF(0,0)<1;1,0> F_BF000000(0,0)<0;1,0>\n" \
  "mad (M1_NM, 32) OUT_BF(1,0)<1> F_7E000000(0,0)<0;1,0> OUT_BF(1,0)<1;1,0> F_BF000000(0,0)<0;1,0>\n" \
  "mul (M1_NM, 32) OUT_BF(0,0)<1> OUT_BF(0,0)<1;1,0> MULS_F(0,0)<1;1,0>\n"                            \
  "mul (M1_NM, 32) OUT_BF(1,0)<1> OUT_BF(1,0)<1;1,0> MULS_F(0,0)<1;1,0>\n"

#define MOE_INT8_BF16_ZP_FOLDED_TAIL                                                            \
  ".decl OUT_BF v_type=G type=BF num_elts=64 alias=<%0,0>\n"                                    \
  ".decl F_7E000000 v_type=G type=F num_elts=1 alias=<%2,0>\n"                                  \
  ".decl ZP_F v_type=G type=F num_elts=32 alias=<%3,0>\n"                                       \
  ".decl MULS_F v_type=G type=F num_elts=32 alias=<%4,0>\n"                                     \
  "mad (M1_NM, 32) OUT_BF(0,0)<1> F_7E000000(0,0)<0;1,0> OUT_BF(0,0)<1;1,0> ZP_F(0,0)<1;1,0>\n" \
  "mad (M1_NM, 32) OUT_BF(1,0)<1> F_7E000000(0,0)<0;1,0> OUT_BF(1,0)<1;1,0> ZP_F(0,0)<1;1,0>\n" \
  "mul (M1_NM, 32) OUT_BF(0,0)<1> OUT_BF(0,0)<1;1,0> MULS_F(0,0)<1;1,0>\n"                      \
  "mul (M1_NM, 32) OUT_BF(1,0)<1> OUT_BF(1,0)<1;1,0> MULS_F(0,0)<1;1,0>\n"

#define MOE_UINT8B128_BF16_UNSCALED_TAIL                                                              \
  ".decl OUT_HF v_type=G type=HF num_elts=64 alias=<%0,0>\n"                                          \
  ".decl OUT_BF v_type=G type=BF num_elts=64 alias=<%0,0>\n"                                          \
  ".decl F_7E000000 v_type=G type=F num_elts=1 alias=<%2,0>\n"                                        \
  ".decl F_BF000000 v_type=G type=F num_elts=1 alias=<%3,0>\n"                                        \
  "mad (M1_NM, 32) OUT_BF(0,0)<1> F_7E000000(0,0)<0;1,0> OUT_BF(0,0)<1;1,0> F_BF000000(0,0)<0;1,0>\n" \
  "mad (M1_NM, 32) OUT_BF(1,0)<1> F_7E000000(0,0)<0;1,0> OUT_BF(1,0)<1;1,0> F_BF000000(0,0)<0;1,0>\n" \
  "mul (M1_NM, 32) OUT_HF(0,0)<1> OUT_HF(0,0)<1;1,0> 0x4000:hf\n"                                     \
  "mul (M1_NM, 32) OUT_HF(1,0)<1> OUT_HF(1,0)<1;1,0> 0x4000:hf\n"

template <ReorderKind Kind>
CUTE_DEVICE void int8_sym_reorder_folded(
    intel::uchar4 const& src0, intel::ushort4& dst0, intel::vector_t<float, 2> const& muls, bool is_signed_int8) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_INTEL_TARGET) && SYCL_INTEL_TARGET == 20
  const uint32_t scale_c = 0x7E000000u;
  const uint32_t shift_c = 0xBF000000u;
  const uint16_t xor_mask = is_signed_int8 ? 0x0080u : 0x0000u;
  if constexpr (Kind == ReorderKind::UU) {
    asm("{\n"
        ".decl IN_UB v_type=G type=UB num_elts=64 alias=<%1,0>\n"
        ".decl OUT_UW v_type=G type=UW num_elts=64 alias=<%0,0>\n"
        ".decl XOR_UW v_type=G type=UW num_elts=1 alias=<%5,0>\n"
        "xor (M1_NM, 32) OUT_UW(0,0)<1> IN_UB(0,0)<1;1,0> XOR_UW(0,0)<0;1,0>\n"
        "xor (M1_NM, 32) OUT_UW(1,0)<1> IN_UB(0,32)<1;1,0> XOR_UW(0,0)<0;1,0>\n" MOE_INT8_BF16_SYM_FOLDED_TAIL "}\n"
        : "=rw"(dst0)
        : "rw"(src0), "rw.u"(scale_c), "rw.u"(shift_c), "rw"(muls), "rw.u"(xor_mask));
  } else {
    static_assert(Kind == ReorderKind::VV, "unsupported INT8 -> BF16 reorder kind for folded dequant");
    asm("{\n"
        ".decl IN_UB v_type=G type=UB num_elts=64 alias=<%1,0>\n"
        ".decl OUT_UW v_type=G type=UW num_elts=64 alias=<%0,0>\n"
        ".decl XOR_UW v_type=G type=UW num_elts=1 alias=<%5,0>\n"
        "xor (M1_NM, 32) OUT_UW(0,0)<1> IN_UB(0,0)<4;2,1> XOR_UW(0,0)<0;1,0>\n"
        "xor (M1_NM, 32) OUT_UW(1,0)<1> IN_UB(0,2)<4;2,1> XOR_UW(0,0)<0;1,0>\n" MOE_INT8_BF16_SYM_FOLDED_TAIL "}\n"
        : "=rw"(dst0)
        : "rw"(src0), "rw.u"(scale_c), "rw.u"(shift_c), "rw"(muls), "rw.u"(xor_mask));
  }
#elif defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_INTEL_TARGET)
#error "W8A16 folded reorder is only supported on SYCL_INTEL_TARGET 20 (BMG/Xe20)"
#endif
}

template <ReorderKind Kind>
CUTE_DEVICE void uint8_zp_reorder_folded(
    intel::uchar4 const& src0,
    intel::ushort4& dst0,
    intel::vector_t<float, 2> const& zp_shifts,
    intel::vector_t<float, 2> const& muls) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_INTEL_TARGET) && SYCL_INTEL_TARGET == 20
  const uint32_t scale_c = 0x7E000000u;
  if constexpr (Kind == ReorderKind::UU) {
    asm("{\n"
        ".decl IN_UB v_type=G type=UB num_elts=64 alias=<%1,0>\n"
        ".decl OUT_UW v_type=G type=UW num_elts=64 alias=<%0,0>\n"
        "mov (M1_NM, 32) OUT_UW(0,0)<1> IN_UB(0,0)<1;1,0>\n"
        "mov (M1_NM, 32) OUT_UW(1,0)<1> IN_UB(0,32)<1;1,0>\n" MOE_INT8_BF16_ZP_FOLDED_TAIL "}\n"
        : "=rw"(dst0)
        : "rw"(src0), "rw.u"(scale_c), "rw"(zp_shifts), "rw"(muls));
  } else {
    static_assert(Kind == ReorderKind::VV, "unsupported UINT8 -> BF16 reorder kind for folded dequant");
    asm("{\n"
        ".decl IN_UB v_type=G type=UB num_elts=64 alias=<%1,0>\n"
        ".decl OUT_UW v_type=G type=UW num_elts=64 alias=<%0,0>\n"
        "mov (M1_NM, 32) OUT_UW(0,0)<1> IN_UB(0,0)<4;2,1>\n"
        "mov (M1_NM, 32) OUT_UW(1,0)<1> IN_UB(0,2)<4;2,1>\n" MOE_INT8_BF16_ZP_FOLDED_TAIL "}\n"
        : "=rw"(dst0)
        : "rw"(src0), "rw.u"(scale_c), "rw"(zp_shifts), "rw"(muls));
  }
#endif
}

template <ReorderKind Kind>
CUTE_DEVICE void uint8b128_reorder_unscaled(intel::uchar4 const& src0, intel::ushort4& dst0) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_INTEL_TARGET) && SYCL_INTEL_TARGET == 20
  const uint32_t scale_c = 0x7E000000u;
  const uint32_t shift_c = 0xBF000000u;
  if constexpr (Kind == ReorderKind::UU) {
    asm("{\n"
        ".decl IN_UB v_type=G type=UB num_elts=64 alias=<%1,0>\n"
        ".decl OUT_UW v_type=G type=UW num_elts=64 alias=<%0,0>\n"
        "mov (M1_NM, 32) OUT_UW(0,0)<1> IN_UB(0,0)<1;1,0>\n"
        "mov (M1_NM, 32) OUT_UW(1,0)<1> IN_UB(0,32)<1;1,0>\n" MOE_UINT8B128_BF16_UNSCALED_TAIL "}\n"
        : "=rw"(dst0)
        : "rw"(src0), "rw.u"(scale_c), "rw.u"(shift_c));
  } else {
    static_assert(Kind == ReorderKind::VV, "unsupported UINT8 -> BF16 reorder kind");
    asm("{\n"
        ".decl IN_UB v_type=G type=UB num_elts=64 alias=<%1,0>\n"
        ".decl OUT_UW v_type=G type=UW num_elts=64 alias=<%0,0>\n"
        "mov (M1_NM, 32) OUT_UW(0,0)<1> IN_UB(0,0)<4;2,1>\n"
        "mov (M1_NM, 32) OUT_UW(1,0)<1> IN_UB(0,2)<4;2,1>\n" MOE_UINT8B128_BF16_UNSCALED_TAIL "}\n"
        : "=rw"(dst0)
        : "rw"(src0), "rw.u"(scale_c), "rw.u"(shift_c));
  }
#endif
}

template <class T>
struct reorder_kind_of;
template <ReorderKind K, class S, class D>
struct reorder_kind_of<Xe_Reorder<K, S, D>> {
  static constexpr ReorderKind value = K;
};

template <class SEngine, class SLayoutWI, class SLayout, class DEngine, class DLayoutWI, class DLayout, class ChunkFn>
CUTE_DEVICE void w8a16_visit_reorder_chunks(
    SubgroupTensor<SEngine, SLayoutWI, SLayout> const& src,
    SubgroupTensor<DEngine, DLayoutWI, DLayout>& dst,
    ChunkFn&& chunk_fn) {
  using SType = typename SEngine::element_type;
  using DType = typename DEngine::element_type;
  static_assert(sizeof_bits_v<SType> == 8 && is_same_v<DType, bfloat16_t>, "W8A16 folded reorder is 8-bit -> BF16");
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
    chunk_fn(Int<static_cast<int>(reorder_kind_of<Atom>::value)>{}, Int<dv>{}, *pS, *pD);
  });
}

template <class SEngine, class SLayoutWI, class SLayout, class DEngine, class DLayoutWI, class DLayout, class MulOf>
CUTE_DEVICE void e4m3_reorder_dequant(
    SubgroupTensor<SEngine, SLayoutWI, SLayout> const& src,
    SubgroupTensor<DEngine, DLayoutWI, DLayout>& dst,
    MulOf const& mul_of) {
  w8a16_visit_reorder_chunks(src, dst, [&](auto kind_c, auto dv_c, auto const& s_reg, auto& d_reg) {
    e4m3_reorder_folded<static_cast<ReorderKind>(decltype(kind_c)::value)>(s_reg, d_reg, mul_of(dv_c));
  });
}

template <class SEngine, class SLayoutWI, class SLayout, class DEngine, class DLayoutWI, class DLayout, class MulOf>
CUTE_DEVICE void int8_sym_reorder_dequant(
    SubgroupTensor<SEngine, SLayoutWI, SLayout> const& src,
    SubgroupTensor<DEngine, DLayoutWI, DLayout>& dst,
    MulOf const& mul_of,
    bool is_signed_int8) {
  w8a16_visit_reorder_chunks(src, dst, [&](auto kind_c, auto dv_c, auto const& s_reg, auto& d_reg) {
    int8_sym_reorder_folded<static_cast<ReorderKind>(decltype(kind_c)::value)>(
        s_reg, d_reg, mul_of(dv_c), is_signed_int8);
  });
}

template <
    class SEngine,
    class SLayoutWI,
    class SLayout,
    class DEngine,
    class DLayoutWI,
    class DLayout,
    class ZpOf,
    class MulOf>
CUTE_DEVICE void uint8_zp_reorder_dequant(
    SubgroupTensor<SEngine, SLayoutWI, SLayout> const& src,
    SubgroupTensor<DEngine, DLayoutWI, DLayout>& dst,
    ZpOf const& zp_of,
    MulOf const& mul_of) {
  w8a16_visit_reorder_chunks(src, dst, [&](auto kind_c, auto dv_c, auto const& s_reg, auto& d_reg) {
    uint8_zp_reorder_folded<static_cast<ReorderKind>(decltype(kind_c)::value)>(s_reg, d_reg, zp_of(dv_c), mul_of(dv_c));
  });
}

template <class SEngine, class SLayoutWI, class SLayout, class DEngine, class DLayoutWI, class DLayout>
CUTE_DEVICE void uint8b128_reorder(
    SubgroupTensor<SEngine, SLayoutWI, SLayout> const& src, SubgroupTensor<DEngine, DLayoutWI, DLayout>& dst) {
  w8a16_visit_reorder_chunks(src, dst, [&](auto kind_c, auto /*dv_c*/, auto const& s_reg, auto& d_reg) {
    uint8b128_reorder_unscaled<static_cast<ReorderKind>(decltype(kind_c)::value)>(s_reg, d_reg);
  });
}

}  // namespace moe_w8a16
