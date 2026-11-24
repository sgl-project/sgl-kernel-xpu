/***************************************************************************************************
 * Copyright 2025 SGLang Team. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 **************************************************************************************************/
/*! \file
    \brief Lane-local register primitives for the Xe3P NVFP4 -> MXFP8 WoQ mainloop.

    Register layout facts these primitives are built on.

    A SIMT work-item value of a 32-bit type maps 1:1 onto a hardware lane:
    element w of a per-lane `uint` vector lives in GRF w, dword lane t.  8/16-bit
    SIMT vectors do NOT (they are stored structure-of-arrays, value v of lane t
    at byte 16*v + t), which is what made the old byte-fragment conversion pay
    thousands of cross-lane movs.  Everything here therefore moves 32-bit lane
    arrays between inline-vISA statements and only reinterprets them as bytes /
    halves *inside* vISA, where the physical layout is explicit.

    * DPAS/BDPAS B operand (8-bit, K=32 per atom): 8 GRFs; GRF w, lane n holds
      the four bytes B[k = 4w .. 4w+3][n] (lowest k in the lowest byte).
    * Packed NVFP4 B is (N, K/2) K-contiguous.  A transposed d32 block-2D load
      of H=16 rows x W=8 dwords puts row n in lane n with GRF j = packed dword j,
      i.e. nibbles k = 8j .. 8j+7 of column n (low nibble = even k).
    * Hence each lane converts *its own* column; no data crosses lanes.

    Per input dword j (8 nibbles of one column, 128 values per subgroup):
      4 x shfl_idx4 (16 lanes)  E2M1 byte -> fp16 pair, LUT shuffle
      4 x mul (32 x hf)         times the per-16 multiplier c (exact in fp16)
      4 x fcvt (32 x hf->hf8)   RNE, written with a byte stride of 4 straight
                                into the B-operand dwords
    = 12 instructions / 128 values.  The bytes of output dword w = 2j' (k0..k3)
    and 2j'+1 (k4..k7) come from the decoded pairs of input bytes {0,1} and
    {2,3}; pairing byte 0 with byte 2 (and 1 with 3) in adjacent GRFs lets one
    32-wide fcvt with dst stride 4 fill the same byte slot of both dwords.

    Exactness: E2M1 has 2 significant bits and c (BF16) 8, so c*e2m1 is exact
    in binary16 (|c*e2m1| <= 448 by prep construction).  Products below the
    binary16 normal range are below half the smallest E4M3 subnormal, so they
    round to zero either way, and the subsequent fcvt is the same RNE the host
    reference applies.  No saturation/inf/nan fixups are needed: values are
    finite and in range by construction.
*/

#pragma once

// clang-format off
#include <cstdint>
#include <sycl/sycl.hpp>

#include <cute/util/sycl_vec.hpp>
#include <cute/atom/copy_traits_xe_2d.hpp>

#include "cutlass/cutlass.h"

namespace sgl_xpu::xe {

using u32 = uint32_t;
using cute::intel::uint2;
using cute::intel::uint4;
using cute::intel::uint8;
using cute::intel::float8;
template <int N>
using uintv = cute::intel::vector_t<u32, N>;

/// fp16(e2m1(i)) replicated in both halves of the dword; lane i of the LUT GRF.
CUTLASS_DEVICE u32 e2m1_fp16x2_lut_entry(int i) {
  const u32 e = (static_cast<u32>(i) >> 1) & 3u;
  const u32 m = static_cast<u32>(i) & 1u;
  const u32 s = (static_cast<u32>(i) >> 3) & 1u;
  const u32 mag = e == 0 ? (m ? 0x3800u : 0u) : (((e + 14u) << 10) | (m << 9));
  const u32 h = (s << 15) | mag;
  return h | (h << 16);
}

// ---------------------------------------------------------------------------
// Block-2D payloads
// ---------------------------------------------------------------------------

/// A block-2D address payload.  `x` is in elements of the message data size
/// (bytes for d8, dwords for d32); width and pitch are always bytes.
struct Block2D {
  int* payload;

  CUTLASS_DEVICE void init(const void* base, uint32_t width_bytes, uint32_t height,
                           uint32_t pitch_bytes, int block_w, int block_h, int blocks) {
#ifdef __SYCL_DEVICE_ONLY__
    payload = __builtin_IB_subgroup_createBlock2DAddressPayload(
        reinterpret_cast<long>(base), static_cast<int>(width_bytes) - 1,
        static_cast<int>(height) - 1, static_cast<int>(pitch_bytes) - 1, 0, 0,
        block_w, block_h, blocks);
#endif
  }
  CUTLASS_DEVICE void set_xy(int x, int y) const {
#ifdef __SYCL_DEVICE_ONLY__
    __builtin_IB_subgroup_setBlock2DAddressPayloadBlockX(payload, x);
    __builtin_IB_subgroup_setBlock2DAddressPayloadBlockY(payload, y);
#endif
  }
  CUTLASS_DEVICE void set_x(int x) const {
#ifdef __SYCL_DEVICE_ONLY__
    __builtin_IB_subgroup_setBlock2DAddressPayloadBlockX(payload, x);
#endif
  }
  CUTLASS_DEVICE void set_y(int y) const {
#ifdef __SYCL_DEVICE_ONLY__
    __builtin_IB_subgroup_setBlock2DAddressPayloadBlockY(payload, y);
#endif
  }
};

#ifdef __SYCL_DEVICE_ONLY__
#define SGL_XE_ASM asm volatile
#define SGL_XE_ASM_PURE asm
#endif

/// d32 transposed load, W dwords x H rows at (x + DX, y + DY);
/// lane n <- row (y + DY + n), GRF j <- dword (x + DX + j).
template <int W, int H, int DY = 0, int DX = 0, class T>
CUTLASS_DEVICE void load_d32_t(Block2D const& p, T& dst) {
  static_assert(int(sizeof(T)) * 16 == W * H * 4, "destination size must match the block");
#ifdef __SYCL_DEVICE_ONLY__
  SGL_XE_ASM("lsc_load_block2d.ugm.ca.ca.uc (M1, 1)  %0:d32.%2x%3tn flat[%1+(%5,%4)]"
             : "=rw"(dst)
             : "rw.u"(p.payload), "P"(W), "P"(H), "P"(DY), "P"(DX));
#endif
}

/// d8 load, `Count` blocks of BW bytes x H rows at (x + DX, y + DY), row-major per block.
template <int Count, int BW, int H, int DY = 0, int DX = 0, class T>
CUTLASS_DEVICE void load_d8(Block2D const& p, T& dst) {
#ifdef __SYCL_DEVICE_ONLY__
  SGL_XE_ASM("lsc_load_block2d.ugm.ca.ca.uc (M1, 1)  %0:d8.%2x%3x%4nn flat[%1+(%6,%5)]"
             : "=rw"(dst)
             : "rw.u"(p.payload), "P"(Count), "P"(BW), "P"(H), "P"(DY), "P"(DX));
#endif
}

/// d32 store of one 8x16 fp32 accumulator (GRF r = row r, lane n = column n),
/// at (x + DX, y + DY) in dwords / rows.  Out-of-surface rows/columns are dropped.
template <int DX, int DY>
CUTLASS_DEVICE void store_d32_16x8(Block2D const& p, float8 const& src) {
#ifdef __SYCL_DEVICE_ONLY__
  SGL_XE_ASM("lsc_store_block2d.ugm (M1, 1)  flat[%1+(%2,%3)]  %0:d32.16x8nn"
             :: "rw"(src), "rw.u"(p.payload), "P"(DX), "P"(DY));
#endif
}

/// Prefetch (no destination) at the payload's (x + DX, y + DY).
template <int Bits, int W, int H, int DY = 0, int DX = 0>
CUTLASS_DEVICE void prefetch_2d(Block2D const& p) {
#ifdef __SYCL_DEVICE_ONLY__
  SGL_XE_ASM("lsc_load_block2d.ugm.ca.ca.uc (M1, 1)  %%null:d%1.%2x%3nn flat[%0+(%5,%4)]"
             :: "rw.u"(p.payload), "P"(Bits), "P"(W), "P"(H), "P"(DY), "P"(DX));
#endif
}

// ---------------------------------------------------------------------------
// NVFP4 column -> E4M3 bdpas B operand (one K=32 atom of one 16-column slice)
// ---------------------------------------------------------------------------

#ifdef __SYCL_DEVICE_ONLY__
// One packed input dword (operand %R = its GRF row in RAW) -> output dwords W, W+1.
#define SGL_NVFP4_J(J, R, CH, W)                                                             \
  "shfl_idx4 (M1_NM, 16) HA" #J ".0  LUT_UD(0,0)<1;1,0> RAW_UB(%" #R ",0)<4;1,0>\n"          \
  "shfl_idx4 (M1_NM, 16) HA" #J ".64 LUT_UD(0,0)<1;1,0> RAW_UB(%" #R ",2)<4;1,0>\n"          \
  "shfl_idx4 (M1_NM, 16) HB" #J ".0  LUT_UD(0,0)<1;1,0> RAW_UB(%" #R ",1)<4;1,0>\n"          \
  "shfl_idx4 (M1_NM, 16) HB" #J ".64 LUT_UD(0,0)<1;1,0> RAW_UB(%" #R ",3)<4;1,0>\n"          \
  "mul (M1_NM, 32) HA" #J "_HF(0,0)<1> HA" #J "_HF(0,0)<1;1,0> CH(" #CH ",0)<1;1,0>\n"       \
  "mul (M1_NM, 32) HA" #J "_HF(1,0)<1> HA" #J "_HF(1,0)<1;1,0> CH(" #CH ",0)<1;1,0>\n"       \
  "mul (M1_NM, 32) HB" #J "_HF(0,0)<1> HB" #J "_HF(0,0)<1;1,0> CH(" #CH ",0)<1;1,0>\n"       \
  "mul (M1_NM, 32) HB" #J "_HF(1,0)<1> HB" #J "_HF(1,0)<1;1,0> CH(" #CH ",0)<1;1,0>\n"       \
  "fcvt (M1_NM, 32) OUT_B(" #W ",0)<4> HA" #J "_HF(0,0)<2;1,0>\n"                            \
  "fcvt (M1_NM, 32) OUT_B(" #W ",1)<4> HA" #J "_HF(0,1)<2;1,0>\n"                            \
  "fcvt (M1_NM, 32) OUT_B(" #W ",2)<4> HB" #J "_HF(0,0)<2;1,0>\n"                            \
  "fcvt (M1_NM, 32) OUT_B(" #W ",3)<4> HB" #J "_HF(0,1)<2;1,0>\n"

#define SGL_NVFP4_DECL_J(J)                                                                  \
  ".decl HA" #J " v_type=G type=UD num_elts=32 align=GRF\n"                                  \
  ".decl HB" #J " v_type=G type=UD num_elts=32 align=GRF\n"                                  \
  ".decl HA" #J "_HF v_type=G type=HF num_elts=64 alias=<HA" #J ",0>\n"                      \
  ".decl HB" #J "_HF v_type=G type=HF num_elts=64 alias=<HB" #J ",0>\n"
#endif

/// Decode/scale/requantize dwords 4*KK .. 4*KK+3 of a transposed packed-B load.
///
///   raw : uintv<8|16> per lane, GRF j lane n = packed dword j of column n
///   cc  : uintv<*> per lane; GRF CROW lane n = bf16 (c[n][2a], c[n][2a+1]),
///         the two 16-block multipliers of this K=32 atom
///   lut : uint per lane (1 GRF), lane i = e2m1_fp16x2_lut_entry(i)
///   ret : uint8 per lane (8 GRFs) = E4M3 bdpas B operand for the atom
template <int KK, int CROW, class RawV, class CcV>
CUTLASS_DEVICE uint8 nvfp4_to_e4m3_atom(RawV const& raw, CcV const& cc, u32 lut) {
  static_assert(int(sizeof(RawV)) >= 16 * (KK + 1), "raw too small for this atom");
  static_assert(int(sizeof(CcV)) >= 4 * (CROW + 1), "cc too small for this atom");
  uint8 out;
#ifdef __SYCL_DEVICE_ONLY__
  SGL_XE_ASM_PURE(
      "{\n"
      ".decl RAW_UB v_type=G type=UB num_elts=%9 alias=<%1,0>\n"
      ".decl CC_BF v_type=G type=BF num_elts=%10 alias=<%2,0>\n"
      ".decl LUT_UD v_type=G type=UD num_elts=16 alias=<%3,0>\n"
      ".decl OUT_B v_type=G type=B num_elts=512 alias=<%0,0>\n"
      ".decl TF0 v_type=G type=F num_elts=16 align=GRF\n"
      ".decl TF1 v_type=G type=F num_elts=16 align=GRF\n"
      ".decl CH v_type=G type=HF num_elts=64 align=GRF\n"
      SGL_NVFP4_DECL_J(0) SGL_NVFP4_DECL_J(1) SGL_NVFP4_DECL_J(2) SGL_NVFP4_DECL_J(3)
      // c for the two 16-blocks of this atom, bf16 -> f32 -> fp16, replicated
      // into both halves of each lane's dword.
      "mov (M1_NM, 16) TF0(0,0)<1> CC_BF(%4,0)<2;1,0>\n"
      "mov (M1_NM, 16) TF1(0,0)<1> CC_BF(%4,1)<2;1,0>\n"
      "mov (M1_NM, 16) CH(0,0)<2> TF0(0,0)<1;1,0>\n"
      "mov (M1_NM, 16) CH(0,1)<2> TF0(0,0)<1;1,0>\n"
      "mov (M1_NM, 16) CH(1,0)<2> TF1(0,0)<1;1,0>\n"
      "mov (M1_NM, 16) CH(1,1)<2> TF1(0,0)<1;1,0>\n"
      SGL_NVFP4_J(0, 5, 0, 0)
      SGL_NVFP4_J(1, 6, 0, 2)
      SGL_NVFP4_J(2, 7, 1, 4)
      SGL_NVFP4_J(3, 8, 1, 6)
      "}\n"
      : "=rw"(out)
      : "rw"(raw), "rw"(cc), "rw"(lut), "P"(CROW), "P"(4 * KK + 0), "P"(4 * KK + 1),
        "P"(4 * KK + 2), "P"(4 * KK + 3), "P"(int(sizeof(RawV)) * 16),
        "P"(int(sizeof(CcV)) * 8));
#endif
  return out;
}

/// acc(8x16 f32) += A(8x32 e4m3) * B(32x16 e4m3), block-scaled by UE8M0 sfa/sfb.
///
/// `a` is the raw A register block, OFFA the byte offset of this 8x32 atom
/// (row-major, 256 bytes).  `sa`/`sb` are raw UE8M0 scale registers; OFFSA /
/// OFFSB are byte offsets of the 8 row scales / 16 column scales.
template <int OFFA, int OFFSA, int OFFSB, class AV, class SAV, class SBV>
CUTLASS_DEVICE void bdpas_e4m3(float8& acc, AV const& a, uint8 const& b, SAV const& sa,
                               SBV const& sb) {
  static_assert(OFFA % 64 == 0 && OFFA + 256 <= int(sizeof(AV)) * 16, "A atom out of range");
  static_assert(OFFSA + 8 <= int(sizeof(SAV)) * 16, "A scale out of range");
  static_assert(OFFSB + 16 <= int(sizeof(SBV)) * 16, "B scale out of range");
#ifdef __SYCL_DEVICE_ONLY__
  SGL_XE_ASM_PURE(
      "{\n"
      ".decl DST v_type=G type=F num_elts=128 alias=<%0,0>\n"
      ".decl SRC1_UD v_type=G type=UD num_elts=128 alias=<%2,0>\n"
      ".decl SRC2_UD v_type=G type=UD num_elts=64 alias=<%1,%5>\n"
      ".decl SRC3_UB v_type=G type=UB num_elts=%9 alias=<%4,%7>\n"
      ".decl SRC4_UB v_type=G type=UB num_elts=%8 alias=<%3,%6>\n"
      "bdpas.hf8.hf8.8.8 (M1, 16) DST.0 DST.0 SRC1_UD.0 SRC2_UD.0 SRC3_UB(0,0) SRC4_UB(0,0)\n"
      "}\n"
      : "+rw"(acc)
      : "rw"(a), "rw"(b), "rw"(sa), "rw"(sb), "P"(OFFA), "P"(OFFSA), "P"(OFFSB),
        "P"(int(sizeof(SAV)) * 16 - OFFSA), "P"(int(sizeof(SBV)) * 16 - OFFSB));
#endif
}

}  // namespace sgl_xpu::xe
