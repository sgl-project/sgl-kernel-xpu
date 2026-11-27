// SPDX-License-Identifier: Apache-2.0
//
// Row-batched building blocks of the decode kernels. Batched decode (NB = 2,
// 3, 4, 6, 8 rows; one token per request; SGLang pads the
// batch with rows of KV slot 0, sequence length 1 and state slot 0): the same
// one-kernel-per-layer structure as the single-row kernels, every GEMV as DPAS with
// the activations as the A matrix (NB rows, zero padded to 8) and the weights
// as B. All fp8 weights are [N][K] (rows = outputs, routed experts included):
// lane = output row, 16 B per 16-K chunk, fp8 -> fp16 with fp8x2_lo / hi; the K
// order inside a chunk is permuted (0,2 | 1,3 | ...) on both sides (A tiles
// "perm"). A 16-row tile lies in one 128-row block and 8 chunks in one 128-K
// block, so a tile's partial over 8 chunks takes one block scale.
// A tiles (8 rows x 16 K per chunk, DPAS A layout): dword a_dw(kc, b, p) holds
// the K pair p of row b. MoE: the union U of the rows' top-8 experts; tasks
// (expert, row eighth) are dealt to the work-groups, the rows' routing weights
// are folded into the hidden activations, down is split by output rows.
//
// Also the batched scratch and the mailbox regions of the row configurations.
//
// Stage code: every function that depends on the model's shape takes the
// traits Tr (Dims<Shape>, dims.hpp) as its first template argument.
#pragma once
#include "sgl_kernel/stage/xe20/args.hpp"

namespace sgl_kernel::stage::xe20 {

inline int a_dw(int kc, int b, int p) {
  return (kc * 16 + (b & 1) * 8 + p) * 4 + (b >> 1);
}
// position of original element i (0..15) of a chunk in the permuted order (swap
// bits 0 and 1)
inline int perm_pos(int i) {
  return (i & ~3) | ((i & 1) << 1) | ((i >> 1) & 1);
}
// one element of row b at K index k into perm A tiles (16-bit store)
inline void a_put_perm(uint32_t* xa, int b, int k, half v) {
  const int pos = perm_pos(k & 15);
  reinterpret_cast<half*>(xa)[a_dw(k >> 4, b, pos >> 1) * 2 + (pos & 1)] = v;
}
inline void a_put_nat(uint32_t* xa, int b, int k, half v) {
  const int pos = k & 15;
  reinterpret_cast<half*>(xa)[a_dw(k >> 4, b, pos >> 1) * 2 + (pos & 1)] = v;
}
// 4 consecutive elements k..k+3 (k % 4 == 0)
inline void a_put4_perm(uint32_t* xa, int b, int k, const half (&x)[4]) {
  const int kc = k >> 4, q = (k & 15) >> 2;
  xa[a_dw(kc, b, 2 * q)] = pack_h2(x[0], x[2]);
  xa[a_dw(kc, b, 2 * q + 1)] = pack_h2(x[1], x[3]);
}
// zero the rows >= NB of chunks [kc0, kc0 + nch)
template <int NB>
inline void a_zero_pad(uint32_t* xa, int kc0, int nch, int t, int nt) {
  if constexpr (NB < 8) {
    for (int u = t; u < nch * 64; u += nt) {
      const int j = u & 3, l = (u >> 2) & 15;
      if (2 * j + (l >> 3) >= NB) xa[kc0 * 64 + u] = 0u;
    }
  }
}

// 16 fp8 weights of a [N][K] row (16 K) -> DPAS B (VNNI) in the permuted K
// order
inline v8u cvt16(sycl::uint4 w) {
  v8u b;
  b[0] = fp8x2_lo(w.x());
  b[1] = fp8x2_hi(w.x());
  b[2] = fp8x2_lo(w.y());
  b[3] = fp8x2_hi(w.y());
  b[4] = fp8x2_lo(w.z());
  b[5] = fp8x2_hi(w.z());
  b[6] = fp8x2_lo(w.w());
  b[7] = fp8x2_hi(w.w());
  return b;
}
// 16 16-bit weights of a [N][K] row (two uint4, natural order; DT) -> DPAS B in
// fp16 and the permuted K order: dwords (w0,w1),(w2,w3) -> (w0,w2),(w1,w3)
template <typename DT>
inline v8u cvt16_w16(sycl::uint4 lo, sycl::uint4 hi) {
  uint32_t d[8] = {lo.x(), lo.y(), lo.z(), lo.w(), hi.x(), hi.y(), hi.z(), hi.w()};
  if constexpr (!std::is_same_v<DT, half>) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint16_t a = sycl::bit_cast<uint16_t>((half)f32<DT>((uint16_t)(d[i] & 0xffffu)));
      const uint16_t b = sycl::bit_cast<uint16_t>((half)f32<DT>((uint16_t)(d[i] >> 16)));
      d[i] = pack16(a, b);
    }
  }
  v8u b;
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    b[2 * q] = (d[2 * q] & 0xffffu) | (d[2 * q + 1] << 16);
    b[2 * q + 1] = (d[2 * q] >> 16) | (d[2 * q + 1] & 0xffff0000u);
  }
  return b;
}
// rows GEMV pieces: NC chunks of 16 K of this lane's weight row
template <int NC>
inline void ld_rows(sycl::uint4 (&w)[NC], const uint8_t* p) {
#pragma unroll
  for (int t = 0; t < NC; ++t)
    w[t] = *reinterpret_cast<const sycl::uint4*>(p + t * 16);
}
// c += sum over the NC chunks (from chunk kc0, a multiple of 8) of A . B, each
// 8-chunk K block scaled by srow[block] (srow: the tile's N-block scale row)
template <int NC>
inline void mm_rows(v8f& c, const sycl::uint4 (&w)[NC], const uint32_t* xa, int kc0, int lane, const float* srow) {
  static_assert(NC % 8 == 0, "whole K blocks");
#pragma unroll
  for (int b8 = 0; b8 < NC / 8; ++b8) {
    v8f t = v8f(0.f);
#pragma unroll
    for (int i = 0; i < 8; ++i)
      dpas_hf(t, *reinterpret_cast<const v4u*>(xa + ((kc0 + b8 * 8 + i) * 16 + lane) * 4), cvt16(w[b8 * 8 + i]));
    c += t * srow[(kc0 >> 3) + b8];
  }
}
// as mm_rows, with the NC / 8 block scales already in registers
template <int NC>
inline void
mm_rows_r(v8f& c, const sycl::uint4 (&w)[NC], const uint32_t* xa, int kc0, int lane, const float (&sr)[NC / 8]) {
#pragma unroll
  for (int b8 = 0; b8 < NC / 8; ++b8) {
    v8f t = v8f(0.f);
#pragma unroll
    for (int i = 0; i < 8; ++i)
      dpas_hf(t, *reinterpret_cast<const v4u*>(xa + ((kc0 + b8 * 8 + i) * 16 + lane) * 4), cvt16(w[b8 * 8 + i]));
    c += t * sr[b8];
  }
}
// unscaled: c += A . B over NC chunks (one K block, scaled by the caller)
template <int NC>
inline void mm_rows_u(v8f& c, const sycl::uint4 (&w)[NC], const uint32_t* xa, int kc0, int lane) {
#pragma unroll
  for (int t = 0; t < NC; ++t)
    dpas_hf(c, *reinterpret_cast<const v4u*>(xa + ((kc0 + t) * 16 + lane) * 4), cvt16(w[t]));
}
// SG partial (lane = column / row of the tile, element i = batch row i) ->
// red[s][i][lane]
inline void red_put(float* red, int s, int lane, const v8f& c) {
#pragma unroll
  for (int i = 0; i < 8; ++i)
    red[(s * 8 + i) * 16 + lane] = c[i];
}
inline float red_get(const float* red, int s, int i, int lane) {
  return red[(s * 8 + i) * 16 + lane];
}

// batched scratch (one per layer, next to Scratch), sized for NBMAX rows
struct ScratchB {
  half* qkv;       // [8][CONVD]
  half* core;      // [8][VD]
  half* z;         // [8][VD]
  float* part_rt;  // [NWG][8][NE]
  float* part_sh;  // [NWG][8][2 * INTER]
  float* part_sg;  // [NWG][8][2]   (shared gate partial, sum of squares)
  uint32_t* hs_a;  // shared-expert h, perm A tiles over K = INTER: [INTER / 16][64]
  uint32_t* hA;    // routed h per union expert, perm A tiles: [UMAX][INTER / 16][64]
  float* qkv_raw;  // ATTN: [8][QKV_ROWS] fp32 GEMV outputs
  half* attn_out;  // ATTN: [8][NQ * HD] gated attention output
  half* pa_o;      // ATTN: per context chunk (80 tokens of one sequence) partial o
                   // [APASS * 512][NQ][HD] (shared
  float* pa_ml;    //   by all attention layers: they run one at a time), (m, l)
                   //   [APASS * 512][NQ][2]
};
template <class Tr>
inline size_t attn_part_bytes() {  // partial o, (m, l), then per slot the row
                                   // [APASS * APAIRS] int, per pair the count
  return (size_t)Tr::APASS * Tr::APAIRS * Tr::NQ * (Tr::HD * 2 + 2 * 4) + (size_t)Tr::APASS * Tr::APAIRS * 4 +
         (size_t)Tr::APAIRS * 4;
}
template <class Tr>
inline size_t scratch_b_bytes() {
  size_t n = 0;
  auto add = [&](size_t b) { n += (b + 255) & ~size_t(255); };
  add(Tr::NBMAX * Tr::CONVD * 2);
  add(Tr::NBMAX * Tr::VD * 2);
  add(Tr::NBMAX * Tr::VD * 2);
  add((size_t)Tr::NWG * Tr::NBMAX * Tr::NE * 4);
  add((size_t)Tr::NWG * Tr::NBMAX * 2 * Tr::INTER * 4);
  add(Tr::NWG * Tr::NBMAX * 2 * 4);
  add(Tr::INTER / 16 * 64 * 4);
  add((size_t)Tr::UMAX * Tr::INTER / 16 * 64 * 4);
  add(Tr::NBMAX * Tr::QKV_ROWS * 4);
  add(Tr::NBMAX * Tr::NQ * Tr::HD * 2);
  return n;
}
template <class Tr>
inline ScratchB scratch_b_carve(void* base) {
  ScratchB B{};
  uint8_t* p = static_cast<uint8_t*>(base);
  auto take = [&](size_t b) {
    void* r = p;
    p += (b + 255) & ~size_t(255);
    return r;
  };
  B.qkv = (half*)take(Tr::NBMAX * Tr::CONVD * 2);
  B.core = (half*)take(Tr::NBMAX * Tr::VD * 2);
  B.z = (half*)take(Tr::NBMAX * Tr::VD * 2);
  B.part_rt = (float*)take((size_t)Tr::NWG * Tr::NBMAX * Tr::NE * 4);
  B.part_sh = (float*)take((size_t)Tr::NWG * Tr::NBMAX * 2 * Tr::INTER * 4);
  B.part_sg = (float*)take(Tr::NWG * Tr::NBMAX * 2 * 4);
  B.hs_a = (uint32_t*)take(Tr::INTER / 16 * 64 * 4);
  B.hA = (uint32_t*)take((size_t)Tr::UMAX * Tr::INTER / 16 * 64 * 4);
  B.qkv_raw = (float*)take(Tr::NBMAX * Tr::QKV_ROWS * 4);
  B.attn_out = (half*)take(Tr::NBMAX * Tr::NQ * Tr::HD * 2);
  B.pa_o = nullptr;
  B.pa_ml = nullptr;  // attn_part_bytes(), set by the owner
  return B;
}
template <class Tr, int NB>
constexpr size_t mb_region() {
  static_assert(NB == 2 || NB == 3 || NB == 4 || NB == 6 || NB == 8, "batched rows");
  return NB == 2 ? 2 * Tr::HID : NB == 4 ? 4 * Tr::HID : NB == 8 ? 8 * Tr::HID : NB == 3 ? 16 * Tr::HID : 19 * Tr::HID;
}
template <int NB>
constexpr int mb_tag() {
  return NB == 2 ? MBT + 1 : NB == 4 ? MBT + 2 : NB == 8 ? MBT + 3 : NB == 3 ? MBT2 : MBT2 + 1;
}

}  // namespace sgl_kernel::stage::xe20
