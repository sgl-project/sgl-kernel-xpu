// SPDX-License-Identifier: Apache-2.0
//
// Batched decode (NB = 2, 4, 8 rows; one token per request; SGLang pads the
// batch with rows of KV slot 0, sequence length 1 and state slot 0): the same
// one-kernel-per-layer structure as mono_layers.hpp, every GEMV as DPAS with
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
#pragma once
#include "mono_layers.hpp"

namespace sgl_mono {

constexpr int NBMAX = 8;
constexpr int UMAX = NBMAX * TOPK;  // experts in the union

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
constexpr int ACH = 80;                // cached tokens per SG pair and pass
constexpr int APAIRS = NWG * NSG / 2;  // 512
constexpr int APASS = NBMAX;           // partial slots reserved for up to NBMAX passes (an upper bound of the buffers)
inline size_t attn_part_bytes() {      // partial o, (m, l), then per slot the row
                                       // [APASS * APAIRS] int, per pair the count
  return (size_t)APASS * APAIRS * NQ * (HD * 2 + 2 * 4) + (size_t)APASS * APAIRS * 4 + (size_t)APAIRS * 4;
}
inline size_t scratch_b_bytes() {
  size_t n = 0;
  auto add = [&](size_t b) { n += (b + 255) & ~size_t(255); };
  add(NBMAX * CONVD * 2);
  add(NBMAX * VD * 2);
  add(NBMAX * VD * 2);
  add((size_t)NWG * NBMAX * NE * 4);
  add((size_t)NWG * NBMAX * 2 * INTER * 4);
  add(NWG * NBMAX * 2 * 4);
  add(INTER / 16 * 64 * 4);
  add((size_t)UMAX * INTER / 16 * 64 * 4);
  add(NBMAX * QKV_ROWS * 4);
  add(NBMAX * NQ * HD * 2);
  return n;
}
inline ScratchB scratch_b_carve(void* base) {
  ScratchB B{};
  uint8_t* p = static_cast<uint8_t*>(base);
  auto take = [&](size_t b) {
    void* r = p;
    p += (b + 255) & ~size_t(255);
    return r;
  };
  B.qkv = (half*)take(NBMAX * CONVD * 2);
  B.core = (half*)take(NBMAX * VD * 2);
  B.z = (half*)take(NBMAX * VD * 2);
  B.part_rt = (float*)take((size_t)NWG * NBMAX * NE * 4);
  B.part_sh = (float*)take((size_t)NWG * NBMAX * 2 * INTER * 4);
  B.part_sg = (float*)take(NWG * NBMAX * 2 * 4);
  B.hs_a = (uint32_t*)take(INTER / 16 * 64 * 4);
  B.hA = (uint32_t*)take((size_t)UMAX * INTER / 16 * 64 * 4);
  B.qkv_raw = (float*)take(NBMAX * QKV_ROWS * 4);
  B.attn_out = (half*)take(NBMAX * NQ * HD * 2);
  B.pa_o = nullptr;
  B.pa_ml = nullptr;  // attn_part_bytes(), set by the owner
  return B;
}
// mailbox words of one layer: a region per configuration (rows 1: 2 HID fp32
// words; rows NB: 2 all-reduces x NB HID / 2 words of two fp16 each), rows 2 /
// 4 / 8 / 3 / 6 at 2 / 4 / 8 / 16 / 19 HID
constexpr size_t MBOX_WORDS_B = 32 * (size_t)HID;
template <int NB>
constexpr size_t mb_region() {
  static_assert(NB == 2 || NB == 3 || NB == 4 || NB == 6 || NB == 8, "batched rows");
  return NB == 2 ? 2 * HID : NB == 4 ? 4 * HID : NB == 8 ? 8 * HID : NB == 3 ? 16 * HID : 19 * HID;
}
template <int NB>
constexpr int mb_tag() {
  return NB == 2 ? MBT + 1 : NB == 4 ? MBT + 2 : NB == 8 ? MBT + 3 : NB == 3 ? MBT2 : MBT2 + 1;
}
static_assert(MBOX_WORDS_B >= MBOX_WORDS, "the batched mailbox covers the single-row one");

// -------------------------------------------------------- SLM layout (batched)
constexpr int B_XA = 0,             // A tiles of the current activations: 128 chunks x 256 B
    B_RED = B_XA + 32768,           // SG partials: [32][8][16] f32
    B_QK = B_RED + 16384,           // GDN: per row q[128] k[128] v[64] beta g (384 f32 per row)
    B_SM = B_QK + 8 * 384 * 4,      // small reductions: 1024 f32
    B_RB = B_SM + 4096,             // tail: res2 block [8][64] f32
    B_XAN = B_RB + 2048,            // tail: router A tiles (nat, 4 chunks)
    B_XAP = B_XAN + 1024,           // tail: shared gate|up A tiles (perm, 4 chunks)
    B_LG = B_XAP + 1024,            // logits [8][256] f32
    B_S6R = B_RED,                  // S6 SG partials [32][2 tiles][8][16] f32 / S7 [32][4][8][16]
                                    // over RED .. LG (dead in S6 / S7)
    B_RT = B_RED + 65536,           // route_e [8][8] int, route_w [8][8] f32
    B_U = B_RT + 512,               // union: e [64] int, w [64][8] f32, count
    B_SGT = B_U + 256 + 2048 + 64,  // shared-expert gate per row [8] f32
    B_CS = B_SGT + 64,              // GDN: conv states of the WG's 320 channels [8][3][320] f16
    B_BYTES = B_CS + 8 * 3 * 320 * 2;
static_assert(B_LG + 8192 <= B_RT, "S6 partials alias only dead tail SLM");

struct SlmB {
  uint32_t* xa;
  float* red;
  float* sm;
  float* rb;
  uint32_t* xan;
  uint32_t* xap;
  float* lg;
  int* re;
  float* rw;
  int* uexp;
  float* uw;
  int* nu;
  float* s6r;
  float* sgt;
};
inline SlmB slm_b(uint8_t* slm) {
  SlmB L;
  L.xa = reinterpret_cast<uint32_t*>(slm + B_XA);
  L.red = reinterpret_cast<float*>(slm + B_RED);
  L.sm = reinterpret_cast<float*>(slm + B_SM);
  L.rb = reinterpret_cast<float*>(slm + B_RB);
  L.xan = reinterpret_cast<uint32_t*>(slm + B_XAN);
  L.xap = reinterpret_cast<uint32_t*>(slm + B_XAP);
  L.lg = reinterpret_cast<float*>(slm + B_LG);
  L.re = reinterpret_cast<int*>(slm + B_RT);
  L.rw = reinterpret_cast<float*>(slm + B_RT + 256);
  L.uexp = reinterpret_cast<int*>(slm + B_U);
  L.uw = reinterpret_cast<float*>(slm + B_U + 256);
  L.nu = reinterpret_cast<int*>(slm + B_U + 256 + 2048);
  L.s6r = reinterpret_cast<float*>(slm + B_S6R);
  L.sgt = reinterpret_cast<float*>(slm + B_SGT);
  return L;
}

// res = h + r per row ; x = GemmaRMS(res) -> perm A tiles (rows >= NB zero)
template <int NB, typename DT>
inline void input_norm_b(sycl::nd_item<1> it, const Ptrs& P, const SlmB& L) {
  const int lid = it.get_local_linear_id();
  sycl::sub_group sg = it.get_sub_group();
  const int s = sg.get_group_linear_id(), lane = sg.get_local_linear_id();
  float v[NB][4];
#pragma unroll
  for (int b = 0; b < NB; ++b) {
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int k = lid * 4 + i;
      v[b][i] = rnd<DT>(ld16<DT>(P.hs_in, b * HID + k) + ld16<DT>(P.res_in, b * HID + k));
      ss += v[b][i] * v[b][i];
    }
    ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
    if (lane == 0) L.sm[b * 32 + s] = ss;
  }
  a_zero_pad<NB>(L.xa, 0, 128, lid, WGS);
  wg_bar(it.get_group());
#pragma unroll
  for (int b = 0; b < NB; ++b) {
    float ss = 0.f;
#pragma unroll
    for (int w2 = 0; w2 < NSG; ++w2)
      ss += L.sm[b * 32 + w2];
    const float rs = sycl::rsqrt(ss / HID + EPS);
    half x[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
      x[i] = toh<DT>(v[b][i] * rs * (1.f + ld16<DT>(P.in_norm_w, lid * 4 + i)));
    a_put4_perm(L.xa, b, lid * 4, x);
  }
}

// all-reduce of two values per work-item: one mailbox word of two fp16 + call
// tag, half the PCIe writes of one fp32 per word. Both ranks add the same
// fp16-rounded partials (a + b == b + a).
inline void
ar_two(uint64_t* peer, uint64_t* mine, size_t word, float& v0, float& v1, uint32_t call, const uint32_t* cnt) {
  const half a0 = (half)v0, a1 = (half)v1;
  mb_put(peer + word, mb_word(pack_h2(a0, a1), call));
  const uint64_t w = mb_wait(mine + word, call, cnt);
  const h2 b = sycl::bit_cast<h2>(mb_val(w));
  v0 = (float)a0 + (float)b[0];
  v1 = (float)a1 + (float)b[1];
}

// -------------------------------------------------------------- tail (batched)
// Entry: the mixer output (NB rows) is in L.xa as perm A tiles; ow = this SG's
// o_proj / out_proj piece (rows wg*64 + (s>>3)*16 + lane, K chunks
// (s&7)*16..+16), prt_r / prt_s = router / shared gate|up B pieces of this WG.
template <int NB, typename DT>
inline void tail_b(
    sycl::nd_item<1> it,
    const Ptrs& P,
    const Scratch& S,
    const ScratchB& SB,
    const SlmB& L,
    uint32_t call,
    int c_o,
    const sycl::uint4 (&ow)[16],
    const sycl::uint4 (&prt_r)[4],
    const sycl::uint4 (&prt_s)[4]) {
  const uint32_t mtag = uc_load32(S.cnt + mb_tag<NB>()) + 1u;  // this configuration's mailbox region and tag
  constexpr size_t MB0 = mb_region<NB>(), MB1 = mb_region<NB>() + (size_t)NB * HID / 2;
  const int wg = it.get_group(0);
  const int lid = it.get_local_linear_id();
  sycl::sub_group sg = it.get_sub_group();
  const int s = sg.get_group_linear_id();
  const int lane = sg.get_local_linear_id();
  auto grp = it.get_group();
  const int n0 = wg * 64;
  // ---- out_proj: 4 row tiles x 8 K slices (K 2048, the WG's 64 rows: one N
  // block)
  {
    v8f c = v8f(0.f);
    mm_rows<16>(c, ow, L.xa, (s & 7) * 16, lane, P.out_s + (size_t)(n0 / QB) * KB_HID);
    red_put(L.red, s, lane, c);
  }
  wg_bar(grp);
  const int rb = lid >> 6, rr = lid & 63;  // (row, column of the WG's 64) of
                                           // this work-item, rows < NB
  if (lid < NB * 32) {                     // all-reduce #1, a column pair per work-item
    const int pb = lid >> 5, pc = (lid & 31) * 2, t = pc >> 4, l = pc & 15;
    float v0 = 0.f, v1 = 0.f;
#pragma unroll
    for (int k2 = 0; k2 < 8; ++k2) {
      v0 += red_get(L.red, t * 8 + k2, pb, l);
      v1 += red_get(L.red, t * 8 + k2, pb, l + 1);
    }
    v0 *= 256.f;
    v1 *= 256.f;
    ar_two(S.peer, S.mbox, MB0 + (((size_t)pb * HID + n0 + pc) >> 1), v0, v1, mtag, S.cnt);
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int n = n0 + pc + e;
      const float res = rnd<DT>(ld16<DT>(P.hs_in, pb * HID + n) + ld16<DT>(P.res_in, pb * HID + n));
      const float r2 = rnd<DT>((e ? v1 : v0) + res);
      st16<DT>(P.res_out, pb * HID + n, r2);
      L.rb[pb * 64 + pc + e] = r2;
    }
  }
  a_zero_pad<NB>(L.xan, 0, 4, lid, WGS);
  a_zero_pad<NB>(L.xap, 0, 4, lid, WGS);
  wg_bar(grp);
  if (rb < NB) {
    const half xb = (half)(L.rb[rb * 64 + rr] * (1.f + ld16<DT>(P.post_norm_w, n0 + rr)));
    a_put_nat(L.xan, rb, rr, xb);
    a_put_perm(L.xap, rb, rr, xb);
    // per-row partials over the WG's 64 dims: sum of squares, shared-expert
    // gate
    float q = L.rb[rb * 64 + rr] * L.rb[rb * 64 + rr];
    float gq = ld16<DT>(P.sgate_w, n0 + rr) * (float)xb;
    q = sycl::reduce_over_group(sg, q, sycl::plus<float>());
    gq = sycl::reduce_over_group(sg, gq, sycl::plus<float>());
    if (lane == 0) {
      L.sm[rb * 8 + (rr >> 4)] = q;
      L.sm[64 + rb * 8 + (rr >> 4)] = gq;
    }
  }
  wg_bar(grp);
  if (lid < NB) {
    float q = 0.f, gq = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      q += L.sm[lid * 8 + i];
      gq += L.sm[64 + lid * 8 + i];
    }
    SB.part_sg[(wg * NBMAX + lid) * 2] = gq;
    SB.part_sg[(wg * NBMAX + lid) * 2 + 1] = q;
  }
  {  // router partials (16 expert tiles x 2 chunk pairs), weights (as fp16)
     // in natural K order
    v8f c = v8f(0.f);
#pragma unroll
    for (int t = 0; t < 2; ++t) {
      const int kc = (s & 1) * 2 + t;
      v8u bw;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        bw[q] = prt_r[2 * t][q];
        bw[4 + q] = prt_r[2 * t + 1][q];
      }
      dpas_hf(c, *reinterpret_cast<const v4u*>(L.xan + (kc * 16 + lane) * 4), bw);
    }
    red_put(L.red, s, lane, c);
    // shared gate|up partials: rows s*16 + lane of shared w13 (one N block),
    // the WG's 4 chunks (one K block, complete per SG): scaled here
    v8f d = v8f(0.f);
    mm_rows_u<4>(d, prt_s, L.xap, 0, lane);
    const float ssc = P.sw13_s[(s * 16 / QB) * KB_HID + n0 / QB];
#pragma unroll
    for (int b = 0; b < NB; ++b)
      SB.part_sh[((size_t)wg * NBMAX + b) * 2 * INTER + s * 16 + lane] = d[b] * ssc;
  }
  wg_bar(grp);
  for (int idx = lid; idx < NB * NE; idx += WGS) {
    const int b = idx / NE, e = idx % NE, t = e >> 4, l = e & 15;
    SB.part_rt[((size_t)wg * NBMAX + b) * NE + e] = red_get(L.red, 2 * t, b, l) + red_get(L.red, 2 * t + 1, b, l);
  }
  signal(it, S.cnt + c_o * 16);

  // --- S5': reduce the partials (fixed order) per row ------------------------
  wait_for(it, S.cnt + c_o * 16, NWG * call);
  if (lid < NB) {
    float ss = 0.f, ga = 0.f;
#pragma unroll
    for (int w2i = 0; w2i < NWG; ++w2i) {
      ss += SB.part_sg[(w2i * NBMAX + lid) * 2 + 1];
      ga += SB.part_sg[(w2i * NBMAX + lid) * 2];
    }
    const float rs = sycl::rsqrt(ss / HID + EPS);
    L.sm[128 + lid] = rs;
    L.sgt[lid] = 1.f / (1.f + sycl::exp(-ga * rs));  // shared-expert gate
  }
  a_zero_pad<NB>(L.xa, 0, 128, lid, WGS);
  wg_bar(grp);
#pragma unroll
  for (int b = 0; b < NB; ++b) {  // x2 = GemmaRMS(res2) -> perm A tiles (routed experts)
    const float rs = L.sm[128 + b];
    half x2[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
      x2[i] = toh<DT>(ld16<DT>(P.res_out, b * HID + lid * 4 + i) * rs * (1.f + ld16<DT>(P.post_norm_w, lid * 4 + i)));
    a_put4_perm(L.xa, b, lid * 4, x2);
  }
  for (int idx = lid; idx < NB * NE; idx += WGS) {
    const int b = idx / NE, e = idx % NE;
    float a = 0.f;
#pragma unroll
    for (int w2i = 0; w2i < NWG; ++w2i)
      a += SB.part_rt[((size_t)w2i * NBMAX + b) * NE + e];
    L.lg[b * NE + e] = a * L.sm[128 + b];
  }
  {  // shared h, the WG's SPW inter rows, all rows (padding rows -> 0): 8
     // work-items per (row, inter row), 4 WGs' partials each, combined over the
     // 8 lanes
    const int pr = lid >> 3, w8 = lid & 7;
    const int b = pr / SPW, i = wg * SPW + pr % SPW;
    float g = 0.f, u = 0.f;
    if (b < NB) {
#pragma unroll
      for (int k = 0; k < NWG / 8; ++k) {
        const int w2i = w8 * (NWG / 8) + k;
        g += SB.part_sh[((size_t)w2i * NBMAX + b) * 2 * INTER + i];
        u += SB.part_sh[((size_t)w2i * NBMAX + b) * 2 * INTER + INTER + i];
      }
    }
#pragma unroll
    for (int m = 4; m >= 1; m >>= 1) {
      g += sycl::permute_group_by_xor(sg, g, m);
      u += sycl::permute_group_by_xor(sg, u, m);
    }
    if (w8 == 0 && pr < NBMAX * SPW) {
      half hv = (half)0.f;
      if (b < NB) {
        const float sc = 256.f * L.sm[128 + b];
        g *= sc;
        u *= sc;
        hv = toh<DT>(g / (1.f + sycl::exp(-g)) * u);
      }
      a_put_perm(SB.hs_a, b, i, hv);
    }
  }
  wg_bar(grp);
  // top-8 per row (ties: lower index first), softmax over the selection
  if (s < NB) {
    const float* lgs = L.lg + s * NE;
    float lg[16];
#pragma unroll
    for (int i = 0; i < 16; ++i)
      lg[i] = lgs[lane * 16 + i];
    float lmax = -INFINITY, sel_v[TOPK];
    int seli[TOPK];
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      float bv = -INFINITY;
      int bi = 1 << 20;
#pragma unroll
      for (int i = 0; i < 16; ++i)
        if (lg[i] > bv) {
          bv = lg[i];
          bi = lane * 16 + i;
        }
      const float mv = sycl::reduce_over_group(sg, bv, sycl::maximum<float>());
      const int mi = sycl::reduce_over_group(sg, bv == mv ? bi : (1 << 20), sycl::minimum<int>());
#pragma unroll
      for (int i = 0; i < 16; ++i)
        if (lane * 16 + i == mi) lg[i] = -INFINITY;
      sel_v[k] = mv;
      seli[k] = mi;
      if (k == 0) lmax = mv;
    }
    float sum = 0.f, e[TOPK];
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      e[k] = sycl::exp(sel_v[k] - lmax);
      sum += e[k];
    }
    if (lane == 0)
#pragma unroll
      for (int k = 0; k < TOPK; ++k) {
        L.re[s * TOPK + k] = seli[k] & (NE - 1);
        L.rw[s * TOPK + k] = e[k] / sum;
      }
  }
  wg_bar(grp);
  if (s == 0) {  // union of the selections, ascending expert id: lane owns
                 // experts 16 lane .. +15 (bit masks, SG scan)
    uint32_t m16 = 0u;
#pragma unroll
    for (int j = 0; j < NB * TOPK; ++j) {
      const int e = L.re[j];
      if ((e >> 4) == lane) m16 |= 1u << (e & 15);
    }
    const int cnt = sycl::popcount(m16);
    int pos = sycl::exclusive_scan_over_group(sg, cnt, sycl::plus<int>());
    if (lane == 15) L.nu[0] = pos + cnt;
#pragma unroll
    for (int i = 0; i < 16; ++i)
      if (m16 >> i & 1u) L.uexp[pos++] = lane * 16 + i;
  }
  wg_bar(grp);
  {  // per (union expert, row) the row's routing weight (0 if the row did not
     // select it)
    const int nu0 = L.nu[0];
    if (lid < nu0 * NBMAX) {
      const int u = lid / NBMAX, b = lid % NBMAX, e = L.uexp[u];
      float w = 0.f;
      if (b < NB) {
#pragma unroll
        for (int k = 0; k < TOPK; ++k)
          if (L.re[b * TOPK + k] == e) w = L.rw[b * TOPK + k];
      }
      L.uw[u * NBMAX + b] = w;
    }
  }
  wg_bar(grp);
  const int nu = L.nu[0];

  // --- S6: routed gate|up, tasks (union expert u, row eighth j) dealt to the
  // WGs. Task rows: gate 32j .. +32 and the same up rows (+256) of w13 [E][512]
  // [2048]. SG: gate (s < 16) or up (s >= 16), K slice s&15 (128 K: 8 chunks,
  // one K block), two 16-row tiles (lane = row). Eight tasks per expert keep
  // the WGs' task counts within one small task of each other.
  {
    const int ntask = nu * 8, gu = s >> 4, ks = s & 15;
    auto wrow = [&](int task, int tl) {  // this lane's row of tile tl, K slice ks
      return P.w13 + ((size_t)L.uexp[task >> 3] * 2 * INTER + gu * INTER + (task & 7) * 32 + tl * 16 + lane) * HID +
             ks * QB;
    };
    // the task's block scale, loaded ahead of its weights (behind them it
    // waits for their last bytes)
    auto wsc = [&](int task) {
      return P
          .w13_s[((size_t)L.uexp[task >> 3] * (2 * INTER / QB) + (gu * INTER + (task & 7) * 32) / QB) * KB_HID + ks];
    };
    sycl::uint4 wb[2][8];
    float scn = 0.f;
    if (wg < ntask) {
      scn = wsc(wg);
#pragma unroll
      for (int tl = 0; tl < 2; ++tl)
        ld_rows<8>(wb[tl], wrow(wg, tl));
    }
    for (int task = wg; task < ntask; task += NWG) {
      const int u = task >> 3, j8 = task & 7;
      const int nxt = task + NWG;
      const bool more = nxt < ntask;
      // one N block (32 rows inside 128) and one K block per SG
      const float sc = scn;
      if (more) scn = wsc(nxt);
      v8f c[2];
#pragma unroll
      for (int tl = 0; tl < 2; ++tl) {  // each consumed tile is refilled with the next task's
        c[tl] = v8f(0.f);
        mm_rows_u<8>(c[tl], wb[tl], L.xa, ks * 8, lane);
        if (more) ld_rows<8>(wb[tl], wrow(nxt, tl));
      }
#pragma unroll
      for (int tl = 0; tl < 2; ++tl)
#pragma unroll
        for (int i = 0; i < 8; ++i)
          L.s6r[((s * 2 + tl) * 8 + i) * 16 + lane] = c[tl][i] * sc;
      wg_bar(grp);
      if (lid < NBMAX * 32) {  // h[b][32 j8 + i] = silu(g) * u * w_b -> perm A
                               // tiles of expert u (padding rows 0)
        const int b = lid >> 5, i = lid & 31;
        const int tl = i >> 4, l = i & 15;
        half hv = (half)0.f;
        if (b < NB) {
          float g = 0.f, up = 0.f;
#pragma unroll
          for (int k2 = 0; k2 < 16; ++k2) {
            g += L.s6r[((k2 * 2 + tl) * 8 + b) * 16 + l];
            up += L.s6r[(((16 + k2) * 2 + tl) * 8 + b) * 16 + l];
          }
          g *= 256.f;
          up *= 256.f;
          hv = toh<DT>(g / (1.f + sycl::exp(-g)) * up * L.uw[u * NBMAX + b]);
        }
        a_put_perm(SB.hA + (size_t)u * (INTER / 16) * 64, b, j8 * 32 + i, hv);
      }
      wg_bar(grp);
    }
  }
  signal(it, S.cnt + C_GU * 16);

  // --- S7: down for this WG's 64 output rows ; shared down ; all-reduce #2.
  // Blocks (union expert u, K quarter q: 64 K = 4 chunks = one 64-B line of
  // each row, row-tile pair p) of 2 x 16 rows (lane = row of w2 [E][2048][256]),
  // dealt round-robin to the SGs (8 per expert: the SGs' counts differ by at
  // most one block), 4 blocks in flight per SG; the first 2 are in flight
  // during the hand-off (they need the union only).
  const int nblk = nu * 8;
  auto brow = [&](int j, int ct) {  // j = (u, q, p): u = j >> 3, q = (j >> 1) & 3, p = j & 1
    return P.w2 + ((size_t)L.uexp[j >> 3] * HID + n0 + ((j & 1) * 2 + ct) * 16 + lane) * INTER + ((j >> 1) & 3) * 64;
  };
  // block j's scale (the WG's 64 rows: one N block; K block q / 2), loaded
  // ahead of the block's weights
  auto bsc = [&](int j) {
    return P.w2_s[((size_t)L.uexp[j >> 3] * KB_HID + n0 / QB) * (INTER / QB) + (((j >> 1) & 3) >> 1)];
  };
  constexpr int S7PRE = 2;
  sycl::uint4 wb7[4][2][4];
  float sc7[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int t = 0; t < S7PRE; ++t) {  // (not all 4: the counter poll would queue behind them)
    const int j = s + NSG * t;
    if (j < nblk) {
      sc7[t] = bsc(j);
#pragma unroll
      for (int ct = 0; ct < 2; ++ct)
        ld_rows<4>(wb7[t][ct], brow(j, ct));
    }
  }
  wait_for(it, S.cnt + C_GU * 16, NWG * call);
#pragma unroll
  for (int t = S7PRE; t < 4; ++t) {
    const int j = s + NSG * t;
    if (j < nblk) {
      sc7[t] = bsc(j);
#pragma unroll
      for (int ct = 0; ct < 2; ++ct)
        ld_rows<4>(wb7[t][ct], brow(j, ct));
    }
  }
  if (wg == 0 && lid == 0) {  // every WG has read the epoch and the mailbox tag
    cnt_ref(S.cnt[EPOCH]).store(call);
    cnt_ref(S.cnt[mb_tag<NB>()]).store(mtag);
  }
  {
    v8f acc[4];
#pragma unroll
    for (int ct = 0; ct < 4; ++ct)
      acc[ct] = v8f(0.f);
    for (int j0 = s; j0 < nblk; j0 += 4 * NSG) {
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const int j = j0 + NSG * t;
        if (j < nblk) {
          const int u = j >> 3, q = (j >> 1) & 3;
          const uint32_t* ha = SB.hA + (size_t)u * (INTER / 16) * 64;
          const float sc = sc7[t];
          v8f c[2];
#pragma unroll
          for (int ct = 0; ct < 2; ++ct) {
            c[ct] = v8f(0.f);
            mm_rows_u<4>(c[ct], wb7[t][ct], ha, q * 4, lane);
          }
          const int jn = j + 4 * NSG;
          if (jn < nblk) {
            sc7[t] = bsc(jn);
#pragma unroll
            for (int ct = 0; ct < 2; ++ct)
              ld_rows<4>(wb7[t][ct], brow(jn, ct));
          }
          if (j & 1) {  // (a uniform branch: indexing acc by p would move it to private memory)
            acc[2] += c[0] * sc;
            acc[3] += c[1] * sc;
          } else {
            acc[0] += c[0] * sc;
            acc[1] += c[1] * sc;
          }
        }
      }
    }
    {  // shared down: rows n0 + 16 tile + lane of shared w2 [2048][256], K
       // chunks 2*kg..+2 (K block kg / 4)
      const int tile = s & 3, kg = s >> 2;
      sycl::uint4 sw[2];
      ld_rows<2>(sw, P.sw2 + (size_t)(n0 + tile * 16 + lane) * INTER + kg * 32);
      v8f d = v8f(0.f);
      mm_rows_u<2>(d, sw, SB.hs_a, 2 * kg, lane);
      const float sc = P.sw2_s[(n0 / QB) * (INTER / QB) + (kg >> 2)];
#pragma unroll
      for (int i = 0; i < 8; ++i)
        acc[tile][i] += d[i] * sc * (i < NB ? L.sgt[i] : 0.f);
    }
#pragma unroll
    for (int ct = 0; ct < 4; ++ct)
#pragma unroll
      for (int i = 0; i < 8; ++i)
        L.s6r[((s * 4 + ct) * 8 + i) * 16 + lane] = acc[ct][i];
  }
  wg_bar(grp);
  if (lid < NB * 32) {  // all-reduce #2, a column pair per work-item
    const int pb = lid >> 5, pc = (lid & 31) * 2, t = pc >> 4, l = pc & 15;
    float a0 = 0.f, a1 = 0.f;
#pragma unroll 8
    for (int sg2 = 0; sg2 < NSG; ++sg2) {
      a0 += L.s6r[((sg2 * 4 + t) * 8 + pb) * 16 + l];
      a1 += L.s6r[((sg2 * 4 + t) * 8 + pb) * 16 + l + 1];
    }
    a0 *= 256.f;
    a1 *= 256.f;
    ar_two(S.peer, S.mbox, MB1 + (((size_t)pb * HID + n0 + pc) >> 1), a0, a1, mtag, S.cnt);
    st16<DT>(P.hid_out, pb * HID + n0 + pc, a0);
    st16<DT>(P.hid_out, pb * HID + n0 + pc + 1, a1);
  }
}

// 8 bf16 -> 8 fp16 (the same rounding as a host-side bf16 -> fp16 conversion)
inline sycl::uint4 bf16x8_to_half(sycl::uint4 v) {
  sycl::uint4 r;
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const uint32_t w = v[q];
    const sycl::half lo = static_cast<sycl::half>(sycl::bit_cast<float>(w << 16));
    const sycl::half hi = static_cast<sycl::half>(sycl::bit_cast<float>(w & 0xffff0000u));
    r[q] = (uint32_t)sycl::bit_cast<uint16_t>(lo) | ((uint32_t)sycl::bit_cast<uint16_t>(hi) << 16);
  }
  return r;
}

// PRT pieces of this WG: router (fp16 values, tile s>>1 = experts 16t.., chunk
// pair s&1 of the WG's 64 dims, 2 uint4 per chunk), shared gate|up (fp8, rows
// s*16 + lane, the WG's 4 chunks)
template <typename DT>
inline void load_prt_b(const Ptrs& P, int wg, int s, int lane, sycl::uint4 (&prt_r)[4], sycl::uint4 (&prt_s)[4]) {
  // the router weight is read in the model's dtype (no fp16 copy); a bf16 one is
  // rounded to fp16 here, for the fp16 DPAS
  const e16* rr = P.router_w + (size_t)((s >> 1) * 16 + lane) * HID + wg * 64 + (s & 1) * 32;
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    prt_r[t] = *reinterpret_cast<const sycl::uint4*>(rr + t * 8);
    if constexpr (std::is_same_v<DT, bf16>) prt_r[t] = bf16x8_to_half(prt_r[t]);
  }
  ld_rows<4>(prt_s, P.sw13 + (size_t)(s * 16 + lane) * HID + wg * 64);
}

// the conv states (3 taps) of the WG's 320 channels of NR states -> csl
// [r][tap][320] (8 channels = 24 contiguous halves a load); conv_of(r) is the
// state's slot pointer, ch its elements per channel
template <int NR, typename ConvOf>
inline void load_conv_b(e16* csl, ConvOf conv_of, const int (&chs)[NR], int h, int c, int kh, int lid) {
  for (int v = lid; v < NR * 40; v += WGS) {
    const int r = v / 40, j8 = (v % 40) * 8;
    const int ch = j8 < 128   ? kh * DK + j8
                   : j8 < 256 ? KD + kh * DK + (j8 - 128)
                              : 2 * KD + h * DV + c * 64 + (j8 - 256);
    const e16* src = conv_of(r) + (size_t)ch * chs[r];
    e16 t[24];
    if (chs[r] == 3) {
      const sycl::vec<uint16_t, 8> a = *reinterpret_cast<const sycl::vec<uint16_t, 8>*>(src);
      const sycl::vec<uint16_t, 8> b = *reinterpret_cast<const sycl::vec<uint16_t, 8>*>(src + 8);
      const sycl::vec<uint16_t, 8> d = *reinterpret_cast<const sycl::vec<uint16_t, 8>*>(src + 16);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        t[i].u = a[i];
        t[8 + i].u = b[i];
        t[16 + i].u = d[i];
      }
    } else {
#pragma unroll
      for (int k = 0; k < 8; ++k)
#pragma unroll
        for (int i = 0; i < 3; ++i)
          t[k * 3 + i] = src[(size_t)k * chs[r] + i];
    }
#pragma unroll
    for (int k = 0; k < 8; ++k)
#pragma unroll
      for (int i = 0; i < 3; ++i)
        csl[(r * 3 + i) * 320 + j8 + k] = t[k * 3 + i];
  }
}

// --------------------------------------------------------- GDN layer (batched)
template <int NB, typename DT>
inline void gdn_layer_b(sycl::nd_item<1> it, uint8_t* slm, const Ptrs& P, const Scratch& S, const ScratchB& SB) {
  const int wg = it.get_group(0);
  const int lid = it.get_local_linear_id();
  sycl::sub_group sg = it.get_sub_group();
  const int s = sg.get_group_linear_id();
  const int lane = sg.get_local_linear_id();
  auto grp = it.get_group();
  const uint32_t call = uc_load32(S.cnt + EPOCH) + 1u;
  const SlmB L = slm_b(slm);
  float* qk = reinterpret_cast<float*>(slm + B_QK);  // per row r: q[0..127] k[128..255] v[256..319] beta g [320, 321]

  const int h = wg >> 1, c = wg & 1, kh = h >> 1;  // S2 role: v head h (k head h/2), v rows c*64..+64
  int sidx[NB], chs[NB];
#pragma unroll
  for (int b = 0; b < NB; ++b) {
    sidx[b] = P.state_idx[b];
    chs[b] = sidx[b] >= 0 ? P.conv_ch : 3;
  }
  auto conv_of = [&](int b) { return sidx[b] >= 0 ? P.conv_cache + (size_t)sidx[b] * P.conv_stride : S.dummy_conv; };
  auto ssm_of = [&](int b) { return sidx[b] >= 0 ? P.ssm_cache + (size_t)sidx[b] * P.ssm_stride : S.dummy_ssm; };

  // ---- kernel start: S1 weights in flight, the rows' recurrent-state blocks
  // of this WG -> L2
  const int tile1 = s >> 2, ks1 = s & 3;  // S1: 8 row tiles x 4 K slices (512)
  const int tz = s >> 3, kz = s & 7;
  // block scales of the S1 and z pieces, loaded ahead of the weight stream
  // (behind it they arrive with its last bytes and stall the first DPAS)
  float s1c[4], szc[2];
#pragma unroll
  for (int j = 0; j < 4; ++j)
    s1c[j] = P.qkvz_s[(size_t)((wg * 128 + tile1 * 16) / QB) * KB_HID + ks1 * 4 + j];
#pragma unroll
  for (int j = 0; j < 2; ++j)
    szc[j] = P.qkvz_s[(size_t)((CONVD + h * DV + c * 64 + tz * 16) / QB) * KB_HID + kz * 2 + j];
  sycl::uint4 w1[32];
  ld_rows<32>(w1, P.qkvz_w + (size_t)(wg * 128 + tile1 * 16 + lane) * HID + ks1 * 512);
#pragma unroll
  for (int b = 0; b < NB; ++b)  // 32 KB per row, one 64-B line per work-item
    syclex::prefetch(
        (void*)(ssm_of(b) + ((size_t)h * DV + c * 64) * DK + lid * 16), syclex::properties{syclex::prefetch_hint_L2});
  const int cch = lid < 128   ? kh * DK + lid
                  : lid < 256 ? KD + kh * DK + (lid - 128)
                              : 2 * KD + h * DV + c * 64 + (lid - 256);
  float cwt[4];
  e16* csl = reinterpret_cast<e16*>(slm + B_CS);  // [b][i][320]: conv state i of row b, the WG's channels
  load_conv_b<NB>(csl, conv_of, chs, h, c, kh, lid);
  if (lid < 320) {
    const sycl::vec<uint16_t, 4> w4 = *reinterpret_cast<const sycl::vec<uint16_t, 4>*>(P.conv_w + (size_t)cch * 4);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      cwt[i] = f32<DT>(w4[i]);
  }
  input_norm_b<NB, DT>(it, P, L);
  wg_bar(grp);

  // --- S1: q|k|v rows (128 per WG) -------------------------------------------
  sycl::uint4 zw[16], bw[16];
  {
    v8f cc = v8f(0.f);
    mm_rows_r<32>(cc, w1, L.xa, ks1 * 32, lane, s1c);
    // z rows of this WG (4 tiles x 8 K slices), b|a rows (2 tiles x 16 K
    // slices, 16-bit: two uint4 a chunk): in flight from here. Only head h's
    // column of each b|a tile is kept, so the other lanes feed zeros instead
    // of streaming all 32 rows into every WG.
    ld_rows<16>(zw, P.qkvz_w + (size_t)(CONVD + h * DV + c * 64 + tz * 16 + lane) * HID + kz * 256);
    if (lane == h)
      ld_rows<16>(bw, reinterpret_cast<const uint8_t*>(P.ba_h + (size_t)((s >> 4) * 16 + h) * HID + (s & 15) * 128));
    else
#pragma unroll
      for (int t = 0; t < 16; ++t)
        bw[t] = sycl::uint4(0u);
    red_put(L.red, s, lane, cc);
  }
  wg_bar(grp);
  {
    for (int idx = lid; idx < NB * 128; idx += WGS) {
      const int b = idx >> 7, r = idx & 127, t = r >> 4, l = r & 15;
      float v = 0.f;
#pragma unroll
      for (int k2 = 0; k2 < 4; ++k2)
        v += red_get(L.red, t * 4 + k2, b, l);
      SB.qkv[b * CONVD + wg * 128 + r] = toh<DT>(v * 256.f);
    }
  }
  signal(it, S.cnt + C_A * 16);
  {
    v8f cz = v8f(0.f), cb = v8f(0.f);
    mm_rows_r<16>(cz, zw, L.xa, kz * 16, lane, szc);
#pragma unroll
    for (int t = 0; t < 8; ++t)
      dpas_hf(
          cb,
          *reinterpret_cast<const v4u*>(L.xa + (((s & 15) * 8 + t) * 16 + lane) * 4),
          cvt16_w16<half>(bw[2 * t], bw[2 * t + 1]));
    wg_bar(grp);  // the S1 partials in L.red are consumed
    red_put(L.red, s, lane, cz);
    if (lane == h) {  // b (SGs 0..15) / a (SGs 16..31) of v head h
#pragma unroll
      for (int b = 0; b < NB; ++b)
        L.sm[256 + (s >> 4) * 128 + b * 16 + (s & 15)] = cb[b];
    }
  }
  wg_bar(grp);
  {
    for (int idx = lid; idx < NB * 64; idx += WGS) {
      const int b = idx >> 6, r = idx & 63, t = r >> 4, l = r & 15;
      float v = 0.f;
#pragma unroll
      for (int k2 = 0; k2 < 8; ++k2)
        v += red_get(L.red, t * 8 + k2, b, l);
      SB.z[b * VD + h * DV + c * 64 + r] = toh<DT>(v * 256.f);
    }
    if (lid < 2 * NB) {  // b, a of head h per row (DT, as SGLang's GEMM output)
      const int which = lid / NB, b = lid % NB;
      float v = 0.f;
#pragma unroll
      for (int k2 = 0; k2 < 16; ++k2)
        v += L.sm[256 + which * 128 + b * 16 + k2];
      L.sm[512 + which * 8 + b] = rnd<DT>(v);
    }
  }
  // recurrent state of all rows: rows v = c*64 + 2s + rr, k = lane*8..+8 (from
  // L2)
  sycl::vec<float, 8> st[NB][2];
#pragma unroll
  for (int b = 0; b < NB; ++b)
#pragma unroll
    for (int rr = 0; rr < 2; ++rr)
      st[b][rr] = *reinterpret_cast<const sycl::vec<float, 8>*>(
          ssm_of(b) + ((size_t)h * DV + c * 64 + 2 * s + rr) * DK + lane * 8);

  // --- S2: conv1d + SiLU, L2 norm, gated delta rule, all rows ----------------
  wait_for(it, S.cnt + C_A * 16, NWG * call);
  if (lid < 320) {
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      const float xnew = (float)SB.qkv[b * CONVD + cch];
      const e16 c0 = csl[(b * 3) * 320 + lid], c1 = csl[(b * 3 + 1) * 320 + lid], c2 = csl[(b * 3 + 2) * 320 + lid];
      float acc = f32<DT>(c0.u) * cwt[0];
      acc += f32<DT>(c1.u) * cwt[1];
      acc += f32<DT>(c2.u) * cwt[2];
      acc += xnew * cwt[3];
      qk[b * 384 + lid] = rnd<DT>(acc / (1.f + sycl::exp(-acc)));
      if (lid >= 256 || ((h & 1) == 0 && c == 0)) {
        e16* cv = conv_of(b) + (size_t)cch * chs[b];
        cv[0] = c1;
        cv[1] = c2;
        st16<DT>(cv, 2, xnew);
      }
    }
  }
  if (lid < NB) {  // beta, g of v head h
    const float bb = L.sm[512 + lid], a = L.sm[520 + lid] + ld16<DT>(P.dt_bias, h);
    const float sp = a < 20.f ? sycl::log(1.f + sycl::exp(a)) : a;
    qk[lid * 384 + 320] = 1.f / (1.f + sycl::exp(-bb));
    qk[lid * 384 + 321] = sycl::exp(-sycl::exp(P.A_log[h]) * sp);
  }
  wg_bar(grp);
  if (s < 2 * NB) {  // L2 norms: SG (row s/2, q | k), 8 values per lane
    const int b = s >> 1, which = s & 1;
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float v = qk[b * 384 + which * 128 + lane * 8 + i];
      ss += v * v;
    }
    ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
    if (lane == 0) L.sm[600 + b * 2 + which] = ss;
  }
  wg_bar(grp);
#pragma unroll
  for (int b = 0; b < NB; ++b) {
    const float* q_ = qk + b * 384;
    const float iq = 1.f / sycl::sqrt(L.sm[600 + b * 2] + 1e-6f), ik = 1.f / sycl::sqrt(L.sm[600 + b * 2 + 1] + 1e-6f);
    const float scale = 0.08838834764831845f;  // 1 / sqrt(128)
    float qv[8], kv8[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      qv[i] = q_[lane * 8 + i] * iq * scale;
      kv8[i] = q_[128 + lane * 8 + i] * ik;
    }
    const float beta = q_[320], g = q_[321];
    float kvm[2], o[2];
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
      float d = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        st[b][rr][i] *= g;
        d += st[b][rr][i] * kv8[i];
      }
      kvm[rr] = d;
    }
#pragma unroll
    for (int rr = 0; rr < 2; ++rr)
      kvm[rr] = sycl::reduce_over_group(sg, kvm[rr], sycl::plus<float>());
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
      const float delta = (q_[256 + 2 * s + rr] - kvm[rr]) * beta;
      float d = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        st[b][rr][i] += kv8[i] * delta;
        d += st[b][rr][i] * qv[i];
      }
      o[rr] = d;
    }
#pragma unroll
    for (int rr = 0; rr < 2; ++rr)
      o[rr] = sycl::reduce_over_group(sg, o[rr], sycl::plus<float>());
    if (lane < 2) SB.core[b * VD + h * DV + c * 64 + 2 * s + lane] = toh<DT>(lane == 0 ? o[0] : o[1]);
  }
  signal(it, S.cnt + C_B * 16);
#pragma unroll
  for (int b = 0; b < NB; ++b)
#pragma unroll
    for (int rr = 0; rr < 2; ++rr)  // state write-back after the hand-off
      st_store8(ssm_of(b) + ((size_t)h * DV + c * 64 + 2 * s + rr) * DK + lane * 8, st[b][rr]);
  sycl::uint4 ow[16], prt_r[4], prt_s[4];
  ld_rows<16>(ow, P.out_w + (size_t)(wg * 64 + (s >> 3) * 16 + lane) * VD + (s & 7) * 256);
  load_prt_b<DT>(P, wg, s, lane, prt_r, prt_s);

  // --- S3: gated RMSNorm (all heads, all rows) -> perm A tiles ; tail --------
  wait_for(it, S.cnt + C_B * 16, NWG * call);
  {
    sycl::vec<half, 4> o4[NB], z4[NB];
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      o4[b] = *reinterpret_cast<const sycl::vec<half, 4>*>(SB.core + b * VD + lid * 4);
      z4[b] = *reinterpret_cast<const sycl::vec<half, 4>*>(SB.z + b * VD + lid * 4);
      float ss = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i)
        ss += (float)o4[b][i] * (float)o4[b][i];
      ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
      if (lane == 0) L.sm[640 + b * 32 + s] = ss;  // head = 2 SGs
    }
    wg_bar(grp);
    const int hh = s >> 1;
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      const float rs = sycl::rsqrt((L.sm[640 + b * 32 + 2 * hh] + L.sm[640 + b * 32 + 2 * hh + 1]) / DV + EPS);
      half y[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float zf = (float)z4[b][i];
        y[i] = toh<DT>(
            (float)o4[b][i] * rs * ld16<DT>(P.gnorm_w, (lid * 4 + i) & (DV - 1)) * (zf / (1.f + sycl::exp(-zf))));
      }
      a_put4_perm(L.xa, b, lid * 4, y);
    }
    wg_bar(grp);
  }
  tail_b<NB, DT>(it, P, S, SB, L, call, C_O, ow, prt_r, prt_s);
}

// ---------------------------------------------- full-attention layer (batched)
// SLM (beyond the shared batched layout): q per row in DPAS A order, RoPE
// tables, new k|v, normalized q, merge weights
constexpr int ADYN_MIN = 96 * 1024;  // cached tokens per step from which the
                                     // attention schedules dynamically
constexpr int ATK = EPOCH + 8;       // batched attention: ticket counter per row [8] (reset each call)
constexpr int AB_QPK = 49152, AB_CS = AB_QPK + NBMAX * 4096, AB_KVN = AB_CS + NBMAX * 256,
              AB_QS = AB_KVN + NBMAX * 1024, AB_W = AB_QS + NQ * HD * 2, AB_SM3 = AB_W + 2048,
              AB_LST = AB_SM3 + 1024,  // [8][512] i16 (no reader in the current kernels; kept so that the offsets
                                       // after it, and the SLM layout of the measured kernels, do not move)
    AB_WL = AB_LST + NBMAX * 512 * 2,  // [8][512] f32 (likewise)
    AB_TK = AB_WL + NBMAX * 512 * 4,   // ticket ring [16][2] + ack [16]
    AB_BYTES = AB_TK + 256;
static_assert(AB_BYTES <= 131072, "SLM of the batched attention layer");
constexpr int AB_SLM = AB_BYTES > B_BYTES ? AB_BYTES : B_BYTES;  // the tail reuses the (dead) attention data
static_assert(AB_SLM <= 131072, "SLM of the batched attention layer + tail");

// q norm + RoPE of (row b, head hh) straight into the dpas A order:
// qpk[(kt*16 + l)*4 + j] = (q[2j + l/8][kt*16 + 2(l%8)], +1): head hh, dim
// lane*16 + i -> half ((lane*16 + (hh&1)*8 + i/2)*4 + hh/2)*2 + i%2
template <typename DT>
inline void
q_norm_rope_b(sycl::sub_group sg, int lane, const Ptrs& P, const float* src, const float* cs, e16* qh, int hh) {
  float y[16];
#pragma unroll
  for (int i = 0; i < 16; ++i)
    y[i] = rnd<DT>(src[lane * 16 + i]);
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 16; ++i)
    ss += y[i] * y[i];
  ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
  const float rs = sycl::rsqrt(ss / HD + EPS);
#pragma unroll
  for (int i = 0; i < 16; ++i)
    y[i] = y[i] * rs * (1.f + ld16<DT>(P.q_norm_w, lane * 16 + i));
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const float other = sycl::permute_group_by_xor(sg, y[i], 2);
    if (lane < 4) {
      const int fi = (lane & 1) * 16 + i;
      y[i] = lane < 2 ? y[i] * cs[fi] - other * cs[32 + fi] : y[i] * cs[fi] + other * cs[32 + fi];
    }
  }
#pragma unroll
  for (int i = 0; i < 16; ++i)
    st16<DT>(qh, ((lane * 16 + (hh & 1) * 8 + (i >> 1)) * 4 + (hh >> 1)) * 2 + (i & 1), y[i]);
}

// the context-chunk attention of the batched / verify kernels: rows' chunks
// (row b: lcs[b] cached tokens of request rq(b)) over the SG pairs -> partials
// SB.pa_o / pa_ml; coff: chunk offsets (static) or pair groups (dyn)
template <int NB, typename DT, typename RqOf>
inline void attn_chunks_b(
    sycl::nd_item<1> it,
    uint8_t* slm,
    const Ptrs& P,
    const Scratch& S,
    const ScratchB& SB,
    const int (&lcs)[NB],
    const int (&coff)[NB + 1],
    bool dyn,
    int ach,
    RqOf rq) {
  const int wg = it.get_group(0);
  sycl::sub_group sg = it.get_sub_group();
  const int s = sg.get_group_linear_id();
  const int lane = sg.get_local_linear_id();
  const uint32_t* qpk_all = reinterpret_cast<const uint32_t*>(slm + AB_QPK);
  if (dyn) {
    // row groups: row b owns the pairs coff[b] .. coff[b + 1] - 1 (in
    // proportion to its length, at least one if not empty); the group takes
    // units of 16 tokens as tickets from the row's counter (tickets of 4
    // units for the first 3/4, then single units), so the pairs' speed
    // differences even out inside the group. One partial per pair (slot =
    // pair): the merge is the chunk one.
    const int dh = s & 1, pl = s >> 1, pr = wg * 16 + pl;
    int* tkr = reinterpret_cast<int*>(slm + AB_TK);  // [16][2]
    int* ack = tkr + 32;                             // [16]
    int b = 0;
#pragma unroll
    for (int bb = 1; bb < NB; ++bb)
      if (coff[bb] <= pr) b = bb;
    if (pr < coff[NB]) {
      const int lc = lcs[b];
      const int* btab = P.block_table + (size_t)rq(b) * P.bt_stride;
      const uint32_t* qpk = qpk_all + b * 1024;
      v8f O[8];
#pragma unroll
      for (int nt = 0; nt < 8; ++nt)
        O[nt] = v8f(0.f);
      float m[8], lp[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        m[i] = -INFINITY;
        lp[i] = 0.f;
      }
      using lat = sycl::atomic_ref<
          int,
          sycl::memory_order::relaxed,
          sycl::memory_scope::work_group,
          sycl::access::address_space::local_space>;
      const int nu = (lc + 15) / 16, na = nu * 3 / 16;
#pragma unroll 1
      for (int k = 0;; ++k) {
        int t = 0;
        if (lane == 0) {
          if (dh == 0) {
            if (k >= 2)
              while (lat(ack[pl]).load(sycl::memory_order::acquire) < k - 2) {
              }
            t = (int)cnt_ref(S.cnt[ATK + b]).fetch_add(1u, sycl::memory_order::relaxed);
            lat(tkr[pl * 2 + (k & 1)]).store((k << 16) | sycl::min(t, 65535), sycl::memory_order::release);
          } else {
            int v;
            do {
              v = lat(tkr[pl * 2 + (k & 1)]).load(sycl::memory_order::acquire);
            } while ((v >> 16) != k);
            t = v & 0xffff;
            lat(ack[pl]).store(k, sycl::memory_order::release);
          }
        }
        t = sycl::select_from_group(sg, t, 0);
        const int u = t < na ? 4 * t : 4 * na + (t - na);
        if (u >= nu) break;
        const int t0 = u * 16, t_end = sycl::min(lc, (u + (t < na ? 4 : 1)) * 16);
        auto trow = [&](int tk) {   // slot of token tk of row b; past the
          if (tk >= t_end) tk = 0;  // segment: token 0 (masked by the score select)
          return kv_slot(btab, P.ps, tk);
        };
        if ((lc & 15) && (lc & ~15) >= t0 && (lc & ~15) < t_end)  // stale V rows of the partial tile -> 0
          zero_v_tail(P, trow(lc & ~15), lc & 15, dh, lane);
        {
          const int lc = t_end;  // the step masks columns >= lc
          int tq = t0;
#pragma unroll 1
          for (; tq + 48 < lc; tq += 64)
            MONO_ATTN_STEP(tq, 4);
#pragma unroll 1
          for (; tq < lc; tq += 16)
            MONO_ATTN_STEP(tq, 1);
        }
      }
      const int g = pr;
#pragma unroll
      for (int i = 0; i < 8; ++i) {  // partial: normalized o of this dims half, (m, l)
        const float l = sycl::reduce_over_group(sg, lp[i], sycl::plus<float>());
        const float il = l > 0.f ? 1.f / l : 0.f;
        half* dst = SB.pa_o + ((size_t)g * NQ + i) * HD + dh * 128;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt)
          dst[nt * 16 + lane] = (half)(O[nt][i] * il);
        if (dh == 0 && lane == 0) {
          SB.pa_ml[((size_t)g * NQ + i) * 2] = m[i];
          SB.pa_ml[((size_t)g * NQ + i) * 2 + 1] = l;
        }
      }
    }
  } else {
    const int dh = s & 1, g = wg * 16 + (s >> 1);
    if (g < coff[NB]) {
      int b = 0;
#pragma unroll
      for (int bb = 1; bb < NB; ++bb)
        if (coff[bb] <= g) b = bb;
      const int t0 = (g - coff[b]) * ach;
      const int lcr = lcs[b];
      const int lc = sycl::min(lcr, t0 + ach);  // end of this chunk (masks the tokens of the next one)
      const int* btab = P.block_table + (size_t)rq(b) * P.bt_stride;
      const uint32_t* qpk = qpk_all + b * 1024;
      auto trow = [&](int t) {  // slot of token t of row b; past the chunk:
        if (t >= lc) t = 0;     // token 0 (masked by the score select)
        return kv_slot(btab, P.ps, t);
      };
      v8f O[8];
#pragma unroll
      for (int nt = 0; nt < 8; ++nt)
        O[nt] = v8f(0.f);
      float m[8], lp[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        m[i] = -INFINITY;
        lp[i] = 0.f;
      }
      if ((lcr & 15) && (lcr & ~15) >= t0 && (lcr & ~15) < lc)  // stale V rows of the partial tile -> 0
        zero_v_tail(P, trow(lcr & ~15), lcr & 15, dh, lane);
      int t = t0;
#pragma unroll 1
      for (; t + 48 < lc; t += 64)
        MONO_ATTN_STEP(t, 4);
#pragma unroll 1
      for (; t < lc; t += 16)
        MONO_ATTN_STEP(t, 1);
#pragma unroll
      for (int i = 0; i < 8; ++i) {  // chunk partial: normalized o of this dims half, (m, l)
        const float l = sycl::reduce_over_group(sg, lp[i], sycl::plus<float>());
        const float il = l > 0.f ? 1.f / l : 0.f;
        half* dst = SB.pa_o + ((size_t)g * NQ + i) * HD + dh * 128;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt)
          dst[nt * 16 + lane] = (half)(O[nt][i] * il);
        if (dh == 0 && lane == 0) {
          SB.pa_ml[((size_t)g * NQ + i) * 2] = m[i];
          SB.pa_ml[((size_t)g * NQ + i) * 2 + 1] = l;
        }
      }
    }
  }
}

// the chunk schedule of NB rows of lcs[b] cached tokens: dynamic row groups
// for long steps, static chunks (no ticket atomics) for short ones
template <int NB>
inline bool attn_schedule(const int (&lcs)[NB], int (&coff)[NB + 1], int& ach) {
  int tot = 0;
#pragma unroll
  for (int b = 0; b < NB; ++b)
    tot += lcs[b];
  const bool dyn = tot >= ADYN_MIN;
  ach = 0;
  if (dyn) {  // row groups: coff[b] = first pair of row b, by length; at least
              // one pair per non-empty row
    int ne = 0;
#pragma unroll
    for (int b = 0; b < NB; ++b)
      ne += lcs[b] > 0;
    coff[0] = 0;
    long cum = 0;
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      cum += lcs[b];
      ne -= lcs[b] > 0;
      int e = tot > 0 ? (int)((cum * APAIRS + tot / 2) / tot) : 0;
      e = sycl::max(e, coff[b] + (lcs[b] > 0 ? 1 : 0));
      e = sycl::min(e, APAIRS - ne);
      coff[b + 1] = e;
    }
  } else {  // chunks: coff[b] = first chunk of row b
    ach = sycl::max(16, ((tot + APAIRS - NB - 1) / (APAIRS - NB) + 15) & ~15);
    coff[0] = 0;
#pragma unroll
    for (int b = 0; b < NB; ++b)
      coff[b + 1] = coff[b] + (lcs[b] + ach - 1) / ach;
  }
  return dyn;
}

template <int NB, typename DT>
inline void attn_layer_b(sycl::nd_item<1> it, uint8_t* slm, const Ptrs& P, const Scratch& S, const ScratchB& SB) {
  const int wg = it.get_group(0);
  const int lid = it.get_local_linear_id();
  sycl::sub_group sg = it.get_sub_group();
  const int s = sg.get_group_linear_id();
  const int lane = sg.get_local_linear_id();
  auto grp = it.get_group();
  const uint32_t call = uc_load32(S.cnt + EPOCH) + 1u;
  const SlmB L = slm_b(slm);
  // the single-row kernel takes gate/k/v chunks from a ticket counter whose
  // base it derives from the call number: every call must consume the same
  // number of tickets
  if (wg == 0 && lid == 0) cnt_ref(S.cnt[C_TKT * 16]).fetch_add((uint32_t)GKV_CHUNKS + NSG * NWG);
  uint32_t* qpk_all = reinterpret_cast<uint32_t*>(slm + AB_QPK);
  float* rope_all = reinterpret_cast<float*>(slm + AB_CS);
  e16* kvn_all = reinterpret_cast<e16*>(slm + AB_KVN);
  float* wgt = reinterpret_cast<float*>(slm + AB_W);
  float* sm3 = reinterpret_cast<float*>(slm + AB_SM3);
  auto qrow = [](int qi) { return (qi >> 8) * 512 + (qi & 255); };
  auto grow = [](int gi) { return gi < NQ * HD ? (gi >> 8) * 512 + 256 + (gi & 255) : NQ * 512 + (gi - NQ * HD); };
  // context chunks: one per SG pair at most; the chunk length ach (a multiple
  // of 16) is sized per call so that the rows' chunks (row b: ceil(lc_b / ach),
  // chunks coff[b] .. coff[b + 1] - 1) fit the APAIRS pairs
  int lcs[NB], coff[NB + 1];
#pragma unroll
  for (int b = 0; b < NB; ++b)
    lcs[b] = sycl::max(P.seq_lens[b] - 1, 0);
  int ach;
  const bool dyn = attn_schedule<NB>(lcs, coff, ach);

  // ---- S1: q rows (64 per WG: 4 tiles x 8 K slices), weights in flight before
  // the input norm
  sycl::uint4 wq[16];
  const int qt = qrow(wg * 64 + (s >> 3) * 16);
  float sq[2];  // block scales, loaded ahead of the weight stream
#pragma unroll
  for (int j = 0; j < 2; ++j)
    sq[j] = P.qkv_s[(size_t)(qt / QB) * KB_HID + (s & 7) * 2 + j];
  ld_rows<16>(wq, P.qkv_w + (size_t)(qt + lane) * HID + (s & 7) * 256);
  input_norm_b<NB, DT>(it, P, L);
  wg_bar(grp);
  {
    v8f cc = v8f(0.f);
    mm_rows_r<16>(cc, wq, L.xa, (s & 7) * 16, lane, sq);
    red_put(L.red, s, lane, cc);
  }
  wg_bar(grp);
  {
    for (int idx = lid; idx < NB * 64; idx += WGS) {
      const int b = idx >> 6, r = idx & 63, t = r >> 4, l = r & 15;
      float v = 0.f;
#pragma unroll
      for (int k2 = 0; k2 < 8; ++k2)
        v += red_get(L.red, t * 8 + k2, b, l);
      SB.qkv_raw[b * QKV_ROWS + qrow(wg * 64 + r)] = v * 256.f;
    }
  }
  signal(it, S.cnt + C_A * 16);
  if (lid < 32 * NB) rope_cs((float)P.pos[lid >> 5], lid & 31, rope_all + (lid >> 5) * 64);

  if (lid < 48) reinterpret_cast<int*>(slm + AB_TK)[lid] = -1;  // ticket ring / acks: nothing yet
  // ---- S2: q norm + RoPE per row, attention over the flattened context chunks
  // -> partials (global)
  wait_for(it, S.cnt + C_A * 16, NWG * call);
  for (int p = s; p < NB * NQ; p += NSG) {  // (row, head) per SG
    const int b = p / NQ, hh = p % NQ;
    q_norm_rope_b<DT>(
        sg,
        lane,
        P,
        SB.qkv_raw + b * QKV_ROWS + hh * 512,
        rope_all + b * 64,
        reinterpret_cast<e16*>(qpk_all + b * 1024),
        hh);
  }
  wg_bar(grp);
  attn_chunks_b<NB, DT>(it, slm, P, S, SB, lcs, coff, dyn, ach, [](int b) { return b; });
  // ---- gate / k / v rows (80 per WG: 5 tiles x 4 K slices, SGs 0..19)
  {
    sycl::uint4 wg4[32];
    const int t = s >> 2, ks = s & 3;
    const int gt = grow(wg * 80 + t * 16);
    float sg4[4] = {0.f, 0.f, 0.f, 0.f};  // block scales, ahead of the weights
    if (s < 20)
#pragma unroll
      for (int j = 0; j < 4; ++j)
        sg4[j] = P.qkv_s[(size_t)(gt / QB) * KB_HID + ks * 4 + j];
    if (s < 20) ld_rows<32>(wg4, P.qkv_w + (size_t)(gt + lane) * HID + ks * 512);
    v8f cc = v8f(0.f);
    if (s < 20) mm_rows_r<32>(cc, wg4, L.xa, ks * 32, lane, sg4);
    red_put(L.red, s, lane, cc);
  }
  wg_bar(grp);
  {
    for (int idx = lid; idx < NB * 80; idx += WGS) {
      const int b = idx / 80, r = idx % 80, t = r >> 4, l = r & 15;
      float v = 0.f;
#pragma unroll
      for (int k2 = 0; k2 < 4; ++k2)
        v += red_get(L.red, t * 4 + k2, b, l);
      SB.qkv_raw[b * QKV_ROWS + grow(wg * 80 + r)] = v * 256.f;
    }
  }
  signal(it, S.cnt + C_B * 16);
  sycl::uint4 ow[16], prt_r[4], prt_s[4];
  ld_rows<16>(ow, P.out_w + (size_t)(wg * 64 + (s >> 3) * 16 + lane) * (NQ * HD) + (s & 7) * 256);

  // ---- S3: per row: new k (norm + RoPE) / v, K / V write (WG 0), merge the
  // row's chunk partials + the new token, gate
  wait_for(it, S.cnt + C_B * 16, NWG * call);
  {
    const int hh = wg >> 2, d0 = (wg & 3) * 64;
    if (s < NB) {
      const int b = s;
      const int64_t slot = P.slot[b];
      head_norm_rope<DT>(
          sg,
          lane,
          SB.qkv_raw + b * QKV_ROWS + NQ * 512,
          P.k_norm_w,
          rope_all + b * 64,
          kvn_all + b * 512,
          (wg == 0 && slot >= 0) ? P.kc + slot * HD : nullptr);
    } else if (s >= 16 && s < 16 + NB) {
      const int b = s - 16;
      const int64_t slot = P.slot[b];
      head_norm_rope<DT>(
          sg,
          lane,
          SB.qkv_raw + b * QKV_ROWS + NQ * 512 + HD,
          nullptr,
          rope_all + b * 64,
          kvn_all + b * 512 + HD,
          (wg == 0 && slot >= 0) ? P.vc + slot * HD : nullptr);
    }
    wg_bar(grp);
    if (s < NB) {  // new-token score of head hh, row s
      float d = 0.f;
#pragma unroll
      for (int i = 0; i < 16; ++i) {  // q[hh][dd] from the packed q: kt = dd/16, l = (hh&1)*8 +
                                      // (dd%16)/2, j = hh/2
        const int dd = lane * 16 + i, kt = dd >> 4, l = (hh & 1) * 8 + ((dd & 15) >> 1);
        const float qv =
            ld16<DT>(reinterpret_cast<const e16*>(qpk_all + s * 1024), ((kt * 16 + l) * 4 + (hh >> 1)) * 2 + (dd & 1));
        d += qv * ld16<DT>(kvn_all, s * 512 + dd);
      }
      d = sycl::reduce_over_group(sg, d, sycl::plus<float>());
      if (lane == 0) sm3[s] = d * SM_SCALE;
    }
    wg_bar(grp);
    {  // merge, all rows at once: a work-item per chunk for the maxima /
       // weights, T = WGS / NB work-items per row
      // for the weighted sums (deterministic: atomic max only, fixed-order
      // sums)
      int* mkey = reinterpret_cast<int*>(sm3 + 16);
      float* lwp = sm3 + 32;  // [NB][TW / 16] per-SG weight sums
      auto key = [](float f) {
        const int k = sycl::bit_cast<int>(f);
        return k >= 0 ? k : k ^ 0x7fffffff;
      };
      auto unkey = [](int k) { return sycl::bit_cast<float>(k >= 0 ? k : k ^ 0x7fffffff); };
      if (lid < NB) mkey[lid] = key(sm3[lid]);
      wg_bar(grp);
      const int g = lid;
      int gb = 0;
#pragma unroll
      for (int bb = 1; bb < NB; ++bb)
        if (coff[bb] <= g) gb = bb;
      const bool live = g < coff[NB];
      float mg = -INFINITY, lg = 0.f;
      if (live) {
        mg = SB.pa_ml[((size_t)g * NQ + hh) * 2];
        lg = SB.pa_ml[((size_t)g * NQ + hh) * 2 + 1];
      }
      if (live && lg > 0.f) {
        sycl::atomic_ref<
            int,
            sycl::memory_order::relaxed,
            sycl::memory_scope::work_group,
            sycl::access::address_space::local_space>
            a(mkey[gb]);
        a.fetch_max(key(mg));
      }
      wg_bar(grp);
      if (live) wgt[g] = lg > 0.f ? lg * sycl::native::exp2(mg - unkey(mkey[gb])) : 0.f;
      wg_bar(grp);
      constexpr int TW = WGS / NB / 64 * 64, PARTS = TW / 64;  // whole SGs per row (NB = 3, 6: idle tail)
      const int b = lid / TW, r = lid % TW, part = r >> 6, d = r & 63;
      if (b < NB) {
        const int g0 = coff[b], n = coff[b + 1] - coff[b];
        {
          float lw = 0.f;
          for (int i2 = r; i2 < n; i2 += TW)
            lw += wgt[g0 + i2];
          lw = sycl::reduce_over_group(sg, lw, sycl::plus<float>());
          if (lane == 0) lwp[b * (TW / 16) + (r >> 4)] = lw;
        }
        float acc = 0.f;
        {  // 8 independent loads in flight (the loop is latency bound otherwise)
          float a8[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
          int i2 = part;
          for (; i2 + 7 * PARTS < n; i2 += 8 * PARTS) {
#pragma unroll
            for (int q = 0; q < 8; ++q)
              a8[q] +=
                  wgt[g0 + i2 + q * PARTS] * (float)SB.pa_o[((size_t)(g0 + i2 + q * PARTS) * NQ + hh) * HD + d0 + d];
          }
          for (; i2 < n; i2 += PARTS)
            a8[0] += wgt[g0 + i2] * (float)SB.pa_o[((size_t)(g0 + i2) * NQ + hh) * HD + d0 + d];
#pragma unroll
          for (int q = 0; q < 8; ++q)
            acc += a8[q];
        }
        L.red[(b * PARTS + part) * 64 + d] = acc;
      }
      wg_bar(grp);
      if (lid < NB * 64) {
        const int b2 = lid >> 6, d2 = lid & 63;
        const float M = unkey(mkey[b2]);
        const float en = sycl::native::exp2(sm3[b2] - M);
        float Lw = en;
        for (int k2 = 0; k2 < TW / 16; ++k2)
          Lw += lwp[b2 * (TW / 16) + k2];
        float a = en * ld16<DT>(kvn_all, b2 * 512 + HD + d0 + d2);
#pragma unroll
        for (int pp = 0; pp < PARTS; ++pp)
          a += L.red[(b2 * PARTS + pp) * 64 + d2];
        const float o = a / Lw;
        const float gate = SB.qkv_raw[b2 * QKV_ROWS + hh * 512 + 256 + d0 + d2];
        SB.attn_out[b2 * NQ * HD + hh * HD + d0 + d2] = toh<DT>(o / (1.f + sycl::exp(-gate)));
      }
    }
  }
  signal(it, S.cnt + C_C * 16);
  load_prt_b<DT>(P, wg, s, lane, prt_r, prt_s);

  // ---- S4: o (all heads, all rows) -> perm A tiles ; tail
  wait_for(it, S.cnt + C_C * 16, NWG * call);
  if (wg == 0 && lid < NB) cnt_ref(S.cnt[ATK + lid]).store(0u);  // every pair is done with the tickets
#pragma unroll
  for (int b = 0; b < NB; ++b) {
    const sycl::vec<half, 4> o4 = *reinterpret_cast<const sycl::vec<half, 4>*>(SB.attn_out + b * NQ * HD + lid * 4);
    const half y[4] = {o4[0], o4[1], o4[2], o4[3]};
    a_put4_perm(L.xa, b, lid * 4, y);
  }
  wg_bar(grp);
  tail_b<NB, DT>(it, P, S, SB, L, call, C_O, ow, prt_r, prt_s);
}

template <int NB, typename DT>
class MonoAttnBK;  // kernel names
template <int NB, typename DT>
class MonoGdnBK;

template <int NB, typename DT>
void launch_attn_b(sycl::queue& q, const Ptrs& P, const Scratch& S, const ScratchB& SB) {
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint8_t, 1> slm(sycl::range<1>(AB_SLM), h);
    syclex::properties props{syclex::sub_group_size<16>, intelex::grf_size<256>};
    h.parallel_for<MonoAttnBK<NB, DT>>(sycl::nd_range<1>(NWG * WGS, WGS), props, [=](sycl::nd_item<1> it) {
      attn_layer_b<NB, DT>(it, slm.get_multi_ptr<sycl::access::decorated::no>().get(), P, S, SB);
    });
  });
}

template <int NB, typename DT>
void launch_gdn_b(sycl::queue& q, const Ptrs& P, const Scratch& S, const ScratchB& SB) {
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint8_t, 1> slm(sycl::range<1>(B_BYTES), h);
    syclex::properties props{syclex::sub_group_size<16>, intelex::grf_size<256>};
    h.parallel_for<MonoGdnBK<NB, DT>>(sycl::nd_range<1>(NWG * WGS, WGS), props, [=](sycl::nd_item<1> it) {
      gdn_layer_b<NB, DT>(it, slm.get_multi_ptr<sycl::access::decorated::no>().get(), P, S, SB);
    });
  });
}

// decode launchers of one layer kind and dtype, 1, 2, 3, 4, 6 or 8 rows: each
// instance is compiled in its own translation unit and library
// (mono_{gdn,attn}_{fp16,bf16}, built with -fvisibility=hidden) and called
// from MonoDecodeXe20: exported
#pragma GCC visibility push(default)
template <typename DT>
void launch_gdn(sycl::queue& q, int nb, const Ptrs& P, const Scratch& S, const ScratchB& SB) {
  if (nb == 1)
    launch_t<false, DT>(q, P, S);
  else if (nb == 2)
    launch_gdn_b<2, DT>(q, P, S, SB);
  else if (nb == 3)
    launch_gdn_b<3, DT>(q, P, S, SB);
  else if (nb == 4)
    launch_gdn_b<4, DT>(q, P, S, SB);
  else if (nb == 6)
    launch_gdn_b<6, DT>(q, P, S, SB);
  else
    launch_gdn_b<8, DT>(q, P, S, SB);
}
template <typename DT>
void launch_attn(sycl::queue& q, int nb, const Ptrs& P, const Scratch& S, const ScratchB& SB) {
  if (nb == 1)
    launch_t<true, DT>(q, P, S);
  else if (nb == 2)
    launch_attn_b<2, DT>(q, P, S, SB);
  else if (nb == 3)
    launch_attn_b<3, DT>(q, P, S, SB);
  else if (nb == 4)
    launch_attn_b<4, DT>(q, P, S, SB);
  else if (nb == 6)
    launch_attn_b<6, DT>(q, P, S, SB);
  else
    launch_attn_b<8, DT>(q, P, S, SB);
}
#pragma GCC visibility pop

}  // namespace sgl_mono
