// SPDX-License-Identifier: Apache-2.0
//
// The full-attention decoder layer: input norm, qkv (q|gate per head, k,
// v), q/k norm + RoPE, paged attention split along the context with DPAS,
// merge with the new token, sigmoid gate, K / V write, then the MoE tail
// (moe.hpp). Single-row (attn_layer), batched (attn_layer_b) and
// speculative-decoding verify (attn_layer_s) versions.
//
// Verify kernels:
// Speculative-decoding verify kernels (mono decode, TP2): NB = NRQ requests x
// T tokens (T = 1 + draft tokens), rows request-major (row = b * T + t). The
// projections, the tail and the MoE are the batched code for NB rows; the GDN
// mixer runs each request's tokens in order.
//
// GatedDeltaNet state (SGLang's target-verify convention): the committed state
// is read, never written, and the state after each token goes to the request's
// scratch row (its batch position b):
// - recurrent state: from slot state_idx[b]; after token t, to
//   inter_ssm[b][t] ([rows][T][16][128][128] fp32);
// - conv state: the three taps of slot state_idx[b]; the window after each
//   token goes to inter_conv[b] ([rows][4096][T0 + 2] of the bound scratch's width
//   T0 >= T, channel stride inter_conv_cstride, viewed by SGLang as
//   [rows][T][4096][3] with window t = entries t .. t + 2): taps 1, 2 of the
//   committed state, then the T inputs.
// Attention: request b's context is seq_lens[b] - T cached tokens (page table
// row b); the T new k / v are written to out_cache_loc and merged causally.
//
// Stage code: every function that depends on the model's shape takes the
// traits Tr (Dims<Shape>, dims.hpp) as its first template argument.
#pragma once
#include "sgl_kernel/stage/xe20/moe.hpp"

namespace sgl_kernel::stage::xe20 {

// K tile: 16 token rows x 16 dims at dims dw*2.. (dw in dwords) of the K
// buffer (token pitch HD halves) -> VNNI B of K^T (lane = token)
template <class Tr>
inline v8u load_k_tile(const void* row0, int dw) {
#ifdef __SYCL_DEVICE_ONLY__
  return __builtin_IB_subgroup_block_read_flat_transpose_u32_k8(
      (intptr_t)row0, Tr::HD * 2 - 1, 15, Tr::HD * 2 - 1, v2i{dw, 0});
#else
  (void)row0;
  (void)dw;
  return v8u{};
#endif
}
// V tile: 16 token rows x 16 dims (VNNI-transformed) of a 128-dim half of the
// V buffer (token pitch HD halves)
template <class Tr>
inline v8u load_v_tile(const void* row0, int col) {
#ifdef __SYCL_DEVICE_ONLY__
  return __builtin_IB_subgroup_block_read_flat_transform_u16_k16(
      (intptr_t)row0, 128 * 2 - 1, 15, Tr::HD * 2 - 1, v2i{col, 0});
#else
  (void)row0;
  (void)col;
  return v8u{};
#endif
}
// token t of a request -> its slot (page table of page indices, ps tokens a
// page); a 16-token tile never crosses a page (ps % 16 == 0)
inline int64_t kv_slot(const int* btab, int ps, int t) {
  return (int64_t)btab[t / ps] * ps + (t % ps);
}

// one head (256) -> (GemmaRMS + NeoX RoPE on dims 0..63 if nw) -> DT dst (and
// dst2).
template <class Tr, typename DT>
inline void
head_norm_rope(sycl::sub_group sg, int lane, const float* src, const e16* nw, const float* cs, e16* dst, e16* dst2) {
  float y[16];
#pragma unroll
  for (int i = 0; i < 16; ++i)
    y[i] = rnd<DT>(src[lane * 16 + i]);
  if (nw) {
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i)
      ss += y[i] * y[i];
    ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
    const float rs = sycl::rsqrt(ss / Tr::HD + Tr::EPS);
#pragma unroll
    for (int i = 0; i < 16; ++i)
      y[i] = y[i] * rs * (1.f + ld16<DT>(nw, lane * 16 + i));
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const float other = sycl::permute_group_by_xor(sg, y[i], 2);
      if (lane < 4) {
        const int fi = (lane & 1) * 16 + i;
        y[i] = lane < 2 ? y[i] * cs[fi] - other * cs[32 + fi] : y[i] * cs[fi] + other * cs[32 + fi];
      }
    }
  }
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    st16<DT>(dst, lane * 16 + i, y[i]);
    if (dst2) st16<DT>(dst2, lane * 16 + i, y[i]);
  }
}

// RoPE cos / sin of position p for the 32 rotated pairs (i < 32): Cody-Waite
// reduction by 2 pi, then cos / sin of |r| <= pi
template <class Tr>
inline void rope_cs(float p, int i, float* cs) {
  const float inv = 1.0f / sycl::pow(Tr::THETA, (float)(2 * i) / 64.f);
  const float f = p * inv;
  const float k = sycl::rint(f * 0.15915494309189535f);
  float r = sycl::fma(-k, 6.28125f, f);
  r = sycl::fma(-k, 1.9353071795864769e-3f, r);
  cs[i] = sycl::cos(r);
  cs[32 + i] = sycl::sin(r);
}

// ----------------------------------------------------------------------------
// full-attention layer
// one flash step over NT 16-token tiles from token tb: trow(t) gives the slot
// of token t (past the context: a valid slot, masked by the score select)
#define MONO_ATTN_STEP(tb, NT)                                                                                 \
  do {                                                                                                         \
    const e16* kb[NT];                                                                                         \
    _Pragma("unroll") for (int tt = 0; tt < NT; ++tt) kb[tt] = P.kc + trow((tb) + tt * 16) * Tr::HD;           \
    v8f S4[NT];                                                                                                \
    _Pragma("unroll") for (int tt = 0; tt < NT; ++tt) {                                                        \
      S4[tt] = v8f(0.f);                                                                                       \
      _Pragma("unroll") for (int kt = 0; kt < 16; ++kt) {                                                      \
        const v8u bk = load_k_tile<Tr>(kb[tt], kt * 8);                                                        \
        const v4u qa = *reinterpret_cast<const v4u*>(qpk + (kt * 16 + lane) * 4);                              \
        dpas16<DT>(S4[tt], qa, bk);                                                                            \
      }                                                                                                        \
    }                                                                                                          \
    float sv[NT][8];                                                                                           \
    _Pragma("unroll") for (int tt = 0; tt < NT; ++tt) {                                                        \
      const bool ok = (tb) + tt * 16 + lane < lc; /* token = column = lane */                                  \
      _Pragma("unroll") for (int i = 0; i < 8; ++i) sv[tt][i] = ok ? S4[tt][i] * Tr::SM_SCALE : -INFINITY;     \
    }                                                                                                          \
    float alpha[8];                                                                                            \
    _Pragma("unroll") for (int i = 0; i < 8; ++i) {                                                            \
      float lm = sv[0][i];                                                                                     \
      _Pragma("unroll") for (int tt = 1; tt < NT; ++tt) lm = sycl::fmax(lm, sv[tt][i]);                        \
      const float mn = sycl::fmax(m[i], sycl::reduce_over_group(sg, lm, sycl::maximum<float>()));              \
      alpha[i] = mn == -INFINITY ? 1.f : sycl::native::exp2(m[i] - mn);                                        \
      m[i] = mn;                                                                                               \
      lp[i] *= alpha[i];                                                                                       \
    }                                                                                                          \
    _Pragma("unroll") for (int nt = 0; nt < 8; ++nt) _Pragma("unroll") for (int i = 0; i < 8; ++i) O[nt][i] *= \
        alpha[i];                                                                                              \
    _Pragma("unroll") for (int tt = 0; tt < NT; ++tt) {                                                        \
      uint16_t pv[8];                                                                                          \
      _Pragma("unroll") for (int i = 0; i < 8; ++i) {                                                          \
        const float p = m[i] == -INFINITY ? 0.f : sycl::native::exp2(sv[tt][i] - m[i]);                        \
        lp[i] += p;                                                                                            \
        pv[i] = bits<DT>(p);                                                                                   \
      }                                                                                                        \
      /* P (lane = token, element = head) -> dpas A: lane l, element j = P[2j                                  \
       * + l/8][2(l%8) + {0,1}] */                                                                             \
      v4u pa;                                                                                                  \
      _Pragma("unroll") for (int j = 0; j < 4; ++j) {                                                          \
        const int t0_ = 2 * (lane & 7);                                                                        \
        const uint16_t a0 = sycl::select_from_group(sg, pv[2 * j], t0_);                                       \
        const uint16_t a1 = sycl::select_from_group(sg, pv[2 * j + 1], t0_);                                   \
        const uint16_t b0 = sycl::select_from_group(sg, pv[2 * j], t0_ + 1);                                   \
        const uint16_t b1 = sycl::select_from_group(sg, pv[2 * j + 1], t0_ + 1);                               \
        pa[j] = (lane < 8) ? pack16(a0, b0) : pack16(a1, b1);                                                  \
      }                                                                                                        \
      const e16* vb = P.vc + (kb[tt] - P.kc) + dh * 128;                                                       \
      _Pragma("unroll") for (int nt = 0; nt < 8; ++nt) dpas16<DT>(O[nt], pa, load_v_tile<Tr>(vb, nt * 16));    \
    }                                                                                                          \
  } while (0)

// zero the V rows r0 .. 15 of the 16-token tile at slot `tile` (the stale rows
// after the context in the request's own page: a reused page may hold NaN, and
// 0 * NaN poisons P.V; later tokens of the request overwrite them anyway)
template <class Tr>
inline void zero_v_tail(const Ptrs& P, int64_t tile, int r0, int dh, int lane) {
  e16* vz = P.vc + tile * Tr::HD + dh * 128;
  for (int r = r0; r < 16; ++r)
    *reinterpret_cast<sycl::vec<half, 8>*>(vz + (size_t)r * Tr::HD + lane * 8) = sycl::vec<half, 8>(0.f);
  sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::work_group);
}

template <class Tr, typename DT, bool SPLIT = false>
inline void attn_layer(sycl::nd_item<1> it, uint8_t* slm, const Ptrs& P, const Scratch& S, const Split& X = Split{}) {
  const int wg = it.get_group(0);
  const int lid = it.get_local_linear_id();
  sycl::sub_group sg = it.get_sub_group();
  const int s = sg.get_group_linear_id();
  const int lane = sg.get_local_linear_id();
  auto grp = it.get_group();
  const uint32_t call = uc_load32(S.cnt + EPOCH) + 1u;

  const Slm L = slm_tail(slm, Tr::A_X, Tr::A_XN, Tr::A_KV, Tr::A_RED, Tr::A_ROUTE, Tr::A_AO, Tr::A_MR, Tr::A_HB);
  half* xp = L.xp;
  const uint32_t* xp32 = L.xp32;
  e16* qs = reinterpret_cast<e16*>(slm + Tr::A_Q);
  e16* kvn = reinterpret_cast<e16*>(slm + Tr::A_KV);
  float* red = L.red;
  float* st = reinterpret_cast<float*>(slm + Tr::A_ST);
  half* ao = reinterpret_cast<half*>(slm + Tr::A_AO);
  float* rope = reinterpret_cast<float*>(slm + Tr::A_CS);
  auto qrow = [](int qi) { return (qi >> 8) * 512 + (qi & 255); };
  auto grow = [](int gi) {
    return gi < Tr::NQ * Tr::HD ? (gi >> 8) * 512 + 256 + (gi & 255) : Tr::NQ * 512 + (gi - Tr::NQ * Tr::HD);
  };

  const int seqlen = P.seq_lens[0];
  const int lc = seqlen - 1;  // cached tokens (the new one is merged separately)
  const int* btab = P.block_table;
  // SG pair p = wg*16 + s/2 owns the cached tokens [p*T, p*T + T) of every
  // pass of APASS1 = 40960 tokens: 5 tiles of 16 a pass. The first pass is one
  // straight-line body (a loop around it is much slower; the 5th tile only
  // runs past 32K tokens); longer contexts take further passes in a loop.
  constexpr int T = 80;
  constexpr int APASS1 = Tr::NWG * 16 * T;
  const int t0 = (wg * 16 + (s >> 1)) * T;
  auto trow = [&](int t) {  // slot of token t; past the context: token 0
    if (t >= lc) t = 0;     // (masked by the score select)
    return kv_slot(btab, P.ps, t);
  };

  input_norm<Tr, DT>(it, P, nullptr, xp);
  wg_bar(grp);

  // ---------------- S1: q rows (2 per SG, one head, one N block) -------------
  if ((!SPLIT || X.seg == SEG_F1)) {
    float* qr = S.qkv_raw;
    const int q0 = (wg * Tr::NSG + s) * 2;
    sycl::uint4 w[2][8];
    float sc[8];
#pragma unroll
    for (int b = 0; b < 2; ++b)
      load_row<8>(w[b], P.qkv_w + (size_t)qrow(q0 + b) * Tr::HID, lane);
    load_scales<8>(sc, P.qkv_s + (size_t)(qrow(q0) / Tr::QB) * Tr::KB_HID, lane);
#pragma unroll
    for (int b = 0; b < 2; ++b) {
      const float v = dot_row_regs<8>(sg, lane, w[b], xp32, sc);
      if (lane == 0) qr[qrow(q0 + b)] = v * 256.f;
    }
    signal(it, S.cnt + C_A * 16);
  }
  if (SPLIT && X.seg == SEG_F1) return;
  {  // the q hand-off + q norm leave DRAM idle: warm L2 with the first K tile
     // of this SG pair
    if (t0 < lc) {
      const e16* kb = P.kc + trow(t0) * Tr::HD;
#pragma unroll
      for (int u = lane; u < 8 * 8; u += 16)
        syclex::prefetch(
            (void*)(kb + (size_t)((s & 1) * 8 + u / 8) * Tr::HD + (u % 8) * 32),
            syclex::properties{syclex::prefetch_hint_L2});
    }
  }
  if (lid < 32) rope_cs<Tr>((float)P.pos[0], lid, rope);

  // ---------------- S2: q norm + RoPE, attention split along the context -----
  // (split: the merge segment needs the normalized q too)
  if (!SPLIT || X.seg == SEG_F2 || X.seg == SEG_F3) {
    wait_for(it, S.cnt + C_A * 16, Tr::NWG * call);
    if (s < Tr::NQ) head_norm_rope<Tr, DT>(sg, lane, S.qkv_raw + s * 512, P.q_norm_w, rope, qs + s * Tr::HD, nullptr);
    wg_bar(grp);
  }
  if ((!SPLIT || X.seg == SEG_F2)) {
    // q in dpas A order: qpk[(kt*16 + l)*4 + j] = (q[2j + l/8][kt*16 + 2(l%8)],
    // .. + 1)
    uint32_t* qpk = reinterpret_cast<uint32_t*>(slm + Tr::A_QPK);
    for (int u = lid; u < 16 * 16 * 4; u += Tr::WGS) {
      const int j = u & 3, l = (u >> 2) & 15, kt = u >> 6;
      const e16* src = qs + (2 * j + (l >> 3)) * Tr::HD + kt * 16 + 2 * (l & 7);
      qpk[u] = pack16(src[0].u, src[1].u);
    }
    wg_bar(grp);
    {
      // SG unit: dh = s&1 (output dims half), chunks (wg*16 + s/2) + 512 i.
      // S^T = q[8 heads x 256] . K^T (A = q, B = K rows col_major); O[8 x 128] +=
      // P . V (B row_major). acc / A / B layout on Xe2: lane = column, element i
      // = row i.
      const int dh = s & 1;

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
      if (lc > 0 && t0 < lc) {
        // In the one partial tile the owner zeroes the stale V rows first
        if ((lc & 15) && (lc & ~15) >= t0 && (lc & ~15) < t0 + T) zero_v_tail<Tr>(P, trow(lc & ~15), lc & 15, dh, lane);
        // one flash step over NT tiles starting at token tb (a macro, not a
        // lambda: matrices captured by reference become generic pointers, which
        // IGC cannot lower)
        MONO_ATTN_STEP(t0, 4);
        if (t0 + 64 < lc) MONO_ATTN_STEP(t0 + 64, 1);
      }
      // further passes (contexts past APASS1), one tile a step: a second copy of
      // the first pass's body would slow the common case
#pragma unroll 1
      for (int pb = t0 + APASS1; pb < lc; pb += APASS1) {
#pragma unroll 1
        for (int tb = pb; tb < pb + T && tb < lc; tb += 16) {
          if ((lc & 15) && (lc & ~15) == tb) zero_v_tail<Tr>(P, trow(tb), lc & 15, dh, lane);
          MONO_ATTN_STEP(tb, 1);
        }
      }
      sycl::uint4 gmid[Tr::GKV_MID][8];  // gate/k/v rows of this SG, consumed after
                                         // the in-WG combine
      float gsc[8];
#pragma unroll
      for (int r = 0; r < Tr::GKV_MID; ++r)
        load_row<8>(gmid[r], P.qkv_w + (size_t)grow((wg * Tr::NSG + s) * Tr::GKV_MID + r) * Tr::HID, lane);
      load_scales<8>(gsc, P.qkv_s + (size_t)(grow((wg * Tr::NSG + s) * Tr::GKV_MID) / Tr::QB) * Tr::KB_HID, lane);
      {  // SG partial -> SLM: ao[s][head i][128 dims] (normalized fp16), st[s][i]
         // = (m, l)
        float il[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const float l = sycl::reduce_over_group(sg, lp[i], sycl::plus<float>());
          il[i] = l > 0.f ? 1.f / l : 0.f;
          if (lane == 0) {
            st[(s * 8 + i) * 2] = m[i];
            st[(s * 8 + i) * 2 + 1] = l;
          }
        }
#pragma unroll
        for (int nt = 0; nt < 8; ++nt)
#pragma unroll
          for (int i = 0; i < 8; ++i)
            ao[(s * 8 + i) * 128 + nt * 16 + lane] = (half)(O[nt][i] * il[i]);
      }
      wg_bar(grp);
      if (s < 2 * Tr::NQ) {  // in-WG combine: SG -> (head hh, dims half dh2); sources
                             // s' = gg*2 + dh2
        const int hh = s >> 1, dh2 = s & 1;
        float M = -INFINITY;
#pragma unroll
        for (int gg = 0; gg < 16; ++gg)
          M = sycl::fmax(M, st[((gg * 2 + dh2) * 8 + hh) * 2]);
        float Lw = 0.f, acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if (M != -INFINITY) {
#pragma unroll
          for (int gg = 0; gg < 16; ++gg) {
            const int ss2 = gg * 2 + dh2;
            const float wgt = st[(ss2 * 8 + hh) * 2 + 1] * sycl::native::exp2(st[(ss2 * 8 + hh) * 2] - M);
            Lw += wgt;
            const half* src = ao + (ss2 * 8 + hh) * 128 + lane * 8;
#pragma unroll
            for (int i = 0; i < 8; ++i)
              acc[i] += wgt * (float)src[i];
          }
        }
        const float iL = Lw > 0.f ? 1.f / Lw : 0.f;
        float* dst = S.part_o + ((size_t)wg * Tr::NQ + hh) * Tr::HD + dh2 * 128 + lane * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i)
          dst[i] = acc[i] * iL;
        if (dh2 == 0 && lane == 0) {
          S.part_ml[(wg * Tr::NQ + hh) * 2] = M;
          S.part_ml[(wg * Tr::NQ + hh) * 2 + 1] = Lw;
        }
      }
#pragma unroll
      for (int r = 0; r < Tr::GKV_MID; ++r) {
        const float v = dot_row_regs<8>(sg, lane, gmid[r], xp32, gsc);
        if (lane == 0) S.qkv_raw[grow((wg * Tr::NSG + s) * Tr::GKV_MID + r)] = v * 256.f;
      }
    }
    // remaining gate / k / v rows taken dynamically by sub-groups that finish
    // early (results land at fixed rows). The ticket counter is monotonic: every
    // call consumes exactly GKV_CHUNKS + NSG*NWG tickets.
    {
      float* qr = S.qkv_raw;
      const uint32_t base = (call - 1) * ((uint32_t)Tr::GKV_CHUNKS + Tr::NSG * Tr::NWG);
      while (true) {
        uint32_t t = 0;
        if (lane == 0) t = cnt_ref(S.cnt[C_TKT * 16]).fetch_add(1u, sycl::memory_order::relaxed);
        const int chunk = (int)(sycl::group_broadcast(sg, t, 0) - base);
        if (chunk >= Tr::GKV_CHUNKS) break;
        const int g0 = Tr::GKV_MID * Tr::NSG * Tr::NWG + chunk * Tr::GKV_ROWS;
        sycl::uint4 w[Tr::GKV_ROWS][8];
        float sc[8];
#pragma unroll
        for (int r = 0; r < Tr::GKV_ROWS; ++r)
          load_row<8>(w[r], P.qkv_w + (size_t)grow(g0 + r) * Tr::HID, lane);
        load_scales<8>(sc, P.qkv_s + (size_t)(grow(g0) / Tr::QB) * Tr::KB_HID, lane);
#pragma unroll
        for (int r = 0; r < Tr::GKV_ROWS; ++r) {
          const float v = dot_row_regs<8>(sg, lane, w[r], xp32, sc);
          if (lane == 0) qr[grow(g0 + r)] = v * 256.f;
        }
      }
    }
    signal(it, S.cnt + C_B * 16);
  }
  if (SPLIT && X.seg == SEG_F2) return;
  sycl::uint4 ow[2][8];  // o_proj rows of this SG -> registers, consumed after
                         // the merge
  float osc[8];
  load_row<8>(ow[0], P.out_w + (size_t)(wg * 64 + s * 2) * (Tr::NQ * Tr::HD), lane);
  load_row<8>(ow[1], P.out_w + (size_t)(wg * 64 + s * 2 + 1) * (Tr::NQ * Tr::HD), lane);
  load_scales<8>(osc, P.out_s + (size_t)((wg * 64) / Tr::QB) * (Tr::NQ * Tr::HD / Tr::QB), lane);

  // ---------------- S3: merge partials + new token, gate ; WG = (head, 64
  // dims) -----------------
  if ((!SPLIT || X.seg == SEG_F3)) {
    wait_for(it, S.cnt + C_B * 16, Tr::NWG * call);
    {
      const int hh = (wg * 64) / Tr::HD, d0 = (wg * 64) % Tr::HD;
      float* mrg = reinterpret_cast<float*>(slm + Tr::A_AO);  // [32][64]
      {  // new token: k (norm + rope) and v; WG 0 writes them to the cache
        const int64_t slot = P.slot[0];
        const bool wr = wg == 0 && slot >= 0;
        if (s == 0)
          head_norm_rope<Tr, DT>(
              sg, lane, S.qkv_raw + Tr::NQ * 512, P.k_norm_w, rope, kvn, wr ? P.kc + slot * Tr::HD : nullptr);
        else if (s == 1)
          head_norm_rope<Tr, DT>(
              sg,
              lane,
              S.qkv_raw + Tr::NQ * 512 + Tr::HD,
              nullptr,
              rope,
              kvn + Tr::HD,
              wr ? P.vc + slot * Tr::HD : nullptr);
        wg_bar(grp);
      }
      if (s == 0) {  // new-token score
        float d = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i)
          d += ld16<DT>(qs, hh * Tr::HD + lane * 16 + i) * ld16<DT>(kvn, lane * 16 + i);
        d = sycl::reduce_over_group(sg, d, sycl::plus<float>());
        if (lane == 0) red[96] = d * Tr::SM_SCALE;
      }
      if (lane == 0) red[s] = S.part_ml[(s * Tr::NQ + hh) * 2];
      wg_bar(grp);
      float M = red[96];
      for (int i = 0; i < 32; ++i)
        M = sycl::fmax(M, red[i]);
      const float wgt = S.part_ml[(s * Tr::NQ + hh) * 2 + 1] * sycl::native::exp2(red[s] - M);
      const float* src = S.part_o + ((size_t)s * Tr::NQ + hh) * Tr::HD + d0;
#pragma unroll
      for (int i = 0; i < 4; ++i)
        mrg[s * 64 + lane * 4 + i] = wgt * src[lane * 4 + i];
      if (lane == 0) red[32 + s] = wgt;
      wg_bar(grp);
      if (lid < 64) {
        const int d = d0 + lid;
        float Lw = sycl::native::exp2(red[96] - M), acc = Lw * ld16<DT>(kvn, Tr::HD + d);
        for (int i = 0; i < 32; ++i) {
          acc += mrg[i * 64 + lid];
          Lw += red[32 + i];
        }
        const float o = acc / Lw;
        const float gate = S.qkv_raw[hh * 512 + 256 + d];
        S.attn_out[hh * Tr::HD + d] = toh<DT>(o / (1.f + sycl::exp(-gate)));
      }
    }
    signal(it, S.cnt + C_C * 16);
  }
  if (SPLIT && X.seg == SEG_F3) return;
  sycl::uint4 prt_r[4], prt_s[Tr::SHR / 4];
  load_prt<Tr>(P, wg, s, lane, prt_r, prt_s);

  // ---------------- S4: o (all heads) -> SLM ; tail
  // -------------------------------------------
  if (!SPLIT || X.seg == SEG_T1 || X.seg == SEG_T1A || X.seg == SEG_T1B) {
    wait_for(it, S.cnt + C_C * 16, Tr::NWG * call);
    for (int k = lid; k < Tr::NQ * Tr::HD; k += Tr::WGS)
      store_perm(xp, k, S.attn_out[k]);
    wg_bar(grp);
  }
  tail<Tr, DT, SPLIT>(it, P, S, L, call, C_O, ow, osc, prt_r, prt_s, X);
}
// attention schedules dynamically
constexpr int ATK = EPOCH + 8;  // batched attention: ticket counter per row [8] (reset each call)

// q norm + RoPE of (row b, head hh) straight into the dpas A order:
// qpk[(kt*16 + l)*4 + j] = (q[2j + l/8][kt*16 + 2(l%8)], +1): head hh, dim
// lane*16 + i -> half ((lane*16 + (hh&1)*8 + i/2)*4 + hh/2)*2 + i%2
template <class Tr, typename DT>
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
  const float rs = sycl::rsqrt(ss / Tr::HD + Tr::EPS);
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
template <class Tr, int NB, typename DT, typename RqOf>
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
  const uint32_t* qpk_all = reinterpret_cast<const uint32_t*>(slm + Tr::AB_QPK);
  if (dyn) {
    // row groups: row b owns the pairs coff[b] .. coff[b + 1] - 1 (in
    // proportion to its length, at least one if not empty); the group takes
    // units of 16 tokens as tickets from the row's counter (tickets of 4
    // units for the first 3/4, then single units), so the pairs' speed
    // differences even out inside the group. One partial per pair (slot =
    // pair): the merge is the chunk one.
    const int dh = s & 1, pl = s >> 1, pr = wg * 16 + pl;
    int* tkr = reinterpret_cast<int*>(slm + Tr::AB_TK);  // [16][2]
    int* ack = tkr + 32;                                 // [16]
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
          zero_v_tail<Tr>(P, trow(lc & ~15), lc & 15, dh, lane);
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
        half* dst = SB.pa_o + ((size_t)g * Tr::NQ + i) * Tr::HD + dh * 128;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt)
          dst[nt * 16 + lane] = (half)(O[nt][i] * il);
        if (dh == 0 && lane == 0) {
          SB.pa_ml[((size_t)g * Tr::NQ + i) * 2] = m[i];
          SB.pa_ml[((size_t)g * Tr::NQ + i) * 2 + 1] = l;
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
        zero_v_tail<Tr>(P, trow(lcr & ~15), lcr & 15, dh, lane);
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
        half* dst = SB.pa_o + ((size_t)g * Tr::NQ + i) * Tr::HD + dh * 128;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt)
          dst[nt * 16 + lane] = (half)(O[nt][i] * il);
        if (dh == 0 && lane == 0) {
          SB.pa_ml[((size_t)g * Tr::NQ + i) * 2] = m[i];
          SB.pa_ml[((size_t)g * Tr::NQ + i) * 2 + 1] = l;
        }
      }
    }
  }
}

// the chunk schedule of NB rows of lcs[b] cached tokens: dynamic row groups
// for long steps, static chunks (no ticket atomics) for short ones
template <class Tr, int NB>
inline bool attn_schedule(const int (&lcs)[NB], int (&coff)[NB + 1], int& ach) {
  int tot = 0;
#pragma unroll
  for (int b = 0; b < NB; ++b)
    tot += lcs[b];
  const bool dyn = tot >= Tr::ADYN_MIN;
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
      int e = tot > 0 ? (int)((cum * Tr::APAIRS + tot / 2) / tot) : 0;
      e = sycl::max(e, coff[b] + (lcs[b] > 0 ? 1 : 0));
      e = sycl::min(e, Tr::APAIRS - ne);
      coff[b + 1] = e;
    }
  } else {  // chunks: coff[b] = first chunk of row b
    ach = sycl::max(16, ((tot + Tr::APAIRS - NB - 1) / (Tr::APAIRS - NB) + 15) & ~15);
    coff[0] = 0;
#pragma unroll
    for (int b = 0; b < NB; ++b)
      coff[b + 1] = coff[b] + (lcs[b] + ach - 1) / ach;
  }
  return dyn;
}

template <class Tr, int NB, typename DT, bool SPLIT = false>
inline void attn_layer_b(
    sycl::nd_item<1> it, uint8_t* slm, const Ptrs& P, const Scratch& S, const ScratchB& SB, const Split& X = Split{}) {
  const int wg = it.get_group(0);
  const int lid = it.get_local_linear_id();
  sycl::sub_group sg = it.get_sub_group();
  const int s = sg.get_group_linear_id();
  const int lane = sg.get_local_linear_id();
  auto grp = it.get_group();
  const uint32_t call = uc_load32(S.cnt + EPOCH) + 1u;
  const SlmB L = slm_b<Tr>(slm);
  // the single-row kernel takes gate/k/v chunks from a ticket counter whose
  // base it derives from the call number: every call must consume the same
  // number of tickets
  if ((!SPLIT || X.seg == SEG_F1) && wg == 0 && lid == 0)
    cnt_ref(S.cnt[C_TKT * 16]).fetch_add((uint32_t)Tr::GKV_CHUNKS + Tr::NSG * Tr::NWG);
  uint32_t* qpk_all = reinterpret_cast<uint32_t*>(slm + Tr::AB_QPK);
  float* rope_all = reinterpret_cast<float*>(slm + Tr::AB_CS);
  e16* kvn_all = reinterpret_cast<e16*>(slm + Tr::AB_KVN);
  float* wgt = reinterpret_cast<float*>(slm + Tr::AB_W);
  float* sm3 = reinterpret_cast<float*>(slm + Tr::AB_SM3);
  auto qrow = [](int qi) { return (qi >> 8) * 512 + (qi & 255); };
  auto grow = [](int gi) {
    return gi < Tr::NQ * Tr::HD ? (gi >> 8) * 512 + 256 + (gi & 255) : Tr::NQ * 512 + (gi - Tr::NQ * Tr::HD);
  };
  // context chunks: one per SG pair at most; the chunk length ach (a multiple
  // of 16) is sized per call so that the rows' chunks (row b: ceil(lc_b / ach),
  // chunks coff[b] .. coff[b + 1] - 1) fit the APAIRS pairs
  int lcs[NB], coff[NB + 1];
#pragma unroll
  for (int b = 0; b < NB; ++b)
    lcs[b] = sycl::max(P.seq_lens[b] - 1, 0);
  int ach;
  const bool dyn = attn_schedule<Tr, NB>(lcs, coff, ach);

  // ---- S1: q rows (64 per WG: 4 tiles x 8 K slices), weights in flight before
  // the input norm
  sycl::uint4 wq[16];
  const int qt = qrow(wg * 64 + (s >> 3) * 16);
  float sq[2];  // block scales, loaded ahead of the weight stream
#pragma unroll
  for (int j = 0; j < 2; ++j)
    sq[j] = P.qkv_s[(size_t)(qt / Tr::QB) * Tr::KB_HID + (s & 7) * 2 + j];
  ld_rows<16>(wq, P.qkv_w + (size_t)(qt + lane) * Tr::HID + (s & 7) * 256);
  input_norm_b<Tr, NB, DT>(it, P, L);
  wg_bar(grp);
  if ((!SPLIT || X.seg == SEG_F1)) {
    {
      v8f cc = v8f(0.f);
      mm_rows_r<16>(cc, wq, L.xa, (s & 7) * 16, lane, sq);
      red_put(L.red, s, lane, cc);
    }
    wg_bar(grp);
    {
      for (int idx = lid; idx < NB * 64; idx += Tr::WGS) {
        const int b = idx >> 6, r = idx & 63, t = r >> 4, l = r & 15;
        float v = 0.f;
#pragma unroll
        for (int k2 = 0; k2 < 8; ++k2)
          v += red_get(L.red, t * 8 + k2, b, l);
        SB.qkv_raw[b * Tr::QKV_ROWS + qrow(wg * 64 + r)] = v * 256.f;
      }
    }
    signal(it, S.cnt + C_A * 16);
  }
  if (SPLIT && X.seg == SEG_F1) return;
  if (lid < 32 * NB) rope_cs<Tr>((float)P.pos[lid >> 5], lid & 31, rope_all + (lid >> 5) * 64);

  if (lid < 48) reinterpret_cast<int*>(slm + Tr::AB_TK)[lid] = -1;  // ticket ring / acks: nothing yet
  // ---- S2: q norm + RoPE per row, attention over the flattened context chunks
  // -> partials (global)
  // (split: the merge segment needs the normalized q too)
  if (!SPLIT || X.seg == SEG_F2 || X.seg == SEG_F3) {
    wait_for(it, S.cnt + C_A * 16, Tr::NWG * call);
    for (int p = s; p < NB * Tr::NQ; p += Tr::NSG) {  // (row, head) per SG
      const int b = p / Tr::NQ, hh = p % Tr::NQ;
      q_norm_rope_b<Tr, DT>(
          sg,
          lane,
          P,
          SB.qkv_raw + b * Tr::QKV_ROWS + hh * 512,
          rope_all + b * 64,
          reinterpret_cast<e16*>(qpk_all + b * 1024),
          hh);
    }
    wg_bar(grp);
  }
  if ((!SPLIT || X.seg == SEG_F2)) {
    attn_chunks_b<Tr, NB, DT>(it, slm, P, S, SB, lcs, coff, dyn, ach, [](int b) { return b; });
    // ---- gate / k / v rows (80 per WG: 5 tiles x 4 K slices, SGs 0..19)
    {
      sycl::uint4 wg4[32];
      const int t = s >> 2, ks = s & 3;
      const int gt = grow(wg * 80 + t * 16);
      float sg4[4] = {0.f, 0.f, 0.f, 0.f};  // block scales, ahead of the weights
      if (s < 20)
#pragma unroll
        for (int j = 0; j < 4; ++j)
          sg4[j] = P.qkv_s[(size_t)(gt / Tr::QB) * Tr::KB_HID + ks * 4 + j];
      if (s < 20) ld_rows<32>(wg4, P.qkv_w + (size_t)(gt + lane) * Tr::HID + ks * 512);
      v8f cc = v8f(0.f);
      if (s < 20) mm_rows_r<32>(cc, wg4, L.xa, ks * 32, lane, sg4);
      red_put(L.red, s, lane, cc);
    }
    wg_bar(grp);
    {
      for (int idx = lid; idx < NB * 80; idx += Tr::WGS) {
        const int b = idx / 80, r = idx % 80, t = r >> 4, l = r & 15;
        float v = 0.f;
#pragma unroll
        for (int k2 = 0; k2 < 4; ++k2)
          v += red_get(L.red, t * 4 + k2, b, l);
        SB.qkv_raw[b * Tr::QKV_ROWS + grow(wg * 80 + r)] = v * 256.f;
      }
    }
    signal(it, S.cnt + C_B * 16);
  }
  if (SPLIT && X.seg == SEG_F2) return;
  sycl::uint4 ow[16], prt_r[4], prt_s[4];
  ld_rows<16>(ow, P.out_w + (size_t)(wg * 64 + (s >> 3) * 16 + lane) * (Tr::NQ * Tr::HD) + (s & 7) * 256);

  // ---- S3: per row: new k (norm + RoPE) / v, K / V write (WG 0), merge the
  // row's chunk partials + the new token, gate
  if ((!SPLIT || X.seg == SEG_F3)) {
    wait_for(it, S.cnt + C_B * 16, Tr::NWG * call);
    {
      const int hh = wg >> 2, d0 = (wg & 3) * 64;
      if (s < NB) {
        const int b = s;
        const int64_t slot = P.slot[b];
        head_norm_rope<Tr, DT>(
            sg,
            lane,
            SB.qkv_raw + b * Tr::QKV_ROWS + Tr::NQ * 512,
            P.k_norm_w,
            rope_all + b * 64,
            kvn_all + b * 512,
            (wg == 0 && slot >= 0) ? P.kc + slot * Tr::HD : nullptr);
      } else if (s >= 16 && s < 16 + NB) {
        const int b = s - 16;
        const int64_t slot = P.slot[b];
        head_norm_rope<Tr, DT>(
            sg,
            lane,
            SB.qkv_raw + b * Tr::QKV_ROWS + Tr::NQ * 512 + Tr::HD,
            nullptr,
            rope_all + b * 64,
            kvn_all + b * 512 + Tr::HD,
            (wg == 0 && slot >= 0) ? P.vc + slot * Tr::HD : nullptr);
      }
      wg_bar(grp);
      if (s < NB) {  // new-token score of head hh, row s
        float d = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {  // q[hh][dd] from the packed q: kt = dd/16, l = (hh&1)*8 +
                                        // (dd%16)/2, j = hh/2
          const int dd = lane * 16 + i, kt = dd >> 4, l = (hh & 1) * 8 + ((dd & 15) >> 1);
          const float qv = ld16<DT>(
              reinterpret_cast<const e16*>(qpk_all + s * 1024), ((kt * 16 + l) * 4 + (hh >> 1)) * 2 + (dd & 1));
          d += qv * ld16<DT>(kvn_all, s * 512 + dd);
        }
        d = sycl::reduce_over_group(sg, d, sycl::plus<float>());
        if (lane == 0) sm3[s] = d * Tr::SM_SCALE;
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
          mg = SB.pa_ml[((size_t)g * Tr::NQ + hh) * 2];
          lg = SB.pa_ml[((size_t)g * Tr::NQ + hh) * 2 + 1];
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
        constexpr int TW = Tr::WGS / NB / 64 * 64, PARTS = TW / 64;  // whole SGs per row (NB = 3, 6: idle tail)
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
                a8[q] += wgt[g0 + i2 + q * PARTS] *
                         (float)SB.pa_o[((size_t)(g0 + i2 + q * PARTS) * Tr::NQ + hh) * Tr::HD + d0 + d];
            }
            for (; i2 < n; i2 += PARTS)
              a8[0] += wgt[g0 + i2] * (float)SB.pa_o[((size_t)(g0 + i2) * Tr::NQ + hh) * Tr::HD + d0 + d];
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
          float a = en * ld16<DT>(kvn_all, b2 * 512 + Tr::HD + d0 + d2);
#pragma unroll
          for (int pp = 0; pp < PARTS; ++pp)
            a += L.red[(b2 * PARTS + pp) * 64 + d2];
          const float o = a / Lw;
          const float gate = SB.qkv_raw[b2 * Tr::QKV_ROWS + hh * 512 + 256 + d0 + d2];
          SB.attn_out[b2 * Tr::NQ * Tr::HD + hh * Tr::HD + d0 + d2] = toh<DT>(o / (1.f + sycl::exp(-gate)));
        }
      }
    }
    signal(it, S.cnt + C_C * 16);
  }
  if (SPLIT && X.seg == SEG_F3) return;
  load_prt_b<Tr, DT>(P, wg, s, lane, prt_r, prt_s);

  // ---- S4: o (all heads, all rows) -> perm A tiles ; tail
  if (!SPLIT || X.seg == SEG_T1 || X.seg == SEG_T1A || X.seg == SEG_T1B) {
    wait_for(it, S.cnt + C_C * 16, Tr::NWG * call);
    if (wg == 0 && lid < NB) cnt_ref(S.cnt[ATK + lid]).store(0u);  // every pair is done with the tickets
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      const sycl::vec<half, 4> o4 =
          *reinterpret_cast<const sycl::vec<half, 4>*>(SB.attn_out + b * Tr::NQ * Tr::HD + lid * 4);
      const half y[4] = {o4[0], o4[1], o4[2], o4[3]};
      a_put4_perm(L.xa, b, lid * 4, y);
    }
    wg_bar(grp);
  }
  tail_b<Tr, NB, DT, SPLIT>(it, P, S, SB, L, call, C_O, ow, prt_r, prt_s, X);
}

// ------------------------------------------------ full attention (spec verify)
// The batched attention with row r = (request r / T, token r % T):
// - the context chunks of a row are its request's cached tokens
//   (P.seq_lens[request] - T, page table row = request); the T rows of a
//   request read the same KV, mostly from L2;
// - the T new k / v are written to the cache (P.slot[row]) and merged
//   causally: row r takes the new tokens 0 .. r % T of its request.
template <class Tr, int NB, int T, typename DT, bool SPLIT = false>
inline void attn_layer_s(
    sycl::nd_item<1> it, uint8_t* slm, const Ptrs& P, const Scratch& S, const ScratchB& SB, const Split& X = Split{}) {
  static_assert(NB % T == 0 && T >= 2, "spec verify: NB = requests x tokens");
  const int wg = it.get_group(0);
  const int lid = it.get_local_linear_id();
  sycl::sub_group sg = it.get_sub_group();
  const int s = sg.get_group_linear_id();
  const int lane = sg.get_local_linear_id();
  auto grp = it.get_group();
  const uint32_t call = uc_load32(S.cnt + EPOCH) + 1u;
  const SlmB L = slm_b<Tr>(slm);
  // the single-row kernel takes gate/k/v chunks from a ticket counter whose
  // base it derives from the call number: every call must consume the same
  // number of tickets
  if ((!SPLIT || X.seg == SEG_F1) && wg == 0 && lid == 0)
    cnt_ref(S.cnt[C_TKT * 16]).fetch_add((uint32_t)Tr::GKV_CHUNKS + Tr::NSG * Tr::NWG);
  uint32_t* qpk_all = reinterpret_cast<uint32_t*>(slm + Tr::AB_QPK);
  float* rope_all = reinterpret_cast<float*>(slm + Tr::AB_CS);
  e16* kvn_all = reinterpret_cast<e16*>(slm + Tr::AB_KVN);
  float* wgt = reinterpret_cast<float*>(slm + Tr::AB_W);
  float* sm3 = reinterpret_cast<float*>(slm + Tr::AB_SM3);
  auto qrow = [](int qi) { return (qi >> 8) * 512 + (qi & 255); };
  auto grow = [](int gi) {
    return gi < Tr::NQ * Tr::HD ? (gi >> 8) * 512 + 256 + (gi & 255) : Tr::NQ * 512 + (gi - Tr::NQ * Tr::HD);
  };
  int lcs[NB], coff[NB + 1];
#pragma unroll
  for (int b = 0; b < NB; ++b)
    lcs[b] = sycl::max(P.seq_lens[b / T] - T, 0);
  int ach;
  const bool dyn = attn_schedule<Tr, NB>(lcs, coff, ach);

  // ---- S1: q rows (64 per WG: 4 tiles x 8 K slices), weights in flight before
  // the input norm
  sycl::uint4 wq[16];
  const int qt = qrow(wg * 64 + (s >> 3) * 16);
  float sq[2];  // block scales, loaded ahead of the weight stream
#pragma unroll
  for (int j = 0; j < 2; ++j)
    sq[j] = P.qkv_s[(size_t)(qt / Tr::QB) * Tr::KB_HID + (s & 7) * 2 + j];
  ld_rows<16>(wq, P.qkv_w + (size_t)(qt + lane) * Tr::HID + (s & 7) * 256);
  input_norm_b<Tr, NB, DT>(it, P, L);
  wg_bar(grp);
  if ((!SPLIT || X.seg == SEG_F1)) {
    {
      v8f cc = v8f(0.f);
      mm_rows_r<16>(cc, wq, L.xa, (s & 7) * 16, lane, sq);
      red_put(L.red, s, lane, cc);
    }
    wg_bar(grp);
    {
      for (int idx = lid; idx < NB * 64; idx += Tr::WGS) {
        const int b = idx >> 6, r = idx & 63, t = r >> 4, l = r & 15;
        float v = 0.f;
#pragma unroll
        for (int k2 = 0; k2 < 8; ++k2)
          v += red_get(L.red, t * 8 + k2, b, l);
        SB.qkv_raw[b * Tr::QKV_ROWS + qrow(wg * 64 + r)] = v * 256.f;
      }
    }
    signal(it, S.cnt + C_A * 16);
  }
  if (SPLIT && X.seg == SEG_F1) return;
  if (lid < 32 * NB) rope_cs<Tr>((float)P.pos[lid >> 5], lid & 31, rope_all + (lid >> 5) * 64);

  if (lid < 48) reinterpret_cast<int*>(slm + Tr::AB_TK)[lid] = -1;  // ticket ring / acks: nothing yet
  // ---- S2: q norm + RoPE per row, attention over the flattened context chunks
  // -> partials (global)
  // (split: the merge segment needs the normalized q too)
  if (!SPLIT || X.seg == SEG_F2 || X.seg == SEG_F3) {
    wait_for(it, S.cnt + C_A * 16, Tr::NWG * call);
    for (int p = s; p < NB * Tr::NQ; p += Tr::NSG) {  // (row, head) per SG
      const int b = p / Tr::NQ, hh = p % Tr::NQ;
      q_norm_rope_b<Tr, DT>(
          sg,
          lane,
          P,
          SB.qkv_raw + b * Tr::QKV_ROWS + hh * 512,
          rope_all + b * 64,
          reinterpret_cast<e16*>(qpk_all + b * 1024),
          hh);
    }
    wg_bar(grp);
  }
  if ((!SPLIT || X.seg == SEG_F2)) {
    attn_chunks_b<Tr, NB, DT>(it, slm, P, S, SB, lcs, coff, dyn, ach, [](int b) { return b / T; });
    // ---- gate / k / v rows (80 per WG: 5 tiles x 4 K slices, SGs 0..19)
    {
      sycl::uint4 wg4[32];
      const int t = s >> 2, ks = s & 3;
      const int gt = grow(wg * 80 + t * 16);
      float sg4[4] = {0.f, 0.f, 0.f, 0.f};  // block scales, ahead of the weights
      if (s < 20)
#pragma unroll
        for (int j = 0; j < 4; ++j)
          sg4[j] = P.qkv_s[(size_t)(gt / Tr::QB) * Tr::KB_HID + ks * 4 + j];
      if (s < 20) ld_rows<32>(wg4, P.qkv_w + (size_t)(gt + lane) * Tr::HID + ks * 512);
      v8f cc = v8f(0.f);
      if (s < 20) mm_rows_r<32>(cc, wg4, L.xa, ks * 32, lane, sg4);
      red_put(L.red, s, lane, cc);
    }
    wg_bar(grp);
    {
      for (int idx = lid; idx < NB * 80; idx += Tr::WGS) {
        const int b = idx / 80, r = idx % 80, t = r >> 4, l = r & 15;
        float v = 0.f;
#pragma unroll
        for (int k2 = 0; k2 < 4; ++k2)
          v += red_get(L.red, t * 4 + k2, b, l);
        SB.qkv_raw[b * Tr::QKV_ROWS + grow(wg * 80 + r)] = v * 256.f;
      }
    }
    signal(it, S.cnt + C_B * 16);
  }
  if (SPLIT && X.seg == SEG_F2) return;
  sycl::uint4 ow[16], prt_r[4], prt_s[4];
  ld_rows<16>(ow, P.out_w + (size_t)(wg * 64 + (s >> 3) * 16 + lane) * (Tr::NQ * Tr::HD) + (s & 7) * 256);

  // ---- S3: per row: new k (norm + RoPE) / v, K / V write (WG 0), merge the
  // row's chunk partials + the request's new tokens, gate
  if ((!SPLIT || X.seg == SEG_F3)) {
    wait_for(it, S.cnt + C_B * 16, Tr::NWG * call);
    {
      const int hh = wg >> 2, d0 = (wg & 3) * 64;
      if (s < NB) {
        const int b = s;
        const int64_t slot = P.slot[b];
        head_norm_rope<Tr, DT>(
            sg,
            lane,
            SB.qkv_raw + b * Tr::QKV_ROWS + Tr::NQ * 512,
            P.k_norm_w,
            rope_all + b * 64,
            kvn_all + b * 512,
            (wg == 0 && slot >= 0) ? P.kc + slot * Tr::HD : nullptr);
      } else if (s >= 16 && s < 16 + NB) {
        const int b = s - 16;
        const int64_t slot = P.slot[b];
        head_norm_rope<Tr, DT>(
            sg,
            lane,
            SB.qkv_raw + b * Tr::QKV_ROWS + Tr::NQ * 512 + Tr::HD,
            nullptr,
            rope_all + b * 64,
            kvn_all + b * 512 + Tr::HD,
            (wg == 0 && slot >= 0) ? P.vc + slot * Tr::HD : nullptr);
      }
      wg_bar(grp);
      float* sn = sm3 + 128;  // [NB][T] new-token scores of head hh (row r, token
                              // j of its request; -inf: j > r % T)
      for (int p = s; p < NB * T; p += Tr::NSG) {
        const int r = p / T, j = p % T, kr = (r / T) * T + j;
        float d = 0.f;
        if (j <= r % T) {
#pragma unroll
          for (int i = 0; i < 16; ++i) {  // q[hh][dd] from the packed q: kt = dd/16, l = (hh&1)*8 +
                                          // (dd%16)/2, j = hh/2
            const int dd = lane * 16 + i, kt = dd >> 4, l = (hh & 1) * 8 + ((dd & 15) >> 1);
            const float qv = ld16<DT>(
                reinterpret_cast<const e16*>(qpk_all + r * 1024), ((kt * 16 + l) * 4 + (hh >> 1)) * 2 + (dd & 1));
            d += qv * ld16<DT>(kvn_all, kr * 512 + dd);
          }
          d = sycl::reduce_over_group(sg, d, sycl::plus<float>());
        }
        if (lane == 0) sn[p] = j <= r % T ? d * Tr::SM_SCALE : -INFINITY;
      }
      wg_bar(grp);
      {  // merge, all rows at once: a work-item per chunk for the maxima /
         // weights, TW work-items per row for the weighted sums (deterministic:
         // atomic max only, fixed-order sums)
        int* mkey = reinterpret_cast<int*>(sm3 + 16);
        float* lwp = sm3 + 32;  // [NB][TW / 16] per-SG weight sums
        auto key = [](float f) {
          const int k = sycl::bit_cast<int>(f);
          return k >= 0 ? k : k ^ 0x7fffffff;
        };
        auto unkey = [](int k) { return sycl::bit_cast<float>(k >= 0 ? k : k ^ 0x7fffffff); };
        if (lid < NB) {
          float mx = sn[lid * T];
#pragma unroll
          for (int j = 1; j < T; ++j)
            mx = sycl::fmax(mx, sn[lid * T + j]);
          mkey[lid] = key(mx);
        }
        wg_bar(grp);
        const int g = lid;
        int gb = 0;
#pragma unroll
        for (int bb = 1; bb < NB; ++bb)
          if (coff[bb] <= g) gb = bb;
        const bool live = g < coff[NB];
        float mg = -INFINITY, lg = 0.f;
        if (live) {
          mg = SB.pa_ml[((size_t)g * Tr::NQ + hh) * 2];
          lg = SB.pa_ml[((size_t)g * Tr::NQ + hh) * 2 + 1];
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
        constexpr int TW = Tr::WGS / NB / 64 * 64, PARTS = TW / 64;  // whole SGs per row (NB = 3, 6: idle tail)
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
                a8[q] += wgt[g0 + i2 + q * PARTS] *
                         (float)SB.pa_o[((size_t)(g0 + i2 + q * PARTS) * Tr::NQ + hh) * Tr::HD + d0 + d];
            }
            for (; i2 < n; i2 += PARTS)
              a8[0] += wgt[g0 + i2] * (float)SB.pa_o[((size_t)(g0 + i2) * Tr::NQ + hh) * Tr::HD + d0 + d];
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
          float Lw = 0.f, a = 0.f;
#pragma unroll
          for (int j = 0; j < T; ++j) {  // the request's new tokens 0 .. b2 % T (others: exp2(-inf) = 0)
            const float en = sycl::native::exp2(sn[b2 * T + j] - M);
            Lw += en;
            a += en * ld16<DT>(kvn_all, ((b2 / T) * T + j) * 512 + Tr::HD + d0 + d2);
          }
          for (int k2 = 0; k2 < TW / 16; ++k2)
            Lw += lwp[b2 * (TW / 16) + k2];
#pragma unroll
          for (int pp = 0; pp < PARTS; ++pp)
            a += L.red[(b2 * PARTS + pp) * 64 + d2];
          const float o = a / Lw;
          const float gate = SB.qkv_raw[b2 * Tr::QKV_ROWS + hh * 512 + 256 + d0 + d2];
          SB.attn_out[b2 * Tr::NQ * Tr::HD + hh * Tr::HD + d0 + d2] = toh<DT>(o / (1.f + sycl::exp(-gate)));
        }
      }
    }
    signal(it, S.cnt + C_C * 16);
  }
  if (SPLIT && X.seg == SEG_F3) return;
  load_prt_b<Tr, DT>(P, wg, s, lane, prt_r, prt_s);

  // ---- S4: o (all heads, all rows) -> perm A tiles ; tail
  if (!SPLIT || X.seg == SEG_T1 || X.seg == SEG_T1A || X.seg == SEG_T1B) {
    wait_for(it, S.cnt + C_C * 16, Tr::NWG * call);
    if (wg == 0 && lid < NB) cnt_ref(S.cnt[ATK + lid]).store(0u);  // every pair is done with the tickets
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      const sycl::vec<half, 4> o4 =
          *reinterpret_cast<const sycl::vec<half, 4>*>(SB.attn_out + b * Tr::NQ * Tr::HD + lid * 4);
      const half y[4] = {o4[0], o4[1], o4[2], o4[3]};
      a_put4_perm(L.xa, b, lid * 4, y);
    }
    wg_bar(grp);
  }
  tail_b<Tr, NB, DT, SPLIT>(it, P, S, SB, L, call, C_O, ow, prt_r, prt_s, X);
}

}  // namespace sgl_kernel::stage::xe20
