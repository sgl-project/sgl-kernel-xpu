// SPDX-License-Identifier: Apache-2.0
//
// The GatedDeltaNet (linear attention) decoder layer: input norm, in_proj
// q|k|v|z (fp8) + b|a (16-bit), causal conv1d, gated delta rule, gated
// RMSNorm, then the MoE tail (moe.hpp). Single-row (gdn_layer), batched
// (gdn_layer_b) and speculative-decoding verify (gdn_layer_s, see attn.hpp
// for the verify row layout) versions.
//
// Stage code: every function that depends on the model's shape takes the
// traits Tr (Dims<Shape>, dims.hpp) as its first template argument.
#pragma once
#include "sgl_kernel/stage/xe20/moe.hpp"

namespace sgl_kernel::stage::xe20 {

// the conv states (3 taps) of the WG's 320 channels of NR states -> csl
// [r][tap][320] (8 channels = 24 contiguous halves a load); conv_of(r) is the
// state's slot pointer, ch its elements per channel
template <class Tr, int NR, typename ConvOf>
inline void load_conv_b(e16* csl, ConvOf conv_of, const int (&chs)[NR], int h, int c, int kh, int lid) {
  for (int v = lid; v < NR * 40; v += Tr::WGS) {
    const int r = v / 40, j8 = (v % 40) * 8;
    const int ch = j8 < 128   ? kh * Tr::DK + j8
                   : j8 < 256 ? Tr::KD + kh * Tr::DK + (j8 - 128)
                              : 2 * Tr::KD + h * Tr::DV + c * 64 + (j8 - 256);
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

// ----------------------------------------------------------------------------
// GDN layer
template <class Tr, typename DT, bool SPLIT = false>
inline void gdn_layer(sycl::nd_item<1> it, uint8_t* slm, const Ptrs& P, const Scratch& S, const Split& X = Split{}) {
  const int wg = (SPLIT && X.vwg >= 0) ? X.vwg : (int)it.get_group(0);
  const int lid = it.get_local_linear_id();
  sycl::sub_group sg = it.get_sub_group();
  const int s = sg.get_group_linear_id();
  const int lane = sg.get_local_linear_id();
  auto grp = it.get_group();
  // epoch: plain L1-uncached load (an atomic load is an atomic_or RMW: the 1024
  // threads would serialize)
  const uint32_t call = uc_load32(S.cnt + EPOCH) + 1u;

  const Slm L = slm_tail(slm, Tr::G_X, Tr::G_XN, Tr::G_QK, Tr::G_RED, Tr::G_ROUTE, Tr::G_AO, Tr::G_MR, Tr::G_HB);
  float* qk = L.qk;  // q[0..127], k[128..255], v[256..319], beta / g [320..321]
  float* red = L.red;

  const int h = wg >> 1, c = wg & 1, kh = h >> 1;  // S2 role: v head h (k head h/2), v rows c*64..+64
  const int sidx = P.state_idx[0];
  e16* conv = sidx >= 0 ? P.conv_cache + (size_t)sidx * P.conv_stride : S.dummy_conv;
  float* ssm = sidx >= 0 ? P.ssm_cache + (size_t)sidx * P.ssm_stride : S.dummy_ssm;
  const int cst_ch = sidx >= 0 ? P.conv_ch : 3;

  // ---------------- kernel start: inputs of S2 that do not depend on this
  // call's activations ----------
  sycl::vec<float, 8> st[2];  // recurrent state block: rows v = c*64 + 2s + rr, k = lane*8..+8
#pragma unroll
  for (int rr = 0; rr < 2; ++rr)
    st[rr] = *reinterpret_cast<const sycl::vec<float, 8>*>(
        ssm + ((size_t)h * Tr::DV + c * 64 + 2 * s + rr) * Tr::DK + lane * 8);
  const int cch = lid < 128   ? kh * Tr::DK + lid
                  : lid < 256 ? Tr::KD + kh * Tr::DK + (lid - 128)
                              : 2 * Tr::KD + h * Tr::DV + c * 64 + (lid - 256);
  e16 cst[3];
  float cwt[4];
  if (lid < 320) {
#pragma unroll
    for (int i = 0; i < 3; ++i)
      cst[i] = conv[(size_t)cch * cst_ch + i];
    const sycl::vec<uint16_t, 4> w4 = *reinterpret_cast<const sycl::vec<uint16_t, 4>*>(P.conv_w + (size_t)cch * 4);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      cwt[i] = f32<DT>(w4[i]);
  }
  // S1's block scales (this SG's rows lie in one 128-row block), ahead of the
  // norm: loaded after S1's weights they would hold up its first dot product
  float sc1[8];
  load_scales<8>(sc1, P.qkvz_s + (size_t)(((wg * Tr::NSG + s) * 4) / Tr::QB) * Tr::KB_HID, lane);
  // b (SG 0) and a (SG 1) rows of v head h, 16-bit: the first half in flight
  // during the norm, the second during S1
  sycl::uint4 baw[8];
  const e16* barow = P.ba_w + (size_t)(s * Tr::NVH + h) * Tr::HID;
  if (s < 2) {
#pragma unroll
    for (int j = 0; j < 8; ++j)
      baw[j] = *reinterpret_cast<const sycl::uint4*>(barow + j * 128 + lane * 8);
  }
  input_norm<Tr, DT>(it, P, L.xn, L.xp);
  wg_bar(grp);
  float bav = 0.f;
  if (s < 2) {
#pragma unroll
    for (int j = 0; j < 8; ++j)
      bav += dot8_w16<DT>(baw[j], L.xn + j * 128 + lane * 8);
#pragma unroll
    for (int j = 0; j < 8; ++j)
      baw[j] = *reinterpret_cast<const sycl::uint4*>(barow + (8 + j) * 128 + lane * 8);
  }

  // ---------------- S1: q|k|v rows of in_proj_qkvz (4 per SG) ----------------
  if ((!SPLIT || X.seg == SEG_F1)) {
    const int r0 = (wg * Tr::NSG + s) * 4;
    sycl::uint4 w[4][8];
#pragma unroll
    for (int b = 0; b < 4; ++b)
      load_row<8>(w[b], P.qkvz_w + (size_t)(r0 + b) * Tr::HID, lane);
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const float v = dot_row_regs<8>(sg, lane, w[b], L.xp32, sc1);
      if (lane == 0) S.qkv[r0 + b] = toh<DT>(v * 256.f);
    }
    signal(it, S.cnt + C_A * 16);
  }
  if (SPLIT && X.seg == SEG_F1) return;
  sycl::uint4 zw[2][8];  // z rows of this WG: channel h*128 + c*64 + 2s + rr
                         // (consumed after the recurrence)
  float zsc[8];
  {
    const int zr = Tr::CONVD + h * Tr::DV + c * 64 + 2 * s;
#pragma unroll
    for (int rr = 0; rr < 2; ++rr)
      load_row<8>(zw[rr], P.qkvz_w + (size_t)(zr + rr) * Tr::HID, lane);
    load_scales<8>(zsc, P.qkvz_s + (size_t)(zr / Tr::QB) * Tr::KB_HID, lane);
  }
  if (s < 2) {  // b, a of v head h (16-bit rows, 8 halves a lane a chunk; DT, as SGLang's GEMM output)
#pragma unroll
    for (int j = 0; j < 8; ++j)
      bav += dot8_w16<DT>(baw[j], L.xn + (8 + j) * 128 + lane * 8);
    const float v = sycl::reduce_over_group(sg, bav, sycl::plus<float>());
    if (lane == 0) red[s] = rnd<DT>(v);
  }

  // ---------------- S2: conv1d + SiLU, L2 norm, gated delta rule -------------
  if ((!SPLIT || X.seg == SEG_F2)) {
    wait_for(it, S.cnt, C_A, Tr::NWG * call);
    if (lid < 320) {
      const float xnew = (float)S.qkv[cch];
      float acc = ld16<DT>(cst, 0) * cwt[0];
      acc += ld16<DT>(cst, 1) * cwt[1];
      acc += ld16<DT>(cst, 2) * cwt[2];
      acc += xnew * cwt[3];
      qk[lid] = rnd<DT>(acc / (1.f + sycl::exp(-acc)));
      // conv state <- (s1, s2, x): q/k channels by the (even h, c = 0) WG of the
      // k head (split: in the T1 segment, see there), v channels by the owner
      if (lid >= 256 || (!SPLIT && (h & 1) == 0 && c == 0)) {
        e16* cp = conv + (size_t)cch * cst_ch;
        cp[0] = cst[1];
        cp[1] = cst[2];
        st16<DT>(cp, 2, xnew);
      }
    }
    if (lid == 0) {  // beta, g of v head h
      const float b = red[0], a = red[1] + ld16<DT>(P.dt_bias, h);
      const float sp = a < 20.f ? sycl::log(1.f + sycl::exp(a)) : a;
      qk[320] = 1.f / (1.f + sycl::exp(-b));
      qk[321] = sycl::exp(-sycl::exp(P.A_log[h]) * sp);
    }
    wg_bar(grp);
    if (s < 16) {  // L2 norms: q = SGs 0..7, k = SGs 8..15
      const float v = qk[lid];
      const float ss = sycl::reduce_over_group(sg, v * v, sycl::plus<float>());
      if (lane == 0) red[8 + s] = ss;
    }
    wg_bar(grp);
    {
      float qn = 0.f, kn = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        qn += red[8 + i];
        kn += red[16 + i];
      }
      const float iq = 1.f / sycl::sqrt(qn + 1e-6f), ik = 1.f / sycl::sqrt(kn + 1e-6f);
      const float scale = 0.08838834764831845f;  // 1 / sqrt(128)
      float qv[8], kv8[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        qv[i] = qk[lane * 8 + i] * iq * scale;
        kv8[i] = qk[128 + lane * 8 + i] * ik;
      }
      const float beta = qk[320], g = qk[321];
      float kvm[2], o[2];
#pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
        float d = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          st[rr][i] *= g;
          d += st[rr][i] * kv8[i];
        }
        kvm[rr] = d;
      }
#pragma unroll
      for (int rr = 0; rr < 2; ++rr)
        kvm[rr] = sycl::reduce_over_group(sg, kvm[rr], sycl::plus<float>());
#pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
        const float delta = (qk[256 + 2 * s + rr] - kvm[rr]) * beta;
        float d = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          st[rr][i] += kv8[i] * delta;
          d += st[rr][i] * qv[i];
        }
        o[rr] = d;
      }
#pragma unroll
      for (int rr = 0; rr < 2; ++rr)
        o[rr] = sycl::reduce_over_group(sg, o[rr], sycl::plus<float>());
      if (lane < 2) S.core[h * Tr::DV + c * 64 + 2 * s + lane] = toh<DT>(lane == 0 ? o[0] : o[1]);
    }
    {  // z rows (registers since S1)
      const float z0 = dot_row_regs<8>(sg, lane, zw[0], L.xp32, zsc),
                  z1 = dot_row_regs<8>(sg, lane, zw[1], L.xp32, zsc);
      if (lane < 2) S.zbuf[h * Tr::DV + c * 64 + 2 * s + lane] = toh<DT>((lane == 0 ? z0 : z1) * 256.f);
    }
    signal(it, S.cnt + C_B * 16);
#pragma unroll
    for (int rr = 0; rr < 2; ++rr)  // state write-back after the hand-off
      st_store8(ssm + ((size_t)h * Tr::DV + c * 64 + 2 * s + rr) * Tr::DK + lane * 8, st[rr]);
  }
  if (SPLIT && X.seg == SEG_F2) return;
  sycl::uint4 ow[2][8], prt_r[4], prt_s[Tr::SHR / 4];
  float osc[8];
  load_row<8>(ow[0], P.out_w + (size_t)(wg * 64 + s * 2) * Tr::VD, lane);
  load_row<8>(ow[1], P.out_w + (size_t)(wg * 64 + s * 2 + 1) * Tr::VD, lane);
  load_scales<8>(osc, P.out_s + (size_t)((wg * 64) / Tr::QB) * (Tr::VD / Tr::QB), lane);
  load_prt<Tr>(P, wg, s, lane, prt_r, prt_s);

  // ---------------- S3: gated RMSNorm (all heads) -> o in SLM ; tail
  // -------------------------
  if (!SPLIT || X.seg == SEG_T1 || X.seg == SEG_T1A || X.seg == SEG_T1B) {
    wait_for(it, S.cnt, C_B, Tr::NWG * call);
    if constexpr (SPLIT) {
      // the q/k conv windows, read at F2's start by the k head's 4 WGs: one
      // kernel orders the reads before the write by C_A; split, the write waits
      // for C_B (every WG's F2 done)
      if ((X.seg == SEG_T1 || X.seg == SEG_T1A) && lid < 256 && (h & 1) == 0 && c == 0) {
        e16* cp = conv + (size_t)cch * cst_ch;
        const e16 t1 = cp[1], t2 = cp[2];
        cp[0] = t1;
        cp[1] = t2;
        st16<DT>(cp, 2, (float)S.qkv[cch]);
      }
    }
    const sycl::vec<half, 4> o4 = *reinterpret_cast<const sycl::vec<half, 4>*>(S.core + lid * 4);
    const sycl::vec<half, 4> z4 = *reinterpret_cast<const sycl::vec<half, 4>*>(S.zbuf + lid * 4);
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i)
      ss += (float)o4[i] * (float)o4[i];
    ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
    if (lane == 0) red[64 + s] = ss;  // head = 2 SGs
    wg_bar(grp);
    const int hh = s >> 1;
    const float rs = sycl::rsqrt((red[64 + 2 * hh] + red[64 + 2 * hh + 1]) / Tr::DV + Tr::EPS);
    half y[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float zf = (float)z4[i];
      y[i] = toh<DT>(
          (float)o4[i] * rs * ld16<DT>(P.gnorm_w, (lid * 4 + i) & (Tr::DV - 1)) * (zf / (1.f + sycl::exp(-zf))));
    }
    *reinterpret_cast<sycl::vec<half, 4>*>(L.xp + lid * 4) = sycl::vec<half, 4>(y[0], y[2], y[1], y[3]);
    wg_bar(grp);
  }
  tail<Tr, DT, SPLIT>(it, P, S, L, call, C_O, ow, osc, prt_r, prt_s, X);
}

// --------------------------------------------------------- GDN layer (batched)
template <class Tr, int NB, typename DT, bool SPLIT = false>
inline void gdn_layer_b(
    sycl::nd_item<1> it, uint8_t* slm, const Ptrs& P, const Scratch& S, const ScratchB& SB, const Split& X = Split{}) {
  const int wg = (SPLIT && X.vwg >= 0) ? X.vwg : (int)it.get_group(0);
  const int lid = it.get_local_linear_id();
  sycl::sub_group sg = it.get_sub_group();
  const int s = sg.get_group_linear_id();
  const int lane = sg.get_local_linear_id();
  auto grp = it.get_group();
  const uint32_t call = uc_load32(S.cnt + EPOCH) + 1u;
  const SlmB L = slm_b<Tr>(slm);
  float* qk =
      reinterpret_cast<float*>(slm + Tr::B_QK);  // per row r: q[0..127] k[128..255] v[256..319] beta g [320, 321]

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
    s1c[j] = P.qkvz_s[(size_t)((wg * 128 + tile1 * 16) / Tr::QB) * Tr::KB_HID + ks1 * 4 + j];
#pragma unroll
  for (int j = 0; j < 2; ++j)
    szc[j] = P.qkvz_s[(size_t)((Tr::CONVD + h * Tr::DV + c * 64 + tz * 16) / Tr::QB) * Tr::KB_HID + kz * 2 + j];
  sycl::uint4 w1[32];
  ld_rows<32>(w1, P.qkvz_w + (size_t)(wg * 128 + tile1 * 16 + lane) * Tr::HID + ks1 * 512);
#pragma unroll
  for (int b = 0; b < NB; ++b)  // 32 KB per row, one 64-B line per work-item
    syclex::prefetch(
        (void*)(ssm_of(b) + ((size_t)h * Tr::DV + c * 64) * Tr::DK + lid * 16),
        syclex::properties{syclex::prefetch_hint_L2});
  const int cch = lid < 128   ? kh * Tr::DK + lid
                  : lid < 256 ? Tr::KD + kh * Tr::DK + (lid - 128)
                              : 2 * Tr::KD + h * Tr::DV + c * 64 + (lid - 256);
  float cwt[4];
  e16* csl = reinterpret_cast<e16*>(slm + Tr::B_CS);  // [b][i][320]: conv state i of row b, the WG's channels
  load_conv_b<Tr, NB>(csl, conv_of, chs, h, c, kh, lid);
  if (lid < 320) {
    const sycl::vec<uint16_t, 4> w4 = *reinterpret_cast<const sycl::vec<uint16_t, 4>*>(P.conv_w + (size_t)cch * 4);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      cwt[i] = f32<DT>(w4[i]);
  }
  input_norm_b<Tr, NB, DT>(it, P, L);
  wg_bar(grp);

  // --- S1: q|k|v rows (128 per WG) -------------------------------------------
  sycl::uint4 zw[16], bw[16];
  {
    v8f cc = v8f(0.f);
    if ((!SPLIT || X.seg == SEG_F1)) mm_rows_r<32>(cc, w1, L.xa, ks1 * 32, lane, s1c);
    // z rows of this WG (4 tiles x 8 K slices), b|a rows (2 tiles x 16 K
    // slices, 16-bit: two uint4 a chunk): in flight from here. Only head h's
    // column of each b|a tile is kept, so the other lanes feed zeros instead
    // of streaming all 32 rows into every WG.
    ld_rows<16>(zw, P.qkvz_w + (size_t)(Tr::CONVD + h * Tr::DV + c * 64 + tz * 16 + lane) * Tr::HID + kz * 256);
    if (lane == h)
      ld_rows<16>(
          bw, reinterpret_cast<const uint8_t*>(P.ba_h + (size_t)((s >> 4) * 16 + h) * Tr::HID + (s & 15) * 128));
    else
#pragma unroll
      for (int t = 0; t < 16; ++t)
        bw[t] = sycl::uint4(0u);
    if ((!SPLIT || X.seg == SEG_F1)) red_put(L.red, s, lane, cc);
  }
  wg_bar(grp);
  if ((!SPLIT || X.seg == SEG_F1)) {
    for (int idx = lid; idx < NB * 128; idx += Tr::WGS) {
      const int b = idx >> 7, r = idx & 127, t = r >> 4, l = r & 15;
      float v = 0.f;
#pragma unroll
      for (int k2 = 0; k2 < 4; ++k2)
        v += red_get(L.red, t * 4 + k2, b, l);
      SB.qkv[b * Tr::CONVD + wg * 128 + r] = toh<DT>(v * 256.f);
    }
    signal(it, S.cnt + C_A * 16);
  }
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
    for (int idx = lid; idx < NB * 64; idx += Tr::WGS) {
      const int b = idx >> 6, r = idx & 63, t = r >> 4, l = r & 15;
      float v = 0.f;
#pragma unroll
      for (int k2 = 0; k2 < 8; ++k2)
        v += red_get(L.red, t * 8 + k2, b, l);
      if ((!SPLIT || X.seg == SEG_F1)) SB.z[b * Tr::VD + h * Tr::DV + c * 64 + r] = toh<DT>(v * 256.f);
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
  if (SPLIT && X.seg == SEG_F1) return;
  // recurrent state of all rows: rows v = c*64 + 2s + rr, k = lane*8..+8 (from
  // L2)
  sycl::vec<float, 8> st[NB][2];
#pragma unroll
  for (int b = 0; b < NB; ++b)
#pragma unroll
    for (int rr = 0; rr < 2; ++rr)
      st[b][rr] = *reinterpret_cast<const sycl::vec<float, 8>*>(
          ssm_of(b) + ((size_t)h * Tr::DV + c * 64 + 2 * s + rr) * Tr::DK + lane * 8);

  // --- S2: conv1d + SiLU, L2 norm, gated delta rule, all rows ----------------
  if ((!SPLIT || X.seg == SEG_F2)) {
    wait_for(it, S.cnt, C_A, Tr::NWG * call);
    if (lid < 320) {
#pragma unroll
      for (int b = 0; b < NB; ++b) {
        const float xnew = (float)SB.qkv[b * Tr::CONVD + cch];
        const e16 c0 = csl[(b * 3) * 320 + lid], c1 = csl[(b * 3 + 1) * 320 + lid], c2 = csl[(b * 3 + 2) * 320 + lid];
        float acc = f32<DT>(c0.u) * cwt[0];
        acc += f32<DT>(c1.u) * cwt[1];
        acc += f32<DT>(c2.u) * cwt[2];
        acc += xnew * cwt[3];
        qk[b * 384 + lid] = rnd<DT>(acc / (1.f + sycl::exp(-acc)));
        if (lid >= 256 || (!SPLIT && (h & 1) == 0 && c == 0)) {  // (split q/k: T1)
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
      const float iq = 1.f / sycl::sqrt(L.sm[600 + b * 2] + 1e-6f),
                  ik = 1.f / sycl::sqrt(L.sm[600 + b * 2 + 1] + 1e-6f);
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
      if (lane < 2) SB.core[b * Tr::VD + h * Tr::DV + c * 64 + 2 * s + lane] = toh<DT>(lane == 0 ? o[0] : o[1]);
    }
    signal(it, S.cnt + C_B * 16);
#pragma unroll
    for (int b = 0; b < NB; ++b)
#pragma unroll
      for (int rr = 0; rr < 2; ++rr)  // state write-back after the hand-off
        st_store8(ssm_of(b) + ((size_t)h * Tr::DV + c * 64 + 2 * s + rr) * Tr::DK + lane * 8, st[b][rr]);
  }
  if (SPLIT && X.seg == SEG_F2) return;
  sycl::uint4 ow[16], prt_r[4], prt_s[4];
  ld_rows<16>(ow, P.out_w + (size_t)(wg * 64 + (s >> 3) * 16 + lane) * Tr::VD + (s & 7) * 256);
  load_prt_b<Tr, DT>(P, wg, s, lane, prt_r, prt_s);

  // --- S3: gated RMSNorm (all heads, all rows) -> perm A tiles ; tail --------
  if (!SPLIT || X.seg == SEG_T1 || X.seg == SEG_T1A || X.seg == SEG_T1B) {
    wait_for(it, S.cnt, C_B, Tr::NWG * call);
    if constexpr (SPLIT) {  // the q/k conv windows (see gdn_layer)
      if ((X.seg == SEG_T1 || X.seg == SEG_T1A) && lid < 256 && (h & 1) == 0 && c == 0) {
#pragma unroll
        for (int b = 0; b < NB; ++b) {
          e16* cv = conv_of(b) + (size_t)cch * chs[b];
          const e16 t1 = cv[1], t2 = cv[2];
          cv[0] = t1;
          cv[1] = t2;
          st16<DT>(cv, 2, (float)SB.qkv[b * Tr::CONVD + cch]);
        }
      }
    }
    sycl::vec<half, 4> o4[NB], z4[NB];
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      o4[b] = *reinterpret_cast<const sycl::vec<half, 4>*>(SB.core + b * Tr::VD + lid * 4);
      z4[b] = *reinterpret_cast<const sycl::vec<half, 4>*>(SB.z + b * Tr::VD + lid * 4);
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
      const float rs = sycl::rsqrt((L.sm[640 + b * 32 + 2 * hh] + L.sm[640 + b * 32 + 2 * hh + 1]) / Tr::DV + Tr::EPS);
      half y[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float zf = (float)z4[b][i];
        y[i] = toh<DT>(
            (float)o4[b][i] * rs * ld16<DT>(P.gnorm_w, (lid * 4 + i) & (Tr::DV - 1)) * (zf / (1.f + sycl::exp(-zf))));
      }
      a_put4_perm(L.xa, b, lid * 4, y);
    }
    wg_bar(grp);
  }
  tail_b<Tr, NB, DT, SPLIT>(it, P, S, SB, L, call, C_O, ow, prt_r, prt_s, X);
}

template <class Tr, int NB, int T, typename DT, bool SPLIT = false>
inline void gdn_layer_s(
    sycl::nd_item<1> it, uint8_t* slm, const Ptrs& P, const Scratch& S, const ScratchB& SB, const Split& X = Split{}) {
  constexpr int NRQ = NB / T;
  static_assert(NRQ * T == NB && T >= 2, "spec verify: NB = requests x tokens");
  const int wg = (SPLIT && X.vwg >= 0) ? X.vwg : (int)it.get_group(0);
  const int lid = it.get_local_linear_id();
  sycl::sub_group sg = it.get_sub_group();
  const int s = sg.get_group_linear_id();
  const int lane = sg.get_local_linear_id();
  auto grp = it.get_group();
  const uint32_t call = uc_load32(S.cnt + EPOCH) + 1u;
  const SlmB L = slm_b<Tr>(slm);
  float* qk =
      reinterpret_cast<float*>(slm + Tr::B_QK);  // per row r: q[0..127] k[128..255] v[256..319] beta g [320, 321]

  const int h = wg >> 1, c = wg & 1, kh = h >> 1;  // S2 role: v head h (k head h/2), v rows c*64..+64
  int sidx[NRQ], chs[NRQ];
#pragma unroll
  for (int b = 0; b < NRQ; ++b) {
    sidx[b] = P.state_idx[b];
    chs[b] = sidx[b] >= 0 ? P.conv_ch : 3;
  }
  auto conv_of = [&](int b) { return sidx[b] >= 0 ? P.conv_cache + (size_t)sidx[b] * P.conv_stride : S.dummy_conv; };
  auto ssm_of = [&](int b) { return sidx[b] >= 0 ? P.ssm_cache + (size_t)sidx[b] * P.ssm_stride : S.dummy_ssm; };
  auto islot = [&](int b) { return P.inter_idx ? P.inter_idx[b] : b; };

  // ---- kernel start: S1 weights in flight, the requests' initial
  // recurrent-state blocks of this WG -> L2
  const int tile1 = s >> 2, ks1 = s & 3;
  const int tz = s >> 3, kz = s & 7;
  // block scales of the S1 and z pieces, loaded ahead of the weight stream
  float s1c[4], szc[2];
#pragma unroll
  for (int j = 0; j < 4; ++j)
    s1c[j] = P.qkvz_s[(size_t)((wg * 128 + tile1 * 16) / Tr::QB) * Tr::KB_HID + ks1 * 4 + j];
#pragma unroll
  for (int j = 0; j < 2; ++j)
    szc[j] = P.qkvz_s[(size_t)((Tr::CONVD + h * Tr::DV + c * 64 + tz * 16) / Tr::QB) * Tr::KB_HID + kz * 2 + j];
  sycl::uint4 w1[32];
  ld_rows<32>(w1, P.qkvz_w + (size_t)(wg * 128 + tile1 * 16 + lane) * Tr::HID + ks1 * 512);
#pragma unroll
  for (int b = 0; b < NRQ; ++b)
    syclex::prefetch(
        (void*)(ssm_of(b) + ((size_t)h * Tr::DV + c * 64) * Tr::DK + lid * 16),
        syclex::properties{syclex::prefetch_hint_L2});
  const int cch = lid < 128   ? kh * Tr::DK + lid
                  : lid < 256 ? Tr::KD + kh * Tr::DK + (lid - 128)
                              : 2 * Tr::KD + h * Tr::DV + c * 64 + (lid - 256);
  float cwt[4];
  e16* csl = reinterpret_cast<e16*>(slm + Tr::B_CS);  // [b][i][320]: tap i of request b, the WG's channels
  load_conv_b<Tr, NRQ>(csl, conv_of, chs, h, c, kh, lid);
  if (lid < 320) {
    const sycl::vec<uint16_t, 4> w4 = *reinterpret_cast<const sycl::vec<uint16_t, 4>*>(P.conv_w + (size_t)cch * 4);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      cwt[i] = f32<DT>(w4[i]);
  }
  input_norm_b<Tr, NB, DT>(it, P, L);
  wg_bar(grp);

  // --- S1: q|k|v rows (128 per WG), z, b|a: as the batched kernel ------------
  sycl::uint4 zw[16], bw[16];
  {
    v8f cc = v8f(0.f);
    if ((!SPLIT || X.seg == SEG_F1)) mm_rows_r<32>(cc, w1, L.xa, ks1 * 32, lane, s1c);
    ld_rows<16>(zw, P.qkvz_w + (size_t)(Tr::CONVD + h * Tr::DV + c * 64 + tz * 16 + lane) * Tr::HID + kz * 256);
    ld_rows<16>(
        bw, reinterpret_cast<const uint8_t*>(P.ba_h + (size_t)((s >> 4) * 16 + lane) * Tr::HID + (s & 15) * 128));
    if ((!SPLIT || X.seg == SEG_F1)) red_put(L.red, s, lane, cc);
  }
  wg_bar(grp);
  if ((!SPLIT || X.seg == SEG_F1)) {
    for (int idx = lid; idx < NB * 128; idx += Tr::WGS) {
      const int b = idx >> 7, r = idx & 127, t = r >> 4, l = r & 15;
      float v = 0.f;
#pragma unroll
      for (int k2 = 0; k2 < 4; ++k2)
        v += red_get(L.red, t * 4 + k2, b, l);
      SB.qkv[b * Tr::CONVD + wg * 128 + r] = toh<DT>(v * 256.f);
    }
    signal(it, S.cnt + C_A * 16);
  }
  {
    v8f cz = v8f(0.f), cb = v8f(0.f);
    mm_rows_r<16>(cz, zw, L.xa, kz * 16, lane, szc);
#pragma unroll
    for (int t = 0; t < 8; ++t)
      dpas_hf(
          cb,
          *reinterpret_cast<const v4u*>(L.xa + (((s & 15) * 8 + t) * 16 + lane) * 4),
          cvt16_w16<half>(bw[2 * t], bw[2 * t + 1]));
    wg_bar(grp);
    red_put(L.red, s, lane, cz);
    if (lane == h) {
#pragma unroll
      for (int b = 0; b < NB; ++b)
        L.sm[256 + (s >> 4) * 128 + b * 16 + (s & 15)] = cb[b];
    }
  }
  wg_bar(grp);
  {
    for (int idx = lid; idx < NB * 64; idx += Tr::WGS) {
      const int b = idx >> 6, r = idx & 63, t = r >> 4, l = r & 15;
      float v = 0.f;
#pragma unroll
      for (int k2 = 0; k2 < 8; ++k2)
        v += red_get(L.red, t * 8 + k2, b, l);
      if ((!SPLIT || X.seg == SEG_F1)) SB.z[b * Tr::VD + h * Tr::DV + c * 64 + r] = toh<DT>(v * 256.f);
    }
    if (lid < 2 * NB) {
      const int which = lid / NB, b = lid % NB;
      float v = 0.f;
#pragma unroll
      for (int k2 = 0; k2 < 16; ++k2)
        v += L.sm[256 + which * 128 + b * 16 + k2];
      L.sm[512 + which * 8 + b] = rnd<DT>(v);
    }
  }
  if (SPLIT && X.seg == SEG_F1) return;
  // initial recurrent state of each request (from L2)
  sycl::vec<float, 8> st[NRQ][2];
#pragma unroll
  for (int b = 0; b < NRQ; ++b)
#pragma unroll
    for (int rr = 0; rr < 2; ++rr)
      st[b][rr] = *reinterpret_cast<const sycl::vec<float, 8>*>(
          ssm_of(b) + ((size_t)h * Tr::DV + c * 64 + 2 * s + rr) * Tr::DK + lane * 8);

  // --- S2: conv1d over each request's window (all tokens at once), gated delta
  // rule in token order ---
  if ((!SPLIT || X.seg == SEG_F2)) {
    wait_for(it, S.cnt, C_A, Tr::NWG * call);
    if (lid < 320) {
#pragma unroll
      for (int b = 0; b < NRQ; ++b) {
        float win[T + 3];  // DT values
        win[0] = f32<DT>(csl[(b * 3) * 320 + lid].u);
        win[1] = f32<DT>(csl[(b * 3 + 1) * 320 + lid].u);
        win[2] = f32<DT>(csl[(b * 3 + 2) * 320 + lid].u);
#pragma unroll
        for (int t = 0; t < T; ++t)
          win[3 + t] = (float)SB.qkv[(b * T + t) * Tr::CONVD + cch];
#pragma unroll
        for (int t = 0; t < T; ++t) {
          float acc = win[t] * cwt[0];
          acc += win[t + 1] * cwt[1];
          acc += win[t + 2] * cwt[2];
          acc += win[t + 3] * cwt[3];
          qk[(b * T + t) * 384 + lid] = rnd<DT>(acc / (1.f + sycl::exp(-acc)));
        }
        // the windows: taps 1, 2, then the T inputs (to inter_conv, not the conv
        // state read above: no order needed between the k head's 4 WGs)
        if (lid >= 256 || ((h & 1) == 0 && c == 0)) {
          e16* ln = P.inter_conv + (size_t)islot(b) * P.inter_conv_stride + (size_t)cch * P.inter_conv_cstride;
#pragma unroll
          for (int j = 0; j < T + 2; ++j)
            st16<DT>(ln, j, win[j + 1]);
        }
      }
    }
    if (lid < NB) {  // beta, g of v head h per row
      const float bb = L.sm[512 + lid], a = L.sm[520 + lid] + ld16<DT>(P.dt_bias, h);
      const float sp = a < 20.f ? sycl::log(1.f + sycl::exp(a)) : a;
      qk[lid * 384 + 320] = 1.f / (1.f + sycl::exp(-bb));
      qk[lid * 384 + 321] = sycl::exp(-sycl::exp(P.A_log[h]) * sp);
    }
    wg_bar(grp);
    if (s < 2 * NB) {  // L2 norms: SG (row s/2, q | k)
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
    for (int b = 0; b < NRQ; ++b) {
      float* ib = P.inter_ssm + (size_t)islot(b) * P.inter_ssm_stride;
#pragma unroll
      for (int t = 0; t < T; ++t) {
        const int row = b * T + t;
        const float* q_ = qk + row * 384;
        const float iq = 1.f / sycl::sqrt(L.sm[600 + row * 2] + 1e-6f),
                    ik = 1.f / sycl::sqrt(L.sm[600 + row * 2 + 1] + 1e-6f);
        const float scale = 0.08838834764831845f;
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
        if (lane < 2) SB.core[row * Tr::VD + h * Tr::DV + c * 64 + 2 * s + lane] = toh<DT>(lane == 0 ? o[0] : o[1]);
        float* dst = ib + (size_t)t * P.inter_ssm_tstride;  // the state after token t
#pragma unroll
        for (int rr = 0; rr < 2; ++rr)
          st_store8(dst + ((size_t)h * Tr::DV + c * 64 + 2 * s + rr) * Tr::DK + lane * 8, st[b][rr]);
      }
    }
    signal(it, S.cnt + C_B * 16);
  }
  if (SPLIT && X.seg == SEG_F2) return;
  sycl::uint4 ow[16], prt_r[4], prt_s[4];
  ld_rows<16>(ow, P.out_w + (size_t)(wg * 64 + (s >> 3) * 16 + lane) * Tr::VD + (s & 7) * 256);
  load_prt_b<Tr, DT>(P, wg, s, lane, prt_r, prt_s);

  // --- S3: gated RMSNorm (all heads, all rows) -> perm A tiles ; tail --------
  if (!SPLIT || X.seg == SEG_T1 || X.seg == SEG_T1A || X.seg == SEG_T1B) {
    wait_for(it, S.cnt, C_B, Tr::NWG * call);
    sycl::vec<half, 4> o4[NB], z4[NB];
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      o4[b] = *reinterpret_cast<const sycl::vec<half, 4>*>(SB.core + b * Tr::VD + lid * 4);
      z4[b] = *reinterpret_cast<const sycl::vec<half, 4>*>(SB.z + b * Tr::VD + lid * 4);
      float ss = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i)
        ss += (float)o4[b][i] * (float)o4[b][i];
      ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
      if (lane == 0) L.sm[640 + b * 32 + s] = ss;
    }
    wg_bar(grp);
    const int hh = s >> 1;
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      const float rs = sycl::rsqrt((L.sm[640 + b * 32 + 2 * hh] + L.sm[640 + b * 32 + 2 * hh + 1]) / Tr::DV + Tr::EPS);
      half y[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float zf = (float)z4[b][i];
        y[i] = toh<DT>(
            (float)o4[b][i] * rs * ld16<DT>(P.gnorm_w, (lid * 4 + i) & (Tr::DV - 1)) * (zf / (1.f + sycl::exp(-zf))));
      }
      a_put4_perm(L.xa, b, lid * 4, y);
    }
    wg_bar(grp);
  }
  tail_b<Tr, NB, DT, SPLIT>(it, P, S, SB, L, call, C_O, ow, prt_r, prt_s, X);
}

}  // namespace sgl_kernel::stage::xe20
