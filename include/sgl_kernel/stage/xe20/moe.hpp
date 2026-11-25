// SPDX-License-Identifier: Apache-2.0
//
// The MoE tail every decoder layer ends with, single-row and batched:
// out_proj / o_proj -> all-reduce #1 -> residual -> post-attention norm ->
// router / shared-expert partials -> fixed-order reduction -> top-8 -> routed
// gate|up + shared expert -> down -> all-reduce #2. The mixers (gdn.hpp,
// attn.hpp) prefetch the PRT column blocks (load_prt, load_prt_b) before
// they hand over to it.
//
// Stage code: every function that depends on the model's shape takes the
// traits Tr (Dims<Shape>, dims.hpp) as its first template argument.
#pragma once
#include "sgl_kernel/stage/xe20/layer.hpp"

namespace sgl_kernel::stage::xe20 {

// ----------------------------------------------------------------------------
// tail Entry: the mixer output o (2048 fp16) is in SLM xp (permuted), the
// out_proj rows ow[] of this SG are in registers (issued after the previous
// signal) with their scales osc, prt_r / prt_s the router / shared gate|up
// column blocks of this WG.
template <class Tr, typename DT, bool SPLIT = false>
inline void tail(
    sycl::nd_item<1> it,
    const Ptrs& P,
    const Scratch& S,
    const Slm& L,
    uint32_t call,
    int c_o,
    const sycl::uint4 (&ow)[2][8],
    const float (&osc)[8],
    const sycl::uint4 (&prt_r)[4],
    const sycl::uint4 (&prt_s)[Tr::SHR / 4],
    const Split& X = Split{}) {
  const uint32_t mtag = uc_load32(S.cnt + MBT) + 1u;  // rows-1 mailbox region
  const int wg = (SPLIT && X.vwg >= 0) ? X.vwg : (int)it.get_group(0);
  const int lid = it.get_local_linear_id();
  sycl::sub_group sg = it.get_sub_group();
  const int s = sg.get_group_linear_id();
  const int lane = sg.get_local_linear_id();
  auto grp = it.get_group();
  // split: the tail's first part runs in its segments only (the later ones
  // read its results from global memory)
  if (!SPLIT || X.seg == SEG_T1 || X.seg == SEG_T1A || X.seg == SEG_T1B) {
    const int r0 = wg * 64 + s * 2;
    const float o0 = dot_row_regs<8>(sg, lane, ow[0], L.xp32, osc), o1 = dot_row_regs<8>(sg, lane, ow[1], L.xp32, osc);
    float ov = (lane == 0 ? o0 : o1) * 256.f;
    {  // all-reduce #1 over this WG's 64 rows, one contiguous block of 64
       // work-items. a + b == b + a
      float* ob = reinterpret_cast<float*>(L.ao + 1536);
      if (lane < 2) ob[s * 2 + lane] = ov;
      wg_bar(grp);
      if (lid < 32) {  // two rows a word
        float v0 = ob[2 * lid], v1 = ob[2 * lid + 1];
        if ((SPLIT && X.ar)) {  // mode 2: the caller all-reduces ar between T1A and T1B
          float* ar = X.ar + wg * 64 + 2 * lid;
          if (X.seg == SEG_T1A) {
            ar[0] = ar_send(v0);
            ar[1] = ar_send(v1);
          } else {
            v0 = ar[0];
            v1 = ar[1];
          }
        } else {
          mb_ar2(S.peer + wg * 32 + lid, S.mbox + wg * 32 + lid, v0, v1, mtag, S.cnt);
        }
        ob[2 * lid] = v0;
        ob[2 * lid + 1] = v1;
      }
      wg_bar(grp);
      if (SPLIT && X.seg == SEG_T1A) return;
      if (lane < 2) ov = ob[s * 2 + lane];
    }
    float* rblk = reinterpret_cast<float*>(L.ao);     // this WG's 64 res2 rows
    half* xbn = reinterpret_cast<half*>(L.ao + 256);  // (1+w) * res2, natural
    half* xbp = reinterpret_cast<half*>(L.ao + 384);  // same, permuted
    if (lane < 2) {
      const int r = r0 + lane;
      const float res = rnd<DT>(ld16<DT>(P.hs_in, r) + ld16<DT>(P.res_in, r));
      const float r2 = rnd<DT>(ov + res);
      st16<DT>(P.res_out, r, r2);
      rblk[s * 2 + lane] = r2;
    }
    wg_bar(grp);
    if (lid < 64) xbn[lid] = (half)(rblk[lid] * (1.f + ld16<DT>(P.post_norm_w, wg * 64 + lid)));
    if (s == 2) {
      float q = rblk[lane * 4] * rblk[lane * 4] + rblk[lane * 4 + 1] * rblk[lane * 4 + 1] +
                rblk[lane * 4 + 2] * rblk[lane * 4 + 2] + rblk[lane * 4 + 3] * rblk[lane * 4 + 3];
      q = sycl::reduce_over_group(sg, q, sycl::plus<float>());
      if (lane == 0) S.part_sg[wg * 2 + 1] = q;
    }
    wg_bar(grp);
    if (lid < 64) store_perm(xbp, lid, xbn[lid]);
    wg_bar(grp);
    {  // router partials: 8 experts per SG
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const sycl::vec<uint16_t, 8> wh = sycl::bit_cast<sycl::vec<uint16_t, 8>>(prt_r[t]);
        float v = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i)
          v += f32<DT>(wh[i]) * (float)xbn[(lane & 7) * 8 + i];
        v += sycl::permute_group_by_xor(sg, v, 4);
        v += sycl::permute_group_by_xor(sg, v, 2);
        v += sycl::permute_group_by_xor(sg, v, 1);
        if ((lane & 7) == 0) S.part_rt[wg * Tr::NE + s * 8 + 2 * t + (lane >> 3)] = v;
      }
      // shared gate|up partials: SHR rows per SG (rows s*16.., one N block),
      // the WG's 64 columns (one K block): scaled here
      const uint32_t* xb32 = reinterpret_cast<const uint32_t*>(xbp);
      const float ssc = P.sw13_s[(s * Tr::SHR / Tr::QB) * Tr::KB_HID + (wg * 64) / Tr::QB];
#pragma unroll
      for (int t = 0; t < Tr::SHR / 4; ++t) {
        float v = dot16_fp8(prt_s[t], xb32 + (lane & 3) * 8);
        v += sycl::permute_group_by_xor(sg, v, 2);
        v += sycl::permute_group_by_xor(sg, v, 1);
        if ((lane & 3) == 0) S.part_sh[wg * 2 * Tr::INTER + s * Tr::SHR + 4 * t + (lane >> 2)] = v * ssc;
      }
      if (s == Tr::NSG - 1) {  // shared-expert gate partial
        float v = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i)
          v += ld16<DT>(P.sgate_w, wg * 64 + lane * 4 + i) * (float)xbn[lane * 4 + i];
        v = sycl::reduce_over_group(sg, v, sycl::plus<float>());
        if (lane == 0) S.part_sg[wg * 2] = v;
      }
    }
  }
  if (!SPLIT || X.seg == SEG_T1 || X.seg == SEG_T1A || X.seg == SEG_T1B) signal(it, S.cnt + c_o * 16);
  if (SPLIT && (!SPLIT || X.seg == SEG_T1 || X.seg == SEG_T1A || X.seg == SEG_T1B)) return;

  // ---------------- S5': reduce the partials (fixed order): rs, x2, logits,
  // shared slice, gate -----
  wait_for(it, S.cnt + c_o * 16, Tr::NWG * call);
  float* lgs = reinterpret_cast<float*>(L.ao + 512);  // 256 logits
  {
    float ss = 0.f;
#pragma unroll
    for (int w2i = 0; w2i < Tr::NWG; ++w2i)
      ss += S.part_sg[w2i * 2 + 1];
    const float rs = sycl::rsqrt(ss / Tr::HID + Tr::EPS);
    half x2[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
      x2[i] = toh<DT>(ld16<DT>(P.res_out, lid * 4 + i) * rs * (1.f + ld16<DT>(P.post_norm_w, lid * 4 + i)));
    // x2, permuted (the routed gate|up rows; xp's mixer output is consumed)
    *reinterpret_cast<sycl::vec<half, 4>*>(L.xp + lid * 4) = sycl::vec<half, 4>(x2[0], x2[2], x2[1], x2[3]);
    if (lid < Tr::NE) {
      float a = 0.f;
#pragma unroll
      for (int w2i = 0; w2i < Tr::NWG; ++w2i)
        a += S.part_rt[w2i * Tr::NE + lid];
      lgs[lid] = a * rs;
    } else if (lid < Tr::NE + Tr::SPW) {
      const int i = wg * Tr::SPW + lid - Tr::NE;
      float g = 0.f, u = 0.f;
#pragma unroll
      for (int w2i = 0; w2i < Tr::NWG; ++w2i) {
        g += S.part_sh[w2i * 2 * Tr::INTER + i];
        u += S.part_sh[w2i * 2 * Tr::INTER + Tr::INTER + i];
      }
      const float sc = 256.f * rs;
      g *= sc;
      u *= sc;
      if ((!SPLIT || X.seg == SEG_S56)) S.hbuf[Tr::TOPK * Tr::INTER + i] = toh<DT>(g / (1.f + sycl::exp(-g)) * u);
    } else if (lid == Tr::NE + 32) {
      float a = 0.f;
#pragma unroll
      for (int w2i = 0; w2i < Tr::NWG; ++w2i)
        a += S.part_sg[w2i * 2];
      L.sgate[0] = 1.f / (1.f + sycl::exp(-a * rs));
    }
    wg_bar(grp);
  }
  // top-8 + softmax over the selected logits (ties: lower index first)
  if (s == 0) {
    float lg[16];
#pragma unroll
    for (int i = 0; i < 16; ++i)
      lg[i] = lgs[lane * 16 + i];
    float lmax = -INFINITY, sel_v[Tr::TOPK];
    int seli[Tr::TOPK];
    for (int k = 0; k < Tr::TOPK; ++k) {
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
    float sum = 0.f, e[Tr::TOPK];
    for (int k = 0; k < Tr::TOPK; ++k) {
      e[k] = sycl::exp(sel_v[k] - lmax);
      sum += e[k];
    }
    if (lane == 0)
      for (int k = 0; k < Tr::TOPK; ++k) {
        L.route_e[k] = seli[k] & (Tr::NE - 1);
        L.route_w[k] = e[k] / sum;
      }
  }
  wg_bar(grp);

  // ---------------- S6: routed gate|up, layout w13[E][N = 512][K = 2048]
  // WG = (slot j = wg/4, quarter q4): gate rows q4*64..+64 (SGs 0..15) and the
  // same up rows (+256, SGs 16..31), 4 rows per SG, each row 8 chunks of 16 B a
  // lane (one N block per SG).
  // the down rows of this WG's 64 outputs (routed slot s/4, rows (s&3)*16 +
  // lane, a whole 256-B row a lane) and the shared down rows: issued right
  // after the gate|up rows, in flight during S6 and the hand-off
  sycl::uint4 wd[16], wsd[2];
  float dsc[2];
  {
    const int j = wg >> 2, q4 = wg & 3, e = L.route_e[j];
    const int r0 = (s >> 4) * Tr::INTER + q4 * 64 + (s & 15) * 4;
    // split: the S7 segments only need the down rows (not the gate|up ones)
    sycl::uint4 gw[4][8];
    float gsc[8];
    if (!SPLIT || X.seg == SEG_S56) {
#pragma unroll
      for (int b = 0; b < 4; ++b)
        load_row<8>(gw[b], P.w13 + ((size_t)e * 2 * Tr::INTER + r0 + b) * Tr::HID, lane);
      load_scales<8>(gsc, P.w13_s + ((size_t)e * (2 * Tr::INTER / Tr::QB) + r0 / Tr::QB) * Tr::KB_HID, lane);
    }
    {
      const int jd = s >> 2, ed = L.route_e[jd];
      const int n = wg * 64 + (s & 3) * 16 + lane;
      const uint8_t* rowd = P.w2 + ((size_t)ed * Tr::HID + n) * Tr::INTER;
#pragma unroll
      for (int c = 0; c < 16; ++c)
        wd[c] = *reinterpret_cast<const sycl::uint4*>(rowd + c * 16);
      const float* sd = P.w2_s + ((size_t)ed * Tr::KB_HID + (wg * 64) / Tr::QB) * (Tr::INTER / Tr::QB);
      dsc[0] = sd[0];
      dsc[1] = sd[1];
#pragma unroll
      for (int rr = 0; rr < 2; ++rr)
        wsd[rr] = *reinterpret_cast<const sycl::uint4*>(P.sw2 + (size_t)(wg * 64 + 2 * s + rr) * Tr::INTER + lane * 16);
    }
    if (!SPLIT || X.seg == SEG_S56) {
      float v[4];
#pragma unroll
      for (int b = 0; b < 4; ++b)
        v[b] = dot_row_regs<8>(sg, lane, gw[b], L.xp32, gsc);
      if (lane < 4)
        L.qk[(s >> 4) * 64 + (s & 15) * 4 + lane] = (lane == 0   ? v[0]
                                                     : lane == 1 ? v[1]
                                                     : lane == 2 ? v[2]
                                                                 : v[3]) *
                                                    256.f;
      wg_bar(grp);
      if (lid < 64) {
        const float g = L.qk[lid], u = L.qk[64 + lid];
        S.hbuf[j * Tr::INTER + q4 * 64 + lid] = toh<DT>(g / (1.f + sycl::exp(-g)) * u);
      }
    }
  }
  if ((!SPLIT || X.seg == SEG_S56)) signal(it, S.cnt + C_GU * 16);
  if (SPLIT && X.seg == SEG_S56) return;

  // ---------------- S7: down for this WG's 64 rows ; all-reduce #2
  // -----------------------------
  wait_for(it, S.cnt + C_GU * 16, Tr::NWG * call);
  if (!SPLIT && wg == 0 && lid == 0) {  // every WG has read the epoch and the tag (split: split_advance)
    cnt_ref(S.cnt[EPOCH]).store(call);
    cnt_ref(S.cnt[MBT]).store(mtag);
  }
  // routed + shared h -> SLM (every lane reads all of its slot's h)
  for (int i = lid; i < (Tr::TOPK + 1) * Tr::INTER / 8; i += Tr::WGS)
    *reinterpret_cast<sycl::uint4*>(L.hb + i * 8) = *reinterpret_cast<const sycl::uint4*>(S.hbuf + i * 8);
  wg_bar(grp);
  {
    const int jd = s >> 2;
    const float coef = L.route_w[jd] * 256.f;
    const half* hj = L.hb + jd * Tr::INTER;
    float a0 = 0.f, a1 = 0.f;
#pragma unroll
    for (int c = 0; c < 8; ++c)
      a0 += dot16_fp8_nat(wd[c], *reinterpret_cast<const sycl::vec<half, 16>*>(hj + c * 16));
#pragma unroll
    for (int c = 8; c < 16; ++c)
      a1 += dot16_fp8_nat(wd[c], *reinterpret_cast<const sycl::vec<half, 16>*>(hj + c * 16));
    L.mr[jd * 64 + (s & 3) * 16 + lane] = (a0 * dsc[0] + a1 * dsc[1]) * coef;
    // shared down: rows wg*64 + 2s + rr, this lane's 16 K (K block lane / 8)
    const sycl::vec<half, 16> hx =
        *reinterpret_cast<const sycl::vec<half, 16>*>(L.hb + Tr::TOPK * Tr::INTER + lane * 16);
    const float ssc = P.sw2_s[((wg * 64) / Tr::QB) * (Tr::INTER / Tr::QB) + (lane >> 3)];
    float d0 = dot16_fp8_nat(wsd[0], hx) * ssc, d1 = dot16_fp8_nat(wsd[1], hx) * ssc;
    d0 = sycl::reduce_over_group(sg, d0, sycl::plus<float>());
    d1 = sycl::reduce_over_group(sg, d1, sycl::plus<float>());
    const float cs = L.sgate[0] * 256.f;
    if (lane < 2) L.red[2 * s + lane] = (lane == 0 ? d0 : d1) * cs;
    wg_bar(grp);
    if (lid < 64) {
      float a = 0.f;
#pragma unroll
      for (int k = 0; k < Tr::TOPK; ++k)
        a += L.mr[k * 64 + lid];
      L.red[lid] += a;  // this row's partial (each work-item its own element)
    }
    wg_bar(grp);
    if (lid < 32) {  // all-reduce #2, two rows a word
      float a0 = L.red[2 * lid], a1 = L.red[2 * lid + 1];
      if ((SPLIT && X.ar)) {  // mode 2: the caller all-reduces ar between S7A and S7B
        float* ar = X.ar + wg * 64 + 2 * lid;
        if (X.seg == SEG_S7A) {
          ar[0] = ar_send(a0);
          ar[1] = ar_send(a1);
          return;
        }
        a0 = ar[0];
        a1 = ar[1];
      } else {
        mb_ar2(S.peer + Tr::HID + wg * 32 + lid, S.mbox + Tr::HID + wg * 32 + lid, a0, a1, mtag, S.cnt);
      }
      st16<DT>(P.hid_out, wg * 64 + 2 * lid, a0);
      st16<DT>(P.hid_out, wg * 64 + 2 * lid + 1, a1);
    }
  }
}

// PRT column blocks of this WG (router 16-bit, shared gate|up fp8) -> registers
template <class Tr>
inline void
load_prt(const Ptrs& P, int wg, int s, int lane, sycl::uint4 (&prt_r)[4], sycl::uint4 (&prt_s)[Tr::SHR / 4]) {
#pragma unroll
  for (int t = 0; t < 4; ++t)  // experts s*8 + 2t + (lane>>3), halves (lane&7)*8..+8
    prt_r[t] = *reinterpret_cast<const sycl::uint4*>(
        P.router_w + (size_t)(s * 8 + 2 * t + (lane >> 3)) * Tr::HID + wg * 64 + (lane & 7) * 8);
#pragma unroll
  for (int t = 0; t < Tr::SHR / 4; ++t)  // rows s*SHR + 4t + (lane>>2), bytes (lane&3)*16..+16
    prt_s[t] = *reinterpret_cast<const sycl::uint4*>(
        P.sw13 + (size_t)(s * Tr::SHR + 4 * t + (lane >> 2)) * Tr::HID + wg * 64 + (lane & 3) * 16);
}

// -------------------------------------------------------------- tail (batched)
// Entry: the mixer output (NB rows) is in L.xa as perm A tiles; ow = this SG's
// o_proj / out_proj piece (rows wg*64 + (s>>3)*16 + lane, K chunks
// (s&7)*16..+16), prt_r / prt_s = router / shared gate|up B pieces of this WG.
template <class Tr, int NB, typename DT, bool SPLIT = false>
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
    const sycl::uint4 (&prt_s)[4],
    const Split& X = Split{}) {
  const uint32_t mtag = uc_load32(S.cnt + mb_tag<NB>()) + 1u;  // this configuration's mailbox region and tag
  constexpr size_t MB0 = mb_region<Tr, NB>(), MB1 = mb_region<Tr, NB>() + (size_t)NB * Tr::HID / 2;
  const int wg = (SPLIT && X.vwg >= 0) ? X.vwg : (int)it.get_group(0);
  const int lid = it.get_local_linear_id();
  sycl::sub_group sg = it.get_sub_group();
  const int s = sg.get_group_linear_id();
  const int lane = sg.get_local_linear_id();
  auto grp = it.get_group();
  const int n0 = wg * 64;
  // split: the tail's first part runs in its segments only (the later ones
  // read its results from global memory)
  if (!SPLIT || X.seg == SEG_T1 || X.seg == SEG_T1A || X.seg == SEG_T1B) {
    // ---- out_proj: 4 row tiles x 8 K slices (K 2048, the WG's 64 rows: one N
    // block)
    {
      v8f c = v8f(0.f);
      mm_rows<16>(c, ow, L.xa, (s & 7) * 16, lane, P.out_s + (size_t)(n0 / Tr::QB) * Tr::KB_HID);
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
      if ((SPLIT && X.ar)) {  // mode 2: the caller all-reduces ar between T1A and T1B
        float* ar = X.ar + (size_t)pb * Tr::HID + n0 + pc;
        if (X.seg == SEG_T1A) {
          ar[0] = ar_send(v0);
          ar[1] = ar_send(v1);
        } else {
          v0 = ar[0];
          v1 = ar[1];
        }
      } else {
        ar_two(S.peer, S.mbox, MB0 + (((size_t)pb * Tr::HID + n0 + pc) >> 1), v0, v1, mtag, S.cnt);
      }
      if (!(SPLIT && X.seg == SEG_T1A)) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int n = n0 + pc + e;
          const float res = rnd<DT>(ld16<DT>(P.hs_in, pb * Tr::HID + n) + ld16<DT>(P.res_in, pb * Tr::HID + n));
          const float r2 = rnd<DT>((e ? v1 : v0) + res);
          st16<DT>(P.res_out, pb * Tr::HID + n, r2);
          L.rb[pb * 64 + pc + e] = r2;
        }
      }
    }
    if (SPLIT && X.seg == SEG_T1A) return;
    a_zero_pad<NB>(L.xan, 0, 4, lid, Tr::WGS);
    a_zero_pad<NB>(L.xap, 0, 4, lid, Tr::WGS);
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
      SB.part_sg[(wg * Tr::NBMAX + lid) * 2] = gq;
      SB.part_sg[(wg * Tr::NBMAX + lid) * 2 + 1] = q;
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
      const float ssc = P.sw13_s[(s * 16 / Tr::QB) * Tr::KB_HID + n0 / Tr::QB];
#pragma unroll
      for (int b = 0; b < NB; ++b)
        SB.part_sh[((size_t)wg * Tr::NBMAX + b) * 2 * Tr::INTER + s * 16 + lane] = d[b] * ssc;
    }
    wg_bar(grp);
    for (int idx = lid; idx < NB * Tr::NE; idx += Tr::WGS) {
      const int b = idx / Tr::NE, e = idx % Tr::NE, t = e >> 4, l = e & 15;
      SB.part_rt[((size_t)wg * Tr::NBMAX + b) * Tr::NE + e] =
          red_get(L.red, 2 * t, b, l) + red_get(L.red, 2 * t + 1, b, l);
    }
    signal(it, S.cnt + c_o * 16);
  }
  if (SPLIT && (!SPLIT || X.seg == SEG_T1 || X.seg == SEG_T1A || X.seg == SEG_T1B)) return;

  // --- S5': reduce the partials (fixed order) per row ------------------------
  wait_for(it, S.cnt + c_o * 16, Tr::NWG * call);
  if (lid < NB) {
    float ss = 0.f, ga = 0.f;
#pragma unroll
    for (int w2i = 0; w2i < Tr::NWG; ++w2i) {
      ss += SB.part_sg[(w2i * Tr::NBMAX + lid) * 2 + 1];
      ga += SB.part_sg[(w2i * Tr::NBMAX + lid) * 2];
    }
    const float rs = sycl::rsqrt(ss / Tr::HID + Tr::EPS);
    L.sm[128 + lid] = rs;
    L.sgt[lid] = 1.f / (1.f + sycl::exp(-ga * rs));  // shared-expert gate
  }
  a_zero_pad<NB>(L.xa, 0, 128, lid, Tr::WGS);
  wg_bar(grp);
#pragma unroll
  for (int b = 0; b < NB; ++b) {  // x2 = GemmaRMS(res2) -> perm A tiles (routed experts)
    const float rs = L.sm[128 + b];
    half x2[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
      x2[i] =
          toh<DT>(ld16<DT>(P.res_out, b * Tr::HID + lid * 4 + i) * rs * (1.f + ld16<DT>(P.post_norm_w, lid * 4 + i)));
    a_put4_perm(L.xa, b, lid * 4, x2);
  }
  for (int idx = lid; idx < NB * Tr::NE; idx += Tr::WGS) {
    const int b = idx / Tr::NE, e = idx % Tr::NE;
    float a = 0.f;
#pragma unroll
    for (int w2i = 0; w2i < Tr::NWG; ++w2i)
      a += SB.part_rt[((size_t)w2i * Tr::NBMAX + b) * Tr::NE + e];
    L.lg[b * Tr::NE + e] = a * L.sm[128 + b];
  }
  {  // shared h, the WG's SPW inter rows, all rows (padding rows -> 0): 8
     // work-items per (row, inter row), 4 WGs' partials each, combined over the
     // 8 lanes
    const int pr = lid >> 3, w8 = lid & 7;
    const int b = pr / Tr::SPW, i = wg * Tr::SPW + pr % Tr::SPW;
    float g = 0.f, u = 0.f;
    if (b < NB) {
#pragma unroll
      for (int k = 0; k < Tr::NWG / 8; ++k) {
        const int w2i = w8 * (Tr::NWG / 8) + k;
        g += SB.part_sh[((size_t)w2i * Tr::NBMAX + b) * 2 * Tr::INTER + i];
        u += SB.part_sh[((size_t)w2i * Tr::NBMAX + b) * 2 * Tr::INTER + Tr::INTER + i];
      }
    }
#pragma unroll
    for (int m = 4; m >= 1; m >>= 1) {
      g += sycl::permute_group_by_xor(sg, g, m);
      u += sycl::permute_group_by_xor(sg, u, m);
    }
    if (w8 == 0 && pr < Tr::NBMAX * Tr::SPW) {
      half hv = (half)0.f;
      if (b < NB) {
        const float sc = 256.f * L.sm[128 + b];
        g *= sc;
        u *= sc;
        hv = toh<DT>(g / (1.f + sycl::exp(-g)) * u);
      }
      if ((!SPLIT || X.seg == SEG_S56)) a_put_perm(SB.hs_a, b, i, hv);  // (split: S7 reads it)
    }
  }
  wg_bar(grp);
  // top-8 per row (ties: lower index first), softmax over the selection
  if (s < NB) {
    const float* lgs = L.lg + s * Tr::NE;
    float lg[16];
#pragma unroll
    for (int i = 0; i < 16; ++i)
      lg[i] = lgs[lane * 16 + i];
    float lmax = -INFINITY, sel_v[Tr::TOPK];
    int seli[Tr::TOPK];
#pragma unroll
    for (int k = 0; k < Tr::TOPK; ++k) {
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
    float sum = 0.f, e[Tr::TOPK];
#pragma unroll
    for (int k = 0; k < Tr::TOPK; ++k) {
      e[k] = sycl::exp(sel_v[k] - lmax);
      sum += e[k];
    }
    if (lane == 0)
#pragma unroll
      for (int k = 0; k < Tr::TOPK; ++k) {
        L.re[s * Tr::TOPK + k] = seli[k] & (Tr::NE - 1);
        L.rw[s * Tr::TOPK + k] = e[k] / sum;
      }
  }
  wg_bar(grp);
  if (s == 0) {  // union of the selections, ascending expert id: lane owns
                 // experts 16 lane .. +15 (bit masks, SG scan)
    uint32_t m16 = 0u;
#pragma unroll
    for (int j = 0; j < NB * Tr::TOPK; ++j) {
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
    if (lid < nu0 * Tr::NBMAX) {
      const int u = lid / Tr::NBMAX, b = lid % Tr::NBMAX, e = L.uexp[u];
      float w = 0.f;
      if (b < NB) {
#pragma unroll
        for (int k = 0; k < Tr::TOPK; ++k)
          if (L.re[b * Tr::TOPK + k] == e) w = L.rw[b * Tr::TOPK + k];
      }
      L.uw[u * Tr::NBMAX + b] = w;
    }
  }
  wg_bar(grp);
  const int nu = L.nu[0];

  // --- S6: routed gate|up, tasks (union expert u, row eighth j) dealt to the
  // WGs. Task rows: gate 32j .. +32 and the same up rows (+256) of w13 [E][512]
  // [2048]. SG: gate (s < 16) or up (s >= 16), K slice s&15 (128 K: 8 chunks,
  // one K block), two 16-row tiles (lane = row). Eight tasks per expert keep
  // the WGs' task counts within one small task of each other.
  if ((!SPLIT || X.seg == SEG_S56)) {  // (split: the S7 segments skip it)
    const int ntask = nu * 8, gu = s >> 4, ks = s & 15;
    auto wrow = [&](int task, int tl) {  // this lane's row of tile tl, K slice ks
      return P.w13 +
             ((size_t)L.uexp[task >> 3] * 2 * Tr::INTER + gu * Tr::INTER + (task & 7) * 32 + tl * 16 + lane) * Tr::HID +
             ks * Tr::QB;
    };
    // the task's block scale, loaded ahead of its weights (behind them it
    // waits for their last bytes)
    auto wsc = [&](int task) {
      return P.w13_s
          [((size_t)L.uexp[task >> 3] * (2 * Tr::INTER / Tr::QB) + (gu * Tr::INTER + (task & 7) * 32) / Tr::QB) *
               Tr::KB_HID +
           ks];
    };
    sycl::uint4 wb[2][8];
    float scn = 0.f;
    if (wg < ntask) {
      scn = wsc(wg);
#pragma unroll
      for (int tl = 0; tl < 2; ++tl)
        ld_rows<8>(wb[tl], wrow(wg, tl));
    }
    for (int task = wg; task < ntask; task += Tr::NWG) {
      const int u = task >> 3, j8 = task & 7;
      const int nxt = task + Tr::NWG;
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
      if (lid < Tr::NBMAX * 32) {  // h[b][32 j8 + i] = silu(g) * u * w_b -> perm A
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
          hv = toh<DT>(g / (1.f + sycl::exp(-g)) * up * L.uw[u * Tr::NBMAX + b]);
        }
        a_put_perm(SB.hA + (size_t)u * (Tr::INTER / 16) * 64, b, j8 * 32 + i, hv);
      }
      wg_bar(grp);
    }
  }
  if ((!SPLIT || X.seg == SEG_S56)) signal(it, S.cnt + C_GU * 16);
  if (SPLIT && X.seg == SEG_S56) return;

  // --- S7: down for this WG's 64 output rows ; shared down ; all-reduce #2.
  // Blocks (union expert u, K quarter q: 64 K = 4 chunks = one 64-B line of
  // each row, row-tile pair p) of 2 x 16 rows (lane = row of w2 [E][2048][256]),
  // dealt round-robin to the SGs (8 per expert: the SGs' counts differ by at
  // most one block), 4 blocks in flight per SG; the first 2 are in flight
  // during the hand-off (they need the union only).
  const int nblk = nu * 8;
  auto brow = [&](int j, int ct) {  // j = (u, q, p): u = j >> 3, q = (j >> 1) & 3, p = j & 1
    return P.w2 + ((size_t)L.uexp[j >> 3] * Tr::HID + n0 + ((j & 1) * 2 + ct) * 16 + lane) * Tr::INTER +
           ((j >> 1) & 3) * 64;
  };
  // block j's scale (the WG's 64 rows: one N block; K block q / 2), loaded
  // ahead of the block's weights
  auto bsc = [&](int j) {
    return P.w2_s[((size_t)L.uexp[j >> 3] * Tr::KB_HID + n0 / Tr::QB) * (Tr::INTER / Tr::QB) + (((j >> 1) & 3) >> 1)];
  };
  constexpr int S7PRE = 2;
  sycl::uint4 wb7[4][2][4];
  float sc7[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int t = 0; t < S7PRE; ++t) {  // (not all 4: the counter poll would queue behind them)
    const int j = s + Tr::NSG * t;
    if (j < nblk) {
      sc7[t] = bsc(j);
#pragma unroll
      for (int ct = 0; ct < 2; ++ct)
        ld_rows<4>(wb7[t][ct], brow(j, ct));
    }
  }
  wait_for(it, S.cnt + C_GU * 16, Tr::NWG * call);
#pragma unroll
  for (int t = S7PRE; t < 4; ++t) {
    const int j = s + Tr::NSG * t;
    if (j < nblk) {
      sc7[t] = bsc(j);
#pragma unroll
      for (int ct = 0; ct < 2; ++ct)
        ld_rows<4>(wb7[t][ct], brow(j, ct));
    }
  }
  if (!SPLIT && wg == 0 && lid == 0) {  // every WG has read the epoch and the mailbox tag (split: split_advance)
    cnt_ref(S.cnt[EPOCH]).store(call);
    cnt_ref(S.cnt[mb_tag<NB>()]).store(mtag);
  }
  {
    v8f acc[4];
#pragma unroll
    for (int ct = 0; ct < 4; ++ct)
      acc[ct] = v8f(0.f);
    for (int j0 = s; j0 < nblk; j0 += 4 * Tr::NSG) {
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const int j = j0 + Tr::NSG * t;
        if (j < nblk) {
          const int u = j >> 3, q = (j >> 1) & 3;
          const uint32_t* ha = SB.hA + (size_t)u * (Tr::INTER / 16) * 64;
          const float sc = sc7[t];
          v8f c[2];
#pragma unroll
          for (int ct = 0; ct < 2; ++ct) {
            c[ct] = v8f(0.f);
            mm_rows_u<4>(c[ct], wb7[t][ct], ha, q * 4, lane);
          }
          const int jn = j + 4 * Tr::NSG;
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
      ld_rows<2>(sw, P.sw2 + (size_t)(n0 + tile * 16 + lane) * Tr::INTER + kg * 32);
      v8f d = v8f(0.f);
      mm_rows_u<2>(d, sw, SB.hs_a, 2 * kg, lane);
      const float sc = P.sw2_s[(n0 / Tr::QB) * (Tr::INTER / Tr::QB) + (kg >> 2)];
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
    for (int sg2 = 0; sg2 < Tr::NSG; ++sg2) {
      a0 += L.s6r[((sg2 * 4 + t) * 8 + pb) * 16 + l];
      a1 += L.s6r[((sg2 * 4 + t) * 8 + pb) * 16 + l + 1];
    }
    a0 *= 256.f;
    a1 *= 256.f;
    if ((SPLIT && X.ar)) {  // mode 2: the caller all-reduces ar between S7A and S7B
      float* ar = X.ar + (size_t)pb * Tr::HID + n0 + pc;
      if (X.seg == SEG_S7A) {
        ar[0] = ar_send(a0);
        ar[1] = ar_send(a1);
        return;
      }
      a0 = ar[0];
      a1 = ar[1];
    } else {
      ar_two(S.peer, S.mbox, MB1 + (((size_t)pb * Tr::HID + n0 + pc) >> 1), a0, a1, mtag, S.cnt);
    }
    st16<DT>(P.hid_out, pb * Tr::HID + n0 + pc, a0);
    st16<DT>(P.hid_out, pb * Tr::HID + n0 + pc + 1, a1);
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
template <class Tr, typename DT>
inline void load_prt_b(const Ptrs& P, int wg, int s, int lane, sycl::uint4 (&prt_r)[4], sycl::uint4 (&prt_s)[4]) {
  // the router weight is read in the model's dtype (no fp16 copy); a bf16 one is
  // rounded to fp16 here, for the fp16 DPAS
  const e16* rr = P.router_w + (size_t)((s >> 1) * 16 + lane) * Tr::HID + wg * 64 + (s & 1) * 32;
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    prt_r[t] = *reinterpret_cast<const sycl::uint4*>(rr + t * 8);
    if constexpr (std::is_same_v<DT, bf16>) prt_r[t] = bf16x8_to_half(prt_r[t]);
  }
  ld_rows<4>(prt_s, P.sw13 + (size_t)(s * 16 + lane) * Tr::HID + wg * 64);
}

}  // namespace sgl_kernel::stage::xe20
