// SPDX-License-Identifier: Apache-2.0
//
// Single-row decode layer kernels (one token of one request), see
// mono_common.hpp for the shapes and layouts. Two layer kernels share one tail:
//   GDN  front (linear_attention): input norm, in_proj q|k|v|z (fp8) + b|a
//        (16-bit), causal conv1d, gated delta rule, gated RMSNorm -> o (2048,
//        SLM)
//   ATTN front (full_attention): input norm, qkv (q|gate per head, k, v), q/k
//        norm + RoPE, paged attention split along the context with DPAS, merge
//        with the new token, sigmoid gate, K / V write -> o (2048)
//   tail: out_proj / o_proj (64 rows per WG) -> all-reduce #1 -> residual ->
//        router / shared-expert partials of the WG's own 64 rows ->
//        fixed-order reduction -> top-8 -> routed gate|up (SGLang layout
//        [E][N][K]) + shared expert -> down of the WG's 64 output rows ->
//        all-reduce #2
//
// Grid: 32 WG x 512 work-items (32 SIMD16 sub-groups), 256 GRF -> one WG per Xe
// core, all resident. Hand-offs between stages are monotonic per-stage arrival
// counters (target = 32 * call). `call` is a device-side epoch, so launches can
// be recorded into a graph. Every launch on a scratch must be matched by one
// launch on the peer card.
#pragma once
#include "mono_common.hpp"

namespace sgl_mono {

// SLM layout GDN
constexpr int G_X = 0, G_XN = 4096, G_QK = 8192, G_RED = G_QK + 2048, G_ROUTE = G_RED + 512, G_AO = G_ROUTE + 128,
              G_MR = G_AO + 8192, G_HB = G_MR + 16384, G_BYTES = G_HB + (TOPK + 1) * INTER * 2;
// ATTN
constexpr int A_X = 0, A_XN = 8192, A_Q = A_XN + 4096, A_KV = A_Q + 8192, A_RED = A_KV + 2048, A_ROUTE = A_RED + 512,
              A_CS = A_ROUTE + 128, A_ST = A_CS + 256, A_QPK = A_ST + 2048, A_AO = A_QPK + 4096, A_MR = A_AO + 8192,
              A_HB = A_MR + 16384, A_BYTES = A_AO + 65536;
static_assert(A_HB + (TOPK + 1) * INTER * 2 <= A_BYTES, "tail SLM inside the attention partial area");

struct Slm {  // tail view
  half* xp;
  const uint32_t* xp32;
  half* xn;
  float* red;
  int* route_e;
  float* route_w;
  float* sgate;
  uint8_t* ao;
  float* mr;
  float* qk;
  half* hb;  // routed + shared h, (TOPK + 1) * INTER
};

// ----------------------------------------------------------------------------
// tail Entry: the mixer output o (2048 fp16) is in SLM xp (permuted), the
// out_proj rows ow[] of this SG are in registers (issued after the previous
// signal) with their scales osc, prt_r / prt_s the router / shared gate|up
// column blocks of this WG.
template <typename DT>
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
    const sycl::uint4 (&prt_s)[SHR / 4]) {
  const uint32_t mtag = uc_load32(S.cnt + MBT) + 1u;  // rows-1 mailbox region
  const int wg = it.get_group(0);
  const int lid = it.get_local_linear_id();
  sycl::sub_group sg = it.get_sub_group();
  const int s = sg.get_group_linear_id();
  const int lane = sg.get_local_linear_id();
  auto grp = it.get_group();
  {
    const int r0 = wg * 64 + s * 2;
    const float o0 = dot_row_regs<8>(sg, lane, ow[0], L.xp32, osc), o1 = dot_row_regs<8>(sg, lane, ow[1], L.xp32, osc);
    float ov = (lane == 0 ? o0 : o1) * 256.f;
    {  // all-reduce #1 over this WG's 64 rows, one contiguous block of 64
       // work-items. a + b == b + a
      float* ob = reinterpret_cast<float*>(L.ao + 1536);
      if (lane < 2) ob[s * 2 + lane] = ov;
      wg_bar(grp);
      if (lid < 64) {
        float v = ob[lid];
        mb_send(S.peer + wg * 64 + lid, v, mtag);
        v += mb_recv(S.mbox + wg * 64 + lid, mtag, S.cnt);
        ob[lid] = v;
      }
      wg_bar(grp);
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
        if ((lane & 7) == 0) S.part_rt[wg * NE + s * 8 + 2 * t + (lane >> 3)] = v;
      }
      // shared gate|up partials: SHR rows per SG (rows s*16.., one N block),
      // the WG's 64 columns (one K block): scaled here
      const uint32_t* xb32 = reinterpret_cast<const uint32_t*>(xbp);
      const float ssc = P.sw13_s[(s * SHR / QB) * KB_HID + (wg * 64) / QB];
#pragma unroll
      for (int t = 0; t < SHR / 4; ++t) {
        float v = dot16_fp8(prt_s[t], xb32 + (lane & 3) * 8);
        v += sycl::permute_group_by_xor(sg, v, 2);
        v += sycl::permute_group_by_xor(sg, v, 1);
        if ((lane & 3) == 0) S.part_sh[wg * 2 * INTER + s * SHR + 4 * t + (lane >> 2)] = v * ssc;
      }
      if (s == NSG - 1) {  // shared-expert gate partial
        float v = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i)
          v += ld16<DT>(P.sgate_w, wg * 64 + lane * 4 + i) * (float)xbn[lane * 4 + i];
        v = sycl::reduce_over_group(sg, v, sycl::plus<float>());
        if (lane == 0) S.part_sg[wg * 2] = v;
      }
    }
  }
  signal(it, S.cnt + c_o * 16);

  // ---------------- S5': reduce the partials (fixed order): rs, x2, logits,
  // shared slice, gate -----
  wait_for(it, S.cnt + c_o * 16, NWG * call);
  float* lgs = reinterpret_cast<float*>(L.ao + 512);  // 256 logits
  {
    float ss = 0.f;
#pragma unroll
    for (int w2i = 0; w2i < NWG; ++w2i)
      ss += S.part_sg[w2i * 2 + 1];
    const float rs = sycl::rsqrt(ss / HID + EPS);
    half x2[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
      x2[i] = toh<DT>(ld16<DT>(P.res_out, lid * 4 + i) * rs * (1.f + ld16<DT>(P.post_norm_w, lid * 4 + i)));
    // x2, permuted (the routed gate|up rows; xp's mixer output is consumed)
    *reinterpret_cast<sycl::vec<half, 4>*>(L.xp + lid * 4) = sycl::vec<half, 4>(x2[0], x2[2], x2[1], x2[3]);
    if (lid < NE) {
      float a = 0.f;
#pragma unroll
      for (int w2i = 0; w2i < NWG; ++w2i)
        a += S.part_rt[w2i * NE + lid];
      lgs[lid] = a * rs;
    } else if (lid < NE + SPW) {
      const int i = wg * SPW + lid - NE;
      float g = 0.f, u = 0.f;
#pragma unroll
      for (int w2i = 0; w2i < NWG; ++w2i) {
        g += S.part_sh[w2i * 2 * INTER + i];
        u += S.part_sh[w2i * 2 * INTER + INTER + i];
      }
      const float sc = 256.f * rs;
      g *= sc;
      u *= sc;
      S.hbuf[TOPK * INTER + i] = toh<DT>(g / (1.f + sycl::exp(-g)) * u);
    } else if (lid == NE + 32) {
      float a = 0.f;
#pragma unroll
      for (int w2i = 0; w2i < NWG; ++w2i)
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
    float lmax = -INFINITY, sel_v[TOPK];
    int seli[TOPK];
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
    for (int k = 0; k < TOPK; ++k) {
      e[k] = sycl::exp(sel_v[k] - lmax);
      sum += e[k];
    }
    if (lane == 0)
      for (int k = 0; k < TOPK; ++k) {
        L.route_e[k] = seli[k] & (NE - 1);
        L.route_w[k] = e[k] / sum;
      }
  }
  wg_bar(grp);

  // ---------------- S6: routed gate|up, layout w13[E][N = 512][K = 2048]
  // WG = (slot j = wg/4, quarter q4): gate rows q4*64..+64 (SGs 0..15) and the
  // same up rows (+256, SGs 16..31), 4 rows per SG, each row 8 chunks of 16 B a
  // lane (one N block per SG).
  {
    const int j = wg >> 2, q4 = wg & 3, e = L.route_e[j];
    const int r0 = (s >> 4) * INTER + q4 * 64 + (s & 15) * 4;
    sycl::uint4 gw[4][8];
    float gsc[8];
#pragma unroll
    for (int b = 0; b < 4; ++b)
      load_row<8>(gw[b], P.w13 + ((size_t)e * 2 * INTER + r0 + b) * HID, lane);
    load_scales<8>(gsc, P.w13_s + ((size_t)e * (2 * INTER / QB) + r0 / QB) * KB_HID, lane);
    float v[4];
#pragma unroll
    for (int b = 0; b < 4; ++b)
      v[b] = dot_row_regs<8>(sg, lane, gw[b], L.xp32, gsc);
    if (lane < 4)
      L.qk[(s >> 4) * 64 + (s & 15) * 4 + lane] = (lane == 0 ? v[0] : lane == 1 ? v[1] : lane == 2 ? v[2] : v[3]) * 256.f;
    wg_bar(grp);
    if (lid < 64) {
      const float g = L.qk[lid], u = L.qk[64 + lid];
      S.hbuf[j * INTER + q4 * 64 + lid] = toh<DT>(g / (1.f + sycl::exp(-g)) * u);
    }
  }
  signal(it, S.cnt + C_GU * 16);
  // the down rows of this WG's 64 outputs (routed slot s/4, rows (s&3)*16 +
  // lane, a whole 256-B row a lane) and the shared down rows: in flight during
  // the hand-off
  sycl::uint4 wd[16], wsd[2];
  float dsc[2];
  {
    const int jd = s >> 2, ed = L.route_e[jd];
    const int n = wg * 64 + (s & 3) * 16 + lane;
    const uint8_t* rowd = P.w2 + ((size_t)ed * HID + n) * INTER;
#pragma unroll
    for (int c = 0; c < 16; ++c)
      wd[c] = *reinterpret_cast<const sycl::uint4*>(rowd + c * 16);
    const float* sd = P.w2_s + ((size_t)ed * KB_HID + (wg * 64) / QB) * (INTER / QB);
    dsc[0] = sd[0];
    dsc[1] = sd[1];
#pragma unroll
    for (int rr = 0; rr < 2; ++rr)
      wsd[rr] = *reinterpret_cast<const sycl::uint4*>(P.sw2 + (size_t)(wg * 64 + 2 * s + rr) * INTER + lane * 16);
  }

  // ---------------- S7: down for this WG's 64 rows ; all-reduce #2
  // -----------------------------
  wait_for(it, S.cnt + C_GU * 16, NWG * call);
  if (wg == 0 && lid == 0) {  // every WG has read the epoch and the tag
    cnt_ref(S.cnt[EPOCH]).store(call);
    cnt_ref(S.cnt[MBT]).store(mtag);
  }
  // routed + shared h -> SLM (every lane reads all of its slot's h)
  for (int i = lid; i < (TOPK + 1) * INTER / 8; i += WGS)
    *reinterpret_cast<sycl::uint4*>(L.hb + i * 8) = *reinterpret_cast<const sycl::uint4*>(S.hbuf + i * 8);
  wg_bar(grp);
  {
    const int jd = s >> 2;
    const float coef = L.route_w[jd] * 256.f;
    const half* hj = L.hb + jd * INTER;
    float a0 = 0.f, a1 = 0.f;
#pragma unroll
    for (int c = 0; c < 8; ++c)
      a0 += dot16_fp8_nat(wd[c], *reinterpret_cast<const sycl::vec<half, 16>*>(hj + c * 16));
#pragma unroll
    for (int c = 8; c < 16; ++c)
      a1 += dot16_fp8_nat(wd[c], *reinterpret_cast<const sycl::vec<half, 16>*>(hj + c * 16));
    L.mr[jd * 64 + (s & 3) * 16 + lane] = (a0 * dsc[0] + a1 * dsc[1]) * coef;
    // shared down: rows wg*64 + 2s + rr, this lane's 16 K (K block lane / 8)
    const sycl::vec<half, 16> hx = *reinterpret_cast<const sycl::vec<half, 16>*>(L.hb + TOPK * INTER + lane * 16);
    const float ssc = P.sw2_s[((wg * 64) / QB) * (INTER / QB) + (lane >> 3)];
    float d0 = dot16_fp8_nat(wsd[0], hx) * ssc, d1 = dot16_fp8_nat(wsd[1], hx) * ssc;
    d0 = sycl::reduce_over_group(sg, d0, sycl::plus<float>());
    d1 = sycl::reduce_over_group(sg, d1, sycl::plus<float>());
    const float cs = L.sgate[0] * 256.f;
    if (lane < 2) L.red[2 * s + lane] = (lane == 0 ? d0 : d1) * cs;
    wg_bar(grp);
    if (lid < 64) {
      float a = 0.f;
#pragma unroll
      for (int k = 0; k < TOPK; ++k)
        a += L.mr[k * 64 + lid];
      a += L.red[lid];
      mb_send(S.peer + HID + wg * 64 + lid, a, mtag);  // all-reduce #2
      a += mb_recv(S.mbox + HID + wg * 64 + lid, mtag, S.cnt);
      st16<DT>(P.hid_out, wg * 64 + lid, a);
    }
  }
}

// PRT column blocks of this WG (router 16-bit, shared gate|up fp8) -> registers
inline void
load_prt(const Ptrs& P, int wg, int s, int lane, sycl::uint4 (&prt_r)[4], sycl::uint4 (&prt_s)[SHR / 4]) {
#pragma unroll
  for (int t = 0; t < 4; ++t)  // experts s*8 + 2t + (lane>>3), halves (lane&7)*8..+8
    prt_r[t] = *reinterpret_cast<const sycl::uint4*>(
        P.router_w + (size_t)(s * 8 + 2 * t + (lane >> 3)) * HID + wg * 64 + (lane & 7) * 8);
#pragma unroll
  for (int t = 0; t < SHR / 4; ++t)  // rows s*SHR + 4t + (lane>>2), bytes (lane&3)*16..+16
    prt_s[t] = *reinterpret_cast<const sycl::uint4*>(
        P.sw13 + (size_t)(s * SHR + 4 * t + (lane >> 2)) * HID + wg * 64 + (lane & 3) * 16);
}

// res = h + r ; x = GemmaRMS(res) -> xn (natural) and xp (permuted)
template <typename DT>
inline void input_norm(sycl::nd_item<1> it, const Ptrs& P, half* xn, half* xp) {
  const int lid = it.get_local_linear_id();
  float v[4], ss = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int k = lid * 4 + i;
    v[i] = rnd<DT>(ld16<DT>(P.hs_in, k) + ld16<DT>(P.res_in, k));
    ss += v[i] * v[i];
  }
  ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
  const float rs = sycl::rsqrt(ss / HID + EPS);
  half x[4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
    x[i] = toh<DT>(v[i] * rs * (1.f + ld16<DT>(P.in_norm_w, lid * 4 + i)));
  if (xn) *reinterpret_cast<sycl::vec<half, 4>*>(xn + lid * 4) = sycl::vec<half, 4>(x[0], x[1], x[2], x[3]);
  *reinterpret_cast<sycl::vec<half, 4>*>(xp + lid * 4) = sycl::vec<half, 4>(x[0], x[2], x[1], x[3]);
}

inline Slm slm_tail(uint8_t* slm, int x, int xn, int qk, int red, int route, int ao, int mr, int hb) {
  Slm L;
  L.xp = reinterpret_cast<half*>(slm + x);
  L.xp32 = reinterpret_cast<const uint32_t*>(slm + x);
  L.xn = reinterpret_cast<half*>(slm + xn);
  L.qk = reinterpret_cast<float*>(slm + qk);
  L.red = reinterpret_cast<float*>(slm + red);
  L.route_e = reinterpret_cast<int*>(slm + route);
  L.route_w = reinterpret_cast<float*>(slm + route + 32);
  L.sgate = reinterpret_cast<float*>(slm + route + 64);
  L.ao = slm + ao;
  L.mr = reinterpret_cast<float*>(slm + mr);
  L.hb = reinterpret_cast<half*>(slm + hb);
  return L;
}

// ----------------------------------------------------------------------------
// GDN layer
template <typename DT>
inline void gdn_layer(sycl::nd_item<1> it, uint8_t* slm, const Ptrs& P, const Scratch& S) {
  const int wg = it.get_group(0);
  const int lid = it.get_local_linear_id();
  sycl::sub_group sg = it.get_sub_group();
  const int s = sg.get_group_linear_id();
  const int lane = sg.get_local_linear_id();
  auto grp = it.get_group();
  // epoch: plain L1-uncached load (an atomic load is an atomic_or RMW: the 1024
  // threads would serialize)
  const uint32_t call = uc_load32(S.cnt + EPOCH) + 1u;

  const Slm L = slm_tail(slm, G_X, G_XN, G_QK, G_RED, G_ROUTE, G_AO, G_MR, G_HB);
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
    st[rr] = *reinterpret_cast<const sycl::vec<float, 8>*>(ssm + ((size_t)h * DV + c * 64 + 2 * s + rr) * DK + lane * 8);
  const int cch = lid < 128   ? kh * DK + lid
                  : lid < 256 ? KD + kh * DK + (lid - 128)
                              : 2 * KD + h * DV + c * 64 + (lid - 256);
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
  input_norm<DT>(it, P, L.xn, L.xp);
  wg_bar(grp);

  // ---------------- S1: q|k|v rows of in_proj_qkvz (4 per SG) ----------------
  {
    const int r0 = (wg * NSG + s) * 4;
    sycl::uint4 w[4][8];
    float sc[8];
#pragma unroll
    for (int b = 0; b < 4; ++b)
      load_row<8>(w[b], P.qkvz_w + (size_t)(r0 + b) * HID, lane);
    load_scales<8>(sc, P.qkvz_s + (size_t)(r0 / QB) * KB_HID, lane);
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const float v = dot_row_regs<8>(sg, lane, w[b], L.xp32, sc);
      if (lane == 0) S.qkv[r0 + b] = toh<DT>(v * 256.f);
    }
  }
  signal(it, S.cnt + C_A * 16);
  sycl::uint4 zw[2][8];  // z rows of this WG: channel h*128 + c*64 + 2s + rr
                         // (consumed after the recurrence)
  float zsc[8];
  {
    const int zr = CONVD + h * DV + c * 64 + 2 * s;
#pragma unroll
    for (int rr = 0; rr < 2; ++rr)
      load_row<8>(zw[rr], P.qkvz_w + (size_t)(zr + rr) * HID, lane);
    load_scales<8>(zsc, P.qkvz_s + (size_t)(zr / QB) * KB_HID, lane);
  }
  if (s < 2) {  // b, a of v head h (16-bit rows, 8 halves a lane a chunk; DT, as SGLang's GEMM output)
    sycl::uint4 baw[16];
    const e16* row = P.ba_w + (size_t)(s * NVH + h) * HID;
#pragma unroll
    for (int j = 0; j < 16; ++j)
      baw[j] = *reinterpret_cast<const sycl::uint4*>(row + j * 128 + lane * 8);
    float v = 0.f;
#pragma unroll
    for (int j = 0; j < 16; ++j)
      v += dot8_w16<DT>(baw[j], L.xn + j * 128 + lane * 8);
    v = sycl::reduce_over_group(sg, v, sycl::plus<float>());
    if (lane == 0) red[s] = rnd<DT>(v);
  }

  // ---------------- S2: conv1d + SiLU, L2 norm, gated delta rule -------------
  wait_for(it, S.cnt + C_A * 16, NWG * call);
  if (lid < 320) {
    const float xnew = (float)S.qkv[cch];
    float acc = ld16<DT>(cst, 0) * cwt[0];
    acc += ld16<DT>(cst, 1) * cwt[1];
    acc += ld16<DT>(cst, 2) * cwt[2];
    acc += xnew * cwt[3];
    qk[lid] = rnd<DT>(acc / (1.f + sycl::exp(-acc)));
    // conv state <- (s1, s2, x): q/k channels by the (even h, c = 0) WG of the
    // k head, v channels by the owner
    if (lid >= 256 || ((h & 1) == 0 && c == 0)) {
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
    if (lane < 2) S.core[h * DV + c * 64 + 2 * s + lane] = toh<DT>(lane == 0 ? o[0] : o[1]);
  }
  {  // z rows (registers since S1)
    const float z0 = dot_row_regs<8>(sg, lane, zw[0], L.xp32, zsc), z1 = dot_row_regs<8>(sg, lane, zw[1], L.xp32, zsc);
    if (lane < 2) S.zbuf[h * DV + c * 64 + 2 * s + lane] = toh<DT>((lane == 0 ? z0 : z1) * 256.f);
  }
  signal(it, S.cnt + C_B * 16);
#pragma unroll
  for (int rr = 0; rr < 2; ++rr)  // state write-back after the hand-off
    st_store8(ssm + ((size_t)h * DV + c * 64 + 2 * s + rr) * DK + lane * 8, st[rr]);
  sycl::uint4 ow[2][8], prt_r[4], prt_s[SHR / 4];
  float osc[8];
  load_row<8>(ow[0], P.out_w + (size_t)(wg * 64 + s * 2) * VD, lane);
  load_row<8>(ow[1], P.out_w + (size_t)(wg * 64 + s * 2 + 1) * VD, lane);
  load_scales<8>(osc, P.out_s + (size_t)((wg * 64) / QB) * (VD / QB), lane);
  load_prt(P, wg, s, lane, prt_r, prt_s);

  // ---------------- S3: gated RMSNorm (all heads) -> o in SLM ; tail
  // -------------------------
  wait_for(it, S.cnt + C_B * 16, NWG * call);
  {
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
    const float rs = sycl::rsqrt((red[64 + 2 * hh] + red[64 + 2 * hh + 1]) / DV + EPS);
    half y[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float zf = (float)z4[i];
      y[i] = toh<DT>((float)o4[i] * rs * ld16<DT>(P.gnorm_w, (lid * 4 + i) & (DV - 1)) * (zf / (1.f + sycl::exp(-zf))));
    }
    *reinterpret_cast<sycl::vec<half, 4>*>(L.xp + lid * 4) = sycl::vec<half, 4>(y[0], y[2], y[1], y[3]);
    wg_bar(grp);
  }
  tail<DT>(it, P, S, L, call, C_O, ow, osc, prt_r, prt_s);
}

// ----------------------------------------------------------------------------
// full-attention layer
// one flash step over NT 16-token tiles from token tb: trow(t) gives the slot
// of token t (past the context: a valid slot, masked by the score select)
#define MONO_ATTN_STEP(tb, NT)                                                                         \
  do {                                                                                                 \
    const e16* kb[NT];                                                                                 \
    _Pragma("unroll") for (int tt = 0; tt < NT; ++tt) kb[tt] = P.kc + trow((tb) + tt * 16) * HD;       \
    v8f S4[NT];                                                                                        \
    _Pragma("unroll") for (int tt = 0; tt < NT; ++tt) {                                                \
      S4[tt] = v8f(0.f);                                                                               \
      _Pragma("unroll") for (int kt = 0; kt < 16; ++kt) {                                              \
        const v8u bk = load_k_tile(kb[tt], kt * 8);                                                    \
        const v4u qa = *reinterpret_cast<const v4u*>(qpk + (kt * 16 + lane) * 4);                      \
        dpas16<DT>(S4[tt], qa, bk);                                                                    \
      }                                                                                                \
    }                                                                                                  \
    float sv[NT][8];                                                                                   \
    _Pragma("unroll") for (int tt = 0; tt < NT; ++tt) {                                                \
      const bool ok = (tb) + tt * 16 + lane < lc; /* token = column = lane */                          \
      _Pragma("unroll") for (int i = 0; i < 8; ++i) sv[tt][i] = ok ? S4[tt][i] * SM_SCALE : -INFINITY; \
    }                                                                                                  \
    float alpha[8];                                                                                    \
    _Pragma("unroll") for (int i = 0; i < 8; ++i) {                                                    \
      float lm = sv[0][i];                                                                             \
      _Pragma("unroll") for (int tt = 1; tt < NT; ++tt) lm = sycl::fmax(lm, sv[tt][i]);                \
      const float mn = sycl::fmax(m[i], sycl::reduce_over_group(sg, lm, sycl::maximum<float>()));      \
      alpha[i] = mn == -INFINITY ? 1.f : sycl::native::exp2(m[i] - mn);                                \
      m[i] = mn;                                                                                       \
      lp[i] *= alpha[i];                                                                               \
    }                                                                                                  \
    _Pragma("unroll") for (int nt = 0; nt < 8; ++nt) _Pragma("unroll") for (int i = 0; i < 8; ++i)     \
        O[nt][i] *= alpha[i];                                                                          \
    _Pragma("unroll") for (int tt = 0; tt < NT; ++tt) {                                                \
      uint16_t pv[8];                                                                                  \
      _Pragma("unroll") for (int i = 0; i < 8; ++i) {                                                  \
        const float p = m[i] == -INFINITY ? 0.f : sycl::native::exp2(sv[tt][i] - m[i]);                \
        lp[i] += p;                                                                                    \
        pv[i] = bits<DT>(p);                                                                           \
      }                                                                                                \
      /* P (lane = token, element = head) -> dpas A: lane l, element j = P[2j                          \
       * + l/8][2(l%8) + {0,1}] */                                                                     \
      v4u pa;                                                                                          \
      _Pragma("unroll") for (int j = 0; j < 4; ++j) {                                                  \
        const int t0_ = 2 * (lane & 7);                                                                \
        const uint16_t a0 = sycl::select_from_group(sg, pv[2 * j], t0_);                               \
        const uint16_t a1 = sycl::select_from_group(sg, pv[2 * j + 1], t0_);                           \
        const uint16_t b0 = sycl::select_from_group(sg, pv[2 * j], t0_ + 1);                           \
        const uint16_t b1 = sycl::select_from_group(sg, pv[2 * j + 1], t0_ + 1);                       \
        pa[j] = (lane < 8) ? pack16(a0, b0) : pack16(a1, b1);                                          \
      }                                                                                                \
      const e16* vb = P.vc + (kb[tt] - P.kc) + dh * 128;                                               \
      _Pragma("unroll") for (int nt = 0; nt < 8; ++nt) dpas16<DT>(O[nt], pa, load_v_tile(vb, nt * 16)); \
    }                                                                                                  \
  } while (0)

// zero the V rows r0 .. 15 of the 16-token tile at slot `tile` (the stale rows
// after the context in the request's own page: a reused page may hold NaN, and
// 0 * NaN poisons P.V; later tokens of the request overwrite them anyway)
inline void zero_v_tail(const Ptrs& P, int64_t tile, int r0, int dh, int lane) {
  e16* vz = P.vc + tile * HD + dh * 128;
  for (int r = r0; r < 16; ++r)
    *reinterpret_cast<sycl::vec<half, 8>*>(vz + (size_t)r * HD + lane * 8) = sycl::vec<half, 8>(0.f);
  sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::work_group);
}

template <typename DT>
inline void attn_layer(sycl::nd_item<1> it, uint8_t* slm, const Ptrs& P, const Scratch& S) {
  const int wg = it.get_group(0);
  const int lid = it.get_local_linear_id();
  sycl::sub_group sg = it.get_sub_group();
  const int s = sg.get_group_linear_id();
  const int lane = sg.get_local_linear_id();
  auto grp = it.get_group();
  const uint32_t call = uc_load32(S.cnt + EPOCH) + 1u;

  const Slm L = slm_tail(slm, A_X, A_XN, A_KV, A_RED, A_ROUTE, A_AO, A_MR, A_HB);
  half* xp = L.xp;
  const uint32_t* xp32 = L.xp32;
  e16* qs = reinterpret_cast<e16*>(slm + A_Q);
  e16* kvn = reinterpret_cast<e16*>(slm + A_KV);
  float* red = L.red;
  float* st = reinterpret_cast<float*>(slm + A_ST);
  half* ao = reinterpret_cast<half*>(slm + A_AO);
  float* rope = reinterpret_cast<float*>(slm + A_CS);
  auto qrow = [](int qi) { return (qi >> 8) * 512 + (qi & 255); };
  auto grow = [](int gi) { return gi < NQ * HD ? (gi >> 8) * 512 + 256 + (gi & 255) : NQ * 512 + (gi - NQ * HD); };

  const int seqlen = P.seq_lens[0];
  const int lc = seqlen - 1;  // cached tokens (the new one is merged separately)
  const int* btab = P.block_table;
  // SG pair p = wg*16 + s/2 owns the cached tokens [p*T, p*T + T) of every
  // pass of APASS1 = 40960 tokens: 5 tiles of 16 a pass. The first pass is one
  // straight-line body (a loop around it is much slower; the 5th tile only
  // runs past 32K tokens); longer contexts take further passes in a loop.
  constexpr int T = 80;
  constexpr int APASS1 = NWG * 16 * T;
  const int t0 = (wg * 16 + (s >> 1)) * T;
  auto trow = [&](int t) {  // slot of token t; past the context: token 0
    if (t >= lc) t = 0;     // (masked by the score select)
    return kv_slot(btab, P.ps, t);
  };

  input_norm<DT>(it, P, nullptr, xp);
  wg_bar(grp);

  // ---------------- S1: q rows (2 per SG, one head, one N block) -------------
  {
    float* qr = S.qkv_raw;
    const int q0 = (wg * NSG + s) * 2;
    sycl::uint4 w[2][8];
    float sc[8];
#pragma unroll
    for (int b = 0; b < 2; ++b)
      load_row<8>(w[b], P.qkv_w + (size_t)qrow(q0 + b) * HID, lane);
    load_scales<8>(sc, P.qkv_s + (size_t)(qrow(q0) / QB) * KB_HID, lane);
#pragma unroll
    for (int b = 0; b < 2; ++b) {
      const float v = dot_row_regs<8>(sg, lane, w[b], xp32, sc);
      if (lane == 0) qr[qrow(q0 + b)] = v * 256.f;
    }
  }
  signal(it, S.cnt + C_A * 16);
  {  // the q hand-off + q norm leave DRAM idle: warm L2 with the first K tile
     // of this SG pair
    if (t0 < lc) {
      const e16* kb = P.kc + trow(t0) * HD;
#pragma unroll
      for (int u = lane; u < 8 * 8; u += 16)
        syclex::prefetch(
            (void*)(kb + (size_t)((s & 1) * 8 + u / 8) * HD + (u % 8) * 32),
            syclex::properties{syclex::prefetch_hint_L2});
    }
  }
  if (lid < 32) rope_cs((float)P.pos[0], lid, rope);

  // ---------------- S2: q norm + RoPE, attention split along the context -----
  wait_for(it, S.cnt + C_A * 16, NWG * call);
  if (s < NQ) head_norm_rope<DT>(sg, lane, S.qkv_raw + s * 512, P.q_norm_w, rope, qs + s * HD, nullptr);
  wg_bar(grp);
  // q in dpas A order: qpk[(kt*16 + l)*4 + j] = (q[2j + l/8][kt*16 + 2(l%8)],
  // .. + 1)
  uint32_t* qpk = reinterpret_cast<uint32_t*>(slm + A_QPK);
  for (int u = lid; u < 16 * 16 * 4; u += WGS) {
    const int j = u & 3, l = (u >> 2) & 15, kt = u >> 6;
    const e16* src = qs + (2 * j + (l >> 3)) * HD + kt * 16 + 2 * (l & 7);
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
      if ((lc & 15) && (lc & ~15) >= t0 && (lc & ~15) < t0 + T) zero_v_tail(P, trow(lc & ~15), lc & 15, dh, lane);
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
        if ((lc & 15) && (lc & ~15) == tb) zero_v_tail(P, trow(tb), lc & 15, dh, lane);
        MONO_ATTN_STEP(tb, 1);
      }
    }
    sycl::uint4 gmid[GKV_MID][8];  // gate/k/v rows of this SG, consumed after
                                   // the in-WG combine
    float gsc[8];
#pragma unroll
    for (int r = 0; r < GKV_MID; ++r)
      load_row<8>(gmid[r], P.qkv_w + (size_t)grow((wg * NSG + s) * GKV_MID + r) * HID, lane);
    load_scales<8>(gsc, P.qkv_s + (size_t)(grow((wg * NSG + s) * GKV_MID) / QB) * KB_HID, lane);
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
    if (s < 2 * NQ) {  // in-WG combine: SG -> (head hh, dims half dh2); sources
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
      float* dst = S.part_o + ((size_t)wg * NQ + hh) * HD + dh2 * 128 + lane * 8;
#pragma unroll
      for (int i = 0; i < 8; ++i)
        dst[i] = acc[i] * iL;
      if (dh2 == 0 && lane == 0) {
        S.part_ml[(wg * NQ + hh) * 2] = M;
        S.part_ml[(wg * NQ + hh) * 2 + 1] = Lw;
      }
    }
#pragma unroll
    for (int r = 0; r < GKV_MID; ++r) {
      const float v = dot_row_regs<8>(sg, lane, gmid[r], xp32, gsc);
      if (lane == 0) S.qkv_raw[grow((wg * NSG + s) * GKV_MID + r)] = v * 256.f;
    }
  }
  // remaining gate / k / v rows taken dynamically by sub-groups that finish
  // early (results land at fixed rows). The ticket counter is monotonic: every
  // call consumes exactly GKV_CHUNKS + NSG*NWG tickets.
  {
    float* qr = S.qkv_raw;
    const uint32_t base = (call - 1) * ((uint32_t)GKV_CHUNKS + NSG * NWG);
    while (true) {
      uint32_t t = 0;
      if (lane == 0) t = cnt_ref(S.cnt[C_TKT * 16]).fetch_add(1u, sycl::memory_order::relaxed);
      const int chunk = (int)(sycl::group_broadcast(sg, t, 0) - base);
      if (chunk >= GKV_CHUNKS) break;
      const int g0 = GKV_MID * NSG * NWG + chunk * GKV_ROWS;
      sycl::uint4 w[GKV_ROWS][8];
      float sc[8];
#pragma unroll
      for (int r = 0; r < GKV_ROWS; ++r)
        load_row<8>(w[r], P.qkv_w + (size_t)grow(g0 + r) * HID, lane);
      load_scales<8>(sc, P.qkv_s + (size_t)(grow(g0) / QB) * KB_HID, lane);
#pragma unroll
      for (int r = 0; r < GKV_ROWS; ++r) {
        const float v = dot_row_regs<8>(sg, lane, w[r], xp32, sc);
        if (lane == 0) qr[grow(g0 + r)] = v * 256.f;
      }
    }
  }
  signal(it, S.cnt + C_B * 16);
  sycl::uint4 ow[2][8];  // o_proj rows of this SG -> registers, consumed after
                         // the merge
  float osc[8];
  load_row<8>(ow[0], P.out_w + (size_t)(wg * 64 + s * 2) * (NQ * HD), lane);
  load_row<8>(ow[1], P.out_w + (size_t)(wg * 64 + s * 2 + 1) * (NQ * HD), lane);
  load_scales<8>(osc, P.out_s + (size_t)((wg * 64) / QB) * (NQ * HD / QB), lane);

  // ---------------- S3: merge partials + new token, gate ; WG = (head, 64
  // dims) -----------------
  wait_for(it, S.cnt + C_B * 16, NWG * call);
  {
    const int hh = (wg * 64) / HD, d0 = (wg * 64) % HD;
    float* mrg = reinterpret_cast<float*>(slm + A_AO);  // [32][64]
    {  // new token: k (norm + rope) and v; WG 0 writes them to the cache
      const int64_t slot = P.slot[0];
      const bool wr = wg == 0 && slot >= 0;
      if (s == 0)
        head_norm_rope<DT>(sg, lane, S.qkv_raw + NQ * 512, P.k_norm_w, rope, kvn, wr ? P.kc + slot * HD : nullptr);
      else if (s == 1)
        head_norm_rope<DT>(sg, lane, S.qkv_raw + NQ * 512 + HD, nullptr, rope, kvn + HD, wr ? P.vc + slot * HD : nullptr);
      wg_bar(grp);
    }
    if (s == 0) {  // new-token score
      float d = 0.f;
#pragma unroll
      for (int i = 0; i < 16; ++i)
        d += ld16<DT>(qs, hh * HD + lane * 16 + i) * ld16<DT>(kvn, lane * 16 + i);
      d = sycl::reduce_over_group(sg, d, sycl::plus<float>());
      if (lane == 0) red[96] = d * SM_SCALE;
    }
    if (lane == 0) red[s] = S.part_ml[(s * NQ + hh) * 2];
    wg_bar(grp);
    float M = red[96];
    for (int i = 0; i < 32; ++i)
      M = sycl::fmax(M, red[i]);
    const float wgt = S.part_ml[(s * NQ + hh) * 2 + 1] * sycl::native::exp2(red[s] - M);
    const float* src = S.part_o + ((size_t)s * NQ + hh) * HD + d0;
#pragma unroll
    for (int i = 0; i < 4; ++i)
      mrg[s * 64 + lane * 4 + i] = wgt * src[lane * 4 + i];
    if (lane == 0) red[32 + s] = wgt;
    wg_bar(grp);
    if (lid < 64) {
      const int d = d0 + lid;
      float Lw = sycl::native::exp2(red[96] - M), acc = Lw * ld16<DT>(kvn, HD + d);
      for (int i = 0; i < 32; ++i) {
        acc += mrg[i * 64 + lid];
        Lw += red[32 + i];
      }
      const float o = acc / Lw;
      const float gate = S.qkv_raw[hh * 512 + 256 + d];
      S.attn_out[hh * HD + d] = toh<DT>(o / (1.f + sycl::exp(-gate)));
    }
  }
  signal(it, S.cnt + C_C * 16);
  sycl::uint4 prt_r[4], prt_s[SHR / 4];
  load_prt(P, wg, s, lane, prt_r, prt_s);

  // ---------------- S4: o (all heads) -> SLM ; tail
  // -------------------------------------------
  wait_for(it, S.cnt + C_C * 16, NWG * call);
  for (int k = lid; k < NQ * HD; k += WGS)
    store_perm(xp, k, S.attn_out[k]);
  wg_bar(grp);
  tail<DT>(it, P, S, L, call, C_O, ow, osc, prt_r, prt_s);
}

// ----------------------------------------------------------------------------
// launch One template instance per layer kind: each translation unit
// instantiates only the kernel it launches (an inline non-template launcher
// with per-file bodies would be an ODR violation).
template <bool ATTN, typename DT>
class MonoLayerK;  // kernel names

template <bool ATTN, typename DT>
void launch_t(sycl::queue& q, const Ptrs& P, const Scratch& S) {
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint8_t, 1> slm(sycl::range<1>(ATTN ? A_BYTES : G_BYTES), h);
    // 256 GRF: the 32 sub-groups of a work-group must all be resident on one
    // Xe core (8 XVEs x 4 threads), and the register-held weights need it
    syclex::properties props{syclex::sub_group_size<16>, intelex::grf_size<256>};
    h.parallel_for<MonoLayerK<ATTN, DT>>(
        sycl::nd_range<1>(NWG * WGS, WGS), props, [=](sycl::nd_item<1> it) {
      if constexpr (ATTN)
        attn_layer<DT>(it, slm.get_multi_ptr<sycl::access::decorated::no>().get(), P, S);
      else
        gdn_layer<DT>(it, slm.get_multi_ptr<sycl::access::decorated::no>().get(), P, S);
    });
  });
}

}  // namespace sgl_mono
