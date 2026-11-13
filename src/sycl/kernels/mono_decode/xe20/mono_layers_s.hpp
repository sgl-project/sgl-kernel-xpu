// SPDX-License-Identifier: Apache-2.0
//
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
#pragma once
#include "mono_layers_b.hpp"

namespace sgl_mono {

template <int NB, int T, typename DT>
inline void gdn_layer_s(sycl::nd_item<1> it, uint8_t* slm, const Ptrs& P, const Scratch& S, const ScratchB& SB) {
  constexpr int NRQ = NB / T;
  static_assert(NRQ * T == NB && T >= 2, "spec verify: NB = requests x tokens");
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
    s1c[j] = P.qkvz_s[(size_t)((wg * 128 + tile1 * 16) / QB) * KB_HID + ks1 * 4 + j];
#pragma unroll
  for (int j = 0; j < 2; ++j)
    szc[j] = P.qkvz_s[(size_t)((CONVD + h * DV + c * 64 + tz * 16) / QB) * KB_HID + kz * 2 + j];
  sycl::uint4 w1[32];
  ld_rows<32>(w1, P.qkvz_w + (size_t)(wg * 128 + tile1 * 16 + lane) * HID + ks1 * 512);
#pragma unroll
  for (int b = 0; b < NRQ; ++b)
    syclex::prefetch(
        (void*)(ssm_of(b) + ((size_t)h * DV + c * 64) * DK + lid * 16), syclex::properties{syclex::prefetch_hint_L2});
  const int cch = lid < 128   ? kh * DK + lid
                  : lid < 256 ? KD + kh * DK + (lid - 128)
                              : 2 * KD + h * DV + c * 64 + (lid - 256);
  float cwt[4];
  e16* csl = reinterpret_cast<e16*>(slm + B_CS);  // [b][i][320]: tap i of request b, the WG's channels
  load_conv_b<NRQ>(csl, conv_of, chs, h, c, kh, lid);
  if (lid < 320) {
    const sycl::vec<uint16_t, 4> w4 = *reinterpret_cast<const sycl::vec<uint16_t, 4>*>(P.conv_w + (size_t)cch * 4);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      cwt[i] = f32<DT>(w4[i]);
  }
  input_norm_b<NB, DT>(it, P, L);
  wg_bar(grp);

  // --- S1: q|k|v rows (128 per WG), z, b|a: as the batched kernel ------------
  sycl::uint4 zw[16], bw[16];
  {
    v8f cc = v8f(0.f);
    mm_rows_r<32>(cc, w1, L.xa, ks1 * 32, lane, s1c);
    ld_rows<16>(zw, P.qkvz_w + (size_t)(CONVD + h * DV + c * 64 + tz * 16 + lane) * HID + kz * 256);
    ld_rows<16>(bw, reinterpret_cast<const uint8_t*>(P.ba_h + (size_t)((s >> 4) * 16 + lane) * HID + (s & 15) * 128));
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
    for (int idx = lid; idx < NB * 64; idx += WGS) {
      const int b = idx >> 6, r = idx & 63, t = r >> 4, l = r & 15;
      float v = 0.f;
#pragma unroll
      for (int k2 = 0; k2 < 8; ++k2)
        v += red_get(L.red, t * 8 + k2, b, l);
      SB.z[b * VD + h * DV + c * 64 + r] = toh<DT>(v * 256.f);
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
  // initial recurrent state of each request (from L2)
  sycl::vec<float, 8> st[NRQ][2];
#pragma unroll
  for (int b = 0; b < NRQ; ++b)
#pragma unroll
    for (int rr = 0; rr < 2; ++rr)
      st[b][rr] = *reinterpret_cast<const sycl::vec<float, 8>*>(
          ssm_of(b) + ((size_t)h * DV + c * 64 + 2 * s + rr) * DK + lane * 8);

  // --- S2: conv1d over each request's window (all tokens at once), gated delta
  // rule in token order ---
  wait_for(it, S.cnt + C_A * 16, NWG * call);
  if (lid < 320) {
#pragma unroll
    for (int b = 0; b < NRQ; ++b) {
      float win[T + 3];  // DT values
      win[0] = f32<DT>(csl[(b * 3) * 320 + lid].u);
      win[1] = f32<DT>(csl[(b * 3 + 1) * 320 + lid].u);
      win[2] = f32<DT>(csl[(b * 3 + 2) * 320 + lid].u);
#pragma unroll
      for (int t = 0; t < T; ++t)
        win[3 + t] = (float)SB.qkv[(b * T + t) * CONVD + cch];
#pragma unroll
      for (int t = 0; t < T; ++t) {
        float acc = win[t] * cwt[0];
        acc += win[t + 1] * cwt[1];
        acc += win[t + 2] * cwt[2];
        acc += win[t + 3] * cwt[3];
        qk[(b * T + t) * 384 + lid] = rnd<DT>(acc / (1.f + sycl::exp(-acc)));
      }
      if (lid >= 256 || ((h & 1) == 0 && c == 0)) {  // the windows: taps 1, 2, then the T inputs
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
      if (lane < 2) SB.core[row * VD + h * DV + c * 64 + 2 * s + lane] = toh<DT>(lane == 0 ? o[0] : o[1]);
      float* dst = ib + (size_t)t * P.inter_ssm_tstride;  // the state after token t
#pragma unroll
      for (int rr = 0; rr < 2; ++rr)
        st_store8(dst + ((size_t)h * DV + c * 64 + 2 * s + rr) * DK + lane * 8, st[b][rr]);
    }
  }
  signal(it, S.cnt + C_B * 16);
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
      if (lane == 0) L.sm[640 + b * 32 + s] = ss;
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

// ------------------------------------------------ full attention (spec verify)
// The batched attention with row r = (request r / T, token r % T):
// - the context chunks of a row are its request's cached tokens
//   (P.seq_lens[request] - T, page table row = request); the T rows of a
//   request read the same KV, mostly from L2;
// - the T new k / v are written to the cache (P.slot[row]) and merged
//   causally: row r takes the new tokens 0 .. r % T of its request.
template <int NB, int T, typename DT>
inline void attn_layer_s(sycl::nd_item<1> it, uint8_t* slm, const Ptrs& P, const Scratch& S, const ScratchB& SB) {
  static_assert(NB % T == 0 && T >= 2, "spec verify: NB = requests x tokens");
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
  int lcs[NB], coff[NB + 1];
#pragma unroll
  for (int b = 0; b < NB; ++b)
    lcs[b] = sycl::max(P.seq_lens[b / T] - T, 0);
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
  attn_chunks_b<NB, DT>(it, slm, P, S, SB, lcs, coff, dyn, ach, [](int b) { return b / T; });
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
  // row's chunk partials + the request's new tokens, gate
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
    float* sn = sm3 + 128;  // [NB][T] new-token scores of head hh (row r, token
                            // j of its request; -inf: j > r % T)
    for (int p = s; p < NB * T; p += NSG) {
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
      if (lane == 0) sn[p] = j <= r % T ? d * SM_SCALE : -INFINITY;
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
        float Lw = 0.f, a = 0.f;
#pragma unroll
        for (int j = 0; j < T; ++j) {  // the request's new tokens 0 .. b2 % T (others: exp2(-inf) = 0)
          const float en = sycl::native::exp2(sn[b2 * T + j] - M);
          Lw += en;
          a += en * ld16<DT>(kvn_all, ((b2 / T) * T + j) * 512 + HD + d0 + d2);
        }
        for (int k2 = 0; k2 < TW / 16; ++k2)
          Lw += lwp[b2 * (TW / 16) + k2];
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

template <int NB, int T, typename DT>
class MonoGdnSK;  // kernel names
template <int NB, int T, typename DT>
class MonoAttnSK;

template <int NB, int T, typename DT>
void launch_gdn_s(sycl::queue& q, const Ptrs& P, const Scratch& S, const ScratchB& SB) {
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint8_t, 1> slm(sycl::range<1>(B_BYTES), h);
    syclex::properties props{syclex::sub_group_size<16>, intelex::grf_size<256>};
    h.parallel_for<MonoGdnSK<NB, T, DT>>(sycl::nd_range<1>(NWG * WGS, WGS), props, [=](sycl::nd_item<1> it) {
      gdn_layer_s<NB, T, DT>(it, slm.get_multi_ptr<sycl::access::decorated::no>().get(), P, S, SB);
    });
  });
}

template <int NB, int T, typename DT>
void launch_attn_s(sycl::queue& q, const Ptrs& P, const Scratch& S, const ScratchB& SB) {
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint8_t, 1> slm(sycl::range<1>(AB_SLM), h);
    syclex::properties props{syclex::sub_group_size<16>, intelex::grf_size<256>};
    h.parallel_for<MonoAttnSK<NB, T, DT>>(sycl::nd_range<1>(NWG * WGS, WGS), props, [=](sycl::nd_item<1> it) {
      attn_layer_s<NB, T, DT>(it, slm.get_multi_ptr<sycl::access::decorated::no>().get(), P, S, SB);
    });
  });
}

// the (rows, tokens) shapes of the verify kernels: every requests x tokens
// width the decode kernels have (1, 2, 3, 4, 6, 8 rows)
#define MONO_VERIFY_SHAPES(X) X(2, 2) X(4, 2) X(6, 2) X(8, 2) X(3, 3) X(6, 3) X(4, 4) X(8, 4) X(6, 6) X(8, 8)

// verify launchers of one layer kind and dtype (mono_{gdn,attn}_verify_
// {fp16,bf16}): false for an unsupported shape (nothing launched)
#pragma GCC visibility push(default)
template <typename DT>
bool launch_gdn_verify(sycl::queue& q, int nb, int t, const Ptrs& P, const Scratch& S, const ScratchB& SB) {
#define X(NB_, T_)                          \
  if (nb == NB_ && t == T_) {               \
    launch_gdn_s<NB_, T_, DT>(q, P, S, SB); \
    return true;                            \
  }
  MONO_VERIFY_SHAPES(X)
#undef X
  return false;
}

template <typename DT>
bool launch_attn_verify(sycl::queue& q, int nb, int t, const Ptrs& P, const Scratch& S, const ScratchB& SB) {
#define X(NB_, T_)                           \
  if (nb == NB_ && t == T_) {                \
    launch_attn_s<NB_, T_, DT>(q, P, S, SB); \
    return true;                             \
  }
  MONO_VERIFY_SHAPES(X)
#undef X
  return false;
}
#pragma GCC visibility pop

}  // namespace sgl_mono
