// SPDX-License-Identifier: Apache-2.0
//
// MTP drafter prologue (Qwen3.5 / Qwen3.6 MoE, TP2) as one kernel for NB <= 8
// rows: x = fc([norm_e(e); norm_h(h)]), with e the token embedding (looked up
// by the caller), h the hidden state the drafter continues from, norm_e /
// norm_h Gemma RMSNorms and fc SGLang's 16-bit nn.Linear(4096, 2048), the
// same full [2048][4096] on both ranks: each rank computes its half of the
// output rows and the halves are gathered through the P2P mailbox (a region
// and a tag counter for each row count). The draft
// decoder layer that follows is an attention layer kernel.
//
// Ptrs fields (the decoder-layer struct, reused; 16-bit data in the dtype DT):
// - hs_in [NB][2048]: e; res_in [NB][2048]: h; hid_out [NB][2048]: x.
// - in_norm_w / post_norm_w: pre_fc_norm_embedding / pre_fc_norm_hidden;
//   fc_w: fc [2048][4096].
#pragma once
#include "mono_layers_b.hpp"

namespace sgl_mono {

constexpr size_t MTP_MB_FC = 0;  // mailbox words of the fc gather
// A row count has its own tag counter and its own mailbox region (the rows
// 1, 2, 3, 4, 6, 8 regions follow each other), as the decoder layers' row
// configurations do: every word of a region is rewritten by every use of its
// row count, so a word left from an older use is at most one tag old and the
// 16-bit tag cannot match it. (With one region and counter for all row counts,
// the words of the rows a narrower call does not write keep their old tag, and
// a wider call 65536 calls later would take them for fresh.)
template <int NB>
constexpr int mtp_idx() {
  static_assert(NB == 1 || NB == 2 || NB == 3 || NB == 4 || NB == 6 || NB == 8, "MTP prologue rows");
  return NB == 1 ? 0 : NB == 2 ? 1 : NB == 3 ? 2 : NB == 4 ? 3 : NB == 6 ? 4 : 5;
}
template <int NB>
constexpr size_t mtp_region() {  // words from the start of the mailbox
  constexpr int rows_before = NB == 1 ? 0 : NB == 2 ? 1 : NB == 3 ? 3 : NB == 4 ? 6 : NB == 6 ? 10 : 16;
  return MTP_MB_FC + (size_t)rows_before * (MTP_FC_OUT / 2);
}
constexpr int MTP_ROWS_ALL = 1 + 2 + 3 + 4 + 6 + 8;
static_assert(MTP_MB_FC + (size_t)MTP_ROWS_ALL * MTP_FC_OUT / 2 <= MBOX_WORDS_B, "MTP prologue mailbox");
static_assert(MTPT + 6 <= HEALTH_PTR, "MTP tag counters");
constexpr int MTP_SLM = NBMAX * MTP_FC_IN * 2 + 4096;  // z rows (fp16) + reductions
constexpr int MTP_KCH = MTP_FC_IN / (16 * 8);          // 16-B chunks of a fc row per lane (32)

template <int NB, typename DT>
inline void mtp_prologue(sycl::nd_item<1> it, uint8_t* slm, const Ptrs& P, const Scratch& S, int rank) {
  const int wg = it.get_group(0);
  const int lid = it.get_local_linear_id();
  sycl::sub_group sg = it.get_sub_group();
  const int s = sg.get_group_linear_id();
  const int lane = sg.get_local_linear_id();
  auto grp = it.get_group();
  const uint32_t call = uc_load32(S.cnt + EPOCH) + 1u;
  const uint32_t mtag = uc_load32(S.cnt + MTPT + mtp_idx<NB>()) + 1u;  // this row count's tag
  half* z = reinterpret_cast<half*>(slm);                              // [NB][4096]
  float* red = reinterpret_cast<float*>(slm + NBMAX * MTP_FC_IN * 2);  // [32 SGs][2]
  float* part = S.part_rt;                                             // [NWG][2]: sums of squares e | h

  // ---- sums of squares: work-item = (row, word of two dims); 1024 words per
  // row, two WGs per row
  {
    const int gid = wg * WGS + lid;
    float se = 0.f, sh = 0.f;
    if (gid < NB * HID / 2) {
      const int b = gid / (HID / 2), w = gid % (HID / 2);
      const sycl::vec<uint16_t, 2> ev = *reinterpret_cast<const sycl::vec<uint16_t, 2>*>(P.hs_in + b * HID + 2 * w);
      const sycl::vec<uint16_t, 2> hv = *reinterpret_cast<const sycl::vec<uint16_t, 2>*>(P.res_in + b * HID + 2 * w);
      const float e0 = f32<DT>(ev[0]), e1 = f32<DT>(ev[1]);
      const float h0 = f32<DT>(hv[0]), h1 = f32<DT>(hv[1]);
      se = e0 * e0 + e1 * e1;
      sh = h0 * h0 + h1 * h1;
    }
    se = sycl::reduce_over_group(sg, se, sycl::plus<float>());
    sh = sycl::reduce_over_group(sg, sh, sycl::plus<float>());
    if (lane == 0) {
      red[s * 2] = se;
      red[s * 2 + 1] = sh;
    }
    wg_bar(grp);
    if (lid < 2) {  // fixed-order WG sums (both ranks run the same order)
      float a = 0.f;
#pragma unroll
      for (int i = 0; i < NSG; ++i)
        a += red[i * 2 + lid];
      part[wg * 2 + lid] = a;
    }
  }
  signal(it, S.cnt + C_A * 16);

  // ---- norms of every row -> z = [norm_e(e); norm_h(h)] (DT values in fp16)
  // in SLM
  wait_for(it, S.cnt + C_A * 16, NWG * call);
  if (wg == 0 && lid == 0) {  // every WG has read the epoch and the mailbox tag
    cnt_ref(S.cnt[EPOCH]).store(call);
    cnt_ref(S.cnt[MTPT + mtp_idx<NB>()]).store(mtag);
  }
#pragma unroll
  for (int b = 0; b < NB; ++b) {
    const float ve = part[(2 * b) * 2] + part[(2 * b + 1) * 2], vh = part[(2 * b) * 2 + 1] + part[(2 * b + 1) * 2 + 1];
    const float re = sycl::rsqrt(ve / HID + EPS), rh = sycl::rsqrt(vh / HID + EPS);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int k = lid * 4 + i;
      z[b * MTP_FC_IN + k] = toh<DT>(ld16<DT>(P.hs_in, b * HID + k) * re * (1.f + ld16<DT>(P.in_norm_w, k)));
      z[b * MTP_FC_IN + HID + k] = toh<DT>(ld16<DT>(P.res_in, b * HID + k) * rh * (1.f + ld16<DT>(P.post_norm_w, k)));
    }
  }
  wg_bar(grp);

  // ---- fc: SG = one output row of this rank's half (32 per WG), 4096 16-bit
  // weights per row: 16 lanes x 32 chunks of 8, in two passes
  const int j = wg * NSG + s;
  float acc[NB];
#pragma unroll
  for (int b = 0; b < NB; ++b)
    acc[b] = 0.f;
  {
    const e16* row = P.fc_w + (size_t)(rank * MTP_FC_OUT + j) * MTP_FC_IN;
#pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
      sycl::uint4 w[MTP_KCH / 2];
#pragma unroll
      for (int c = 0; c < MTP_KCH / 2; ++c)
        w[c] = *reinterpret_cast<const sycl::uint4*>(row + ((pass * (MTP_KCH / 2) + c) * 16 + lane) * 8);
#pragma unroll
      for (int c = 0; c < MTP_KCH / 2; ++c) {
        const int k0 = ((pass * (MTP_KCH / 2) + c) * 16 + lane) * 8;
#pragma unroll
        for (int b = 0; b < NB; ++b)
          acc[b] += dot8_w16<DT>(w[c], z + b * MTP_FC_IN + k0);
      }
    }
  }
  e16* yo = reinterpret_cast<e16*>(red);  // [NB][32] this WG's outputs, DT (reuses the reduction SLM)
  wg_bar(grp);
#pragma unroll
  for (int b = 0; b < NB; ++b) {
    const float v = sycl::reduce_over_group(sg, acc[b], sycl::plus<float>());
    if (lane == 0) st16<DT>(yo, b * NSG + s, v);
  }
  wg_bar(grp);

  // ---- gather: word = two outputs of one row; both ranks use the same words
  if (lid < NB * NSG / 2) {
    const int b = lid / (NSG / 2), p = lid % (NSG / 2);
    const uint32_t mine = pack16(yo[b * NSG + 2 * p].u, yo[b * NSG + 2 * p + 1].u);
    const size_t word = mtp_region<NB>() + (size_t)b * (MTP_FC_OUT / 2) + wg * (NSG / 2) + p;
    mb_put(S.peer + word, mb_word(mine, mtag));
    const int col = wg * NSG + 2 * p;
    *reinterpret_cast<uint32_t*>(P.hid_out + b * HID + rank * MTP_FC_OUT + col) = mine;
    const uint64_t pw = mb_wait(S.mbox + word, mtag, S.cnt);
    *reinterpret_cast<uint32_t*>(P.hid_out + b * HID + (1 - rank) * MTP_FC_OUT + col) = mb_val(pw);
  }
}

template <int NB, typename DT>
class MonoMtpK;  // kernel names

template <int NB, typename DT>
sycl::event launch_mtp_prologue(sycl::queue& q, const Ptrs& P, const Scratch& S, int rank) {
  return q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint8_t, 1> slm(sycl::range<1>(MTP_SLM), h);
    syclex::properties props{intelex::grf_size<256>};
    h.parallel_for<MonoMtpK<NB, DT>>(
        sycl::nd_range<1>(NWG * WGS, WGS), props, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
          mtp_prologue<NB, DT>(it, slm.get_multi_ptr<sycl::access::decorated::no>().get(), P, S, rank);
        });
  });
}

// one dtype's launcher (mono_mtp_{fp16,bf16}): false for an unsupported row
// count (nothing launched)
template <typename DT>
bool launch_mtp(sycl::queue& q, int nb, const Ptrs& P, const Scratch& S, int rank) {
  if (nb == 1)
    launch_mtp_prologue<1, DT>(q, P, S, rank);
  else if (nb == 2)
    launch_mtp_prologue<2, DT>(q, P, S, rank);
  else if (nb == 3)
    launch_mtp_prologue<3, DT>(q, P, S, rank);
  else if (nb == 4)
    launch_mtp_prologue<4, DT>(q, P, S, rank);
  else if (nb == 6)
    launch_mtp_prologue<6, DT>(q, P, S, rank);
  else if (nb == 8)
    launch_mtp_prologue<8, DT>(q, P, S, rank);
  else
    return false;
  return true;
}

}  // namespace sgl_mono
