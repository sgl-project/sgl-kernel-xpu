// SPDX-License-Identifier: Apache-2.0
//
// The MTP drafter prologue (../mtp.hpp) on fewer than NWG work-groups (see
// launch.hpp in this directory): the same two phases (sums of squares; norms,
// fc and the mailbox gather), each run for the NWG work-groups' parts in turn,
// with the one wait between the phases. Bitwise the 32-work-group prologue.
#pragma once
#include "../mtp.hpp"

namespace sgl_mono {

template <int NB, typename DT>
inline void mtp_prologue_b60(sycl::nd_item<1> it, uint8_t* slm, const Ptrs& P, const Scratch& S, int rank) {
  const int w = (int)it.get_group(0), pw = (int)it.get_group_range(0);
  const int lid = it.get_local_linear_id();
  sycl::sub_group sg = it.get_sub_group();
  const int s = sg.get_group_linear_id();
  const int lane = sg.get_local_linear_id();
  auto grp = it.get_group();
  const uint32_t call = uc_load32(S.cnt + EPOCH) + 1u;
  const uint32_t mtag = uc_load32(S.cnt + MTPT + mtp_idx<NB>()) + 1u;
  half* z = reinterpret_cast<half*>(slm);
  float* red = reinterpret_cast<float*>(slm + NBMAX * MTP_FC_IN * 2);
  float* part = S.part_rt;

  // ---- sums of squares, each part's (signal: one arrival a part)
  for (int wg = w; wg < NWG; wg += pw) {
    const int gid = wg * WGS + lid;
    float se = 0.f, sh = 0.f;
    if (gid < NB * HID / 2) {
      const int b = gid / (HID / 2), wd = gid % (HID / 2);
      const sycl::vec<uint16_t, 2> ev = *reinterpret_cast<const sycl::vec<uint16_t, 2>*>(P.hs_in + b * HID + 2 * wd);
      const sycl::vec<uint16_t, 2> hv = *reinterpret_cast<const sycl::vec<uint16_t, 2>*>(P.res_in + b * HID + 2 * wd);
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
    if (lid < 2) {
      float a = 0.f;
#pragma unroll
      for (int i = 0; i < NSG; ++i)
        a += red[i * 2 + lid];
      part[wg * 2 + lid] = a;
    }
    signal(it, S.cnt + C_A * 16);
  }

  wait_for(it, S.cnt + C_A * 16, NWG * call);
  if (w == 0 && lid == 0) {  // every work-group has read the epoch and the tag
    cnt_ref(S.cnt[EPOCH]).store(call);
    cnt_ref(S.cnt[MTPT + mtp_idx<NB>()]).store(mtag);
  }

  // ---- norms, fc and gather, each part's
  for (int wg = w; wg < NWG; wg += pw) {
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      const float ve = part[(2 * b) * 2] + part[(2 * b + 1) * 2],
                  vh = part[(2 * b) * 2 + 1] + part[(2 * b + 1) * 2 + 1];
      const float re = sycl::rsqrt(ve / HID + EPS), rh = sycl::rsqrt(vh / HID + EPS);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int k = lid * 4 + i;
        z[b * MTP_FC_IN + k] = toh<DT>(ld16<DT>(P.hs_in, b * HID + k) * re * (1.f + ld16<DT>(P.in_norm_w, k)));
        z[b * MTP_FC_IN + HID + k] = toh<DT>(ld16<DT>(P.res_in, b * HID + k) * rh * (1.f + ld16<DT>(P.post_norm_w, k)));
      }
    }
    wg_bar(grp);

    const int j = wg * NSG + s;
    float acc[NB];
#pragma unroll
    for (int b = 0; b < NB; ++b)
      acc[b] = 0.f;
    {
      const e16* row = P.fc_w + (size_t)(rank * MTP_FC_OUT + j) * MTP_FC_IN;
#pragma unroll
      for (int pass = 0; pass < 2; ++pass) {
        sycl::uint4 wv[MTP_KCH / 2];
#pragma unroll
        for (int c = 0; c < MTP_KCH / 2; ++c)
          wv[c] = *reinterpret_cast<const sycl::uint4*>(row + ((pass * (MTP_KCH / 2) + c) * 16 + lane) * 8);
#pragma unroll
        for (int c = 0; c < MTP_KCH / 2; ++c) {
          const int k0 = ((pass * (MTP_KCH / 2) + c) * 16 + lane) * 8;
#pragma unroll
          for (int b = 0; b < NB; ++b)
            acc[b] += dot8_w16<DT>(wv[c], z + b * MTP_FC_IN + k0);
        }
      }
    }
    e16* yo = reinterpret_cast<e16*>(red);
    wg_bar(grp);
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      const float v = sycl::reduce_over_group(sg, acc[b], sycl::plus<float>());
      if (lane == 0) st16<DT>(yo, b * NSG + s, v);
    }
    wg_bar(grp);

    if (lid < NB * NSG / 2) {
      const int b = lid / (NSG / 2), p = lid % (NSG / 2);
      const uint32_t mine = pack16(yo[b * NSG + 2 * p].u, yo[b * NSG + 2 * p + 1].u);
      const size_t word = mtp_region<NB>() + (size_t)b * (MTP_FC_OUT / 2) + wg * (NSG / 2) + p;
      mb_put(S.peer + word, mb_word(mine, mtag));
      const int col = wg * NSG + 2 * p;
      *reinterpret_cast<uint32_t*>(P.hid_out + b * HID + rank * MTP_FC_OUT + col) = mine;
      const uint64_t pw_ = mb_wait(S.mbox + word, mtag, S.cnt);
      *reinterpret_cast<uint32_t*>(P.hid_out + b * HID + (1 - rank) * MTP_FC_OUT + col) = mb_val(pw_);
    }
    // the next part rewrites z and yo
    sycl::group_barrier(grp);
  }
}

template <int NB, typename DT>
class MonoB60MtpK;  // kernel names

template <int NB, typename DT>
void b60_launch_mtp_prologue(sycl::queue& q, int pw, const Ptrs& P, const Scratch& S, int rank) {
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint8_t, 1> slm(sycl::range<1>(MTP_SLM), h);
    syclex::properties props{intelex::grf_size<256>};
    h.parallel_for<MonoB60MtpK<NB, DT>>(
        sycl::nd_range<1>(pw * WGS, WGS), props, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
          mtp_prologue_b60<NB, DT>(it, slm.get_multi_ptr<sycl::access::decorated::no>().get(), P, S, rank);
        });
  });
}

// one dtype's launcher (qwen35_b60_mtp_launch_{fp16,bf16}): false for an
// unsupported row count (nothing launched)
#pragma GCC visibility push(default)
template <typename DT>
bool b60_launch_mtp(sycl::queue& q, int pw, int nb, const Ptrs& P, const Scratch& S, int rank) {
  if (nb == 1)
    b60_launch_mtp_prologue<1, DT>(q, pw, P, S, rank);
  else if (nb == 2)
    b60_launch_mtp_prologue<2, DT>(q, pw, P, S, rank);
  else if (nb == 3)
    b60_launch_mtp_prologue<3, DT>(q, pw, P, S, rank);
  else if (nb == 4)
    b60_launch_mtp_prologue<4, DT>(q, pw, P, S, rank);
  else if (nb == 6)
    b60_launch_mtp_prologue<6, DT>(q, pw, P, S, rank);
  else if (nb == 8)
    b60_launch_mtp_prologue<8, DT>(q, pw, P, S, rank);
  else
    return false;
  return true;
}
#pragma GCC visibility pop

}  // namespace sgl_mono
