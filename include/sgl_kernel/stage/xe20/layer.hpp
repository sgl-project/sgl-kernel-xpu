// SPDX-License-Identifier: Apache-2.0
//
// The frame every decoder-layer kernel shares: the SLM layouts (single-row
// and batched), the input RMSNorm, and the two-value all-reduce.
//
// Stage code: every function that depends on the model's shape takes the
// traits Tr (Dims<Shape>, dims.hpp) as its first template argument.
#pragma once
#include "sgl_kernel/stage/xe20/rows.hpp"

namespace sgl_kernel::stage::xe20 {

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

// res = h + r ; x = GemmaRMS(res) -> xn (natural) and xp (permuted)
template <class Tr, typename DT>
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
  const float rs = sycl::rsqrt(ss / Tr::HID + Tr::EPS);
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
template <class Tr>
inline SlmB slm_b(uint8_t* slm) {
  SlmB L;
  L.xa = reinterpret_cast<uint32_t*>(slm + Tr::B_XA);
  L.red = reinterpret_cast<float*>(slm + Tr::B_RED);
  L.sm = reinterpret_cast<float*>(slm + Tr::B_SM);
  L.rb = reinterpret_cast<float*>(slm + Tr::B_RB);
  L.xan = reinterpret_cast<uint32_t*>(slm + Tr::B_XAN);
  L.xap = reinterpret_cast<uint32_t*>(slm + Tr::B_XAP);
  L.lg = reinterpret_cast<float*>(slm + Tr::B_LG);
  L.re = reinterpret_cast<int*>(slm + Tr::B_RT);
  L.rw = reinterpret_cast<float*>(slm + Tr::B_RT + 256);
  L.uexp = reinterpret_cast<int*>(slm + Tr::B_U);
  L.uw = reinterpret_cast<float*>(slm + Tr::B_U + 256);
  L.nu = reinterpret_cast<int*>(slm + Tr::B_U + 256 + 2048);
  L.s6r = reinterpret_cast<float*>(slm + Tr::B_S6R);
  L.sgt = reinterpret_cast<float*>(slm + Tr::B_SGT);
  return L;
}

// res = h + r per row ; x = GemmaRMS(res) -> perm A tiles (rows >= NB zero)
template <class Tr, int NB, typename DT>
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
      v[b][i] = rnd<DT>(ld16<DT>(P.hs_in, b * Tr::HID + k) + ld16<DT>(P.res_in, b * Tr::HID + k));
      ss += v[b][i] * v[b][i];
    }
    ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
    if (lane == 0) L.sm[b * 32 + s] = ss;
  }
  a_zero_pad<NB>(L.xa, 0, 128, lid, Tr::WGS);
  wg_bar(it.get_group());
#pragma unroll
  for (int b = 0; b < NB; ++b) {
    float ss = 0.f;
#pragma unroll
    for (int w2 = 0; w2 < Tr::NSG; ++w2)
      ss += L.sm[b * 32 + w2];
    const float rs = sycl::rsqrt(ss / Tr::HID + Tr::EPS);
    half x[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
      x[i] = toh<DT>(v[b][i] * rs * (1.f + ld16<DT>(P.in_norm_w, lid * 4 + i)));
    a_put4_perm(L.xa, b, lid * 4, x);
  }
}

// all-reduce of two values per work-item: one mailbox word of two bf16 + call
// tag, half the PCIe writes of one fp32 per word (mb_ar2).
inline void
ar_two(uint64_t* peer, uint64_t* mine, size_t word, float& v0, float& v1, uint32_t call, const uint32_t* cnt) {
  mb_ar2(peer + word, mine + word, v0, v1, call, cnt);
}

}  // namespace sgl_kernel::stage::xe20
