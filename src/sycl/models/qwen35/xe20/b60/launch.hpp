// SPDX-License-Identifier: Apache-2.0
//
// The decoder layers on an Xe2 GPU with fewer Xe cores than the layer
// kernels' NWG = 32 work-groups (Arc Pro B60: 20). Functional, not tuned.
//
// A launch has as many work-groups as the GPU keeps resident (pw: one a Xe
// core, 20 <= pw < 32), and runs the 32 work-groups' parts of the layer in
// turn: work-group w runs the parts of work-groups w, w + pw, ... (Split.vwg).
// The layer runs as its split segments (include/sgl_kernel/stage/xe20/args.hpp,
// SplitSeg; the same code as modes 1 / 2): a segment keeps no state in
// registers or SLM for the next one, and starts with the wait for the
// previous segment's 32 arrivals. A work-group runs all of its parts of one
// segment before it starts the next, so every arrival comes from a resident
// work-group that waits for nothing earlier: no deadlock. The 32 parts and
// their data are those of the 32-work-group layer, so the results are bitwise
// the same.
//
// One launch runs a list of segments of one layer: all of them (mode 0, one
// kernel a layer) or one (modes 1 / 2). The epoch and the mailbox tag advance
// after the layer's last segment, in a separate task (as in modes 1 / 2).
#pragma once
#include "../launch.hpp"

namespace sgl_mono {

// the segments of one launch, run in order
struct Segs {
  int n = 0;
  int seg[8] = {};
};

// runs segments G of one layer, work-group parts v = w, w + pw, ... of each;
// f(X) runs one part of one segment
template <class F>
inline void b60_run(sycl::nd_item<1> it, const Segs& G, const Split& X0, F&& f) {
  const int w = (int)it.get_group(0), pw = (int)it.get_group_range(0);
  for (int i = 0; i < G.n; ++i) {
    for (int v = w; v < NWG; v += pw) {
      Split X = X0;
      X.seg = G.seg[i];
      X.vwg = v;
      f(X);
      // the next part reuses the SLM
      sycl::group_barrier(it.get_group());
    }
  }
}

template <bool ATTN, typename DT>
class MonoB60LayerK;  // kernel names
template <int NB, typename DT>
class MonoB60AttnBK;
template <int NB, typename DT>
class MonoB60GdnBK;
template <int NB, int T, typename DT>
class MonoB60AttnSK;
template <int NB, int T, typename DT>
class MonoB60GdnSK;

template <bool ATTN, typename DT>
void b60_launch_t(sycl::queue& q, int pw, const Ptrs& P, const Scratch& S, const Split& X, const Segs& G) {
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint8_t, 1> slm(sycl::range<1>(ATTN ? A_BYTES : G_BYTES), h);
    syclex::properties props{syclex::sub_group_size<16>, intelex::grf_size<256>};
    h.parallel_for<MonoB60LayerK<ATTN, DT>>(sycl::nd_range<1>(pw * WGS, WGS), props, [=](sycl::nd_item<1> it) {
      uint8_t* m = slm.get_multi_ptr<sycl::access::decorated::no>().get();
      b60_run(it, G, X, [&](const Split& Y) {
        if constexpr (ATTN)
          attn_layer<Q35, DT, true>(it, m, P, S, Y);
        else
          gdn_layer<Q35, DT, true>(it, m, P, S, Y);
      });
    });
  });
}

template <int NB, typename DT>
void b60_launch_attn_b(
    sycl::queue& q, int pw, const Ptrs& P, const Scratch& S, const ScratchB& SB, const Split& X, const Segs& G) {
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint8_t, 1> slm(sycl::range<1>(AB_SLM), h);
    syclex::properties props{syclex::sub_group_size<16>, intelex::grf_size<256>};
    h.parallel_for<MonoB60AttnBK<NB, DT>>(sycl::nd_range<1>(pw * WGS, WGS), props, [=](sycl::nd_item<1> it) {
      uint8_t* m = slm.get_multi_ptr<sycl::access::decorated::no>().get();
      b60_run(it, G, X, [&](const Split& Y) { attn_layer_b<Q35, NB, DT, true>(it, m, P, S, SB, Y); });
    });
  });
}

template <int NB, typename DT>
void b60_launch_gdn_b(
    sycl::queue& q, int pw, const Ptrs& P, const Scratch& S, const ScratchB& SB, const Split& X, const Segs& G) {
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint8_t, 1> slm(sycl::range<1>(B_BYTES), h);
    syclex::properties props{syclex::sub_group_size<16>, intelex::grf_size<256>};
    h.parallel_for<MonoB60GdnBK<NB, DT>>(sycl::nd_range<1>(pw * WGS, WGS), props, [=](sycl::nd_item<1> it) {
      uint8_t* m = slm.get_multi_ptr<sycl::access::decorated::no>().get();
      b60_run(it, G, X, [&](const Split& Y) { gdn_layer_b<Q35, NB, DT, true>(it, m, P, S, SB, Y); });
    });
  });
}

template <int NB, int T, typename DT>
void b60_launch_attn_s(
    sycl::queue& q, int pw, const Ptrs& P, const Scratch& S, const ScratchB& SB, const Split& X, const Segs& G) {
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint8_t, 1> slm(sycl::range<1>(AB_SLM), h);
    syclex::properties props{syclex::sub_group_size<16>, intelex::grf_size<256>};
    h.parallel_for<MonoB60AttnSK<NB, T, DT>>(sycl::nd_range<1>(pw * WGS, WGS), props, [=](sycl::nd_item<1> it) {
      uint8_t* m = slm.get_multi_ptr<sycl::access::decorated::no>().get();
      b60_run(it, G, X, [&](const Split& Y) { attn_layer_s<Q35, NB, T, DT, true>(it, m, P, S, SB, Y); });
    });
  });
}

template <int NB, int T, typename DT>
void b60_launch_gdn_s(
    sycl::queue& q, int pw, const Ptrs& P, const Scratch& S, const ScratchB& SB, const Split& X, const Segs& G) {
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint8_t, 1> slm(sycl::range<1>(B_BYTES), h);
    syclex::properties props{syclex::sub_group_size<16>, intelex::grf_size<256>};
    h.parallel_for<MonoB60GdnSK<NB, T, DT>>(sycl::nd_range<1>(pw * WGS, WGS), props, [=](sycl::nd_item<1> it) {
      uint8_t* m = slm.get_multi_ptr<sycl::access::decorated::no>().get();
      b60_run(it, G, X, [&](const Split& Y) { gdn_layer_s<Q35, NB, T, DT, true>(it, m, P, S, SB, Y); });
    });
  });
}

// launchers of one layer kind and dtype (qwen35_b60_{gdn,attn}{,_verify}_
// launch_{fp16,bf16}, each its own library): segments G of a decode (tokens 1)
// or verify step on pw work-groups; false for a shape without a kernel
// (nothing launched)
#pragma GCC visibility push(default)
template <typename DT>
bool b60_launch_gdn(
    sycl::queue& q,
    int pw,
    int nb,
    const Ptrs& P,
    const Scratch& S,
    const ScratchB& SB,
    const Split& X,
    const Segs& G) {
  if (nb == 1)
    b60_launch_t<false, DT>(q, pw, P, S, X, G);
  else if (nb == 2)
    b60_launch_gdn_b<2, DT>(q, pw, P, S, SB, X, G);
  else if (nb == 3)
    b60_launch_gdn_b<3, DT>(q, pw, P, S, SB, X, G);
  else if (nb == 4)
    b60_launch_gdn_b<4, DT>(q, pw, P, S, SB, X, G);
  else if (nb == 6)
    b60_launch_gdn_b<6, DT>(q, pw, P, S, SB, X, G);
  else if (nb == 8)
    b60_launch_gdn_b<8, DT>(q, pw, P, S, SB, X, G);
  else
    return false;
  return true;
}
template <typename DT>
bool b60_launch_attn(
    sycl::queue& q,
    int pw,
    int nb,
    const Ptrs& P,
    const Scratch& S,
    const ScratchB& SB,
    const Split& X,
    const Segs& G) {
  if (nb == 1)
    b60_launch_t<true, DT>(q, pw, P, S, X, G);
  else if (nb == 2)
    b60_launch_attn_b<2, DT>(q, pw, P, S, SB, X, G);
  else if (nb == 3)
    b60_launch_attn_b<3, DT>(q, pw, P, S, SB, X, G);
  else if (nb == 4)
    b60_launch_attn_b<4, DT>(q, pw, P, S, SB, X, G);
  else if (nb == 6)
    b60_launch_attn_b<6, DT>(q, pw, P, S, SB, X, G);
  else if (nb == 8)
    b60_launch_attn_b<8, DT>(q, pw, P, S, SB, X, G);
  else
    return false;
  return true;
}
template <typename DT>
bool b60_launch_gdn_verify(
    sycl::queue& q,
    int pw,
    int nb,
    int t,
    const Ptrs& P,
    const Scratch& S,
    const ScratchB& SB,
    const Split& X,
    const Segs& G) {
#define X_(NB_, T_)                                       \
  if (nb == NB_ && t == T_) {                             \
    b60_launch_gdn_s<NB_, T_, DT>(q, pw, P, S, SB, X, G); \
    return true;                                          \
  }
  MONO_VERIFY_SHAPES(X_)
#undef X_
  return false;
}
template <typename DT>
bool b60_launch_attn_verify(
    sycl::queue& q,
    int pw,
    int nb,
    int t,
    const Ptrs& P,
    const Scratch& S,
    const ScratchB& SB,
    const Split& X,
    const Segs& G) {
#define X_(NB_, T_)                                        \
  if (nb == NB_ && t == T_) {                              \
    b60_launch_attn_s<NB_, T_, DT>(q, pw, P, S, SB, X, G); \
    return true;                                           \
  }
  MONO_VERIFY_SHAPES(X_)
#undef X_
  return false;
}
#pragma GCC visibility pop

}  // namespace sgl_mono
