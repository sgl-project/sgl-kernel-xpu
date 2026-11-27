// SPDX-License-Identifier: Apache-2.0
//
// The kernels and launchers of the decoder layers: one SYCL kernel per layer
// kind, row count (and verify width) and dtype. Each launcher source
// (qwen35_*_launch_*.cpp) instantiates only the launcher it exports.
//
// Grid: 32 WG x 512 work-items (32 SIMD16 sub-groups), 256 GRF -> one WG per
// Xe core, all resident. Hand-offs between stages are monotonic per-stage
// arrival counters (target = 32 * call). `call` is a device-side epoch, so
// launches can be recorded into a graph. Every launch on a scratch must be
// matched by one launch on the peer card.
#pragma once
#include "common.hpp"

namespace sgl_mono {

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
    h.parallel_for<MonoLayerK<ATTN, DT>>(sycl::nd_range<1>(NWG * WGS, WGS), props, [=](sycl::nd_item<1> it) {
      if constexpr (ATTN)
        attn_layer<Q35, DT>(it, slm.get_multi_ptr<sycl::access::decorated::no>().get(), P, S);
      else
        gdn_layer<Q35, DT>(it, slm.get_multi_ptr<sycl::access::decorated::no>().get(), P, S);
    });
  });
}

// the split (mode 1 / 2) kernel of a layer kind: one kernel for all of its
// segments, the segment a runtime argument (include/sgl_kernel/stage/xe20/
// args.hpp, Split)
template <bool ATTN, typename DT>
class MonoLayerXK;  // kernel names

template <bool ATTN, typename DT>
void launch_tx(sycl::queue& q, const Ptrs& P, const Scratch& S, const Split& X) {
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint8_t, 1> slm(sycl::range<1>(ATTN ? A_BYTES : G_BYTES), h);
    syclex::properties props{syclex::sub_group_size<16>, intelex::grf_size<256>};
    h.parallel_for<MonoLayerXK<ATTN, DT>>(sycl::nd_range<1>(NWG * WGS, WGS), props, [=](sycl::nd_item<1> it) {
      if constexpr (ATTN)
        attn_layer<Q35, DT, true>(it, slm.get_multi_ptr<sycl::access::decorated::no>().get(), P, S, X);
      else
        gdn_layer<Q35, DT, true>(it, slm.get_multi_ptr<sycl::access::decorated::no>().get(), P, S, X);
    });
  });
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
      attn_layer_b<Q35, NB, DT>(it, slm.get_multi_ptr<sycl::access::decorated::no>().get(), P, S, SB);
    });
  });
}

template <int NB, typename DT>
void launch_gdn_b(sycl::queue& q, const Ptrs& P, const Scratch& S, const ScratchB& SB) {
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint8_t, 1> slm(sycl::range<1>(B_BYTES), h);
    syclex::properties props{syclex::sub_group_size<16>, intelex::grf_size<256>};
    h.parallel_for<MonoGdnBK<NB, DT>>(sycl::nd_range<1>(NWG * WGS, WGS), props, [=](sycl::nd_item<1> it) {
      gdn_layer_b<Q35, NB, DT>(it, slm.get_multi_ptr<sycl::access::decorated::no>().get(), P, S, SB);
    });
  });
}

template <int NB, typename DT>
class MonoAttnBXK;  // kernel names (split)
template <int NB, typename DT>
class MonoGdnBXK;

template <int NB, typename DT>
void launch_attn_bx(sycl::queue& q, const Ptrs& P, const Scratch& S, const ScratchB& SB, const Split& X) {
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint8_t, 1> slm(sycl::range<1>(AB_SLM), h);
    syclex::properties props{syclex::sub_group_size<16>, intelex::grf_size<256>};
    h.parallel_for<MonoAttnBXK<NB, DT>>(sycl::nd_range<1>(NWG * WGS, WGS), props, [=](sycl::nd_item<1> it) {
      attn_layer_b<Q35, NB, DT, true>(it, slm.get_multi_ptr<sycl::access::decorated::no>().get(), P, S, SB, X);
    });
  });
}

template <int NB, typename DT>
void launch_gdn_bx(sycl::queue& q, const Ptrs& P, const Scratch& S, const ScratchB& SB, const Split& X) {
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint8_t, 1> slm(sycl::range<1>(B_BYTES), h);
    syclex::properties props{syclex::sub_group_size<16>, intelex::grf_size<256>};
    h.parallel_for<MonoGdnBXK<NB, DT>>(sycl::nd_range<1>(NWG * WGS, WGS), props, [=](sycl::nd_item<1> it) {
      gdn_layer_b<Q35, NB, DT, true>(it, slm.get_multi_ptr<sycl::access::decorated::no>().get(), P, S, SB, X);
    });
  });
}

// decode launchers of one layer kind and dtype, 1, 2, 3, 4, 6 or 8 rows: each
// instance is compiled in its own translation unit and library
// (qwen35_{gdn,attn}_launch_{fp16,bf16}, built with -fvisibility=hidden) and
// called from qwen35_decode: exported
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
// split launchers (modes 1 / 2) of the decode kernels: segment X.seg of the
// layer; false for a row count without a split kernel (nothing launched)
template <typename DT>
bool launch_gdn_split(sycl::queue& q, int nb, const Ptrs& P, const Scratch& S, const ScratchB& SB, const Split& X) {
  if (nb == 1)
    launch_tx<false, DT>(q, P, S, X);
  else if (nb == 2)
    launch_gdn_bx<2, DT>(q, P, S, SB, X);
  else if (nb == 3)
    launch_gdn_bx<3, DT>(q, P, S, SB, X);
  else if (nb == 4)
    launch_gdn_bx<4, DT>(q, P, S, SB, X);
  else if (nb == 6)
    launch_gdn_bx<6, DT>(q, P, S, SB, X);
  else if (nb == 8)
    launch_gdn_bx<8, DT>(q, P, S, SB, X);
  else
    return false;
  return true;
}
template <typename DT>
bool launch_attn_split(sycl::queue& q, int nb, const Ptrs& P, const Scratch& S, const ScratchB& SB, const Split& X) {
  if (nb == 1)
    launch_tx<true, DT>(q, P, S, X);
  else if (nb == 2)
    launch_attn_bx<2, DT>(q, P, S, SB, X);
  else if (nb == 3)
    launch_attn_bx<3, DT>(q, P, S, SB, X);
  else if (nb == 4)
    launch_attn_bx<4, DT>(q, P, S, SB, X);
  else if (nb == 6)
    launch_attn_bx<6, DT>(q, P, S, SB, X);
  else if (nb == 8)
    launch_attn_bx<8, DT>(q, P, S, SB, X);
  else
    return false;
  return true;
}
#pragma GCC visibility pop

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
      gdn_layer_s<Q35, NB, T, DT>(it, slm.get_multi_ptr<sycl::access::decorated::no>().get(), P, S, SB);
    });
  });
}

template <int NB, int T, typename DT>
void launch_attn_s(sycl::queue& q, const Ptrs& P, const Scratch& S, const ScratchB& SB) {
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint8_t, 1> slm(sycl::range<1>(AB_SLM), h);
    syclex::properties props{syclex::sub_group_size<16>, intelex::grf_size<256>};
    h.parallel_for<MonoAttnSK<NB, T, DT>>(sycl::nd_range<1>(NWG * WGS, WGS), props, [=](sycl::nd_item<1> it) {
      attn_layer_s<Q35, NB, T, DT>(it, slm.get_multi_ptr<sycl::access::decorated::no>().get(), P, S, SB);
    });
  });
}

template <int NB, int T, typename DT>
class MonoGdnSXK;  // kernel names (split)
template <int NB, int T, typename DT>
class MonoAttnSXK;

template <int NB, int T, typename DT>
void launch_gdn_sx(sycl::queue& q, const Ptrs& P, const Scratch& S, const ScratchB& SB, const Split& X) {
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint8_t, 1> slm(sycl::range<1>(B_BYTES), h);
    syclex::properties props{syclex::sub_group_size<16>, intelex::grf_size<256>};
    h.parallel_for<MonoGdnSXK<NB, T, DT>>(sycl::nd_range<1>(NWG * WGS, WGS), props, [=](sycl::nd_item<1> it) {
      gdn_layer_s<Q35, NB, T, DT, true>(it, slm.get_multi_ptr<sycl::access::decorated::no>().get(), P, S, SB, X);
    });
  });
}

template <int NB, int T, typename DT>
void launch_attn_sx(sycl::queue& q, const Ptrs& P, const Scratch& S, const ScratchB& SB, const Split& X) {
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint8_t, 1> slm(sycl::range<1>(AB_SLM), h);
    syclex::properties props{syclex::sub_group_size<16>, intelex::grf_size<256>};
    h.parallel_for<MonoAttnSXK<NB, T, DT>>(sycl::nd_range<1>(NWG * WGS, WGS), props, [=](sycl::nd_item<1> it) {
      attn_layer_s<Q35, NB, T, DT, true>(it, slm.get_multi_ptr<sycl::access::decorated::no>().get(), P, S, SB, X);
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
// split verify launchers (modes 1 / 2): false for a shape without one
template <typename DT>
bool launch_gdn_verify_split(
    sycl::queue& q, int nb, int t, const Ptrs& P, const Scratch& S, const ScratchB& SB, const Split& XS) {
#define X(NB_, T_)                               \
  if (nb == NB_ && t == T_) {                    \
    launch_gdn_sx<NB_, T_, DT>(q, P, S, SB, XS); \
    return true;                                 \
  }
  MONO_VERIFY_SHAPES(X)
#undef X
  return false;
}
template <typename DT>
bool launch_attn_verify_split(
    sycl::queue& q, int nb, int t, const Ptrs& P, const Scratch& S, const ScratchB& SB, const Split& XS) {
#define X(NB_, T_)                                \
  if (nb == NB_ && t == T_) {                     \
    launch_attn_sx<NB_, T_, DT>(q, P, S, SB, XS); \
    return true;                                  \
  }
  MONO_VERIFY_SHAPES(X)
#undef X
  return false;
}
#pragma GCC visibility pop

}  // namespace sgl_mono
