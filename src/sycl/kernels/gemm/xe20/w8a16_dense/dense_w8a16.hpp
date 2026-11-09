/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

// Dedicated DENSE weight-only FP8 (W8A16) GEMM for Xe2 (BMG), SYCL-TLA only
// (CuTe tensors + Xe block-2D copy atoms + XE_DPAS_TT MMA atoms + reorder):
//
//   out[M, N] = (x[M, K] @ dequant(w[N, K]).T) (+ bias[N])
//   x bf16 row-major, w float8_e4m3fn [N, K] row-major (K contiguous), out bf16,
//   scale fp32: per-tensor (numel 1) | per-channel [N] | 128x128 block
//   [ceil(N/128), K/128]. Accumulation fp32 throughout.
//
// Kernel family (M-bucketed dispatch ladder):
//   * DECODE (M <= 64; also M <= 128 when N <= 2048, and any M with M * N <= 131072):
//     skinny DPAS GEMM, M tiles of MT = 8 / 16 / 32 / 64 rows (block scales: at most
//     32, larger M runs several M tiles). One work-group = one BN-column N tile x
//     all of K; its KS sub-groups split K (contiguous K ranges, whole 128-groups in
//     block mode). Per sub-group: FP8 weight block [BN x BK] via a U8 block-2D load
//     (prefetched DPF k-steps ahead), FP8 -> bf16 in registers (CuTe reorder),
//     x block [MT x BK] (M padded to MT by the 2D load's out-of-bounds zero fill),
//     XE_DPAS 8x16x16 bf16 -> fp32. The KS fp32 partial tiles are reduced in SLM
//     (one barrier, fixed order -> deterministic, no atomics), then sub-group 0 applies
//     scale/bias and stores bf16 with a block-2D store (rows >= M are clipped).
//   * PREFILL (M > 16): 2D-tiled DPAS GEMM, own mainloop (cute tutorial structure):
//     WG tile BM x BN x 32 over an SgM x SgN sub-group grid, A = bf16 block-2D,
//     B = U8 block-2D + reorder to bf16, L1 prefetch 3 k-tiles ahead, split barrier.
//   * Scale: per-tensor / per-channel in the epilogue (one code path: scale index =
//     n * s_stride, s_stride 0 = per-tensor). Block: a per-128-K-group fp32 partial
//     accumulator folded into the main accumulator with s[n/128][k/128] at each group
//     end (template BLOCK).

#include <cstdint>
#include <cute/tensor.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

namespace dense_w8a16 {
using namespace cute;

static constexpr int kSG = 16;
// Tuned on Arc Pro B65 for the Qwen3.6-35B-A3B TP=2/TP=4 dense shapes.
static constexpr int kDecodePrefetch = 2;           // decode: L1 prefetch distance (k-tiles)
static constexpr int kDecodeTargetSubgroups = 320;  // decode: split-K until ~this many sub-groups

struct Params {
  const void* x;      // bf16 [M, K]
  const void* w;      // fp8 e4m3 [N, K]
  const float* s;     // scale
  const float* bias;  // fp32 [N] or nullptr
  void* out;          // bf16 [M, N]
  int M, N, K;
  int s_stride;  // per-tensor 0, per-channel 1 (unused in block mode)
  int s_ld;      // block mode: K/128 (scale row stride)
  int ks_per;    // decode: k-tiles per sub-group slice
  int dpf;       // decode: L1 prefetch distance (k-tiles)
};

using DpasOp = XE_DPAS_TT<8, float, cute::bfloat16_t>;

template <class TA, class TB, class TC>
inline auto make_operands(const Params& p) {
  Tensor A = make_tensor(
      make_gmem_ptr(static_cast<const cute::bfloat16_t*>(p.x)),
      make_layout(make_shape(p.M, p.K), make_stride(p.K, _1{})));
  Tensor B = make_tensor(
      make_gmem_ptr(static_cast<const cute::float_e4m3_t*>(p.w)),
      make_layout(make_shape(p.N, p.K), make_stride(p.K, _1{})));
  Tensor C = make_tensor(
      make_gmem_ptr(static_cast<cute::bfloat16_t*>(p.out)), make_layout(make_shape(p.M, p.N), make_stride(p.N, _1{})));
  return cute::make_tuple(A, B, C);
}

// Epilogue: per-tensor/per-channel scale (non-block) + bias on the fp32 accumulator,
// using the identity-tensor coordinates of each accumulator element.
template <bool BLOCK, class Acc, class Coord>
inline void apply_epilogue(const Params& p, Acc& acc, const Coord& tCgC) {
  CUTE_UNROLL
  for (int i = 0; i < cute::size(acc.layout()); ++i) {
    const int n = get<1>(tCgC(i));
    const bool ok = n < p.N;
    float v = acc(i);
    if constexpr (!BLOCK) v *= ok ? p.s[n * p.s_stride] : 0.f;
    if (p.bias != nullptr) v += ok ? p.bias[n] : 0.f;
    acc(i) = v;
  }
}

// ----------------------------------------------------------------------------
// DECODE: in-WG split-K skinny DPAS GEMM
// ----------------------------------------------------------------------------
template <int MT, int BN, int BK, bool BLOCK, int GRF = 256>
struct DecodeKernel {
  using MMA = typename TiledMMAHelper<
      MMA_Atom<DpasOp>,
      Layout<Shape<Int<MT>, Int<BN>, Int<BK>>>,
      Layout<Shape<_1, _1, _1>, Stride<_1, _1, _0>>>::TiledMMA;
  static constexpr int kGroupK = 128 / BK;  // k-tiles per 128-wide scale group
  static_assert(BN <= 128 && (128 % BN) == 0, "block scale needs BN | 128");
  Params p;
  sycl::local_accessor<float, 1> slm;

  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> it) const {
    const int wg_lin = int(it.get_group(0));
    const int ntn = p.N / BN;
    const int wg = wg_lin % ntn;  // N tile
    const int wm = wg_lin / ntn;  // M tile (M > MT: several M tiles re-read the weight via L2)
    auto sg = it.get_sub_group();
    const int sgid = int(sg.get_group_linear_id());
    const int nsg = int(sg.get_group_linear_range());
    const int lane = int(sg.get_local_linear_id());

    auto [A, B, C] = make_operands<void, void, void>(p);
    MMA mma;
    Tensor cA = make_identity_tensor(A.shape());
    Tensor cB = make_identity_tensor(B.shape());
    Tensor cC = make_identity_tensor(C.shape());
    auto wg_tile = mma.tile_mnk();
    Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(wm, _));
    Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(wg, _));
    Tensor gC = local_tile(cC, wg_tile, make_coord(wm, wg, 0), Step<_1, _1, X>{});

    auto copy_a = make_block_2d_copy_A(mma, A);
    auto copy_b = make_block_2d_copy_B(mma, B);
    auto thr_mma = mma.get_slice(lane);
    auto thr_copy_a = copy_a.get_slice(lane);
    auto thr_copy_b = copy_b.get_slice(lane);
    auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
    auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));
    auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
    auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));
    Tensor tAgA = thr_copy_a.partition_S(gA);
    Tensor tBgB = thr_copy_b.partition_S(gB);
    auto tCrC = thr_mma.partition_sg_fragment_C(gC);
    Tensor tCgC = thr_mma.partition_C(gC);
    auto prefetch_b = make_block_2d_prefetch(copy_b);
    auto thr_pf_b = prefetch_b.get_slice(lane);
    auto pBgB = thr_pf_b.partition_S(gB);

    const int ktiles = p.K / BK;
    const int k0 = sgid * p.ks_per;
    const int k1 = sycl::min(k0 + p.ks_per, ktiles);

    clear(tCrC);
    const int dpf = p.dpf;
    for (int d = 0; d < dpf; ++d)
      if (k0 + d < k1) prefetch(prefetch_b, pBgB(_, _, _, k0 + d));

    if constexpr (!BLOCK) {
      for (int kt = k0; kt < k1; ++kt) {
        copy(copy_b, tBgB(_, _, _, kt), tBrB);
        copy(copy_a, tAgA(_, _, _, kt), tArA);
        if (kt + dpf < k1) prefetch(prefetch_b, pBgB(_, _, _, kt + dpf));
        reorder(tBrB, tCrB);
        reorder(tArA, tCrA);
        gemm(mma, tCrA, tCrB, tCrC);
      }
    } else {
      auto tCrG = thr_mma.partition_sg_fragment_C(gC);
      const int nb = get<1>(tCgC(0)) >> 7;  // SG-uniform: BN | 128, tile aligned
      const float* srow = p.s + (int64_t)nb * p.s_ld;
      for (int kg = k0; kg < k1; kg += kGroupK) {
        clear(tCrG);
        CUTE_UNROLL
        for (int j = 0; j < kGroupK; ++j) {
          const int kt = kg + j;
          copy(copy_b, tBgB(_, _, _, kt), tBrB);
          copy(copy_a, tAgA(_, _, _, kt), tArA);
          if (kt + dpf < k1) prefetch(prefetch_b, pBgB(_, _, _, kt + dpf));
          reorder(tBrB, tCrB);
          reorder(tArA, tCrA);
          gemm(mma, tCrA, tCrB, tCrG);
        }
        const float sc = srow[kg / kGroupK];
        CUTE_UNROLL
        for (int i = 0; i < cute::size(tCrC.layout()); ++i)
          tCrC(i) += tCrG(i) * sc;
      }
    }

    // ---- split-K reduction across the WG's sub-groups (fixed order) ----
    constexpr int NV = decltype(cute::size(tCrC.layout()))::value;
    if (nsg > 1) {
      float* my = &slm[0] + (sgid * NV) * kSG + lane;
      CUTE_UNROLL
      for (int i = 0; i < NV; ++i)
        my[i * kSG] = tCrC(i);
      sycl::group_barrier(it.get_group());
      if (sgid != 0) return;
      for (int s = 1; s < nsg; ++s) {
        const float* o = &slm[0] + (s * NV) * kSG + lane;
        CUTE_UNROLL
        for (int i = 0; i < NV; ++i)
          tCrC(i) += o[i * kSG];
      }
    }

    apply_epilogue<BLOCK>(p, tCrC, tCgC);
    auto copy_c = make_block_2d_copy_D(mma, C);
    auto thr_copy_c = copy_c.get_slice(lane);
    auto tCrD = thr_copy_c.partition_sg_fragment_S(gC);
    Tensor tDgC = thr_copy_c.partition_D(gC);
    reorder(tCrC, tCrD);
    copy(copy_c, tCrD, tDgC);
  }
};

// ----------------------------------------------------------------------------
// PREFILL: 2D-tiled DPAS GEMM
// ----------------------------------------------------------------------------
template <int BM, int BN, int SGM, int SGN, bool BLOCK>
struct PrefillKernel {
  static constexpr int BK = 32;
  using MMA = typename TiledMMAHelper<
      MMA_Atom<DpasOp>,
      Layout<Shape<Int<BM>, Int<BN>, Int<BK>>>,
      Layout<Shape<Int<SGM>, Int<SGN>, _1>, Stride<Int<SGN>, _1, _0>>>::TiledMMA;
  static constexpr int kWG = SGM * SGN * kSG;
  static constexpr int kGroupK = 128 / BK;
  static constexpr int kDPF = 3;
  static_assert((BN / SGN) <= 128 && (128 % (BN / SGN)) == 0, "block scale needs SG N extent | 128");
  Params p;
  int tiles_n;

  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> it) const {
    const int wgid = int(it.get_group(0));
    const int wg_n = wgid % tiles_n;
    const int wg_m = wgid / tiles_n;
    const int local_id = int(it.get_local_id(0));

    auto [A, B, C] = make_operands<void, void, void>(p);
    MMA mma;
    Tensor cA = make_identity_tensor(A.shape());
    Tensor cB = make_identity_tensor(B.shape());
    Tensor cC = make_identity_tensor(C.shape());
    auto wg_tile = mma.tile_mnk();
    Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(wg_m, _));
    Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(wg_n, _));
    Tensor gC = local_tile(cC, wg_tile, make_coord(wg_m, wg_n, 0), Step<_1, _1, X>{});

    auto copy_a = make_block_2d_copy_A(mma, A);
    auto copy_b = make_block_2d_copy_B(mma, B);
    auto thr_mma = mma.get_slice(local_id);
    auto thr_copy_a = copy_a.get_slice(local_id);
    auto thr_copy_b = copy_b.get_slice(local_id);
    auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
    auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));
    auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
    auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));
    Tensor tAgA = thr_copy_a.partition_S(gA);
    Tensor tBgB = thr_copy_b.partition_S(gB);
    auto tCrC = thr_mma.partition_sg_fragment_C(gC);
    Tensor tCgC = thr_mma.partition_C(gC);
    auto prefetch_a = make_block_2d_prefetch(copy_a);
    auto prefetch_b = make_block_2d_prefetch(copy_b);
    auto thr_pf_a = prefetch_a.get_slice(local_id);
    auto thr_pf_b = prefetch_b.get_slice(local_id);
    auto pAgA = thr_pf_a.partition_S(gA);
    auto pBgB = thr_pf_b.partition_S(gB);

    constexpr SPIRVScope barrier_scope = ScopeWorkgroup;
    const int ktiles = p.K / BK;
    int kpf = 0;
    clear(tCrC);
    CUTE_UNROLL
    for (; kpf < kDPF; ++kpf) {
      prefetch(prefetch_a, pAgA(_, _, _, kpf));
      prefetch(prefetch_b, pBgB(_, _, _, kpf));
    }

    if constexpr (!BLOCK) {
      for (int kt = 0; kt < ktiles; ++kt, ++kpf) {
        barrier_arrive(barrier_scope);
        copy(copy_a, tAgA(_, _, _, kt), tArA);
        copy(copy_b, tBgB(_, _, _, kt), tBrB);
        prefetch(prefetch_a, pAgA(_, _, _, kpf));
        prefetch(prefetch_b, pBgB(_, _, _, kpf));
        reorder(tArA, tCrA);
        reorder(tBrB, tCrB);
        gemm(mma, tCrA, tCrB, tCrC);
        barrier_wait(barrier_scope);
      }
    } else {
      auto tCrG = thr_mma.partition_sg_fragment_C(gC);
      const int nb = get<1>(tCgC(0)) >> 7;
      const float* srow = p.s + (int64_t)sycl::min(nb, (p.N - 1) >> 7) * p.s_ld;
      for (int kg = 0; kg < ktiles; kg += kGroupK) {
        clear(tCrG);
        CUTE_UNROLL
        for (int j = 0; j < kGroupK; ++j, ++kpf) {
          const int kt = kg + j;
          barrier_arrive(barrier_scope);
          copy(copy_a, tAgA(_, _, _, kt), tArA);
          copy(copy_b, tBgB(_, _, _, kt), tBrB);
          prefetch(prefetch_a, pAgA(_, _, _, kpf));
          prefetch(prefetch_b, pBgB(_, _, _, kpf));
          reorder(tArA, tCrA);
          reorder(tBrB, tCrB);
          gemm(mma, tCrA, tCrB, tCrG);
          barrier_wait(barrier_scope);
        }
        const float sc = srow[kg / kGroupK];
        CUTE_UNROLL
        for (int i = 0; i < cute::size(tCrC.layout()); ++i)
          tCrC(i) += tCrG(i) * sc;
      }
    }

    apply_epilogue<BLOCK>(p, tCrC, tCgC);
    auto copy_c = make_block_2d_copy_D(mma, C);
    auto thr_copy_c = copy_c.get_slice(local_id);
    auto tCrD = thr_copy_c.partition_sg_fragment_S(gC);
    Tensor tDgC = thr_copy_c.partition_D(gC);
    reorder(tCrC, tCrD);
    copy(copy_c, tCrD, tDgC);
  }
};

namespace syclex = sycl::ext::oneapi::experimental;
namespace intelex = sycl::ext::intel::experimental;

template <int MT, int BN, int BK, bool BLOCK, int GRF = 256>
static void launch_decode(sycl::queue& q, Params p) {
  using Kern = DecodeKernel<MT, BN, BK, BLOCK, GRF>;
  const int nwg = p.N / BN * ((p.M + MT - 1) / MT);
  const int ktiles = p.K / BK;
  const int g = BLOCK ? (128 / BK) : 1;
  const int ngroups = ktiles / g;
  // KS: enough sub-groups to fill the device (~1024 SGs), each slice >= 2 groups
  int ks = 1;
  while (ks < 32 && nwg * ks < kDecodeTargetSubgroups && (ngroups / (ks * 2)) >= (BLOCK ? 1 : 2))
    ks *= 2;
  if (ks > ngroups) ks = ngroups;
  const int per_g = (ngroups + ks - 1) / ks;
  p.ks_per = per_g * g;
  ks = (ktiles + p.ks_per - 1) / p.ks_per;
  constexpr int NV = MT * BN / kSG;
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> slm(sycl::range<1>((size_t)ks * NV * kSG), h);
    h.parallel_for(
        sycl::nd_range<1>((size_t)nwg * ks * kSG, (size_t)ks * kSG),
        syclex::properties{intelex::grf_size<GRF>},
        Kern{p, slm});
  });
}

template <int BM, int BN, int SGM, int SGN, bool BLOCK>
static void launch_prefill(sycl::queue& q, Params p) {
  using Kern = PrefillKernel<BM, BN, SGM, SGN, BLOCK>;
  const int tn = (p.N + BN - 1) / BN;
  const int tm = (p.M + BM - 1) / BM;
  q.submit([&](sycl::handler& h) {
    h.parallel_for(
        sycl::nd_range<1>((size_t)tn * tm * Kern::kWG, (size_t)Kern::kWG),
        syclex::properties{intelex::grf_size<256>},
        Kern{p, tn});
  });
}

inline int nwg_of(int M, int N, int BM, int BN) {
  return ((M + BM - 1) / BM) * ((N + BN - 1) / BN);
}

// scale_mode: 0 per-tensor, 1 per-channel, 2 block
static void
run(sycl::queue& q,
    void* out,
    const void* x,
    const void* w,
    const float* s,
    const float* bias,
    int M,
    int N,
    int K,
    int scale_mode) {
  if (M <= 0) return;
  Params p{x, w, s, bias, out, M, N, K, scale_mode == 0 ? 0 : 1, K / 128, 0, kDecodePrefetch};
  const bool blk = scale_mode == 2;
  // Decode-style (split-K skinny DPAS, M tiles of <= 64) vs 2D-tiled prefill: measured
  // crossover on B65 -- decode wins up to M=64 everywhere, up to M=128 for N <= 2048 and
  // while M*N <= 256*512 (the prefill tile loop cannot fill the device for small N).
  const bool use_dec = M <= 64 || (M <= 128 && N <= 2048) || (long)M * N <= 131072L;
  if (use_dec) {
    // tuned on B65 (side harness, TP=2/TP=4 shapes): BN=16 (more WGs/streams), BK=64
    // (full 64-B cache-line rows per block-2D load), ~320 SGs per launch.
#define DW8_DEC(MT_, BN_, BK_)                   \
  {                                              \
    if (blk)                                     \
      launch_decode<MT_, BN_, BK_, true>(q, p);  \
    else                                         \
      launch_decode<MT_, BN_, BK_, false>(q, p); \
  }
    if (M <= 8)
      DW8_DEC(8, 16, 64)
    else if (M <= 16)
      DW8_DEC(16, 16, 64)
    else if (M <= 32)
      DW8_DEC(32, 16, 32)
    else if (blk)
      DW8_DEC(32, 16, 32)  // block MT=64 needs a second 64-row accumulator (IGC retry); use 2 M tiles of 32
    else
      launch_decode<64, 16, 32, false>(q, p);
#undef DW8_DEC
    return;
  }
  if (!blk && nwg_of(M, N, 256, 256) >= 40) {
    launch_prefill<256, 256, 4, 8, false>(q, p);
  } else if (nwg_of(M, N, 128, 256) >= 40) {
    if (blk)
      launch_prefill<128, 256, 4, 8, true>(q, p);
    else
      launch_prefill<128, 256, 4, 8, false>(q, p);
  } else {
    if (blk)
      launch_prefill<64, 128, 2, 4, true>(q, p);
    else
      launch_prefill<64, 128, 2, 4, false>(q, p);
  }
}

}  // namespace dense_w8a16
