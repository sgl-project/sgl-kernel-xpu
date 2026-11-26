/***************************************************************************************************
 * Copyright (C) 2026 SGLang Team. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 **************************************************************************************************/

#pragma once

// Dedicated W8A8 INT8 scaled GEMM for Xe2 (BMG), SYCL-TLA only
// (CuTe tensors + Xe block-2D copy atoms + native XE_DPAS_TT<8, int32_t, int8_t>):
//
//   out[M, N] = out_dtype( float32(mat_a[M, K] @ mat_b[K, N]) * scales_a[M] * scales_b[N] (+ bias[N]) )
//
// Physical memory layouts:
//   a:       int8_t  [M, K], row-major    (stride = (lda, 1))
//   b:       int8_t  [N, K], row-major    (stride = (ldb, 1); i.e. mat_b[K, N] column-major)
//   scale_a: float   [M], contiguous
//   scale_b: float   [N], contiguous
//   bias:    16-bit  [N], contiguous (bf16 if is_bf16 else fp16, nullptr if absent)
//   out:     16-bit  [M, N], row-major    (stride = (ldd, 1), bf16 if is_bf16 else fp16)
//
// Kernel family (M/N/wave-aware dispatch ladder tuned on Intel Arc Pro B60, 20 Xe-cores):
//   * DECODE (M <= 32, or small-N M <= 64):
//     In-WG split-K skinny DPAS GEMM (MT = 8/16/32, BN = 16, BK = 64; GRF 128 for MT <= 16,
//     GRF 256 for MT = 32). Exact int32 SLM reduction across KS sub-groups (1 barrier,
//     deterministic, no atomics), then SG 0 runs the fused scale/bias/cast epilogue and
//     writes 16-bit output via block-2D store. Sustains ~380-430 GB/s (~84-94% GDDR6 peak).
//   * PREFILL (M > 32):
//     2D-tiled DPAS GEMM with split work-group barrier, 3-stage L1/L2 prefetch, and
//     M-grouped serpentine L2 swizzling:
//       - 128x256x64 (4x4 SGs = 16 SGs/WG = 40 WGs/wave, SwizM8 + snake): highest AI for
//         large compute-bound shapes when wave efficiency is high.
//       - 128x128x64 (4x4 SGs = 16 SGs/WG = 40 WGs/wave, SwizM8 + snake): eliminates
//         in-order tail-wave quantization on 20 Xe-cores (e.g. M=1024/2048, M=128).
//       -  64x128x64 (2x4 SGs =  8 SGs/WG = 80 WGs/wave, SwizM4): optimal for M=33..64
//         and narrow-N shapes.
//
// Single-instantiation 16-bit output: both bf16 and fp16 share the exact same int8 mainloop
// and 16-bit block-2D store (uint16_t), branching uniformly on is_bf16 only in the epilogue.
// This halves compile time, peak compile RSS, and binary size with zero mainloop overhead.

#include <bit>
#include <cstdint>
#include <cute/tensor.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

namespace int8_scaled_mm_xe20 {
using namespace cute;

static constexpr int kSG = 16;
static constexpr int kDecodeTargetSubgroups = 640;
static constexpr int kDecodePrefetch = 2;
static constexpr int kXe2WaveSlots16SG = 40;  // 20 Xe-cores * 2 WGs/Xe-core @ 256 GRF

using DpasOp = XE_DPAS_TT<8, int32_t, int8_t>;

struct Params {
  const int8_t* a;       // [M, K], stride (lda, 1)
  const int8_t* b;       // [N, K], stride (ldb, 1)
  const float* scale_a;  // [M]
  const float* scale_b;  // [N]
  const void* bias;      // [N] of bf16/fp16, or nullptr
  uint16_t* out;         // [M, N] of bf16/fp16 bits, stride (ldd, 1)
  int M, N, K;
  int lda, ldb, ldd;
  int ks_per;            // decode: k-tiles per sub-group slice
  int dpf;               // decode: L1 prefetch distance (k-tiles)
  bool is_bf16;          // true -> bfloat16, false -> float16
};

inline auto make_operands(const Params& p) {
  Tensor A = make_tensor(
      make_gmem_ptr(p.a),
      make_layout(make_shape(p.M, p.K), make_stride(p.lda, _1{})));
  Tensor B = make_tensor(
      make_gmem_ptr(p.b),
      make_layout(make_shape(p.N, p.K), make_stride(p.ldb, _1{})));
  Tensor C = make_tensor(
      make_gmem_ptr(p.out),
      make_layout(make_shape(p.M, p.N), make_stride(p.ldd, _1{})));
  return cute::make_tuple(A, B, C);
}

// Register-hoisted epilogue:
// In XE_DPAS_TT<8, int32_t, int8_t>, fragment shape is (V=8, AtomM, AtomN) where each
// sub-group lane owns one fixed column n per AtomN across all V*AtomM rows.
template <class AccInt32, class OutU16, class Coord>
inline void apply_epilogue(const Params& p, const AccInt32& acc_i32, OutU16& out_u16, const Coord& tCgC) {
  constexpr int nV = decltype(cute::size<0>(acc_i32.layout()))::value;
  constexpr int nM = decltype(cute::size<1>(acc_i32.layout()))::value;
  constexpr int nN = decltype(cute::size<2>(acc_i32.layout()))::value;

  float sb_cache[nN];
  float bias_cache[nN];
  const auto* bias_bf16 = static_cast<const cute::bfloat16_t*>(p.bias);
  const auto* bias_fp16 = static_cast<const cute::half_t*>(p.bias);
  CUTE_UNROLL
  for (int j = 0; j < nN; ++j) {
    const int n = get<1>(tCgC(0, 0, j));
    const bool ok_n = n < p.N;
    sb_cache[j] = ok_n ? p.scale_b[n] : 0.0f;
    if (p.bias != nullptr && ok_n) {
      bias_cache[j] = p.is_bf16 ? static_cast<float>(bias_bf16[n]) : static_cast<float>(bias_fp16[n]);
    } else {
      bias_cache[j] = 0.0f;
    }
  }

  CUTE_UNROLL
  for (int i = 0; i < nM; ++i) {
    CUTE_UNROLL
    for (int v = 0; v < nV; ++v) {
      const int m = get<0>(tCgC(v, i, 0));
      const float sa = (m < p.M) ? p.scale_a[m] : 0.0f;
      CUTE_UNROLL
      for (int j = 0; j < nN; ++j) {
        float val = static_cast<float>(acc_i32(v, i, j)) * sa * sb_cache[j];
        if (p.bias != nullptr) val += bias_cache[j];
        if (p.is_bf16) {
          out_u16(v, i, j) = std::bit_cast<uint16_t>(static_cast<cute::bfloat16_t>(val));
        } else {
          out_u16(v, i, j) = std::bit_cast<uint16_t>(static_cast<cute::half_t>(val));
        }
      }
    }
  }
}

// ----------------------------------------------------------------------------
// DECODE: in-WG split-K skinny DPAS GEMM
// ----------------------------------------------------------------------------
template <int MT, int BN, int BK, int GRF>
struct DecodeKernel {
  using MMA = typename TiledMMAHelper<
      MMA_Atom<DpasOp>,
      Layout<Shape<Int<MT>, Int<BN>, Int<BK>>>,
      Layout<Shape<_1, _1, _1>, Stride<_1, _1, _0>>>::TiledMMA;
  Params p;
  sycl::local_accessor<int32_t, 1> slm;

  void operator()(sycl::nd_item<1> it) const {
    const int wg_lin = int(it.get_group(0));
    const int ntm = (p.M + MT - 1) / MT;
    const int wg = wg_lin / ntm;  // N tile
    const int wm = wg_lin % ntm;  // M tile
    auto sg = it.get_sub_group();
    const int sgid = int(sg.get_group_linear_id());
    const int nsg = int(sg.get_group_linear_range());
    const int lane = int(sg.get_local_linear_id());

    auto [A, B, C] = make_operands(p);
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
    auto pBgB = prefetch_b.get_slice(lane).partition_S(gB);

    const int ktiles = (p.K + BK - 1) / BK;
    const int k0 = sgid * p.ks_per;
    const int k1 = sycl::min(k0 + p.ks_per, ktiles);

    clear(tCrC);
    const int dpf = p.dpf;
    for (int d = 0; d < dpf; ++d) {
      if (k0 + d < k1) prefetch(prefetch_b, pBgB(_, _, _, k0 + d));
    }

    for (int kt = k0; kt < k1; ++kt) {
      copy(copy_b, tBgB(_, _, _, kt), tBrB);
      copy(copy_a, tAgA(_, _, _, kt), tArA);
      if (kt + dpf < k1) prefetch(prefetch_b, pBgB(_, _, _, kt + dpf));
      reorder(tBrB, tCrB);
      reorder(tArA, tCrA);
      gemm(mma, tCrA, tCrB, tCrC);
    }

    // Exact int32 split-K reduction across the WG sub-groups (deterministic, no atomics).
    constexpr int NV = decltype(cute::size(tCrC.layout()))::value;
    if (nsg > 1) {
      int32_t* my = &slm[0] + (sgid * NV) * kSG + lane;
      CUTE_UNROLL
      for (int i = 0; i < NV; ++i)
        my[i * kSG] = tCrC(i);
      sycl::group_barrier(it.get_group());
      if (sgid != 0) return;
      for (int s = 1; s < nsg; ++s) {
        const int32_t* o = &slm[0] + (s * NV) * kSG + lane;
        CUTE_UNROLL
        for (int i = 0; i < NV; ++i)
          tCrC(i) += o[i * kSG];
      }
    }

    auto tCrU16 = make_subgroup_tensor(make_tensor<uint16_t>(tCrC.layout()), tCrC.tv_layout());
    apply_epilogue(p, tCrC, tCrU16, tCgC);

    auto copy_c = make_block_2d_copy_D(mma, C);
    auto thr_copy_c = copy_c.get_slice(lane);
    auto tCrD = thr_copy_c.partition_sg_fragment_S(gC);
    Tensor tDgC = thr_copy_c.partition_D(gC);
    reorder(tCrU16, tCrD);
    copy(copy_c, tCrD, tDgC);
  }
};

// ----------------------------------------------------------------------------
// PREFILL: 2D-tiled DPAS GEMM with split barrier + M-grouped L2 swizzling
// ----------------------------------------------------------------------------
template <int BM, int BN, int BK, int SGM, int SGN, int kDPF, int SWIZZLE_M, bool SNAKE>
struct PrefillKernel {
  using MMA = typename TiledMMAHelper<
      MMA_Atom<DpasOp>,
      Layout<Shape<Int<BM>, Int<BN>, Int<BK>>>,
      Layout<Shape<Int<SGM>, Int<SGN>, _1>, Stride<Int<SGN>, _1, _0>>>::TiledMMA;
  static constexpr int kWG = SGM * SGN * kSG;
  Params p;
  int tiles_m;
  int tiles_n;

  void operator()(sycl::nd_item<1> it) const {
    const int wgid = int(it.get_group(0));
    int wg_m, wg_n;
    if constexpr (SWIZZLE_M > 1) {
      const int num_pid_in_group = SWIZZLE_M * tiles_n;
      const int group_id = wgid / num_pid_in_group;
      const int first_m = group_id * SWIZZLE_M;
      const int gm = sycl::min(tiles_m - first_m, SWIZZLE_M);
      const int rem = wgid % num_pid_in_group;
      wg_m = first_m + (rem % gm);
      wg_n = rem / gm;
      if constexpr (SNAKE) {
        if (group_id & 1) wg_n = (tiles_n - 1) - wg_n;
      }
    } else {
      wg_n = wgid % tiles_n;
      wg_m = wgid / tiles_n;
    }
    const int local_id = int(it.get_local_id(0));

    auto [A, B, C] = make_operands(p);
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
    auto pAgA = prefetch_a.get_slice(local_id).partition_S(gA);
    auto pBgB = prefetch_b.get_slice(local_id).partition_S(gB);

    constexpr SPIRVScope barrier_scope = ScopeWorkgroup;
    const int ktiles = (p.K + BK - 1) / BK;
    int kpf = 0;
    clear(tCrC);
    CUTE_UNROLL
    for (; kpf < kDPF; ++kpf) {
      if (kpf < ktiles) {
        prefetch(prefetch_a, pAgA(_, _, _, kpf));
        prefetch(prefetch_b, pBgB(_, _, _, kpf));
      }
    }

    for (int kt = 0; kt < ktiles; ++kt, ++kpf) {
      barrier_arrive(barrier_scope);
      copy(copy_a, tAgA(_, _, _, kt), tArA);
      copy(copy_b, tBgB(_, _, _, kt), tBrB);
      if (kpf < ktiles) {
        prefetch(prefetch_a, pAgA(_, _, _, kpf));
        prefetch(prefetch_b, pBgB(_, _, _, kpf));
      }
      reorder(tArA, tCrA);
      reorder(tBrB, tCrB);
      gemm(mma, tCrA, tCrB, tCrC);
      barrier_wait(barrier_scope);
    }

    auto tCrU16 = make_subgroup_tensor(make_tensor<uint16_t>(tCrC.layout()), tCrC.tv_layout());
    apply_epilogue(p, tCrC, tCrU16, tCgC);

    auto copy_c = make_block_2d_copy_D(mma, C);
    auto thr_copy_c = copy_c.get_slice(local_id);
    auto tCrD = thr_copy_c.partition_sg_fragment_S(gC);
    Tensor tDgC = thr_copy_c.partition_D(gC);
    reorder(tCrU16, tCrD);
    copy(copy_c, tCrD, tDgC);
  }
};

namespace syclex = sycl::ext::oneapi::experimental;
namespace intelex = sycl::ext::intel::experimental;

template <int MT, int BN, int BK, int GRF>
static void launch_decode(sycl::queue& q, Params p) {
  using Kern = DecodeKernel<MT, BN, BK, GRF>;
  const int nwg = ((p.N + BN - 1) / BN) * ((p.M + MT - 1) / MT);
  const int ktiles = (p.K + BK - 1) / BK;
  int ks = 1;
  while (ks < 32 && nwg * ks < kDecodeTargetSubgroups && (ktiles / (ks * 2)) >= 2)
    ks *= 2;
  if (ks > ktiles) ks = ktiles;
  if (ks < 1) ks = 1;
  p.ks_per = (ktiles + ks - 1) / ks;
  ks = (ktiles + p.ks_per - 1) / p.ks_per;
  p.dpf = kDecodePrefetch;
  constexpr int NV = MT * BN / kSG;
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<int32_t, 1> slm(sycl::range<1>((size_t)ks * NV * kSG), h);
    h.parallel_for(
        sycl::nd_range<1>((size_t)nwg * ks * kSG, (size_t)ks * kSG),
        syclex::properties{syclex::sub_group_size<16>, intelex::grf_size<GRF>},
        Kern{p, slm});
  });
}

template <int BM, int BN, int BK, int SGM, int SGN, int kDPF, int SWIZZLE_M, bool SNAKE>
static void launch_prefill(sycl::queue& q, Params p) {
  using Kern = PrefillKernel<BM, BN, BK, SGM, SGN, kDPF, SWIZZLE_M, SNAKE>;
  const int tm = (p.M + BM - 1) / BM;
  const int tn = (p.N + BN - 1) / BN;
  q.submit([&](sycl::handler& h) {
    h.parallel_for(
        sycl::nd_range<1>((size_t)tm * tn * Kern::kWG, (size_t)Kern::kWG),
        syclex::properties{syclex::sub_group_size<16>, intelex::grf_size<256>},
        Kern{p, tm, tn});
  });
}

inline int nwg_of(int M, int N, int BM, int BN) {
  return ((M + BM - 1) / BM) * ((N + BN - 1) / BN);
}

inline float wave_efficiency(int nwg, int wave_slots) {
  const int waves = (nwg + wave_slots - 1) / wave_slots;
  return static_cast<float>(nwg) / static_cast<float>(waves * wave_slots);
}

static void run(
    sycl::queue& q,
    void* out,
    const int8_t* a,
    const int8_t* b,
    const float* scale_a,
    const float* scale_b,
    const void* bias,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldd,
    bool is_bf16) {
  if (M <= 0 || N <= 0) return;
  Params p{
      a,
      b,
      scale_a,
      scale_b,
      bias,
      static_cast<uint16_t*>(out),
      M,
      N,
      K,
      lda,
      ldb,
      ldd,
      0,
      kDecodePrefetch,
      is_bf16};

  const bool use_dec = M <= 32 || (M <= 64 && N <= 1024);
  if (use_dec) {
    if (M <= 8) {
      launch_decode<8, 16, 64, 128>(q, p);
    } else if (M <= 16) {
      launch_decode<16, 16, 64, 128>(q, p);
    } else {
      launch_decode<32, 16, 64, 256>(q, p);
    }
    return;
  }

  if (M <= 64 || N < 1024) {
    launch_prefill<64, 128, 64, 2, 4, 3, 4, false>(q, p);
    return;
  }

  // Wave-aware tile selection on 20 Xe-cores (40 concurrent 16-SG WGs/wave):
  // 128x256 has higher arithmetic intensity, while 128x128 doubles WG count to eliminate
  // severe in-order tail-wave quantization (e.g. 128 WGs -> 3.2 waves vs 256 WGs -> 6.4 waves).
  const float eff_256 = wave_efficiency(nwg_of(M, N, 128, 256), kXe2WaveSlots16SG);
  const float eff_128 = wave_efficiency(nwg_of(M, N, 128, 128), kXe2WaveSlots16SG);
  if (eff_128 > eff_256 * 1.03f) {
    launch_prefill<128, 128, 64, 4, 4, 3, 8, true>(q, p);
  } else {
    launch_prefill<128, 256, 64, 4, 4, 3, 8, true>(q, p);
  }
}

}  // namespace int8_scaled_mm_xe20
