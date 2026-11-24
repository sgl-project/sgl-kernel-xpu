/***************************************************************************************************
 * Copyright 2025 SGLang Team. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 **************************************************************************************************/
/*! \file
    \brief Fused NVFP4 weight-only-quantized MXFP8 grouped GEMM kernel for Xe3P.

    Xe3P's `bdpas` block-scaled MMA only accepts UE8M0 scale factors with a
    fixed block size of 32.  NVFP4 (E2M1 payload, E4M3 scales per 16, one FP32
    per-tensor global scale) is therefore emulated: B is fetched from memory as
    4-bit NVFP4, requantized to MXFP8 **in registers**, and the MMA runs as a
    plain MXFP8 x MXFP8 `bdpas`.  A is already MXFP8.  Weight traffic stays at
    NVFP4 density (0.5 B/element).

    Numerics.  A one-time prep pass (nvfp4_woq_prep, src/sycl/nvfp4_woq.cpp)
    splits the NVFP4 scales into

      sfmx (N, K/32) UE8M0   requantization scale 2^k, k minimal such that
                             448 * 2^k >= 6 * max(s_2j, s_2j+1)
      c    (N, K/16) BF16    per-16 multiplier gs_dq * sf16_j * 2^-k

    so the mainloop evaluates, per element, w = e4m3_rne(c[n][k/16] * e2m1) and
    hands `sfmx` to `bdpas` unchanged.  This is bit-exact against host
    requantization feeding the plain MXFP8 grouped GEMM: the accumulation order
    (one bdpas chain per 8x16 accumulator, k ascending) is the same.

    Register design (see xe_subbyte_upconvert.hpp for the layout facts).
    Everything a subgroup touches is moved as 32-bit lane arrays and fed to
    inline-vISA statements, never through cute's byte fragments:

      * packed B   : transposed d32 block-2D load, 16 columns x 64 k per message;
                     lane n owns column n, GRF j = nibbles k = 8j..8j+7
      * c          : transposed d32 block-2D load of (N, K/16) -> lane n owns its
                     four bf16 multipliers of the k tile
      * sfmx       : d8 block-2D load of the MN-major (K/32, N) surface, consumed
                     by bdpas through byte offsets
      * A          : d8 block-2D load, already in the hardware DPAS A layout
      * A scales   : one 2-byte gather per lane (16 rows) straight from the
                     row-major (sum_m, K/32) tensor, split into the two k-block
                     byte rows bdpas reads; no transposed scratch copy
      * conversion : per (16-column slice, K=32 atom) one vISA block of
                     shfl_idx4 + packed-HF mul + fcvt writing the E4M3 B operand
                     dwords in place (12 instructions / 128 values)
      * MMA        : raw `bdpas` on those registers
      * epilogue   : one d32 16x8 block-2D store per 8x16 accumulator

    Every block-2D surface spans all experts (global base, expert selected by
    the y coordinate), so no per-expert pointer table or base alignment is
    needed: B / c / sfmx rows beyond an expert's N belong to the next expert and
    only feed columns the epilogue never stores; A rows and output rows are
    bounded by surface height = expert row offset + expert rows.  Out-of-range
    rows and columns are zero-filled by the hardware on load and dropped on
    store, so there are no masks and no divergent branches in the k loop.

    Two k-loop flavours (template flag SHARE):
      * per-subgroup (decode / small M): each subgroup converts its own SG_N
        columns and runs its bdpas; no SLM, no barriers.
      * SLM-shared (large M): the SGM subgroups of an N-group split the
        conversion of its columns, publish the E4M3 operands through a
        double-buffered SLM stage and synchronise with one split barrier
        (arrive after the store, wait at the top of the next step) per 64-wide
        k step, so B is converted once per workgroup tile and the barrier
        latency hides behind the subgroup's own bdpas.

    Scheduling.  The kernel is persistent.  Each subgroup locates tile t by
    walking the device-resident problem sizes 16 experts at a time with a
    subgroup scan (no SLM, early exit), for t = group, group + num_groups, ...
    CUTLASS's group scheduler instead walks the list with one dependent global
    load per expert, which cost ~70 us per launch at E = 128 decode.
*/

#pragma once

// clang-format off
#include <cstdint>
#include <utility>

#include <sycl/sycl.hpp>

#include "cutlass/cutlass.h"

#include <cute/util/xe_split_barrier.hpp>

#include "xe_subbyte_upconvert.hpp"

namespace sgl_xpu {

/// Compile-time unrolled loop handing the index to `f` as an integral_constant,
/// so it can feed template arguments / inline-asm immediates.
template <int N, class F>
CUTLASS_DEVICE void static_for(F&& f) {
  [&]<int... I>(std::integer_sequence<int, I...>) {
    (f(std::integral_constant<int, I>{}), ...);
  }(std::make_integer_sequence<int, N>{});
}

struct NvFp4WoqMxfp8Params {
  int32_t const* problem_sizes;   // (E, 3) per-expert (M, N, K); only M is read
  int32_t const* expert_offsets;  // (E,) first row of each expert in A / out
  uint8_t const* a;               // (total_m, K) E4M3
  uint8_t const* sa;              // (total_m, K/32) UE8M0, row-major
  uint8_t const* b;               // (E, N, K/2) packed E2M1
  uint8_t const* sb;              // (E, K/32, N) UE8M0, MN-major
  uint8_t const* c;               // (E, N, K/16) BF16
  float* out;                     // (total_m, N) FP32
  int num_experts;
  int n;
  int k;
  int total_m;
};

/// Workgroup tile BM x BN, SGM x SGN subgroups.
///
///   KT : k extent of one mainloop step (64 or 128).  128 consumes every
///        64-byte line of packed B in one message, so no line has to survive
///        in L1 until the next step; a 64-wide tail step covers K % 128 == 64.
///   CB : k extent of one c load (64..512).  c rows hold 32 multipliers (512 k)
///        per 64-byte line; loading CB/32 dwords per row into registers and
///        picking per step keeps those lines out of L1 as well.
///   PF : prefetch distance in steps.
///   SHARE : convert each B element once per workgroup instead of once per
///        subgroup.  The SGM subgroups of an N-group split its N_ITER 16-column
///        slices, write the converted E4M3 operands to a double-buffered SLM
///        stage, and after one barrier per k step every subgroup reads all
///        N_ITER slices back.  Pays off once the conversion, not memory, bounds
///        the kernel (large M).
template <int BM, int BN, int SGM, int SGN, int KT_, int CB_, int PF, bool SHARE_ = false>
struct NvFp4WoqMxfp8Kernel {
  static constexpr bool SHARE = SHARE_;
  static constexpr int SG_NUMS_M = SGM;
  static constexpr int SG_NUMS_N = SGN;
  static constexpr int SG_M = BM / SGM;
  static constexpr int SG_N = BN / SGN;
  static constexpr int BLK_M = BM;
  static constexpr int BLK_N = BN;
  static constexpr int KT = KT_;
  static constexpr int CB = CB_;
  static constexpr int WG_SIZE = 16 * SGM * SGN;

  static_assert(BM % SGM == 0 && BN % SGN == 0, "uneven subgroup split");
  static_assert(SG_M % 8 == 0 && SG_M <= 64, "SG_M must be 8..64 (bdpas repeat count 8)");
  static_assert(SG_N == 16 || SG_N == 32 || SG_N == 64, "SG_N must be 16, 32 or 64");
  static_assert(KT == 64 || KT == 128, "KT must be 64 or 128");
  static_assert(CB % KT == 0 && CB <= 512, "CB must be a multiple of KT, at most 512");

  static constexpr int M_ITER = SG_M / 8;
  static constexpr int N_ITER = SG_N / 16;

  // A: per 64-k half, A_LOADS d8 block-2D loads of 2 x 32 bytes (k) x A_H rows.
  static constexpr int A_H = SG_M < 32 ? SG_M : 32;
  static constexpr int A_LOADS = SG_M / A_H;
  static constexpr int A_ATOMS_PER_LOAD = A_H / 8;
  using AVec = xe::uintv<A_H>;
  // A scales: per 64-k half one 16-row gather -> [kb0: 16 rows][kb1: 16 rows].
  static constexpr int SA_GATHERS = (SG_M + 15) / 16;
  using SaVec = cute::intel::vector_t<uint8_t, 2 * SA_GATHERS>;
  // B scales: d8 block-2D of KT/32 k-blocks x SFB_W bytes (one per column).
  static constexpr int SFB_W = SG_N <= 32 ? 32 : 64;
  // c: CB/32 dwords per column.
  using CVec = xe::uintv<CB / 32>;
  // B prefetch: d8 rows of KT/2 bytes.
  static constexpr int PF_H = SG_N < 32 ? SG_N : 32;
  static constexpr int PF_COUNT = SG_N / PF_H;

  // SLM sharing: slices converted per subgroup, and the stage size in dwords.
  static constexpr int SL_PER_SG = SHARE ? N_ITER / SGM : N_ITER;
  static constexpr int SLM_ATOM_DW = 128;  // one 8-GRF E4M3 B operand
  static constexpr int SLM_STAGE_DW = (BN / 16) * (KT / 32) * SLM_ATOM_DW;
  static constexpr int SLM_DWORDS = SHARE ? 2 * SLM_STAGE_DW : 1;
  static_assert(!SHARE || (SGM > 1 && N_ITER % SGM == 0 && KT == 64),
                "SLM sharing needs N_ITER divisible by SGM and a 64-wide k step");

  NvFp4WoqMxfp8Params p;
  sycl::local_accessor<uint32_t, 1> slm;  // SHARE only

  /// Locate tile `t`: expert, its first tile index and m-tile count.  Walks the
  /// problem sizes 16 experts at a time with a subgroup scan; returns false when
  /// t is past the last tile.  Uniform across the subgroup.
  CUTLASS_DEVICE bool find_tile(sycl::sub_group const& sg, int lane, int n_tiles, int t,
                                int& e_out, int& start_out, int& m_tiles_out) const {
    int carry = 0;
    for (int base = 0; base < p.num_experts; base += 16) {
      const int e = base + lane;
      const int m = e < p.num_experts ? p.problem_sizes[3 * e] : 0;
      const int cnt = ((m + BM - 1) / BM) * n_tiles;
      const int incl = sycl::inclusive_scan_over_group(sg, cnt, sycl::plus<int>());
      const int chunk = sycl::group_broadcast(sg, incl, 15);
      if (t < carry + chunk) {
        const bool hit = (t < carry + incl) && (t >= carry + incl - cnt);
        const int l = sycl::reduce_over_group(sg, hit ? lane : 16, sycl::minimum<int>());
        e_out = base + l;
        start_out = carry + sycl::select_from_group(sg, incl - cnt, l);
        m_tiles_out = sycl::select_from_group(sg, cnt, l) / n_tiles;
        return true;
      }
      carry += chunk;
    }
    return false;
  }

  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> item) const {
#ifdef __SYCL_DEVICE_ONLY__
    auto sg = item.get_sub_group();
    const int sg_id = static_cast<int>(sg.get_group_linear_id());
    const int lane = static_cast<int>(sg.get_local_linear_id());
    const int n_tiles = (p.n + BN - 1) / BN;
    const int sg_m = sg_id / SG_NUMS_N;
    const int sg_n = sg_id % SG_NUMS_N;
    const uint32_t lut = xe::e2m1_fp16x2_lut_entry(lane);

    for (int t = static_cast<int>(item.get_group_linear_id());;
         t += static_cast<int>(item.get_group_range(0))) {
      int e, start, m_tiles;
      if (!find_tile(sg, lane, n_tiles, t, e, start, m_tiles)) {
        break;
      }
      const int local = t - start;
      const int m_tile = local % m_tiles;
      const int n_tile = local / m_tiles;
      const int expert_m = p.problem_sizes[3 * e];
      const int row0 = p.expert_offsets[e];
      const int m0 = m_tile * BM + sg_m * SG_M;
      const int n0 = n_tile * BN + sg_n * SG_N;
      if constexpr (SHARE) {
        // Every subgroup of the workgroup takes part in the conversion and the
        // per-step barrier, live rows or not.
        run_tile_shared(e, expert_m, row0, m0, n0, sg_m, lane, lut);
      } else if (m0 < expert_m && n0 < p.n) {
        run_tile(e, expert_m, row0, m0, n0, lane, lut);
      }
    }
#endif
  }

  struct Surfaces {
    // The payload's block shape must match the message, so the 64-wide tail
    // step (KT == 128 only) gets its own B / B-scale payloads.
    xe::Block2D pa, pb, pc, psb, pfb, pfc, pfa, pb_tail, psb_tail;
    uint16_t const* sa_rows[SA_GATHERS];
    int sb_row0;
  };

  /// One mainloop step of S (= KT or 64) k at k0.  `cdw` is the dword of k0's
  /// multipliers within the current c registers.
  template <int S>
  CUTLASS_DEVICE void k_step(Surfaces const& s, int k0, CVec const (&creg)[N_ITER], int cdw,
                             xe::float8 (&acc)[M_ITER * N_ITER], uint32_t lut) const {
    using namespace xe;
    constexpr int HALVES = S / 64;
    constexpr int ATOMS = S / 32;
    using RawV = uintv<S / 8>;
    using CcV = uintv<ATOMS>;
    using SfbVec = uintv<SFB_W * ATOMS / 64>;

    RawV braw[N_ITER];
    AVec abuf[HALVES * A_LOADS];
    SaVec sfa[HALVES];
    SfbVec sfb;
    CcV cc[N_ITER];

    Block2D const& pb = S == KT ? s.pb : s.pb_tail;
    Block2D const& psb = S == KT ? s.psb : s.psb_tail;
    pb.set_x(k0 / 8);
    static_for<N_ITER>([&](auto n) {
      load_d32_t<S / 8, 16, decltype(n)::value * 16>(pb, braw[n]);
    });
    s.pa.set_x(k0);
    static_for<HALVES>([&](auto h) {
      static_for<A_LOADS>([&](auto i) {
        load_d8<2, 32, A_H, decltype(i)::value * A_H, decltype(h)::value * 64>(
            s.pa, abuf[decltype(h)::value * A_LOADS + decltype(i)::value]);
      });
    });
    psb.set_y(s.sb_row0 + k0 / 32);
    load_d8<1, SFB_W, ATOMS>(psb, sfb);
    static_for<HALVES>([&](auto h) {
      static_for<SA_GATHERS>([&](auto i) {
        constexpr int hh = decltype(h)::value;
        constexpr int ii = decltype(i)::value;
        const uint16_t v = s.sa_rows[ii][k0 / 64 + hh];
        sfa[hh][2 * ii] = static_cast<uint8_t>(v);
        sfa[hh][2 * ii + 1] = static_cast<uint8_t>(v >> 8);
      });
    });
    static_for<N_ITER>([&](auto n) {
      static_for<ATOMS>([&](auto a) {
        cc[decltype(n)::value][decltype(a)::value] = creg[n][cdw + decltype(a)::value];
      });
    });

    static_for<ATOMS>([&](auto kk_) {
      constexpr int kk = decltype(kk_)::value;
      constexpr int h = kk / 2;
      constexpr int kh = kk % 2;
      static_for<N_ITER>([&](auto n_) {
        constexpr int n = decltype(n_)::value;
        uint8 bop = nvfp4_to_e4m3_atom<kk, kk>(braw[n], cc[n], lut);
        static_for<M_ITER>([&](auto m_) {
          constexpr int m = decltype(m_)::value;
          constexpr int ml = m % A_ATOMS_PER_LOAD;
          bdpas_e4m3<kh * 32 * A_H + ml * 256,
                     32 * (m / 2) + 16 * kh + 8 * (m % 2),
                     kk * SFB_W + 16 * n>(
              acc[m + M_ITER * n], abuf[h * A_LOADS + m / A_ATOMS_PER_LOAD], bop, sfa[h], sfb);
        });
      });
    });
  }

  CUTLASS_DEVICE void prefetch_step(Surfaces const& s, int k0) const {
    using namespace xe;
    s.pfb.set_x(k0 / 2);
    static_for<PF_COUNT>([&](auto i) {
      prefetch_2d<8, KT / 2, PF_H, decltype(i)::value * PF_H>(s.pfb);
    });
    s.pfa.set_x(k0);
    static_for<KT / 64>([&](auto h) {
      static_for<A_LOADS>([&](auto i) {
        prefetch_2d<8, 64, A_H, decltype(i)::value * A_H, decltype(h)::value * 64>(s.pfa);
      });
    });
  }

  CUTLASS_DEVICE void load_c(Surfaces const& s, int kc, CVec (&creg)[N_ITER]) const {
    using namespace xe;
    s.pc.set_x(kc / 32);
    static_for<N_ITER>([&](auto n) {
      load_d32_t<CB / 32, 16, decltype(n)::value * 16>(s.pc, creg[n]);
    });
  }

  CUTLASS_DEVICE void prefetch_c(Surfaces const& s, int kc) const {
    using namespace xe;
    s.pfc.set_x(kc / 32);
    static_for<N_ITER>([&](auto n) {
      prefetch_2d<32, CB / 32, 16, decltype(n)::value * 16>(s.pfc);
    });
  }

  CUTLASS_DEVICE void run_tile(int e, int expert_m, int row0, int m0, int n0, int lane,
                               uint32_t lut) const {
#ifdef __SYCL_DEVICE_ONLY__
    using namespace xe;
    const int K = p.k;
    const int N = p.n;
    const int kb_count = K / 32;
    const int row_end = row0 + expert_m;  // surface height for A and out

    xe::float8 acc[M_ITER * N_ITER];
    static_for<M_ITER * N_ITER>([&](auto i) {
      acc[i] = 0.0f;
    });

    Surfaces s;
    s.pa.init(p.a, K, row_end, K, 32, A_H, 2);
    s.pb.init(p.b, K / 2, p.num_experts * N, K / 2, KT / 8, 16, 1);
    s.pc.init(p.c, K / 8, p.num_experts * N, K / 8, CB / 32, 16, 1);
    s.psb.init(p.sb, N, p.num_experts * kb_count, N, SFB_W, KT / 32, 1);
    s.pfb.init(p.b, K / 2, p.num_experts * N, K / 2, KT / 2, PF_H, 1);
    s.pfc.init(p.c, K / 8, p.num_experts * N, K / 8, CB / 32, 16, 1);
    s.pfa.init(p.a, K, row_end, K, 64, A_H, 1);
    const int brow = e * N + n0;
    if constexpr (KT != 64) {
      s.pb_tail.init(p.b, K / 2, p.num_experts * N, K / 2, 8, 16, 1);
      s.psb_tail.init(p.sb, N, p.num_experts * kb_count, N, SFB_W, 2, 1);
      s.pb_tail.set_y(brow);
      s.psb_tail.set_x(n0);
    }
    s.sb_row0 = e * kb_count;
    s.pa.set_y(row0 + m0);
    s.pb.set_y(brow);
    s.pc.set_y(brow);
    s.psb.set_x(n0);
    s.pfb.set_y(brow);
    s.pfc.set_y(brow);
    s.pfa.set_y(row0 + m0);
    // A-scale gather rows (clamped into the tensor; rows past the expert only
    // scale rows that are never stored).
    static_for<SA_GATHERS>([&](auto i) {
      int r = row0 + m0 + 16 * decltype(i)::value + lane;
      r = r < p.total_m ? r : p.total_m - 1;
      s.sa_rows[i] =
          reinterpret_cast<uint16_t const*>(p.sa + static_cast<int64_t>(r) * kb_count);
    });

    const int k_main = K - K % KT;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < PF; ++i) {
      if (i * KT < k_main) {
        prefetch_step(s, i * KT);
      }
    }
    prefetch_c(s, 0);

    CVec creg[N_ITER];
    for (int k0 = 0; k0 < k_main; k0 += KT) {
      const int kc = k0 % CB;
      if (kc == 0) {
        load_c(s, k0, creg);
        if (k0 + CB < K) {
          prefetch_c(s, k0 + CB);
        }
      }
      if (k0 + PF * KT < k_main) {
        prefetch_step(s, k0 + PF * KT);
      }
      k_step<KT>(s, k0, creg, kc / 32, acc, lut);
    }
    if constexpr (KT == 128) {
      if (k_main < K) {  // 64-wide tail
        if (k_main % CB == 0) {
          load_c(s, k_main, creg);
        }
        k_step<64>(s, k_main, creg, (k_main % CB) / 32, acc, lut);
      }
    }

    // Epilogue: rows >= row_end and columns >= N are outside the surface.
    Block2D pd;
    pd.init(p.out, N * 4, row_end, N * 4, 16, 8, 1);
    pd.set_xy(n0, row0 + m0);
    static_for<N_ITER>([&](auto n_) {
      constexpr int n = decltype(n_)::value;
      static_for<M_ITER>([&](auto m_) {
        constexpr int m = decltype(m_)::value;
        store_d32_16x8<16 * n, 8 * m>(pd, acc[m + M_ITER * n]);
      });
    });
#endif
  }

  /// SHARE path: see the class comment.  KT == 64, no tail.
  CUTLASS_DEVICE void run_tile_shared(int e, int expert_m, int row0, int m0, int n0, int sg_m,
                                      int lane, uint32_t lut) const {
#ifdef __SYCL_DEVICE_ONLY__
    using namespace xe;
    constexpr int ATOMS = KT / 32;
    using RawV = uintv<KT / 8>;
    using CcV = uintv<ATOMS>;
    using SfbVec = uintv<SFB_W * ATOMS / 64>;
    const int K = p.k;
    const int N = p.n;
    const int kb_count = K / 32;
    const int row_end = row0 + expert_m;
    const bool live = m0 < expert_m && n0 < N;
    const int sg_n = (n0 / SG_N) % SG_NUMS_N;

    xe::float8 acc[M_ITER * N_ITER];
    static_for<M_ITER * N_ITER>([&](auto i) {
      acc[i] = 0.0f;
    });

    // Converted slices of this subgroup: j = sg_m * SL_PER_SG + jl of its N-group.
    const int conv_row = e * N + n0 + 16 * sg_m * SL_PER_SG;
    Block2D pa, pb, pc, psb, pfb, pfa;
    pa.init(p.a, K, row_end, K, 32, A_H, 2);
    pb.init(p.b, K / 2, p.num_experts * N, K / 2, KT / 8, 16, 1);
    pc.init(p.c, K / 8, p.num_experts * N, K / 8, CB / 32, 16, 1);
    psb.init(p.sb, N, p.num_experts * kb_count, N, SFB_W, ATOMS, 1);
    pfb.init(p.b, K / 2, p.num_experts * N, K / 2, KT / 2, 16, 1);
    pfa.init(p.a, K, row_end, K, 64, A_H, 1);
    pa.set_y(row0 + m0);
    pb.set_y(conv_row);
    pc.set_y(conv_row);
    psb.set_x(n0);
    pfb.set_y(conv_row);
    pfa.set_y(row0 + m0);
    uint16_t const* sa_rows[SA_GATHERS];
    static_for<SA_GATHERS>([&](auto i) {
      int r = row0 + m0 + 16 * decltype(i)::value + lane;
      r = r < p.total_m ? r : p.total_m - 1;
      sa_rows[i] = reinterpret_cast<uint16_t const*>(p.sa + static_cast<int64_t>(r) * kb_count);
    });

    auto slm_base = slm.template get_multi_ptr<sycl::access::decorated::yes>().get();
    const int conv_slot = (sg_n * N_ITER + sg_m * SL_PER_SG) * ATOMS * SLM_ATOM_DW;
    const int read_slot = sg_n * N_ITER * ATOMS * SLM_ATOM_DW;

    auto prefetch_k = [&](int k0) {
      pfb.set_x(k0 / 2);
      static_for<SL_PER_SG>([&](auto j) {
        prefetch_2d<8, KT / 2, 16, decltype(j)::value * 16>(pfb);
      });
      pfa.set_x(k0);
      static_for<A_LOADS>([&](auto i) {
        prefetch_2d<8, 64, A_H, decltype(i)::value * A_H>(pfa);
      });
    };
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < PF; ++i) {
      if (i * KT < K) {
        prefetch_k(i * KT);
      }
    }

    // Software pipeline with a split barrier, one arrive/wait pair per k step.
    // Step k:  wait(k)   -> stage k%2 holds step k's operands (every subgroup
    //                       stored them before its arrive(k)), and stage
    //                       (k+1)%2 is free (its reads, step k-1, also precede
    //                       arrive(k))
    //          read stage k%2 into registers, convert step k+1 into stage
    //          (k+1)%2, arrive(k+1), then the bdpas of step k.
    // The barrier latency therefore hides behind this subgroup's own MMA work,
    // and the packed-B / c registers are loaded one step ahead.  The step after
    // the last one converts zero-filled out-of-range data into the idle stage,
    // which keeps the loop body branch-free (a branch would split the basic
    // block and stop the scheduler from interleaving conversion and bdpas).
    static_assert(!SHARE || CB == KT, "the SLM-shared path loads c per step");
    constexpr int SEM_REL = SemanticsRelease | SemanticsWGMemory;
    constexpr int SEM_ACQ = SemanticsAcquire | SemanticsWGMemory;
    RawV braw[SL_PER_SG];
    CVec creg[SL_PER_SG];
    auto load_raw = [&](int k0) {
      pc.set_x(k0 / 32);
      static_for<SL_PER_SG>([&](auto j) {
        load_d32_t<CB / 32, 16, decltype(j)::value * 16>(pc, creg[j]);
      });
      pb.set_x(k0 / 8);
      static_for<SL_PER_SG>([&](auto j) {
        load_d32_t<KT / 8, 16, decltype(j)::value * 16>(pb, braw[j]);
      });
    };
    auto convert_store = [&](int stage) {
      auto stage_ptr = slm_base + stage * SLM_STAGE_DW;
      static_for<SL_PER_SG>([&](auto j_) {
        constexpr int j = decltype(j_)::value;
        static_for<ATOMS>([&](auto kk_) {
          constexpr int kk = decltype(kk_)::value;
          uint8 bop = nvfp4_to_e4m3_atom<kk, kk>(braw[j], creg[j], lut);
          __spirv_SubgroupBlockWriteINTEL(
              stage_ptr + conv_slot + (j * ATOMS + kk) * SLM_ATOM_DW, bop);
        });
      });
    };

    load_raw(0);
    convert_store(0);
    __spirv_ControlBarrierArriveINTEL(ScopeWorkgroup, ScopeWorkgroup, SEM_REL);
    load_raw(KT);

    // Two copies of the k loop, selected outside it: a subgroup with no live
    // rows (the tail M-tile of an expert, e.g. 257 rows on a 256-row tile) only
    // converts its slices and keeps the barrier cadence.
    auto k_loop = [&](auto with_mma) {
      constexpr bool MMA = decltype(with_mma)::value;
      int stage = 0;
      for (int k0 = 0; k0 < K; k0 += KT, stage ^= 1) {
        prefetch_k(k0 + PF * KT);  // out-of-range prefetches are dropped
        AVec abuf[A_LOADS];
        SaVec sfa;
        SfbVec sfb;
        if constexpr (MMA) {
          pa.set_x(k0);
          static_for<A_LOADS>([&](auto i) {
            load_d8<2, 32, A_H, decltype(i)::value * A_H>(pa, abuf[i]);
          });
          psb.set_y(e * kb_count + k0 / 32);
          load_d8<1, SFB_W, ATOMS>(psb, sfb);
          static_for<SA_GATHERS>([&](auto i) {
            constexpr int ii = decltype(i)::value;
            const uint16_t v = sa_rows[ii][k0 / 64];
            sfa[2 * ii] = static_cast<uint8_t>(v);
            sfa[2 * ii + 1] = static_cast<uint8_t>(v >> 8);
          });
        }

        __spirv_ControlBarrierWaitINTEL(ScopeWorkgroup, ScopeWorkgroup, SEM_ACQ);
        uint8 bsl[MMA ? ATOMS * N_ITER : 1];
        if constexpr (MMA) {
          auto stage_ptr = slm_base + stage * SLM_STAGE_DW;
          static_for<ATOMS * N_ITER>([&](auto q) {
            bsl[q] = __spirv_SubgroupBlockReadINTEL<uint8>(
                stage_ptr + read_slot + decltype(q)::value * SLM_ATOM_DW);
          });
        }
        convert_store(stage ^ 1);
        __spirv_ControlBarrierArriveINTEL(ScopeWorkgroup, ScopeWorkgroup, SEM_REL);
        load_raw(k0 + 2 * KT);

        if constexpr (MMA) {
          static_for<ATOMS>([&](auto kk_) {
            constexpr int kk = decltype(kk_)::value;
            static_for<N_ITER>([&](auto n_) {
              constexpr int n = decltype(n_)::value;
              uint8 const& bop = bsl[n * ATOMS + kk];
              static_for<M_ITER>([&](auto m_) {
                constexpr int m = decltype(m_)::value;
                constexpr int ml = m % A_ATOMS_PER_LOAD;
                bdpas_e4m3<kk * 32 * A_H + ml * 256,
                           32 * (m / 2) + 16 * kk + 8 * (m % 2),
                           kk * SFB_W + 16 * n>(
                    acc[m + M_ITER * n], abuf[m / A_ATOMS_PER_LOAD], bop, sfa, sfb);
              });
            });
          });
        }
      }
    };
    if (live) {
      k_loop(std::true_type{});
    } else {
      k_loop(std::false_type{});
    }
    // Match the last arrive; afterwards no subgroup still reads either stage,
    // so the next tile may start overwriting them.
    __spirv_ControlBarrierWaitINTEL(ScopeWorkgroup, ScopeWorkgroup, SEM_ACQ);

    if (live) {
      Block2D pd;
      pd.init(p.out, N * 4, row_end, N * 4, 16, 8, 1);
      pd.set_xy(n0, row0 + m0);
      static_for<N_ITER>([&](auto n_) {
        constexpr int n = decltype(n_)::value;
        static_for<M_ITER>([&](auto m_) {
          constexpr int m = decltype(m_)::value;
          store_d32_16x8<16 * n, 8 * m>(pd, acc[m + M_ITER * n]);
        });
      });
    }
#endif
  }
};

}  // namespace sgl_xpu
