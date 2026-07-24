/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

// FP8 (E4M3) W8A8 MoE grouped-GEMM mainloop for Xe2 (BMG).
//
// Fork of src/sycl/kernels/moe/xe20/mxfp4_w4a16/moe_mainloop.hpp. Tile
// cadence and pipeline structure (prefetch, barrier, bias, fused-activation
// epilogue) are unchanged. The deltas vs. the MXFP4 W4A16 mainloop:
//
//   - Both A and B are fp8 e4m3 (byte-per-element, no sub-byte packing),
//     so BOTH operands need a register-level decode step, not just B.
//     Decode+relayout reuses the same `cute::reorder` "ConvertRelayout"
//     dispatch MXFP4 uses for E2M1->bf16 (generic NumericArrayConverter
//     path), just with float_e4m3_t->half_t instead of float_e2m1_t->bf16.
//   - Xe2 DPAS has no native fp8xfp8 MMA atom (confirmed by the existing
//     nsa/fp8_mqa_gemm_xe20.hpp and the upstream cutlass-sycl
//     `MainloopIntelW8A8`/`MainloopIntelXeXMX16FP8Scaling` collectives,
//     both of which decode fp8->fp16 before the DPAS). This mainloop
//     follows the same convention: decode to fp16 (not bf16) for the MMA,
//     and only casts back to bf16 at the very end when writing D.
//   - Weight (B) is block-quantized: one fp32 direct-multiplier scale per
//     (N-row, K-group) with FP8_GROUP_SIZE_K elements per K-group (128,
//     matching the common DeepSeek-style block-quant convention). Callers
//     with a genuinely 2-D-blocked (N-group, K-group) scale tensor are
//     expected to pre-expand it to per-N-row on the host/Python side
//     (repeat_interleave along N) before calling into this kernel - see
//     the op-level comment in GroupGemmFp8W8A8Xe20.cpp. Per-tensor
//     (single-scalar) weight scale is NOT supported by this first version;
//     only block-quant is wired up (see xpu_fp8_moe_minimal_plan.md).
//   - Activation (A) is quantized per-token (one fp32 scale per M-row, no
//     K-grouping). Because a per-token scale does not vary across K, it
//     is algebraically a common factor of the whole K-sum and is applied
//     ONCE to the fp32 accumulator after the K-loop (before bias), instead
//     of being threaded through the k-tile loop like the B-side scale.
//     This is cheaper than MXFP4-style per-tile scale application and
//     avoids adding any A-side scale-reload machinery.
//
// BLK_K stays 32 (same as every other MoE tile in this repo) rather than
// being raised to 128 to exactly match FP8_GROUP_SIZE_K. This means the
// B-scale is reloaded only once every (FP8_GROUP_SIZE_K / BLK_K) k-tiles
// instead of every iteration like MXFP4. Whether BLK_K should instead be
// raised to 128 (fewer, bigger iterations, less loop overhead, more
// register pressure from bigger A/B copy fragments) is an open tuning
// question - not decided here, flagged for later benchmarking.

#pragma once

#include <cute/tensor.hpp>
#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include "../common/activation.hpp"
#include "cutlass/float8.h"
#include "cutlass/half.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/sycl_event_manager.hpp"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace MoE_FP8_W8A8 {

using namespace cute;

inline constexpr int SILU = moe_xe20::ACT_SILU;
inline constexpr int SWIGLU_GPT_OSS = moe_xe20::ACT_SWIGLU_GPT_OSS;

// Weight block-quant K-group size (DeepSeek-style convention). This is a
// compile-time constant rather than a runtime `group_size` (unlike the
// upstream `MainloopIntelXeXMX16FP8Scaling`) to keep the first version
// simple; promote to a template parameter if a checkpoint with a different
// block size shows up.
static constexpr int FP8_GROUP_SIZE_K = 128;

// Number of work-items per subgroup on Xe (SIMD lane count).
static constexpr int SUBGROUP_SIZE = 16;

// Collaborative weight-scale load - identical technique to MXFP4's
// load_scale_slice (see mxfp4_w4a16/moe_mainloop.hpp for the full
// rationale/comment): each WI reads its own slice of the SG's [SG_N]
// scale row, then the SG broadcasts via select_from_group so every WI
// ends up with the full slice. `k_group_idx` is the weight-scale K-group
// index (NOT the k_tile index - see RELOAD_CADENCE in the mainloop body).
template <int SG_N, int ATOM_N, int BLK_N>
CUTLASS_DEVICE void load_weight_scale_slice(
    const float* scales_gmem_ptr,
    int row_stride,
    int wg_n,
    int k_group_idx,
    int thr_id,
    float* scale_out /* [SG_N] */) {
  static constexpr int N_per_wi = SG_N / SUBGROUP_SIZE;
  static_assert(SG_N % SUBGROUP_SIZE == 0, "SG_N must be a multiple of SUBGROUP_SIZE");

  const int sg_n_coord = (thr_id / SUBGROUP_SIZE) % ATOM_N;
  const int lane = thr_id % SUBGROUP_SIZE;
  const int n_base = wg_n * BLK_N + sg_n_coord * SG_N;

  float wi_local[N_per_wi];
  CUTE_UNROLL
  for (int sn = 0; sn < N_per_wi; ++sn) {
    const int n = n_base + lane * N_per_wi + sn;
    wi_local[sn] = scales_gmem_ptr[n * row_stride + k_group_idx];
  }

  auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
  CUTE_UNROLL
  for (int src_lane = 0; src_lane < SUBGROUP_SIZE; ++src_lane) {
    CUTE_UNROLL
    for (int sn = 0; sn < N_per_wi; ++sn) {
      scale_out[src_lane * N_per_wi + sn] = sycl::select_from_group(sg, wi_local[sn], src_lane);
    }
  }
}

// Apply per-(N-row, K-group) weight scale to an already-upcasted fp16
// MMA-B fragment. Same pattern as MXFP4's apply_B_scales_mma, but the
// multiplier is cast to half_t (fp16) instead of bf16 to match this
// mainloop's fp16 MMA compute type.
template <int SG_N_v, class FragFp16, class CoordFrag>
CUTLASS_DEVICE void
apply_B_weight_scale(FragFp16& frag, CoordFrag const& coord_frag, const float* scales_sg, int n_sg_base) {
  using FragLayout = typename FragFp16::layout_type;
  constexpr int frag_size = cute::size_v<FragLayout>;

  cutlass::half_t scale_fp16[SG_N_v];
  CUTE_UNROLL
  for (int i = 0; i < SG_N_v; ++i) {
    scale_fp16[i] = cutlass::half_t(scales_sg[i]);
  }

  CUTE_UNROLL
  for (int idx = 0; idx < frag_size; ++idx) {
    auto coord = coord_frag(idx);
    int n_in_sg = get<0>(coord) - n_sg_base;
    frag(idx) = frag(idx) * scale_fp16[n_in_sg];
  }
}

// Apply the per-token (per-M-row) activation descale to the fp32
// accumulator, once, after the K-loop. `coord_frag` is a coordinate
// companion for tCrC built from `thr_mma.partition_C(...)` on an identity
// tile, so `get<0>(coord)` gives each accumulator element's M-offset
// within the WG tile - this avoids hand-deriving the thread-id -> M
// mapping (which `add_bias` below never needed, since bias is N-only).
template <class FragF32, class CoordFrag>
CUTLASS_DEVICE void
apply_A_token_scale(FragF32& frag, CoordFrag const& coord_frag, const float* a_scale_gmem, int m_tile_start) {
  using FragLayout = typename FragF32::layout_type;
  constexpr int frag_size = cute::size_v<FragLayout>;

  CUTLASS_PRAGMA_UNROLL
  for (int idx = 0; idx < frag_size; ++idx) {
    auto coord = coord_frag(idx);
    int m_in_tile = get<0>(coord);
    frag(idx) = frag(idx) * a_scale_gmem[m_tile_start + m_in_tile];
  }
}

template <class FragFp16, class CoordFrag>
CUTLASS_DEVICE void apply_A_group_scale(
    FragFp16& frag,
    CoordFrag const& coord_frag,
    const float* a_scale_gmem,
    int m_tile_start,
    int k_group_idx,
    int k_groups) {
  using FragLayout = typename FragFp16::layout_type;
  constexpr int frag_size = cute::size_v<FragLayout>;

  CUTLASS_PRAGMA_UNROLL
  for (int idx = 0; idx < frag_size; ++idx) {
    auto coord = coord_frag(idx);
    int m_in_tile = get<0>(coord);
    cutlass::half_t scale_fp16 = cutlass::half_t(a_scale_gmem[(m_tile_start + m_in_tile) * k_groups + k_group_idx]);
    frag(idx) = frag(idx) * scale_fp16;
  }
}

template <int Stages>
class XeDefault {};

template <
    class DispatchPolicy_,
    class TiledCopyA_,
    class TiledCopyBPacked_,
    class TiledCopyD_,
    class ATensor_,
    class BPackedTensor_,
    class DTensor_,
    class BiasTensor_,
    class TiledMMA_,
    int ActType,
    bool WithBias>
struct MoEMainloopFp8W8A8 {
  static_assert(cutlass::detail::dependent_false<DispatchPolicy_>, "Could not find a mainloop specialization.");
};

template <
    int Stages,
    class TiledCopyA_,
    class TiledCopyBPacked_,
    class TiledCopyD_,
    class ATensor_,
    class BPackedTensor_,
    class DTensor_,
    class BiasTensor_,
    class TiledMMA_,
    int ActType,
    bool WithBias>
struct MoEMainloopFp8W8A8<
    XeDefault<Stages>,
    TiledCopyA_,
    TiledCopyBPacked_,
    TiledCopyD_,
    ATensor_,
    BPackedTensor_,
    DTensor_,
    BiasTensor_,
    TiledMMA_,
    ActType,
    WithBias> {
  using TiledMMA = TiledMMA_;
  using TiledCopyA = TiledCopyA_;
  using TiledCopyBPacked = TiledCopyBPacked_;
  using TiledCopyD = TiledCopyD_;
  using ATensor = ATensor_;
  using BPackedTensor = BPackedTensor_;
  using DTensor = DTensor_;
  using BiasTensor = BiasTensor_;

  MoEMainloopFp8W8A8() {}

  // -------------------------------------------------------------------------
  // Non-fused-activation path: one B, one weight-scale pointer. Used for
  // the down-projection GEMM, and (for now, see moe_kernel.hpp comment)
  // also for GEMM1 in place of a dedicated "unfused" variant.
  // -------------------------------------------------------------------------
  template <typename Coord>
  CUTLASS_DEVICE void operator()(
      ATensor& A,                 // (M,K)              fp8 e4m3
      BPackedTensor& Bp,          // (N,K)               fp8 e4m3
      const float* w_scale_gmem,  // fp32 [N, K/FP8_GROUP_SIZE_K] direct multiplier
      int w_scale_row_stride,     // fp32 stride per weight-scale N-row (= K/FP8_GROUP_SIZE_K)
      const float* a_scale_gmem,  // fp32 [M] per-token direct multiplier
      DTensor& D,                 // (M,N)              bf16
      Coord blk_coord,
      TiledMMA mma,
      int thr_id,
      BiasTensor Bias,
      bool act_scale_grouped,
      int act_scale_k_groups) {
    auto wg_m = get<0>(blk_coord);
    auto wg_n = get<1>(blk_coord);

    Tensor cA = make_identity_tensor(A.shape());
    Tensor cBp = make_identity_tensor(Bp.shape());
    Tensor cD = make_identity_tensor(D.shape());

    auto wg_tile = mma.tile_mnk();
    auto wg_coord = make_coord(wg_m, wg_n, 0);

    constexpr int BLK_M = get<0>(decltype(wg_tile){});
    constexpr int BLK_N = get<1>(decltype(wg_tile){});
    constexpr int BLK_K = get<2>(decltype(wg_tile){});
    static_assert(
        FP8_GROUP_SIZE_K % BLK_K == 0,
        "FP8_GROUP_SIZE_K must be a multiple of BLK_K so weight-scale groups align to whole k-tiles");
    constexpr int RELOAD_CADENCE = FP8_GROUP_SIZE_K / BLK_K;

    constexpr int ATOM_N_V = get<2>(typename TiledMMA::ThrLayoutVMNK{}.shape());
    constexpr int SG_N = BLK_N / ATOM_N_V;
    constexpr int N_per_wi = SG_N / SUBGROUP_SIZE;
    static_assert(N_per_wi >= 1, "SG_N must be at least SUBGROUP_SIZE");

    Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(wg_m, _));
    Tensor gBp = local_tile(cBp, select<1, 2>(wg_tile), make_coord(wg_n, _));
    Tensor gD = local_tile(cD, wg_tile, wg_coord, Step<_1, _1, X>{});

    TiledCopyA tiled_copy_a{A};
    TiledCopyBPacked tiled_copy_b{Bp};
    TiledCopyD tiled_copy_d{D};

    auto thr_copy_a = tiled_copy_a.get_slice(thr_id);
    auto thr_copy_b = tiled_copy_b.get_slice(thr_id);
    auto thr_copy_d = tiled_copy_d.get_slice(thr_id);
    auto thr_mma = mma.get_slice(thr_id);

    auto tAgA = thr_copy_a.partition_S(gA);
    auto tBgBp = thr_copy_b.partition_S(gBp);

    // Copy-layout fp8 fragments (raw 2D-block-load targets).
    auto tArA_packed = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
    auto tBrB_packed = thr_copy_b.partition_sg_fragment_D(gBp(_, _, 0));

    // fp16 MMA-layout fragments. `reorder` does the fp8->fp16 upcast and
    // copy-layout -> MMA-layout permute in one call (same ConvertRelayout
    // dispatch MXFP4 uses for E2M1->bf16).
    auto tSrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
    auto tSrB = thr_mma.partition_sg_fragment_B(gBp(_, _, 0));

    // Coord companions for scale indexing.
    auto cBp_coord_tile = make_identity_tensor(make_shape(Int<BLK_N>{}, Int<BLK_K>{}));
    auto tCrB_coord = thr_mma.partition_B(cBp_coord_tile);
    auto cA_coord_tile = make_identity_tensor(make_shape(Int<BLK_M>{}, Int<BLK_K>{}));
    auto tCrA_coord = thr_mma.partition_A(cA_coord_tile);
    auto cC_coord_tile = make_identity_tensor(make_shape(Int<BLK_M>{}, Int<BLK_N>{}));
    auto tCrC_coord = thr_mma.partition_C(cC_coord_tile);

    float w_scale_sg[SG_N];

    SubgroupTensor tCrC = thr_mma.partition_sg_fragment_C(gD);

    using TD = typename DTensor::element_type;
    TD tCrD_final_frag[tCrC.size()];
    Tensor tCrD_final_tensor = make_tensor(make_rmem_ptr(tCrD_final_frag), tCrC.layout());
    SubgroupTensor tCrD_final_sg_tensor = make_subgroup_tensor(tCrD_final_tensor, tCrC.tv_layout());
    Tensor tCgD = thr_mma.partition_C(gD);

    auto prefetch_a = make_block_2d_prefetch(tiled_copy_a);
    auto prefetch_b = make_block_2d_prefetch(tiled_copy_b);

    auto pAgA = prefetch_a.get_slice(thr_id).partition_S(gA);
    auto pBgBp = prefetch_b.get_slice(thr_id).partition_S(gBp);

    constexpr SPIRVScope barrier_scope = ScopeWorkgroup;
    int k_start_idx = 0;
    int prefetch_k = k_start_idx;
    const int prefetch_dist = Stages;
    int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));

    CUTE_UNROLL
    for (; prefetch_k < prefetch_dist; ++prefetch_k) {
      prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
      prefetch(prefetch_b, pBgBp(_, _, _, prefetch_k));
    }

    const int sg_n_coord = (thr_id / SUBGROUP_SIZE) % ATOM_N_V;
    const int n_sg_base = sg_n_coord * SG_N;

    for (int k_tile = k_start_idx; k_tile < k_tile_count; ++k_tile, ++prefetch_k) {
      barrier_arrive(barrier_scope);

      copy(tiled_copy_a, tAgA(_, _, _, k_tile), tArA_packed);
      copy(tiled_copy_b, tBgBp(_, _, _, k_tile), tBrB_packed);

      // Reload the weight-scale slice only when crossing into a new
      // FP8_GROUP_SIZE_K-wide K-group (every RELOAD_CADENCE k-tiles),
      // instead of every iteration like MXFP4 (whose group size equals
      // BLK_K exactly). See the file header for the BLK_K=32 vs 128
      // tradeoff this is deferring.
      if (k_tile % RELOAD_CADENCE == 0) {
        load_weight_scale_slice<SG_N, ATOM_N_V, BLK_N>(
            w_scale_gmem, w_scale_row_stride, wg_n, /*k_group_idx=*/k_tile / RELOAD_CADENCE, thr_id, w_scale_sg);
      }

      if (prefetch_k < k_tile_count) {
        prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
        prefetch(prefetch_b, pBgBp(_, _, _, prefetch_k));
      }

      reorder(tArA_packed, tSrA);
      reorder(tBrB_packed, tSrB);

      if (act_scale_grouped) {
        apply_A_group_scale<decltype(tSrA)>(
            tSrA, tCrA_coord, a_scale_gmem, wg_m * BLK_M, k_tile / RELOAD_CADENCE, act_scale_k_groups);
      }

      apply_B_weight_scale<SG_N>(tSrB, tCrB_coord, w_scale_sg, n_sg_base);

      cute::gemm(mma, tSrA, tSrB, tCrC);
      barrier_wait(barrier_scope);
    }

    // Descale (per-token activation scale factors out of the K-sum, see
    // file header), then bias - both operate on the real/dequantized
    // scale, matching standard quantized-GEMM epilogue ordering.
    if (!act_scale_grouped) {
      apply_A_token_scale(tCrC, tCrC_coord, a_scale_gmem, wg_m * BLK_M);
    }

    if constexpr (WithBias) {
      add_bias<decltype(tCrC), BLK_M, BLK_N>(Bias, tCrC, mma, wg_n, thr_id);
    }

    reorder(tCrC, tCrD_final_sg_tensor);
    copy(tiled_copy_d, tCrD_final_sg_tensor, tCgD);
  }

  // -------------------------------------------------------------------------
  // Fused-activation path: two Bs (gate + up), two weight-scale pointers,
  // one shared activation scale (both halves read the same A).
  // -------------------------------------------------------------------------
  template <typename Coord>
  CUTLASS_DEVICE void operator()(
      ATensor& A,                  // (M,K)
      BPackedTensor& Bp0,          // (N/2,K)  fp8 e4m3 (gate)
      BPackedTensor& Bp1,          // (N/2,K)  fp8 e4m3 (up)
      const float* w_scale0_gmem,  // fp32 gate weight scales
      const float* w_scale1_gmem,  // fp32 up weight scales
      int w_scale_row_stride,      // fp32 stride per weight-scale N-row (same for both halves)
      const float* a_scale_gmem,   // fp32 [M] per-token direct multiplier
      DTensor& D,
      Coord blk_coord,
      TiledMMA mma,
      int thr_id,
      BiasTensor Bias0,
      BiasTensor Bias1,
      float gemm1_alpha,
      float gemm1_limit,
      bool act_scale_grouped,
      int act_scale_k_groups) {
    auto wg_m = get<0>(blk_coord);
    auto wg_n = get<1>(blk_coord);
    auto wg_n1 = get<2>(blk_coord);

    Tensor cA = make_identity_tensor(A.shape());
    Tensor cBp0 = make_identity_tensor(Bp0.shape());
    Tensor cBp1 = make_identity_tensor(Bp1.shape());
    Tensor cC0 = make_identity_tensor(D.shape());
    Tensor cC1 = make_identity_tensor(D.shape());

    auto wg_tile = mma.tile_mnk();
    auto wg_coord = make_coord(wg_m, wg_n, 0);

    constexpr int BLK_M = get<0>(decltype(wg_tile){});
    constexpr int BLK_N = get<1>(decltype(wg_tile){});
    constexpr int BLK_K = get<2>(decltype(wg_tile){});
    static_assert(FP8_GROUP_SIZE_K % BLK_K == 0, "FP8_GROUP_SIZE_K must be a multiple of BLK_K");
    constexpr int RELOAD_CADENCE = FP8_GROUP_SIZE_K / BLK_K;

    constexpr int ATOM_N_V = get<2>(typename TiledMMA::ThrLayoutVMNK{}.shape());
    constexpr int SG_N = BLK_N / ATOM_N_V;
    constexpr int N_per_wi = SG_N / SUBGROUP_SIZE;
    static_assert(N_per_wi >= 1, "SG_N must be at least SUBGROUP_SIZE");

    Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(wg_m, _));
    Tensor gB = local_tile(cBp0, select<1, 2>(wg_tile), make_coord(wg_n, _));
    Tensor gC0 = local_tile(cC0, wg_tile, wg_coord, Step<_1, _1, X>{});
    Tensor gC1 = local_tile(cC1, wg_tile, wg_coord, Step<_1, _1, X>{});

    TiledCopyA tiled_copy_a{A};
    TiledCopyBPacked tiled_copy_b0{Bp0};
    TiledCopyBPacked tiled_copy_b1{Bp1};
    TiledCopyD tiled_copy_d{D};

    auto thr_copy_a = tiled_copy_a.get_slice(thr_id);
    auto thr_copy_b0 = tiled_copy_b0.get_slice(thr_id);
    auto thr_copy_b1 = tiled_copy_b1.get_slice(thr_id);
    auto thr_copy_d = tiled_copy_d.get_slice(thr_id);
    auto thr_mma = mma.get_slice(thr_id);

    auto tAgA = thr_copy_a.partition_S(gA);
    auto tBgB0 = thr_copy_b0.partition_S(gB);
    auto tBgB1 = thr_copy_b1.partition_S(gB);

    auto tArA_packed = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
    auto tSrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));

    auto tBrB0_packed = thr_copy_b0.partition_sg_fragment_D(gB(_, _, 0));
    auto tBrB1_packed = thr_copy_b1.partition_sg_fragment_D(gB(_, _, 0));
    auto tSrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

    auto cB_coord_tile = make_identity_tensor(make_shape(Int<BLK_N>{}, Int<BLK_K>{}));
    auto tCrB_coord = thr_mma.partition_B(cB_coord_tile);
    auto cA_coord_tile = make_identity_tensor(make_shape(Int<BLK_M>{}, Int<BLK_K>{}));
    auto tCrA_coord = thr_mma.partition_A(cA_coord_tile);
    auto cC_coord_tile = make_identity_tensor(make_shape(Int<BLK_M>{}, Int<BLK_N>{}));
    auto tCrC_coord = thr_mma.partition_C(cC_coord_tile);

    float w_scale0_sg[SG_N];
    float w_scale1_sg[SG_N];

    SubgroupTensor tCrC0 = thr_mma.partition_sg_fragment_C(gC0);
    SubgroupTensor tCrC1 = thr_mma.partition_sg_fragment_C(gC1);

    using TD = typename DTensor::element_type;
    TD tCrD_final_frag0[tCrC0.size()];
    Tensor tCrD_final_tensor0 = make_tensor(make_rmem_ptr(tCrD_final_frag0), tCrC0.layout());
    SubgroupTensor tCrD_final_sg_tensor0 = make_subgroup_tensor(tCrD_final_tensor0, tCrC0.tv_layout());

    Tensor tCgD = thr_mma.partition_C(gC0);

    auto prefetch_a = make_block_2d_prefetch(tiled_copy_a);
    auto prefetch_b0 = make_block_2d_prefetch(tiled_copy_b0);
    auto prefetch_b1 = make_block_2d_prefetch(tiled_copy_b1);

    auto pAgA = prefetch_a.get_slice(thr_id).partition_S(gA);
    auto pBgB0 = prefetch_b0.get_slice(thr_id).partition_S(gB);
    auto pBgB1 = prefetch_b1.get_slice(thr_id).partition_S(gB);

    constexpr SPIRVScope barrier_scope = ScopeWorkgroup;
    int k_start_idx = 0;
    int prefetch_k = k_start_idx;
    const int prefetch_dist = Stages;
    int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));

    CUTE_UNROLL
    for (; prefetch_k < prefetch_dist; ++prefetch_k) {
      prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
      prefetch(prefetch_b0, pBgB0(_, _, _, prefetch_k));
      prefetch(prefetch_b1, pBgB1(_, _, _, prefetch_k));
    }

    const int sg_n_coord = (thr_id / SUBGROUP_SIZE) % ATOM_N_V;
    const int n_sg_base = sg_n_coord * SG_N;

    for (int k_tile = k_start_idx; k_tile < k_tile_count; ++k_tile, ++prefetch_k) {
      barrier_arrive(barrier_scope);

      copy(tiled_copy_a, tAgA(_, _, _, k_tile), tArA_packed);
      reorder(tArA_packed, tSrA);

      if (act_scale_grouped) {
        apply_A_group_scale<decltype(tSrA)>(
            tSrA, tCrA_coord, a_scale_gmem, wg_m * BLK_M, k_tile / RELOAD_CADENCE, act_scale_k_groups);
      }

      if (k_tile % RELOAD_CADENCE == 0) {
        int k_group_idx = k_tile / RELOAD_CADENCE;
        load_weight_scale_slice<SG_N, ATOM_N_V, BLK_N>(
            w_scale0_gmem, w_scale_row_stride, wg_n, k_group_idx, thr_id, w_scale0_sg);
        load_weight_scale_slice<SG_N, ATOM_N_V, BLK_N>(
            w_scale1_gmem, w_scale_row_stride, wg_n1, k_group_idx, thr_id, w_scale1_sg);
      }

      copy(tiled_copy_b0, tBgB0(_, _, _, k_tile), tBrB0_packed);
      reorder(tBrB0_packed, tSrB);
      apply_B_weight_scale<SG_N>(tSrB, tCrB_coord, w_scale0_sg, n_sg_base);
      cute::gemm(mma, tSrA, tSrB, tCrC0);

      copy(tiled_copy_b1, tBgB1(_, _, _, k_tile), tBrB1_packed);
      reorder(tBrB1_packed, tSrB);
      apply_B_weight_scale<SG_N>(tSrB, tCrB_coord, w_scale1_sg, n_sg_base);
      cute::gemm(mma, tSrA, tSrB, tCrC1);

      if (prefetch_k < k_tile_count) {
        prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
        prefetch(prefetch_b0, pBgB0(_, _, _, prefetch_k));
        prefetch(prefetch_b1, pBgB1(_, _, _, prefetch_k));
      }

      barrier_wait(barrier_scope);
    }

    // Descale both halves (same A, same per-token scale) before bias and
    // the fused nonlinearity - order matters, see file header.
    if (!act_scale_grouped) {
      apply_A_token_scale(tCrC0, tCrC_coord, a_scale_gmem, wg_m * BLK_M);
      apply_A_token_scale(tCrC1, tCrC_coord, a_scale_gmem, wg_m * BLK_M);
    }

    if constexpr (WithBias) {
      add_bias<decltype(tCrC0), BLK_M, BLK_N>(Bias0, tCrC0, mma, wg_n, thr_id);
      add_bias<decltype(tCrC1), BLK_M, BLK_N>(Bias1, tCrC1, mma, wg_n1, thr_id);
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tCrC0.size(); ++i) {
      tCrC0(i) = moe_xe20::apply_fused_activation<ActType>(tCrC0(i), tCrC1(i), gemm1_alpha, gemm1_limit);
    }

    reorder(tCrC0, tCrD_final_sg_tensor0);
    copy(tiled_copy_d, tCrD_final_sg_tensor0, tCgD);
  }

  template <typename tCrC_t, int tile_m, int tile_n>
  void add_bias(const BiasTensor& Bias, tCrC_t& tCrC, const TiledMMA& mma, int wg_n, int thr_id) {
    static constexpr auto ATOM_M = get<1>(typename TiledMMA::ThrLayoutVMNK{}.shape());
    static constexpr auto ATOM_N = get<2>(typename TiledMMA::ThrLayoutVMNK{}.shape());

    static constexpr int sg_local_range = 16;
    int sg_local_n_coord = (thr_id / sg_local_range) % ATOM_N;
    int sg_local_id = (thr_id % sg_local_range);

    static constexpr auto SG_M = tile_m / ATOM_M;
    static constexpr auto SG_N = tile_n / ATOM_N;

    int n_tile_start = wg_n * tile_n;
    int n_sg_start = sg_local_n_coord * SG_N;

    CUTLASS_PRAGMA_UNROLL
    for (int sn = 0; sn < SG_N / sg_local_range; ++sn) {
      int sg_local_n = sn * sg_local_range + sg_local_id;
      float bias = static_cast<float>(Bias(n_tile_start + n_sg_start + sg_local_n));
      CUTLASS_PRAGMA_UNROLL
      for (int sm = 0; sm < SG_M; ++sm) {
        tCrC(sn * SG_M + sm) += bias;
      }
    }
  }
};

}  // namespace MoE_FP8_W8A8
