/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

// MXFP4-B × BF16-A MoE grouped-GEMM mainloop for Xe2 (BMG).
//
// Fork of src/sycl/kernels/moe/xe20/moe_mainloop.hpp. Tile cadence is
// unchanged from the BF16 mainloop (BLK_K == 32, one cute::gemm per k-tile).
// The only delta is a register-level B-dequantization step inserted between
// the B-tile load and cute::gemm:
//
//   - B is stored in GMEM as packed MXFP4 (uint8 bytes, two E2M1 nibbles each,
//     low nibble = smaller-K element).
//   - A separate 2D block load brings in [SG_N x 1] UE8M0 scale bytes per
//     k-tile. BLK_K is hardcoded to MXFP4 GroupSize=32 (the only E2M1 group
//     size the hardware cares about) so each k-tile sees exactly one scale
//     byte per N-row, which simplifies the dequant loop.
//   - In registers, each work-item unpacks nibbles into a 16-entry signed
//     bf16 LUT and multiplies by the bf16 multiplier (2^(byte - 127)). The
//     result lands in a bf16 MMA-B fragment consumed by cute::gemm unchanged.
//
// Design pattern mirrors xe_mma_mixed_input's upconvert-in-registers flow:
// the packed-B fragment is allocated with the *same layout* as the bf16 MMA
// target, but with uint8 element storage. The 2D block load uses retile_D
// onto this fragment so bytes land in MMA-layout order rather than the
// natural 8-bit-atom order. Dequant then walks both fragments via linear
// index — layout is identical up to element-type-induced byte width.
//
// No change to prefetch pipeline, bias addition, fused-activation path, or
// tile scheduler semantics — all copied verbatim from moe_mainloop.hpp.

#pragma once

#include <cute/tensor.hpp>
#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

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

#define SILU 0
#define GELU 1
#define SWIGLU_GPT_OSS 2

namespace MoE_MXFP4 {

using namespace cute;

// MXFP4 invariant: one UE8M0 scale byte per 32 consecutive K-elements.
static constexpr int MXFP4_GROUP_SIZE = 32;

// Number of work-items per subgroup on Xe (SIMD lane count).
static constexpr int SUBGROUP_SIZE = 16;

// Hand-rolled scale load.
//
// The 2D-block-load machinery doesn't fit a (BLK_N, 1)-shaped scale tile
// cleanly, so we load scales with plain pointer arithmetic. Scales are tiny
// (one UE8M0 byte per N-row per k-tile — at most a few bytes per WI per
// k-tile), so we skip prefetch and accept the slow path; the latency is
// fully hidden under the A/B tile loads which are orders of magnitude
// bigger.
//
// Per-WI N-to-byte mapping matches add_bias: WI at (sg_n_coord, lane)
// owns N-positions { sn*SUBGROUP_SIZE + lane : sn ∈ [0, SG_N/16) }, with
// SG starting at n_base = wg_n * BLK_N + sg_n_coord * SG_N. This matches
// the MMA-B fragment's per-WI N layout so dequant can index scales by the
// same `n` value as its N-loop.
//
// scales_gmem_ptr points at the start of this expert's scales
// ([N, K/GROUP_SIZE] uint8, row-stride = K/GROUP_SIZE).
// k_scale_idx is the K-group index for this k-tile (= k_tile since we
// force BLK_K == GROUP_SIZE).
// Load the full SG_N-range of scale bytes for this subgroup's N-slice into
// a per-SG register array (every WI in the SG reads the same SG_N bytes).
// Each WI reads SG_N / SUBGROUP_SIZE contiguous bytes, then broadcasts them
// to the other lanes via a subgroup shuffle / shared writes. For simplicity
// (and because SG_N is small) we just have every WI read the whole range —
// the gmem load cache makes the redundant reads cheap, and this avoids the
// subgroup-shuffle plumbing.
template <int SG_N, int ATOM_N, int BLK_N>
CUTLASS_DEVICE void load_scale_slice(
    const uint8_t* scales_gmem_ptr,
    int row_stride,
    int wg_n,
    int k_scale_idx,
    int thr_id,
    uint8_t* scale_out /* [SG_N] */) {
  const int sg_n_coord = (thr_id / SUBGROUP_SIZE) % ATOM_N;
  const int n_base = wg_n * BLK_N + sg_n_coord * SG_N;

  CUTE_UNROLL
  for (int i = 0; i < SG_N; ++i) {
    scale_out[i] = scales_gmem_ptr[(n_base + i) * row_stride + k_scale_idx];
  }
}

// E2M1 decode LUT (16 entries). Bit 3 = sign, bits 2..0 = magnitude index.
// All values are exactly representable in bf16.
CUTLASS_DEVICE
cutlass::bfloat16_t mxfp4_e2m1_lut(uint8_t nibble) {
  switch (nibble & 0xF) {
    case 0x0: return cutlass::bfloat16_t(0.0f);
    case 0x1: return cutlass::bfloat16_t(0.5f);
    case 0x2: return cutlass::bfloat16_t(1.0f);
    case 0x3: return cutlass::bfloat16_t(1.5f);
    case 0x4: return cutlass::bfloat16_t(2.0f);
    case 0x5: return cutlass::bfloat16_t(3.0f);
    case 0x6: return cutlass::bfloat16_t(4.0f);
    case 0x7: return cutlass::bfloat16_t(6.0f);
    case 0x8: return cutlass::bfloat16_t(0.0f);  // -0.0 arithmetically equal to +0
    case 0x9: return cutlass::bfloat16_t(-0.5f);
    case 0xA: return cutlass::bfloat16_t(-1.0f);
    case 0xB: return cutlass::bfloat16_t(-1.5f);
    case 0xC: return cutlass::bfloat16_t(-2.0f);
    case 0xD: return cutlass::bfloat16_t(-3.0f);
    case 0xE: return cutlass::bfloat16_t(-4.0f);
    case 0xF: return cutlass::bfloat16_t(-6.0f);
  }
  return cutlass::bfloat16_t(0.0f);
}

// UE8M0 biased-exponent byte -> bf16 direct multiplier = 2^(byte - 127).
// Computed by directly writing the biased exponent into fp32's bit layout.
// This side-steps any numerical imprecision or domain-clamping in
// transcendental math paths, and all MXFP4 scales are exact powers of two
// so there's no rounding.
CUTLASS_DEVICE
cutlass::bfloat16_t mxfp4_ue8m0_to_bf16_mul(uint8_t byte) {
  // fp32 layout: sign (1) | exponent (8, biased by 127) | mantissa (23).
  // For 2^(k - 127) with biased_exp = byte, set exponent bits = byte, sign
  // and mantissa = 0. UE8M0 = 0x00 maps to fp32 denormal region; bit-pattern
  // (byte << 23) gives 0 for byte=0 (2^-127 underflows to 0 in fp32 anyway
  // for most realistic cases), which is acceptable for MoE weight scales.
  uint32_t bits = static_cast<uint32_t>(byte) << 23;
  float f = *reinterpret_cast<const float*>(&bits);
  return cutlass::bfloat16_t(f);
}

// Dequantize a packed-B fragment (float_e2m1_t, 4-bit each) + per-SG scale
// array to a bf16 MMA-B fragment, using a companion coordinate fragment to
// resolve each register position's N-row for scale lookup.
//
// Inputs:
//   packed     : SubgroupTensor/Tensor<float_e2m1_t> with MMA-B layout.
//   coord_frag : per-WI coord tensor from thr_mma.partition_B(identity(N,K)).
//                coord_frag(idx) returns a (n, k) tuple for each register
//                position.
//   scales_sg  : uint8_t[SG_N], one UE8M0 byte per N-row in this SG's
//                N-slice. scales_sg[0] is the scale for the SG's first
//                N-row (at n_sg_base_within_wg).
//   n_sg_base  : this SG's base N-index within the WG tile (we subtract it
//                from each element's coord to index into scales_sg).
//   out        : destination bf16 fragment with SAME layout as packed.
//
// Mirrors the upconvert-in-registers pattern in
// cutlass/gemm/collective/xe_mma_mixed_input.hpp::transform_quant — but
// with per-element scale lookup via a coord companion, since the packed
// fragment's linear index doesn't trivially decompose to (n, k).
template <class PackedFrag, class CoordFrag, class OutFrag>
CUTLASS_DEVICE void dequant_B_mxfp4_to_bf16(
    PackedFrag const& packed,
    CoordFrag const& coord_frag,
    const uint8_t* scales_sg,
    int n_sg_base,
    OutFrag& out) {
  using OutLayout = typename OutFrag::layout_type;
  using PackedLayout = typename PackedFrag::layout_type;

  constexpr int out_size = cute::size_v<OutLayout>;
  constexpr int packed_size = cute::size_v<PackedLayout>;
  static_assert(out_size == packed_size,
                "packed fragment and bf16 out fragment must have equal element counts");

  CUTE_UNROLL
  for (int idx = 0; idx < out_size; ++idx) {
    auto coord = coord_frag(idx);
    int n_in_sg = get<0>(coord) - n_sg_base;
    cutlass::bfloat16_t s = mxfp4_ue8m0_to_bf16_mul(scales_sg[n_in_sg]);
    cutlass::float_e2m1_t elt = packed(idx);
    uint8_t nibble = static_cast<uint8_t>(elt.raw()) & 0xF;
    out(idx) = mxfp4_e2m1_lut(nibble) * s;
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
    bool WithBias,
    int ActType>
struct MoEMainloopMxfp4 {
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
    bool WithBias,
    int ActType>
struct MoEMainloopMxfp4<
    XeDefault<Stages>,
    TiledCopyA_,
    TiledCopyBPacked_,
    TiledCopyD_,
    ATensor_,
    BPackedTensor_,
    DTensor_,
    BiasTensor_,
    TiledMMA_,
    WithBias,
    ActType> {
  using TiledMMA = TiledMMA_;
  using TiledCopyA = TiledCopyA_;
  using TiledCopyBPacked = TiledCopyBPacked_;
  using TiledCopyD = TiledCopyD_;
  using ATensor = ATensor_;
  using BPackedTensor = BPackedTensor_;
  using DTensor = DTensor_;
  using BiasTensor = BiasTensor_;

  MoEMainloopMxfp4() {}

  // -------------------------------------------------------------------------
  // Non-fused-activation path: one B, one scale-B.
  // -------------------------------------------------------------------------
  template <typename Coord>
  CUTLASS_DEVICE void operator()(
      ATensor& A,                      // (M,K)              bf16
      BPackedTensor& Bp,               // (N, K)             float_e2m1_t (4-bit)
      const uint8_t* scales_gmem,      // UE8M0 byte buffer (rows span N via scale_row_stride)
      int scale_row_stride,            // byte stride per scale N-row
      DTensor& D,                      // (M,N)              bf16
      Coord blk_coord,
      TiledMMA mma,
      int thr_id,
      BiasTensor Bias) {
    auto wg_m = get<0>(blk_coord);
    auto wg_n = get<1>(blk_coord);

    Tensor cA = make_identity_tensor(A.shape());
    Tensor cBp = make_identity_tensor(Bp.shape());
    Tensor cD = make_identity_tensor(D.shape());

    auto wg_tile = mma.tile_mnk();
    auto wg_coord = make_coord(wg_m, wg_n, 0);

    constexpr int BLK_N = get<1>(decltype(wg_tile){});
    constexpr int BLK_K = get<2>(decltype(wg_tile){});
    static_assert(BLK_K == MXFP4_GROUP_SIZE,
                  "MXFP4 mainloop assumes BLK_K == GROUP_SIZE == 32 so each k-tile has one scale per N-row");

    // Compute subgroup-local N dimensions for scale loading. ATOM_N is the
    // SG replication along N (mode 2 of TiledMMA::ThrLayoutVMNK); SG_N is
    // BLK_N / ATOM_N.
    constexpr int ATOM_N_V = get<2>(typename TiledMMA::ThrLayoutVMNK{}.shape());
    constexpr int SG_N = BLK_N / ATOM_N_V;
    constexpr int N_per_wi = SG_N / SUBGROUP_SIZE;
    static_assert(N_per_wi >= 1, "SG_N must be at least SUBGROUP_SIZE");

    // B tile uses the full logical K stride; the packed (K/2 byte) footprint
    // is encoded in CuTe's float_e2m1_t sub-byte arithmetic.
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

    // A fragments.
    auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
    auto tSrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));

    // B fragments — two-step layout path (mirrors BF16 mainloop's reorder
    // pattern plus a dequant step):
    //   tBrB_packed  : float_e2m1_t, COPY layout — destination of the
    //                  8-bit 2D-load. SubgroupTensor with tv_layout for
    //                  reorder bookkeeping.
    //   tBrB_bf16    : bf16, SAME COPY layout as tBrB_packed — dequant
    //                  writes here element-wise.
    //   tSrB         : bf16, MMA-B layout — reorder target; consumed by
    //                  cute::gemm.
    auto tBrB_packed = thr_copy_b.partition_sg_fragment_D(gBp(_, _, 0));
    cutlass::bfloat16_t tBrB_bf16_storage[tBrB_packed.size()];
    Tensor tBrB_bf16_rmem = make_tensor(make_rmem_ptr(tBrB_bf16_storage), tBrB_packed.layout());
    auto tBrB_bf16 = make_subgroup_tensor(tBrB_bf16_rmem, tBrB_packed.tv_layout());
    auto tSrB = thr_mma.partition_sg_fragment_B(gBp(_, _, 0));

    // Coord companion in the COPY layout — same partition as tBrB_packed.
    // Using partition_S on a coord gmem tile gives per-WI N/K coordinates
    // for every element of the copy-layout fragment, so the dequant loop
    // can resolve the correct scale byte per packed element.
    auto cBp_coord_tile = make_identity_tensor(make_shape(Int<BLK_N>{}, Int<BLK_K>{}));
    auto tCrB_coord = thr_copy_b.partition_S(cBp_coord_tile);

    // Per-SG scale register array — one byte per N-row in this SG's slice.
    uint8_t scale_sg[SG_N];

    // Partition C/D.
    SubgroupTensor tCrC = thr_mma.partition_sg_fragment_C(gD);

    using TD = typename DTensor::element_type;
    TD tCrD_final_frag[tCrC.size()];
    Tensor tCrD_final_tensor = make_tensor(make_rmem_ptr(tCrD_final_frag), tCrC.layout());
    SubgroupTensor tCrD_final_sg_tensor = make_subgroup_tensor(tCrD_final_tensor, tCrC.tv_layout());
    Tensor tCgD = thr_mma.partition_C(gD);

    // Prefetches for A and B (no prefetch for scales — direct loads are
    // small enough that the hit in the A/B-prefetch-hidden critical path
    // is negligible).
    auto prefetch_a = make_block_2d_prefetch(tiled_copy_a);
    auto prefetch_b = make_block_2d_prefetch(tiled_copy_b);

    auto pAgA = prefetch_a.get_slice(thr_id).partition_S(gA);
    auto pBgBp = prefetch_b.get_slice(thr_id).partition_S(gBp);

    constexpr int barrier_scope = 2;
    int k_start_idx = 0;
    int prefetch_k = k_start_idx;
    const int prefetch_dist = Stages;
    int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));

    CUTE_UNROLL
    for (; prefetch_k < prefetch_dist; ++prefetch_k) {
      prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
      prefetch(prefetch_b, pBgBp(_, _, _, prefetch_k));
    }

    for (int k_tile = k_start_idx; k_tile < k_tile_count; ++k_tile, ++prefetch_k) {
      barrier_arrive(barrier_scope);

      copy(tiled_copy_a, tAgA(_, _, _, k_tile), tArA);
      // Standard copy — copy-layout fragment as target (no retile_D).
      copy(tiled_copy_b, tBgBp(_, _, _, k_tile), tBrB_packed);
      load_scale_slice<SG_N, ATOM_N_V, BLK_N>(
          scales_gmem, scale_row_stride, wg_n, /*k_scale_idx=*/k_tile, thr_id, scale_sg);

      if (prefetch_k < k_tile_count) {
        prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
        prefetch(prefetch_b, pBgBp(_, _, _, prefetch_k));
      }

      reorder(tArA, tSrA);
      // Dequant in the copy-layout (linear index over tBrB_packed matches
      // tBrB_bf16 matches tCrB_coord), then reorder bf16 → MMA-B layout.
      const int sg_n_coord = (thr_id / SUBGROUP_SIZE) % ATOM_N_V;
      const int n_sg_base = sg_n_coord * SG_N;
      dequant_B_mxfp4_to_bf16(tBrB_packed, tCrB_coord, scale_sg, n_sg_base, tBrB_bf16);
      reorder(tBrB_bf16, tSrB);

      cute::gemm(mma, tSrA, tSrB, tCrC);
      barrier_wait(barrier_scope);
    }

    if constexpr (WithBias) {
      constexpr int BLK_M = get<0>(decltype(wg_tile){});
      add_bias<decltype(tCrC), BLK_M, BLK_N>(Bias, tCrC, mma, wg_n, thr_id);
    }

    reorder(tCrC, tCrD_final_sg_tensor);
    copy(tiled_copy_d, tCrD_final_sg_tensor, tCgD);
  }

  // -------------------------------------------------------------------------
  // Fused-activation path: two Bs (gate + up), two scale-B pointers.
  // -------------------------------------------------------------------------
  template <typename Coord>
  CUTLASS_DEVICE void operator()(
      ATensor& A,                       // (M,K)
      BPackedTensor& Bp0,               // (N/2, K)  float_e2m1_t
      BPackedTensor& Bp1,               // (N/2, K)  float_e2m1_t
      const uint8_t* scales0_gmem,      // UE8M0 gate scales
      const uint8_t* scales1_gmem,      // UE8M0 up scales
      int scale_row_stride,             // byte stride per scale N-row (same for both halves)
      DTensor& D,
      Coord blk_coord,
      TiledMMA mma,
      int thr_id,
      BiasTensor Bias0,
      BiasTensor Bias1,
      float gemm1_alpha,
      float gemm1_limit) {
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
    static_assert(BLK_K == MXFP4_GROUP_SIZE,
                  "MXFP4 mainloop assumes BLK_K == GROUP_SIZE == 32");

    constexpr int ATOM_N_V = get<2>(typename TiledMMA::ThrLayoutVMNK{}.shape());
    constexpr int SG_N = BLK_N / ATOM_N_V;
    constexpr int N_per_wi = SG_N / SUBGROUP_SIZE;
    static_assert(N_per_wi >= 1, "SG_N must be at least SUBGROUP_SIZE");

    Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(wg_m, _));
    Tensor gBp = local_tile(cBp0, select<1, 2>(wg_tile), make_coord(wg_n, _));
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
    auto tBgBp0 = thr_copy_b0.partition_S(gBp);
    auto tBgBp1 = thr_copy_b1.partition_S(gBp);

    auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
    auto tSrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));

    // Two packed-B fragments in the COPY layout, each with a bf16 shadow
    // in the same layout for dequant. tSrB is the MMA-B layout target
    // (shared across gate and up gemms since they run in series).
    auto tBrB_packed0 = thr_copy_b0.partition_sg_fragment_D(gBp(_, _, 0));
    auto tBrB_packed1 = thr_copy_b1.partition_sg_fragment_D(gBp(_, _, 0));
    cutlass::bfloat16_t tBrB_bf160_storage[tBrB_packed0.size()];
    cutlass::bfloat16_t tBrB_bf161_storage[tBrB_packed1.size()];
    Tensor tBrB_bf160_rmem = make_tensor(make_rmem_ptr(tBrB_bf160_storage), tBrB_packed0.layout());
    Tensor tBrB_bf161_rmem = make_tensor(make_rmem_ptr(tBrB_bf161_storage), tBrB_packed1.layout());
    auto tBrB_bf160 = make_subgroup_tensor(tBrB_bf160_rmem, tBrB_packed0.tv_layout());
    auto tBrB_bf161 = make_subgroup_tensor(tBrB_bf161_rmem, tBrB_packed1.tv_layout());
    auto tSrB = thr_mma.partition_sg_fragment_B(gBp(_, _, 0));

    // Coord companion in the COPY layout (same partition as tBrB_packed0).
    auto cBp_coord_tile = make_identity_tensor(make_shape(Int<BLK_N>{}, Int<BLK_K>{}));
    auto tCrB_coord = thr_copy_b0.partition_S(cBp_coord_tile);

    // Per-SG scale register arrays (gate + up).
    uint8_t scale0_sg[SG_N];
    uint8_t scale1_sg[SG_N];

    SubgroupTensor tCrC0 = thr_mma.partition_sg_fragment_C(gC0);
    SubgroupTensor tCrC1 = thr_mma.partition_sg_fragment_C(gC1);

    using TD = typename DTensor::element_type;
    TD tCrD_final_frag0[tCrC0.size()];
    Tensor tCrD_final_tensor0 = make_tensor(make_rmem_ptr(tCrD_final_frag0), tCrC0.layout());
    SubgroupTensor tCrD_final_sg_tensor0 = make_subgroup_tensor(tCrD_final_tensor0, tCrC0.tv_layout());
    TD tCrD_final_frag1[tCrC1.size()];
    Tensor tCrD_final_tensor1 = make_tensor(make_rmem_ptr(tCrD_final_frag1), tCrC1.layout());
    SubgroupTensor tCrD_final_sg_tensor1 = make_subgroup_tensor(tCrD_final_tensor1, tCrC1.tv_layout());

    Tensor tCgD = thr_mma.partition_C(gC0);

    auto prefetch_a = make_block_2d_prefetch(tiled_copy_a);
    auto prefetch_b0 = make_block_2d_prefetch(tiled_copy_b0);
    auto prefetch_b1 = make_block_2d_prefetch(tiled_copy_b1);

    auto pAgA = prefetch_a.get_slice(thr_id).partition_S(gA);
    auto pBgBp0 = prefetch_b0.get_slice(thr_id).partition_S(gBp);
    auto pBgBp1 = prefetch_b1.get_slice(thr_id).partition_S(gBp);

    constexpr int barrier_scope = 2;
    int k_start_idx = 0;
    int prefetch_k = k_start_idx;
    const int prefetch_dist = Stages;
    int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));

    CUTE_UNROLL
    for (; prefetch_k < prefetch_dist; ++prefetch_k) {
      prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
      prefetch(prefetch_b0, pBgBp0(_, _, _, prefetch_k));
      prefetch(prefetch_b1, pBgBp1(_, _, _, prefetch_k));
    }

    for (int k_tile = k_start_idx; k_tile < k_tile_count; ++k_tile, ++prefetch_k) {
      barrier_arrive(barrier_scope);

      copy(tiled_copy_a, tAgA(_, _, _, k_tile), tArA);
      const int sg_n_coord = (thr_id / SUBGROUP_SIZE) % ATOM_N_V;
      const int n_sg_base = sg_n_coord * SG_N;

      // GEMM0 (gate)
      copy(tiled_copy_b0, tBgBp0(_, _, _, k_tile), tBrB_packed0);
      load_scale_slice<SG_N, ATOM_N_V, BLK_N>(
          scales0_gmem, scale_row_stride, wg_n, k_tile, thr_id, scale0_sg);
      reorder(tArA, tSrA);
      dequant_B_mxfp4_to_bf16(tBrB_packed0, tCrB_coord, scale0_sg, n_sg_base, tBrB_bf160);
      reorder(tBrB_bf160, tSrB);
      cute::gemm(mma, tSrA, tSrB, tCrC0);

      // GEMM1 (up)
      copy(tiled_copy_b1, tBgBp1(_, _, _, k_tile), tBrB_packed1);
      load_scale_slice<SG_N, ATOM_N_V, BLK_N>(
          scales1_gmem, scale_row_stride, wg_n1, k_tile, thr_id, scale1_sg);
      dequant_B_mxfp4_to_bf16(tBrB_packed1, tCrB_coord, scale1_sg, n_sg_base, tBrB_bf161);
      reorder(tBrB_bf161, tSrB);
      cute::gemm(mma, tSrA, tSrB, tCrC1);

      if (prefetch_k < k_tile_count) {
        prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
        prefetch(prefetch_b0, pBgBp0(_, _, _, prefetch_k));
        prefetch(prefetch_b1, pBgBp1(_, _, _, prefetch_k));
      }

      barrier_wait(barrier_scope);
    }

    if constexpr (WithBias) {
      add_bias<decltype(tCrC0), BLK_M, BLK_N>(Bias0, tCrC0, mma, wg_n, thr_id);
      add_bias<decltype(tCrC1), BLK_M, BLK_N>(Bias1, tCrC1, mma, wg_n1, thr_id);
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tCrC0.size(); ++i) {
      float x = tCrC0(i);
      float y = tCrC1(i);
      float s;
      if constexpr (ActType == SILU) {
        s = 1.0f / (1.0f + sycl::native::exp(-x));
        tCrC0(i) = x * s * y;
      } else if constexpr (ActType == SWIGLU_GPT_OSS) {
        float gate = sycl::fmin(x, gemm1_limit);
        float up = sycl::fmax(-gemm1_limit, sycl::fmin(y, gemm1_limit));
        float t = gate * gemm1_alpha;
        s = 1.0f / (1.0f + sycl::native::exp(-t));
        tCrC0(i) = gate * s * (up + 1.0f);
      } else {  // GELU (tanh approx)
        constexpr float kBeta = 0.7978845608028654f;
        constexpr float kAlpha = 0.044715f;
        float x_cube = x * x * x;
        float tanh_arg = kBeta * (x + kAlpha * x_cube);
        s = 0.5f * (1.0f + std::tanh(tanh_arg));
        tCrC0(i) = x * s * y;
      }
    }

    reorder(tCrC0, tCrD_final_sg_tensor0);
    copy(tiled_copy_d, tCrD_final_sg_tensor0, tCgD);
  }

  // Bias helper (identical to BF16 mainloop).
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

}  // namespace MoE_MXFP4
