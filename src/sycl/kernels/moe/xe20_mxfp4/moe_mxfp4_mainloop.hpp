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
//   - B is stored in GMEM as packed MXFP4 (int8 bytes, two E2M1 nibbles each,
//     low nibble = smaller-K element).
//   - A separate collaborative load brings in [SG_N] fp32 scale elements per
//     k-tile. BLK_K is hardcoded to MXFP4 GroupSize=32 (the only E2M1 group
//     size the hardware cares about) so each k-tile sees exactly one scale
//     element per N-row, which simplifies the dequant loop. The producer
//     decodes UE8M0 bytes to fp32 direct multipliers ahead of the kernel.
//   - In registers, each work-item unpacks nibbles into a 16-entry signed
//     bf16 LUT and multiplies by the bf16 multiplier (fp32 scale cast to
//     bf16 once per SG per k-tile). The result lands in a bf16 MMA-B
//     fragment consumed by cute::gemm unchanged.
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

// MXFP4 invariant: one scale value per 32 consecutive K-elements.
static constexpr int MXFP4_GROUP_SIZE = 32;

// Number of work-items per subgroup on Xe (SIMD lane count).
static constexpr int SUBGROUP_SIZE = 16;

// Hand-rolled scale load.
//
// The 2D-block-load machinery doesn't fit a (BLK_N, 1)-shaped scale tile
// cleanly, so we load scales with plain pointer arithmetic. Scales are tiny
// (one fp32 element per N-row per k-tile — at most a few elements per WI
// per k-tile), so we skip prefetch and accept the slow path; the latency
// is fully hidden under the A/B tile loads which are orders of magnitude
// bigger.
//
// Per-WI N-to-element mapping matches add_bias: WI at (sg_n_coord, lane)
// owns N-positions { sn*SUBGROUP_SIZE + lane : sn ∈ [0, SG_N/16) }, with
// SG starting at n_base = wg_n * BLK_N + sg_n_coord * SG_N. This matches
// the MMA-B fragment's per-WI N layout so dequant can index scales by the
// same `n` value as its N-loop.
//
// scales_gmem_ptr points at the start of this expert's scales
// ([N, K/GROUP_SIZE] fp32, row-stride = K/GROUP_SIZE).
// k_scale_idx is the K-group index for this k-tile (= k_tile since we
// force BLK_K == GROUP_SIZE).
// Collaborative scale load. Each WI in the SG reads SG_N / SUBGROUP_SIZE
// fp32 elements from gmem, then the SG uses sycl::select_from_group to
// broadcast each WI's local elements into every other WI's full SG_N
// array. Total gmem traffic per SG per k-tile: SG_N * 4 bytes (coalesced)
// instead of 16*SG_N*4 with the old naive approach.
//
// The broadcast is done as SG_N separate select_from_group ops — one per
// element position in the output array. Each call asks the SG for the
// value stored in lane `src_lane` at local array slot `local_slot`, which
// is cheap on Xe (single sub-group exchange instruction).
template <int SG_N, int ATOM_N, int BLK_N>
CUTLASS_DEVICE void load_scale_slice(
    const float* scales_gmem_ptr,
    int row_stride,
    int wg_n,
    int k_scale_idx,
    int thr_id,
    float* scale_out /* [SG_N] */) {
  static constexpr int N_per_wi = SG_N / SUBGROUP_SIZE;
  static_assert(SG_N % SUBGROUP_SIZE == 0,
                "SG_N must be a multiple of SUBGROUP_SIZE");

  const int sg_n_coord = (thr_id / SUBGROUP_SIZE) % ATOM_N;
  const int lane = thr_id % SUBGROUP_SIZE;
  const int n_base = wg_n * BLK_N + sg_n_coord * SG_N;

  // Each WI loads its own N_per_wi fp32 elements from gmem (its assigned
  // rows interleaved by lane: WI `lane` owns rows
  // { lane*N_per_wi + k : k<N_per_wi }). Matches the add_bias per-WI N
  // mapping convention.
  float wi_local[N_per_wi];
  CUTE_UNROLL
  for (int sn = 0; sn < N_per_wi; ++sn) {
    const int n = n_base + lane * N_per_wi + sn;
    wi_local[sn] = scales_gmem_ptr[n * row_stride + k_scale_idx];
  }

  // Broadcast each lane's elements to all SG lanes, materializing the full
  // SG_N fp32 scale array in every WI's registers.
  auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
  CUTE_UNROLL
  for (int src_lane = 0; src_lane < SUBGROUP_SIZE; ++src_lane) {
    CUTE_UNROLL
    for (int sn = 0; sn < N_per_wi; ++sn) {
      scale_out[src_lane * N_per_wi + sn] =
          sycl::select_from_group(sg, wi_local[sn], src_lane);
    }
  }
}

// Apply per-block MXFP4 scales to an already-upcasted bf16 MMA-B fragment.
// Used after cute::reorder does the E2M1→bf16 conversion + layout permute.
//
// The N-row for each fragment element comes from the MMA-B coord companion:
// get<0>(coord_frag(idx)) is the WG-tile-N coord. We subtract n_sg_base to
// index into scale_bf16, which is the fp32→bf16-cast SG-local scale array.
template <int SG_N_v, class FragBf16, class CoordFrag>
CUTLASS_DEVICE void apply_B_scales_mma(
    FragBf16& frag,
    CoordFrag const& coord_frag,
    const float* scales_sg,
    int n_sg_base) {
  using FragLayout = typename FragBf16::layout_type;
  constexpr int frag_size = cute::size_v<FragLayout>;

  // Cast SG_N fp32 scales → bf16 multipliers once; amortized across every
  // fragment element referencing the same N-row.
  cutlass::bfloat16_t scale_bf16[SG_N_v];
  CUTE_UNROLL
  for (int i = 0; i < SG_N_v; ++i) {
    scale_bf16[i] = cutlass::bfloat16_t(scales_sg[i]);
  }

  CUTE_UNROLL
  for (int idx = 0; idx < frag_size; ++idx) {
    auto coord = coord_frag(idx);
    int n_in_sg = get<0>(coord) - n_sg_base;
    cutlass::bfloat16_t s = scale_bf16[n_in_sg];
    frag(idx) = frag(idx) * s;
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
      const float* scales_gmem,        // fp32 scale buffer (rows span N via scale_row_stride)
      int scale_row_stride,            // fp32 stride per scale N-row
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

    // B fragments — fused reorder path that collapses dequant+reorder into
    // a single cute::reorder call with built-in E2M1→bf16 conversion:
    //   tBrB_packed   : float_e2m1_t, COPY layout — destination of the
    //                   8-bit 2D-load.
    //   tSrB          : bf16, MMA-B layout — reorder produces this;
    //                   element-wise scale is applied in-place here.
    // We no longer allocate an intermediate bf16 fragment in the copy
    // layout; cute::reorder handles both the E2M1→bf16 conversion and
    // the copy-layout→MMA-layout permutation via its ConvertRelayout
    // dispatch path.
    auto tBrB_packed = thr_copy_b.partition_sg_fragment_D(gBp(_, _, 0));
    auto tSrB = thr_mma.partition_sg_fragment_B(gBp(_, _, 0));

    // Coord companion in the MMA-B layout — maps each tSrB element's
    // linear index to its (n, k) coord for scale indexing.
    auto cBp_coord_tile = make_identity_tensor(make_shape(Int<BLK_N>{}, Int<BLK_K>{}));
    auto tCrB_coord = thr_mma.partition_B(cBp_coord_tile);

    // Per-SG scale register array — one fp32 value per N-row in this SG's slice.
    float scale_sg[SG_N];

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
      // Fused type-conversion + layout-reorder. This picks the
      // ConvertRelayout dispatch inside cute::reorder, which handles both
      // E2M1 → bf16 upcast and copy-layout → MMA-B layout permutation.
      reorder(tBrB_packed, tSrB);

      // Apply per-block MXFP4 scales to tSrB in-place. Each element's
      // N-coord is looked up in the MMA-B coord companion; scale_sg holds
      // the per-SG fp32 scales are cast to bf16 in apply_B_scales_mma.
      const int sg_n_coord = (thr_id / SUBGROUP_SIZE) % ATOM_N_V;
      const int n_sg_base = sg_n_coord * SG_N;
      apply_B_scales_mma<SG_N>(tSrB, tCrB_coord, scale_sg, n_sg_base);

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
      const float* scales0_gmem,        // fp32 gate scales
      const float* scales1_gmem,        // fp32 up scales
      int scale_row_stride,             // fp32 stride per scale N-row (same for both halves)
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

    // Two packed-B fragments in the COPY layout; tSrB is the MMA-B layout
    // target, shared across gate and up gemms (they run in series).
    // cute::reorder with the ConvertRelayout dispatch does both E2M1→bf16
    // conversion and layout permute in one call — no copy-layout bf16
    // intermediate needed.
    auto tBrB_packed0 = thr_copy_b0.partition_sg_fragment_D(gBp(_, _, 0));
    auto tBrB_packed1 = thr_copy_b1.partition_sg_fragment_D(gBp(_, _, 0));
    auto tSrB = thr_mma.partition_sg_fragment_B(gBp(_, _, 0));

    // Coord companion in the MMA-B layout for scale indexing on tSrB.
    auto cBp_coord_tile = make_identity_tensor(make_shape(Int<BLK_N>{}, Int<BLK_K>{}));
    auto tCrB_coord = thr_mma.partition_B(cBp_coord_tile);

    // Per-SG scale register arrays (gate + up).
    float scale0_sg[SG_N];
    float scale1_sg[SG_N];

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
      reorder(tBrB_packed0, tSrB);
      apply_B_scales_mma<SG_N>(tSrB, tCrB_coord, scale0_sg, n_sg_base);
      cute::gemm(mma, tSrA, tSrB, tCrC0);

      // GEMM1 (up)
      copy(tiled_copy_b1, tBgBp1(_, _, _, k_tile), tBrB_packed1);
      load_scale_slice<SG_N, ATOM_N_V, BLK_N>(
          scales1_gmem, scale_row_stride, wg_n1, k_tile, thr_id, scale1_sg);
      reorder(tBrB_packed1, tSrB);
      apply_B_scales_mma<SG_N>(tSrB, tCrB_coord, scale1_sg, n_sg_base);
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
