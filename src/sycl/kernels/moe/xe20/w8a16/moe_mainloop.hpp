/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

// FP8 E4M3 weight, BF16 activation grouped-GEMM mainloop for Xe2 (BMG).
// Weight scales are either one scalar per expert/projection or one value per
// 128x128 weight block.

#pragma once

#include <cute/tensor.hpp>
#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include "../common/scale.hpp"
#include "cutlass/float8.h"
#include "cutlass/half.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"
#include "cutlass/tensor_ref.h"
#include "sycl/SYCLHelpers.h"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace moe_w8a16 {

using namespace cute;

static constexpr int FP8_GROUP_SIZE_K = 128;

// Number of work-items per subgroup on Xe (SIMD lane count).
static constexpr int SUBGROUP_SIZE = 16;

template <int Stages>
class W8A16MainloopPolicy {};

template <
    class DispatchPolicy_,
    class TiledCopyA_,
    class TiledCopyBPacked_,
    class TiledCopyD_,
    class ATensor_,
    class BPackedTensor_,
    class DTensor_,
    class TiledMMA_,
    bool WeightScalePerExpert = false,
    bool WeightScaleBlocked = false>
struct Fp8W8A16Mainloop {
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
    class TiledMMA_,
    bool WeightScalePerExpert,
    bool WeightScaleBlocked>
struct Fp8W8A16Mainloop<
    W8A16MainloopPolicy<Stages>,
    TiledCopyA_,
    TiledCopyBPacked_,
    TiledCopyD_,
    ATensor_,
    BPackedTensor_,
    DTensor_,
    TiledMMA_,
    WeightScalePerExpert,
    WeightScaleBlocked> {
  using TiledMMA = TiledMMA_;
  using TiledCopyA = TiledCopyA_;
  using TiledCopyBPacked = TiledCopyBPacked_;
  using TiledCopyD = TiledCopyD_;
  using ATensor = ATensor_;
  using BPackedTensor = BPackedTensor_;
  using DTensor = DTensor_;

  Fp8W8A16Mainloop() {}

  template <typename Coord>
  CUTLASS_DEVICE void run_w8a16_block(
      ATensor& A,
      BPackedTensor& Bp,
      const float* w_scale_gmem,
      int w_scale_row_stride,
      DTensor& D,
      Coord blk_coord,
      TiledMMA mma,
      int thr_id,
      const float* Bias,
      int gemm_n) {
    auto wg_m = get<0>(blk_coord);
    auto wg_n = get<1>(blk_coord);
    auto wg_tile = mma.tile_mnk();
    auto wg_coord = make_coord(wg_m, wg_n, 0);
    constexpr int BLK_M = get<0>(decltype(wg_tile){});
    constexpr int BLK_N = get<1>(decltype(wg_tile){});
    constexpr int BLK_K = get<2>(decltype(wg_tile){});
    constexpr int ATOM_M_V = get<1>(typename TiledMMA::ThrLayoutVMNK{}.shape());
    constexpr int ATOM_N_V = get<2>(typename TiledMMA::ThrLayoutVMNK{}.shape());
    constexpr int SG_M = BLK_M / ATOM_M_V;
    constexpr int SG_N = BLK_N / ATOM_N_V;
    constexpr int N_ATOMS = SG_N / SUBGROUP_SIZE;
    constexpr int RELOAD_CADENCE = FP8_GROUP_SIZE_K / BLK_K;
    static_assert(
        RELOAD_CADENCE == 4 || RELOAD_CADENCE == 8,
        "W8A16 block fast path supports K tiles of 32 or 16 per scale group");
    static_assert(BLK_N <= 128, "W8A16 block fast path requires one N tile per weight-scale block");

    Tensor cA = make_identity_tensor(A.shape());
    Tensor cBp = make_identity_tensor(Bp.shape());
    Tensor cD = make_identity_tensor(D.shape());
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
    using CopyAFragment = decltype(thr_copy_a.partition_sg_fragment_D(gA(_, _, 0)));
    using CopyBFragment = decltype(thr_copy_b.partition_sg_fragment_D(gBp(_, _, 0)));
    using MmaAFragment = decltype(thr_mma.partition_sg_fragment_A(gA(_, _, 0)));
    using MmaBFragment = decltype(thr_mma.partition_sg_fragment_B(gBp(_, _, 0)));
    CopyAFragment tArA_packed = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
    CopyBFragment tBrB_packed = thr_copy_b.partition_sg_fragment_D(gBp(_, _, 0));
    MmaAFragment tSrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
    MmaBFragment tSrB = thr_mma.partition_sg_fragment_B(gBp(_, _, 0));
    SubgroupTensor tCrC = thr_mma.partition_sg_fragment_C(gD);
    cute::clear(tCrC);

    auto prefetch_a = make_block_2d_prefetch(tiled_copy_a);
    auto prefetch_b = make_block_2d_prefetch(tiled_copy_b);
    auto pAgA = prefetch_a.get_slice(thr_id).partition_S(gA);
    auto pBgBp = prefetch_b.get_slice(thr_id).partition_S(gBp);
    constexpr SPIRVScope barrier_scope = ScopeWorkgroup;
    const int k_tile_count = ceil_div(shape<1>(A), BLK_K);
    const int full_group_count = k_tile_count / RELOAD_CADENCE;
    CUTE_UNROLL
    for (int prefetch_k = 0; prefetch_k < Stages; ++prefetch_k) {
      if (prefetch_k < k_tile_count) {
        prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
        prefetch(prefetch_b, pBgBp(_, _, _, prefetch_k));
      }
    }

    float w_scale = 1.0f;
    const int scale_n = (wg_n * BLK_N) / 128;
    auto load_group_scale = [&](int group) { w_scale = w_scale_gmem[scale_n * w_scale_row_stride + group]; };
    load_group_scale(0);

    auto run_k_tile = [&](int k_tile) {
      barrier_arrive(barrier_scope);
      copy(tiled_copy_a, tAgA(_, _, _, k_tile), tArA_packed);
      copy(tiled_copy_b, tBgBp(_, _, _, k_tile), tBrB_packed);
      const int prefetch_idx = k_tile + Stages;
      if (prefetch_idx < k_tile_count) {
        prefetch(prefetch_a, pAgA(_, _, _, prefetch_idx));
        prefetch(prefetch_b, pBgBp(_, _, _, prefetch_idx));
      }
      reorder(tArA_packed, tSrA);
      reorder(tBrB_packed, tSrB);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tSrB.size(); ++i) {
        tSrB(i) = moe_xe20::apply_bf16_or_fp16_scale(tSrB(i), w_scale);
      }
      cute::gemm(mma, tSrA, tSrB, tCrC);
      barrier_wait(barrier_scope);
    };

    for (int group = 0; group < full_group_count; ++group) {
      CUTE_UNROLL
      for (int group_offset = 0; group_offset < RELOAD_CADENCE; ++group_offset) {
        run_k_tile(group * RELOAD_CADENCE + group_offset);
      }
      if (group + 1 < full_group_count) {
        load_group_scale(group + 1);
      }
    }

    if (Bias != nullptr) {
      add_bias<SG_M, SG_N, BLK_N>(Bias, tCrC, wg_n, thr_id, gemm_n);
    }
    SubgroupTensor tCrD = thr_copy_d.partition_sg_fragment_S(gD);
    Tensor tCgD = thr_copy_d.partition_D(gD);
    reorder(tCrC, tCrD);
    copy(tiled_copy_d, tCrD, tCgD);
  }

  template <typename Coord>
  CUTLASS_DEVICE void run_w8a16_scalar(
      ATensor& A,
      BPackedTensor& Bp,
      const float* w_scale_gmem,
      int weight_scale_count,
      DTensor& D,
      Coord blk_coord,
      TiledMMA mma,
      int thr_id,
      const float* Bias,
      int gemm_n) {
    auto wg_m = get<0>(blk_coord);
    auto wg_n = get<1>(blk_coord);
    auto wg_tile = mma.tile_mnk();
    auto wg_coord = make_coord(wg_m, wg_n, 0);
    constexpr int BLK_M = get<0>(decltype(wg_tile){});
    constexpr int BLK_N = get<1>(decltype(wg_tile){});
    constexpr int ATOM_M_V = get<1>(typename TiledMMA::ThrLayoutVMNK{}.shape());
    constexpr int ATOM_N_V = get<2>(typename TiledMMA::ThrLayoutVMNK{}.shape());
    constexpr int SG_M = BLK_M / ATOM_M_V;

    Tensor cA = make_identity_tensor(A.shape());
    Tensor cBp = make_identity_tensor(Bp.shape());
    Tensor cD = make_identity_tensor(D.shape());
    Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(wg_m, _));
    Tensor gBp = local_tile(cBp, select<1, 2>(wg_tile), make_coord(wg_n, _));
    Tensor gD = local_tile(cD, wg_tile, wg_coord, Step<_1, _1, X>{});

    TiledCopyA tiled_copy_a{A};
    TiledCopyBPacked tiled_copy_b{Bp};
    auto thr_copy_a = tiled_copy_a.get_slice(thr_id);
    auto thr_copy_b = tiled_copy_b.get_slice(thr_id);
    auto thr_mma = mma.get_slice(thr_id);
    auto tAgA = thr_copy_a.partition_S(gA);
    auto tBgBp = thr_copy_b.partition_S(gBp);
    using CopyAFragment = decltype(thr_copy_a.partition_sg_fragment_D(gA(_, _, 0)));
    using CopyBFragment = decltype(thr_copy_b.partition_sg_fragment_D(gBp(_, _, 0)));
    using MmaAFragment = decltype(thr_mma.partition_sg_fragment_A(gA(_, _, 0)));
    using MmaBFragment = decltype(thr_mma.partition_sg_fragment_B(gBp(_, _, 0)));
    CopyAFragment tArA_packed = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
    CopyBFragment tBrB_packed = thr_copy_b.partition_sg_fragment_D(gBp(_, _, 0));
    MmaAFragment tSrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
    MmaBFragment tSrB = thr_mma.partition_sg_fragment_B(gBp(_, _, 0));
    SubgroupTensor tCrC = thr_mma.partition_sg_fragment_C(gD);
    cute::clear(tCrC);

    auto prefetch_a = make_block_2d_prefetch(tiled_copy_a);
    auto prefetch_b = make_block_2d_prefetch(tiled_copy_b);
    auto pAgA = prefetch_a.get_slice(thr_id).partition_S(gA);
    auto pBgBp = prefetch_b.get_slice(thr_id).partition_S(gBp);
    constexpr SPIRVScope barrier_scope = ScopeWorkgroup;
    const int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));
    int k_tile_prefetch = 0;

    CUTE_UNROLL
    for (; k_tile_prefetch < Stages; ++k_tile_prefetch) {
      if (k_tile_prefetch < k_tile_count) {
        prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
        prefetch(prefetch_b, pBgBp(_, _, _, k_tile_prefetch));
      }
    }

    for (int k_tile = 0; k_tile < k_tile_count; ++k_tile, ++k_tile_prefetch) {
      barrier_arrive(barrier_scope);

      copy(tiled_copy_a, tAgA(_, _, _, k_tile), tArA_packed);
      copy(tiled_copy_b, tBgBp(_, _, _, k_tile), tBrB_packed);

      if (k_tile_prefetch < k_tile_count) {
        prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
        prefetch(prefetch_b, pBgBp(_, _, _, k_tile_prefetch));
      }

      reorder(tArA_packed, tSrA);
      reorder(tBrB_packed, tSrB);
      cute::gemm(mma, tSrA, tSrB, tCrC);

      barrier_wait(barrier_scope);
    }

    constexpr int SG_N = BLK_N / ATOM_N_V;
    auto sg_local_n_coord = cutlass::get_sub_group_id() % ATOM_N_V;
    int sg_local_id = cutlass::get_sub_group_local_id();
    constexpr int sg_local_range = 16;
    int n_tile_start = wg_n * BLK_N;
    int n_sg_start = sg_local_n_coord * SG_N;
    CUTLASS_PRAGMA_UNROLL
    for (int sn = 0; sn < SG_N / sg_local_range; ++sn) {
      int global_n = n_tile_start + n_sg_start + sn * sg_local_range + sg_local_id;
      float weight_scale = w_scale_gmem[weight_scale_count == 2 && global_n >= gemm_n / 2 ? 1 : 0];
      CUTLASS_PRAGMA_UNROLL
      for (int sm = 0; sm < SG_M; ++sm) {
        tCrC(sn * SG_M + sm) *= weight_scale;
      }
    }
    if (Bias != nullptr) {
      add_bias<SG_M, SG_N, BLK_N>(Bias, tCrC, wg_n, thr_id, gemm_n);
    }
    TiledCopyD tiled_copy_d{D};
    auto thr_copy_d = tiled_copy_d.get_slice(thr_id);
    SubgroupTensor tCrD = thr_copy_d.partition_sg_fragment_S(gD);
    Tensor tCgD = thr_copy_d.partition_D(gD);
    reorder(tCrC, tCrD);
    copy(tiled_copy_d, tCrD, tCgD);
  }

  template <typename Coord>
  CUTLASS_DEVICE void operator()(
      ATensor& A,
      BPackedTensor& Bp,
      const float* w_scale_gmem,
      int w_scale_row_stride,
      DTensor& D,
      Coord blk_coord,
      TiledMMA mma,
      int thr_id,
      const float* Bias,
      int gemm_n) {
    if constexpr (WeightScalePerExpert) {
      run_w8a16_scalar(A, Bp, w_scale_gmem, w_scale_row_stride, D, blk_coord, mma, thr_id, Bias, gemm_n);
    } else {
      run_w8a16_block(A, Bp, w_scale_gmem, w_scale_row_stride, D, blk_coord, mma, thr_id, Bias, gemm_n);
    }
  }

  template <int SG_M, int SG_N, int BLK_N, typename tCrC_t>
  void add_bias(const float* Bias, tCrC_t& tCrC, int wg_n, int thr_id, int gemm_n) {
    static constexpr auto ATOM_N = get<2>(typename TiledMMA::ThrLayoutVMNK{}.shape());
    constexpr int N_ATOMS = SG_N / SUBGROUP_SIZE;
    int sg_local_n_coord = (thr_id / SUBGROUP_SIZE) % ATOM_N;
    int lane = thr_id % SUBGROUP_SIZE;

    CUTLASS_PRAGMA_UNROLL
    for (int na = 0; na < N_ATOMS; ++na) {
      int n = wg_n * BLK_N + sg_local_n_coord * SG_N + na * SUBGROUP_SIZE + lane;
      float bias = (n < gemm_n) ? Bias[n] : 0.0f;
      CUTLASS_PRAGMA_UNROLL
      for (int sm = 0; sm < SG_M; ++sm) {
        tCrC(na * SG_M + sm) += bias;
      }
    }
  }
};

}  // namespace moe_w8a16
