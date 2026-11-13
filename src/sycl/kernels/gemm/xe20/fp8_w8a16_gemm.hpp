/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

// Dense FP8 E4M3 weight, BF16 activation GEMM for Xe2 (BMG) with 128x128 FP32 block scales:
// D[M, N] = A[M, K] @ (B[N, K] * S[N / 128, K / 128])^T.
//
// Decode-sized GEMMs have too few (M, N) tiles to fill the GPU, so each work-group runs KSlices
// copies of the subgroup layout over disjoint K ranges and reduces their accumulators through SLM.

#pragma once

#include <cute/tensor.hpp>
#include <sycl/sycl.hpp>

#include "../../moe/xe20/common/block_2d_copy_d.hpp"
#include "../../moe/xe20/common/scale.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/float8.h"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace fp8_w8a16_xe20 {
using namespace cute;

static constexpr int kScaleBlock = 128;

template <class TiledMMA, class TensorA, class TensorB, class TensorD, int KSlices>
struct DenseW8A16Kernel {
  using ElementA = cutlass::bfloat16_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementD = cutlass::bfloat16_t;
  using TiledCopyA = decltype(make_block_2d_copy_A(TiledMMA{}, TensorA{}));
  using TiledCopyB = decltype(make_block_2d_copy_B(TiledMMA{}, TensorB{}));
  using TiledCopyD = decltype(moe_xe20::make_moe_block_2d_copy_D<void>(TiledMMA{}, TensorD{}));

  static constexpr int Stages = 3;
  static constexpr int SliceThreads = size(TiledMMA{});
  static constexpr int WGThreads = SliceThreads * KSlices;
  using WGTile = decltype(TiledMMA{}.tile_mnk());
  static constexpr int BLK_M = get<0>(WGTile{});
  static constexpr int BLK_N = get<1>(WGTile{});
  static constexpr int BLK_K = get<2>(WGTile{});
  static constexpr int TilesPerScaleGroup = kScaleBlock / BLK_K;
  static_assert(kScaleBlock % BLK_K == 0, "K tile must divide the 128-wide scale block");
  static_assert(kScaleBlock % BLK_N == 0, "N tile must stay inside one 128-row scale block");

  // Accumulator floats per work-item; the SLM reduction buffer holds them for slices 1..KSlices-1.
  static constexpr int SG_M = BLK_M / get<1>(typename TiledMMA::ThrLayoutVMNK{}.shape());
  static constexpr int SG_N = BLK_N / get<2>(typename TiledMMA::ThrLayoutVMNK{}.shape());
  static constexpr int AccPerThread = SG_M * SG_N / 16;
  static constexpr int SlmFloats = KSlices > 1 ? (KSlices - 1) * SliceThreads* AccPerThread : 1;

  struct Params {
    const ElementA* A;
    const ElementB* B;
    const float* scales;  // [ceil(N / 128), K / 128], row-major
    ElementD* D;
    int M;
    int N;
    int K;
    int ld_a;
    int ld_b;
    int ld_d;
    int ld_s;
  };

  CUTLASS_DEVICE void operator()(Params const& p, sycl::nd_item<3> item, float* slm) const {
    TiledMMA mma{};
    const int local_id = static_cast<int>(item.get_local_linear_id());
    const int slice = local_id / SliceThreads;
    const int thr_id = local_id % SliceThreads;
    const int n_tiles = ceil_div(p.N, BLK_N);
    const int wg_id = static_cast<int>(item.get_group_linear_id());
    const int wg_m = wg_id / n_tiles;
    const int wg_n = wg_id % n_tiles;

    auto A = make_tensor(make_gmem_ptr(p.A), make_layout(make_shape(p.M, p.K), make_stride(p.ld_a, _1{})));
    auto B = make_tensor(make_gmem_ptr(p.B), make_layout(make_shape(p.N, p.K), make_stride(p.ld_b, _1{})));
    auto D = make_tensor(make_gmem_ptr(p.D), make_layout(make_shape(p.M, p.N), make_stride(p.ld_d, _1{})));

    auto wg_tile = mma.tile_mnk();
    Tensor cA = make_identity_tensor(A.shape());
    Tensor cB = make_identity_tensor(B.shape());
    Tensor cD = make_identity_tensor(D.shape());
    Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(wg_m, _));
    Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(wg_n, _));
    Tensor gD = local_tile(cD, wg_tile, make_coord(wg_m, wg_n, 0), Step<_1, _1, X>{});

    TiledCopyA tiled_copy_a{A};
    TiledCopyB tiled_copy_b{B};
    auto thr_copy_a = tiled_copy_a.get_slice(thr_id);
    auto thr_copy_b = tiled_copy_b.get_slice(thr_id);
    auto thr_mma = mma.get_slice(thr_id);
    auto tAgA = thr_copy_a.partition_S(gA);
    auto tBgB = thr_copy_b.partition_S(gB);
    auto tArA_packed = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
    auto tBrB_packed = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));
    auto tSrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
    auto tSrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));
    SubgroupTensor tCrC = thr_mma.partition_sg_fragment_C(gD);
    cute::clear(tCrC);

    auto prefetch_a = make_block_2d_prefetch(tiled_copy_a);
    auto prefetch_b = make_block_2d_prefetch(tiled_copy_b);
    auto pAgA = prefetch_a.get_slice(thr_id).partition_S(gA);
    auto pBgB = prefetch_b.get_slice(thr_id).partition_S(gB);

    // The host picks KSlices dividing K / 128, so every slice runs the same trip count and the
    // work-group-scope split barrier below stays matched.
    const int groups_per_slice = (p.K / kScaleBlock) / KSlices;
    const int group_begin = slice * groups_per_slice;
    const int k_tile_begin = group_begin * TilesPerScaleGroup;
    const int k_tile_end = k_tile_begin + groups_per_slice * TilesPerScaleGroup;

    CUTE_UNROLL
    for (int s = 0; s < Stages; ++s) {
      if (k_tile_begin + s < k_tile_end) {
        prefetch(prefetch_a, pAgA(_, _, _, k_tile_begin + s));
        prefetch(prefetch_b, pBgB(_, _, _, k_tile_begin + s));
      }
    }

    const float* scale_row = p.scales + static_cast<int64_t>((wg_n * BLK_N) / kScaleBlock) * p.ld_s;
    constexpr SPIRVScope barrier_scope = ScopeWorkgroup;
    for (int g = 0; g < groups_per_slice; ++g) {
      const float w_scale = scale_row[group_begin + g];
      CUTE_UNROLL
      for (int t = 0; t < TilesPerScaleGroup; ++t) {
        const int k_tile = k_tile_begin + g * TilesPerScaleGroup + t;
        barrier_arrive(barrier_scope);
        copy(tiled_copy_a, tAgA(_, _, _, k_tile), tArA_packed);
        copy(tiled_copy_b, tBgB(_, _, _, k_tile), tBrB_packed);
        if (k_tile + Stages < k_tile_end) {
          prefetch(prefetch_a, pAgA(_, _, _, k_tile + Stages));
          prefetch(prefetch_b, pBgB(_, _, _, k_tile + Stages));
        }
        reorder(tArA_packed, tSrA);
        reorder(tBrB_packed, tSrB);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrB.size(); ++i) {
          tSrB(i) = moe_xe20::apply_bf16_or_fp16_scale(tSrB(i), w_scale);
        }
        cute::gemm(mma, tSrA, tSrB, tCrC);
        barrier_wait(barrier_scope);
      }
    }

    if constexpr (KSlices > 1) {
      // Work-items with the same thr_id in different slices own the same accumulator elements.
      if (slice > 0) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < AccPerThread; ++i) {
          slm[((slice - 1) * AccPerThread + i) * SliceThreads + thr_id] = tCrC(i);
        }
      }
      sycl::group_barrier(item.get_group());
      if (slice > 0) {
        return;
      }
      CUTLASS_PRAGMA_UNROLL
      for (int s = 1; s < KSlices; ++s) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < AccPerThread; ++i) {
          tCrC(i) += slm[((s - 1) * AccPerThread + i) * SliceThreads + thr_id];
        }
      }
    }

    TiledCopyD tiled_copy_d{D};
    auto thr_copy_d = tiled_copy_d.get_slice(thr_id);
    SubgroupTensor tCrD = thr_copy_d.partition_sg_fragment_S(gD);
    Tensor tCgD = thr_copy_d.partition_D(gD);
    reorder(tCrC, tCrD);
    copy(tiled_copy_d, tCrD, tCgD);
  }
};

}  // namespace fp8_w8a16_xe20
