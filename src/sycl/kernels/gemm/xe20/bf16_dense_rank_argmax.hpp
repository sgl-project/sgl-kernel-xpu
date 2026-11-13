/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

// BF16 dense GEMM with an exact caller-provided comparison-class epilogue.
// Tile128x128x32, subgroup grid4x4.
#pragma once
#include <cstdint>

#include "../../moe/xe20/bf16/moe_kernel.hpp"

// The mainloop below is derived from the original MoEMainloop single-GEMM
// implementation. Input copy/DPAS ordering is unchanged; the output store is
// replaced by BF16 rounding, rank lookup, and compact subgroup reductions.
namespace sgl::gemm {

template <class Base, class MMA, class Element>
struct Bf16DenseRankPartials {
  struct Params {
    const Element* a;
    const Element* w;
    const std::uint16_t* ranks;
    std::uint32_t* partial;
    int m, n, k;
    MMA mma;
  };
  using Mainloop = typename Base::CollectiveMainloop;

  void operator()(const Params& p, sycl::nd_item<3> item, std::uint32_t* scratch) {
    using namespace cute;
    constexpr int BM = 128, BN = 128, NSG = 4;
    static_assert(int(get<0>(MMA{}.tile_mnk())) == BM);
    static_assert(int(get<1>(MMA{}.tile_mnk())) == BN);
    const int tiles_m = (p.m + BM - 1) / BM, tiles_n = (p.n + BN - 1) / BN;
    const int tid = int(item.get_local_linear_id());
    auto sg = item.get_sub_group();
    const int lane = int(sg.get_local_linear_id());
    const int sgid = tid / 16, sg_m = sgid / NSG, sg_n = sgid % NSG;
    auto A = make_tensor(
        make_gmem_ptr(const_cast<Element*>(p.a)), make_layout(make_shape(p.m, p.k), make_stride(p.k, _1{})));
    Base base;
    auto bs = base.make_B_tensors(const_cast<Element*>(p.w), p.n, p.k, p.k);
    auto B = get<0>(bs);
    Tensor cA = make_identity_tensor(A.shape());
    Tensor cB = make_identity_tensor(B.shape());
    Tensor cD = make_identity_tensor(make_shape(p.m, p.n));
    auto tile = p.mma.tile_mnk();
    typename Mainloop::TiledCopyA copy_a{A};
    typename Mainloop::TiledCopyB copy_b{B};
    auto thr_a = copy_a.get_slice(tid);
    auto thr_b = copy_b.get_slice(tid);
    auto thr_mma = p.mma.get_slice(tid);
    auto prefetch_a = make_block_2d_prefetch(copy_a);
    auto prefetch_b = make_block_2d_prefetch(copy_b);
    const int ktiles = ceil_div(shape<1>(A), get<2>(tile));
    constexpr SPIRVScope barrier_scope = ScopeWorkgroup;
    for (std::int64_t task = item.get_group_linear_id(); task < tiles_m * tiles_n; task += item.get_group_range(1)) {
      // The launcher bounds valid tile indices to int32. Keep tile-coordinate
      // division in that range, but widen the increment past the final tile.
      const int task_index = static_cast<int>(task);
      const int tile_m = task_index % tiles_m, tile_n = task_index / tiles_m;
      auto gA = local_tile(cA, select<0, 2>(tile), make_coord(tile_m, _));
      auto gB = local_tile(cB, select<1, 2>(tile), make_coord(tile_n, _));
      auto gD = local_tile(cD, tile, make_coord(tile_m, tile_n, 0), Step<_1, _1, X>{});
      auto tAgA = thr_a.partition_S(gA);
      auto tBgB = thr_b.partition_S(gB);
      auto tArA = thr_a.partition_sg_fragment_D(gA(_, _, 0));
      auto tSrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
      auto tBrB = thr_b.partition_sg_fragment_D(gB(_, _, 0));
      auto tSrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));
      SubgroupTensor acc = thr_mma.partition_sg_fragment_C(gD);
      auto pAgA = prefetch_a.get_slice(tid).partition_S(gA);
      auto pBgB = prefetch_b.get_slice(tid).partition_S(gB);
      int prefetch_k = 0;
      CUTE_UNROLL
      for (; prefetch_k < Base::Stages; ++prefetch_k) {
        prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
        prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
      }
      for (int k = 0; k < ktiles; ++k, ++prefetch_k) {
        barrier_arrive(barrier_scope);
        copy(copy_a, tAgA(_, _, _, k), tArA);
        copy(copy_b, tBgB(_, _, _, k), tBrB);
        if (prefetch_k < ktiles) {
          prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
          prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
        }
        reorder(tArA, tSrA);
        reorder(tBrB, tSrB);
        cute::gemm(p.mma, tSrA, tSrB, acc);
        barrier_wait(barrier_scope);
      }
      // CPU CuTe coordinate proof: each lane holds rows0..31 at
      // columns lane and lane+16. Subgroup grid indexes N, then M.
      // Round the accumulator exactly as the ordinary BF16 output store.
      CUTE_UNROLL
      for (int r = 0; r < 32; ++r) {
        std::uint32_t best = 0;
        CUTE_UNROLL
        for (int c = 0; c < 2; ++c) {
          const int column = tile_n * BN + sg_n * 32 + lane + c * 16;
          const Element rounded = Element(acc(r + c * 32));
          const auto bits = sycl::bit_cast<std::uint16_t>(rounded);
          const auto rank = p.ranks[bits];
          const std::uint32_t key =
              column < p.n ? (std::uint32_t(rank) << 7) | std::uint32_t(127 - (sg_n * 32 + lane + c * 16)) : 0;
          best = sycl::max(best, key);
        }
        best = sycl::reduce_over_group(sg, best, sycl::maximum<std::uint32_t>());
        if (lane == 0) scratch[(sg_m * 32 + r) * NSG + sg_n] = best;
      }
      item.barrier(sycl::access::fence_space::local_space);
      // 128 threads emit one compact key per valid row and N tile.
      if (tid < BM && tile_m * BM + tid < p.m) {
        std::uint32_t best = 0;
        CUTE_UNROLL
        for (int nsg = 0; nsg < NSG; ++nsg)
          best = sycl::max(best, scratch[tid * NSG + nsg]);
        p.partial[std::int64_t(tile_m * BM + tid) * tiles_n + tile_n] = best;
      }
      item.barrier(sycl::access::fence_space::local_space);
    }
  }
};

}  // namespace sgl::gemm
