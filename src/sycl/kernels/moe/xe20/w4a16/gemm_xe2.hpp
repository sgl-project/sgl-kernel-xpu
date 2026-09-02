/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include <cute/tensor.hpp>
#include <cute/util/compat.hpp>
#include <cute/util/xe_split_barrier.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include "../common/block_2d_copy_d.hpp"
#include "../common/scale.hpp"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"
#include "cutlass/tensor_ref.h"
#include "mxfp4_dequant.hpp"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace moe_w4a16 {

using namespace cute;

template <
    class GmemTiledCopyA,
    class GmemTiledCopyB,
    class GmemTiledCopyC,
    class ATensor,
    class BTensor,
    class DTensor,
    class TiledMMA,
    typename ElementS,
    typename ElementBI>
CUTE_DEVICE void xe_gemm(
    ATensor const& A,  // (M,K)
    BTensor const& B,  // (N,K)
    const ElementS* Scales,
    const ElementBI* Bias,
    DTensor& C,  // (M,N)
    Coord<int, int, cute::Underscore, int> blk_coord,
    TiledMMA const& mma) {
  using TA = typename ATensor::element_type;
  using TB = typename BTensor::element_type;
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  auto wg_m = get<0>(blk_coord);
  auto wg_n = get<1>(blk_coord);
  int local_id = item.get_local_linear_id();

  Tensor cA = make_identity_tensor(A.shape());
  Tensor cB = make_identity_tensor(B.shape());
  Tensor cC = make_identity_tensor(C.shape());

  auto wg_tile = mma.tile_mnk();
  auto wg_coord = make_coord(wg_m, wg_n, 0);

  Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(wg_m, _));  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(wg_n, _));  // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(cC, wg_tile, wg_coord, Step<_1, _1, X>{});        // (BLK_M,BLK_N)

  auto copy_a = get_block_2d_copy_A<GmemTiledCopyA>(mma, A);
  auto copy_b = get_block_2d_copy_B<GmemTiledCopyB>(mma, B);
  auto copy_c = moe_xe20::make_moe_block_2d_copy_D<GmemTiledCopyC>(mma, C);

  auto thr_mma = mma.get_slice(local_id);
  auto thr_copy_a = copy_a.get_slice(local_id);
  auto thr_copy_b = copy_b.get_slice(local_id);
  auto thr_copy_c = copy_c.get_slice(local_id);

  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
  auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgB = thr_copy_b.partition_S(gB);

  /* Partition C */
  auto tCrC = thr_mma.partition_sg_fragment_C(gC);
  auto tCrC_out = thr_copy_c.partition_sg_fragment_S(gC);
  auto tCgC = thr_copy_c.partition_D(gC);

  auto prefetch_a = make_block_2d_prefetch(copy_a);
  auto prefetch_b = make_block_2d_prefetch(copy_b);

  auto thr_prefetch_A = prefetch_a.get_slice(local_id);
  auto thr_prefetch_B = prefetch_b.get_slice(local_id);

  auto pAgA = thr_prefetch_A.partition_S(gA);
  auto pBgB = thr_prefetch_B.partition_S(gB);

  const int prefetch_dist = 3;

  constexpr SPIRVScope barrier_scope = ScopeWorkgroup;

  int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));
  int k_tile_prefetch = 0;

  clear(tCrC);

  using ElementB = typename BTensor::element_type;
  static constexpr bool is_B_fp8_type =
      std::is_same_v<ElementB, cutlass::float_e5m2_t> || std::is_same_v<ElementB, cutlass::float_e4m3_t>;

  CUTE_UNROLL
  for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
    prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
  }

  for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
    barrier_arrive(barrier_scope);

    copy(copy_a, tAgA(_, _, _, k_tile), tArA);
    copy(copy_b, tBgB(_, _, _, k_tile), tBrB);

    if (k_tile_prefetch < k_tile_count) {
      prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
      prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
    }

    reorder(tArA, tCrA);
    reorder(tBrB, tCrB);

    cute::gemm(mma, tCrA, tCrB, tCrC);

    barrier_wait(barrier_scope);
  }

  if constexpr (is_B_fp8_type) {
    float B_scale = Scales[0];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tCrC.size(); ++i) {
      tCrC(i) *= B_scale;
    }
  }

  if (Bias != nullptr) {
    static constexpr auto ATOM_M = get<1>(typename TiledMMA::ThrLayoutVMNK{}.shape());
    static constexpr auto ATOM_N = get<2>(typename TiledMMA::ThrLayoutVMNK{}.shape());

    auto sg_local_n_coord = cutlass::get_sub_group_id() % ATOM_N;

    static constexpr auto tile_m = get<0>(wg_tile);
    static constexpr auto tile_n = get<1>(wg_tile);

    // 32 * 64
    static constexpr auto SG_M = tile_m / ATOM_M;  // BLK_M / ATOM_M;
    static constexpr auto SG_N = tile_n / ATOM_N;  // BLK_N / ATOM_N;

    int sg_local_id = cutlass::get_sub_group_local_id();
    static constexpr int sg_local_range = 16;

    int n_tile_start = wg_n * tile_n;
    int n_sg_start = sg_local_n_coord * SG_N;

    CUTLASS_PRAGMA_UNROLL
    for (int sn = 0; sn < SG_N / sg_local_range; ++sn) {
      int sg_local_n = sn * sg_local_range + sg_local_id;
      float b_float = Bias[n_tile_start + n_sg_start + sg_local_n];
      CUTLASS_PRAGMA_UNROLL
      for (int sm = 0; sm < SG_M; ++sm) {
        tCrC(sn * SG_M + sm) += b_float;
      }
    }
  }

  reorder(tCrC, tCrC_out);
  copy(copy_c, tCrC_out, tCgC);
}

// A view of `Blocks` M-blocks of a (V, M, ...) register fragment, starting at
// M-block `First`. The M mode of an A or C fragment is the dpas atom's 8-row
// block, so a view is what lets one gemm() cover a row range that is a whole
// number of blocks -- there is no way to give dpas a partial block.
// Non-const on purpose: a view built from a `Tensor const&` carries a const
// element type, and dpas writes its accumulator.
template <int First, int Blocks, class Engine, class Layout>
CUTE_DEVICE auto m_block_view(Tensor<Engine, Layout>& t) {
  static_assert(rank(Layout{}) == 3, "an A/C subgroup fragment is (V, M, K|N)");
  auto layout = make_layout(get<0>(t.layout()), make_layout(Int<Blocks>{}, stride<1>(t.layout())), get<2>(t.layout()));
  return make_tensor(t.data() + First * stride<1>(t.layout()), layout);
}

template <
    class GmemTiledCopyA,
    class GmemTiledCopyB,
    class GmemTiledCopyC,
    int GroupSize,
    bool HasZero,
    int PrefetchDist,
    bool MainloopBarrier,
    bool SkipPaddedN,
    class ATensor,
    class BTensor,
    class DTensor,
    class TiledMMA,
    typename ElementS,
    typename ElementBI>
CUTE_DEVICE void xe_gemm_4bits(
    ATensor const& A,  // (M,K)
    BTensor const& B,  // (N,K)
    const ElementS* Scales,
    const ElementS* Zeros,
    const ElementBI* Bias,
    DTensor& C,  // (M,N)
    Coord<int, int, cute::Underscore, int> blk_coord,
    TiledMMA const& mma) {
  using TA = typename ATensor::element_type;
  using TB = typename BTensor::element_type;
  static constexpr int group_size = GroupSize;
  static constexpr int sg_local_range = 16;
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  auto wg_m = get<0>(blk_coord);
  auto wg_n = get<1>(blk_coord);
  int local_id = item.get_local_linear_id();

  Tensor cA = make_identity_tensor(A.shape());
  Tensor cB = make_identity_tensor(B.shape());
  Tensor cC = make_identity_tensor(C.shape());

  auto wg_tile = mma.tile_mnk();
  auto wg_coord = make_coord(wg_m, wg_n, 0);

  Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(wg_m, _));  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(wg_n, _));  // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(cC, wg_tile, wg_coord, Step<_1, _1, X>{});        // (BLK_M,BLK_N)

  auto copy_a = get_block_2d_copy_A<GmemTiledCopyA>(mma, A);
  auto copy_b = get_block_2d_copy_B<GmemTiledCopyB>(mma, B);
  auto copy_c = moe_xe20::make_moe_block_2d_copy_D<GmemTiledCopyC>(mma, C);

  auto thr_mma = mma.get_slice(local_id);
  auto thr_copy_a = copy_a.get_slice(local_id);
  auto thr_copy_b = copy_b.get_slice(local_id);
  auto thr_copy_c = copy_c.get_slice(local_id);

  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
  auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgB = thr_copy_b.partition_S(gB);

  /* Partition C */
  auto tCrC = thr_mma.partition_sg_fragment_C(gC);
  auto tCrC_out = thr_copy_c.partition_sg_fragment_S(gC);
  auto tCgC = thr_copy_c.partition_D(gC);

  auto prefetch_a = make_block_2d_prefetch(copy_a);
  auto prefetch_b = make_block_2d_prefetch(copy_b);

  auto thr_prefetch_A = prefetch_a.get_slice(local_id);
  auto thr_prefetch_B = prefetch_b.get_slice(local_id);

  auto pAgA = thr_prefetch_A.partition_S(gA);
  auto pBgB = thr_prefetch_B.partition_S(gB);

  // Keep this a plain const int, in this position: the mainloop's IGC schedule is
  // sensitive to the shape of the prologue, and reordering even one declaration in
  // it costs 1.4-8.3% (worst on the short-M tiles).
  static_assert(PrefetchDist > 0, "prefetch distance must be positive");
  const int prefetch_dist = PrefetchDist;

  constexpr SPIRVScope barrier_scope = ScopeWorkgroup;

  int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));
  int k_tile_prefetch = 0;

  static constexpr auto ATOM_M = get<1>(typename TiledMMA::ThrLayoutVMNK{}.shape());
  static constexpr auto ATOM_N = get<2>(typename TiledMMA::ThrLayoutVMNK{}.shape());
  static constexpr auto ATOM_K = get<3>(typename TiledMMA::ThrLayoutVMNK{}.shape());

  static constexpr auto tile_m = get<0>(wg_tile);
  static constexpr auto tile_n = get<1>(wg_tile);
  static constexpr auto tile_k = get<2>(wg_tile);

  static constexpr auto SG_M = tile_m / ATOM_M;  // BLK_M / ATOM_M;
  static constexpr auto SG_N = tile_n / ATOM_N;  // BLK_N / ATOM_N;
  static constexpr auto SG_K = tile_k / ATOM_K;  // BLK_K / ATOM_K;

  static constexpr auto thr_N = get<1>(tCrB.shape());
  static constexpr auto channel_num = get<0>(get<0>(tCrB.shape()));
  auto n_tile_start = wg_n * tile_n;

  auto sg_local_n_coord = cutlass::get_sub_group_id() % ATOM_N;
  int sg_local_id = cutlass::get_sub_group_local_id();
  int n_sg_start = sg_local_n_coord * SG_N;
  int group_num = get<1>(A.shape()) / group_size;
  int x_idx = sg_local_id / channel_num;

  using scaleStoreType = conditional_t<is_same_v<TA, half_t>, half_t, float>;
  scaleStoreType scales[thr_N * channel_num];
  conditional_t<HasZero, TA, uint8_t> zeros[thr_N * channel_num];

  // Where each (n, channel) slot gathers its group scales from, i.e. the start of
  // its B column inside this expert's scale matrix. The N direction is tiled to a
  // BLK_N multiple, so the last tile of an expert whose N is not a multiple can
  // hold columns that do not exist; their scales are clamped to the last real
  // column, because past the last expert they would be past the whole array.
  // (The clamped columns are not stored: the D copy is bounds-checked.)
  // Carried as pointers, not as (base, index) pairs: consecutive gathers always
  // advance by one group, so the loop pays one 64-bit add per slot instead of the
  // int add + widen + 64-bit add IGC needs to rebuild `Scales[offset + group_idx]`
  // from scratch every k-tile.
  const int scale_col_bound = size<0>(B.shape()) - 1;
  const ElementS* scale_ptr[thr_N * channel_num];
  const ElementS* zero_ptr[HasZero ? thr_N * channel_num : 1];
  CUTLASS_PRAGMA_UNROLL
  for (int n = 0; n < thr_N; ++n) {
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < channel_num; ++c) {
      const int col = n_tile_start + n_sg_start + n * sg_local_range + x_idx + c * (sg_local_range / channel_num);
      const int offset = cute::min(col, scale_col_bound) * group_num;
      scale_ptr[n * channel_num + c] = Scales + offset;
      if constexpr (HasZero) {
        zero_ptr[n * channel_num + c] = Zeros + offset;
      }
    }
  }

  // Whether to prefetch the next group's scales. They are one E8M0 byte per B
  // column per group, so a whole subgroup's worth is 16-32 B: the 2D-block
  // prefetch below costs a message and a k-loop test to fetch what the gather
  // will pull into L1 anyway one k-tile later. Off is worth ~3%.
  //
  // It is also not safe as written. The prefetch covers SG_N columns starting at
  // this subgroup's first one, so on the last tile of the last expert it reads
  // past the end of the scale array -- with a 2D-block message, past what the
  // clamp above can fix. That is a device fault, not a wrong answer, and it only
  // shows up when the allocation ends exactly at the array (E=128, N=768,
  // K=2880). The bound test below is the fix if this is ever turned back on.
  static constexpr bool kScalePrefetch = false;
  static constexpr int kScalePrefetchSlop = 64;
  const int scale_prefetch_bound = (scale_col_bound + 1) * group_num - kScalePrefetchSlop;

  // MXFP4 without zero points is dequantized inside the E2M1 -> BF16 reorder,
  // which ends in a multiply anyway; see mxfp4_dequant.hpp. Everything else needs
  // the separate pass over the B fragment further down.
  static constexpr bool kFuseDequant = std::is_same_v<TB, float_e2m1_t> && std::is_same_v<TA, bfloat16_t> && !HasZero;

  // The multipliers a reorder chunk needs, from the chunk's first value index:
  // the fragment's innermost mode alternates between the two columns a work-item
  // covers (channels), and so between their two group scales, while a whole
  // 8-value chunk stays inside one n-block.
  using BFragLayout = decltype(tCrB.layout());
  static constexpr int frag_mode0 = size<0>(tCrB.shape());
  intel::vector_t<float, 2> mul_pairs[kFuseDequant ? thr_N : 1];
  auto mul_of = [&](auto dv) -> intel::vector_t<float, 2> const& {
    constexpr int n = (decltype(dv)::value / frag_mode0) % thr_N;
    return mul_pairs[n];
  };
  if constexpr (kFuseDequant) {
    static_assert(channel_num == 2, "the folded multiply covers exactly two channels");
    static_assert(std::is_same_v<scaleStoreType, float>, "the folded multiply takes f32 multipliers");
    // ((channel, x), n, k):((1, channel), mode0, ...) -- the n mode carries a zero
    // stride when it is degenerate, which the index above handles.
    static_assert(
        stride<0, 0>(BFragLayout{}) == 1 && stride<0, 1>(BFragLayout{}) == channel_num &&
            (thr_N == 1 || stride<1>(BFragLayout{}) == frag_mode0),
        "the folded dequant needs a channel-innermost B fragment with n above mode 0");
    static_assert(frag_mode0 % 8 == 0, "an 8-value reorder chunk must stay inside one n-block");
  }

  clear(tCrC);

  using ElementB = typename BTensor::element_type;
  static constexpr bool is_B_fp8_type =
      std::is_same_v<ElementB, cutlass::float_e5m2_t> || std::is_same_v<ElementB, cutlass::float_e4m3_t>;

  // How many of this subgroup's 8-row dpas blocks hold rows that exist.
  //
  // A tile spans tile_m rows of the expert's A/D slice, but an expert's last tile
  // is partial, and that is where nearly all of this kernel's lost throughput
  // lives: on the gpt-oss-120b l0 histogram (128 experts, 16384 rows) 23% of the
  // dpas work at BLK_M = 64 is on rows past the end of an expert.
  //
  // Skipping the padded blocks costs nothing else: the tile keeps its shape, so B
  // is loaded and dequantized exactly once per k-tile as it is for a full tile
  // (which is why a narrower tile loses instead -- see the tile registry), and the
  // D store was already bounds-checked and never wrote those rows.
  //
  // Only for a tile whose work-group is one M subgroup wide (ATOM_M == 1, which is
  // every tile the production dispatch selects), and only for a subgroup tile that
  // holds more than one dpas M block: at SG_M = 8 (the decode tile) a live tile
  // always has its one block live, so the predicate can never fire and all that is
  // left of it is a branch the compiler cannot fold.
  static constexpr int kMBlocks = size<1>(tCrA.shape());
  static constexpr int kMBlockRows = SG_M / kMBlocks;
  static_assert(kMBlocks * kMBlockRows == SG_M, "the M mode of the A fragment must tile SG_M");
  static constexpr bool kSkipPaddedM = ATOM_M == 1 && kMBlocks > 1;
  const int sg_m_row0 = (cutlass::get_sub_group_id() / ATOM_N) * SG_M;
  const int sg_m_blocks = cute::max(
      0, cute::min(kMBlocks, ceil_div(int(size<0>(C.shape())) - int(wg_m) * int(tile_m) - sg_m_row0, kMBlockRows)));

  // The other side of the padding: a work-group tile is a whole BLK_N wide, so an
  // expert whose N is not a multiple of it -- gpt-oss GEMM2 is N = 2880 against
  // BLK_N = 128 -- ends with subgroups whose columns are all past the end of D.
  // Those subgroups have nothing to store (the D copy is bounds-checked and never
  // wrote those columns), so they skip the mainloop's loads, dequant, dpas and
  // epilogue, and leave their issue slots, their share of the L1 and their share of
  // the XMX array to the subgroups that do have work. The work-stealing loop is
  // outside this function and every subgroup still reaches it, so this cannot
  // deadlock.
  //
  // The skip is not free, and whether it pays is a property of the shape: a skipped
  // subgroup drops its share of the prefetch with it (make_block_2d_prefetch is
  // sliced by the work-item id, so the WG's A and B tile prefetches are partitioned
  // across all of its subgroups and the live ones then miss on the lines the dead
  // one would have fetched), and the `sg_alive` branch itself costs codegen even
  // where nothing is ever skipped. Hence the shape rule in want_nskip().
  //
  // The M skip above cannot leave a subgroup dead -- it needs ATOM_M == 1, and then
  // the subgroup owning row 0 is the only one there is and always has a live block
  // -- so only the N skip appears here.
  static constexpr bool kSkipPaddedN = SkipPaddedN;
  const bool sg_alive = !kSkipPaddedN || n_tile_start + n_sg_start < int(size<1>(C.shape()));

  // Whether the policy's per-k-tile split barrier request (MainloopBarrier, see
  // gemm_xe2_policy.hpp) can actually be honoured. A split barrier is only legal if
  // *every* subgroup of the work-group reaches it, and a subgroup that finds
  // `sg_alive` false runs no k-loop, so the ones that do would wait on an arrive
  // that never comes and the work-group hangs. Whether the N skip fires is a
  // run-time property, so the compile-time test has to be the template bit itself.
  static constexpr bool kMainloopBarrier = MainloopBarrier && !kSkipPaddedN;

  auto prefetch_scale_group = [&](int scale_k_tile) {
    if constexpr (!kScalePrefetch) {
      return;
    }
    if (scale_k_tile >= k_tile_count || scale_k_tile * tile_k % group_size != 0) {
      return;
    }

    int scale_group_idx = scale_k_tile * tile_k / group_size;
    if ((n_tile_start + n_sg_start + SG_N - 1) * group_num + scale_group_idx > scale_prefetch_bound) {
      return;
    }
    auto next_scales_tensor = make_tensor(
        make_gmem_ptr(
            reinterpret_cast<const ElementS*>(Scales + (n_tile_start + n_sg_start) * group_num + scale_group_idx)),
        make_layout(make_shape(Int<SG_N>{}, Int<1>{}), make_stride(group_num, Int<1>{})));
    auto prefetch_scales = make_block_2d_prefetch<1>(make_shape(Int<SG_N>{}, Int<1>{}), next_scales_tensor);
    auto thr_prefetch_scales = prefetch_scales.get_slice(sg_local_id);
    auto pSgS = thr_prefetch_scales.partition_S(make_identity_tensor(make_shape(Int<SG_N>{}, Int<1>{})));
    prefetch(prefetch_scales, pSgS(_, 0, 0));
  };

  // No columns: no mainloop at all -- not even the prefetch -- and no epilogue
  // either. Hoisted out of the k-loop rather than tested in it, so the loop the
  // compiler schedules is the one a live subgroup runs.
  if (!sg_alive) {
    return;
  }

  CUTE_UNROLL
  for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
    prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
    prefetch_scale_group(k_tile_prefetch);
  }

  for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
    if constexpr (kMainloopBarrier) {
      barrier_arrive(barrier_scope);
    }

    copy(copy_a, tAgA(_, _, _, k_tile), tArA);
    copy(copy_b, tBgB(_, _, _, k_tile), tBrB);

    if (k_tile * tile_k % group_size == 0) {
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < thr_N; ++n) {
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < channel_num; ++c) {
          const int slot = n * channel_num + c;
          const ElementS* sp = scale_ptr[slot];
          scale_ptr[slot] = sp + 1;
          const ElementS* zp = nullptr;
          if constexpr (HasZero) {
            zp = zero_ptr[slot];
            zero_ptr[slot] = zp + 1;
          }
          scaleStoreType scale;
          if constexpr (std::is_same_v<TB, int4_t>) {
            scale = sp[0];
          } else if constexpr (std::is_same_v<TB, uint4_t>) {
            scale = static_cast<scaleStoreType>(sp[0]);
            if constexpr (HasZero) {
              zeros[slot] = static_cast<TA>(zp[0]);
            }
          } else if constexpr (std::is_same_v<TB, float_e2m1_t>) {
            const uint32_t e8m0 = static_cast<uint32_t>(sp[0]);
            if constexpr (kFuseDequant) {
              // Folded into the reorder's own multiply, so it carries the
              // conversion constant and the 2^-kFoldShift bias with it.
              scale = mxfp4_fold_multiplier(e8m0);
            } else {
              uint32_t scale_u32 = e8m0 << 23;
              scale = static_cast<scaleStoreType>(reinterpret_cast<float&>(scale_u32));
            }
          }

          scales[slot] = scale;
          if constexpr (kFuseDequant) {
            mul_pairs[n][c] = scale;
          }
        }
      }
    }

    if (k_tile_prefetch < k_tile_count) {
      prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
      prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
    }
    prefetch_scale_group(k_tile_prefetch);

    reorder(tArA, tCrA);
    if constexpr (kFuseDequant) {
      mxfp4_reorder_dequant(tBrB, tCrB, mul_of);
    } else {
      reorder(tBrB, tCrB);
    }

    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < thr_N && !kFuseDequant; ++n) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < channel_num; ++c) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tCrB.size() / thr_N / channel_num; ++i) {
          if constexpr (HasZero) {
            TA value = tCrB(cute::tuple(c, _), n, _)[i] - zeros[n * channel_num + c];
            if constexpr (std::is_same_v<TA, half_t>) {
              tCrB(cute::tuple(c, _), n, _)[i] = value * scales[n * channel_num + c];
            } else {
              tCrB(cute::tuple(c, _), n, _)[i] = moe_xe20::apply_bf16_or_fp16_scale(value, scales[n * channel_num + c]);
            }
          } else {
            if constexpr (std::is_same_v<TA, half_t>) {
              tCrB(cute::tuple(c, _), n, _)[i] *= scales[n * channel_num + c];
            } else {
              tCrB(cute::tuple(c, _), n, _)[i] =
                  moe_xe20::apply_bf16_or_fp16_scale(tCrB(cute::tuple(c, _), n, _)[i], scales[n * channel_num + c]);
            }
          }
        }
      }
    }

    if constexpr (kSkipPaddedM) {
      if (sg_m_blocks == kMBlocks) {
        cute::gemm(mma, tCrA, tCrB, tCrC);
      } else {
        // One gemm over the whole valid block range, picked by an equality test on
        // a subgroup-uniform, loop-invariant count. Predicating each block
        // separately instead costs far more than the dpas it saves: the branch
        // between two blocks is a basic-block boundary, so what should be one run
        // of independent dpas becomes a chain of dependent pairs, and the penalty
        // grows with the block count. Measured cost of one partial tile, in units
        // of a full tile of the same shape (l0 GEMM1, 90 k-tiles, three full tiles
        // alongside it so the launch is not bandwidth-bound):
        //
        //   valid blocks   1     2     4     6     7     8
        //   per block    0.803 0.830 0.915 1.220 1.292 1.000
        //   one range    0.79 (a single range at 6 blocks measures 0.808)
        //
        // A 6-of-8 tile was *slower* than computing all eight. The floor near 0.79
        // is the per-k-tile stream that does not depend on the block count (the A
        // and B loads, the scale gather, the prefetches, the two reorders); no
        // predication can go below it, which is why the histogram of tail blocks --
        // 59 of 128 l0 experts land on 5, 6 or 7 -- decides whether the skip pays.
        for_each(make_int_sequence<kMBlocks - 1>{}, [&](auto mb) {
          constexpr int blocks = decltype(mb)::value + 1;
          if (sg_m_blocks == blocks) {
            cute::gemm(mma, m_block_view<0, blocks>(tCrA), tCrB, m_block_view<0, blocks>(tCrC));
          }
        });
      }
    } else {
      cute::gemm(mma, tCrA, tCrB, tCrC);
    }

    if constexpr (kMainloopBarrier) {
      barrier_wait(barrier_scope);
    }
  }

  // Every product carried a 2^-kFoldShift from the folded multiplier.
  if constexpr (kFuseDequant) {
    mxfp4_unfold(tCrC);
  }

  if (Bias != nullptr) {
    CUTLASS_PRAGMA_UNROLL
    for (int sn = 0; sn < SG_N / sg_local_range; ++sn) {
      int sg_local_n = sn * sg_local_range + sg_local_id;
      float b_float = Bias[n_tile_start + n_sg_start + sg_local_n];
      CUTLASS_PRAGMA_UNROLL
      for (int sm = 0; sm < SG_M; ++sm) {
        tCrC(sn * SG_M + sm) += b_float;
      }
    }
  }

  reorder(tCrC, tCrC_out);
  copy(copy_c, tCrC_out, tCgC);
}

}  // namespace moe_w4a16
