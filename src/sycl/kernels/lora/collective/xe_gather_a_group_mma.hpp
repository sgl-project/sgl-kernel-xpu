/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
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
/*!
  \file
  \brief Gathered-A mainloop of the chunked-SGMV LoRA kernels, a collective mainloop plugged into
         sycl-tla's grouped GemmUniversal (xe_gemm_array_cooperative.hpp).

  GatherAGroupMma implements the collective mainloop interface the grouped kernel calls; its
  dispatch policy MainloopXeGatherAGroup<SGK, BLOCK_B> has Schedule = KernelXePtrArrayCooperative,
  so GemmUniversal picks sycl-tla's grouped Xe kernel.

    * MMA     : the TiledMma (XE_DPAS_TT) of NSG_N sub-groups side by side along N.
    * B       : the adapter's whole [N, K] weight matrix as a cute tensor; the tile's first
                column enters as a coordinate offset (domain_offset), so the surface base stays
                64-byte aligned for any slice layout.
                BLOCK_B = true  : make_block_2d_copy_B + reorder into the MMA fragment.
                BLOCK_B = false : for weights a 2D block load cannot read (block2d_rows_ok()),
                                  the B fragment is filled from the coordinates cute assigns it.
    * A       : rows are gathered (x[perm[r]]), so they are not one block-2D surface. cute's A
                fragment element s * BM + i is (row i, k0 + 16 s + lane): one sub-group block
                read per row and 16-wide k step.
    * split-K : the work-group holds SGK copies of the TiledMma (MaxThreadsPerBlock); copy sg_k
                takes k-blocks sg_k, sg_k + SGK, ... and the epilogue reduces the partials.
*/

#pragma once

#include <cstdint>
#include <cute/tensor.hpp>
#include <sycl/sycl.hpp>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "sycl/kernels/lora/collective/xe_gather_scatter_copy.hpp"
#include "sycl/kernels/lora/collective/xe_segment_problem_shape.hpp"

namespace cutlass::lora::collective {

// Dispatch policy of GatherAGroupMma. Schedule = KernelXePtrArrayCooperative makes
// GemmUniversal pick sycl-tla's grouped Xe kernel.
template <int SGK_, bool BLOCK_B_>
struct MainloopXeGatherAGroup {
  constexpr static int Stages = 1;
  constexpr static int SubgroupSize = 16;
  constexpr static int SGK = SGK_;
  constexpr static bool BLOCK_B = BLOCK_B_;
  using ArchTag = cutlass::arch::IntelXe;
  using Schedule = cutlass::gemm::KernelXePtrArrayCooperative;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
};

// The grouped kernel needs only the collective mainloop interface (the types below, Arguments,
// Base, to_base_arguments(), operator()), not a CollectiveMma specialization.
template <
    class DispatchPolicy_,
    class TileShape_,
    class ElementA_,
    class StrideA_,
    class ElementB_,
    class StrideB_,
    class TiledMma_>
struct GatherAGroupMma {
  static constexpr int SGK = DispatchPolicy_::SGK;
  static constexpr bool BLOCK_B = DispatchPolicy_::BLOCK_B;
  using DispatchPolicy = DispatchPolicy_;
  using WorkgroupTileShape = TileShape_;
  using TiledMma = TiledMma_;
  using ArchTag = typename DispatchPolicy::ArchTag;
  using ElementA = ElementA_;  // cute::bfloat16_t / cute::half_t
  using ElementB = ElementB_;
  // Strides are fixed by the gather (A rows through perm, B [N, K] row-major); these satisfy
  // the kernel's rank-3 checks only.
  using StrideA = StrideA_;
  using InternalStrideA = StrideA_;
  using StrideB = StrideB_;
  using InternalStrideB = StrideB_;
  using ElementAccumulator = float;
  // Descriptive types GemmUniversalAdapter reads; the copies themselves are built in operator().
  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;
  using TransformA = cute::identity;
  using TransformB = cute::identity;
  using MmaAtomShape = typename TiledMma::AtomShape_MNK;

  static constexpr int SubgroupSize = DispatchPolicy::SubgroupSize;
  static constexpr int BM = cute::get<0>(WorkgroupTileShape{});
  static constexpr int BN = cute::get<1>(WorkgroupTileShape{});
  static constexpr int BK = cute::get<2>(WorkgroupTileShape{});
  static constexpr int ThreadsPerMma = cute::size(TiledMma{});
  static constexpr int NSG_N = ThreadsPerMma / SubgroupSize;
  static constexpr uint32_t MaxThreadsPerBlock = ThreadsPerMma * SGK;
  using SubgroupTileShape = cute::Shape<cute::Int<BM>, cute::Int<BN / NSG_N>, cute::Int<BK>>;
  static_assert(BK % 16 == 0, "k-blocks are whole 16-wide DPAS steps");

  struct Arguments {
    const uint16_t* a;    // [num_rows, a_ld], rows in physical order
    int64_t a_ld;         // elements per A row
    int64_t a_slice_ld;   // A column offset of slice p is p * a_slice_ld
    const uint16_t* b;    // [num_adapters, N, K]
    const int64_t* perm;  // logical -> physical row, or nullptr (identity)
    bool a_rows_aligned;  // A rows 16-byte aligned: block reads for full k-blocks
    SegmentSliceProblemShape groups;
  };
  using Params = Arguments;

  // Per-group view, built on the device by to_base_arguments() when the work-group's group changes.
  struct Base {
    struct Arguments {
      const uint16_t* a;  // first column of this slice's A band
      int64_t a_ld;
      const uint16_t* b;  // this adapter's [N, K] matrix
      int N;              // weight rows (all slices)
      int K;
      int n_lo;  // first output column of the slice
      int row0;  // first logical row of the segment
      int rows;  // segment rows
      const int64_t* perm;
      bool a_rows_aligned;
    };
    using Params = Arguments;
    template <class ProblemShape>
    static constexpr Params to_underlying_arguments(ProblemShape const&, Arguments const& args, void*) {
      return args;
    }
  };

  GatherAGroupMma() = default;

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const&, Arguments const& args, void*) {
    return args;
  }

  template <class ProblemShape>
  static bool can_implement(ProblemShape const&, Arguments const& args) {
    return !BLOCK_B || block2d_rows_ok(args.b, args.groups.N, args.groups.K);
  }

  CUTLASS_DEVICE static typename Base::Arguments to_base_arguments(Arguments const& args, int g) {
    auto const& gr = args.groups;
    return {
        args.a + gr.slice(g) * args.a_slice_ld,
        args.a_ld,
        args.b + gr.adapter(g) * gr.N * static_cast<int64_t>(gr.K),
        gr.N,
        gr.K,
        gr.n_lo(g),
        gr.row0(g),
        gr.rows(g),
        args.perm,
        args.a_rows_aligned};
  }

  // accum (this thread's C fragment, zeroed by the kernel) += A @ B^T over this copy's k-blocks.
  // blk_coord is the tile's (m, n) inside the group.
  template <class FrgTensorD, class TensorA, class TensorB, class FrgTensorC, class KTileIterator, class BlkCoord>
  CUTLASS_DEVICE void operator()(
      FrgTensorD& accum,
      TensorA const&,
      TensorB const&,
      FrgTensorC const&,
      KTileIterator,
      int k_tile_count,
      BlkCoord const& blk_coord,
      int const& /*K*/,
      int thread_idx,
      typename Base::Params const& p) const {
    using namespace cute;
    const int lane = thread_idx % SubgroupSize;
    const int sg_k = thread_idx / ThreadsPerMma;
    const int m0 = get<0>(blk_coord) * BM;
    const int rows = sycl::min(BM, p.rows - m0);
    const int n0 = p.n_lo + get<1>(blk_coord) * BN;

    TiledMma mma{};
    auto thr_mma = mma.get_slice(thread_idx % ThreadsPerMma);
    auto B = make_tensor(
        make_gmem_ptr(reinterpret_cast<const ElementB*>(p.b)),
        make_layout(make_shape(p.N, p.K), make_stride(p.K, _1{})));
    auto cB = domain_offset(make_coord(n0, 0), make_identity_tensor(B.shape()));
    auto gB = local_tile(cB, select<1, 2>(mma.tile_mnk()), make_coord(0, _));  // (BN, BK, k)
    auto tSrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

    // Gathered A rows; rows past the segment clamp onto its last row.
    const uint16_t* arow[BM];
    CUTE_UNROLL
    for (int r = 0; r < BM; ++r) {
      const int lr = p.row0 + m0 + sycl::min(r, rows - 1);
      arow[r] = p.a + (p.perm ? p.perm[lr] : static_cast<int64_t>(lr)) * p.a_ld;
    }
    auto tSrA = thr_mma.partition_sg_fragment_A(make_identity_tensor(make_shape(Int<BM>{}, Int<BK>{})));

    auto load_a = [&](int c) {
      const int k0 = c * BK;
      if (p.a_rows_aligned && k0 + BK <= p.K) {
        CUTE_UNROLL
        for (int i = 0; i < BM; ++i) {
          const auto v = sg_block_read<BK / 16>(arow[i] + k0);
          CUTE_UNROLL
          for (int s = 0; s < BK / 16; ++s) {
            tSrA(s * BM + i) = ElementA::bitcast(v[s]);
          }
        }
      } else {
        CUTE_UNROLL
        for (int s = 0; s < BK / 16; ++s) {
          const int k = k0 + s * 16 + lane;
          CUTE_UNROLL
          for (int i = 0; i < BM; ++i) {
            tSrA(s * BM + i) = ElementA::bitcast(k < p.K ? arow[i][k] : uint16_t(0));
          }
        }
      }
    };

    if constexpr (BLOCK_B) {
      auto tiled_copy_b = make_block_2d_copy_B(mma, B);
      auto thr_copy_b = tiled_copy_b.get_slice(thread_idx % ThreadsPerMma);
      auto tBgB = thr_copy_b.partition_S(gB);
      auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));
      for (int c = sg_k; c < k_tile_count; c += SGK) {
        copy(tiled_copy_b, tBgB(_, _, _, c), tBrB);
        load_a(c);
        reorder(tBrB, tSrB);
        cute::gemm(mma, tSrA, tSrB, accum);
      }
    } else {
      auto tCcB = thr_mma.partition_B(gB(_, _, 0));  // coordinate of each tSrB element
      for (int c = sg_k; c < k_tile_count; c += SGK) {
        const int k0 = c * BK;
        CUTE_UNROLL
        for (int i = 0; i < tCcB.size(); ++i) {
          const int n = get<0>(tCcB(i)), k = get<1>(tCcB(i)) + k0;
          tSrB(i) = ElementB::bitcast(n < p.N && k < p.K ? p.b[static_cast<int64_t>(n) * p.K + k] : uint16_t(0));
        }
        load_a(c);
        cute::gemm(mma, tSrA, tSrB, accum);
      }
    }
  }
};

}  // namespace cutlass::lora::collective
