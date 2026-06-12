/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
/*! \file
    \brief GEMM + Square Sum kernel definition
*/

#pragma once

#include "cute/algorithm/copy.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/util/compat/dims.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "../collective/xe_gemm_sqrsum_mainloop.hpp"
#include "../collective/xe_gemm_sqrsum_epilogue.hpp"

namespace cutlass::gemm_sqrsum::kernel {
using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <class CollectiveMainloop_>
class GemmSqrSumKernel {
public:
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using ElementA = typename CollectiveMainloop::ElementA;
  using ElementB = typename CollectiveMainloop::ElementB;
  using ElementC = typename CollectiveMainloop::ElementC;
  using ElementSqrSum = typename CollectiveMainloop::ElementSqrSum;

  static constexpr auto BLK_M = get<0>(TileShape{});
  static constexpr auto BLK_N = get<1>(TileShape{});
  static constexpr auto BLK_K = get<2>(TileShape{});

  // Kernel launch parameters
  struct Params {
    typename CollectiveMainloop::Params mainloop;
    cutlass::KernelHardwareInfo hw_info;

    // Problem shape
    int M;
    int K;
    int N;

    // Tensor pointers and strides
    ElementA const* ptr_A;
    int64_t stride_A_m;
    int64_t stride_A_k;

    ElementB const* ptr_B;
    int64_t stride_B_k;
    int64_t stride_B_n;

    ElementC* ptr_C;
    int64_t stride_C_m;
    int64_t stride_C_n;

    ElementSqrSum* ptr_sqrsum;  // Output: [M] row-wise square sums
  };

  struct Arguments {
    typename CollectiveMainloop::Arguments mainloop;

    // Problem shape
    int M;
    int K;
    int N;

    // Tensor pointers and strides
    ElementA const* ptr_A;
    int64_t stride_A_m;
    int64_t stride_A_k;

    ElementB const* ptr_B;
    int64_t stride_B_k;
    int64_t stride_B_n;

    ElementC* ptr_C;
    int64_t stride_C_m;
    int64_t stride_C_n;

    ElementSqrSum* ptr_sqrsum;
  };

  struct SharedStorage {
    typename CollectiveMainloop::SharedStorage mainloop;
  };

  // Convert arguments to parameters
  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    return Params{
        CollectiveMainloop::to_underlying_arguments(args.mainloop, workspace),
        cutlass::KernelHardwareInfo{},
        args.M,
        args.K,
        args.N,
        args.ptr_A,
        args.stride_A_m,
        args.stride_A_k,
        args.ptr_B,
        args.stride_B_k,
        args.stride_B_n,
        args.ptr_C,
        args.stride_C_m,
        args.stride_C_n,
        args.ptr_sqrsum};
  }

  static bool can_implement(Arguments const& args) {
    return CollectiveMainloop::can_implement(args.mainloop);
  }

  static int get_workspace_size(Arguments const& args) {
    return 0;
  }

  // Grid/block dimensions for kernel launch
  static compat::dim3 get_grid_shape(Params const& params) {
    int grid_m = (params.M + BLK_M - 1) / BLK_M;
    int grid_n = (params.N + BLK_N - 1) / BLK_N;
    return compat::dim3(1, grid_n, grid_m);
  }

  static compat::dim3 get_block_shape() {
    // Number of work-items per workgroup MUST equal the TiledMMA thread count
    // (num_subgroups * sg_size). cute::size(TiledMMA{}) gives exactly this, as
    // the canonical Xe GEMM does (`local = {size(mma), 1}`).
    constexpr int num_threads = cute::size(typename CollectiveMainloop::TiledMMA{});
    return compat::dim3(num_threads, 1, 1);
  }

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  // Kernel entry point
  CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
    using namespace sycl::ext::oneapi;

    // Cast shared memory buffer to SharedStorage
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    int thr_id = this_work_item::get_nd_item<3>().get_local_id(0);
    int blk_m = this_work_item::get_nd_item<3>().get_group(2);
    int blk_n = this_work_item::get_nd_item<3>().get_group(1);

    // Create global tensor layouts.
    // A: (M,K) row-major (contiguous K).
    auto layout_A = make_layout(make_shape(params.M, params.K), make_stride(params.stride_A_m, Int<1>{}));
    // B: PyTorch passes [K,N] row-major (ptr[k*N+n]). The canonical Xe GEMM (and
    // the block-2D copy_B / Step<X,_1,_1> tiling in the mainloop) expect B as
    // (N,K). The SAME memory is exactly (N,K) with stride (1, N): element (n,k)
    // lives at n*1 + k*N == k*N+n. stride_B_n==1 (contiguous), stride_B_k==N.
    auto layout_B = make_layout(make_shape(params.N, params.K), make_stride(Int<1>{}, params.stride_B_k));
    auto layout_C = make_layout(make_shape(params.M, params.N), make_stride(params.stride_C_m, Int<1>{}));
    auto layout_sqrsum = make_layout(make_shape(params.M), make_stride(Int<1>{}));

    Tensor A = make_tensor(make_gmem_ptr(params.ptr_A), layout_A);
    Tensor B = make_tensor(make_gmem_ptr(params.ptr_B), layout_B);  // (N,K)
    Tensor C = make_tensor(make_gmem_ptr(params.ptr_C), layout_C);
    Tensor SqrSum = make_tensor(make_gmem_ptr(params.ptr_sqrsum), layout_sqrsum);

    // Create 2D views
    auto A_2D = A(append<rank_v<decltype(A)>>(make_coord(_, _), 0));
    auto B_2D = B(append<rank_v<decltype(B)>>(make_coord(_, _), 0));

    // Initialize mainloop
    CollectiveMainloop mainloop(params.mainloop, shared_storage.mainloop);

    // Create output accumulators
    typename CollectiveMainloop::FragGemm tC;
    typename CollectiveMainloop::FragSqrSum tSqrSum;

    // Run mainloop: computes tC = A @ B and tSqrSum = sum(A^2) per thread
    mainloop(A_2D, B_2D, tC, tSqrSum, make_coord(blk_m, blk_n), thr_id);

    //
    // Epilogue: Write C matrix and sqrsum
    //

    // 1. Write C matrix using the canonical Xe GEMM block-2D store (xe_gemm.cpp):
    //      gC     = local_tile of C's coords, Step<_1,_1,X>  -> (BLK_M,BLK_N)
    //      copy_c = make_block_2d_copy_D(mma, C)
    //      tCgC   = thr_mma.partition_C(gC)
    //      copy(copy_c, <frag>, tCgC)
    //    The example's C is float (== accumulator), so it stores tC directly.
    //    Our C is fp16, so the block-2D store atom (16-bit) would mismatch the
    //    float (32-bit) accumulator. We first convert tC -> a SAME-LAYOUT fp16
    //    fragment (element-wise, no coordinate logic), then run the proven copy.
    typename CollectiveMainloop::TiledMMA mma{};
    auto thr_mma = mma.get_slice(thr_id);

    auto cC = make_identity_tensor(C.shape());                                   // global (M,N) coords
    auto gC = local_tile(cC, mma.tile_mnk(), make_coord(blk_m, blk_n, 0),
                         Step<_1, _1, X>{});                                     // (BLK_M,BLK_N)
    auto copy_c = make_block_2d_copy_D(mma, C);
    auto tCgC = thr_mma.partition_C(gC);

    // Convert the float accumulator to an fp16 fragment with identical layout.
    auto tCrC = make_fragment_like<ElementC>(tC);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tC); ++i) {
      tCrC(i) = static_cast<ElementC>(tC(i));
    }
    copy(copy_c, tCrC, tCgC);

    int row_offset = blk_m * BLK_M;

    // 2. Write sqrsum - one value per row of A
    // tSqrSum is an array[BLK_M] holding this work-item's per-row partial sums.
    // Each work-item only touched a subset of the tile's rows; atomics reduce the
    // partial sums across all work-items that touched each row.
    //
    // sqrsum depends only on A (not B or N), so every workgroup along the N grid
    // dimension computes the SAME square sums for its M-tile. Only the blk_n == 0
    // column writes them out; otherwise the result would be (grid_n) times too large.
    if (blk_n == 0) {
      CUTLASS_PRAGMA_UNROLL
      for (int m_local = 0; m_local < BLK_M; ++m_local) {
        int global_row = row_offset + m_local;

        if (global_row < params.M) {
          ElementSqrSum local_sum = tSqrSum[m_local];

          if (local_sum != ElementSqrSum(0)) {  // Skip work-items that missed this row
            sycl::atomic_ref<ElementSqrSum,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                atomic_sqrsum(params.ptr_sqrsum[global_row]);
            atomic_sqrsum.fetch_add(local_sum);
          }
        }
      }
    }
  }
};

}  // namespace cutlass::gemm_sqrsum::kernel
