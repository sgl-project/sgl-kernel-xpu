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

    // (M,N) float scratch: the square-sum-as-GEMM accumulator is stored here via
    // the proven block-2D copy engine. Every column equals sqrsum[row]; the host
    // dispatch slices column 0 into the [M] sqrsum output.
    ElementSqrSum* ptr_sqrsum_scratch;
    int64_t stride_sqsc_m;
    int64_t stride_sqsc_n;
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

    ElementSqrSum* ptr_sqrsum_scratch;
    int64_t stride_sqsc_m;
    int64_t stride_sqsc_n;
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
        args.ptr_sqrsum,
        args.ptr_sqrsum_scratch,
        args.stride_sqsc_m,
        args.stride_sqsc_n};
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

    // Work-item / workgroup IDs. compat::dim3(x,y,z) -> sycl::range<3> maps
    // x->dim2, y->dim1, z->dim0. get_block_shape() = dim3(512,1,1) puts the 512
    // work-items on nd-range dim 2 (== ThreadIdxX == get_local_id(2)).
    // get_grid_shape() = dim3(1, grid_n, grid_m) puts grid_m on dim 0, grid_n on
    // dim 1. (The previous code read local_id(0)/group(2), which are the size-1
    // dims, so every work-item saw thr_id 0 and only one subgroup tile computed.)
    auto nd = this_work_item::get_nd_item<3>();
    int thr_id = int(nd.get_local_id(2));
    int blk_m = int(nd.get_group(0));
    int blk_n = int(nd.get_group(1));

    // Create global tensor layouts.
    // A: (M,K) row-major (contiguous K).
    auto layout_A = make_layout(make_shape(params.M, params.K), make_stride(params.stride_A_m, Int<1>{}));
    // B: passed as genuine [N,K] row-major (fn = [24,16384] = [N,K]). Element
    // (n,k) lives at n*stride_B_k + k, i.e. stride (stride_B_k==K, 1) with K
    // contiguous. This is the canonical "Bᵀ" orientation the block-2D copy_B
    // and the select<1,2> mainloop tiling expect (xe_gemm.cpp). No more
    // col-major-view-of-[K,N] trick.
    auto layout_B = make_layout(make_shape(params.N, params.K), make_stride(params.stride_B_k, Int<1>{}));
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
    //      copy(copy_c, tC, tCgC)
    //    C is now fp32 == the MMA accumulator type, exactly like the canonical
    //    example, so the float accumulator stores straight through the block-2D
    //    copy with no element-wise convert (the old fp16 C needed a convert
    //    because the 16-bit store atom mismatched the 32-bit accumulator).
    typename CollectiveMainloop::TiledMMA mma{};
    auto thr_mma = mma.get_slice(thr_id);

    auto cC = make_identity_tensor(C.shape());                                   // global (M,N) coords
    auto gC = local_tile(cC, mma.tile_mnk(), make_coord(blk_m, blk_n, 0),
                         Step<_1, _1, X>{});                                     // (BLK_M,BLK_N)
    auto copy_c = make_block_2d_copy_D(mma, C);
    auto tCgC = thr_mma.partition_C(gC);

    copy(copy_c, tC, tCgC);

    // 2. Write the square-sum accumulator to the (M,N) float scratch using the
    //    SAME proven block-2D store engine as C. tSqrSum is float and the scratch
    //    is float, so no conversion is needed and the store atom matches. Manual
    //    get<0>/get<1> on the MMA C-partition mis-decodes rows (the Xe coordinate
    //    is linearized), so we must let the copy engine handle the layout — just
    //    as it does for C. The host then slices column 0 of the scratch into the
    //    [M] sqrsum output (every column of tSqrSum equals sum_k A[row,k]^2).
    auto layout_Ssc = make_layout(make_shape(params.M, params.N),
                                  make_stride(params.stride_sqsc_m, Int<1>{}));
    Tensor Ssc = make_tensor(make_gmem_ptr(params.ptr_sqrsum_scratch), layout_Ssc);
    auto cSsc = make_identity_tensor(Ssc.shape());
    auto gSsc = local_tile(cSsc, mma.tile_mnk(), make_coord(blk_m, blk_n, 0),
                           Step<_1, _1, X>{});
    auto copy_sq = make_block_2d_copy_D(mma, Ssc);
    auto tSgSsc = thr_mma.partition_C(gSsc);
    copy(copy_sq, tSqrSum, tSgSsc);
  }
};

}  // namespace cutlass::gemm_sqrsum::kernel
