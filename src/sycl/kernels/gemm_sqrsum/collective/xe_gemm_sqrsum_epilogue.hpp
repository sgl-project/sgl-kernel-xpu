/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
/*! \file
    \brief GEMM + Square Sum Epilogue - writes C and sqrsum to global memory
*/

#pragma once

#include <sycl/sycl.hpp>

#include "cute/algorithm/copy.hpp"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"

namespace cutlass::gemm_sqrsum::collective {
using namespace cute;

template <class TiledMMA_, class TileShape_, class TensorC_, class TensorSqrSum_>
struct XeGemmSqrSumEpilogue {
  using TiledMMA = TiledMMA_;
  using TileShape = TileShape_;
  using TensorC = TensorC_;
  using TensorSqrSum = TensorSqrSum_;
  using ElementC = typename TensorC::value_type;
  using ElementSqrSum = typename TensorSqrSum::value_type;

  struct Arguments {
    ElementC* ptr_C = nullptr;
    int64_t stride_C_m = 0;
    int64_t stride_C_n = 0;
    ElementSqrSum* ptr_sqrsum = nullptr;
  };

  using Params = Arguments;
  struct SharedStorage {};

  Params params;
  SharedStorage& shared;

  XeGemmSqrSumEpilogue(Params const& params_, SharedStorage& shared_)
      : params(params_), shared(shared_) {}

  static constexpr Params to_underlying_arguments(Arguments const& args, void* /* workspace */) {
    return args;
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) {
    return true;
  }

  // Write C matrix and sqrsum vector to global memory
  template <typename TensorC2D, typename FragC, typename FragSqrSum, typename BlkCoord>
  CUTLASS_DEVICE void operator()(
      TensorC2D const& C,        // Global C tensor (2D view)
      FragC const& tC,           // GEMM result accumulator
      FragSqrSum const& tSqrSum, // Square sum accumulator
      BlkCoord blk_coord,        // Block coordinates (blk_m, blk_n)
      int thr_id,                // Thread ID
      int M, int N) {            // Problem dimensions

    using namespace cute;
    using namespace sycl::ext::oneapi;

    int blk_m = get<0>(blk_coord);
    int blk_n = get<1>(blk_coord);

    //
    // Write C matrix using naive element-wise stores
    //
    // Create coordinate tensor for the C tile
    auto cC = make_identity_tensor(C.shape());
    auto gC = local_tile(cC, select<0,1>(TileShape{}), blk_coord);

    // Partition for this thread using TiledMMA
    auto thr_mma = TiledMMA{}.get_slice(thr_id);
    auto tCcC = thr_mma.partition_C(gC);  // Coordinates

    // Write C elements
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tC); ++i) {
      auto coord = tCcC(i);
      int row = get<0>(coord);
      int col = get<1>(coord);
      int global_row = blk_m * get<0>(TileShape{}) + row;
      int global_col = blk_n * get<1>(TileShape{}) + col;

      if (global_row < M && global_col < N) {
        params.ptr_C[global_row * params.stride_C_m + global_col * params.stride_C_n] =
            static_cast<ElementC>(tC(i));
      }
    }

    //
    // Write sqrsum using atomics with proper row mapping
    //
    // tSqrSum is a row fragment - each element corresponds to a row in the tile
    // We need to map fragment indices to global row indices
    auto sg = this_work_item::get_sub_group();

    // Partition identity tensor to get row coordinates
    auto tCcC_row = thr_mma.partition_C(make_identity_tensor(select<0,1>(TileShape{})));

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tSqrSum); ++i) {
      // Get the row index for this fragment element
      // Since tSqrSum is reduced over columns, we look at any element in that row
      int local_row = get<0>(tCcC_row(i * size<1>(tCcC)));  // First element of row i
      int global_row = blk_m * get<0>(TileShape{}) + local_row;

      if (global_row < M) {
        // Reduce across subgroup (in case multiple threads have same row)
        ElementSqrSum sum_val = tSqrSum(i);
        ElementSqrSum reduced_val = sycl::reduce_over_group(sg, sum_val, sycl::plus<ElementSqrSum>());

        // Thread 0 in subgroup writes
        if (sg.get_local_linear_id() == 0) {
          sycl::atomic_ref<ElementSqrSum,
                           sycl::memory_order::relaxed,
                           sycl::memory_scope::device,
                           sycl::access::address_space::global_space>
              atomic_sqrsum(params.ptr_sqrsum[global_row]);
          atomic_sqrsum.fetch_add(reduced_val);
        }
      }
    }
  }
};

}  // namespace cutlass::gemm_sqrsum::collective
