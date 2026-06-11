/***************************************************************************************************
 * Copyright 2025 SGLang Team. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 **************************************************************************************************/
/*! \file
    \brief XPU Row-wise Square Sum Collective Mainloop
*/

#pragma once

#include <cute/tensor.hpp>
#include "cutlass/cutlass.h"

namespace cutlass::reduction::collective {
using namespace cute;

///////////////////////////////////////////////////////////////////////////////
/// Row-wise square sum mainloop: loads row data and computes partial sum of squares
///
/// Template Parameters:
///   TileShape_: Shape of tile to process (e.g. Shape<_1, _256>)
///   Element_: Input element type
///   StrideA_: Input stride type
///////////////////////////////////////////////////////////////////////////////
template <
    class TileShape_,
    class Element_,
    class StrideA_>
struct XeRowSquareSumMainloop {
  using TileShape = TileShape_;
  using Element = Element_;
  using StrideA = StrideA_;
  using ElementAccumulator = float;  // Always use float for accumulation

  static constexpr int kTileM = size<0>(TileShape{});  // Rows per tile (usually 1)
  static constexpr int kTileN = size<1>(TileShape{});  // Columns per tile

  // No shared storage needed for simple row-wise sum
  struct SharedStorage {};

  ///////////////////////////////////////////////////////////////////////////////
  /// Arguments: pointers and problem size
  ///////////////////////////////////////////////////////////////////////////////
  struct Arguments {
    const Element* ptr_A = nullptr;
    StrideA stride_A{};
    int M = 0;  // Number of rows
    int N = 0;  // Number of columns
  };

  ///////////////////////////////////////////////////////////////////////////////
  /// Params: runtime kernel parameters
  ///////////////////////////////////////////////////////////////////////////////
  struct Params {
    const Element* ptr_A;
    StrideA stride_A;
    int M;
    int N;
  };

  ///////////////////////////////////////////////////////////////////////////////
  /// to_underlying_arguments: convert Arguments to Params
  ///////////////////////////////////////////////////////////////////////////////
  static Params to_underlying_arguments(Arguments const& args) {
    return {
        args.ptr_A,
        args.stride_A,
        args.M,
        args.N};
  }

  ///////////////////////////////////////////////////////////////////////////////
  /// can_implement: validate problem can be executed
  ///////////////////////////////////////////////////////////////////////////////
  static bool can_implement(Arguments const& args) {
    return args.M > 0 && args.N > 0 && args.ptr_A != nullptr;
  }

  ///////////////////////////////////////////////////////////////////////////////
  /// Mainloop operator: load data and compute partial sum
  ///////////////////////////////////////////////////////////////////////////////
  template <class FragmentAcc>
  CUTLASS_DEVICE void operator()(
      Params const& params,
      SharedStorage& /* shared_storage */,
      FragmentAcc& accumulator,
      int row_idx,
      sycl::nd_item<1> item) const {

    const int thread_idx = item.get_local_id(0);
    const int threads_per_row = item.get_local_range(0);

    if (row_idx >= params.M)
      return;

    // Create tensor view for this row
    // params.stride_A is Stride<int, Int<1>>, we need to extract the row stride (first element)
    const Element* row_ptr = params.ptr_A + row_idx * get<0>(params.stride_A);

    // Each thread processes strided elements
    ElementAccumulator thread_sum = 0.0f;

    // Process in tiles for better cache locality
    constexpr int kProcessingTile = 256;  // Process 256 elements at a time
    const int num_tiles = (params.N + kProcessingTile - 1) / kProcessingTile;

#pragma unroll 2
    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
      const int tile_start = tile_idx * kProcessingTile;
      const int tile_end = cute::min(tile_start + kProcessingTile, params.N);

      // Each thread processes its portion of the tile
      // Compute SQUARE sum: sum(x^2) instead of sum(x)
      for (int col = tile_start + thread_idx; col < tile_end; col += threads_per_row) {
        ElementAccumulator value = static_cast<ElementAccumulator>(row_ptr[col]);
        thread_sum += value * value;  // Square each element
      }
    }

    // Store in accumulator
    accumulator = thread_sum;
  }
};

}  // namespace cutlass::reduction::collective
