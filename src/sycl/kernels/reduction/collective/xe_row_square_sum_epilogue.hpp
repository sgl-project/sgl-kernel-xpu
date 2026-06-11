/***************************************************************************************************
 * Copyright 2025 SGLang Team. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 **************************************************************************************************/
/*! \file
    \brief XPU Row-wise Square Sum Collective Epilogue
*/

#pragma once

#include <cute/tensor.hpp>
#include <sycl/sycl.hpp>
#include "cutlass/cutlass.h"

namespace cutlass::reduction::collective {
using namespace cute;

///////////////////////////////////////////////////////////////////////////////
/// Row-wise sum epilogue: reduces thread accumulators and writes output
///
/// Template Parameters:
///   ElementOutput_: Output element type
///   ElementAccumulator_: Accumulator element type
///   StrideD_: Output stride type
///////////////////////////////////////////////////////////////////////////////
template <
    class ElementOutput_,
    class ElementAccumulator_,
    class StrideD_>
struct XeRowSquareSumEpilogue {
  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using StrideD = StrideD_;

  static constexpr int kSubgroupSize = 16;  // Intel XPU subgroup size
  static constexpr int kThreadsPerRow = 256;  // Must match kernel config
  static constexpr int kNumSubgroups = kThreadsPerRow / kSubgroupSize;

  ///////////////////////////////////////////////////////////////////////////////
  /// Shared storage for reduction tree
  ///////////////////////////////////////////////////////////////////////////////
  struct SharedStorage {
    alignas(16) float subgroup_results[kNumSubgroups];
  };

  ///////////////////////////////////////////////////////////////////////////////
  /// Arguments
  ///////////////////////////////////////////////////////////////////////////////
  struct Arguments {
    ElementOutput* ptr_D = nullptr;
    StrideD stride_D{};
    int M = 0;
  };

  ///////////////////////////////////////////////////////////////////////////////
  /// Params
  ///////////////////////////////////////////////////////////////////////////////
  struct Params {
    ElementOutput* ptr_D;
    StrideD stride_D;
    int M;
  };

  ///////////////////////////////////////////////////////////////////////////////
  /// to_underlying_arguments
  ///////////////////////////////////////////////////////////////////////////////
  static Params to_underlying_arguments(Arguments const& args) {
    return {
        args.ptr_D,
        args.stride_D,
        args.M};
  }

  ///////////////////////////////////////////////////////////////////////////////
  /// can_implement
  ///////////////////////////////////////////////////////////////////////////////
  static bool can_implement(Arguments const& args) {
    return args.M > 0 && args.ptr_D != nullptr;
  }

  ///////////////////////////////////////////////////////////////////////////////
  /// Epilogue operator: reduce and write output
  ///////////////////////////////////////////////////////////////////////////////
  template <class FragmentAcc>
  CUTLASS_DEVICE void operator()(
      Params const& params,
      SharedStorage& shared_storage,
      FragmentAcc const& accumulator,
      int row_idx,
      sycl::nd_item<1> item) const {

    const int thread_idx = item.get_local_id(0);
    const int sg_id = thread_idx / kSubgroupSize;
    const int lane_id = thread_idx % kSubgroupSize;

    // Step 1: Subgroup-level reduction using shuffle
    auto sg = item.get_sub_group();
    ElementAccumulator sg_sum = accumulator;

#pragma unroll
    for (int offset = kSubgroupSize / 2; offset > 0; offset /= 2) {
      sg_sum += sycl::shift_group_left(sg, sg_sum, offset);
    }

    // Step 2: First lane of each subgroup writes to shared memory
    if (lane_id == 0) {
      shared_storage.subgroup_results[sg_id] = sg_sum;
    }

    // Synchronize work-group
    item.barrier(sycl::access::fence_space::local_space);

    // Step 3: Final reduction across subgroups (done by first subgroup)
    if (sg_id == 0) {
      ElementAccumulator final_sum = (lane_id < kNumSubgroups) ? shared_storage.subgroup_results[lane_id] : 0.0f;

      // Reduce within first subgroup
#pragma unroll
      for (int offset = kSubgroupSize / 2; offset > 0; offset /= 2) {
        final_sum += sycl::shift_group_left(sg, final_sum, offset);
      }

      // Thread 0 writes final result
      if (lane_id == 0 && row_idx < params.M) {
        // stride_D is Int<1>, just use row_idx directly since it's contiguous
        params.ptr_D[row_idx] = static_cast<ElementOutput>(final_sum);
      }
    }
  }
};

}  // namespace cutlass::reduction::collective
