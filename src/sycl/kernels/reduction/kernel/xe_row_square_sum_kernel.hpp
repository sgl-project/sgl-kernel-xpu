/***************************************************************************************************
 * Copyright 2025 SGLang Team. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 **************************************************************************************************/
/*! \file
    \brief XPU Row-wise Square Sum Kernel
*/

#pragma once

#include <cute/tensor.hpp>
#include <sycl/sycl.hpp>
#include "cutlass/cutlass.h"
#include "sycl/kernels/reduction/collective/xe_row_square_sum_epilogue.hpp"
#include "sycl/kernels/reduction/collective/xe_row_square_sum_mainloop.hpp"

namespace cutlass::reduction::kernel {
using namespace cute;

///////////////////////////////////////////////////////////////////////////////
/// XeRowSquareSumKernel: Orchestrates mainloop and epilogue for row-wise square sum
///
/// Template Parameters:
///   CollectiveMainloop_: Mainloop type (loads and accumulates)
///   CollectiveEpilogue_: Epilogue type (reduces and writes output)
///////////////////////////////////////////////////////////////////////////////
template <class CollectiveMainloop_, class CollectiveEpilogue_>
class XeRowSquareSumKernel {
 public:
  using CollectiveMainloop = CollectiveMainloop_;
  using CollectiveEpilogue = CollectiveEpilogue_;

  // Type aliases
  using Element = typename CollectiveMainloop::Element;
  using ElementOutput = typename CollectiveEpilogue::ElementOutput;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;

  using StrideA = typename CollectiveMainloop::StrideA;
  using StrideD = typename CollectiveEpilogue::StrideD;

  ///////////////////////////////////////////////////////////////////////////////
  /// Shared storage (union to save memory)
  ///////////////////////////////////////////////////////////////////////////////
  using MainloopSharedStorage = typename CollectiveMainloop::SharedStorage;
  using EpilogueSharedStorage = typename CollectiveEpilogue::SharedStorage;

  union SharedStorage {
    MainloopSharedStorage mainloop;
    EpilogueSharedStorage epilogue;
  };

  ///////////////////////////////////////////////////////////////////////////////
  /// Problem shape
  ///////////////////////////////////////////////////////////////////////////////
  struct ProblemShape {
    int M;  // Number of rows
    int N;  // Number of columns
  };

  ///////////////////////////////////////////////////////////////////////////////
  /// Kernel Arguments (user-facing)
  ///////////////////////////////////////////////////////////////////////////////
  struct Arguments {
    ProblemShape shape{};
    const Element* ptr_A = nullptr;
    StrideA stride_A{};
    ElementOutput* ptr_D = nullptr;
    StrideD stride_D{};
  };

  ///////////////////////////////////////////////////////////////////////////////
  /// Kernel Params (device-side)
  ///////////////////////////////////////////////////////////////////////////////
  struct Params {
    ProblemShape shape;
    typename CollectiveMainloop::Params mainloop;
    typename CollectiveEpilogue::Params epilogue;
  };

  ///////////////////////////////////////////////////////////////////////////////
  /// to_underlying_arguments: convert Arguments to Params
  ///////////////////////////////////////////////////////////////////////////////
  static Params to_underlying_arguments(Arguments const& args) {
    // Create mainloop arguments
    typename CollectiveMainloop::Arguments mainloop_args{
        args.ptr_A, args.stride_A, args.shape.M, args.shape.N};

    // Create epilogue arguments
    typename CollectiveEpilogue::Arguments epilogue_args{
        args.ptr_D, args.stride_D, args.shape.M};

    return {
        args.shape,
        CollectiveMainloop::to_underlying_arguments(mainloop_args),
        CollectiveEpilogue::to_underlying_arguments(epilogue_args)};
  }

  ///////////////////////////////////////////////////////////////////////////////
  /// can_implement: validate problem
  ///////////////////////////////////////////////////////////////////////////////
  static bool can_implement(Arguments const& args) {
    if (args.shape.M <= 0 || args.shape.N <= 0)
      return false;

    typename CollectiveMainloop::Arguments mainloop_args{
        args.ptr_A, args.stride_A, args.shape.M, args.shape.N};

    typename CollectiveEpilogue::Arguments epilogue_args{
        args.ptr_D, args.stride_D, args.shape.M};

    return CollectiveMainloop::can_implement(mainloop_args) && CollectiveEpilogue::can_implement(epilogue_args);
  }

  ///////////////////////////////////////////////////////////////////////////////
  /// get_workspace_size: no workspace needed
  ///////////////////////////////////////////////////////////////////////////////
  static size_t get_workspace_size(Arguments const& /* args */) {
    return 0;
  }

  ///////////////////////////////////////////////////////////////////////////////
  /// Kernel operator: main device-side execution
  ///////////////////////////////////////////////////////////////////////////////
  CUTLASS_DEVICE void operator()(Params const& params, SharedStorage& shared_storage, sycl::nd_item<1> item) const {
    // Each work-group processes one row
    const int row_idx = item.get_group(0);

    if (row_idx >= params.shape.M)
      return;

    // Accumulator (thread-local register)
    ElementAccumulator accumulator = 0.0f;

    // Step 1: Mainloop - load and accumulate
    CollectiveMainloop mainloop;
    mainloop(params.mainloop, shared_storage.mainloop, accumulator, row_idx, item);

    // Barrier before epilogue (shared memory reuse)
    item.barrier(sycl::access::fence_space::local_space);

    // Step 2: Epilogue - reduce and write output
    CollectiveEpilogue epilogue;
    epilogue(params.epilogue, shared_storage.epilogue, accumulator, row_idx, item);
  }

  ///////////////////////////////////////////////////////////////////////////////
  /// sycl_ker_config_convention: Setup shared memory for SYCL
  ///////////////////////////////////////////////////////////////////////////////
  CUTLASS_DEVICE void sycl_ker_config_convention(sycl::handler& /* cgh */) {}
};

}  // namespace cutlass::reduction::kernel
