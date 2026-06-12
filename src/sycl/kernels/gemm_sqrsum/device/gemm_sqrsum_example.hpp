/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
/*! \file
    \brief Example usage of GEMM + Square Sum kernel
*/

#pragma once

#include <sycl/sycl.hpp>
#include "cute/tensor.hpp"
#include "../kernel/xe_gemm_sqrsum_kernel.hpp"

namespace cutlass::gemm_sqrsum {

using namespace cute;

// Example kernel configuration for FP16
// Computes C = A @ B and sqrsum[i] = sum(A[i,:]^2)
template <typename Element_>
struct GemmSqrSumConfig {
  using Element = Element_;

  // Tile shape: 256x256x16 (M x N x K)
  using TileShape = Shape<_256, _256, _16>;

  // MMA atom configuration
  using MMA_Atom = MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>;

  // Tiled MMA: replicate the atom across subgroups
  using TiledMMA = TiledMMA<
      MMA_Atom,
      Layout<Shape<_16, _16, _1>>,  // Subgroup layout
      Tile<_32, _32, _16>>;         // Value layout

  // Mainloop dispatch policy
  using DispatchPolicy = cutlass::gemm_sqrsum::XeDefault<1>;

  // Create tensor types
  template <typename Layout>
  using TensorA = decltype(make_tensor(make_gmem_ptr<Element const>(nullptr), Layout{}));

  template <typename Layout>
  using TensorB = decltype(make_tensor(make_gmem_ptr<Element const>(nullptr), Layout{}));

  // Mainloop type
  using CollectiveMainloop = cutlass::gemm_sqrsum::collective::XeGemmSqrSumMainloop<
      DispatchPolicy,
      TiledMMA,
      TensorA<Layout<Shape<_256, _16>>>,  // Example layouts
      TensorB<Layout<Shape<_16, _256>>>,
      void,  // Auto TiledCopyA
      void>; // Auto TiledCopyB

  // Kernel type
  using Kernel = cutlass::gemm_sqrsum::kernel::GemmSqrSumKernel<CollectiveMainloop>;
};

// Launch function example
template <typename Element>
void launch_gemm_sqrsum(
    sycl::queue& queue,
    int M, int K, int N,
    Element const* A,  // [M, K]
    Element const* B,  // [K, N]
    Element* C,        // [M, N] output
    Element* sqrsum) { // [M] row-wise square sums output

  using Config = GemmSqrSumConfig<Element>;
  using Kernel = typename Config::Kernel;
  using Arguments = typename Kernel::Arguments;
  using Params = typename Kernel::Params;

  // Setup arguments
  Arguments args{
      {},  // mainloop args
      M, K, N,
      A, K, 1,      // A: stride_m=K, stride_k=1 (row-major)
      B, N, 1,      // B: stride_k=N, stride_n=1 (row-major)
      C, N, 1,      // C: stride_m=N, stride_n=1 (row-major)
      sqrsum
  };

  Params params = Kernel::to_underlying_arguments(args, nullptr);

  // Calculate grid dimensions
  constexpr int BLK_M = decltype(Config::TileShape{})::value_type::template get<0>();
  constexpr int BLK_N = decltype(Config::TileShape{})::value_type::template get<1>();

  int grid_m = (M + BLK_M - 1) / BLK_M;
  int grid_n = (N + BLK_N - 1) / BLK_N;

  // TODO: Get proper workgroup size from TiledMMA configuration
  constexpr int WORKGROUP_SIZE = 256;

  // Launch kernel
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<typename Kernel::SharedStorage, 0> shared_storage(cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, grid_n, grid_m) * sycl::range<3>(1, 1, WORKGROUP_SIZE),
            sycl::range<3>(1, 1, WORKGROUP_SIZE)),
        [=](sycl::nd_item<3> item) {
          Kernel kernel;
          kernel(params, *shared_storage.get_multi_ptr<sycl::access::decorated::no>().get_raw());
        });
  });
}

}  // namespace cutlass::gemm_sqrsum

/*
 * Usage Example:
 *
 * #include "device/gemm_sqrsum_example.hpp"
 *
 * sycl::queue q;
 * int M = 1024, K = 512, N = 2048;
 *
 * sycl::half* A = sycl::malloc_device<sycl::half>(M * K, q);
 * sycl::half* B = sycl::malloc_device<sycl::half>(K * N, q);
 * sycl::half* C = sycl::malloc_device<sycl::half>(M * N, q);
 * sycl::half* sqrsum = sycl::malloc_device<sycl::half>(M, q);
 *
 * // Fill A and B with data...
 *
 * cutlass::gemm_sqrsum::launch_gemm_sqrsum(q, M, K, N, A, B, C, sqrsum);
 * q.wait();
 *
 * // C now contains A @ B
 * // sqrsum[i] contains sum(A[i,:]^2) for each row i
 */
