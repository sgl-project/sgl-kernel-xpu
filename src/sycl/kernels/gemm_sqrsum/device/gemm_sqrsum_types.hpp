/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
/*!
  \file
  \brief Type definitions and runner for GEMM + Square Sum kernel
*/

#pragma once

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <sycl/sycl.hpp>

#include "../../../Utils.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "sycl/kernels/gemm_sqrsum/collective/xe_gemm_sqrsum_mainloop.hpp"
#include "sycl/kernels/gemm_sqrsum/kernel/xe_gemm_sqrsum_kernel.hpp"
#include "sycl/kernels/gemm_sqrsum/device/gemm_sqrsum_runner.hpp"

using namespace cute;

//----------------- Element type conversion --------------------//
template <typename T>
struct ToCutlassElementType {
  using type = T;
};

template <>
struct ToCutlassElementType<sycl::half> {
  using type = cutlass::half_t;
};

template <>
struct ToCutlassElementType<sycl::ext::oneapi::bfloat16> {
  using type = cutlass::bfloat16_t;
};

template <>
struct ToCutlassElementType<float> {
  using type = float;
};

//----------------- Tile size options --------------------//
template <int TileM_, int TileN_, int TileK_>
struct TileSizeOption {
  static constexpr int TileM = TileM_;
  static constexpr int TileN = TileN_;
  static constexpr int TileK = TileK_;
};

//----------------- Kernel configuration --------------------//
// This creates a specific instantiation of the GEMM+SqrSum kernel
template <typename Element_, typename TileSizeOpt_>
struct GemmSqrSumXe {
  using Element = typename ToCutlassElementType<Element_>::type;
  using TileSizeOpt = TileSizeOpt_;

  // Define tile shape
  using TileShape = Shape<Int<TileSizeOpt::TileM>, Int<TileSizeOpt::TileN>, Int<TileSizeOpt::TileK>>;

  // MMA Atom: XE DPAS operation
  // XE_DPAS_TT<M, AccumType, InputType>
  //   M = 8 (DPAS atom size in M dimension)
  //   AccumType = float (accumulator precision)
  //   InputType = Element (input type: half/bfloat16/float)
  using MMAOperation = XE_DPAS_TT<8, float, Element>;
  using MmaAtom = MMA_Atom<MMAOperation>;

  // Subgroup layout: how subgroups tile the work, matching the canonical Xe
  // GEMM tutorial (examples/cute/tutorial/xe_gemm.cpp). An 8x4 (M x N) layout
  // gives 32 subgroups => 32 * sg_size(16) = 512 work-items per workgroup,
  // which is what get_block_shape() must launch. N-major stride <4,1,0>.
  using SubgroupLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;

  // TiledMMA: tiles MMA atoms across subgroups
  using TiledMma = typename TiledMMAHelper<MmaAtom, Layout<TileShape>, SubgroupLayout>::TiledMMA;

  using DispatchPolicy = cutlass::gemm_sqrsum::XeDefault<1>;

  // Placeholder tensor types used only to deduce TiledCopyA / TiledCopyB. Their
  // SHAPE ORIENTATION and contiguous-stride position must match the runtime
  // tensors the kernel builds, or the deduced copies tile the wrong axes.
  //   A: (M,K) row-major  -> shape (m,k), stride (k,1)
  //   B: (N,K) col-major  -> shape (n,k), stride (1,n)  [the Bᵀ view of PyTorch's
  //      [K,N] row-major buffer; this is what copy_B / the mainloop tiling expect]
  using TensorA = decltype(make_tensor(
      make_gmem_ptr<Element const>(nullptr),
      make_layout(make_shape(Int<256>{}, Int<128>{}), make_stride(Int<128>{}, Int<1>{}))));
  using TensorB = decltype(make_tensor(
      make_gmem_ptr<Element const>(nullptr),
      make_layout(make_shape(Int<256>{}, Int<128>{}), make_stride(Int<1>{}, Int<256>{}))));

  // Assemble collective mainloop
  using CollectiveMainloop = cutlass::gemm_sqrsum::collective::XeGemmSqrSumMainloop<
      DispatchPolicy,
      TiledMma,
      TensorA,
      TensorB>;

  // Assemble the kernel
  using Kernel = cutlass::gemm_sqrsum::kernel::GemmSqrSumKernel<CollectiveMainloop>;
};

//----------------- Launch function --------------------//
template <typename Element, typename TileSizeOpt>
inline void runGemmSqrSumImpl(
    at::Tensor& C,           // Output: [M, N]
    at::Tensor& sqrsum,      // Output: [M] row-wise square sums
    const at::Tensor& A,     // Input: [M, K]
    const at::Tensor& B) {   // Input: [K, N]

  // Get dimensions
  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);

  // Verify shapes
  TORCH_CHECK(A.size(1) == B.size(0), "A.K must match B.K for GEMM");
  TORCH_CHECK(C.size(0) == M && C.size(1) == N, "C shape mismatch");
  TORCH_CHECK(sqrsum.size(0) == M, "sqrsum size mismatch");

  // Initialize outputs to zero (epilogue uses atomic adds)
  C.zero_();
  sqrsum.zero_();

  // Get CUTLASS element type
  using CutlassElement = typename ToCutlassElementType<Element>::type;

  // Get kernel configuration
  using KernelConfig = GemmSqrSumXe<Element, TileSizeOpt>;
  using Kernel = typename KernelConfig::Kernel;
  using Runner = cutlass::gemm_sqrsum::device::GemmSqrSum<Kernel>;

  // Setup kernel arguments
  typename Kernel::Arguments args;
  args.M = M;
  args.K = K;
  args.N = N;

  // Fetch raw pointers via the untyped data_ptr() (returns void*) and reinterpret.
  // Using data_ptr<Element>() would require a PyTorch data_ptr specialization for
  // the kernel element type; sycl::ext::oneapi::bfloat16 has none, so the typed
  // form leaves an undefined symbol at link time. The untyped form sidesteps that.

  // Input A: [M, K] - row-major
  args.ptr_A = reinterpret_cast<CutlassElement const*>(A.data_ptr());
  args.stride_A_m = K;  // Row stride
  args.stride_A_k = 1;  // Column stride

  // Input B: [K, N] - row-major
  args.ptr_B = reinterpret_cast<CutlassElement const*>(B.data_ptr());
  args.stride_B_k = N;  // Row stride
  args.stride_B_n = 1;  // Column stride

  // Output C: [M, N] - row-major
  args.ptr_C = reinterpret_cast<CutlassElement*>(C.data_ptr());
  args.stride_C_m = N;  // Row stride
  args.stride_C_n = 1;  // Column stride

  // Output sqrsum: [M] - 1D vector (always float32)
  args.ptr_sqrsum = sqrsum.data_ptr<float>();

  // Mainloop arguments (empty for now)
  args.mainloop = typename KernelConfig::CollectiveMainloop::Arguments{};

  // Launch kernel
  Runner runner;
  auto status = runner.run(args, nullptr, c10::xpu::getCurrentXPUStream().queue());

  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS GEMM+SqrSum kernel failed with status: ", int(status));
}

template <typename Element, typename TileSizeOpt>
inline void runGemmSqrSum(
    at::Tensor& C,
    at::Tensor& sqrsum,
    const at::Tensor& A,
    const at::Tensor& B) {

  runGemmSqrSumImpl<Element, TileSizeOpt>(C, sqrsum, A, B);
}
