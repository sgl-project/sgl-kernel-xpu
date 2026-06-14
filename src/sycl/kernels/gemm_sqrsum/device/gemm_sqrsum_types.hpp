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
#include "cutlass/numeric_types.h"
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

template <>
struct ToCutlassElementType<cutlass::tfloat32_t> {
  using type = cutlass::tfloat32_t;
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
  //   A: (M,K) row-major  -> shape (m,k), stride (k,1)  [K-contiguous]
  //   B: (N,K) row-major  -> shape (n,k), stride (k,1)  [K-contiguous] — this is
  //      fn = [N,K] = [24,16384] passed DIRECTLY (no col-major-of-[K,N] trick).
  //      K-contiguous B is the canonical "Bᵀ" case in xe_gemm.cpp's
  //      choose_tiled_mma (b_n == false), which the block-2D copy_B and the
  //      select<1,2> mainloop tiling expect.
  using TensorA = decltype(make_tensor(
      make_gmem_ptr<Element const>(nullptr),
      make_layout(make_shape(Int<256>{}, Int<128>{}), make_stride(Int<128>{}, Int<1>{}))));
  using TensorB = decltype(make_tensor(
      make_gmem_ptr<Element const>(nullptr),
      make_layout(make_shape(Int<256>{}, Int<128>{}), make_stride(Int<128>{}, Int<1>{}))));

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
// Contract (mhc_pre GEMM+sqrsum stage):
//   A      : [M, K]  (residual.view(M, hc_hidden)); bf16/fp16/fp32 accepted
//   B      : [N, K]  (fn = [24, 16384]); fp32 — passed as genuine [N,K] row-major
//   C      : [M, N]  fp32  (== gemm_out_mul)
//   sqrsum : [M]     fp32  (== gemm_out_sqrsum), sqrsum[m] = sum_k A[m,k]^2
//
// Element is the MMA compute type. For the production path Element ==
// cutlass::tfloat32_t: A is widened to fp32 and B (already fp32) is taken as-is,
// then both are bit-reinterpreted to tf32 (tfloat32_t stores raw fp32 bits; the
// DPAS unit truncates the mantissa at load — ~10 bits kept vs bf16's 7). This is
// the "proper" mixed-precision path; the bf16-downcast path (HcPreGemm-style)
// lost too much precision on B.
template <typename Element, typename TileSizeOpt>
inline void runGemmSqrSumImpl(
    at::Tensor& C,           // Output: [M, N] fp32
    at::Tensor& sqrsum,      // Output: [M] fp32
    const at::Tensor& A,     // Input: [M, K]
    const at::Tensor& B) {   // Input: [N, K]

  // Get CUTLASS element type
  using CutlassElement = typename ToCutlassElementType<Element>::type;
  constexpr bool kIsTf32 = std::is_same_v<CutlassElement, cutlass::tfloat32_t>;

  // Get dimensions. B is [N, K] (NOT [K, N]).
  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(0);

  // Verify shapes
  TORCH_CHECK(B.size(1) == K, "A.K (", K, ") must match B.K (", B.size(1), ") for GEMM");
  TORCH_CHECK(C.size(0) == M && C.size(1) == N, "C shape mismatch");
  TORCH_CHECK(sqrsum.size(0) == M, "sqrsum size mismatch");

  // Initialize outputs to zero (epilogue uses atomic adds)
  C.zero_();
  sqrsum.zero_();

  // Prepare A/B in the kernel's storage type. tfloat32_t is bit-compatible with
  // float (alignas(4){uint32_t}), so an fp32 buffer reinterprets to tf32 with no
  // copy. A is widened from bf16/fp16; B (fp32) is taken contiguous as-is.
  at::Tensor A_buf, B_buf;
  if constexpr (kIsTf32) {
    A_buf = A.scalar_type() == at::kFloat ? A.contiguous() : A.to(at::kFloat);
    B_buf = B.scalar_type() == at::kFloat ? B.contiguous() : B.to(at::kFloat);
  } else {
    A_buf = A.contiguous();
    B_buf = B.contiguous();
  }

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
  // the kernel element type; tfloat32_t/bfloat16 have none, so the typed form
  // leaves an undefined symbol at link time. The untyped form sidesteps that.

  // Input A: [M, K] - row-major (K contiguous)
  args.ptr_A = reinterpret_cast<CutlassElement const*>(A_buf.data_ptr());
  args.stride_A_m = K;  // M-row stride
  args.stride_A_k = 1;  // K stride (contiguous)

  // Input B: [N, K] - row-major (K contiguous). stride_B_k carries the N-row
  // stride (== K); layout_B is make_stride(stride_B_k, 1) over shape (N,K).
  args.ptr_B = reinterpret_cast<CutlassElement const*>(B_buf.data_ptr());
  args.stride_B_k = K;  // N-row stride
  args.stride_B_n = 1;  // K stride (contiguous)

  // Output C: [M, N] fp32 row-major. ElementC == float (the accumulator type).
  args.ptr_C = reinterpret_cast<typename Kernel::ElementC*>(C.data_ptr());
  args.stride_C_m = N;  // Row stride
  args.stride_C_n = 1;  // Column stride

  // Output sqrsum: [M] - 1D vector (always float32)
  args.ptr_sqrsum = sqrsum.data_ptr<float>();

  // (M,N) float scratch for the square-sum-as-GEMM accumulator. The kernel
  // stores the full A^2 @ ones product here via the proven block-2D copy (whose
  // coordinate handling is correct, unlike a manual per-element store). Every
  // column equals sum_k A[row,k]^2, so we slice column 0 into sqrsum afterward.
  auto sqsc = torch::empty({M, N}, sqrsum.options().dtype(torch::kFloat32));
  args.ptr_sqrsum_scratch = sqsc.data_ptr<float>();
  args.stride_sqsc_m = N;  // row-major
  args.stride_sqsc_n = 1;

  // Mainloop arguments (empty for now)
  args.mainloop = typename KernelConfig::CollectiveMainloop::Arguments{};

  // Launch kernel
  Runner runner;
  auto status = runner.run(args, nullptr, c10::xpu::getCurrentXPUStream().queue());

  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS GEMM+SqrSum kernel failed with status: ", int(status));

  // sqrsum[m] = scratch[m, 0]  (all columns are equal).
  sqrsum.copy_(sqsc.select(/*dim=*/1, /*index=*/0));
}

template <typename Element, typename TileSizeOpt>
inline void runGemmSqrSum(
    at::Tensor& C,
    at::Tensor& sqrsum,
    const at::Tensor& A,
    const at::Tensor& B) {

  runGemmSqrSumImpl<Element, TileSizeOpt>(C, sqrsum, A, B);
}
