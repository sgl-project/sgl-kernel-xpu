/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
/*!
  \file
  \brief Type definitions and runner for the GEMM + Square Sum kernel.

  Single concrete configuration: tf32 x tf32 -> fp32 DPAS, tile 64 x 32 x 16.
  (The kernel was generalized over dtype/tile during bring-up; production is
  tf32-only, so it is specialized here to the one instantiation that ships.)
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

//----------------- Kernel configuration --------------------//
// Concrete instantiation of the GEMM+SqrSum kernel: tf32 compute, tile 64x32x16.
struct GemmSqrSumXe {
  // MMA compute type. The production path is bf16(A) x fp32(B) -> fp32. The DPAS
  // atom is tf32 x tf32 -> fp32, so both operands reach the engine as tf32.
  //   B (fp32) is reinterpreted to tf32 (raw fp32 bits; DPAS truncates to ~10
  //     mantissa bits at load) — the genuine fp32->tf32 step, where the real
  //     precision lives. The bf16-downcast path lost too much precision on B.
  //   A (bf16) is loaded from gmem AS bf16 (ElementALoad) and converted bf16->tf32
  //     in-register at the reorder() before the MMA. A was bf16 to begin with, so
  //     tf32's 10-bit slot holds its 7 bits losslessly — converting in-register is
  //     bit-identical to the old host fp32-widen, but avoids hauling a 2x-wider
  //     fp32 copy of A through memory (the dominant traffic at large M).
  using Element = cutlass::tfloat32_t;

  // Global load type for A: bf16. The block-2D copy loads bf16 from gmem into a
  // bf16 fragment; reorder() converts it to the tf32 MMA fragment. (B has no
  // separate load type — its gmem tensor is fp32 bits reinterpreted as tf32.)
  using ElementALoad = cutlass::bfloat16_t;

  // Tile shape (M, N, K) = (64, 32, 16).
  //   TileM=64 = 8(atom M) * 8(SG_M) * 1 iter. Small M-tile => more workgroups
  //     (grid_m = ceil(M/64)) to fill the GPU, since N=24 gives grid_n=1 and all
  //     parallelism must come from grid_m. Also keeps the fp32 accumulator small
  //     (no register spill).
  //   TileN=32 = 16(atom N) * 2(SG_N) * 1 iter, masked to the production N=24.
  //   TileK=16 = the DPAS K depth for tf32 (256 / sizeof_bits).
  using TileShape = Shape<Int<64>, Int<32>, Int<16>>;

  // MMA Atom: XE DPAS, XE_DPAS_TT<atom_M=8, AccumType=float, InputType=tf32>.
  using MMAOperation = XE_DPAS_TT<8, float, Element>;
  using MmaAtom = MMA_Atom<MMAOperation>;

  // Subgroup layout (M x N x 1). MUST satisfy TileN == 16 * SG_N * iters and
  // TileM == 8 * SG_M * iters (the DPAS atom is Shape_MNK = <8, 16, K>, so N is
  // hardwired to 16 per atom). For TileN=32 we need SG_N = 2 (16*2 = 32, one
  // N-iteration). An 8x2 layout = 16 subgroups => 16 * sg_size(16) = 256
  // work-items/WG, which is what get_block_shape() launches. N-major stride
  // <SG_N,1,0> = <2,1,0>. KEEP IN SYNC with TileShape: 8x2 pairs with TileN=32.
  using SubgroupLayout = Layout<Shape<_8, _2, _1>, Stride<_2, _1, _0>>;

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
  //   A's value_type is ElementALoad (bf16) so the deduced TiledCopyA loads bf16;
  //   B's stays Element (tf32) — its fp32 gmem buffer is reinterpreted to tf32.
  using TensorA = decltype(make_tensor(
      make_gmem_ptr<ElementALoad const>(nullptr),
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
// Contract (mhc_pre GEMM+sqrsum stage — Design B: the kernel writes the K-split
// partials and the downstream hc_pre_big_fuse reduces them, so NOTHING is summed
// here):
//   A      : [M, K]            (residual.view(M, hc_hidden)); bf16/fp16/fp32
//   B      : [N, K]            (fn = [24, 16384]); fp32 — genuine [N,K] row-major
//   C      : [n_splits, M, N]  fp32  (== gemm_out_mul partials)
//   sqrsum : [n_splits, M]     fp32  (== gemm_out_sqrsum partials)
//             sqrsum[s,m] = sum_{k in split s} A[m,k]^2
// The K reduction is partitioned into `split_k = C.size(0)` slices; the caller
// (mhc_pre) picks that count (32 for the split-k path, 1 for the simple path)
// and pre-allocates the partial buffers. The fuse's `for split` loop collapses
// the leading axis, so the K-reduction is folded into work the fuse already does.
inline void runGemmSqrSum(
    at::Tensor& C,           // Output: [n_splits, M, N] fp32
    at::Tensor& sqrsum,      // Output: [n_splits, M] fp32
    const at::Tensor& A,     // Input: [M, K]
    const at::Tensor& B) {   // Input: [N, K]

  using Kernel = typename GemmSqrSumXe::Kernel;
  using Runner = cutlass::gemm_sqrsum::device::GemmSqrSum<Kernel>;

  // Get dimensions. B is [N, K] (NOT [K, N]).
  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(0);

  // Split-K count = the leading axis of the partial buffers. Each K-slice writes
  // one [M,N] slab; the fuse reduces over this axis. No host-side reduction.
  int split_k = C.size(0);

  // Verify shapes
  TORCH_CHECK(B.size(1) == K, "A.K (", K, ") must match B.K (", B.size(1), ") for GEMM");
  TORCH_CHECK(C.dim() == 3 && C.size(1) == M && C.size(2) == N,
              "C must be [n_splits, M, N]");
  TORCH_CHECK(sqrsum.dim() == 2 && sqrsum.size(0) == split_k && sqrsum.size(1) == M,
              "sqrsum must be [n_splits, M] matching C's leading dim");

  // Zero the outputs. The epilogue does a full block-2D store (overwrite, not
  // atomic add) over every (blk_m,blk_n) tile, so this is belt-and-suspenders for
  // the OOB-masked boundary tiles (ragged M, N=24<BLK_N) and any tail K-slice
  // that runs zero loop iterations.
  C.zero_();
  sqrsum.zero_();

  // Prepare A as bf16 and B as fp32 (reinterpreted to tf32 at load).
  //   A: loaded narrow (bf16) and converted bf16->tf32 in-register in the
  //      mainloop, so NO host fp32 widen — that widen was a full extra pass over
  //      A plus a 2x-wider re-read, the dominant memory traffic at large M. A
  //      that arrives as fp32/fp16 is cast down to bf16 (the production input is
  //      already bf16, so this is normally a no-op contiguous view).
  //   B: tfloat32_t is bit-compatible with float (alignas(4){uint32_t}), so an
  //      fp32 buffer reinterprets to tf32 with no copy. This is the genuine
  //      fp32->tf32 precision step; keep B fp32 in, reinterpret at load.
  at::Tensor A_buf = A.scalar_type() == at::kBFloat16 ? A.contiguous() : A.to(at::kBFloat16);
  at::Tensor B_buf = B.scalar_type() == at::kFloat ? B.contiguous() : B.to(at::kFloat);

  // Setup kernel arguments
  typename Kernel::Arguments args;
  args.M = M;
  args.K = K;
  args.N = N;
  args.split_k = split_k;

  // Fetch raw pointers via the untyped data_ptr() (returns void*) and reinterpret.
  // Using data_ptr<tfloat32_t>() would require a PyTorch data_ptr specialization
  // for tfloat32_t (there is none), leaving an undefined symbol at link time.
  // The untyped form sidesteps that.

  // Input A: [M, K] - row-major (K contiguous). Loaded as bf16 (ElementALoad);
  // the block-2D copy reads bf16 and reorder() converts to tf32 for the MMA.
  args.ptr_A = reinterpret_cast<GemmSqrSumXe::ElementALoad const*>(A_buf.data_ptr());
  args.stride_A_m = K;  // M-row stride
  args.stride_A_k = 1;  // K stride (contiguous)

  // Input B: [N, K] - row-major (K contiguous). stride_B_k carries the N-row
  // stride (== K); layout_B is make_stride(stride_B_k, 1) over shape (N,K).
  args.ptr_B = reinterpret_cast<GemmSqrSumXe::Element const*>(B_buf.data_ptr());
  args.stride_B_k = K;  // N-row stride
  args.stride_B_n = 1;  // K stride (contiguous)

  // Output C: [split_k, M, N] fp32 row-major == the caller's gemm_out_mul
  // partials. ElementC == float (the accumulator type). The kernel writes slab
  // `split_idx` directly into this buffer (base offset split_idx*M*N); no host
  // scratch, no host reduction — the fuse reduces the leading axis downstream.
  // stride_C_m/n describe one [M,N] slab.
  args.ptr_C = reinterpret_cast<typename Kernel::ElementC*>(C.data_ptr());
  args.stride_C_m = N;  // Row stride within a slab
  args.stride_C_n = 1;  // Column stride

  // Output sqrsum partials: [split_k, M]. The square-sum-as-GEMM produces an
  // [M,N] slab per split where every column of a row equals that row's partial
  // sum_k A[row,k]^2, so the kernel needs an [M,N]-shaped scratch slab to store
  // through the block-2D copy engine; we then take column 0 into sqrsum[s, :].
  // (A standalone [split_k, M, N] scratch — only the square-sum needs the extra
  // N columns; the final sqrsum the fuse reads is [split_k, M].)
  auto sqsc = torch::empty({split_k, M, N}, sqrsum.options().dtype(torch::kFloat32));
  args.ptr_sqrsum = sqsc.data_ptr<float>();  // base of the [split_k,M,N] scratch
  args.ptr_sqrsum_scratch = sqsc.data_ptr<float>();
  args.stride_sqsc_m = N;  // row-major within a slab
  args.stride_sqsc_n = 1;

  // Mainloop arguments (empty for now)
  args.mainloop = typename GemmSqrSumXe::CollectiveMainloop::Arguments{};

  // Launch kernel
  Runner runner;
  auto status = runner.run(args, nullptr, c10::xpu::getCurrentXPUStream().queue());

  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS GEMM+SqrSum kernel failed with status: ", int(status));

  // Design B: NO split-K reduction here. C already holds the [split_k, M, N]
  // partials in place (the kernel wrote each slab directly). The only host work
  // is extracting the square-sum partials from the [split_k, M, N] scratch into
  // the caller's [split_k, M] buffer: every column of a slab's row is equal, so
  // column 0 is that split's partial sum_k A[m,k]^2. No sum over split_k — the
  // fuse's `for split` loop reduces both C and sqrsum downstream.
  //   sqrsum[s, m] = sqsc[s, m, 0]
  sqrsum.copy_(sqsc.select(/*dim=*/2, /*index=*/0));
}
