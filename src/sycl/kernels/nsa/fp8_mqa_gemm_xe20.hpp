/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Custom FP8 GEMM for MQA logits on Intel BMG (Xe20).
//
// Computes: D(M,N) = A_fp8(M,K) @ B_fp8(N,K)^T   (fp32 output)
//
// A is (M,K) uint8 fp8 with K contiguous (stride = (K, 1)).
// B is (N,K) uint8 fp8 with K contiguous (stride = (K, 1)).
// D is (M,N) float32 with N contiguous (stride = (N, 1)).
//
// NO host-side transpose of B is needed. The custom mainloop handles
// the (N,K) layout directly via auto-selected 2D block copy atoms
// (LD_T for transpose load when needed).
//
// Pipeline: load uint8 → convert FP8→FP16 → reorder → XMX GEMM (FP16 in, FP32 accum)

#pragma once

#include <cute/tensor.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include "cutlass/cutlass.h"
// clang-format off
#include "cutlass/numeric_conversion.h"
// clang-format on
#include "cutlass/fp8_to_fp16.h"
#include "cutlass/half.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/util/sycl_event_manager.hpp"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace nsa {

using namespace cute;

// FP8 MQA GEMM mainloop with custom FP8→FP16 conversion.
//
// Follows the MoE mainloop pattern:
//   1. Load A/B as uint8 using auto-selected 2D block copy atoms
//   2. Convert FP8 e4m3 → FP16 in registers (element-wise)
//   3. Reorder from copy layout to MMA layout
//   4. XMX GEMM with FP16 inputs, FP32 accumulator
template <
    int Stages,
    typename TiledCopyA_,
    typename TiledCopyB_,
    typename TiledCopyD_,
    typename ATensor_,
    typename BTensor_,
    typename DTensor_,
    typename TiledMMA_>
struct Fp8MqaGemmMainloop {
  using TiledCopyA = TiledCopyA_;
  using TiledCopyB = TiledCopyB_;
  using TiledCopyD = TiledCopyD_;
  using ATensor = ATensor_;
  using BTensor = BTensor_;
  using DTensor = DTensor_;
  using TiledMMA = TiledMMA_;

  using MmaType = cutlass::half_t;

  template <typename Coord>
  CUTLASS_DEVICE void operator()(
      ATensor& A,  // (M,K) uint8 fp8
      BTensor& B,  // (N,K) uint8 fp8
      DTensor& D,  // (M,N) float32
      Coord blk_coord,
      TiledMMA mma,
      int thr_id) {
    auto wg_m = get<0>(blk_coord);
    auto wg_n = get<1>(blk_coord);

    Tensor cA = make_identity_tensor(A.shape());
    Tensor cB = make_identity_tensor(B.shape());
    Tensor cD = make_identity_tensor(D.shape());

    auto wg_tile = mma.tile_mnk();
    auto wg_coord = make_coord(wg_m, wg_n, 0);

    Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(wg_m, _));  // (BLK_M, BLK_K, k)
    Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(wg_n, _));  // (BLK_N, BLK_K, k)
    Tensor gD = local_tile(cD, wg_tile, wg_coord, Step<_1, _1, X>{});        // (BLK_M, BLK_N)

    TiledCopyA tiled_copy_a{A};
    TiledCopyB tiled_copy_b{B};
    TiledCopyD tiled_copy_d{D};

    auto thr_copy_a = tiled_copy_a.get_slice(thr_id);
    auto thr_copy_b = tiled_copy_b.get_slice(thr_id);
    auto thr_copy_d = tiled_copy_d.get_slice(thr_id);
    auto thr_mma = mma.get_slice(thr_id);

    // Source partitions for global → register copy
    auto tAgA = thr_copy_a.partition_S(gA);
    auto tBgB = thr_copy_b.partition_S(gB);

    // Copy-layout register fragments (uint8, for raw FP8 data)
    auto tArA_u8 = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
    auto tBrB_u8 = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

    // FP16 intermediates in copy layout (for FP8→FP16 conversion before reorder)
    MmaType tArA_fp16_buf[tArA_u8.size()];
    auto tArA_fp16_tensor = make_tensor(make_rmem_ptr(tArA_fp16_buf), tArA_u8.layout());
    auto tArA_fp16 = make_subgroup_tensor(tArA_fp16_tensor, tArA_u8.tv_layout());

    MmaType tBrB_fp16_buf[tBrB_u8.size()];
    auto tBrB_fp16_tensor = make_tensor(make_rmem_ptr(tBrB_fp16_buf), tBrB_u8.layout());
    auto tBrB_fp16 = make_subgroup_tensor(tBrB_fp16_tensor, tBrB_u8.tv_layout());

    // MMA-layout register fragments (FP16)
    auto tSrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
    auto tSrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

    // Accumulator (FP32)
    SubgroupTensor tCrC = thr_mma.partition_sg_fragment_C(gD);

    // Output fragment for D store
    using TD = typename DTensor::element_type;
    TD tCrD_frag[tCrC.size()];
    Tensor tCrD_tensor = make_tensor(make_rmem_ptr(tCrD_frag), tCrC.layout());
    SubgroupTensor tCrD_sg = make_subgroup_tensor(tCrD_tensor, tCrC.tv_layout());
    Tensor tCgD = thr_mma.partition_C(gD);

    // Prefetch setup
    auto prefetch_a = make_block_2d_prefetch(tiled_copy_a);
    auto prefetch_b = make_block_2d_prefetch(tiled_copy_b);
    auto pAgA = prefetch_a.get_slice(thr_id).partition_S(gA);
    auto pBgB = prefetch_b.get_slice(thr_id).partition_S(gB);

    constexpr int barrier_scope = 2;
    int prefetch_k = 0;
    int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));

    // Prefetch first tiles
    CUTE_UNROLL
    for (; prefetch_k < Stages && prefetch_k < k_tile_count; ++prefetch_k) {
      prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
      prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
    }

    // Main GEMM loop
    for (int k_tile = 0; k_tile < k_tile_count; ++k_tile, ++prefetch_k) {
      barrier_arrive(barrier_scope);

      // Load FP8 data as uint8
      copy(tiled_copy_a, tAgA(_, _, _, k_tile), tArA_u8);
      copy(tiled_copy_b, tBgB(_, _, _, k_tile), tBrB_u8);

      if (prefetch_k < k_tile_count) {
        prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
        prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
      }

      // FP8 e4m3 → FP16 conversion (element-wise on base tensors)
      convert_FP8_to_FP16<cute::float_e4m3_t>(tArA_u8.tensor(), tArA_fp16_tensor);
      convert_FP8_to_FP16<cute::float_e4m3_t>(tBrB_u8.tensor(), tBrB_fp16_tensor);

      // Reorder from copy layout to MMA layout
      reorder(tArA_fp16, tSrA);
      reorder(tBrB_fp16, tSrB);

      // XMX GEMM: FP16 × FP16 → FP32
      cute::gemm(mma, tSrA, tSrB, tCrC);
      barrier_wait(barrier_scope);
    }

    // Store results
    reorder(tCrC, tCrD_sg);
    copy(tiled_copy_d, tCrD_sg, tCgD);
  }
};

// Tile configuration
using GemmTileShape = Shape<_32, _128, _32>;
using GemmSGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
using GemmMmaAtom = MMA_Atom<XE_8x16x16_F32F16F16F32_TT>;
using GemmTiledMMA = typename TiledMMAHelper<GemmMmaAtom, Layout<GemmTileShape>, GemmSGLayout>::TiledMMA;

constexpr int GemmPipelineStages = 2;

// Stride types: both A(M,K) and B(N,K) have K contiguous, D(M,N) has N contiguous
using GemmStrideAB = Stride<int, _1>;
using GemmStrideD = Stride<int, _1>;

// Kernel parameters — POD struct captured by the SYCL lambda.
// Tensors are constructed inside the kernel from these raw pointers/dims.
// Batch strides (in elements) advance the per-batch A/B/D base pointers;
// they are 0 for the single-GEMM case.
struct Fp8MqaGemmParams {
  uint8_t* A_ptr;
  uint8_t* B_ptr;
  float* D_ptr;
  int M, N, K;
  int grid_n;
  int64_t A_batch_stride;
  int64_t B_batch_stride;
  int64_t D_batch_stride;
};

// Kernel name
class Fp8MqaGemmKernelName;

// Launch the custom FP8 MQA GEMM, optionally batched.
// Computes, for each batch b in [0, batch):
//   D_b(M,N) = A_b(M,K) @ B_b(N,K)^T
// where A_b = A_fp8 + b*A_batch_stride, similarly for B and D.
// All batches share the same (M,N,K) tile shape, so a single grid launch
// covers them by adding a batch dimension (grid dim 0). This replaces the
// per-batch serial launch loop with one fused launch.
// A: (M,K) uint8 fp8, K contiguous
// B: (N,K) uint8 fp8, K contiguous
// D: (M,N) float32, N contiguous
inline int fp8_mqa_gemm_batched_launch(
    sycl::queue* queue_ptr,
    const void* A_fp8,
    const void* B_fp8,
    void* D_f32,
    int batch,
    int M,
    int N,
    int K,
    int64_t A_batch_stride,
    int64_t B_batch_stride,
    int64_t D_batch_stride) {
  // Create dummy tensors for type deduction (only stride types matter)
  auto make_dummy = [](auto* ptr, auto stride) {
    return make_tensor(make_gmem_ptr(ptr), make_layout(repeat<rank_v<decltype(stride)>>(1), stride));
  };

  auto dummy_a = make_dummy(static_cast<uint8_t*>(nullptr), GemmStrideAB{});
  auto dummy_b = make_dummy(static_cast<uint8_t*>(nullptr), GemmStrideAB{});
  auto dummy_d = make_dummy(static_cast<float*>(nullptr), GemmStrideD{});

  using TensorA = decltype(dummy_a);
  using TensorB = decltype(dummy_b);
  using TensorD = decltype(dummy_d);

  using TiledCopyA = decltype(make_block_2d_copy_A(GemmTiledMMA{}, dummy_a));
  using TiledCopyB = decltype(make_block_2d_copy_B(GemmTiledMMA{}, dummy_b));
  using TiledCopyD = decltype(make_block_2d_copy_D(GemmTiledMMA{}, dummy_d));

  using Mainloop = Fp8MqaGemmMainloop<
      GemmPipelineStages,
      TiledCopyA,
      TiledCopyB,
      TiledCopyD,
      TensorA,
      TensorB,
      TensorD,
      GemmTiledMMA>;

  GemmTiledMMA mma;
  auto wg_tile = mma.tile_mnk();
  int wg_tile_m = get<0>(wg_tile);
  int wg_tile_n = get<1>(wg_tile);

  // Check alignment: M and N must be multiples of tile dimensions
  if (M % wg_tile_m != 0 || N % wg_tile_n != 0 || K % get<2>(wg_tile) != 0) {
    return 1;  // Caller should use fallback
  }

  int grid_m = M / wg_tile_m;
  int grid_n = N / wg_tile_n;
  int total_tiles = grid_m * grid_n;

  auto max_threads = size(mma);

  sycl::range<3> global_range(batch, total_tiles, max_threads);
  sycl::range<3> local_range(1, 1, max_threads);

  Fp8MqaGemmParams params{
      const_cast<uint8_t*>(static_cast<const uint8_t*>(A_fp8)),
      const_cast<uint8_t*>(static_cast<const uint8_t*>(B_fp8)),
      static_cast<float*>(D_f32),
      M,
      N,
      K,
      grid_n,
      A_batch_stride,
      B_batch_stride,
      D_batch_stride,
  };

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  syclex::properties kernel_props{syclex::sub_group_size<16>, intelex::grf_size<256>};

  queue_ptr->submit([&](sycl::handler& h) {
    h.parallel_for<Fp8MqaGemmKernelName>(
        sycl::nd_range<3>(global_range, local_range), kernel_props, [=](sycl::nd_item<3> item) {
          int batch_id = item.get_group(0);
          int group_id = item.get_group(1);
          int wg_m_idx = group_id / params.grid_n;
          int wg_n_idx = group_id % params.grid_n;
          int thr_id = item.get_local_linear_id();

          auto coord = make_coord(wg_m_idx, wg_n_idx, 0);

          // Per-batch base pointers
          uint8_t* A_base = params.A_ptr + batch_id * params.A_batch_stride;
          uint8_t* B_base = params.B_ptr + batch_id * params.B_batch_stride;
          float* D_base = params.D_ptr + batch_id * params.D_batch_stride;

          // Construct tensors from raw pointers inside the kernel
          auto A = make_tensor(
              make_gmem_ptr(A_base), make_layout(make_shape(params.M, params.K), GemmStrideAB{params.K, _1{}}));
          auto B = make_tensor(
              make_gmem_ptr(B_base), make_layout(make_shape(params.N, params.K), GemmStrideAB{params.K, _1{}}));
          auto D = make_tensor(
              make_gmem_ptr(D_base), make_layout(make_shape(params.M, params.N), GemmStrideD{params.N, _1{}}));

          GemmTiledMMA mma_local;
          Mainloop{}(A, B, D, coord, mma_local, thr_id);
        });
  });

  return 0;
}

// Single-GEMM convenience wrapper (batch=1, zero batch strides).
inline int
fp8_mqa_gemm_launch(sycl::queue* queue_ptr, const void* A_fp8, const void* B_fp8, void* D_f32, int M, int N, int K) {
  return fp8_mqa_gemm_batched_launch(queue_ptr, A_fp8, B_fp8, D_f32, 1, M, N, K, 0, 0, 0);
}

}  // namespace nsa
