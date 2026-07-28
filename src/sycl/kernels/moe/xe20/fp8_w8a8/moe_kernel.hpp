/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

// FP8 (E4M3) W8A8 MoE grouped-GEMM kernel for Xe2 (BMG).
//
// Fork of src/sycl/kernels/moe/xe20/mxfp4_w4a16/moe_kernel.hpp. Identical
// per-expert tile-scheduler loop; the differences vs. the MXFP4 kernel:
//   - Both A and B are fp8 e4m3 (1 byte/element, no sub-byte packing), so B
//     tensor construction uses plain (not halved) K strides, same shape
//     math as the bf16 kernel's make_B_tensors.
//   - A per-expert float32 weight-scale tensor is threaded through
//     alongside B (per N-row, per FP8_GROUP_SIZE_K-wide K-group direct
//     multiplier - see moe_mainloop.hpp for why this granularity, and the
//     Python-side expectation that a genuinely 2-D-blocked scale tensor is
//     pre-expanded to per-N-row before reaching this op).
//   - A per-token (per-M-row) float32 activation-scale array is threaded
//     through as a flat pointer indexed by the same shuffled-token order
//     as Activations (no per-expert offset math beyond `pre_rows`, exactly
//     like Activations itself).
//
// NOTE: unlike the bf16/MXFP4 kernels, there is currently only ONE
// operator() dispatch shape used from Python for GEMM1 (always fuse_act
// for gated activations) - the "unfused GEMM1 for huge-weight/small-M"
// heuristic that GroupGemmXe20.cpp/moe.py apply for bf16 was not ported
// here yet. The non-fused operator() overload below is still fully
// implemented (it's what down-projection/GEMM2 always uses), so wiring up
// an unfused GEMM1 path later is just a dispatch-side change, not a new
// mainloop. Flagged here since this is a real (if second-order) perf gap
// vs. the bf16 path for the huge-weight/small-M corner case.

#pragma once

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/float8.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/platform/platform.h"
#include "cutlass/util/packed_stride.hpp"
#include "moe_mainloop.hpp"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace MoE_FP8_W8A8 {
using namespace cute;

template <
    typename TileShape,
    typename SubgroupLayout,
    typename TensorA,
    typename TensorBPacked,
    typename TensorD,
    typename TensorBias,
    typename TiledMMA,
    int ActType,
    bool FuseAct,
    bool WithBias,
    typename ElementA,
    typename ElementD = ElementA>
class MoEGEMMFp8W8A8 {
 public:
  using TiledCopyA = decltype(make_block_2d_copy_A(TiledMMA{}, TensorA{}));
  using TiledCopyBPacked = decltype(make_block_2d_copy_B(TiledMMA{}, TensorBPacked{}));
  using TiledCopyD = decltype(make_block_2d_copy_D(TiledMMA{}, TensorD{}));
  using SGPerWG = decltype(product(take<1, 4>(shape(typename TiledMMA::ThrLayoutVMNK{}))));

  constexpr static int Stages = 3;
  using MainloopDispatchPolicy = MoE_FP8_W8A8::XeDefault<Stages>;
  using CollectiveMainloop = MoEMainloopFp8W8A8<
      MainloopDispatchPolicy,
      TiledCopyA,
      TiledCopyBPacked,
      TiledCopyD,
      TensorA,
      TensorBPacked,
      TensorD,
      TensorBias,
      TiledMMA,
      WithBias,
      ActType>;

  struct Params {
    const uint8_t* Activations;    // [M_total, K] fp8 e4m3 raw bytes
    const uint8_t* PackedWeights;  // [num_experts, N, K] fp8 e4m3 raw bytes
    const float* WeightScales;     // [num_experts, N, K/FP8_GROUP_SIZE_K] fp32 direct multiplier
    const float* ActScales;        // [M_total] fp32 per-token direct multiplier
    const float* Bias;
    ElementD* Outputs;
    const int32_t* M_per_group;
    const int32_t N;
    const int32_t K;
    const int32_t num_experts;
    int32_t* workspace;
    TiledMMA mma;
    int32_t ld_b;
    float gemm1_alpha = 1.702f;
    float gemm1_limit = 7.0f;
  };

  auto make_A_tensor(uint8_t* ptr_A, int M, int K) {
    auto* e4m3_ptr = reinterpret_cast<cutlass::float_e4m3_t*>(ptr_A);
    return make_tensor(make_gmem_ptr(e4m3_ptr), make_layout(make_shape(M, K), make_stride(K, _1{})));
  }

  // Same shape/stride math as bf16/moe_kernel.hpp's make_B_tensors - fp8
  // e4m3 is a full byte per element (unlike MXFP4's 2-nibbles-per-byte),
  // so no K/2 packing arithmetic is needed here.
  auto make_B_tensors(uint8_t* ptr_B, int N, int K, int ld_b) {
    auto* e4m3_ptr = reinterpret_cast<cutlass::float_e4m3_t*>(ptr_B);
    if constexpr (FuseAct) {
      if constexpr (ActType == SWIGLU_GPT_OSS) {
        auto B0 = make_tensor(make_gmem_ptr(e4m3_ptr), make_layout(make_shape(N / 2, K), make_stride(2 * ld_b, _1{})));
        auto B1 =
            make_tensor(make_gmem_ptr(e4m3_ptr + ld_b), make_layout(make_shape(N / 2, K), make_stride(2 * ld_b, _1{})));
        return cute::make_tuple(B0, B1);
      } else {
        auto B0 = make_tensor(make_gmem_ptr(e4m3_ptr), make_layout(make_shape(N / 2, K), make_stride(ld_b, _1{})));
        auto B1 = make_tensor(
            make_gmem_ptr(e4m3_ptr + static_cast<int64_t>(N / 2) * ld_b),
            make_layout(make_shape(N / 2, K), make_stride(ld_b, _1{})));
        return cute::make_tuple(B0, B1);
      }
    } else {
      auto B = make_tensor(make_gmem_ptr(e4m3_ptr), make_layout(make_shape(N, K), make_stride(ld_b, _1{})));
      return cute::make_tuple(B);
    }
  }

  // Per-expert weight-scale pointers + row stride. Mirrors MXFP4's
  // make_scale_ptrs exactly (same gate/up split conventions), just with
  // K_scale = K / FP8_GROUP_SIZE_K instead of K / MXFP4_GROUP_SIZE.
  struct WeightScalePtrs {
    const float* ptr0;
    const float* ptr1;
    int row_stride;
  };

  WeightScalePtrs make_weight_scale_ptrs(const float* ptr_S, int N, int K) {
    const int K_scale = K / FP8_GROUP_SIZE_K;
    if constexpr (FuseAct) {
      if constexpr (ActType == SWIGLU_GPT_OSS) {
        return WeightScalePtrs{ptr_S, ptr_S + K_scale, 2 * K_scale};
      } else {
        return WeightScalePtrs{ptr_S, ptr_S + (N / 2) * K_scale, K_scale};
      }
    } else {
      return WeightScalePtrs{ptr_S, nullptr, K_scale};
    }
  }

  auto make_Bias_tensors(float* ptr_Bias, int N) {
    if constexpr (WithBias) {
      if constexpr (FuseAct) {
        if constexpr (ActType == SWIGLU_GPT_OSS) {
          auto Bias0 = make_tensor(make_gmem_ptr<float>(ptr_Bias), make_layout(make_shape(N / 2), make_stride(_2{})));
          auto Bias1 =
              make_tensor(make_gmem_ptr<float>(ptr_Bias + 1), make_layout(make_shape(N / 2), make_stride(_2{})));
          return cute::make_tuple(Bias0, Bias1);
        } else {
          auto Bias0 = make_tensor(make_gmem_ptr<float>(ptr_Bias), make_layout(make_shape(N / 2), make_stride(_1{})));
          float* ptr_Bias1 = ptr_Bias + (N / 2);
          auto Bias1 = make_tensor(make_gmem_ptr<float>(ptr_Bias1), make_layout(make_shape(N / 2), make_stride(_1{})));
          return cute::make_tuple(Bias0, Bias1);
        }
      } else {
        auto Bias = make_tensor(make_gmem_ptr<float>(ptr_Bias), make_layout(make_shape(N), make_stride(_1{})));
        return cute::make_tuple(Bias);
      }
    } else {
      if constexpr (FuseAct && ActType == SWIGLU_GPT_OSS) {
        return cute::make_tuple(
            make_tensor(make_gmem_ptr<float>(nullptr), make_layout(make_shape(0), make_stride(_2{}))),
            make_tensor(make_gmem_ptr<float>(nullptr), make_layout(make_shape(0), make_stride(_2{}))));
      } else {
        return cute::make_tuple(
            make_tensor(make_gmem_ptr<float>(nullptr), make_layout(make_shape(0), make_stride(_1{}))),
            make_tensor(make_gmem_ptr<float>(nullptr), make_layout(make_shape(0), make_stride(_1{}))));
      }
    }
  }

  auto make_D_tensors(ElementD* ptr_D, int pre_rows, int M, int N) {
    if constexpr (FuseAct) {
      auto D_tensor = make_tensor(
          make_gmem_ptr<ElementD>(ptr_D + pre_rows * N / 2),
          make_layout(make_shape(M, N / 2), make_stride(N / 2, _1{})));
      return D_tensor;
    } else {
      auto D_tensor = make_tensor(
          make_gmem_ptr<ElementD>(ptr_D + pre_rows * N), make_layout(make_shape(M, N), make_stride(N, _1{})));
      return D_tensor;
    }
  }

  void operator()(Params const& params, sycl::nd_item<3> item, int32_t* slm_mem) {
    auto N = params.N;
    auto K = params.K;
    auto M_per_group = params.M_per_group;
    auto num_experts = params.num_experts;
    auto mma = params.mma;
    auto workspace = params.workspace;

    auto wg_tile = mma.tile_mnk();
    auto wg_tile_m = get<0>(wg_tile);
    auto wg_tile_n = get<1>(wg_tile);

    int group_id = item.get_group_linear_id();
    int N_pad;
    if constexpr (FuseAct) {
      N_pad = ceil_div(N / 2, wg_tile_n) * wg_tile_n;
    } else {
      N_pad = ceil_div(N, wg_tile_n) * wg_tile_n;
    }
    int group_m_id = (group_id * wg_tile_n) / N_pad;
    int group_range = item.get_group_range(1);
    int32_t thr_id = int32_t(item.get_local_linear_id());

    if (group_id == 0 && thr_id == 0) {
      auto atm = sycl::atomic_ref<
          int,
          sycl::memory_order::relaxed,
          sycl::memory_scope::device,
          sycl::access::address_space::global_space>(workspace[0]);
      atm.store(0);
    }

    const int64_t K_scale = K / FP8_GROUP_SIZE_K;

    int pre_rows = 0;
    int pre_tiles = 0;
    for (int i = 0; i < num_experts; ++i) {
      int M = M_per_group[i];
      int cumsum_rows_for_experts = M + pre_rows;
      int cumsum_tiles_for_experts = (M + wg_tile_m - 1) / wg_tile_m + pre_tiles;

      if (group_m_id >= cumsum_tiles_for_experts) {
        pre_rows = cumsum_rows_for_experts;
        pre_tiles = cumsum_tiles_for_experts;
        continue;
      }

      int expert_id = i;
      int ld_b = params.ld_b;
      int64_t B_offset = static_cast<int64_t>(expert_id) * static_cast<int64_t>(N) * static_cast<int64_t>(ld_b);
      int64_t S_offset = static_cast<int64_t>(expert_id) * static_cast<int64_t>(N) * K_scale;

      uint8_t* ptr_A_curr_batch = const_cast<uint8_t*>(params.Activations) + pre_rows * K;
      uint8_t* ptr_B_curr_batch = const_cast<uint8_t*>(params.PackedWeights) + B_offset;
      float* ptr_S_curr_batch = const_cast<float*>(params.WeightScales) + S_offset;
      float* ptr_ActScale_curr_batch = const_cast<float*>(params.ActScales) + pre_rows;
      float* ptr_Bias_curr_batch = nullptr;
      if constexpr (WithBias) {
        ptr_Bias_curr_batch = const_cast<float*>(params.Bias) + expert_id * N;
      }

      auto A_tensor = make_A_tensor(ptr_A_curr_batch, M, K);
      auto B_tensor = make_B_tensors(ptr_B_curr_batch, N, K, ld_b);
      auto weight_scale_ptrs = make_weight_scale_ptrs(ptr_S_curr_batch, N, K);
      auto D_tensor = make_D_tensors(params.Outputs, pre_rows, M, N);
      auto Bias_tensor = make_Bias_tensors(ptr_Bias_curr_batch, N);

      while (group_m_id < cumsum_tiles_for_experts) {
        int n_coord = (group_id * wg_tile_n) % N_pad / wg_tile_n;
        int m_coord = (group_m_id - pre_tiles);

        CollectiveMainloop mainloop;
        if constexpr (FuseAct) {
          auto tile_coord = make_coord(m_coord, n_coord, n_coord);
          mainloop(
              A_tensor,
              get<0>(B_tensor),
              get<1>(B_tensor),
              weight_scale_ptrs.ptr0,
              weight_scale_ptrs.ptr1,
              weight_scale_ptrs.row_stride,
              ptr_ActScale_curr_batch,
              D_tensor,
              tile_coord,
              mma,
              thr_id,
              get<0>(Bias_tensor),
              get<1>(Bias_tensor),
              params.gemm1_alpha,
              params.gemm1_limit);
        } else {
          auto tile_coord = make_coord(m_coord, n_coord, _, 0);
          mainloop(
              A_tensor,
              get<0>(B_tensor),
              weight_scale_ptrs.ptr0,
              weight_scale_ptrs.row_stride,
              ptr_ActScale_curr_batch,
              D_tensor,
              tile_coord,
              mma,
              thr_id,
              get<0>(Bias_tensor));
        }
        if (thr_id == 0) {
          slm_mem[0] = cutlass::atomicAdd(workspace, 1);
        }
        item.barrier(sycl::access::fence_space::local_space);
        group_id = group_range + slm_mem[0];
        group_m_id = (group_id * wg_tile_n) / N_pad;
      }
      pre_rows = cumsum_rows_for_experts;
      pre_tiles = cumsum_tiles_for_experts;
    }
  };
};
}  // namespace MoE_FP8_W8A8
