/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

// FP8 E4M3 weight, BF16 activation grouped GEMM for Xe2 (BMG). Activation is
// applied externally between GEMM1 and GEMM2.

#pragma once

#include "../common/block_2d_copy_d.hpp"
#include "../w4a16/gemm_xe2.hpp"
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

namespace MoE_FP8 {
using namespace cute;

template <
    typename TileShape,
    typename SubgroupLayout,
    typename TensorA,
    typename TensorBPacked,
    typename TensorD,
    typename TensorBias,
    typename TiledMMA,
    bool WithBias,
    bool WeightScalePerExpert = false,
    bool WeightScaleBlocked = false,
    bool SingleWeightScale = false>
class MoEGEMMFp8Weight {
 public:
  using ElementA = cutlass::bfloat16_t;
  using ElementD = cutlass::bfloat16_t;
  using TiledCopyA = decltype(make_block_2d_copy_A(TiledMMA{}, TensorA{}));
  using TiledCopyBPacked = decltype(make_block_2d_copy_B(TiledMMA{}, TensorBPacked{}));
  using TiledCopyD = decltype(moe_xe20::make_moe_block_2d_copy_D<void>(TiledMMA{}, TensorD{}));
  using SGPerWG = decltype(product(take<1, 4>(shape(typename TiledMMA::ThrLayoutVMNK{}))));

  constexpr static int Stages = 3;
  using MainloopDispatchPolicy = MoE_FP8::XeDefault<Stages>;
  using CollectiveMainloop = MoEMainloopFp8Weight<
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
      WeightScalePerExpert,
      WeightScaleBlocked>;

  struct Params {
    const uint8_t* Activations;    // [M_total, K] bf16 bytes
    const uint8_t* PackedWeights;  // [num_experts, N, K] fp8 e4m3 raw bytes
    const float* WeightScales;     // Per-expert scalar or [E, ceil(N/128), K/128] block scales
    const float* Bias;
    ElementD* Outputs;
    const int32_t* M_per_group;
    const int32_t N;
    const int32_t K;
    const int32_t num_experts;
    int32_t* workspace;
    TiledMMA mma;
    int32_t ld_b;
    bool weight_scale_blocked = false;
    bool static_scheduler = false;
  };

  auto make_A_tensor(uint8_t* ptr_A, int M, int K) {
    auto* bf16_ptr = reinterpret_cast<cutlass::bfloat16_t*>(ptr_A);
    return make_tensor(make_gmem_ptr(bf16_ptr), make_layout(make_shape(M, K), make_stride(K, _1{})));
  }

  auto make_B_tensors(uint8_t* ptr_B, int N, int K, int ld_b) {
    auto* e4m3_ptr = reinterpret_cast<cutlass::float_e4m3_t*>(ptr_B);
    auto B = make_tensor(make_gmem_ptr(e4m3_ptr), make_layout(make_shape(N, K), make_stride(ld_b, _1{})));
    return B;
  }

  // Per-expert weight-scale pointers + row stride. Mirrors MXFP4's
  // make_scale_ptrs exactly (same gate/up split conventions), just with
  // K_scale = K / FP8_GROUP_SIZE_K instead of K / MXFP4_GROUP_SIZE.
  auto make_Bias_tensors(float* ptr_Bias, int N) {
    return make_tensor(make_gmem_ptr<float>(ptr_Bias), make_layout(make_shape(N), make_stride(_1{})));
  }

  auto make_D_tensors(ElementD* ptr_D, int pre_rows, int M, int N) {
    return make_tensor(
        make_gmem_ptr<ElementD>(ptr_D + pre_rows * N), make_layout(make_shape(M, N), make_stride(N, _1{})));
  }

  CUTLASS_DEVICE void operator()(Params const& params, sycl::nd_item<3> item, int32_t* slm_mem) {
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
    int N_pad = ceil_div(N, wg_tile_n) * wg_tile_n;
    int group_m_id = (group_id * wg_tile_n) / N_pad;
    int group_range = item.get_group_range(1);
    int32_t thr_id = int32_t(item.get_local_linear_id());

    const int64_t K_scale = K / FP8_GROUP_SIZE_K;
    int64_t scale_n =
        WeightScalePerExpert ? (SingleWeightScale ? 1 : 2) : (params.weight_scale_blocked ? (N + 127) / 128 : N);

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
      int64_t S_offset = static_cast<int64_t>(expert_id) * scale_n * (WeightScalePerExpert ? 1 : K_scale);

      uint8_t* ptr_A_curr_batch = const_cast<uint8_t*>(params.Activations) + pre_rows * K * sizeof(ElementA);
      uint8_t* ptr_B_curr_batch = const_cast<uint8_t*>(params.PackedWeights) + B_offset;
      float* ptr_S_curr_batch = const_cast<float*>(params.WeightScales) + S_offset;
      float* ptr_Bias_curr_batch = nullptr;
      if constexpr (WithBias) {
        ptr_Bias_curr_batch = const_cast<float*>(params.Bias) + expert_id * N;
      }

      auto A_tensor = make_A_tensor(ptr_A_curr_batch, M, K);
      auto B_tensor = make_B_tensors(ptr_B_curr_batch, N, K, ld_b);
      auto D_tensor = make_D_tensors(params.Outputs, pre_rows, M, N);
      auto Bias_tensor = make_Bias_tensors(ptr_Bias_curr_batch, N);

      while (group_m_id < cumsum_tiles_for_experts) {
        int n_coord = (group_id * wg_tile_n) % N_pad / wg_tile_n;
        int m_coord = (group_m_id - pre_tiles);

        auto tile_coord = make_coord(m_coord, n_coord, _, 0);
        if constexpr (SingleWeightScale) {
          moe_w4a16::xe_gemm<void, void, void>(
              A_tensor, B_tensor, ptr_S_curr_batch, ptr_Bias_curr_batch, D_tensor, tile_coord, mma);
        } else {
          CollectiveMainloop mainloop;
          mainloop(
              A_tensor,
              B_tensor,
              ptr_S_curr_batch,
              WeightScalePerExpert ? scale_n : K_scale,
              D_tensor,
              tile_coord,
              mma,
              thr_id,
              Bias_tensor,
              N);
        }

        if (params.static_scheduler) {
          group_id += group_range;
        } else {
          if (thr_id == 0) {
            slm_mem[0] = cutlass::atomicAdd(workspace, 1);
          }
          item.barrier(sycl::access::fence_space::local_space);
          group_id = group_range + slm_mem[0];
        }
        group_m_id = (group_id * wg_tile_n) / N_pad;
      }
      pre_rows = cumsum_rows_for_experts;
      pre_tiles = cumsum_tiles_for_experts;
    }
  };
};
}  // namespace MoE_FP8
