/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

// 8-bit weight (FP8 E4M3 or INT8/UINT8), BF16 activation grouped GEMM for Xe2
// (BMG). Weights use column-major inner layout `(N, K)` with `Stride<_1, int>`
// so LSC hardware performs native 64B VNNI loads (`XE_LOAD_2D_VNNI`).

#pragma once

#include "../common/block_2d_copy_d.hpp"
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

namespace moe_w8a16 {
using namespace cute;

template <
    typename TileShape,
    typename SubgroupLayout,
    typename TensorA,
    typename TensorBPacked,
    typename TensorD,
    typename TiledMMA,
    bool IsInt8 = false>
class W8A16Kernel {
 public:
  using ElementA = cutlass::bfloat16_t;
  using ElementBPacked = typename TensorBPacked::element_type;
  using ElementD = cutlass::bfloat16_t;
  using TiledCopyA = decltype(make_block_2d_copy_A(TiledMMA{}, TensorA{}));
  using TiledCopyBPacked = decltype(make_block_2d_copy_B(TiledMMA{}, TensorBPacked{}));
  using TiledCopyD = decltype(moe_xe20::make_moe_block_2d_copy_D<void>(TiledMMA{}, TensorD{}));
  using SGPerWG = decltype(product(take<1, 4>(shape(typename TiledMMA::ThrLayoutVMNK{}))));

  constexpr static int Stages = 3;
  using MainloopDispatchPolicy = moe_w8a16::W8A16MainloopPolicy<Stages>;
  using CollectiveMainloop = W8A16Mainloop<
      MainloopDispatchPolicy,
      TiledCopyA,
      TiledCopyBPacked,
      TiledCopyD,
      TensorA,
      TensorBPacked,
      TensorD,
      TiledMMA,
      IsInt8>;

  struct Params {
    const uint8_t* Activations;    // [M_total, K] bf16 bytes
    const uint8_t* PackedWeights;  // [num_experts, N, K] with stride (expert_stride_b, 1, ld_b)
    const void* WeightScales;      // Scalar, per-channel, 128x128 block, K-group, or MXFP8 scales
    const uint8_t* WeightZeros;    // Optional [num_experts, N, K/group_size] uint8 zero-points
    const float* Bias;
    ElementD* Outputs;
    const int32_t* M_per_group;
    const int32_t N;
    const int32_t K;
    const int32_t num_experts;
    int32_t* workspace;
    TiledMMA mma;
    int32_t ld_b;
    int64_t expert_stride_b;
    int32_t scale_mode;
    int32_t group_size;
    bool scale_is_bf16 = false;
    bool is_signed_int8 = true;
    bool static_scheduler = false;
  };

  auto make_A_tensor(uint8_t* ptr_A, int M, int K) {
    auto* bf16_ptr = reinterpret_cast<cutlass::bfloat16_t*>(ptr_A);
    return make_tensor(make_gmem_ptr(bf16_ptr), make_layout(make_shape(M, K), make_stride(K, _1{})));
  }

  auto make_B_tensors(uint8_t* ptr_B, int N, int K, int ld_b) {
    auto* b_ptr = reinterpret_cast<ElementBPacked*>(ptr_B);
    return make_tensor(make_gmem_ptr(b_ptr), make_layout(make_shape(N, K), make_stride(_1{}, ld_b)));
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

    const int32_t mode = params.scale_mode;
    const bool is_mxfp8 = !IsInt8 && (mode == static_cast<int32_t>(ScaleMode::BlockMXFP8));
    const int scale_elem_bytes = is_mxfp8 ? 1 : (params.scale_is_bf16 ? 2 : 4);
    int scale_row_stride = 1;
    int64_t expert_scale_elems = 1;
    if (mode == static_cast<int32_t>(ScaleMode::ScalarSingle)) {
      scale_row_stride = 1;
      expert_scale_elems = 1;
    } else if (mode == static_cast<int32_t>(ScaleMode::ScalarGateUp)) {
      scale_row_stride = 2;
      expert_scale_elems = 2;
    } else if (mode == static_cast<int32_t>(ScaleMode::PerChannelN)) {
      scale_row_stride = 1;
      expert_scale_elems = N;
    } else if (mode == static_cast<int32_t>(ScaleMode::BlockFP32)) {
      scale_row_stride = K / FP8_GROUP_SIZE_K;
      expert_scale_elems = static_cast<int64_t>((N + 127) / 128) * scale_row_stride;
    } else {
      // BlockMXFP8 (group_size=32) or GroupNK (arbitrary power-of-two group_size)
      scale_row_stride = K / params.group_size;
      expert_scale_elems = static_cast<int64_t>(N) * scale_row_stride;
    }
    const int64_t expert_zero_elems = static_cast<int64_t>(N) * scale_row_stride;

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
      int64_t B_offset = static_cast<int64_t>(expert_id) * params.expert_stride_b;
      int64_t S_byte_offset = static_cast<int64_t>(expert_id) * expert_scale_elems * scale_elem_bytes;

      uint8_t* ptr_A_curr_batch = const_cast<uint8_t*>(params.Activations) + pre_rows * K * sizeof(ElementA);
      uint8_t* ptr_B_curr_batch = const_cast<uint8_t*>(params.PackedWeights) + B_offset;
      const uint8_t* ptr_S_curr_batch = static_cast<const uint8_t*>(params.WeightScales) + S_byte_offset;
      const uint8_t* ptr_Z_curr_batch =
          params.WeightZeros == nullptr ? nullptr
                                        : (params.WeightZeros + static_cast<int64_t>(expert_id) * expert_zero_elems);
      const float* ptr_Bias_curr_batch =
          params.Bias == nullptr ? nullptr : params.Bias + static_cast<int64_t>(expert_id) * N;

      auto A_tensor = make_A_tensor(ptr_A_curr_batch, M, K);
      auto B_tensor = make_B_tensors(ptr_B_curr_batch, N, K, ld_b);
      auto D_tensor = make_D_tensors(params.Outputs, pre_rows, M, N);

      while (group_m_id < cumsum_tiles_for_experts) {
        int n_coord = (group_id * wg_tile_n) % N_pad / wg_tile_n;
        int m_coord = (group_m_id - pre_tiles);

        auto tile_coord = make_coord(m_coord, n_coord, _, 0);
        CollectiveMainloop mainloop;
        mainloop(
            A_tensor,
            B_tensor,
            ptr_S_curr_batch,
            ptr_Z_curr_batch,
            scale_row_stride,
            D_tensor,
            tile_coord,
            mma,
            thr_id,
            ptr_Bias_curr_batch,
            N,
            params.scale_mode,
            params.group_size,
            params.scale_is_bf16,
            params.is_signed_int8);

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

}  // namespace moe_w8a16
