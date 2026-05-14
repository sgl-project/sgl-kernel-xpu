/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

// MXFP4-B × BF16-A MoE grouped-GEMM kernel for Xe2 (BMG).
//
// Fork of src/sycl/kernels/moe/xe20/moe_kernel.hpp. Identical per-expert
// tile-scheduler loop; the only changes are:
//   - B is int8 packed MXFP4 with row stride K/2 (two E2M1 nibbles per byte).
//   - A per-expert float32 scale tensor S is threaded through alongside B
//     (direct multiplier; producer decodes UE8M0 ahead of the kernel).
//   - B tensor construction uses halved-K strides; S tensor uses stride K/32.
//   - The mainloop call takes both B-tile and S-tile tensors.

#pragma once

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/platform/platform.h"
#include "cutlass/util/packed_stride.hpp"
#include "moe_mxfp4_w4a16_mainloop.hpp"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace MoE_MXFP4_W4A16 {
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
class MoEGEMMMxfp4W4A16 {
 public:
  using TiledCopyA = decltype(make_block_2d_copy_A(TiledMMA{}, TensorA{}));
  using TiledCopyBPacked = decltype(make_block_2d_copy_B(TiledMMA{}, TensorBPacked{}));
  using TiledCopyD = decltype(make_block_2d_copy_D(TiledMMA{}, TensorD{}));
  using SGPerWG = decltype(product(take<1, 4>(shape(typename TiledMMA::ThrLayoutVMNK{}))));

  constexpr static int Stages = 3;
  using MainloopDispatchPolicy = MoE_MXFP4_W4A16::XeDefault<Stages>;
  using CollectiveMainloop = MoEMainloopMxfp4W4A16<
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
    const ElementA* Activations;
    // Packed MXFP4 weights stored as a raw byte buffer of shape
    // [num_experts, N, K/2] uint8. Pointer is kept in byte address space
    // to make per-expert offsetting obvious (1 byte = 2 E2M1 elements);
    // wrapped into a float_e2m1_t CuTe tensor per-expert below.
    const uint8_t* PackedWeights;
    const float* Scales;  // [num_experts, N, K/GROUP_SIZE] fp32 direct multiplier
    const float* Bias;
    ElementD* Outputs;
    const int32_t* M_per_group;
    const int32_t N;
    const int32_t K;
    const int32_t num_experts;
    int32_t* workspace;
    TiledMMA mma;
    float gemm1_alpha = 1.702f;
    float gemm1_limit = 7.0f;
  };

  // Build B-packed tensor(s) from a byte buffer. Inside CuTe, B's element
  // type is cutlass::float_e2m1_t (4 bits). We reinterpret the uint8 byte
  // pointer to float_e2m1_t* and wrap with make_gmem_ptr(raw_ptr) (argument-
  // deduced form) so the resulting type matches the launcher dummy tensor
  // type: gmem_ptr<float_e2m1_t*>, not gmem_ptr<subbyte_iterator<...>>.
  auto make_B_tensors(uint8_t* ptr_B, int N, int K) {
    auto* e2m1_ptr = reinterpret_cast<cutlass::float_e2m1_t*>(ptr_B);
    const int byte_half_K = K / 2;
    if constexpr (FuseAct) {
      if constexpr (ActType == SWIGLU_GPT_OSS) {
        // Interleaved [g0, u0, g1, u1, ...] → gate at byte offset 0,
        // up at byte offset K/2 (one packed row = K/2 bytes).
        auto B0 = make_tensor(make_gmem_ptr(e2m1_ptr), make_layout(make_shape(N / 2, K), make_stride(2 * K, _1{})));
        auto B1 = make_tensor(
            make_gmem_ptr(reinterpret_cast<cutlass::float_e2m1_t*>(ptr_B + byte_half_K)),
            make_layout(make_shape(N / 2, K), make_stride(2 * K, _1{})));
        return cute::make_tuple(B0, B1);
      } else {
        // Block-split: first N/2 rows = gate, last N/2 rows = up.
        // up rows start at byte offset (N/2) * (K/2).
        auto B0 = make_tensor(make_gmem_ptr(e2m1_ptr), make_layout(make_shape(N / 2, K), make_stride(K, _1{})));
        auto B1 = make_tensor(
            make_gmem_ptr(reinterpret_cast<cutlass::float_e2m1_t*>(ptr_B + (N / 2) * byte_half_K)),
            make_layout(make_shape(N / 2, K), make_stride(K, _1{})));
        return cute::make_tuple(B0, B1);
      }
    } else {
      auto B = make_tensor(make_gmem_ptr(e2m1_ptr), make_layout(make_shape(N, K), make_stride(K, _1{})));
      return cute::make_tuple(B);
    }
  }

  // Per-expert scale pointers + row stride. Returns a struct that matches
  // the split-B pattern: gate/up base pointers for the fused-act path, or
  // a single pointer for the non-fused path. Row stride is the N-row fp32
  // element stride (= K/GROUP_SIZE for block-split and non-fused,
  // 2*K/GROUP_SIZE for GPT-OSS interleaved layout).
  struct ScalePtrs {
    const float* ptr0;
    const float* ptr1;
    int row_stride;
  };

  ScalePtrs make_scale_ptrs(const float* ptr_S, int N, int K) {
    const int K_scale = K / MXFP4_GROUP_SIZE;
    if constexpr (FuseAct) {
      if constexpr (ActType == SWIGLU_GPT_OSS) {
        // Interleaved [g0, u0, g1, u1, ...] scales: gate rows at element
        // offset 0, up rows at element offset K_scale; row stride = 2*K_scale.
        return ScalePtrs{ptr_S, ptr_S + K_scale, 2 * K_scale};
      } else {
        // Block-split: up rows start after (N/2) gate rows.
        return ScalePtrs{ptr_S, ptr_S + (N / 2) * K_scale, K_scale};
      }
    } else {
      return ScalePtrs{ptr_S, nullptr, K_scale};
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

    const int64_t K_packed = K / 2;
    const int64_t K_scale = K / MXFP4_GROUP_SIZE;

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
      int64_t B_offset = static_cast<int64_t>(expert_id) * static_cast<int64_t>(N) * K_packed;
      int64_t S_offset = static_cast<int64_t>(expert_id) * static_cast<int64_t>(N) * K_scale;

      ElementA* ptr_A_curr_batch = const_cast<ElementA*>(params.Activations) + pre_rows * K;
      uint8_t* ptr_B_curr_batch = const_cast<uint8_t*>(params.PackedWeights) + B_offset;
      float* ptr_S_curr_batch = const_cast<float*>(params.Scales) + S_offset;
      float* ptr_Bias_curr_batch = nullptr;
      if constexpr (WithBias) {
        ptr_Bias_curr_batch = const_cast<float*>(params.Bias) + expert_id * N;
      }

      auto A_tensor =
          make_tensor(make_gmem_ptr<ElementA>(ptr_A_curr_batch), make_layout(make_shape(M, K), make_stride(K, _1{})));
      auto B_tensor = make_B_tensors(ptr_B_curr_batch, N, K);
      auto scale_ptrs = make_scale_ptrs(ptr_S_curr_batch, N, K);
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
              scale_ptrs.ptr0,
              scale_ptrs.ptr1,
              scale_ptrs.row_stride,
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
              scale_ptrs.ptr0,
              scale_ptrs.row_stride,
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
}  // namespace MoE_MXFP4_W4A16
