/***************************************************************************************************
 * Copyright 2025 Intel corporation. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/platform/platform.h"
#include "cutlass/util/packed_stride.hpp"
#include "moe_mainloop.hpp"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace MoE {
using namespace cute;

template <
    typename TileShape,
    typename SubgroupLayout,
    typename TensorA,
    typename TensorB,
    typename TensorD,
    typename TensorBias,
    typename TiledMMA,
    bool FuseSiLU,
    bool WithBias,
    typename ElementA,
    typename ElementB = ElementA,
    typename ElementS = ElementA,
    typename ElementD = ElementA>
class MoEGEMM {
 public:
  using TiledCopyA = decltype(make_block_2d_copy_A(TiledMMA{}, TensorA{}));
  using TiledCopyB = decltype(make_block_2d_copy_B(TiledMMA{}, TensorB{}));
  using TiledCopyD = decltype(make_block_2d_copy_D(TiledMMA{}, TensorD{}));
  using SGPerWG = decltype(product(take<1, 4>(shape(typename TiledMMA::ThrLayoutVMNK{}))));

  constexpr static int Stages = 3;
  using MainloopDispatchPolicy = MoE::XeDefault<Stages>;
  using CollectiveMainloop = MoEMainloop<
      MainloopDispatchPolicy,
      TiledCopyA,
      TiledCopyB,
      TiledCopyD,
      TensorA,
      TensorB,
      TensorD,
      TensorBias,
      TiledMMA,
      WithBias>;

  struct Params {
    const ElementA* Activations;
    const ElementB* Weights;
    const ElementD* Bias;
    ElementD* Outputs;
    const int32_t* M_per_group;
    const int32_t N;
    const int32_t K;
    const int32_t num_experts;
    int32_t* workspace;
    TiledMMA mma;
  };

  auto make_B_tensors(ElementB* ptr_B, int N, int K) {
    if constexpr (FuseSiLU) {
      auto B0 = make_tensor(make_gmem_ptr<ElementB>(ptr_B), make_layout(make_shape(N / 2, K), make_stride(K, _1{})));
      ElementB* ptr_B1 = ptr_B + (N / 2) * K;
      auto B1 = make_tensor(make_gmem_ptr<ElementB>(ptr_B1), make_layout(make_shape(N / 2, K), make_stride(K, _1{})));
      return cute::make_tuple(B0, B1);
    } else {
      auto B = make_tensor(make_gmem_ptr<ElementB>(ptr_B), make_layout(make_shape(N, K), make_stride(K, _1{})));
      return cute::make_tuple(B);
    }
  }

  auto make_Bias_tensors(ElementD* ptr_Bias, int N) {
    if constexpr (WithBias) {
      if constexpr (FuseSiLU) {
        auto Bias0 = make_tensor(make_gmem_ptr<ElementD>(ptr_Bias), make_layout(make_shape(N / 2), make_stride(_1{})));
        ElementD* ptr_Bias1 = ptr_Bias + (N / 2);
        auto Bias1 = make_tensor(make_gmem_ptr<ElementD>(ptr_Bias1), make_layout(make_shape(N / 2), make_stride(_1{})));
        return cute::make_tuple(Bias0, Bias1);
      } else {
        auto Bias = make_tensor(make_gmem_ptr<ElementD>(ptr_Bias), make_layout(make_shape(N), make_stride(_1{})));
        return cute::make_tuple(Bias);
      }
    } else {
      // return a tuple of empty tensors
      return cute::make_tuple(
          make_tensor(make_gmem_ptr<ElementD>(nullptr), make_layout(make_shape(0), make_stride(_1{}))),
          make_tensor(make_gmem_ptr<ElementD>(nullptr), make_layout(make_shape(0), make_stride(_1{}))));
    }
  }

  auto make_D_tensors(ElementD* ptr_D, int pre_rows, int M, int N) {
    if constexpr (FuseSiLU) {
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
    if constexpr (FuseSiLU) {
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
      int64_t B_offset = static_cast<int64_t>(expert_id) * static_cast<int64_t>(N) * static_cast<int64_t>(K);
      ElementA* ptr_A_curr_batch = const_cast<ElementA*>(params.Activations) + pre_rows * K;
      ElementB* ptr_B_curr_batch = const_cast<ElementB*>(params.Weights) + B_offset;
      ElementD* ptr_Bias_curr_batch = nullptr;
      if constexpr (WithBias) {
        ptr_Bias_curr_batch = const_cast<ElementD*>(params.Bias) + expert_id * N;
      }

      auto A_tensor =
          make_tensor(make_gmem_ptr<ElementA>(ptr_A_curr_batch), make_layout(make_shape(M, K), make_stride(K, _1{})));
      auto B_tensor = make_B_tensors(ptr_B_curr_batch, N, K);
      auto D_tensor = make_D_tensors(params.Outputs, pre_rows, M, N);
      auto Bias_tensor = make_Bias_tensors(ptr_Bias_curr_batch, N);

      while (group_m_id < cumsum_tiles_for_experts) {
        int n_coord = (group_id * wg_tile_n) % N_pad / wg_tile_n;
        int m_coord = (group_m_id - pre_tiles);

        CollectiveMainloop mainloop;
        if constexpr (FuseSiLU) {
          auto tile_coord = make_coord(m_coord, n_coord, 0);
          mainloop(
              A_tensor,
              get<0>(B_tensor),
              get<1>(B_tensor),
              D_tensor,
              tile_coord,
              mma,
              thr_id,
              get<0>(Bias_tensor),
              get<1>(Bias_tensor));
        } else {
          auto tile_coord = make_coord(m_coord, n_coord, _, 0);
          mainloop(A_tensor, get<0>(B_tensor), D_tensor, tile_coord, mma, thr_id);
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
}  // namespace MoE
