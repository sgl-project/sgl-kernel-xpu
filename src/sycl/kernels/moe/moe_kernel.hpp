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

#include <cute/util/compat.hpp>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/platform/platform.h"
#include "cutlass/util/packed_stride.hpp"
#include "moe_mainloop.hpp"
#include "moe_tile_scheduler.hpp"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace MoE {
using namespace cute;

using ProblemShapeMNKL = Shape<int, int, int, int>;
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
using TileScheduler = typename MoE::PersistentTileSchedulerXeMoE<ProblemShape>;
using RasterOrderOptions = typename TileScheduler::RasterOrderOptions;

template <
    typename TileShape,
    typename SubgroupLayout,
    typename TensorA,
    typename TensorB,
    typename TensorD,
    typename ElementA,
    typename ElementB = ElementA,
    typename ElementS = ElementA,
    typename ElementD = ElementA>
class MoEGEMM {
 public:
  // init mma and copy
  static constexpr int SGTileQ = get<0>(shape_div(TileShape{}, shape(SubgroupLayout{})))();
  using MMAOperation = XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, ElementA, ElementB>;
  using TiledMMA = typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<TileShape>, SubgroupLayout>::TiledMMA;

  using TiledCopyA = decltype(make_block_2d_copy_A(TiledMMA{}, TensorA{}));
  using TiledCopyB = decltype(make_block_2d_copy_B(TiledMMA{}, TensorB{}));
  using TiledCopyD = decltype(make_block_2d_copy_D(TiledMMA{}, TensorD{}));
  using SGPerWG = decltype(product(take<1, 4>(shape(typename TiledMMA::ThrLayoutVMNK{}))));

  constexpr static int Stages = 3;
  using MainloopDispatchPolicy = MoE::XeDefault<Stages>;
  using CollectiveMainloop =
      MoEMainloop<MainloopDispatchPolicy, TiledCopyA, TiledCopyB, TiledCopyD, TensorA, TensorB, TensorD, TiledMMA>;

  struct Params {
    const ElementA* Activations;
    const ElementB* Weights;
    ElementD* Outputs;
    const int32_t* M_per_group;
    const int32_t N;
    const int32_t K;
    const int32_t num_experts;
    PersistentTileSchedulerSm90GroupParams<ProblemShape> scheduler_params;
  };

  void operator()(Params const& params, sycl::nd_item<3> item) {
    auto N = params.N;
    auto K = params.K;
    auto M_per_group = params.M_per_group;
    auto num_experts = params.num_experts;
    TileScheduler scheduler{params.scheduler_params, const_cast<int32_t*>(M_per_group), N, K, num_experts};
    auto work_tile_info = scheduler.initial_work_tile_info(Shape<_1, _1, _1>{});

    bool did_group_change = true;
    int32_t curr_group = 0;
    int32_t prev_group = 0;
    int32_t cumulative_M = 0;
    int32_t M = 0;
    int32_t thr_id = int32_t(item.get_local_linear_id());

    if (work_tile_info.is_valid()) {
      // We don't really need this conditional outside the while loop.
      // It simply helps initialize tensors. If using nullptr would be
      // fine for their initialization, then we can remove this conditional.
      curr_group = work_tile_info.L_idx;
      M = M_per_group[curr_group];
    }

    ElementA* ptr_A_curr_batch = const_cast<ElementA*>(params.Activations);
    ElementB* ptr_B_curr_batch = const_cast<ElementB*>(params.Weights);
    ElementD* ptr_D_curr_batch = params.Outputs;
    auto A_tensor =
        make_tensor(make_gmem_ptr<ElementA>(ptr_A_curr_batch), make_layout(make_shape(M, K), make_stride(K, _1{})));
    auto B_tensor =
        make_tensor(make_gmem_ptr<ElementB>(ptr_B_curr_batch), make_layout(make_shape(N, K), make_stride(K, _1{})));
    auto D_tensor =
        make_tensor(make_gmem_ptr<ElementD>(ptr_D_curr_batch), make_layout(make_shape(M, N), make_stride(N, _1{})));

    while (work_tile_info.is_valid()) {
      auto m_coord = work_tile_info.M_idx;
      auto n_coord = work_tile_info.N_idx;
      auto tile_coord = make_coord(m_coord, n_coord);

      if (did_group_change) {
        curr_group = work_tile_info.L_idx;
        M = M_per_group[curr_group];
        // recompute each time because the groups don't necessarily increment by 1
        for (int i = prev_group; i < curr_group; i++) {
          cumulative_M += M_per_group[i];
        }
        prev_group = curr_group;

        ptr_A_curr_batch = const_cast<ElementA*>(params.Activations) + cumulative_M * K;
        ptr_B_curr_batch = const_cast<ElementB*>(params.Weights) + curr_group * K * N;
        ptr_D_curr_batch = params.Outputs + cumulative_M * N;

        auto A_tensor =
            make_tensor(make_gmem_ptr<ElementA>(ptr_A_curr_batch), make_layout(make_shape(M, K), make_stride(K, _1{})));
        auto B_tensor =
            make_tensor(make_gmem_ptr<ElementB>(ptr_B_curr_batch), make_layout(make_shape(N, K), make_stride(K, _1{})));
        auto D_tensor =
            make_tensor(make_gmem_ptr<ElementD>(ptr_D_curr_batch), make_layout(make_shape(M, N), make_stride(N, _1{})));
        did_group_change = false;
      }

      CollectiveMainloop mainloop;
      mainloop(A_tensor, B_tensor, D_tensor, tile_coord, thr_id);

      work_tile_info = scheduler.fetch_next_work(work_tile_info);
      did_group_change = curr_group != work_tile_info.L_idx;
    }  // end while loop
  }
};

}  // namespace MoE
