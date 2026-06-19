/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief XPU Gemm Square Sum Kernel
*/

#pragma once

#include "../collective/xe_hc_pre_gemm_sqr_sum_epilogue.hpp"
#include "../collective/xe_hc_pre_gemm_sqr_sum_mainloop.hpp"
#include "cute/algorithm/copy.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/util/compat/dims.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "hc_pre_gemm_sqr_sum_tile_scheduler.hpp"

namespace cutlass::hc_pre_gemm_sqr_sum::kernel {
using namespace cute;

template <class ProblemShape_, class CollectiveMainloop_, class CollectiveEpilogue_>
class XeHcPreGemmSqrSumKernel {
 public:
  using ProblemShape = ProblemShape_;
  using CollectiveMainloop = CollectiveMainloop_;
  using CollectiveEpilogue = CollectiveEpilogue_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using ElementA = typename CollectiveMainloop::ElementA;
  using ElementB = typename CollectiveMainloop::ElementB;
  using ElementC = typename CollectiveMainloop::ElementC;
  using ElementSqrSum = typename CollectiveMainloop::ElementSqrSum;
  using StrideA = decltype(stride(typename CollectiveMainloop::TensorA{}));
  using StrideB = decltype(stride(typename CollectiveMainloop::TensorB{}));
  using StrideC = cute::Stride<int, cute::_1>;
  using StrideSqsum = cute::Stride<int, cute::_1>;
  using TileScheduler = XeHcPreGemmSqrSumTileScheduler;
  using TileSchedulerParams = typename TileScheduler::Params;

  static constexpr auto BLK_M = get<0>(TileShape{});
  static constexpr auto BLK_N = get<1>(TileShape{});
  static constexpr auto BLK_K = get<2>(TileShape{});

  struct SharedStorage {
    typename CollectiveMainloop::SharedStorage mainloop;
    typename CollectiveEpilogue::SharedStorage epilogue;
  };

  struct KernelArguments {
    ProblemShape shape{};

    ElementA const* ptr_A = nullptr;
    StrideA dA{};

    ElementB const* ptr_B = nullptr;
    StrideB dB{};

    ElementC* ptr_C = nullptr;
    StrideC dC{};

    ElementSqrSum* ptr_sqr_sum = nullptr;
    ElementSqrSum* ptr_sqr_sum_scratch = nullptr;
    StrideSqsum dSqsum{};

    KernelArguments() = default;
  };

  using KernelParams = KernelArguments;

  struct Params {
    KernelArguments kernel{};
    typename CollectiveMainloop::Params mainloop{};
    TileSchedulerParams scheduler{};
    int split_k = 1;
  };

  struct Arguments {
    KernelArguments kernel{};
    typename CollectiveMainloop::Arguments mainloop{};
    cutlass::KernelHardwareInfo hw_info{};
    int split_k = 1;
  };

  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    return Params{
        args.kernel,
        CollectiveMainloop::to_underlying_arguments(args.mainloop, workspace),
        TileScheduler::to_underlying_arguments(args.kernel.shape, args.hw_info, TileShape{}, args.split_k),
        args.split_k};
  }

  static bool can_implement(Arguments const& args) {
    return CollectiveMainloop::can_implement(args.mainloop);
  }

  static int get_workspace_size(Arguments const& args) {
    return 0;
  }

  static compat::dim3 get_grid_shape(Params const& params) {
    return TileScheduler::template get_grid_shape<1>(params.scheduler);
  }

  static compat::dim3 get_block_shape() {
    constexpr int num_threads = cute::size(typename CollectiveMainloop::TiledMMA{});
    return compat::dim3(num_threads, 1, 1);
  }

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
    using namespace sycl::ext::oneapi;

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    auto& p = params.kernel;
    ProblemShape const& s = p.shape;

    int thr_id = int(this_work_item::get_nd_item<3>().get_local_id(2));

    TileScheduler scheduler{params.scheduler};
    auto [blk_m, blk_n, split_idx] = scheduler.get_block_coord();

    int k_tiles_total = (s.K + BLK_K - 1) / BLK_K;
    int split_k = params.split_k;
    int tiles_per_split = (k_tiles_total + split_k - 1) / split_k;
    int k_tile_begin = split_idx * tiles_per_split;
    int k_tile_end = k_tile_begin + tiles_per_split;
    if (k_tile_end > k_tiles_total) k_tile_end = k_tiles_total;

    int64_t c_slab_elems = int64_t(s.M) * int64_t(s.N);
    ElementC* ptr_C_split = p.ptr_C + split_idx * c_slab_elems;
    ElementSqrSum* ptr_sqsum_split = p.ptr_sqr_sum_scratch + int64_t(split_idx) * int64_t(s.M);

    auto layout_A = make_layout(make_shape(s.M, s.K), p.dA);
    auto layout_B = make_layout(make_shape(s.N, s.K), p.dB);
    auto layout_C = make_layout(make_shape(s.M, s.N), p.dC);
    auto layout_Ssq = make_layout(make_shape(s.M, 1), p.dSqsum);

    Tensor A = make_tensor(make_gmem_ptr(p.ptr_A), layout_A);
    Tensor B = make_tensor(make_gmem_ptr(p.ptr_B), layout_B);
    Tensor C = make_tensor(make_gmem_ptr(ptr_C_split), layout_C);
    Tensor Ssq = make_tensor(make_gmem_ptr(ptr_sqsum_split), layout_Ssq);

    auto A_2D = A(append<rank_v<decltype(A)>>(make_coord(_, _), 0));
    auto B_2D = B(append<rank_v<decltype(B)>>(make_coord(_, _), 0));

    CollectiveMainloop mainloop(params.mainloop, shared_storage.mainloop);

    typename CollectiveMainloop::FragGemm tC;
    typename CollectiveMainloop::FragSqrSum tSqrSum;

    mainloop(A_2D, B_2D, tC, tSqrSum, make_coord(blk_m, blk_n), thr_id, k_tile_begin, k_tile_end);

    CollectiveEpilogue epilogue({}, shared_storage.epilogue);
    epilogue(C, Ssq, tC, tSqrSum, make_coord(blk_m, blk_n), thr_id);
  }
};

}  // namespace cutlass::hc_pre_gemm_sqr_sum::kernel
