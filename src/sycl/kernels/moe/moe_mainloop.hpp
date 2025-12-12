/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
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

#include <cute/tensor.hpp>
#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/sycl_event_manager.hpp"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

template <typename T>
struct dump;

namespace MoE {

using namespace cute;

template <int Stages>
class XeDefault {};

template <
    class DispatchPolicy_,
    class TiledCopyA_,
    class TiledCopyB_,
    class TilesCopyD_,
    class ATensor_,
    class BTensor_,
    class DTensor_,
    class TiledMMA_>
struct MoEMainloop {
  static_assert(cutlass::detail::dependent_false<DispatchPolicy_>, "Could not find a mainloop specialization.");
};

template <
    int Stages,
    class TiledCopyA_,
    class TiledCopyB_,
    class TilesCopyD_,
    class ATensor_,
    class BTensor_,
    class DTensor_,
    class TiledMMA_>
struct MoEMainloop<XeDefault<Stages>, TiledCopyA_, TiledCopyB_, TilesCopyD_, ATensor_, BTensor_, DTensor_, TiledMMA_> {
  using TiledMMA = TiledMMA_;
  using TiledCopyA = TiledCopyA_;
  using TiledCopyB = TiledCopyB_;
  using TiledCopyD = TilesCopyD_;
  using ATensor = ATensor_;
  using BTensor = BTensor_;
  using DTensor = DTensor_;
  MoEMainloop() {}

  template <typename Coord>
  CUTLASS_DEVICE void operator()(
      ATensor& A,  // (M,K)
      BTensor& B,  // (N,K)
      DTensor& D,  // (M,N)
      Coord blk_coord,
      TiledMMA mma,
      int thr_id) {  // work-item ID

    auto wg_m = get<0>(blk_coord);
    auto wg_n = get<1>(blk_coord);

    /* Create proxy coordinate tensors for A/B/C */
    Tensor cA = make_identity_tensor(A.shape());  // (M,K)
    Tensor cB = make_identity_tensor(B.shape());  // (N,K)
    Tensor cD = make_identity_tensor(D.shape());  // (M,N)

    /* init mma */
    auto wg_tile = mma.tile_mnk();
    auto wg_coord = make_coord(wg_m, wg_n, 0);

    Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(wg_m, _));  // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(wg_n, _));  // (BLK_N,BLK_K,k)
    Tensor gD = local_tile(cD, wg_tile, wg_coord, Step<_1, _1, X>{});        // (BLK_M,BLK_N)

    /* Create global -> register copies */
    TiledCopyA tiled_copy_a{A};
    TiledCopyB tiled_copy_b{B};
    TiledCopyD tiled_copy_d{D};

    /* Slice TiledCopy/TiledMMA operations down to to work-item level */
    auto thr_copy_a = tiled_copy_a.get_slice(thr_id);
    auto thr_copy_b = tiled_copy_b.get_slice(thr_id);
    auto thr_copy_d = tiled_copy_d.get_slice(thr_id);
    auto thr_mma = mma.get_slice(thr_id);

    /* Partition coordinate tensors for copy */
    auto tAgA = thr_copy_a.partition_S(gA);
    auto tBgB = thr_copy_b.partition_S(gB);

    /* Create register fragments for MMA and copies */
    auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
    auto tSrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));

    auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));
    auto tSrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

    /* Partition C */
    SubgroupTensor tCrC = thr_mma.partition_sg_fragment_C(gD);

    /* Partition D */
    using TD = typename DTensor::element_type;
    TD tCrD_final_frag[tCrC.size()];
    Tensor tCrD_final_tensor = make_tensor(make_rmem_ptr(tCrD_final_frag), tCrC.layout());
    SubgroupTensor tCrD_final_sg_tensor = make_subgroup_tensor(tCrD_final_tensor, tCrC.tv_layout());
    Tensor tCgD = thr_mma.partition_C(gD);

    /* Create TiledCopy objects for prefetches */
    auto prefetch_a = make_block_2d_prefetch(tiled_copy_a);
    auto prefetch_b = make_block_2d_prefetch(tiled_copy_b);

    /* Partition global tensors for prefetch */
    auto pAgA = prefetch_a.get_slice(thr_id).partition_S(gA);
    auto pBgB = prefetch_b.get_slice(thr_id).partition_S(gB);

    constexpr int barrier_scope = 2;
    int k_start_idx = 0;
    int prefetch_k = k_start_idx;
    const int prefetch_dist = Stages;
    int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));

    // ------
    // Compute
    // ------

    CUTE_UNROLL
    for (; prefetch_k < prefetch_dist; prefetch_k++) {
      prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
      prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
    }

    for (int k_tile = k_start_idx; k_tile < k_tile_count; k_tile++, prefetch_k++) {
      barrier_arrive(barrier_scope);

      copy(tiled_copy_a, tAgA(_, _, _, k_tile), tArA);
      copy(tiled_copy_b, tBgB(_, _, _, k_tile), tBrB);

      if (prefetch_k < k_tile_count) {
        prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
        prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
      }

      reorder(tArA, tSrA);
      reorder(tBrB, tSrB);

      cute::gemm(mma, tSrA, tSrB, tCrC);
      barrier_wait(barrier_scope);
    }
    reorder(tCrC, tCrD_final_sg_tensor);
    copy(tiled_copy_d, tCrD_final_sg_tensor, tCgD);
  }
};
}  // namespace MoE
