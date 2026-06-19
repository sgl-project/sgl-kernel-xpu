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
    \brief Gemm Square Sum Epilogue
*/

#pragma once

#include "cute/algorithm/copy.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"

namespace cutlass::gemm_sqrsum::collective {
using namespace cute;

template <class CollectiveMainloop_>
class XeGemmSqrSumEpilogue {
 public:
  using CollectiveMainloop = CollectiveMainloop_;
  using TiledMMA = typename CollectiveMainloop::TiledMMA;
  using FragGemm = typename CollectiveMainloop::FragGemm;
  using FragSqrSum = typename CollectiveMainloop::FragSqrSum;
  struct Arguments {};
  using Params = Arguments;
  struct SharedStorage {};
  Params params;
  SharedStorage& shared;
  CUTLASS_HOST_DEVICE
  XeGemmSqrSumEpilogue(Params const& params_, SharedStorage& shared_) : params(params_), shared(shared_) {}
  static constexpr Params to_underlying_arguments(Arguments const& args, void* /* workspace */) {
    return {};
  }
  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) {
    return true;
  }
  template <typename TensorC, typename TensorSqrSum, typename MNCoord>
  CUTLASS_DEVICE void
  operator()(TensorC const& C, TensorSqrSum const& Ssc, FragGemm& tC, FragSqrSum& tSqrSum, MNCoord blk_mn, int thr_id) {
    TiledMMA mma{};
    auto thr_mma = mma.get_slice(thr_id);
    auto blk_coord = make_coord(get<0>(blk_mn), get<1>(blk_mn), 0);
    auto cC = make_identity_tensor(C.shape());
    auto gC = local_tile(cC, mma.tile_mnk(), blk_coord, Step<_1, _1, X>{});
    auto copy_c = make_block_2d_copy_D(mma, C);
    copy(copy_c, tC, thr_mma.partition_C(gC));
    auto cS = make_identity_tensor(Ssc.shape());
    auto gS = local_tile(cS, mma.tile_mnk(), blk_coord, Step<_1, _1, X>{});
    auto copy_s = make_block_2d_copy_D(mma, Ssc);
    copy(copy_s, tSqrSum, thr_mma.partition_C(gS));
  }
};
}  // namespace cutlass::gemm_sqrsum::collective
