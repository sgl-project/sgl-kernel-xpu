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
    \brief Per-group scalar (alpha/beta) fix for the Intel Xe grouped-GEMM epilogue.

    Problem
    -------
    In a grouped GEMM each group should compute D = alpha[g]*(A@B) + beta[g]*C.
    The stock Xe grouped collective epilogue
    (cutlass::epilogue::collective::CollectiveEpilogue<IntelXeGenericGroup, ...>)
    selects each group's C/D *base pointers* per group in to_base_arguments()
    (the `[idx]` indexing), but the array kernel then hands the fusion a tile
    coordinate whose L (batch/group) slot is hard-coded to 0
    (xe_gemm_array_cooperative.hpp: `make_coord(m_coord, n_coord, _, 0)`). The
    fusion's Sm90ScalarBroadcastPtrArray reads the scalar as
    `*(alpha_ptr_array[l_coord])`, so with l_coord always 0 EVERY group ends up
    using alpha_ptr_array[0] / beta_ptr_array[0] -- per-group scalars are ignored.

    Fix
    ---
    Mirror the C/D pre-offsetting that already works: in to_base_arguments()
    advance the alpha/beta pointer arrays by the group index, so the fusion's
    `[0]` (l_coord==0) resolves to *this group's* scalar. C/D addressing is
    unaffected (their pointers are pre-offset per group and l_coord stays 0).

    Why a subclass (not editing sycl-tla, not a full custom collective)
    -------------------------------------------------------------------
    * The epilogue *math* is unchanged (stock LinearCombination is correct), so a
      full custom collective/mainloop/kernel is overkill.
    * GemmUniversal selects the kernel layer purely from the mainloop dispatch
      policy (see xe_gemm_array_cooperative.hpp: enable_if on
      CollectiveMainloop::DispatchPolicy::Schedule); the CollectiveEpilogue is a
      free type parameter. So substituting a subclass whose static
      to_base_arguments() shadows the base one is picked up automatically -- the
      kernel calls CollectiveEpilogue::to_base_arguments(...) on our type.

    Everything else (Base, Params, Arguments, SharedStorage, operator(),
    to_underlying_arguments, can_implement, get_workspace_size, ...) is inherited
    unchanged. IMPORTANT: we deliberately do NOT redeclare the inherited `Base`
    alias -- the array kernel refers to CollectiveEpilogue::Base (the underlying
    IntelXeGeneric epilogue) and must keep resolving to it.

    Upstream: the real bug is the hard-coded L coord in sycl-tla's
    xe_gemm_array_cooperative.hpp; this subclass is the owned local workaround
    until that is fixed upstream.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/xe_array_epilogue.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"

namespace cutlass::lora::kernel {

// Drop-in replacement for
//   cutlass::epilogue::collective::CollectiveEpilogue<IntelXeGenericGroup, Ts...>
// that applies per-group alpha/beta by pre-offsetting the scalar pointer arrays.
// Same template parameter list as the stock IntelXeGenericGroup specialization.
template <
    class WGTileMNK_,
    class EpilogueTile_,
    class ElementC_,
    class StrideC_,
    class ElementD_,
    class StrideD_,
    class FusionCallbacks_,
    class CopyOpG2R_,
    class CopyOpR2G_>
class GroupedEpiloguePerGroupScalar : public cutlass::epilogue::collective::CollectiveEpilogue<
                                          cutlass::epilogue::IntelXeGenericGroup,
                                          WGTileMNK_,
                                          EpilogueTile_,
                                          ElementC_,
                                          StrideC_,
                                          ElementD_,
                                          StrideD_,
                                          FusionCallbacks_,
                                          CopyOpG2R_,
                                          CopyOpR2G_> {
 private:
  using StockGroupEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
      cutlass::epilogue::IntelXeGenericGroup,
      WGTileMNK_,
      EpilogueTile_,
      ElementC_,
      StrideC_,
      ElementD_,
      StrideD_,
      FusionCallbacks_,
      CopyOpG2R_,
      CopyOpR2G_>;

 public:
  using Arguments = typename StockGroupEpilogue::Arguments;  // host-side epilogue args (fusion thread + C/D ptrs)
  using BaseArguments =
      typename StockGroupEpilogue::BaseArguments;  // per-group args consumed by the underlying epilogue

  // Called once per group by the array kernel (guarded by did_group_change) with
  // idx == curr_group. Identical to the stock method except alpha/beta pointer
  // arrays are advanced by idx so the fusion's l_coord==0 read lands on this
  // group's scalar. Cost: one pointer add per array per group -- outside the
  // compute/epilogue hot loops.
  CUTLASS_DEVICE static constexpr BaseArguments to_base_arguments(Arguments const& args, int idx) {
    auto thread = args.thread;
    if (thread.alpha_ptr_array != nullptr) {
      thread.alpha_ptr_array += idx;
    }
    if (thread.beta_ptr_array != nullptr) {
      thread.beta_ptr_array += idx;
    }
    return BaseArguments{thread, args.ptr_C[idx], args.dC[idx], args.ptr_D[idx], args.dD[idx]};
  }
};

}  // namespace cutlass::lora::kernel
