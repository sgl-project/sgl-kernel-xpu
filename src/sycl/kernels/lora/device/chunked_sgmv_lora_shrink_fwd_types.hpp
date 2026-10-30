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
/*!
  \file
  \brief Tile-configuration option tags for the chunked-SGMV LoRA shrink kernel.

  Adding another tile is a two-step change (mirrors the prefill LoRA kernels):
    1) Define a new option tag here.
    2) Register (tag name, C++ type) in ChunkedSgmvLoraShrinkFwd.cmake.
  The entrypoint (ChunkedSgmvLoraShrinkFwd.cpp) then picks a tag per call.
*/

#pragma once

#include "sycl/kernels/lora/common/gather_scatter_group_gemm_types.hpp"

namespace chunked_sgmv_lora_shrink_fwd_impl {

// Sub-group tile 16 x 32, k-blocks of 32, 8 sub-groups splitting K inside one work-group (the
// split-K epilogue reduces them through SLM). BM = 16 keeps the kernel at 128 GRF (twice the
// hardware threads of a 32-row tile). Measured on BMG (B580) over the decode shapes in
// benchmark/bench_chunked_sgmv_lora_shrink_fwd.py. Types<T, BLOCK_B> is the assembled gather /
// scatter grouped-GEMM bundle (sycl-tla GemmUniversal with the LoRA collectives).
struct ChunkedShrinkTileSplitK {
  template <typename T, bool BLOCK_B>
  using Types = at::native::xpu::lora::GatherScatterGroupGemmTypes<
      T,
      /*BM=*/16,
      /*SG_N=*/32,
      /*BK=*/32,
      /*NSG_N=*/1,
      /*SGK=*/8,
      cutlass::lora::collective::SplitKScatterGroupEpilogue,
      BLOCK_B>;
};

}  // namespace chunked_sgmv_lora_shrink_fwd_impl
