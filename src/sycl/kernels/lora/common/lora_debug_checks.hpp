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
/*!
  \file
  \brief Value-level LoRA input validation, compiled only in debug-check builds.

  These checks read tensor *values* (not just shapes), so each one costs a
  device -> host sync (`.item()`, `.cpu()`): the host stalls until the GPU has
  drained its queue. For small decode batches that costs more than the kernel
  itself, so they are compiled only when the build sets USE_DEBUG_CHECKS=ON
  (which sets SGL_KERNEL_DEBUG_CHECKS=1 in the generated sgl_kernel_debug_config.h):

      USE_DEBUG_CHECKS=ON pip install -e . --no-build-isolation -v

  In a default build every function below is an empty inline function, so call
  sites need no #ifdef and the fast path issues no syncs. Shape / dtype /
  contiguity checks are host-only metadata and stay on in every build.
*/

#pragma once

#include <ATen/ATen.h>
#include <torch/all.h>

#include <algorithm>
#include <cstdint>

#include "sgl_kernel_debug_config.h"  // generated: SGL_KERNEL_DEBUG_CHECKS = 0 / 1

namespace lora_debug_checks {

inline constexpr bool kEnabled = SGL_KERNEL_DEBUG_CHECKS != 0;

// weight_indices[s] in [0, num_loras) for every segment.
inline void check_weight_indices(const torch::Tensor& weight_indices, int64_t num_loras, const char* weights_name) {
  if constexpr (kEnabled) {
    if (weight_indices.numel() == 0) {
      return;
    }
    auto [min_wi, max_wi] = torch::aminmax(weight_indices);
    TORCH_CHECK(
        min_wi.item<int64_t>() >= 0 && max_wi.item<int64_t>() < num_loras,
        "weight_indices values must be in [0, ",
        weights_name,
        ".size(0))");
  }
}

// seg_indptr starts at 0, ends at num_tokens, and is non-decreasing.
inline void check_seg_indptr(const torch::Tensor& seg_indptr, int64_t num_tokens) {
  if constexpr (kEnabled) {
    const int64_t n = seg_indptr.numel();
    if (n < 2) {
      return;  // the host-side shape checks already cover this
    }
    TORCH_CHECK(seg_indptr[0].item<int64_t>() == 0, "seg_indptr[0] must be 0");
    TORCH_CHECK(seg_indptr[n - 1].item<int64_t>() == num_tokens, "seg_indptr[-1] must equal num_tokens");
    auto seg_lens = seg_indptr.slice(0, 1) - seg_indptr.slice(0, 0, n - 1);
    TORCH_CHECK(seg_lens.min().item<int64_t>() >= 0, "seg_indptr must be non-decreasing");
  }
}

// lora_ranks[l] in [0, max_rank] for every adapter.
inline void check_lora_ranks(const torch::Tensor& lora_ranks, int64_t max_rank) {
  if constexpr (kEnabled) {
    if (lora_ranks.numel() == 0) {
      return;
    }
    auto [min_lr, max_lr] = torch::aminmax(lora_ranks);
    TORCH_CHECK(
        min_lr.item<int64_t>() >= 0 && max_lr.item<int64_t>() <= max_rank,
        "All values in lora_ranks must be within the range [0, max_rank]");
  }
}

// permutation (logical -> physical) is a bijection over [0, num_tokens). The
// chunked-SGMV kernels scatter output rows through it, so a repeated index
// would make two work-groups race on the same output row.
inline void check_permutation(const torch::Tensor& permutation, int64_t num_tokens) {
  if constexpr (kEnabled) {
    if (num_tokens == 0) {
      return;
    }
    auto [min_p, max_p] = torch::aminmax(permutation);
    TORCH_CHECK(
        min_p.item<int64_t>() >= 0 && max_p.item<int64_t>() < num_tokens,
        "permutation values must be in [0, x.size(0))");
    auto sorted = std::get<0>(torch::sort(permutation.to(torch::kInt64)));
    auto identity = torch::arange(num_tokens, sorted.options());
    TORCH_CHECK(torch::equal(sorted, identity), "permutation must be a bijection over [0, num_tokens) (no repeats)");
  }
}

// Output-column boundary table [0, ..., n_total] (slice_offsets / output_offset):
// starts at 0, ends at n_total, non-decreasing, and its widest band equals the
// caller-supplied max width.
inline void check_column_offsets(
    const torch::Tensor& offsets,
    int64_t n_total,
    int64_t max_width,
    const char* offsets_name,
    const char* n_total_desc,
    const char* max_width_name) {
  if constexpr (kEnabled) {
    auto off_cpu = offsets.to(torch::kInt64).cpu();
    const int64_t* off = off_cpu.data_ptr<int64_t>();
    const int64_t n_bands = off_cpu.numel() - 1;
    TORCH_CHECK(off[0] == 0, offsets_name, "[0] must be 0");
    TORCH_CHECK(off[n_bands] == n_total, offsets_name, "[-1] must equal ", n_total_desc);
    int64_t widest = 0;
    for (int64_t p = 0; p < n_bands; ++p) {
      TORCH_CHECK(off[p] <= off[p + 1], offsets_name, " must be non-decreasing");
      widest = std::max<int64_t>(widest, off[p + 1] - off[p]);
    }
    TORCH_CHECK(max_width == widest, max_width_name, " must equal the widest band implied by ", offsets_name);
  }
}

}  // namespace lora_debug_checks
