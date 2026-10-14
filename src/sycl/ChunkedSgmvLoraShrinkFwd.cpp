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
    \brief Chunked-SGMV LoRA "shrink" (A-matrix) forward entrypoint.

    Computes the LoRA-A projection D = x @ weights[l]^T as a segmented grouped
    GEMM, where each segment s uses adapter l = weight_indices[s]. Used in the
    decode phase, where the batch interleaves adapters ("zigzag").

    seg_indptr / weight_indices describe the *logical* (adapter-grouped) layout;
    permutation (logical -> physical, a bijection over the token rows) maps that
    to the physical token order of x / output. When permutation is absent the
    batch is already contiguous by adapter (prefill) and seg_indptr /
    weight_indices are in *physical* layout.

    Implementation (device/chunked_sgmv_lora_shrink_fwd_kernel.hpp, launched through
    the generated per-(dtype, tile) launchers declared in
    chunked_sgmv_lora_shrink_fwd_dispatch.hpp): a single DPAS kernel that reads
    x rows through the permutation and scatters output rows through it, with an
    intra-work-group split-K for the long K = input_dim reduction -- no sorted
    copies, no host syncs. Any input_dim, N and pointer alignment is supported
    (masked K tail; alignment-free loads where 16-byte block reads are not possible).
*/

#define SYCL_INTEL_TARGET 20

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <optional>
#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"
#include "kernels/lora/common/lora_debug_checks.hpp"
#include "kernels/lora/device/chunked_sgmv_lora_shrink_fwd_dispatch.hpp"
#include "sgl_kernel_export.h"

namespace {

//----------------- Per-(dtype, tile) dispatch macros --------------------//
// One tile (splitk) for now. Add tiles to DISPATCH_CHUNKED_SGMV_LORA_SHRINK_FWD_TILE
// (and to ChunkedSgmvLoraShrinkFwdXe20.cmake + chunked_sgmv_lora_shrink_fwd_dispatch.hpp
// + chunked_sgmv_lora_shrink_fwd_types.hpp) with a runtime heuristic picking the tag.
#define DISPATCH_CHUNKED_SGMV_LORA_SHRINK_FWD_TILE(ELEM, ...)                                            \
  do {                                                                                                   \
    chunked_sgmv_lora_shrink_fwd_impl::launch_chunked_sgmv_lora_shrink_fwd_##ELEM##_splitk(__VA_ARGS__); \
  } while (0)

#define DISPATCH_CHUNKED_SGMV_LORA_SHRINK_FWD_DTYPE(...)                                                               \
  do {                                                                                                                 \
    switch (weights.scalar_type()) {                                                                                   \
      case torch::kHalf:                                                                                               \
        DISPATCH_CHUNKED_SGMV_LORA_SHRINK_FWD_TILE(half, __VA_ARGS__);                                                 \
        break;                                                                                                         \
      case torch::kBFloat16:                                                                                           \
        DISPATCH_CHUNKED_SGMV_LORA_SHRINK_FWD_TILE(bf16, __VA_ARGS__);                                                 \
        break;                                                                                                         \
      default:                                                                                                         \
        TORCH_CHECK(false, "Unsupported data type for chunked_sgmv_lora_shrink_fwd weights: ", weights.scalar_type()); \
    }                                                                                                                  \
  } while (0)

}  // namespace

//----------------- Main API function --------------------//

SGL_KERNEL_EXPORT void chunked_sgmv_lora_shrink_fwd(
    torch::Tensor& output,                           // [num_tokens, num_slices*max_rank]  (physical token order)
    const torch::Tensor& x,                          // [num_tokens, input_dim]  (physical token order)
    const torch::Tensor& weights,                    // [num_loras, num_slices*max_rank, input_dim]
    const int64_t num_slices,                        // stacked projections (qkv=3, gate_up=2, else 1)
    const int64_t num_segments,                      // currently the code does not rely on it.
    const torch::Tensor& seg_indptr,                 // [num_segments + 1,]   over the logical row order
    const torch::Tensor& weight_indices,             // [num_segments,]
    const torch::Tensor& lora_ranks,                 // [num_loras,]
    const std::optional<torch::Tensor>& permutation  // [num_tokens,] logical -> physical (optional)
) {
  CHECK_INPUT(x);
  CHECK_INPUT(weights);
  CHECK_INPUT(seg_indptr);
  CHECK_INPUT(weight_indices);
  CHECK_INPUT(lora_ranks);
  CHECK_INPUT(output);

  TORCH_CHECK(x.dim() == 2, "x must be a 2D tensor");
  TORCH_CHECK(weights.dim() == 3, "weights must be a 3D tensor");
  TORCH_CHECK(seg_indptr.dim() == 1, "seg_indptr must be a 1D tensor");
  TORCH_CHECK(weight_indices.dim() == 1, "weight_indices must be a 1D tensor");
  TORCH_CHECK(lora_ranks.dim() == 1, "lora_ranks must be a 1D tensor");
  TORCH_CHECK(output.dim() == 2, "output must be a 2D tensor");

  TORCH_CHECK(num_slices > 0, "num_slices must be > 0");
  TORCH_CHECK(weights.size(1) % num_slices == 0, "weights.size(1) must be divisible by num_slices");
  TORCH_CHECK(
      weights.scalar_type() == torch::kHalf || weights.scalar_type() == torch::kBFloat16,
      "Unsupported data type for chunked_sgmv_lora_shrink_fwd weights: ",
      weights.scalar_type());
  TORCH_CHECK(weights.scalar_type() == x.scalar_type(), "x dtype must match weights dtype");
  TORCH_CHECK(weights.size(2) == x.size(1), "weights.size(2) must equal x.size(1)");

  const int64_t num_loras_i64 = weights.size(0);
  const int64_t max_rank_i64 = weights.size(1) / num_slices;
  const int64_t total_n_i64 = weights.size(1);  // num_slices * max_rank
  const int64_t num_tokens_i64 = x.size(0);

  TORCH_CHECK(num_loras_i64 > 0, "weights.size(0) must be greater than 0");
  TORCH_CHECK(lora_ranks.numel() == num_loras_i64, "lora_ranks.numel() must equal weights.size(0)");

  TORCH_CHECK(
      num_tokens_i64 == 0 || seg_indptr.numel() >= 2, "seg_indptr must have at least 2 elements when num_tokens > 0");
  const int64_t num_segments_i64 = seg_indptr.numel() - 1;
  TORCH_CHECK(weight_indices.numel() == num_segments_i64, "weight_indices.numel() must equal seg_indptr.numel() - 1");

  // Validate the caller-allocated output tensor (physical token order).
  TORCH_CHECK(
      output.size(0) == num_tokens_i64 && output.size(1) == total_n_i64,
      "output must have shape (num_tokens, num_slices * max_rank)");
  TORCH_CHECK(output.scalar_type() == weights.scalar_type(), "output dtype must match weights dtype");

  if (num_tokens_i64 == 0) {
    return;
  }
  // K == 0 (input_dim == 0) is a degenerate GEMM: every output element is an
  // empty sum, so the result is the (num_tokens, N) zero matrix.
  if (x.size(1) == 0) {
    output.zero_();
    return;
  }

  // Value-level checks (each a device -> host sync) run in debug-check builds only.
  lora_debug_checks::check_weight_indices(weight_indices, num_loras_i64, "weights");
  lora_debug_checks::check_seg_indptr(seg_indptr, num_tokens_i64);
  lora_debug_checks::check_lora_ranks(lora_ranks, max_rank_i64);
  // lora_ranks does NOT shrink the per-segment GEMM (every segment computes the
  // full N = num_slices * max_rank columns). The caller must pre-zero weight rows
  // beyond each adapter's rank R_l so that the extra columns come out zero-padded.

  // Cast index tensors to int32 for the kernel's device-side tile scheduler.
  auto seg_indptr_i32 = seg_indptr.scalar_type() == torch::kInt32 ? seg_indptr : seg_indptr.to(torch::kInt32);
  auto weight_indices_i32 =
      weight_indices.scalar_type() == torch::kInt32 ? weight_indices : weight_indices.to(torch::kInt32);

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  const int num_segments_ = static_cast<int>(num_segments_i64);

  const int64_t* perm_ptr = nullptr;
  torch::Tensor perm_i64;
  if (permutation.has_value()) {
    const auto& perm = permutation.value();
    CHECK_INPUT(perm);
    TORCH_CHECK(perm.dim() == 1, "permutation must be a 1D tensor");
    TORCH_CHECK(perm.numel() == num_tokens_i64, "permutation.numel() must equal x.size(0)");
    lora_debug_checks::check_permutation(perm, num_tokens_i64);
    perm_i64 = perm.scalar_type() == torch::kInt64 ? perm : perm.to(torch::kInt64);
    perm_ptr = perm_i64.data_ptr<int64_t>();
  }

  // One gather-fused DPAS kernel: rows are read through the permutation and
  // outputs scattered through it, so no sorted copies exist.
  DISPATCH_CHUNKED_SGMV_LORA_SHRINK_FWD_DTYPE(
      x, weights, output, seg_indptr_i32, weight_indices_i32, perm_ptr, num_segments_, queue);
}

#undef DISPATCH_CHUNKED_SGMV_LORA_SHRINK_FWD_TILE
#undef DISPATCH_CHUNKED_SGMV_LORA_SHRINK_FWD_DTYPE
#undef SYCL_INTEL_TARGET
