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
    decode phase, where the batch interleaves adapters ("zigzag"), so tokens are
    reordered to group rows by adapter (gather) before the GEMM and reordered
    back (scatter) afterwards -- the three-kernel "Option A" fused into one op:

      1. gather : x_sorted[i]       = x[permutation[i]]        (physical -> logical)
      2. GEMM   : out_sorted        = x_sorted @ weights^T     (small-N grouped GEMM)
      3. scatter: output[perm[i]]   = out_sorted[i]            (logical -> physical)

    seg_indptr / weight_indices describe the *logical* (adapter-grouped) layout;
    permutation (logical -> physical, a bijection over the token rows) maps that
    to the physical token order of x / output. When permutation is absent the
    batch is already contiguous by adapter (prefill), so gather/scatter are
    skipped and the GEMM runs in place.

    The GEMM uses a dedicated small-N tile (chunked_sgmv_lora_shrink_types.hpp)
    suited to the skinny shrink output; the merged sgemm_lora_a_fwd kernel is
    left untouched.
*/

#define SYCL_INTEL_TARGET 20

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <optional>
#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"
#include "kernels/lora/device/chunked_sgmv_lora_shrink_dispatch.hpp"
#include "kernels/lora/device/lora_permute_rows.hpp"
#include "sgl_kernel_export.h"

namespace {

//----------------- Per-(dtype, tile) dispatch macros --------------------//
// Tile selection currently has a single small-N option. Add tiles to
// DISPATCH_CHUNKED_SGMV_LORA_SHRINK_TILE (and to ChunkedSgmvLoraShrinkXe20.cmake
// + chunked_sgmv_lora_shrink_dispatch.hpp + chunked_sgmv_lora_shrink_types.hpp)
// with a runtime heuristic (e.g. average M per segment) picking the tag.
#define DISPATCH_CHUNKED_SGMV_LORA_SHRINK_TILE(ELEM, ...)                                 \
  do {                                                                                    \
    chunked_sgmv_lora_shrink_impl::launch_chunked_sgmv_lora_shrink_##ELEM##_small(__VA_ARGS__); \
  } while (0)

#define DISPATCH_CHUNKED_SGMV_LORA_SHRINK_DTYPE(...)                                                              \
  do {                                                                                                           \
    switch (weights.scalar_type()) {                                                                             \
      case torch::kHalf:                                                                                         \
        DISPATCH_CHUNKED_SGMV_LORA_SHRINK_TILE(half, __VA_ARGS__);                                               \
        break;                                                                                                   \
      case torch::kBFloat16:                                                                                     \
        DISPATCH_CHUNKED_SGMV_LORA_SHRINK_TILE(bf16, __VA_ARGS__);                                               \
        break;                                                                                                   \
      default:                                                                                                   \
        TORCH_CHECK(false, "Unsupported data type for chunked_sgmv_lora_shrink_forward weights: ",               \
                    weights.scalar_type());                                                                      \
    }                                                                                                            \
  } while (0)

}  // namespace

//----------------- Main API function --------------------//

SGL_KERNEL_EXPORT torch::Tensor chunked_sgmv_lora_shrink_forward(
    const torch::Tensor& x,               // [num_tokens, input_dim]  (physical token order)
    const torch::Tensor& weights,         // [num_loras, num_slices*max_rank, input_dim]
    const int64_t num_slices,             // stacked projections (qkv=3, gate_up=2, else 1)
    const int64_t num_segments,           // number of segments (== batch size)
    const torch::Tensor& seg_indptr,      // [num_segments + 1,]   over the logical row order
    const torch::Tensor& weight_indices,  // [num_segments,]
    const torch::Tensor& lora_ranks,      // [num_loras,]
    const std::optional<torch::Tensor>& permutation  // [num_tokens,] logical -> physical (optional)
) {
  CHECK_INPUT(x);
  CHECK_INPUT(weights);
  CHECK_INPUT(seg_indptr);
  CHECK_INPUT(weight_indices);
  CHECK_INPUT(lora_ranks);

  TORCH_CHECK(x.dim() == 2, "x must be a 2D tensor");
  TORCH_CHECK(weights.dim() == 3, "weights must be a 3D tensor");
  TORCH_CHECK(seg_indptr.dim() == 1, "seg_indptr must be a 1D tensor");
  TORCH_CHECK(weight_indices.dim() == 1, "weight_indices must be a 1D tensor");
  TORCH_CHECK(lora_ranks.dim() == 1, "lora_ranks must be a 1D tensor");

  TORCH_CHECK(num_slices > 0, "num_slices must be > 0");
  TORCH_CHECK(weights.size(1) % num_slices == 0, "weights.size(1) must be divisible by num_slices");
  TORCH_CHECK(weights.scalar_type() == x.scalar_type(), "x dtype must match weights dtype");

  const int64_t num_loras_i64 = weights.size(0);
  const int64_t max_rank_i64 = weights.size(1) / num_slices;
  const int64_t total_n_i64 = weights.size(1);  // num_slices * max_rank
  const int64_t num_tokens_i64 = x.size(0);

  TORCH_CHECK(num_loras_i64 > 0, "weights.size(0) must be greater than 0");
  TORCH_CHECK(lora_ranks.numel() == num_loras_i64, "lora_ranks.numel() must equal weights.size(0)");
  TORCH_CHECK(num_segments >= 0, "num_segments must be non-negative");
  TORCH_CHECK(
      weight_indices.numel() == num_segments, "weight_indices.numel() must equal num_segments");
  TORCH_CHECK(
      seg_indptr.numel() == num_segments + 1, "seg_indptr.numel() must equal num_segments + 1");

  // Allocate the output in physical token order.
  auto out_opts = torch::TensorOptions().dtype(weights.dtype()).device(weights.device());
  auto output = torch::empty({num_tokens_i64, total_n_i64}, out_opts);

  if (num_tokens_i64 == 0 || num_segments == 0) {
    return output;
  }
  // K == 0 (input_dim == 0) is a degenerate GEMM: every output element is an
  // empty sum, so the result is the (num_tokens, N) zero matrix.
  if (x.size(1) == 0) {
    output.zero_();
    return output;
  }

  if (num_segments > 0) {
    auto [min_wi, max_wi] = torch::aminmax(weight_indices);
    TORCH_CHECK(
        min_wi.item<int64_t>() >= 0 && max_wi.item<int64_t>() < num_loras_i64,
        "weight_indices values must be in [0, weights.size(0))");
  }
  TORCH_CHECK(seg_indptr[0].item<int64_t>() == 0, "seg_indptr[0] must be 0");
  TORCH_CHECK(
      seg_indptr[seg_indptr.numel() - 1].item<int64_t>() == num_tokens_i64, "seg_indptr[-1] must equal num_tokens");
  auto seg_len_tensor = seg_indptr.slice(0, 1) - seg_indptr.slice(0, 0, seg_indptr.size(0) - 1);
  auto [seg_len_min, seg_len_max] = torch::aminmax(seg_len_tensor);
  TORCH_CHECK(seg_len_min.item<int>() >= 0, "seg_indptr must be non-decreasing");
  (void)seg_len_max;

  // lora_ranks is only range-validated here; it does NOT shrink the per-segment
  // GEMM (every segment computes the full N = num_slices * max_rank columns).
  // The caller must pre-zero weight rows beyond each adapter's rank R_l so that
  // the extra columns come out zero-padded. See build_grouped_gemm_meta().
  auto [min_lr, max_lr] = torch::aminmax(lora_ranks);
  TORCH_CHECK(
      min_lr.item<int64_t>() >= 0 && max_lr.item<int>() <= max_rank_i64,
      "All values in lora_ranks must be within the range [0, max_rank]");

  // Cast index tensors to int32 for the device-side metadata build.
  auto seg_indptr_i32 = seg_indptr.scalar_type() == torch::kInt32 ? seg_indptr : seg_indptr.to(torch::kInt32);
  auto weight_indices_i32 =
      weight_indices.scalar_type() == torch::kInt32 ? weight_indices : weight_indices.to(torch::kInt32);

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  const int max_rank = static_cast<int>(max_rank_i64);
  const int num_segments_ = static_cast<int>(num_segments);
  const int num_slices_ = static_cast<int>(num_slices);

  if (!permutation.has_value()) {
    // Prefill fast path: rows already contiguous by adapter, GEMM in place.
    DISPATCH_CHUNKED_SGMV_LORA_SHRINK_DTYPE(
        x, weights, seg_indptr_i32, weight_indices_i32, output, num_slices_, max_rank, num_segments_, queue);
    return output;
  }

  // Decode path: gather physical -> logical, GEMM, scatter logical -> physical.
  const auto& perm = permutation.value();
  CHECK_INPUT(perm);
  TORCH_CHECK(perm.dim() == 1, "permutation must be a 1D tensor");
  TORCH_CHECK(perm.numel() == num_tokens_i64, "permutation.numel() must equal x.size(0)");
  {
    auto [min_p, max_p] = torch::aminmax(perm);
    TORCH_CHECK(
        min_p.item<int64_t>() >= 0 && max_p.item<int64_t>() < num_tokens_i64,
        "permutation values must be in [0, x.size(0))");
  }
  auto perm_i64 = perm.scalar_type() == torch::kInt64 ? perm : perm.to(torch::kInt64);

  auto x_sorted = torch::empty({num_tokens_i64, x.size(1)}, x.options());
  lora_permute_rows_impl::permute_rows_dispatch</*GATHER=*/true>(x, x_sorted, perm_i64, queue);

  auto out_sorted = torch::empty({num_tokens_i64, total_n_i64}, out_opts);
  DISPATCH_CHUNKED_SGMV_LORA_SHRINK_DTYPE(
      x_sorted, weights, seg_indptr_i32, weight_indices_i32, out_sorted, num_slices_, max_rank, num_segments_, queue);

  lora_permute_rows_impl::permute_rows_dispatch</*GATHER=*/false>(out_sorted, output, perm_i64, queue);
  return output;
}

#undef DISPATCH_CHUNKED_SGMV_LORA_SHRINK_TILE
#undef DISPATCH_CHUNKED_SGMV_LORA_SHRINK_DTYPE
#undef SYCL_INTEL_TARGET
