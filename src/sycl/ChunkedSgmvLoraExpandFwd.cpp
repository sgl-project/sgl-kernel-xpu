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
    \brief Chunked-SGMV LoRA "expand" (B-matrix) forward entrypoint.

    Computes the scaled LoRA-B projection D = scalings[l] * (x @ weights[l]^T)
    (+ base_output) as a segmented, per-slice grouped GEMM, where each segment s
    uses adapter l = weight_indices[s]. It is the "expand" counterpart of the
    chunked-SGMV shrink: the decode batch interleaves adapters ("zigzag"), so
    tokens are reordered to group rows by adapter (gather) before the GEMM and
    reordered back (scatter) afterwards -- the three-kernel "Option A" fused into
    one op:

      1. gather : x_sorted[i]       = x[permutation[i]]        (physical -> logical)
                  (base_sorted[i]   = base_output[permutation[i]] when supplied)
      2. GEMM   : out_sorted        = scalings * (x_sorted @ weights^T) + base_sorted
      3. scatter: output[perm[i]]   = out_sorted[i]            (logical -> physical)

    The per-slice partition (qkv=3, gate_up=2, else 1) is generic: slice_offsets
    gives the output-column boundaries of each stacked projection, and x packs the
    per-slice rank bands ([num_tokens, num_slices*max_rank]). This is the generic
    n_slices sibling of the fused QKV / gate-up LoRA-B kernels, wrapped in the
    chunked-SGMV gather/scatter.

    seg_indptr / weight_indices describe the *logical* (adapter-grouped) layout;
    permutation (logical -> physical, a bijection over the token rows) maps that
    to the physical token order of x / output. When permutation is absent the
    batch is already contiguous by adapter (prefill), so gather/scatter are
    skipped and the GEMM runs in place and seg_indptr / weight_indices are
    assumed to be in *physical* layout.
*/

#define SYCL_INTEL_TARGET 20

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <algorithm>
#include <optional>
#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"
#include "kernels/lora/device/chunked_sgmv_lora_expand_fwd_dispatch.hpp"
#include "kernels/lora/device/lora_permute_rows.hpp"
#include "sgl_kernel_export.h"

namespace {

//----------------- Per-(dtype, tile) dispatch macros --------------------//
// LoRA-B "expand" is a K-thin, memory-bandwidth-bound grouped GEMM. Tile
// selection currently has a single option (tall).
// Add tiles to DISPATCH_CHUNKED_SGMV_LORA_EXPAND_FWD_TILE (and to
// ChunkedSgmvLoraExpandFwdXe20.cmake + chunked_sgmv_lora_expand_fwd_dispatch.hpp
// + chunked_sgmv_lora_expand_fwd_types.hpp) with a runtime heuristic picking the tag.
#define DISPATCH_CHUNKED_SGMV_LORA_EXPAND_FWD_TILE(ELEM, ...)                                          \
  do {                                                                                                 \
    chunked_sgmv_lora_expand_fwd_impl::launch_chunked_sgmv_lora_expand_fwd_##ELEM##_tall(__VA_ARGS__); \
  } while (0)

#define DISPATCH_CHUNKED_SGMV_LORA_EXPAND_FWD_DTYPE(...)                                                               \
  do {                                                                                                                 \
    switch (weights.scalar_type()) {                                                                                   \
      case torch::kHalf:                                                                                               \
        DISPATCH_CHUNKED_SGMV_LORA_EXPAND_FWD_TILE(half, __VA_ARGS__);                                                 \
        break;                                                                                                         \
      case torch::kBFloat16:                                                                                           \
        DISPATCH_CHUNKED_SGMV_LORA_EXPAND_FWD_TILE(bf16, __VA_ARGS__);                                                 \
        break;                                                                                                         \
      default:                                                                                                         \
        TORCH_CHECK(false, "Unsupported data type for chunked_sgmv_lora_expand_fwd weights: ", weights.scalar_type()); \
    }                                                                                                                  \
  } while (0)

}  // namespace

//----------------- Main API function --------------------//

SGL_KERNEL_EXPORT void chunked_sgmv_lora_expand_fwd(
    torch::Tensor& output,                            // [num_tokens, N_total]  (physical token order)
    const torch::Tensor& x,                           // [num_tokens, num_slices*max_rank]  (physical token order)
    const torch::Tensor& weights,                     // [num_loras, N_total, max_rank]
    const torch::Tensor& slice_offsets,               // [num_slices + 1,]  output-column boundaries
    const int64_t max_slice_size,                     // max per-slice output width implied by slice_offsets
    const int64_t num_slices,                         // stacked projections (qkv=3, gate_up=2, else 1)
    const int64_t num_segments,                       // currently the code does not rely on it.
    const torch::Tensor& seg_indptr,                  // [num_segments + 1,]   over the logical row order
    const torch::Tensor& weight_indices,              // [num_segments,]
    const torch::Tensor& lora_ranks,                  // [num_loras,]
    const torch::Tensor& scalings,                    // [num_loras,]
    const std::optional<torch::Tensor>& permutation,  // [num_tokens,] logical -> physical (optional)
    const std::optional<torch::Tensor>&
        base_output  // [num_tokens, N_total] optional; the base model's output for a fused add
) {
  CHECK_INPUT(x);
  CHECK_INPUT(weights);
  CHECK_INPUT(slice_offsets);
  CHECK_INPUT(seg_indptr);
  CHECK_INPUT(weight_indices);
  CHECK_INPUT(lora_ranks);
  CHECK_INPUT(scalings);
  CHECK_INPUT(output);

  TORCH_CHECK(x.dim() == 2, "x must be a 2D tensor");
  TORCH_CHECK(weights.dim() == 3, "weights must be a 3D tensor");
  TORCH_CHECK(slice_offsets.dim() == 1, "slice_offsets must be a 1D tensor");
  TORCH_CHECK(seg_indptr.dim() == 1, "seg_indptr must be a 1D tensor");
  TORCH_CHECK(weight_indices.dim() == 1, "weight_indices must be a 1D tensor");
  TORCH_CHECK(lora_ranks.dim() == 1, "lora_ranks must be a 1D tensor");
  TORCH_CHECK(scalings.dim() == 1, "scalings must be a 1D tensor");
  TORCH_CHECK(output.dim() == 2, "output must be a 2D tensor");

  TORCH_CHECK(num_slices > 0, "num_slices must be > 0");
  TORCH_CHECK(slice_offsets.numel() == num_slices + 1, "slice_offsets must have num_slices + 1 elements");
  TORCH_CHECK(weights.scalar_type() == x.scalar_type(), "x dtype must match weights dtype");

  const int64_t num_loras_i64 = weights.size(0);
  const int64_t n_total_i64 = weights.size(1);  // N_total (sum of per-slice output dims)
  const int64_t max_rank_i64 = weights.size(2);
  const int64_t num_tokens_i64 = x.size(0);

  TORCH_CHECK(x.size(1) == num_slices * max_rank_i64, "x.size(1) must equal num_slices * max_rank");
  TORCH_CHECK(num_loras_i64 > 0, "weights.size(0) must be greater than 0");
  TORCH_CHECK(lora_ranks.numel() == num_loras_i64, "lora_ranks.numel() must equal weights.size(0)");
  TORCH_CHECK(scalings.numel() == num_loras_i64, "scalings.numel() must equal weights.size(0)");

  TORCH_CHECK(
      num_tokens_i64 == 0 || seg_indptr.numel() >= 2, "seg_indptr must have at least 2 elements when num_tokens > 0");
  const int64_t num_segments_i64 = seg_indptr.numel() - 1;
  TORCH_CHECK(weight_indices.numel() == num_segments_i64, "weight_indices.numel() must equal seg_indptr.numel() - 1");
  if (num_segments_i64 > 0) {
    auto [min_wi, max_wi] = torch::aminmax(weight_indices);
    TORCH_CHECK(
        min_wi.item<int64_t>() >= 0 && max_wi.item<int64_t>() < num_loras_i64,
        "weight_indices values must be in [0, weights.size(0))");
  }

  // Validate the caller-allocated output tensor (physical token order).
  TORCH_CHECK(
      output.size(0) == num_tokens_i64 && output.size(1) == n_total_i64,
      "output must have shape (num_tokens, N_total)");
  TORCH_CHECK(output.scalar_type() == weights.scalar_type(), "output dtype must match weights dtype");
  if (base_output.has_value()) {
    CHECK_INPUT(base_output.value());
    TORCH_CHECK(base_output->dim() == 2, "base_output must be a 2D tensor");
    TORCH_CHECK(
        base_output->size(0) == num_tokens_i64 && base_output->size(1) == n_total_i64,
        "base_output must have shape (num_tokens, N_total)");
    TORCH_CHECK(base_output->scalar_type() == weights.scalar_type(), "base_output dtype must match weights dtype");
  }

  // slice_offsets defines the per-slice output-column bands: [0, ..., N_total].
  // Validate the boundaries frame the full output and are non-decreasing, and
  // that max_slice_size matches the widest slice.
  auto slice_offsets_i32 =
      slice_offsets.scalar_type() == torch::kInt32 ? slice_offsets : slice_offsets.to(torch::kInt32);
  auto so_cpu = slice_offsets_i32.cpu();
  const int32_t* so = so_cpu.data_ptr<int32_t>();
  TORCH_CHECK(so[0] == 0, "slice_offsets[0] must be 0");
  TORCH_CHECK(so[num_slices] == n_total_i64, "slice_offsets[-1] must equal weights.size(1) (N_total)");
  int64_t widest = 0;
  for (int64_t p = 0; p < num_slices; ++p) {
    TORCH_CHECK(so[p] <= so[p + 1], "slice_offsets must be non-decreasing");
    widest = std::max<int64_t>(widest, so[p + 1] - so[p]);
  }
  TORCH_CHECK(max_slice_size == widest, "max_slice_size must equal the widest slice implied by slice_offsets");

  if (num_tokens_i64 == 0) {
    return;
  }
  // K == 0 (max_rank == 0) is a degenerate GEMM: the scaled LoRA term is an empty
  // sum (zero), so the output reduces to the residual when one is supplied, or the
  // zero matrix otherwise -- mirroring the SGEMM / QKV LoRA-B convention.
  if (max_rank_i64 == 0) {
    if (base_output.has_value()) {
      output.copy_(base_output.value());
    } else {
      output.zero_();
    }
    return;
  }

  TORCH_CHECK(seg_indptr[0].item<int64_t>() == 0, "seg_indptr[0] must be 0");
  TORCH_CHECK(
      seg_indptr[seg_indptr.numel() - 1].item<int64_t>() == num_tokens_i64, "seg_indptr[-1] must equal num_tokens");
  auto seg_len_tensor = seg_indptr.slice(0, 1) - seg_indptr.slice(0, 0, seg_indptr.size(0) - 1);
  auto [seg_len_min, seg_len_max] = torch::aminmax(seg_len_tensor);
  TORCH_CHECK(seg_len_min.item<int>() >= 0, "seg_indptr must be non-decreasing");
  (void)seg_len_max;

  // lora_ranks is only range-validated here; it does NOT shrink the per-segment
  // GEMM (every group computes the full K = max_rank reduction). The caller must
  // pre-zero weight columns beyond each adapter's rank R_l.
  auto [min_lr, max_lr] = torch::aminmax(lora_ranks);
  TORCH_CHECK(
      min_lr.item<int64_t>() >= 0 && max_lr.item<int>() <= max_rank_i64,
      "All values in lora_ranks must be within the range [0, max_rank]");

  // Cast index tensors to int32 for the device-side metadata build; the per-group
  // alpha buffer is derived from fp32 scalings.
  auto seg_indptr_i32 = seg_indptr.scalar_type() == torch::kInt32 ? seg_indptr : seg_indptr.to(torch::kInt32);
  auto weight_indices_i32 =
      weight_indices.scalar_type() == torch::kInt32 ? weight_indices : weight_indices.to(torch::kInt32);
  auto scalings_f32 = scalings.scalar_type() == torch::kFloat32 ? scalings : scalings.to(torch::kFloat32);

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  const int n_total = static_cast<int>(n_total_i64);
  const int num_slices_ = static_cast<int>(num_slices);
  const int num_segments_ = static_cast<int>(num_segments_i64);
  (void)num_segments;  // derived from seg_indptr; the explicit arg is informational

  if (!permutation.has_value()) {
    // Prefill fast path: rows already contiguous by adapter, GEMM in place with
    // the residual fused via the epilogue (beta = base_output.has_value()).
    DISPATCH_CHUNKED_SGMV_LORA_EXPAND_FWD_DTYPE(
        x,
        weights,
        slice_offsets_i32,
        seg_indptr_i32,
        weight_indices_i32,
        scalings_f32,
        output,
        base_output,
        n_total,
        num_slices_,
        num_segments_,
        queue);
    return;
  }

  // Decode path: gather physical -> logical, GEMM (residual fused in logical
  // order), scatter logical -> physical.
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

  // Fuse the residual through the epilogue: gather base_output into logical order
  // so it shares the GEMM's row layout, then scatter the combined result back.
  std::optional<torch::Tensor> base_sorted;
  if (base_output.has_value()) {
    base_sorted = torch::empty({num_tokens_i64, n_total_i64}, base_output->options());
    lora_permute_rows_impl::permute_rows_dispatch</*GATHER=*/true>(*base_output, *base_sorted, perm_i64, queue);
  }

  auto out_sorted = torch::empty({num_tokens_i64, n_total_i64}, output.options());
  DISPATCH_CHUNKED_SGMV_LORA_EXPAND_FWD_DTYPE(
      x_sorted,
      weights,
      slice_offsets_i32,
      seg_indptr_i32,
      weight_indices_i32,
      scalings_f32,
      out_sorted,
      base_sorted,
      n_total,
      num_slices_,
      num_segments_,
      queue);

  lora_permute_rows_impl::permute_rows_dispatch</*GATHER=*/false>(out_sorted, output, perm_i64, queue);
}

#undef DISPATCH_CHUNKED_SGMV_LORA_EXPAND_FWD_TILE
#undef DISPATCH_CHUNKED_SGMV_LORA_EXPAND_FWD_DTYPE
#undef SYCL_INTEL_TARGET
