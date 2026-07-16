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
    \brief SGEMM LoRA A Forward Kernel
*/

#define SYCL_INTEL_TARGET 20

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"
#include "kernels/lora/device/sgemm_lora_a_fwd_dispatch.hpp"

namespace {

//----------------- Per-(dtype, tile) dispatch macros --------------------//
// Tile selection currently has a single option (large). Add tiles to
// DISPATCH_SGEMM_LORA_A_FWD_TILE (and to SGEMMLoraAFwdXe20.cmake +
// sgemm_lora_a_fwd_dispatch.hpp + sgemm_lora_a_fwd_types.hpp) with a runtime
// heuristic (e.g. average M per segment) picking the tag.
#define DISPATCH_SGEMM_LORA_A_FWD_TILE(ELEM, ...)                              \
  do {                                                                         \
    sgemm_lora_a_fwd_impl::launch_sgemm_lora_a_fwd_##ELEM##_large(__VA_ARGS__); \
  } while (0)

#define DISPATCH_SGEMM_LORA_A_FWD_DTYPE(...)                                                       \
  do {                                                                                             \
    switch (weights.scalar_type()) {                                                               \
      case torch::kHalf:                                                                           \
        DISPATCH_SGEMM_LORA_A_FWD_TILE(half, __VA_ARGS__);                                         \
        break;                                                                                     \
      case torch::kBFloat16:                                                                       \
        DISPATCH_SGEMM_LORA_A_FWD_TILE(bf16, __VA_ARGS__);                                         \
        break;                                                                                     \
      default:                                                                                     \
        TORCH_CHECK(false, "Unsupported data type for sgemm_lora_a_fwd weights: ", weights.scalar_type()); \
    }                                                                                              \
  } while (0)

}  // namespace

//----------------- Main API function --------------------//

void sgemm_lora_a_fwd(
    torch::Tensor& output,         // [num_tokens, max_rank]
    const torch::Tensor& input_x,  // [num_tokens, input_dim]
    const torch::Tensor& weights,  // [num_loras, stack_num*max_rank, input_dim]
    const int64_t stack_num,
    const torch::Tensor& seg_indptr,      // [num_segments + 1,]
    const torch::Tensor& weight_indices,  // [num_segments,]
    const torch::Tensor& lora_ranks,      // [num_loras,]
    const std::optional<torch::Tensor>&
        seg_lens  // [num_segments,] optional; currently unused, reserved for future per-segment optimizations
) {
  CHECK_INPUT(input_x);
  CHECK_INPUT(weights);
  CHECK_INPUT(seg_indptr);
  CHECK_INPUT(weight_indices);
  CHECK_INPUT(lora_ranks);
  CHECK_INPUT(output);

  TORCH_CHECK(input_x.dim() == 2, "input_x must be a 2D tensor");
  TORCH_CHECK(weights.dim() == 3, "weights must be a 3D tensor");
  TORCH_CHECK(seg_indptr.dim() == 1, "seg_indptr must be a 1D tensor");
  TORCH_CHECK(weight_indices.dim() == 1, "weight_indices must be a 1D tensor");
  TORCH_CHECK(lora_ranks.dim() == 1, "lora_ranks must be a 1D tensor");
  TORCH_CHECK(output.dim() == 2, "output must be a 2D tensor");

  TORCH_CHECK(stack_num > 0, "stack_num must be > 0");
  TORCH_CHECK(weights.size(1) % stack_num == 0, "weights.size(1) must be divisible by stack_num");

  const int64_t num_loras_i64 = weights.size(0);
  const int64_t max_rank_i64 = weights.size(1) / stack_num;
  const int64_t num_tokens_i64 = input_x.size(0);

  TORCH_CHECK(lora_ranks.numel() == num_loras_i64, "lora_ranks.numel() must equal weights.size(0)");
  TORCH_CHECK(num_loras_i64 > 0, "weights.size(0) and lora_ranks.numel() must be greater than 0");
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
  // Validate output tensor size and dtype
  TORCH_CHECK(
      output.size(0) == num_tokens_i64 && output.size(1) == max_rank_i64 * stack_num,
      "Output tensor must have shape (num_tokens, max_rank * stack_num)");
  TORCH_CHECK(output.scalar_type() == weights.scalar_type(), "Output tensor dtype must match weights dtype");
  TORCH_CHECK(weights.scalar_type() == input_x.scalar_type(), "Input tensor dtype must match weights dtype");
  if (num_tokens_i64 == 0) {
    return;
  }
  // K == 0 (input_dim == 0) is a degenerate GEMM: every output element is an
  // empty sum, so the result is the (num_tokens, N) zero matrix.
  if (input_x.size(1) == 0) {
    output.zero_();
    return;
  }

  TORCH_CHECK(seg_indptr[0].item<int64_t>() == 0, "seg_indptr[0] must be 0");
  TORCH_CHECK(
      seg_indptr[seg_indptr.numel() - 1].item<int64_t>() == num_tokens_i64, "seg_indptr[-1] must equal num_tokens");
  auto seg_len_tensor = seg_indptr.slice(0, 1) - seg_indptr.slice(0, 0, seg_indptr.size(0) - 1);
  auto [seg_len_min, seg_len_max] = torch::aminmax(seg_len_tensor);
  TORCH_CHECK(seg_len_min.item<int>() >= 0, "seg_indptr must be non-decreasing");
  (void)seg_len_max;  // not needed: grouped GEMM handles variable M per group

  // lora_ranks is only range-validated here; it does NOT shrink the per-segment
  // GEMM (every segment computes the full N = stack_num * max_rank columns). The
  // caller must pre-zero weight rows beyond each adapter's rank R_l so that the
  // extra columns come out zero-padded. See build_grouped_gemm_meta().
  auto [min_lr, max_lr] = torch::aminmax(lora_ranks);
  TORCH_CHECK(
      min_lr.item<int64_t>() >= 0 && max_lr.item<int>() <= max_rank_i64,
      "All values in lora_ranks must be within the range [0, max_rank]");

  // Cast index tensors to int32 for the host-side metadata read in the launcher.
  auto seg_indptr_i32 = seg_indptr.scalar_type() == torch::kInt32 ? seg_indptr : seg_indptr.to(torch::kInt32);
  auto weight_indices_i32 =
      weight_indices.scalar_type() == torch::kInt32 ? weight_indices : weight_indices.to(torch::kInt32);

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  const int max_rank = static_cast<int>(max_rank_i64);
  const int num_segments = static_cast<int>(num_segments_i64);
  const int stack_num_ = static_cast<int>(stack_num);

  // Dispatch on (dtype, tile). Each launch symbol is defined in a separate
  // generated translation unit (see SGEMMLoraAFwdXe20.cmake).
  DISPATCH_SGEMM_LORA_A_FWD_DTYPE(
      input_x, weights, seg_indptr_i32, weight_indices_i32, output, stack_num_, max_rank, num_segments, queue);
}

#undef DISPATCH_SGEMM_LORA_A_FWD_TILE
#undef DISPATCH_SGEMM_LORA_A_FWD_DTYPE
#undef SYCL_INTEL_TARGET
