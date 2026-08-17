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
    \brief QKV LoRA B Forward Kernel

    Fused LoRA-B projection for the packed q/k/v outputs. This kernel packs three
    SGEMMs (q / k / v) into a single pointer-array grouped GEMM: for each input
    segment it emits three groups, one per projection, accumulating each product
    into its column band of the output. When an adapter's rank is 0 the group is a
    genuine all-zero LoRA term (empty reduction), following the PyTorch convention
    that (m, 0) @ (0, n) is the zero matrix of shape (m, n).
*/

#define SYCL_INTEL_TARGET 20

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <algorithm>
#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"
#include "kernels/lora/device/qkv_lora_b_fwd_dispatch.hpp"
#include "sgl_kernel_export.h"

namespace {

//----------------- Per-(dtype, tile) dispatch macros --------------------//
// QKV LoRA-B is a K-thin (rank 16..64), memory-bandwidth-bound grouped GEMM.
// After trying different shapes, a single tall/thin 32x512 workgroup tile is
// fastest across the shape space, so tile selection currently has a single option (tall).
// Add tiles to DISPATCH_QKV_LORA_B_FWD_TILE (and to QKVLoraBFwdXe20.cmake +
// qkv_lora_b_fwd_dispatch.hpp + qkv_lora_b_fwd_types.hpp) with a runtime
// heuristic picking the tag.
#define DISPATCH_QKV_LORA_B_FWD_TILE(ELEM, ...)                            \
  do {                                                                     \
    qkv_lora_b_fwd_impl::launch_qkv_lora_b_fwd_##ELEM##_tall(__VA_ARGS__); \
  } while (0)
#define DISPATCH_QKV_LORA_B_FWD_DTYPE(...)                                                                     \
  do {                                                                                                         \
    switch (qkv_lora_b.scalar_type()) {                                                                        \
      case torch::kHalf:                                                                                       \
        DISPATCH_QKV_LORA_B_FWD_TILE(half, __VA_ARGS__);                                                       \
        break;                                                                                                 \
      case torch::kBFloat16:                                                                                   \
        DISPATCH_QKV_LORA_B_FWD_TILE(bf16, __VA_ARGS__);                                                       \
        break;                                                                                                 \
      default:                                                                                                 \
        TORCH_CHECK(false, "Unsupported data type for qkv_lora_b_fwd qkv_lora_b: ", qkv_lora_b.scalar_type()); \
    }                                                                                                          \
  } while (0)

}  // namespace

//----------------- Main API function --------------------//

SGL_KERNEL_EXPORT void qkv_lora_b_fwd(
    torch::Tensor& output,                // [num_tokens, N_Q + 2N_{KV}]
    const torch::Tensor& input_x,         // [num_tokens, 3*K] (K = max_rank)
    const torch::Tensor& qkv_lora_b,      // [num_loras, N_Q + 2N_{KV}, K]
    const torch::Tensor& output_offset,   // [4,]
    const int64_t max_qkv_out_dim,        // max(output_q_dim, output_kv_dim)
    const torch::Tensor& seg_indptr,      // [num_segments + 1,]
    const torch::Tensor& weight_indices,  // [num_segments,]
    const torch::Tensor& lora_ranks,      // [num_loras,]
    const torch::Tensor& scalings,        // [num_loras,]
    const std::optional<torch::Tensor>&
        seg_lens,  // [num_segments,] optional; currently unused, reserved for future per-segment optimizations
    const std::optional<torch::Tensor>&
        base_output  // [num_tokens, N_Q + 2N_{KV}] optional; this can be the base model's output for a fused add
) {
  CHECK_INPUT(input_x);
  CHECK_INPUT(qkv_lora_b);
  CHECK_INPUT(output_offset);
  CHECK_INPUT(seg_indptr);
  CHECK_INPUT(weight_indices);
  CHECK_INPUT(lora_ranks);
  CHECK_INPUT(scalings);
  CHECK_INPUT(output);

  const auto dev = input_x.device();
  TORCH_CHECK(qkv_lora_b.device() == dev, "qkv_lora_b must be on the same device as input_x");
  TORCH_CHECK(output_offset.device() == dev, "output_offset must be on the same device as input_x");
  TORCH_CHECK(seg_indptr.device() == dev, "seg_indptr must be on the same device as input_x");
  TORCH_CHECK(weight_indices.device() == dev, "weight_indices must be on the same device as input_x");
  TORCH_CHECK(lora_ranks.device() == dev, "lora_ranks must be on the same device as input_x");
  TORCH_CHECK(scalings.device() == dev, "scalings must be on the same device as input_x");
  TORCH_CHECK(output.device() == dev, "output must be on the same device as input_x");
  if (seg_lens.has_value()) {
    TORCH_CHECK(seg_lens->device() == dev, "seg_lens must be on the same device as input_x");
  }
  if (base_output.has_value()) {
    TORCH_CHECK(base_output->device() == dev, "base_output must be on the same device as input_x");
  }

  TORCH_CHECK(input_x.dim() == 2, "input_x must be a 2D tensor");
  TORCH_CHECK(qkv_lora_b.dim() == 3, "qkv_lora_b must be a 3D tensor");
  TORCH_CHECK(output_offset.dim() == 1, "output_offset must be a 1D tensor");
  TORCH_CHECK(seg_indptr.dim() == 1, "seg_indptr must be a 1D tensor");
  TORCH_CHECK(weight_indices.dim() == 1, "weight_indices must be a 1D tensor");
  TORCH_CHECK(lora_ranks.dim() == 1, "lora_ranks must be a 1D tensor");
  TORCH_CHECK(scalings.dim() == 1, "scalings must be a 1D tensor");
  TORCH_CHECK(output.dim() == 2, "output must be a 2D tensor");

  TORCH_CHECK(output_offset.numel() == 4, "output_offset must have 4 elements (q/k/v output-column boundaries)");

  const int64_t num_loras_i64 = qkv_lora_b.size(0);
  const int64_t n_total_i64 = qkv_lora_b.size(1);  // N_Q + 2 * N_KV
  const int64_t max_rank_i64 = qkv_lora_b.size(2);
  const int64_t num_tokens_i64 = input_x.size(0);

  TORCH_CHECK(input_x.size(1) == 3 * max_rank_i64, "input_x.size(1) must equal 3 * max_rank");
  TORCH_CHECK(lora_ranks.numel() == num_loras_i64, "lora_ranks.numel() must equal qkv_lora_b.size(0)");
  TORCH_CHECK(scalings.numel() == num_loras_i64, "scalings.numel() must equal qkv_lora_b.size(0)");
  TORCH_CHECK(num_loras_i64 > 0, "qkv_lora_b.size(0) and lora_ranks.numel() must be greater than 0");
  TORCH_CHECK(
      num_tokens_i64 == 0 || seg_indptr.numel() >= 2, "seg_indptr must have at least 2 elements when num_tokens > 0");
  const int64_t num_segments_i64 = seg_indptr.numel() - 1;
  TORCH_CHECK(weight_indices.numel() == num_segments_i64, "weight_indices.numel() must equal seg_indptr.numel() - 1");
  if (num_segments_i64 > 0) {
    auto [min_wi, max_wi] = torch::aminmax(weight_indices);
    TORCH_CHECK(
        min_wi.item<int64_t>() >= 0 && max_wi.item<int64_t>() < num_loras_i64,
        "weight_indices values must be in [0, qkv_lora_b.size(0))");
  }
  // Validate output tensor size and dtype.
  TORCH_CHECK(
      output.size(0) == num_tokens_i64 && output.size(1) == n_total_i64,
      "Output tensor must have shape (num_tokens, N_Q + 2 * N_KV)");
  TORCH_CHECK(output.scalar_type() == qkv_lora_b.scalar_type(), "Output tensor dtype must match qkv_lora_b dtype");
  TORCH_CHECK(qkv_lora_b.scalar_type() == input_x.scalar_type(), "Input tensor dtype must match qkv_lora_b dtype");
  if (base_output.has_value()) {
    CHECK_INPUT(base_output.value());
    TORCH_CHECK(base_output->dim() == 2, "base_output must be a 2D tensor");
    TORCH_CHECK(
        base_output->size(0) == num_tokens_i64 && base_output->size(1) == n_total_i64,
        "base_output must have shape (num_tokens, N_Q + 2 * N_KV)");
    TORCH_CHECK(
        base_output->scalar_type() == qkv_lora_b.scalar_type(), "base_output dtype must match qkv_lora_b dtype");
  }

  // output_offset defines the q/k/v output-column bands: [0, N_Q, N_Q + N_KV,
  // N_Q + 2 * N_KV]. Validate the boundaries frame the full output and are
  // non-decreasing, and that max_qkv_out_dim matches the widest projection.
  auto output_offset_i32 =
      output_offset.scalar_type() == torch::kInt32 ? output_offset : output_offset.to(torch::kInt32);
  auto oo_cpu = output_offset_i32.cpu();
  const int32_t* oo = oo_cpu.data_ptr<int32_t>();
  TORCH_CHECK(oo[0] == 0, "output_offset[0] must be 0");
  TORCH_CHECK(oo[3] == n_total_i64, "output_offset[-1] must equal qkv_lora_b.size(1) (N_Q + 2 * N_KV)");
  TORCH_CHECK(oo[0] <= oo[1] && oo[1] <= oo[2] && oo[2] <= oo[3], "output_offset must be non-decreasing");
  const int64_t widest = std::max({oo[1] - oo[0], oo[2] - oo[1], oo[3] - oo[2]});
  TORCH_CHECK(
      max_qkv_out_dim == widest,
      "max_qkv_out_dim must equal max(output_q_dim, output_kv_dim) implied by output_offset");

  if (num_tokens_i64 == 0) {
    return;
  }
  // K == 0 (max_rank == 0) is a degenerate GEMM: the scaled LoRA term is an empty
  // sum (zero), so the output reduces to the residual when one is supplied, or the
  // zero matrix otherwise -- mirroring the SGEMM LoRA-B convention.
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
  (void)seg_len_max;  // not needed: grouped GEMM handles variable M per group

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
  // Keep output_offset_i32 on-device for the metadata kernel.
  output_offset_i32 =
      output_offset_i32.device() == input_x.device() ? output_offset_i32 : output_offset_i32.to(input_x.device());

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  const int n_total = static_cast<int>(n_total_i64);
  const int num_segments = static_cast<int>(num_segments_i64);
  (void)max_qkv_out_dim;  // validated above; per-group N comes from output_offset

  // Dispatch on dtype. Each launch symbol is defined in a separate generated
  // translation unit (see QKVLoraBFwdXe20.cmake).
  DISPATCH_QKV_LORA_B_FWD_DTYPE(
      input_x,
      qkv_lora_b,
      output_offset_i32,
      seg_indptr_i32,
      weight_indices_i32,
      scalings_f32,
      output,
      base_output,
      n_total,
      num_segments,
      queue);
}

#undef DISPATCH_QKV_LORA_B_FWD_DTYPE
#undef DISPATCH_QKV_LORA_B_FWD_TILE
#undef SYCL_INTEL_TARGET
