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
#include <type_traits>
#include <utility>
#include <vector>

#include "SYCLHelpers.h"
#include "Utils.h"
#include "kernels/lora/group_gemm_lora_launcher.hpp"

namespace {

//----------------- Compile-time perf knobs (LoRA-A forward) -----------------//
// Pinned once here for every LoRA-A entry point (fp16/bf16 single GEMM and the
// fp32 3xTF32 path) so the tile / subgroup / layout mix lives in a single place.
//   TileShape    = 256 x 256 x 32   -- canonical tile from upstream 04_bmg_grouped_gemm.
//   ThreadLayout = 8 x 4 x 1        -- 32 subgroups per workgroup.
//   LayoutB      = ColumnMajor      -- free-transpose B (auto copy atom transposes).
//   PipelineStages = 2              -- matches the upstream BMG reference.
using LoraAFwdTileShape = cute::Shape<cute::_256, cute::_256, cute::_32>;
using LoraAFwdThreadLayout =
    cute::Layout<cute::Shape<cute::_8, cute::_4, cute::_1>, cute::Stride<cute::_4, cute::_1, cute::_0>>;
using LoraAFwdLayoutB = cutlass::layout::ColumnMajor;
constexpr int kLoraAFwdPipelineStages = 2;

//----------------- Shared host-side grouped-GEMM metadata --------------------//
//
// Per-segment problem sizes, element byte-offsets, and strides that the CUTLASS
// pointer-array grouped GEMM consumes. Built once on the host and reused by
// every GEMM in a launch (the fp32 3xTF32 path issues three GEMMs off one set).
//
// For each segment s in [0, num_segments):
//   M_s = seg_indptr[s+1] - seg_indptr[s]
//   N   = stack_num * max_rank                (uniform across segments)
//   K   = input_x.size(1) = input_dim         (uniform across segments)
//
//   a_off[s] = seg_indptr[s]        * K * elem_bytes    (into input_x)
//   b_off[s] = weight_indices[s] * N * K * elem_bytes   (into weights)
//   d_off[s] = seg_indptr[s]        * N * elem_bytes    (into output)

struct GroupedGemmMeta {
  torch::Tensor problem_sizes;  // int32 [num_segments, 3]  (M_s, N, K), on device
  torch::Tensor stride_A;       // int64 [num_segments]     leading dim of A = K
  torch::Tensor stride_B;       // int64 [num_segments]     leading dim of B = K
  torch::Tensor stride_D;       // int64 [num_segments]     leading dim of D = N
  std::vector<int64_t> a_off;   // byte offset into A per segment (host)
  std::vector<int64_t> b_off;   // byte offset into B per segment (host)
  std::vector<int64_t> d_off;   // byte offset into D per segment (host)
};

GroupedGemmMeta build_grouped_gemm_meta(
    const torch::Tensor& seg_indptr_i32,      // int32 [num_segments + 1]
    const torch::Tensor& weight_indices_i32,  // int32 [num_segments]
    const int N,
    const int K,
    const int num_segments,
    const int64_t elem_bytes,
    const at::Device device) {
  auto seg_indptr_cpu = seg_indptr_i32.cpu().contiguous();
  auto weight_indices_cpu = weight_indices_i32.cpu().contiguous();
  const int32_t* seg_indptr_h = seg_indptr_cpu.data_ptr<int32_t>();
  const int32_t* weight_indices_h = weight_indices_cpu.data_ptr<int32_t>();

  std::vector<int32_t> problem_sizes_h(static_cast<size_t>(num_segments) * 3);
  GroupedGemmMeta meta;
  meta.a_off.resize(num_segments);
  meta.b_off.resize(num_segments);
  meta.d_off.resize(num_segments);

  const int64_t b_per_lora_bytes = static_cast<int64_t>(N) * K * elem_bytes;
  for (int s = 0; s < num_segments; ++s) {
    const int32_t row_start = seg_indptr_h[s];
    const int32_t M_s = seg_indptr_h[s + 1] - row_start;
    const int32_t lora_id = weight_indices_h[s];

    problem_sizes_h[3 * s + 0] = M_s;
    // N is always the full stack_num * max_rank; per-adapter lora_ranks are NOT
    // folded into the GEMM problem size. Every segment computes all N output
    // columns, so the API contract is that weight rows beyond an adapter's rank
    // R_l (i.e. rows j >= R_l within each stacked block) are pre-zeroed by the
    // caller. Those rows still take part in the GEMM but contribute zeros,
    // yielding the correct zero-padded output for ranks smaller than max_rank.
    problem_sizes_h[3 * s + 1] = N;
    problem_sizes_h[3 * s + 2] = K;

    meta.a_off[s] = static_cast<int64_t>(row_start) * K * elem_bytes;
    meta.b_off[s] = static_cast<int64_t>(lora_id) * b_per_lora_bytes;
    meta.d_off[s] = static_cast<int64_t>(row_start) * N * elem_bytes;
  }

  // Strides in elements (leading dim of A/B = K, D = N).
  std::vector<int64_t> stride_A_h(num_segments, static_cast<int64_t>(K));
  std::vector<int64_t> stride_B_h(num_segments, static_cast<int64_t>(K));
  std::vector<int64_t> stride_D_h(num_segments, static_cast<int64_t>(N));

  auto cpu_i32 = torch::TensorOptions().dtype(torch::kInt32);
  auto cpu_i64 = torch::TensorOptions().dtype(torch::kInt64);
  meta.problem_sizes = torch::from_blob(problem_sizes_h.data(), {num_segments, 3}, cpu_i32).clone().to(device);
  meta.stride_A = torch::from_blob(stride_A_h.data(), {num_segments}, cpu_i64).clone().to(device);
  meta.stride_B = torch::from_blob(stride_B_h.data(), {num_segments}, cpu_i64).clone().to(device);
  meta.stride_D = torch::from_blob(stride_D_h.data(), {num_segments}, cpu_i64).clone().to(device);
  return meta;
}

// Turn a base tensor + host byte-offsets into a device int64 pointer array
// (one absolute device address per segment) for the pointer-array grouped GEMM.
torch::Tensor
make_device_ptrs(const torch::Tensor& base, const std::vector<int64_t>& off_bytes, const at::Device device) {
  const int64_t base_addr = reinterpret_cast<int64_t>(base.data_ptr());
  const int num_segments = static_cast<int>(off_bytes.size());
  std::vector<int64_t> ptrs_h(num_segments);
  for (int s = 0; s < num_segments; ++s) {
    ptrs_h[s] = base_addr + off_bytes[s];
  }
  auto cpu_i64 = torch::TensorOptions().dtype(torch::kInt64);
  return torch::from_blob(ptrs_h.data(), {num_segments}, cpu_i64).clone().to(device);
}

//----------------- fp16 / bf16 launch (single grouped GEMM) ------------------//
template <typename TensorDType>
void launch_sgemm_lora_a_fwd(
    const torch::Tensor& input_x,
    const torch::Tensor& weights,
    const torch::Tensor& seg_indptr_i32,
    const torch::Tensor& weight_indices_i32,
    torch::Tensor& output,
    const int stack_num,
    const int max_rank,
    const int num_segments,
    sycl::queue& queue) {
  const int K = static_cast<int>(input_x.size(1));  // input_dim
  const int N = stack_num * max_rank;               // output columns
  const int64_t elem_bytes = static_cast<int64_t>(sizeof(TensorDType));
  const auto device = input_x.device();

  auto meta = build_grouped_gemm_meta(seg_indptr_i32, weight_indices_i32, N, K, num_segments, elem_bytes, device);

  auto a_ptrs = make_device_ptrs(input_x, meta.a_off, device);
  auto b_ptrs = make_device_ptrs(weights, meta.b_off, device);
  auto d_ptrs = make_device_ptrs(output, meta.d_off, device);

  // The launcher uses the modern (non-legacy) grouped path
  // (MainloopXeL1StagedGroup), which auto-selects the XMX DPAS MMA atom and all
  // gmem load/store/prefetch copy atoms per dtype.
  at::native::xpu::launch_group_gemm_lora_fwd<
      TensorDType,
      LoraAFwdTileShape,
      LoraAFwdThreadLayout,
      LoraAFwdLayoutB,
      kLoraAFwdPipelineStages>(
      queue,
      meta.problem_sizes,
      a_ptrs,
      b_ptrs,
      /*c_ptrs   =*/d_ptrs,
      d_ptrs,
      meta.stride_A,
      meta.stride_B,
      /*stride_C =*/meta.stride_D,
      meta.stride_D,
      num_segments,
      /*alpha    =*/1.0f,
      /*beta     =*/0.0f);
}

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

  // Dispatch kernel based on data type
  if (weights.scalar_type() == torch::kHalf) {
    launch_sgemm_lora_a_fwd<at::Half>(
        input_x, weights, seg_indptr_i32, weight_indices_i32, output, stack_num_, max_rank, num_segments, queue);
  } else if (weights.scalar_type() == torch::kBFloat16) {
    launch_sgemm_lora_a_fwd<at::BFloat16>(
        input_x, weights, seg_indptr_i32, weight_indices_i32, output, stack_num_, max_rank, num_segments, queue);
  } else {
    TORCH_CHECK(false, "Unsupported data type for weights");
  }
}

#undef SYCL_INTEL_TARGET
