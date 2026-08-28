/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
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
#define SYCL_INTEL_TARGET 20

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <sycl/sycl.hpp>

#include "sgl_kernel_export.h"
#include "sycl/Utils.h"
#include "sycl/kernels/moe/xe20/w4a16/gemm_xe2_policy.hpp"
#ifdef USE_MOE_JIT
#include "jit/moe_jit.h"
#endif

namespace moe_w4a16 {
template <typename Policy, typename ElementS, typename ElementA>
void w4a16_launch(
    sycl::queue stream,
    const void* activations,
    const void* packed_weights,
    const void* scales,
    const void* zeros,
    const void* bias,
    void* outputs,
    const int gemm_n,
    const int gemm_k,
    const int* rows_per_expert,
    const int num_experts,
    const int group_size,
    int32_t* atomic_buffer);
}  // namespace moe_w4a16

#define DECLARE_W4A16_EXTERN(Policy, ElementS, ElementA)                               \
  extern template void moe_w4a16::w4a16_launch<moe_w4a16::Policy, ElementS, ElementA>( \
      sycl::queue,                                                                     \
      const void*,                                                                     \
      const void*,                                                                     \
      const void*,                                                                     \
      const void*,                                                                     \
      const void*,                                                                     \
      void*,                                                                           \
      const int,                                                                       \
      const int,                                                                       \
      const int*,                                                                      \
      const int,                                                                       \
      const int,                                                                       \
      int32_t*);

#define DECLARE_W4A16_POLICY(Policy)                                     \
  DECLARE_W4A16_EXTERN(Policy, cutlass::bfloat16_t, cutlass::bfloat16_t) \
  DECLARE_W4A16_EXTERN(Policy, cutlass::half_t, cutlass::half_t)         \
  DECLARE_W4A16_EXTERN(Policy, uint8_t, cutlass::bfloat16_t)             \
  DECLARE_W4A16_EXTERN(Policy, uint8_t, cutlass::half_t)

DECLARE_W4A16_POLICY(w4a16_policy_m_8_n_64)
DECLARE_W4A16_POLICY(w4a16_policy_m_16_n_64)
DECLARE_W4A16_POLICY(w4a16_policy_m_32_n_64)
DECLARE_W4A16_POLICY(w4a16_policy_m_64_n_128)
DECLARE_W4A16_POLICY(w4a16_policy_m_128_n_128)

#undef DECLARE_W4A16_POLICY
#undef DECLARE_W4A16_EXTERN

namespace {

// GEMM shape -> tile policy, weighted by how much of each tile the GEMM fills.
//
// The grouped GEMM tiles every expert's rows independently, so a policy with an
// M tile of T computes ceil(avg_m / T) * T rows per expert whatever avg_m is.
// Picking the widest tile unconditionally therefore falls off a cliff just past
// a multiple of T: at avg_m = 129 an M=128 tile does 256 rows of work for 129
// rows of data and loses half the machine. N tails have the same effect: a
// policy computes ceil(gemm_n / tile_n) * tile_n columns. Scoring each candidate
// by its peak throughput times both fill factors estimates which policy finishes
// first.
//
// The peaks below are measured on Xe2 (Arc Pro B60, bf16 activations) at
// avg_m values that fill each tile exactly, in TFLOP/s. They are only ever
// compared against each other, so the absolute scale does not matter -- what
// matters is that a wider tile is worth more per row but wastes more of a
// partial tile. GPT-OSS gemm1 (N=5760, K=2880) and DeepSeek-V4 gemm1
// (N=4096, K=4096) agree on this ordering.
constexpr int kW4A16TileM[] = {32, 64, 128};
constexpr int kW4A16TileN[] = {64, 128, 128};
constexpr float kW4A16TilePeakTflops[] = {49.0f, 64.0f, 68.0f};

int select_w4a16_tile_m(int avg_m, int gemm_n) {
  int best_tile_m = kW4A16TileM[0];
  float best_score = 0.0f;
  for (size_t i = 0; i < sizeof(kW4A16TileM) / sizeof(kW4A16TileM[0]); ++i) {
    const int tile_m = kW4A16TileM[i];
    const int tile_n = kW4A16TileN[i];
    const int rows_computed = ((avg_m + tile_m - 1) / tile_m) * tile_m;
    const int columns_computed = ((gemm_n + tile_n - 1) / tile_n) * tile_n;
    const float score = kW4A16TilePeakTflops[i] * static_cast<float>(avg_m) / static_cast<float>(rows_computed) *
                        static_cast<float>(gemm_n) / static_cast<float>(columns_computed);
    if (score > best_score) {
      best_score = score;
      best_tile_m = tile_m;
    }
  }
  return best_tile_m;
}

int select_w4a16_policy_id(int avg_m, int gemm_n) {
  if (avg_m <= 4) return 0;
  if (avg_m <= 8) return 1;

  const int tile_m = select_w4a16_tile_m(avg_m, gemm_n);
  if (tile_m <= 32) return 2;
  if (tile_m <= 64) return 3;
  return 4;
}

}  // namespace

SGL_KERNEL_EXPORT void moe_grouped_mm_nt_xe20_w4a16(
    torch::Tensor& output,                   // [total_m, N] bf16/fp16
    const torch::Tensor& activations,        // [total_m, K] bf16 or fp16
    const torch::Tensor& packed_weights,     // [E, N, K/2] int8/uint8 (two 4-bit values per byte)
    const torch::Tensor& scales,             // [E, N, K/group_size]: int4=activation dtype, mxfp4=E8M0 byte
    const std::optional<at::Tensor>& zeros,  // [E, N, K/group_size], same dtype as int4 scales, optional
    const std::optional<at::Tensor>& bias,   // [E, N] float32, optional
    const torch::Tensor& rows_per_expert,    // [E] int32 per-expert row counts
    const int64_t n_experts,
    bool is_int4,
    const int64_t group_size) {
  CHECK_INPUT(output);
  CHECK_INPUT(activations);
  CHECK_INPUT(packed_weights);
  CHECK_INPUT(scales);
  CHECK_INPUT(rows_per_expert);
  TORCH_CHECK(output.device() == activations.device(), "output must be on the same device as activations");
  TORCH_CHECK(
      packed_weights.device() == activations.device(), "packed_weights must be on the same device as activations");
  TORCH_CHECK(scales.device() == activations.device(), "scales must be on the same device as activations");
  TORCH_CHECK(
      rows_per_expert.device() == activations.device(), "rows_per_expert must be on the same device as activations");
  if (zeros.has_value()) {
    const auto& zeros_tensor = *zeros;
    CHECK_INPUT(zeros_tensor);
    TORCH_CHECK(zeros_tensor.device() == activations.device(), "zeros must be on the same device as activations");
  }
  if (bias.has_value()) {
    const auto& bias_tensor = *bias;
    CHECK_INPUT(bias_tensor);
    TORCH_CHECK(bias_tensor.device() == activations.device(), "bias must be on the same device as activations");
  }

  TORCH_CHECK(output.dim() == 2, "output must be 2D [total_m, N]");
  TORCH_CHECK(activations.dim() == 2, "activations must be 2D [total_m, K]");
  TORCH_CHECK(rows_per_expert.dim() == 1, "rows_per_expert must be 1D [E]");
  const int total_m = activations.size(0);
  const int gemm_k = activations.size(1);

  auto pw_shape = packed_weights.sizes().vec();
  TORCH_CHECK(pw_shape.size() == 3, "packed_weights must be 3D [E, N, K/2]");
  const int gemm_n = pw_shape[1];
  TORCH_CHECK(pw_shape[0] == n_experts, "packed_weights.size(0) must equal n_experts");
  TORCH_CHECK(pw_shape[2] == gemm_k / 2, "packed_weights.size(2) must equal K/2 (two 4-bit values per byte)");
  TORCH_CHECK(
      packed_weights.scalar_type() == at::ScalarType::Char || packed_weights.scalar_type() == at::ScalarType::Byte,
      "packed_weights must be int8 or uint8");

  TORCH_CHECK(
      group_size == 32 || group_size == 64 || group_size == 128 || group_size == 256,
      "group_size must be 32, 64, 128 or 256; got ",
      group_size);
  TORCH_CHECK(gemm_k % group_size == 0, "K must be a multiple of group_size");

  auto sc_shape = scales.sizes().vec();
  TORCH_CHECK(sc_shape.size() == 3, "scales must be 3D [E, N, K/group_size]");
  TORCH_CHECK(sc_shape[0] == n_experts, "scales.size(0) must equal n_experts");
  TORCH_CHECK(sc_shape[1] == gemm_n, "scales.size(1) must equal N");
  TORCH_CHECK(sc_shape[2] == gemm_k / group_size, "scales.size(2) must equal K/group_size");
  if (is_int4) {
    TORCH_CHECK(scales.scalar_type() == activations.scalar_type(), "int4 scales dtype must match activations dtype");
  } else {
    TORCH_CHECK(
        scales.scalar_type() == at::ScalarType::Byte || scales.scalar_type() == at::ScalarType::Float8_e8m0fnu,
        "mxfp4 scales must be uint8 or float8_e8m0fnu (E8M0 exponent)");
  }

  TORCH_CHECK(n_experts > 0, "n_experts must be positive");
  TORCH_CHECK(n_experts == rows_per_expert.size(0), "rows_per_expert must have n_experts elements");
  TORCH_CHECK(rows_per_expert.scalar_type() == at::ScalarType::Int, "rows_per_expert must be int32");
  TORCH_CHECK(output.size(0) == total_m, "output rows must match activations rows");
  TORCH_CHECK(output.size(1) == gemm_n, "output must have N columns");
  TORCH_CHECK(gemm_n % 8 == 0, "N must be divisible by 8");
  TORCH_CHECK(
      activations.scalar_type() == at::ScalarType::BFloat16 || activations.scalar_type() == at::ScalarType::Half,
      "activations must be bfloat16 or half");
  TORCH_CHECK(output.scalar_type() == activations.scalar_type(), "output dtype must match activations dtype");

  const void* bias_ptr = nullptr;
  if (bias.has_value()) {
    TORCH_CHECK(bias->scalar_type() == at::kFloat, "bias must be float32");
    TORCH_CHECK(bias->dim() == 2, "bias must be 2D [E, N]");
    TORCH_CHECK(bias->size(0) == n_experts && bias->size(1) == gemm_n, "bias shape must be [E, N]");
    bias_ptr = bias->data_ptr();
  }

  const void* zeros_ptr = nullptr;
  if (zeros.has_value()) {
    TORCH_CHECK(is_int4, "zeros (explicit zero-point) is only supported for int4, not mxfp4");
    TORCH_CHECK(zeros->scalar_type() == scales.scalar_type(), "zeros dtype must match int4 scales dtype");
    auto z_shape = zeros->sizes().vec();
    TORCH_CHECK(z_shape.size() == 3, "zeros must be 3D [E, N, K/group_size]");
    TORCH_CHECK(
        z_shape[0] == n_experts && z_shape[1] == gemm_n && z_shape[2] == gemm_k / group_size,
        "zeros shape must match scales shape [E, N, K/group_size]");
    zeros_ptr = zeros->data_ptr();
  }

  c10::DeviceGuard device_guard(activations.device());
  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();
  at::Tensor atomic_buffer = at::empty({static_cast<long>(1)}, activations.options().dtype(at::kInt));

  const int avg_m = total_m / static_cast<int>(n_experts);
  const int policy_id = select_w4a16_policy_id(avg_m, gemm_n);
  const bool is_fp16_act = activations.scalar_type() == at::ScalarType::Half;
#define LAUNCH_W4A16(Policy)                                                                  \
  do {                                                                                        \
    if (is_int4) {                                                                            \
      if (is_fp16_act) {                                                                      \
        moe_w4a16::w4a16_launch<moe_w4a16::Policy, cutlass::half_t, cutlass::half_t>(         \
            queue,                                                                            \
            activations.data_ptr(),                                                           \
            packed_weights.data_ptr(),                                                        \
            scales.data_ptr(),                                                                \
            zeros_ptr,                                                                        \
            bias_ptr,                                                                         \
            output.data_ptr(),                                                                \
            gemm_n,                                                                           \
            gemm_k,                                                                           \
            rows_per_expert.data_ptr<int>(),                                                  \
            static_cast<int>(n_experts),                                                      \
            static_cast<int>(group_size),                                                     \
            atomic_buffer.data_ptr<int>());                                                   \
      } else {                                                                                \
        moe_w4a16::w4a16_launch<moe_w4a16::Policy, cutlass::bfloat16_t, cutlass::bfloat16_t>( \
            queue,                                                                            \
            activations.data_ptr(),                                                           \
            packed_weights.data_ptr(),                                                        \
            scales.data_ptr(),                                                                \
            zeros_ptr,                                                                        \
            bias_ptr,                                                                         \
            output.data_ptr(),                                                                \
            gemm_n,                                                                           \
            gemm_k,                                                                           \
            rows_per_expert.data_ptr<int>(),                                                  \
            static_cast<int>(n_experts),                                                      \
            static_cast<int>(group_size),                                                     \
            atomic_buffer.data_ptr<int>());                                                   \
      }                                                                                       \
    } else {                                                                                  \
      if (is_fp16_act) {                                                                      \
        moe_w4a16::w4a16_launch<moe_w4a16::Policy, uint8_t, cutlass::half_t>(                 \
            queue,                                                                            \
            activations.data_ptr(),                                                           \
            packed_weights.data_ptr(),                                                        \
            scales.data_ptr(),                                                                \
            zeros_ptr,                                                                        \
            bias_ptr,                                                                         \
            output.data_ptr(),                                                                \
            gemm_n,                                                                           \
            gemm_k,                                                                           \
            rows_per_expert.data_ptr<int>(),                                                  \
            static_cast<int>(n_experts),                                                      \
            static_cast<int>(group_size),                                                     \
            atomic_buffer.data_ptr<int>());                                                   \
      } else {                                                                                \
        moe_w4a16::w4a16_launch<moe_w4a16::Policy, uint8_t, cutlass::bfloat16_t>(             \
            queue,                                                                            \
            activations.data_ptr(),                                                           \
            packed_weights.data_ptr(),                                                        \
            scales.data_ptr(),                                                                \
            zeros_ptr,                                                                        \
            bias_ptr,                                                                         \
            output.data_ptr(),                                                                \
            gemm_n,                                                                           \
            gemm_k,                                                                           \
            rows_per_expert.data_ptr<int>(),                                                  \
            static_cast<int>(n_experts),                                                      \
            static_cast<int>(group_size),                                                     \
            atomic_buffer.data_ptr<int>());                                                   \
      }                                                                                       \
    }                                                                                         \
  } while (0)

#define DISPATCH_W4A16_POLICY()                 \
  do {                                          \
    switch (policy_id) {                        \
      case 0:                                   \
        LAUNCH_W4A16(w4a16_policy_m_8_n_64);    \
        break;                                  \
      case 1:                                   \
        LAUNCH_W4A16(w4a16_policy_m_16_n_64);   \
        break;                                  \
      case 2:                                   \
        LAUNCH_W4A16(w4a16_policy_m_32_n_64);   \
        break;                                  \
      case 3:                                   \
        LAUNCH_W4A16(w4a16_policy_m_64_n_128);  \
        break;                                  \
      case 4:                                   \
        LAUNCH_W4A16(w4a16_policy_m_128_n_128); \
        break;                                  \
    }                                           \
  } while (0)

#ifdef USE_MOE_JIT
  {
    std::string jit_err;
    TORCH_CHECK(
        sgl::moe_jit::w4a16_grouped_gemm_launch(
            policy_id,
            is_int4,
            is_fp16_act,
            &queue,
            activations.data_ptr(),
            packed_weights.data_ptr(),
            scales.data_ptr(),
            zeros_ptr,
            bias_ptr,
            output.data_ptr(),
            gemm_n,
            gemm_k,
            rows_per_expert.data_ptr<int>(),
            static_cast<int>(n_experts),
            static_cast<int>(group_size),
            atomic_buffer.data_ptr<int>(),
            jit_arch_code(),
            &jit_err),
        jit_err);
  }
#else
  DISPATCH_W4A16_POLICY();
#endif

#undef DISPATCH_W4A16_POLICY
#undef LAUNCH_W4A16
}

#undef SYCL_INTEL_TARGET
