/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Torch op dispatcher for the unified int4/mxfp4 W4A16 MoE grouped GEMM.
 * Ported from vllm-xpu-kernels
 * (csrc/xpu/grouped_gemm/xe_2/grouped_gemm_xe2_interface.hpp), adapted to
 * sgl-kernel-xpu's AOT instantiation scheme.
 *
 * A single kernel template (moe_w4a16::MoEGEMM) serves both int4 and mxfp4:
 *   - ElementS = ElementA           => int4 (scale is a direct multiplier;
 *     an optional explicit per-group zero-point tensor -- same layout/dtype
 *     as scales -- may be supplied via `zeros`, dequanting as
 *     `(code - zp) * scale` instead of requiring the zero-point to be
 *     pre-folded into a signed 4-bit code, which overflows for real,
 *     non-symmetric AWQ checkpoints)
 *   - ElementS = uint8_t             => mxfp4 (scale is an E8M0 exponent,
 *     decoded to fp32 in registers via `<< 23`; no zero-point, `zeros` must
 *     be absent)
 * Activations/output dtype (ElementA) is either cutlass::bfloat16_t or
 * cutlass::half_t (fp16), dispatched at runtime off activations.scalar_type(),
 * mirroring vllm-xpu-kernels' A_dtype branch in grouped_gemm_xe2_interface.hpp.
 * group_size (32/64/128/256) is a runtime argument; all four are compiled into
 * every instance, so it does not multiply the AOT instance count.
 *
 * The AOT instantiation matrix (4 policies x {int4, mxfp4} x {bf16, fp16} =
 * 16 units) lives in src/GroupGemmW4A16Xe20.cmake.
 **************************************************************************************************/
#define SYCL_INTEL_TARGET 20

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <sycl/sycl.hpp>

#include "sycl/kernels/moe/xe20/w4a16/gemm_xe2_policy.hpp"

// Primary template declaration for the AOT-instantiated entry points. The
// definition lives in w4a16_launcher.hpp and is compiled into the generated
// translation units (GroupGemmW4A16Xe20LauncherInstance.cpp.in); here we only
// need the declaration plus extern template to avoid pulling the heavy kernel
// into this dispatcher TU.
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

#define DECLARE_W4A16_EXTERN(Policy, ElementS, ElementA)                                       \
  extern template void moe_w4a16::w4a16_launch<moe_w4a16::Policy, ElementS, ElementA>(          \
      sycl::queue,                                                                    \
      const void*,                                                                   \
      const void*,                                                                   \
      const void*,                                                                   \
      const void*,                                                                   \
      const void*,                                                                   \
      void*,                                                                         \
      const int,                                                                     \
      const int,                                                                     \
      const int*,                                                                    \
      const int,                                                                     \
      const int,                                                                     \
      int32_t*);

#define DECLARE_W4A16_POLICY(Policy)                                       \
  DECLARE_W4A16_EXTERN(Policy, cutlass::bfloat16_t, cutlass::bfloat16_t) \
  DECLARE_W4A16_EXTERN(Policy, cutlass::half_t, cutlass::half_t)         \
  DECLARE_W4A16_EXTERN(Policy, uint8_t, cutlass::bfloat16_t)             \
  DECLARE_W4A16_EXTERN(Policy, uint8_t, cutlass::half_t)

DECLARE_W4A16_POLICY(w4a16_policy_m_8)
DECLARE_W4A16_POLICY(w4a16_policy_m_16)
DECLARE_W4A16_POLICY(w4a16_policy_m_32)
DECLARE_W4A16_POLICY(w4a16_policy)

#undef DECLARE_W4A16_POLICY
#undef DECLARE_W4A16_EXTERN

void moe_grouped_mm_nt_xe20_w4a16(
    torch::Tensor& output,                  // [total_m, N] bf16/fp16
    const torch::Tensor& activations,       // [total_m, K] bf16 or fp16
    const torch::Tensor& packed_weights,    // [E, N, K/2] int8 (two 4-bit values per byte)
    const torch::Tensor& scales,            // [E, N, K/group_size]: int4=activation dtype, mxfp4=uint8
    const std::optional<at::Tensor>& zeros, // [E, N, K/group_size], same dtype as int4 scales, optional
    const std::optional<at::Tensor>& bias,  // [E, N] float32, optional
    const torch::Tensor& rows_per_expert,   // [E] int32 per-expert row counts
    const int64_t n_experts,
    bool is_int4,
    const int64_t group_size) {
  const int total_m = activations.size(0);
  const int gemm_k = activations.size(1);

  auto pw_shape = packed_weights.sizes().vec();
  TORCH_CHECK(pw_shape.size() == 3, "packed_weights must be 3D [E, N, K/2]");
  const int gemm_n = pw_shape[1];
  TORCH_CHECK(pw_shape[0] == n_experts, "packed_weights.size(0) must equal n_experts");
  TORCH_CHECK(pw_shape[2] == gemm_k / 2, "packed_weights.size(2) must equal K/2 (two 4-bit values per byte)");
  TORCH_CHECK(packed_weights.scalar_type() == at::ScalarType::Char, "packed_weights must be int8");

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
    TORCH_CHECK(
        scales.scalar_type() == activations.scalar_type(), "int4 scales dtype must match activations dtype");
  } else {
    TORCH_CHECK(scales.scalar_type() == at::ScalarType::Byte, "mxfp4 scales must be uint8 (E8M0 exponent)");
  }

  TORCH_CHECK(n_experts == rows_per_expert.size(0), "rows_per_expert must have n_experts elements");
  TORCH_CHECK(rows_per_expert.scalar_type() == at::ScalarType::Int, "rows_per_expert must be int32");
  TORCH_CHECK(output.size(0) == total_m, "output rows must match activations rows");
  TORCH_CHECK(output.size(1) == gemm_n, "output must have N columns");
  TORCH_CHECK(n_experts % 8 == 0, "n_experts must be a multiple of 8 for the current implementation");
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

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();
  at::Tensor atomic_buffer = at::empty({static_cast<long>(1)}, activations.options().dtype(at::kInt));

  const int avg_m = total_m / static_cast<int>(n_experts);
  const bool is_fp16_act = activations.scalar_type() == at::ScalarType::Half;
#define LAUNCH_W4A16(Policy)                                                                     \
  do {                                                                                           \
    if (is_int4) {                                                                               \
      if (is_fp16_act) {                                                                         \
        moe_w4a16::w4a16_launch<moe_w4a16::Policy, cutlass::half_t, cutlass::half_t>(            \
            queue, activations.data_ptr(), packed_weights.data_ptr(), scales.data_ptr(),         \
            zeros_ptr, bias_ptr, output.data_ptr(), gemm_n, gemm_k,                              \
            rows_per_expert.data_ptr<int>(), static_cast<int>(n_experts),                        \
            static_cast<int>(group_size), atomic_buffer.data_ptr<int>());                        \
      } else {                                                                                   \
        moe_w4a16::w4a16_launch<moe_w4a16::Policy, cutlass::bfloat16_t, cutlass::bfloat16_t>(    \
            queue, activations.data_ptr(), packed_weights.data_ptr(), scales.data_ptr(),         \
            zeros_ptr, bias_ptr, output.data_ptr(), gemm_n, gemm_k,                              \
            rows_per_expert.data_ptr<int>(), static_cast<int>(n_experts),                        \
            static_cast<int>(group_size), atomic_buffer.data_ptr<int>());                        \
      }                                                                                          \
    } else {                                                                                     \
      if (is_fp16_act) {                                                                         \
        moe_w4a16::w4a16_launch<moe_w4a16::Policy, uint8_t, cutlass::half_t>(                    \
            queue, activations.data_ptr(), packed_weights.data_ptr(), scales.data_ptr(),         \
            zeros_ptr, bias_ptr, output.data_ptr(), gemm_n, gemm_k,                              \
            rows_per_expert.data_ptr<int>(), static_cast<int>(n_experts),                        \
            static_cast<int>(group_size), atomic_buffer.data_ptr<int>());                        \
      } else {                                                                                   \
        moe_w4a16::w4a16_launch<moe_w4a16::Policy, uint8_t, cutlass::bfloat16_t>(                \
            queue, activations.data_ptr(), packed_weights.data_ptr(), scales.data_ptr(),         \
            zeros_ptr, bias_ptr, output.data_ptr(), gemm_n, gemm_k,                              \
            rows_per_expert.data_ptr<int>(), static_cast<int>(n_experts),                        \
            static_cast<int>(group_size), atomic_buffer.data_ptr<int>());                        \
      }                                                                                          \
    }                                                                                            \
  } while (0)

#define DISPATCH_W4A16_POLICY()                  \
  do {                                          \
    if (avg_m <= 4) {                           \
      LAUNCH_W4A16(w4a16_policy_m_8);           \
    } else if (avg_m <= 8) {                    \
      LAUNCH_W4A16(w4a16_policy_m_16);          \
    } else if (avg_m <= 128) {                  \
      LAUNCH_W4A16(w4a16_policy_m_32);          \
    } else {                                    \
      LAUNCH_W4A16(w4a16_policy);               \
    }                                           \
  } while (0)

  DISPATCH_W4A16_POLICY();

#undef DISPATCH_W4A16_POLICY
#undef LAUNCH_W4A16
}

#undef SYCL_INTEL_TARGET
