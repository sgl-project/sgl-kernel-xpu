/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

// Shared fused-activation helpers for the Xe2 MoE grouped-GEMM mainloops.
// Both the bf16 and MXFP4 W4A16 mainloops produce a pair of fp32
// accumulator fragments (tCrC0 = gate, tCrC1 = up) and need the same
// gate-and-mul formulas to fuse activation with the GEMM. Centralizing
// the activation IDs and per-element compute here means new activations
// (e.g. PR #194's ReLU²) drop into one file, not two.

#pragma once

#include <cmath>
#include <sycl/sycl.hpp>

namespace moe_xe20 {

// Activation identifiers. Match the ABI exposed by
// torch.ops.sgl_kernel.moe_grouped_mm_nt_xe20{,_mxfp4_w4a16}; the Python
// wrapper translates string activation names to these ints. Plain ints
// (not enum class) so they can be used as int template parameters.
inline constexpr int ACT_SILU = 0;
inline constexpr int ACT_GELU = 1;
inline constexpr int ACT_SWIGLU_GPT_OSS = 2;
// DeepSeek-V4 swiglu: clamp gate to [-inf, limit] and up to [-limit, limit],
// then silu(gate) * up. Differs from GPT-OSS only in alpha==1 (plain silu,
// no 1.702 scale) and no (up + 1) bias. The limit is passed via gemm1_limit.
inline constexpr int ACT_SWIGLU_DEEPSEEK_V4 = 4;

// One-element fused gate-and-mul. `x` is the gate accumulator output, `y`
// is the up accumulator output, both in fp32. Returns the fp32 fused
// activation value, which the caller then writes back into tCrC0(i).
//
// alpha/limit are unused for SILU and GELU; the caller passes them
// unconditionally and the unused branches drop them out at compile time.
template <int ActType>
CUTLASS_DEVICE float apply_fused_activation(float x, float y, float alpha, float limit) {
  if constexpr (ActType == ACT_SILU) {
    float s = 1.0f / (1.0f + sycl::native::exp(-x));
    return x * s * y;
  } else if constexpr (ActType == ACT_SWIGLU_GPT_OSS) {
    float gate = sycl::fmin(x, limit);
    float up = sycl::fmax(-limit, sycl::fmin(y, limit));
    float t = gate * alpha;
    float s = 1.0f / (1.0f + sycl::native::exp(-t));
    return gate * s * (up + 1.0f);
  } else if constexpr (ActType == ACT_SWIGLU_DEEPSEEK_V4) {
    // DeepSeek-V4: clamp gate to max=limit, up to [-limit, limit], then
    // plain silu(gate) * up (alpha == 1, no +1 bias). `alpha` is unused.
    float gate = sycl::fmin(x, limit);
    float up = sycl::fmax(-limit, sycl::fmin(y, limit));
    float s = 1.0f / (1.0f + sycl::native::exp(-gate));
    return gate * s * up;
  } else {                                        // GELU (tanh approx)
    constexpr float kBeta = 0.7978845608028654f;  // sqrt(2.0f / pi)
    constexpr float kAlpha = 0.044715f;
    float x_cube = x * x * x;
    float tanh_arg = kBeta * (x + kAlpha * x_cube);
    float s = 0.5f * (1.0f + std::tanh(tanh_arg));
    return x * s * y;
  }
}

}  // namespace moe_xe20
