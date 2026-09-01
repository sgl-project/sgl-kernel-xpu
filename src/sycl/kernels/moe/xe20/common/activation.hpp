/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

// Fused gated-activation helpers for the BF16 Xe2 MoE grouped-GEMM mainloop.
// W4A16 applies activation in a separate kernel between GEMM1 and GEMM2.
// Non-gated activations such as ReLU2 are handled directly by the BF16
// mainloop.

#pragma once

#include <cmath>
#include <sycl/sycl.hpp>

namespace moe_xe20 {

// Gated-activation identifiers match the corresponding BF16 grouped-GEMM ABI
// values. Plain ints (not enum class) are used as template parameters.
inline constexpr int ACT_SILU = 0;
inline constexpr int ACT_GELU = 1;
inline constexpr int ACT_SWIGLU_GPT_OSS = 2;

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
