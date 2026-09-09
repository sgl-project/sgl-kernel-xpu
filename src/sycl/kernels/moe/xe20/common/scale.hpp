/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#pragma once

#include <cute/util/compat.hpp>
#include <sycl/sycl.hpp>

#include "cutlass/half.h"

namespace moe_xe20 {

template <typename Element>
CUTE_DEVICE Element apply_bf16_or_fp16_scale(Element value, float scale) {
  static_assert(cute::is_same_v<Element, cutlass::bfloat16_t> || cute::is_same_v<Element, cutlass::half_t>);
  uint16_t bits = sycl::bit_cast<uint16_t>(value);
#if defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_INTEL_TARGET)
  if constexpr (cute::is_same_v<Element, cutlass::bfloat16_t>) {
    asm("{\n"
        ".decl Z_BF16 v_type=G type=BF num_elts=16 alias=<%0,0>\n"
        ".decl Y_FP32 v_type=G type=F num_elts=16 alias=<%1,0>\n"
        "mul (M1, 16) Z_BF16(0,0)<1> Z_BF16(0,0)<1;1,0> Y_FP32(0,0)<1;1,0>\n"
        "}\n"
        : "+rw"(bits)
        : "rw"(scale));
  } else {
    asm("{\n"
        ".decl Z_FP16 v_type=G type=HF num_elts=16 alias=<%0,0>\n"
        ".decl Y_FP32 v_type=G type=F num_elts=16 alias=<%1,0>\n"
        "mul (M1, 16) Z_FP16(0,0)<1> Z_FP16(0,0)<1;1,0> Y_FP32(0,0)<1;1,0>\n"
        "}\n"
        : "+rw"(bits)
        : "rw"(scale));
  }
#else
  return Element(static_cast<float>(value) * scale);
#endif
  return sycl::bit_cast<Element>(bits);
}

}  // namespace moe_xe20
