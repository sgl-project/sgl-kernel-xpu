#pragma once
#include <cstdint>
#include <sycl/sycl.hpp>

#include "cutlass/bfloat16.h"
namespace sgl::gemm {
inline uint32_t bf16_round_bits(float value) {
  auto bits = sycl::bit_cast<uint32_t>(value);
  if ((bits & 0x7f800000u) == 0x7f800000u) return (bits >> 16) | ((bits & 0x007fffffu) ? 0x40u : 0u);
  return (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16;
}
// The table contains native BF16 GELU(tanh), including its BF16 intermediate
// rounding. Round both GEMM projections to BF16 before the lookup/multiply;
// the existing mainloop BF16 store performs the final product rounding.
struct Bf16GeluTanhLut {
  const cutlass::bfloat16_t* lut;
  float operator()(float gate, float up, float, float) const {
    auto gate_bits = bf16_round_bits(gate);
    auto up_bits = bf16_round_bits(up);
    return float(lut[gate_bits]) * sycl::bit_cast<float>(up_bits << 16);
  }
};
}  // namespace sgl::gemm
