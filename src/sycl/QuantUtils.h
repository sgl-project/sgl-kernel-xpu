#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

namespace at::native::xpu {

inline uint8_t castToUE8M0(float x) {
  uint32_t bits = sycl::bit_cast<uint32_t>(x);
  uint32_t exp = (bits >> 23) & 0xFF;
  uint32_t round_up = (bits & 0x7FFFFF) != 0 ? 1 : 0;
  return static_cast<uint8_t>(exp + round_up);
}

inline float invScaleUE8M0(uint8_t ue8m0) {
  if (ue8m0 >= 254) return 0.0f;
  uint32_t inv_exp = 254 - ue8m0;
  uint32_t inv_bits = inv_exp << 23;
  return sycl::bit_cast<float>(inv_bits);
}

}  // namespace at::native::xpu
