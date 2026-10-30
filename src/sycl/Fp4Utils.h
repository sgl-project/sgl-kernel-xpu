#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

namespace at::native::xpu::fp4 {

inline uint8_t encode_e2m1_ties_to_even(float value) {
  const float magnitude = sycl::fmin(sycl::fabs(value), 6.0f);
  uint8_t code = 0;
  code += magnitude > 0.25f;
  code += magnitude >= 0.75f;
  code += magnitude > 1.25f;
  code += magnitude >= 1.75f;
  code += magnitude > 2.5f;
  code += magnitude >= 3.5f;
  code += magnitude > 5.0f;
  if (value < 0.0f && code != 0) code |= 0x8u;
  return code;
}

inline uint8_t encode_e2m1_ties_to_zero(float value) {
  const float magnitude = sycl::fmin(sycl::fabs(value), 6.0f);
  uint8_t code = 0;
  code += magnitude > 0.25f;
  code += magnitude > 0.75f;
  code += magnitude > 1.25f;
  code += magnitude > 1.75f;
  code += magnitude > 2.5f;
  code += magnitude > 3.5f;
  code += magnitude > 5.0f;
  if (value < 0.0f && code != 0) code |= 0x8u;
  return code;
}

inline float decode_e2m1(uint8_t code) {
  constexpr float kValues[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float value = kValues[code & 0x7u];
  return (code & 0x8u) ? -value : value;
}

inline float decode_ue8m0_scale(uint8_t ue8m0) {
  if (ue8m0 == 0) return 0x1p-127f;
  if (ue8m0 == 255) return sycl::nan(0u);
  const uint32_t bits = static_cast<uint32_t>(ue8m0) << 23;
  return sycl::bit_cast<float>(bits);
}

}  // namespace at::native::xpu::fp4
