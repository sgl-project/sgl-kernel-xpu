#pragma once

#include <array>
#include <cstdint>

namespace sgl::random {

namespace philox {

// Philox4x32 multipliers and Weyl (key-bump) constants.
constexpr uint32_t kPhiloxM0 = 0xD2511F53u;
constexpr uint32_t kPhiloxM1 = 0xCD9E8D57u;
constexpr uint32_t kPhiloxW0 = 0x9E3779B9u;  // golden ratio
constexpr uint32_t kPhiloxW1 = 0xBB67AE85u;  // sqrt(3) - 1

// One Philox round: mix the 4 counter words with the 2 key words.
inline void philox_round(std::array<uint32_t, 4>& ctr, const std::array<uint32_t, 2>& key) {
  const uint64_t prod0 = static_cast<uint64_t>(kPhiloxM0) * ctr[0];
  const uint64_t prod1 = static_cast<uint64_t>(kPhiloxM1) * ctr[2];
  const uint32_t hi0 = static_cast<uint32_t>(prod0 >> 32);
  const uint32_t lo0 = static_cast<uint32_t>(prod0);
  const uint32_t hi1 = static_cast<uint32_t>(prod1 >> 32);
  const uint32_t lo1 = static_cast<uint32_t>(prod1);

  const std::array<uint32_t, 4> out = {
      static_cast<uint32_t>(hi1 ^ ctr[1] ^ key[0]), lo1, static_cast<uint32_t>(hi0 ^ ctr[3] ^ key[1]), lo0};
  ctr = out;
}

// 10-round Philox4x32; returns the first output word.
inline uint32_t philox4x32_10(std::array<uint32_t, 4> ctr, std::array<uint32_t, 2> key) {
#pragma unroll
  for (int r = 0; r < 10; ++r) {
    philox_round(ctr, key);
    key[0] += kPhiloxW0;
    key[1] += kPhiloxW1;
  }
  return ctr[0];
}

}  // namespace philox

inline float philox_uniform(uint64_t seed, uint64_t offset, uint32_t subsequence, uint32_t round) {
  const std::array<uint32_t, 2> key = {static_cast<uint32_t>(seed), 0u};
  const std::array<uint32_t, 4> counter = {
      static_cast<uint32_t>(offset), static_cast<uint32_t>((offset >> 32) ^ (seed >> 32)), subsequence, round};
  const uint32_t x = philox::philox4x32_10(counter, key);
  // 24-bit mantissa uniform in [0, 1).
  return static_cast<float>(x >> 8) * (1.0f / 16777216.0f);
}

}  // namespace sgl::random
