#pragma once
#include <oneapi/dpl/internal/random_impl/philox_engine.h>

#include <array>
#include <cstdint>

namespace sgl::random {

// One uniform in [0, 1) for a stream keyed by (subsequence, round), seeded from
// the (seed, offset) pair produced by the torch generator's philox engine.
// source:
// https://www.intel.com/content/www/us/en/docs/onedpl/developer-guide/2022-13-0/random-number-generators.html#PREDEFINED-RANDOM-NUMBER-ENGINES
inline float philox_uniform(uint64_t seed, uint64_t offset, uint32_t subsequence, uint32_t round) {
  using philox4x32_engine = oneapi::dpl::experimental::
      philox_engine<std::uint_fast32_t, 32, 4, 10, 0xCD9E8D57, 0x9E3779B9, 0xD2511F53, 0xBB67AE85>;
  philox4x32_engine engine(static_cast<uint32_t>(seed));
  const std::array<uint_fast32_t, 4> counter = {
      static_cast<uint_fast32_t>(offset),
      static_cast<uint_fast32_t>((offset >> 32) ^ (seed >> 32)),
      static_cast<uint_fast32_t>(subsequence),
      static_cast<uint_fast32_t>(round)};
  engine.set_counter(counter);
  const uint32_t x = static_cast<uint32_t>(engine());
  // 24-bit mantissa uniform in [0, 1).
  return static_cast<float>(x >> 8) * (1.0f / 16777216.0f);
}

}  // namespace sgl::random
