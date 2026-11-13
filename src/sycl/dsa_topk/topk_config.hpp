// Adapted from vllm-xpu-kernels csrc/dsa_topk (5bc6a53a).
#pragma once

// DSA top-k configuration, key extraction and coarse-bin boundaries.
// Direct port of topk_impl.cuh (TopKConfig, extract_coarse_bin,
// coarse_bin_bounds, TieValue, TopKProblem).

#include <cstdint>
#include <limits>
#include <sycl/sycl.hpp>

#include "topk_warp.hpp"

namespace vllm {
namespace dsa_topk {

// ---------------------------------------------------------------------------
// Exactness contract (SGL_TOPK_MODE): FAST=0, EXACT=1, DETERMINISTIC=2.
// Chosen at compile time by the DSA_TOPK_MODE macro.
// ---------------------------------------------------------------------------
#ifndef DSA_TOPK_MODE
#define DSA_TOPK_MODE 0
#endif
static_assert(DSA_TOPK_MODE >= 0 && DSA_TOPK_MODE <= 2, "DSA_TOPK_MODE must be 0, 1 or 2");

enum class Exactness : uint32_t { FAST = 0, EXACT = 1, DETERMINISTIC = 2 };
inline constexpr Exactness kExactness = static_cast<Exactness>(DSA_TOPK_MODE);
inline constexpr bool kExact = kExactness != Exactness::FAST;
inline constexpr bool kDeterministic = kExactness == Exactness::DETERMINISTIC;
inline constexpr uint32_t kDeterministicIdxBits = 20;

// Only FAST mode is wired up; overflow is resolved by a separate exact rescan.
static_assert(kExactness == Exactness::FAST, "DSA_TOPK_MODE EXACT/DETERMINISTIC are not supported.");

// ---------------------------------------------------------------------------
// Order-preserving keys.
// ---------------------------------------------------------------------------
// extract_exact_bin is in topk_warp.hpp.

// extract_exact_bin for tie resolution. Deterministic mode orders -0 and +0 as
// the equal floats they are, so its index tie-break spans the two zeros.
inline uint32_t extract_tie_key(float x) {
  if constexpr (kDeterministic) x = (x == 0.0f) ? 0.0f : x;
  return extract_exact_bin(x);
}

inline constexpr float padding_value() {
  return std::numeric_limits<float>::quiet_NaN();
}
inline constexpr float infinity_value() {
  return std::numeric_limits<float>::infinity();
}

// fp32 -> coarse kBits bin via the ordered fp16 key (topk_impl.cuh).
template <uint32_t kBits>
inline uint32_t extract_coarse_bin(float x) {
  static_assert(0 < kBits && kBits < 15);
  const sycl::half h = static_cast<sycl::half>(x);
  const uint32_t bits = static_cast<uint32_t>(sycl::bit_cast<uint16_t>(h)) << 16;
  const uint32_t sign = static_cast<uint32_t>(static_cast<int32_t>(bits) >> 31);
  return (bits ^ (sign | 0x80000000u)) >> (32 - kBits);
}

// ordered16 key (one half) -> fp32. Mirrors topk_impl.cuh's to_fp32, scalar.
inline float ordered16_to_fp32(uint32_t k16 /* low 16 bits used */) {
  const uint16_t bits = static_cast<uint16_t>(k16 ^ 0x8000u ^ (((~k16 >> 15) & 0x1u) * 0x7FFFu));
  const sycl::half h = sycl::bit_cast<sycl::half>(bits);
  return static_cast<float>(h);
}

// fp32 bounds {lo, hi} of coarse bin `bin`: "above" iff v >= hi, "tie" iff
// v >= lo and not above. Port of coarse_bin_bounds (scalar half conversions).
template <uint32_t kBits>
inline sycl::float2 coarse_bin_bounds(uint32_t bin) {
  constexpr uint32_t kShift = 16 - kBits;
  static_assert(kShift >= 1, "edge keys must be even for the rounding rule below");
  constexpr uint32_t kNegInfBin = 0x03FFu >> kShift;  // bin holding the -inf key
  constexpr uint32_t kInfBin = 0xFC00u >> kShift;     // bin holding the +inf key

  const uint32_t key_lo16 = bin << kShift;
  const uint32_t key_hi16 = (bin + 1) << kShift;

  const float v_lo = ordered16_to_fp32(key_lo16);
  const float v_hi = ordered16_to_fp32(key_hi16);
  const float u_lo = ordered16_to_fp32(key_lo16 - 1u);
  const float u_hi = ordered16_to_fp32(key_hi16 - 1u);

  const auto midpoint = [](float a, float b) { return sycl::fmin(sycl::fmax(0.5f * (a + b), -65520.0f), 65520.0f); };
  const int32_t lo = sycl::bit_cast<int32_t>(midpoint(v_lo, u_lo));
  const int32_t hi = sycl::bit_cast<int32_t>(midpoint(v_hi, u_hi));
  // Ties round to even and edge keys are even, so negative edges sit one ulp
  // above the midpoint. A zero hi steps past +0.
  const float lo_edge = sycl::bit_cast<float>(lo + (lo >> 31));
  const float hi_edge = sycl::bit_cast<float>(hi + (hi >> 31) + (hi == 0));
  const float nan = sycl::bit_cast<float>(0x7F800001);
  const float lo_final = bin <= kNegInfBin ? -infinity_value() : (bin > kInfBin ? nan : lo_edge);
  const float hi_final = bin >= kInfBin ? nan : hi_edge;
  return sycl::float2(lo_final, hi_final);
}

// ---------------------------------------------------------------------------
struct alignas(8) TieValue {
  float value;
  uint32_t idx;
  static inline constexpr TieValue invalid() {
    return TieValue{std::numeric_limits<float>::quiet_NaN(), 0xFFFFFFFFu};
  }
};

struct TopKProblem {
  const float* in;
  int32_t* out;  // page_indices [topk]
  uint32_t topk;
  uint32_t seq_len;
  int32_t bias = 0;
  uint32_t input_start = 0;  // ragged mode

  inline void emit(uint32_t pos, uint32_t raw_idx) const {
    out[pos] = static_cast<int32_t>(raw_idx) + bias;
  }
};

// ---------------------------------------------------------------------------
// TopKConfig: block/warp/tie sizing. kBlockSize=1024 so
// kNumWarps==kWarpSize==32 (PVC max work-group = 1024, sub_group = 32), keeping
// handle_tie's static assumption intact.
// ---------------------------------------------------------------------------
struct TopKConfig {
  static constexpr uint32_t kMaxTopK = 2048;
  static constexpr uint32_t kBlockSize = 1024;
  static constexpr uint32_t kNumWarps = kBlockSize / kWarpSize;  // 32
  static constexpr uint32_t kMaxNumTie = 2048;
  static constexpr uint32_t kRadixSize = 1 << 8;
  static constexpr uint32_t kRefineBits = 12;
  static constexpr uint32_t kRefineSize = 1 << kRefineBits;
  static constexpr uint32_t kRefineItems = kRefineSize / kBlockSize;
  static constexpr uint32_t kTopKItems = (kMaxTopK + kBlockSize - 1) / kBlockSize;
  static constexpr uint32_t kTieItems = kMaxNumTie / kBlockSize;
  static_assert(kMaxNumTie >= kMaxTopK && kMaxNumTie % kBlockSize == 0 && kBlockSize % kNumWarps == 0);
  static_assert(kNumWarps == kWarpSize, "handle_tie assumes kNumWarps==kWarpSize");

  struct TieHandleSmem {
    struct MatchBin {
      uint32_t bin;
      uint32_t above_count;
      uint32_t equal_count;
    };
    uint32_t counter;
    uint32_t counter_final;
    MatchBin match;
    uint32_t warp_sum[kNumWarps];
    uint32_t histogram[2][kRadixSize];
  };
};

}  // namespace dsa_topk
}  // namespace vllm
