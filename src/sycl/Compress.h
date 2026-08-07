#pragma once

#include <cstdint>
#include <utility>

namespace at::native::xpu {

struct alignas(16) DecodePlan {
  uint32_t seq_len;
  int32_t write_loc;
  int32_t read_page_0;
  int32_t read_page_1;
};

struct alignas(16) CompressPlan {
  uint32_t seq_len;
  uint16_t ragged_id;
  uint16_t buffer_len;
  int32_t read_page_0;
  int32_t read_page_1;

  static constexpr CompressPlan invalid() {
    return CompressPlan{-1u, 0, 0, -1, -1};
  }

  constexpr bool is_invalid() const {
    return seq_len == -1u;
  }
};

struct alignas(8) WritePlan {
  uint32_t ragged_id;
  int32_t write_loc;

  static constexpr WritePlan invalid() {
    return WritePlan{-1u, -1};
  }

  constexpr bool is_invalid() const {
    return ragged_id == -1u;
  }
};

inline WritePlan pack_w(uint32_t ragged_id, uint32_t batch_id, int32_t seq_len) {
  return {static_cast<uint32_t>(ragged_id | (batch_id << 16)), seq_len};
}

inline std::pair<uint16_t, uint16_t> unpack_w(WritePlan plan) {
  return {static_cast<uint16_t>(plan.ragged_id), static_cast<uint16_t>(plan.ragged_id >> 16)};
}

static_assert(sizeof(DecodePlan) == 16, "DecodePlan must be 16 bytes");
static_assert(sizeof(CompressPlan) == 16, "CompressPlan must be 16 bytes");
static_assert(sizeof(WritePlan) == 8, "WritePlan must be 8 bytes");

}  // namespace at::native::xpu
