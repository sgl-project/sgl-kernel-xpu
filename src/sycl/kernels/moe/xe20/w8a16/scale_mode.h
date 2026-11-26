/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#pragma once

#include <cstdint>

namespace moe_w8a16 {

enum class ScaleMode : int32_t {
  ScalarSingle = 1,  // [E] or [E, 1] per-expert scalar scale
  ScalarGateUp = 2,  // [E, 2] separate scalar scale for gate and up projections
  BlockFP32 = 3,     // [E, ceil(N/block_n), K/block_k] 2D block scales (power-of-two block_n)
  BlockMXFP8 = 4,    // [E, N, K/32] 1x32 UE8M0 uint8 block scales
  PerChannelN = 5,   // [E, N] or [E, N, 1] per-output-channel scales
  GroupNK = 6,       // [E, N, K/group_size] per-channel K-grouped scales
};

inline constexpr bool is_block_scale_mode(int32_t mode) {
  return mode == static_cast<int32_t>(ScaleMode::BlockFP32) || mode == static_cast<int32_t>(ScaleMode::BlockMXFP8) ||
         mode == static_cast<int32_t>(ScaleMode::GroupNK);
}

inline constexpr bool is_epilogue_scale_mode(int32_t mode) {
  return mode == static_cast<int32_t>(ScaleMode::ScalarSingle) ||
         mode == static_cast<int32_t>(ScaleMode::ScalarGateUp) || mode == static_cast<int32_t>(ScaleMode::PerChannelN);
}

inline bool is_block_scale_mode(ScaleMode mode) {
  return is_block_scale_mode(static_cast<int32_t>(mode));
}

}  // namespace moe_w8a16
