/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#pragma once

#include <cstdint>

namespace moe_w8a16 {

enum class ScaleMode : int32_t {
  ScalarSingle = 1,  // [E, 1] scalar scale per expert
  ScalarGateUp = 2,  // [E, 2] separate scalar scale for gate and up projections
  BlockFP32 = 3,     // [E, ceil(N/128), K/128] 128x128 FP32 block scales
  BlockMXFP8 = 4,    // [E, N, K/32] 1x32 UE8M0 uint8 block scales
};

inline bool is_block_scale_mode(int32_t mode) {
  return mode == static_cast<int32_t>(ScaleMode::BlockFP32) || mode == static_cast<int32_t>(ScaleMode::BlockMXFP8);
}

inline bool is_block_scale_mode(ScaleMode mode) {
  return is_block_scale_mode(static_cast<int32_t>(mode));
}

}  // namespace moe_w8a16
