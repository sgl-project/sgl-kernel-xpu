// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>

#include "sgl_kernel/stage/xe20/common.hpp"

namespace sgl_kernel::minicpm5::xe20 {
struct Shape {
  static constexpr int hidden = 2048;
  static constexpr int experts = 160;
  static constexpr int topk = 16;
  static constexpr int head_dim = 128;
  static constexpr int query_heads = 32;
  static constexpr int kv_heads = 2;
  static constexpr int dense_intermediate = 8192;
  static constexpr int expert_intermediate = 512;
  static constexpr int max_rows = 8;
  static constexpr int draft_rows = 7;
  static constexpr int phases = 16;
  static constexpr int control_words = phases * 16 + 16 + max_rows * 4;
  static constexpr int tp2_control_banks = 1 + 3 * max_rows;
  static constexpr std::size_t mailbox_rows = max_rows * (max_rows + 1) / 2;
  static constexpr std::size_t mailbox_words = 2 * mailbox_rows * hidden;
};

using B70 = stage::xe20::Launch<32, 512, Shape::max_rows>;
// Flat, evenly divisible hidden-state tiles. The B60 has 20 Xe cores,
// so all 16 groups in this functional variant can be resident.
using B60 = stage::xe20::Launch<16, 512, Shape::max_rows>;
}  // namespace sgl_kernel::minicpm5::xe20
