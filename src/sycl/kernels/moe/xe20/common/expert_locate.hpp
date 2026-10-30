/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#pragma once

#include <algorithm>
#include <cstdint>
#include <cute/util/compat.hpp>
#include <sycl/sycl.hpp>

namespace moe_xe20 {

struct TileExpert {
  int expert_id;
  int m_tiles_before;  // m-tiles of all experts before expert_id
  int rows_before;     // rows of all experts before expert_id
  int rows;            // rows of expert_id
};

// Finds the expert that owns m-tile `m_tile` when the experts' rows are cut into TileM-row tiles in
// expert order. Every lane of the sub-group gets the same answer; returns false if m_tile is past the
// last tile. This replaces a serial walk over the experts, which costs tens of ns per expert, with
// one sub-group prefix sum per 128 experts: lane l owns experts [8 * l, 8 * l + 8) of each round.
template <int TileM, int SubgroupSize, typename SubGroup>
CUTE_DEVICE bool
locate_m_tile(SubGroup sg, const int32_t* rows_per_expert, int num_experts, int m_tile, TileExpert& out) {
  constexpr int PerLane = 8;
  constexpr int PerRound = SubgroupSize * PerLane;
  const int lane = static_cast<int>(sg.get_local_linear_id());

  int m_tiles_seen = 0;
  int lane_experts_before = 0;
  int lane_tiles_before = 0;
  int64_t lane_rows_before = 0;
  int lane_rows_here = 0;
  for (int round = 0; round < num_experts; round += PerRound) {
    int rows[PerLane];
    int lane_tiles = 0;
#pragma unroll
    for (int j = 0; j < PerLane; ++j) {
      int e = round + lane * PerLane + j;
      rows[j] = e < num_experts ? rows_per_expert[e] : 0;
      lane_tiles += (rows[j] + TileM - 1) / TileM;
    }
    int end = m_tiles_seen + sycl::exclusive_scan_over_group(sg, lane_tiles, sycl::plus<int>());
#pragma unroll
    for (int j = 0; j < PerLane; ++j) {
      int tiles = (rows[j] + TileM - 1) / TileM;
      int begin = end;
      end += tiles;
      bool before = end <= m_tile && round + lane * PerLane + j < num_experts;
      lane_experts_before += before ? 1 : 0;
      lane_tiles_before += before ? tiles : 0;
      lane_rows_before += before ? rows[j] : 0;
      lane_rows_here += (begin <= m_tile && m_tile < end) ? rows[j] : 0;
    }
    m_tiles_seen += sycl::reduce_over_group(sg, lane_tiles, sycl::plus<int>());
  }
  if (m_tile >= m_tiles_seen) {
    return false;
  }
  out.expert_id = sycl::reduce_over_group(sg, lane_experts_before, sycl::plus<int>());
  out.m_tiles_before = sycl::reduce_over_group(sg, lane_tiles_before, sycl::plus<int>());
  out.rows_before = static_cast<int>(sycl::reduce_over_group(sg, lane_rows_before, sycl::plus<int64_t>()));
  out.rows = sycl::reduce_over_group(sg, lane_rows_here, sycl::plus<int>());
  return true;
}

// Upper bound on sum_e ceil(rows_e / TileM): at most total_rows / TileM full tiles plus one partial
// tile per non-empty expert, and never more than one tile per row.
template <int TileM>
inline int64_t m_tiles_upper_bound(int64_t total_rows, int64_t num_experts) {
  return std::min<int64_t>(total_rows, total_rows / TileM + std::min<int64_t>(num_experts, total_rows));
}

}  // namespace moe_xe20
