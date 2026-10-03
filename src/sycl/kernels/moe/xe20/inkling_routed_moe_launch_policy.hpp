#pragma once

#include "sycl/kernels/moe/xe20/w4a16_launch_policy.hpp"

namespace moe_w4a16 {

// Two 48-row tiles cover up to 96 rows without leaving a partly active
// 64-row tile in large down projections.
using w4a16_launch_policy_inkling_down_prefill_low =
    w4a16_expert_row_range_policy<w4a16_launch_policy<w4a16_tile<48, 128, 1, 8, 32, false>, 4, 2, true, false>, 1, 96>;
using w4a16_launch_policy_inkling_down_prefill_high =
    w4a16_expert_row_range_policy<w4a16_launch_policy<w4a16_tile<64, 128, 1, 8, 32, false>, 4, 2, true, false>, 97>;

}  // namespace moe_w4a16
