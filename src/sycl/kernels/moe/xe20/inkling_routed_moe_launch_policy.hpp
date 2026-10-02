#pragma once

#include "sycl/kernels/moe/xe20/w4a16_launch_policy.hpp"

namespace moe_w4a16 {

using w4a16_launch_policy_inkling_m_8_n_64 = w4a16_launch_policy<w4a16_tile<8, 64, 1, 4, 32, false>, 1, 3, true, false>;
using w4a16_launch_policy_inkling_m_16_n_64 =
    w4a16_launch_policy<w4a16_tile<16, 64, 1, 4, 32, false>, 1, 3, false, false>;
using w4a16_launch_policy_inkling_down_m_32_n_256 =
    w4a16_launch_policy<w4a16_tile<32, 256, 1, 16, 32, false>, 1, 2, true, false>;

// Two 48-row tiles cover up to 96 rows without leaving a partly active
// 64-row tile. Larger experts use the higher-throughput 64-row policy.
using w4a16_launch_policy_inkling_gate_prefill_low =
    w4a16_expert_row_range_policy<w4a16_launch_policy<w4a16_tile<48, 256, 1, 16, 32, false>, 1, 2, true, false>, 1, 96>;
using w4a16_launch_policy_inkling_gate_prefill_high =
    w4a16_expert_row_range_policy<w4a16_launch_policy<w4a16_tile<64, 256, 1, 16, 32, false>, 1, 2, true, false>, 97>;
using w4a16_launch_policy_inkling_down_prefill_low =
    w4a16_expert_row_range_policy<w4a16_launch_policy<w4a16_tile<48, 128, 1, 8, 32, false>, 4, 2, true, false>, 1, 96>;
using w4a16_launch_policy_inkling_down_prefill_high =
    w4a16_expert_row_range_policy<w4a16_launch_policy<w4a16_tile<64, 128, 1, 8, 32, false>, 4, 2, true, false>, 97>;

}  // namespace moe_w4a16
