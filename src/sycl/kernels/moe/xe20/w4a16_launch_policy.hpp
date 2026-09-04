#pragma once

// Framework launch-policy layer for the byte-identical fmha-cri W4A16 kernel.
//
// The copied kernel policy header deliberately contains only the upstream policy
// menu plus its generic w4a16_tile building block.  This file keeps production
// tile/scheduling choices in the framework adapter, so importing a new kernel
// revision never requires editing the authoritative kernel source.

#include "sycl/kernels/moe/xe20/w4a16/gemm_xe2_policy.hpp"

namespace moe_w4a16 {

template <class KernelPolicy, int StealChunk_, int PrefetchDist_, bool RowExtend_, bool SkipPaddedN_>
class w4a16_launch_policy : public KernelPolicy {
 public:
  static constexpr int StealChunk = StealChunk_;
  static constexpr int PrefetchDist = PrefetchDist_;
  static constexpr bool RowExtend = RowExtend_;
  static constexpr bool SkipPaddedN = SkipPaddedN_;
};

// Decode-size tiles use the shallow prefetch that fmha-cri measures for the
// small-M band.  The 32-row policy retains its imported split-barrier trait.
using w4a16_launch_policy_m_8_n_64 = w4a16_launch_policy<w4a16_policy_m_8_n_64, 1, 3, false, false>;
using w4a16_launch_policy_m_16_n_64 = w4a16_launch_policy<w4a16_policy_m_16_n_64, 1, 3, false, false>;
using w4a16_launch_policy_m_32_n_64 = w4a16_launch_policy<w4a16_policy_m_32_n_64, 1, 3, false, false>;

// Production prefill tiles from fmha-cri's registry.  SG_N=16 gives one DPAS
// N block per subgroup and avoids the ragged-M cost of multiple M subgroups.
using w4a16_launch_policy_m_64_n_128 = w4a16_launch_policy<w4a16_tile<64, 128, 1, 8>, 4, 2, true, false>;
using w4a16_launch_policy_m_64_n_128_skip = w4a16_launch_policy<w4a16_tile<64, 128, 1, 8>, 4, 2, true, true>;
using w4a16_launch_policy_m_64_n_256 = w4a16_launch_policy<w4a16_tile<64, 256, 1, 16>, 1, 2, true, false>;
using w4a16_launch_policy_m_64_n_256_skip = w4a16_launch_policy<w4a16_tile<64, 256, 1, 16>, 1, 2, true, true>;

}  // namespace moe_w4a16
