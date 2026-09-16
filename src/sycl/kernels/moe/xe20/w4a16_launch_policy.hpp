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

// N-width sweep around the 8-row decode tile, for the small-m band. All of them
// keep SG_N = BlkN/SgCountN = 16 and the StealChunk/PrefetchDist of
// w4a16_launch_policy_m_8_n_64, so the only variable is how many subgroups are
// bundled into one work-group: the per-subgroup tile, and therefore the number of
// 16-wide subgroup tiles the problem decomposes into, is the same in every one.
// Barrier is forced to true to match w4a16_policy_m_8_n_64, which inherits
// MainloopBarrier=true from xe_gemm_policy_base while w4a16_tile would default it
// to false here (w4a16_tile_wants_barrier(8, BlkN, 1) == false); leaving that to
// the default would vary the barrier and the N width at the same time. The
// _nobar entry varies the barrier alone, at the baseline's own width.
//
// RowExtend is on for 128 and 256 and off for 32 and 16, matching whether the
// width divides the decode N: GPT-OSS decode N = 1472 and 2880 are multiples of
// 16, 32 and 64 but not of 128 or 256, and the tile-aligned B height is what
// keeps a tail tile's 2D block load inside allocated memory (the same reason the
// m=64 wide-N entries below set it).
//
// Reachable only through SGL_W4A16_POLICY_ID (see GroupGemmW4A16Xe20.cpp):
// select_w4a16_policy_id() does not return these ids. What the sweep in
// benchmark/bench_moe_w4a16_policy_sweep.py measures on one B60, at the GPT-OSS
// 120b tp=4 decode shapes (gemm1 N=1472 K=2880, gemm2 N=2880 K=736) with 4
// routed experts and 1..8 rows each, is that N width is a ~4% knob at these
// sizes and not the ~2x the grid geometry might suggest: every one of these
// tiles decomposes the problem into the same number of 16-wide subgroup tiles
// (368 for gemm1, 720 for gemm2) and therefore fills the same 0.57/1.12 waves
// of the 640-subgroup persistent grid. Widening N while keeping the mainloop
// split barrier is a small regression (m_8_n_128 0.97x, m_8_n_256 0.97x on
// gemm1 at m=4); the variants that skip the barrier gain a little
// (m_8_n_128_skip 1.05x/1.06x on gemm1/gemm2, m_8_n_16 1.04x, m_8_n_64_nobar
// 1.04x/1.03x). m_8_n_128_skip is the only one that is also faster at
// 64 rows/expert (1.15x/1.09x), so it is the one worth revisiting as a default
// -- which needs a second device and src/jit/moe_jit.cpp's w4a16_policy() table
// kept in sync, so it is deliberately not done here.
using w4a16_launch_policy_m_8_n_128 = w4a16_launch_policy<w4a16_tile<8, 128, 1, 8, 32, true>, 1, 3, true, false>;
using w4a16_launch_policy_m_8_n_128_skip = w4a16_launch_policy<w4a16_tile<8, 128, 1, 8, 32, true>, 1, 3, true, true>;
using w4a16_launch_policy_m_8_n_256 = w4a16_launch_policy<w4a16_tile<8, 256, 1, 16, 32, true>, 1, 3, true, false>;
using w4a16_launch_policy_m_8_n_256_skip = w4a16_launch_policy<w4a16_tile<8, 256, 1, 16, 32, true>, 1, 3, true, true>;
using w4a16_launch_policy_m_8_n_32 = w4a16_launch_policy<w4a16_tile<8, 32, 1, 2, 32, true>, 1, 3, false, false>;
using w4a16_launch_policy_m_8_n_16 = w4a16_launch_policy<w4a16_tile<8, 16, 1, 1, 32, true>, 1, 3, false, false>;
using w4a16_launch_policy_m_8_n_64_nobar = w4a16_launch_policy<w4a16_tile<8, 64, 1, 4, 32, false>, 1, 3, false, false>;

// Production prefill tiles from fmha-cri's registry.  SG_N=16 gives one DPAS
// N block per subgroup and avoids the ragged-M cost of multiple M subgroups.
using w4a16_launch_policy_m_64_n_128 = w4a16_launch_policy<w4a16_tile<64, 128, 1, 8>, 4, 2, true, false>;
using w4a16_launch_policy_m_64_n_128_skip = w4a16_launch_policy<w4a16_tile<64, 128, 1, 8>, 4, 2, true, true>;
using w4a16_launch_policy_m_64_n_256 = w4a16_launch_policy<w4a16_tile<64, 256, 1, 16>, 1, 2, true, false>;
using w4a16_launch_policy_m_64_n_256_skip = w4a16_launch_policy<w4a16_tile<64, 256, 1, 16>, 1, 2, true, true>;

}  // namespace moe_w4a16
