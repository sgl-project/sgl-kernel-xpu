// SPDX-License-Identifier: Apache-2.0
//
// The GatedDeltaNet decode launcher for GPUs with fewer than 32 Xe cores, bf16: its own
// translation unit and library (see launch.hpp in this directory).
#include "launch.hpp"

namespace sgl_mono {
template __attribute__((visibility("default"))) bool
b60_launch_gdn<bf16>(sycl::queue&, int, int, const Ptrs&, const Scratch&, const ScratchB&, const Split&, const Segs&);
}  // namespace sgl_mono
