// SPDX-License-Identifier: Apache-2.0
//
// The attention verify launcher for GPUs with fewer than 32 Xe cores, fp16: its own
// translation unit and library (see launch.hpp in this directory).
#include "launch.hpp"

namespace sgl_mono {
template __attribute__((visibility("default"))) bool b60_launch_attn_verify<sycl::half>(
    sycl::queue&, int, int, int, const Ptrs&, const Scratch&, const ScratchB&, const Split&, const Segs&);
}  // namespace sgl_mono
