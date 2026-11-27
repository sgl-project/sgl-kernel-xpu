// SPDX-License-Identifier: Apache-2.0
//
// The MTP prologue launcher for GPUs with fewer than 32 Xe cores, bf16: its own
// translation unit and library (see launch.hpp in this directory).
#include "mtp.hpp"

namespace sgl_mono {
template __attribute__((visibility("default"))) bool
b60_launch_mtp<bf16>(sycl::queue&, int, int, const Ptrs&, const Scratch&, int);
}  // namespace sgl_mono
