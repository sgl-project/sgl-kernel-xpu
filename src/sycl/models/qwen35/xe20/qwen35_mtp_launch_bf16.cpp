// SPDX-License-Identifier: Apache-2.0
//
// MTP drafter prologue (see mtp.hpp), bf16. Its own translation unit and
// library, like the layer kernels.
#include "mtp.hpp"

namespace sgl_mono {
template __attribute__((visibility("default"))) bool
launch_mtp<bf16>(sycl::queue&, int, const Ptrs&, const Scratch&, int);
}  // namespace sgl_mono
