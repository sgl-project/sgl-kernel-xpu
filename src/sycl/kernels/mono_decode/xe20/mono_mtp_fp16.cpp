// SPDX-License-Identifier: Apache-2.0
//
// MTP drafter prologue (see mono_mtp.hpp), fp16. Its own translation unit and
// library, like the layer kernels.
#include "mono_mtp.hpp"

namespace sgl_mono {
template __attribute__((visibility("default"))) bool
launch_mtp<half>(sycl::queue&, int, const Ptrs&, const Scratch&, int);
}  // namespace sgl_mono
