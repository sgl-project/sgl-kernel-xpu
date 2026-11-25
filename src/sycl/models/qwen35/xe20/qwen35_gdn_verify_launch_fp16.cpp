// SPDX-License-Identifier: Apache-2.0
//
// Speculative-decoding verify: Linear-attention (GatedDeltaNet) + MoE layer for NB = requests x T
// tokens (see gdn.hpp), fp16. Its own translation unit and library,
// like the decode kernels.
#include "launch.hpp"

namespace sgl_mono {
template __attribute__((visibility("default"))) bool
launch_gdn_verify<half>(sycl::queue&, int, int, const Ptrs&, const Scratch&, const ScratchB&);
}  // namespace sgl_mono
