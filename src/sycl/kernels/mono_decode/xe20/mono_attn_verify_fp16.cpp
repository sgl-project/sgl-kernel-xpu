// SPDX-License-Identifier: Apache-2.0
//
// Speculative-decoding verify: Full-attention + MoE layer for NB = requests x T
// tokens (see mono_layers_s.hpp), fp16. Its own translation unit and library,
// like the decode kernels.
#include "mono_layers_s.hpp"

namespace sgl_mono {
template __attribute__((visibility("default"))) bool
launch_attn_verify<half>(sycl::queue&, int, int, const Ptrs&, const Scratch&, const ScratchB&);
}  // namespace sgl_mono
