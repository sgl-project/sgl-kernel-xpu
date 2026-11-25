// SPDX-License-Identifier: Apache-2.0
//
// Full-attention + MoE decoder layer as one kernel (see attn.hpp,
// launch.hpp), bf16. Its own translation unit and library: the layer
// kernels in one device image run slower (code layout); so do the two dtypes.
#include "launch.hpp"

namespace sgl_mono {
template __attribute__((visibility("default"))) void
launch_attn<bf16>(sycl::queue&, int, const Ptrs&, const Scratch&, const ScratchB&);
}  // namespace sgl_mono
