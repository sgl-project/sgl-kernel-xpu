// SPDX-License-Identifier: Apache-2.0
//
// Linear-attention (GatedDeltaNet) + MoE decoder layer as one kernel (see mono_layers.hpp,
// mono_layers_b.hpp), bf16. Its own translation unit and library: the layer
// kernels in one device image run slower (code layout); so do the two dtypes.
#include "mono_layers_b.hpp"

namespace sgl_mono {
template __attribute__((visibility("default"))) void
launch_gdn<bf16>(sycl::queue&, int, const Ptrs&, const Scratch&, const ScratchB&);
}  // namespace sgl_mono
