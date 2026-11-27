// SPDX-License-Identifier: Apache-2.0
//
// Speculative-decoding verify: GatedDeltaNet + MoE layer, the split kernels (modes 1 / 2: one kernel a segment, see
// include/sgl_kernel/stage/xe20/args.hpp), bf16. Their own translation unit
// and library: next to the one-kernel layer they would share its helpers'
// instances, which then stay out of line and change its register allocation.
#include "launch.hpp"

namespace sgl_mono {
template __attribute__((visibility("default"))) bool
launch_gdn_verify_split<bf16>(sycl::queue&, int, int, const Ptrs&, const Scratch&, const ScratchB&, const Split&);
}  // namespace sgl_mono
