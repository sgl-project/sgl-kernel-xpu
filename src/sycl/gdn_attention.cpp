// Fused Gated-DeltaNet (GDN) attention op for Intel GPU.
// Exposes the host entry `gdn_attention(...)`, which is registered as
// torch.ops.sgl_kernel.gdn_attention.
//
// This translation unit compiles the interface, recurrent (decode) path,
// causal-conv1d kernels, and l2norm. Only the chunk delta-rule dispatcher is
// compiled as a separate TU; see src/CMakeLists.txt and src/BuildOnLinux.cmake.

#include "gdn_attn/gdn_attn_interface_impl.hpp"
