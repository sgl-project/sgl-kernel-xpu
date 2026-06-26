// Fused Gated-DeltaNet (GDN) attention op for Intel Xe2 (BMG).
// Exposes the host entry `gdn_attention(...)`, which is registered as
// torch.ops.sgl_kernel.gdn_attention.
//
// This translation unit compiles the interface + recurrent (decode) path and
// the Xe2 causal-conv1d kernels. The Xe2 chunk delta-rule and l2norm kernels are
// compiled as separate TUs (see GdnAttnXe20.cmake).
#define SYCL_INTEL_TARGET 20

#include "kernels/gdn_attn/gdn_attn_interface_impl.hpp"
