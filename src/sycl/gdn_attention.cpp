// Fused Gated-DeltaNet (GDN) attention op for Intel GPU.
// Exposes the host entry `gdn_attention(...)`, which is registered as
// torch.ops.sgl_kernel.gdn_attention.
//
// This translation unit compiles the interface + recurrent (decode) path and
// the causal-conv1d kernels. The chunk delta-rule and l2norm kernels are
// compiled as separate TUs (see GdnAttn.cmake).

#include "gdn_attn/gdn_attn_interface_impl.hpp"
