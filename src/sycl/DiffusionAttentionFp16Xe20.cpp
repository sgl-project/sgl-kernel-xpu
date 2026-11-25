// SPDX-License-Identifier: Apache-2.0
// Compile the FP16 attention implementation.
#define SGLD_KERNEL_TAG CuteFmhaKernelTagFp16
#define SGLD_ATTENTION_BF16 0
#define SGLD_ATTENTION_TWO_EDIT 1
#include "DiffusionAttentionXe20.h"
#include "DiffusionAttentionXe20Impl.inc"

namespace sgl_kernel::diffusion::detail {
at::Tensor sdp_fp16(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {
  return diffusion_sdp(q, k, v);
}
}  // namespace sgl_kernel::diffusion::detail
