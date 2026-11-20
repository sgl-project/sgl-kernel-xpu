// SPDX-License-Identifier: Apache-2.0
#include "DiffusionAttentionXe20.h"

#include <torch/library.h>

namespace {
at::Tensor diffusion_sdp(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {
  using namespace sgl_kernel::diffusion::detail;
  switch (q.scalar_type()) {
    case at::kHalf:
      return sdp_fp16(q, k, v);
    case at::kBFloat16:
      return sdp_bf16(q, k, v);
    default:
      TORCH_CHECK(false, "diffusion_sdp: only fp16/bf16 supported");
  }
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def("diffusion_sdp(Tensor q, Tensor k, Tensor v) -> Tensor");
}
TORCH_LIBRARY_IMPL(sgl_kernel, XPU, m) {
  m.impl("diffusion_sdp", &diffusion_sdp);
}
