// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <ATen/ATen.h>

namespace sgl_kernel::diffusion::detail {
at::Tensor sdp_fp16(const at::Tensor&, const at::Tensor&, const at::Tensor&);
at::Tensor sdp_bf16(const at::Tensor&, const at::Tensor&, const at::Tensor&);
}  // namespace sgl_kernel::diffusion::detail
