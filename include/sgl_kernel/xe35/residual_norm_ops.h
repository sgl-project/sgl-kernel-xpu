#pragma once

#include <optional>
#include <torch/types.h>
#include "sgl_kernel_export.h"

#if SYCL_INTEL_TARGET == 35
namespace at::native::xpu {
SGL_KERNEL_EXPORT void fused_post_norm_residual_pre_norm_xe35(
    torch::Tensor branch,
    torch::Tensor stream,
    torch::Tensor post_weight,
    torch::Tensor pre_weight,
    torch::Tensor alpha,
    torch::Tensor beta,
    torch::Tensor output,
    torch::Tensor new_stream,
    double post_eps,
    double pre_eps,
    bool post_norm_input_scale,
    bool round_pre_norm_input,
    const std::optional<torch::Tensor>& output_fp32);
}
#endif
