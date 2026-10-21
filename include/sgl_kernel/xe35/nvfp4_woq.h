#pragma once

#include <torch/torch.h>
#include <sgl_kernel_export.h>

namespace at::native::xpu {
SGL_KERNEL_EXPORT void nvfp4_woq_prep_xe35(torch::Tensor& out_c, torch::Tensor& out_sfmx, const torch::Tensor& w_s, const torch::Tensor& w_gs);
}
SGL_KERNEL_EXPORT void nvfp4_woq_bf16_grouped_mm_xe35(
    torch::Tensor& output, const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& sfmx_b, const torch::Tensor& c_b,
    const torch::Tensor& problem_sizes, const torch::Tensor& expert_offsets, const torch::Tensor& workspace);
SGL_KERNEL_EXPORT void nvfp4_woq_mxfp8_scaled_grouped_mm_xe35(
    torch::Tensor& output, const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& scales_a, const torch::Tensor& sfmx_b, const torch::Tensor& c_b,
    const torch::Tensor& problem_sizes, const torch::Tensor& expert_offsets, const torch::Tensor& workspace);
SGL_KERNEL_EXPORT void mxfp4_moe_scale_to_mn_major_xe35(
    torch::Tensor& out, const torch::Tensor& in, const torch::Tensor& rows_per_expert);
