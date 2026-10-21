#pragma once

#include <torch/types.h>

#include "sgl_kernel_export.h"

SGL_KERNEL_EXPORT void
bf16_gemm_nt_xpu_xe35(torch::Tensor& output, const torch::Tensor& mat_a, const torch::Tensor& mat_b, int64_t config);
