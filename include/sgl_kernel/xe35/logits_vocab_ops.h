#pragma once

#include <torch/all.h>

#include "sgl_kernel_export.h"

namespace at::native::xpu {
SGL_KERNEL_EXPORT std::tuple<torch::Tensor, torch::Tensor> vocab_mask_xe35(
    torch::Tensor input,
    int64_t org_start,
    int64_t org_end,
    int64_t num_org_padding,
    int64_t added_start,
    int64_t added_end);
SGL_KERNEL_EXPORT void vocab_parallel_embedding_xe35(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    int64_t org_start,
    int64_t org_end,
    int64_t num_org_padding,
    int64_t added_start,
    int64_t added_end);
SGL_KERNEL_EXPORT void softcap_logits_xe35(torch::Tensor logits, double cap);

}
