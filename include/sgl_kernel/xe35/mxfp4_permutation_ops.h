#pragma once

#include <torch/types.h>

#include "sgl_kernel_export.h"

namespace at::native::xpu {
SGL_KERNEL_EXPORT void mxfp4_small_permutation_xe35(
    torch::Tensor topk_ids,
    torch::Tensor padded_counts,
    torch::Tensor perm_src,
    torch::Tensor expanded_to_permuted,
    torch::Tensor scale_map,
    int64_t alignment);
}
