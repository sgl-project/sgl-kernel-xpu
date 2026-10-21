#pragma once

#include <torch/all.h>

#include "sgl_kernel_export.h"

namespace at::native::xpu {
SGL_KERNEL_EXPORT void compute_position_xe35(
    torch::Tensor positions, torch::Tensor extend_start_loc, torch::Tensor prefix_lens, torch::Tensor seq_lens);
SGL_KERNEL_EXPORT void compute_position_with_hint_xe35(
    torch::Tensor positions,
    torch::Tensor extend_start_loc,
    torch::Tensor prefix_lens,
    torch::Tensor seq_lens,
    int64_t max_seq_len_hint);

}
