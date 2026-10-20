#pragma once

#include <torch/all.h>

#include "sgl_kernel_export.h"

namespace at::native::xpu {
SGL_KERNEL_EXPORT void write_req_to_token_pool_xe35(
    torch::Tensor req_to_token,
    torch::Tensor req_pool_indices,
    torch::Tensor prefix_pointers,
    torch::Tensor prefix_lens,
    torch::Tensor seq_lens,
    torch::Tensor extend_lens,
    torch::Tensor out_cache_loc);
SGL_KERNEL_EXPORT void paged_alloc_extend_xe35(
    torch::Tensor prefix_lens,
    torch::Tensor seq_lens,
    torch::Tensor last_loc,
    torch::Tensor free_pages,
    torch::Tensor out_indices,
    int64_t page_size);
SGL_KERNEL_EXPORT void paged_alloc_extend_and_write_xe35(
    torch::Tensor prefix_lens,
    torch::Tensor seq_lens,
    torch::Tensor last_loc,
    torch::Tensor free_pages,
    torch::Tensor out_indices,
    torch::Tensor req_to_token,
    torch::Tensor req_pool_indices,
    torch::Tensor prefix_pointers,
    int64_t page_size);
SGL_KERNEL_EXPORT void paged_alloc_decode_xe35(
    torch::Tensor seq_lens,
    torch::Tensor last_loc,
    torch::Tensor free_pages,
    torch::Tensor out_indices,
    int64_t page_size);

}
