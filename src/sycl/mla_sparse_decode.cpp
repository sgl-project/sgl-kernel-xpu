/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
/*! \file
    \brief Sparse MLA decode dispatch interface for DeepSeek V4.
    Token-level scattered gather with dual KV cache pools + attn_sink.
*/
#define SYCL_INTEL_TARGET 20
#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <sycl/sycl.hpp>

#include "Utils.h"
#include "sycl/kernels/mla_sparse/device/mla_sparse_decode_dispatch.hpp"
#include "sycl/kernels/mla_sparse/device/mla_sparse_decode_types.hpp"

namespace {

#define DISPATCH_MLA_SPARSE_DTYPE()                                              \
  do {                                                                           \
    switch (in_dtype) {                                                          \
      case at::ScalarType::Half:                                                 \
        mla_sparse_decode::launch_mla_sparse_decode_half_128(                    \
            out,                                                                 \
            lse_out,                                                             \
            q,                                                                   \
            k_cache,                                                             \
            indices,                                                             \
            topk_length,                                                         \
            extra_k_cache,                                                       \
            extra_indices,                                                       \
            extra_topk_length,                                                   \
            attn_sink,                                                           \
            sm_scale,                                                            \
            head_dim_v,                                                          \
            is_fp8_kvcache);                                                     \
        break;                                                                   \
      case at::ScalarType::BFloat16:                                             \
        mla_sparse_decode::launch_mla_sparse_decode_bf16_128(                    \
            out,                                                                 \
            lse_out,                                                             \
            q,                                                                   \
            k_cache,                                                             \
            indices,                                                             \
            topk_length,                                                         \
            extra_k_cache,                                                       \
            extra_indices,                                                       \
            extra_topk_length,                                                   \
            attn_sink,                                                           \
            sm_scale,                                                            \
            head_dim_v,                                                          \
            is_fp8_kvcache);                                                     \
        break;                                                                   \
      default:                                                                   \
        TORCH_CHECK(false, "Unsupported input data type for Sparse MLA decode"); \
    }                                                                            \
  } while (0)

}  // namespace

/// @brief Dispatch kernel implementation for V4 Sparse MLA decode.
void flash_mla_sparse_decode(
    at::Tensor& out,                                     // [B, 1, H, head_dim_v]
    at::Tensor& lse_out,                                 // [B, H, 1]
    const at::Tensor& q,                                 // [B, 1, H, D_qk]
    const at::Tensor& k_cache,                           // [num_pages, page_size, 1, D]
    const at::Tensor& indices,                           // [B, 1, topk]
    const std::optional<at::Tensor>& topk_length,        // [B] or nullopt
    const std::optional<at::Tensor>& extra_k_cache,      // [num_ext_pg, page_size, 1, D] or nullopt
    const std::optional<at::Tensor>& extra_indices,      // [B, 1, extra_topk] or nullopt
    const std::optional<at::Tensor>& extra_topk_length,  // [B] or nullopt
    const std::optional<at::Tensor>& attn_sink,          // [H] or nullopt
    double sm_scale,
    int64_t head_dim_v,
    bool is_fp8_kvcache = false) {
  CHECK_INPUT(out);
  CHECK_INPUT(lse_out);
  CHECK_INPUT(q);
  // k_cache may be non-contiguous (FP8 packed uses as_strided with custom stride(0))
  CHECK_DEVICE(k_cache);
  CHECK_INPUT(indices);

  int page_size = k_cache.size(1);

  c10::DeviceGuard guard(q.device());

  auto in_dtype = q.scalar_type();

  TORCH_CHECK(
      in_dtype == at::ScalarType::Half || in_dtype == at::ScalarType::BFloat16,
      "Unsupported input data type for Sparse MLA decode");
  TORCH_CHECK(head_dim_v == 512, "head_dim_v must be 512 for DeepSeek V4 MLA");

  DISPATCH_MLA_SPARSE_DTYPE();
}

#undef DISPATCH_MLA_SPARSE_DTYPE
#undef SYCL_INTEL_TARGET
