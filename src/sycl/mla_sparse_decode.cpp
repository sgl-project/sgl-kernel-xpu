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
#include "sgl_kernel_export.h"
#include "sycl/kernels/mla_sparse/device/mla_sparse_decode_dispatch.hpp"
#include "sycl/kernels/mla_sparse/device/mla_sparse_decode_types.hpp"
#ifdef USE_MLA_JIT
#include "jit/mla_jit.h"
#endif

// Compile-time toggle for the two-stage sparse MLA decode path (gather+dequant to
// HBM, then dense flash-decode). The selector macro
// SGLANG_USE_SPARSE_MLA_2STAGE is defined (default 1) in mla_sparse_decode_types.hpp
// below; set it to 0 there for the fused path, or override at build time with
// -DSGLANG_USE_SPARSE_MLA_2STAGE=<0|1>. Compile-time A/B toggle in the
// SGL_DISABLE_PACKGQA style. The name follows the env-var-conventions naming rule
// (SGLANG_ prefix + USE_ verb for an implementation selector).

namespace {

#define DISPATCH_MLA_SPARSE_DTYPE()                                              \
  do {                                                                           \
    switch (in_dtype) {                                                          \
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

// Two-stage dispatch ladder. Each rung resolves ONE runtime value to a compile-time
// token and is named for the value it switches on (like DISPATCH_MLA_SPARSE_DTYPE):
//
//   DISPATCH_MLA_SPARSE_DTYPE_2STAGE -> ELEM  (in_dtype; bf16 only)
//     DISPATCH_MLA_SPARSE_D_QK       -> D_QK  (q.size(3); 512 only for decode)
//       DISPATCH_MLA_SPARSE_B_H      -> B_H   (select_b_h(h_q); 8/16/32/64)
//         DISPATCH_MLA_SPARSE_SINK   -> SINK  (attn_sink.has_value(); 0/1)
//           DISPATCH_MLA_SPARSE_LAUNCH_2STAGE -> the generated launcher call
//
// The switches are load-bearing, not stylistic: the leaf pastes these four tokens into
// the launcher's name, so each value must be a literal before it is reached. Full
// expansion is 1 D_QK x 4 B_H x 2 SINK = 8 call sites, which is exactly the symbol set
// MlaSparseDecodeXe20.cmake generates and mla_sparse_decode_dispatch.hpp declares --
// the three must stay in lockstep or the TU fails to link.
#define DISPATCH_MLA_SPARSE_LAUNCH_2STAGE(ELEM, D_QK, B_H, SINK)                       \
  mla_sparse_decode::launch_mla_sparse_decode_2stage_##ELEM##_##D_QK##_##B_H##_##SINK( \
      out,                                                                             \
      lse_out,                                                                         \
      q,                                                                               \
      k_cache,                                                                         \
      indices,                                                                         \
      topk_length,                                                                     \
      extra_k_cache,                                                                   \
      extra_indices,                                                                   \
      extra_topk_length,                                                               \
      attn_sink,                                                                       \
      sm_scale,                                                                        \
      head_dim_v,                                                                      \
      is_fp8_kvcache)

// Resolve the runtime attn_sink flag to the compile-time 0/1 launcher variant.
#define DISPATCH_MLA_SPARSE_SINK(ELEM, D_QK, B_H)            \
  do {                                                       \
    if (attn_sink.has_value()) {                             \
      DISPATCH_MLA_SPARSE_LAUNCH_2STAGE(ELEM, D_QK, B_H, 1); \
    } else {                                                 \
      DISPATCH_MLA_SPARSE_LAUNCH_2STAGE(ELEM, D_QK, B_H, 0); \
    }                                                        \
  } while (0)

// Resolve the head-block size B_H (the number of query heads packed into one Stage-2
// tile) for this h_q, threading the already-resolved D_QK through untouched.
// sparse_mla_decode_select_b_h returns an int, which cannot be pasted into a symbol
// name -- this switch is what turns it into one of four literal call sites. Its
// default arm and that function's fallthrough must agree on 64.
#define DISPATCH_MLA_SPARSE_B_H(ELEM, D_QK)                               \
  do {                                                                    \
    switch (mla_sparse_decode::sparse_mla_decode_select_b_h(q.size(2))) { \
      case 8:                                                             \
        DISPATCH_MLA_SPARSE_SINK(ELEM, D_QK, 8);                          \
        break;                                                            \
      case 16:                                                            \
        DISPATCH_MLA_SPARSE_SINK(ELEM, D_QK, 16);                         \
        break;                                                            \
      case 32:                                                            \
        DISPATCH_MLA_SPARSE_SINK(ELEM, D_QK, 32);                         \
        break;                                                            \
      default:                                                            \
        DISPATCH_MLA_SPARSE_SINK(ELEM, D_QK, 64);                         \
        break;                                                            \
    }                                                                     \
  } while (0)

// Decode currently only supports d_qk == 512; resolve the runtime value to the
// compile-time D_QK launcher variant, then dispatch B_H. Structured as a switch (like
// the prefill path's {512, 576}) so a second d_qk can be added without reshaping the
// dispatch -- add a case here + the D_QK to MlaSparseDecodeXe20.cmake / the dispatch
// header declarations.
#define DISPATCH_MLA_SPARSE_D_QK(ELEM)                                                               \
  do {                                                                                               \
    switch (q.size(3)) {                                                                             \
      case 512:                                                                                      \
        DISPATCH_MLA_SPARSE_B_H(ELEM, 512);                                                          \
        break;                                                                                       \
      default:                                                                                       \
        TORCH_CHECK(false, "Unsupported d_qk for Sparse MLA decode (must be 512), got ", q.size(3)); \
    }                                                                                                \
  } while (0)

// bf16 only for now: the 2-stage Stage-2 QK DPAS is bf16 (K/V are the gathered bf16
// latent).
#define DISPATCH_MLA_SPARSE_DTYPE_2STAGE()                                                      \
  do {                                                                                          \
    switch (in_dtype) {                                                                         \
      case at::ScalarType::BFloat16:                                                            \
        DISPATCH_MLA_SPARSE_D_QK(bf16);                                                         \
        break;                                                                                  \
      default:                                                                                  \
        TORCH_CHECK(false, "2-stage Sparse MLA decode currently supports only bfloat16 query"); \
    }                                                                                           \
  } while (0)

}  // namespace

/// @brief Dispatch kernel implementation for V4 Sparse MLA decode.
SGL_KERNEL_EXPORT void flash_mla_sparse_decode(
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
  TORCH_CHECK(q.dim() == 4 && q.size(1) == 1, "q must have shape [B, 1, H, D_qk] (decode-only)");
  TORCH_CHECK(
      indices.dim() == 3 && indices.size(0) == q.size(0) && indices.size(1) == 1,
      "indices must have shape [B, 1, topk]");
  TORCH_CHECK(k_cache.dim() == 4 && k_cache.size(2) == 1, "k_cache must have shape [num_pages, page_size, 1, D]");
  TORCH_CHECK(
      k_cache.scalar_type() == at::ScalarType::Float8_e4m3fn && k_cache.size(3) == 584,
      "k_cache must use the DeepSeek V4 FP8 packed layout: dtype=float8_e4m3fn, last_dim=584");
  TORCH_CHECK((q.size(2) % 8) == 0, "num_heads must be a multiple of 8 (kernel fuses 8 heads per workgroup)");
  TORCH_CHECK(
      (!extra_k_cache.has_value() && !extra_indices.has_value()) ||
          (extra_k_cache.has_value() && extra_indices.has_value()),
      "extra_k_cache and extra_indices must be provided together");
  TORCH_CHECK(in_dtype == at::ScalarType::BFloat16, "Unsupported input data type for Sparse MLA decode");
  TORCH_CHECK(head_dim_v == 512, "head_dim_v must be 512 for DeepSeek V4 MLA");

// The JIT path only covers the 2-stage template (the fused template has no
// SGL_MLA_JIT_ENTRY). When the fused path is selected
// (SGLANG_USE_SPARSE_MLA_2STAGE=0), fall through to the AOT dispatch below so
// the compile-time A/B toggle stays authoritative on both paths.
#if defined(USE_MLA_JIT) && SGLANG_USE_SPARSE_MLA_2STAGE
  {
    const int d_qk = static_cast<int>(q.size(3));
    const int b_h = mla_sparse_decode::sparse_mla_decode_select_b_h(q.size(2));
    std::string jit_err;
    TORCH_CHECK(
        sgl::mla_jit::sparse_decode_launch(
            in_dtype == at::ScalarType::Half,
            d_qk,
            b_h,
            attn_sink.has_value(),
            &out,
            &lse_out,
            &q,
            &k_cache,
            &indices,
            &topk_length,
            &extra_k_cache,
            &extra_indices,
            &extra_topk_length,
            &attn_sink,
            sm_scale,
            head_dim_v,
            is_fp8_kvcache,
            jit_arch_code(),
            &jit_err),
        jit_err);
  }
#else
#if SGLANG_USE_SPARSE_MLA_2STAGE
  DISPATCH_MLA_SPARSE_DTYPE_2STAGE();
#else
#ifndef USE_MLA_SPARSE_FUSED
#error \
    "Fused sparse MLA decode selected (SGLANG_USE_SPARSE_MLA_2STAGE=0) but the fused kernel was not built. Reconfigure with -DUSE_MLA_SPARSE_FUSED=ON (or USE_MLA_SPARSE_FUSED=1)."
#endif
  DISPATCH_MLA_SPARSE_DTYPE();
#endif
#endif
}

#undef DISPATCH_MLA_SPARSE_DTYPE
#undef DISPATCH_MLA_SPARSE_DTYPE_2STAGE
#undef DISPATCH_MLA_SPARSE_D_QK
#undef DISPATCH_MLA_SPARSE_B_H
#undef DISPATCH_MLA_SPARSE_SINK
#undef DISPATCH_MLA_SPARSE_LAUNCH_2STAGE
#undef SYCL_INTEL_TARGET
