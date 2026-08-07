/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
/*! \file
    \brief Sparse MLA prefill dispatch interface for DeepSeek V4.

    Two-stage sparse attention prefill: Stage 1 gathers the dense bf16 KV rows named
    by `indices` into a dense HBM tile; Stage 2 runs the shared dense flash kernel
    (reused from the 2-stage decode path) over that tile, emitting out / max_logits /
    lse. Each query row is mapped to a decode "batch" (see
    device/mla_sparse_prefill_2stage_types.hpp).
*/
#define SYCL_INTEL_TARGET 20
#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <sycl/sycl.hpp>

#include "Utils.h"
#include "sgl_kernel_export.h"
#include "sycl/kernels/mla_sparse/device/mla_sparse_prefill_dispatch.hpp"

namespace {

// Two-stage dispatch ladder. Each rung resolves ONE runtime value to a compile-time
// token and is named for the value it switches on (like DISPATCH_..._DTYPE_2STAGE):
//
//   DISPATCH_MLA_SPARSE_PREFILL_DTYPE_2STAGE -> ELEM  (in_dtype; bf16 only)
//     DISPATCH_MLA_SPARSE_PREFILL_D_QK       -> D_QK  (q.size(2); 512 or 576)
//       DISPATCH_MLA_SPARSE_PREFILL_B_H      -> B_H   (select_b_h(h_q); 8/16/32/64)
//         DISPATCH_MLA_SPARSE_PREFILL_SINK   -> SINK  (attn_sink.has_value(); 0/1)
//           DISPATCH_MLA_SPARSE_PREFILL_LAUNCH_2STAGE -> the generated launcher call
//
// The switches are load-bearing, not stylistic: the leaf pastes these four tokens into
// the launcher's name, so each value must be a literal before it is reached. Full
// expansion is 2 D_QK x 4 B_H x 2 SINK = 16 call sites, which is exactly the symbol set
// MlaSparsePrefillXe20.cmake generates and mla_sparse_prefill_dispatch.hpp declares --
// the three must stay in lockstep or the TU fails to link.
#define DISPATCH_MLA_SPARSE_PREFILL_LAUNCH_2STAGE(ELEM, D_QK, B_H, SINK)                 \
  mla_sparse_prefill::launch_mla_sparse_prefill_2stage_##ELEM##_##D_QK##_##B_H##_##SINK( \
      out, max_logits, lse, q, kv, indices, attn_sink, topk_length, sm_scale, head_dim_v)

// Resolve the runtime attn_sink flag to the compile-time 0/1 launcher variant.
#define DISPATCH_MLA_SPARSE_PREFILL_SINK(ELEM, D_QK, B_H)            \
  do {                                                               \
    if (attn_sink.has_value()) {                                     \
      DISPATCH_MLA_SPARSE_PREFILL_LAUNCH_2STAGE(ELEM, D_QK, B_H, 1); \
    } else {                                                         \
      DISPATCH_MLA_SPARSE_PREFILL_LAUNCH_2STAGE(ELEM, D_QK, B_H, 0); \
    }                                                                \
  } while (0)

// Resolve the head-block size B_H (the number of query heads packed into one Stage-2
// tile) for this h_q, threading the already-resolved D_QK through untouched.
// sparse_mla_prefill_select_b_h returns an int, which cannot be pasted into a symbol
// name -- this switch is what turns it into one of four literal call sites. Its
// default arm and that function's fallthrough must agree on 64.
#define DISPATCH_MLA_SPARSE_PREFILL_B_H(ELEM, D_QK)                         \
  do {                                                                      \
    switch (mla_sparse_prefill::sparse_mla_prefill_select_b_h(q.size(1))) { \
      case 8:                                                               \
        DISPATCH_MLA_SPARSE_PREFILL_SINK(ELEM, D_QK, 8);                    \
        break;                                                              \
      case 32:                                                              \
        DISPATCH_MLA_SPARSE_PREFILL_SINK(ELEM, D_QK, 32);                   \
        break;                                                              \
      default:                                                              \
        DISPATCH_MLA_SPARSE_PREFILL_SINK(ELEM, D_QK, 64);                   \
        break;                                                              \
    }                                                                       \
  } while (0)

// Prefill supports d_qk in {512, 576}; resolve the runtime value to the compile-time
// D_QK launcher variant, then dispatch B_H.
#define DISPATCH_MLA_SPARSE_PREFILL_D_QK(ELEM)                                                               \
  do {                                                                                                       \
    switch (q.size(2)) {                                                                                     \
      case 512:                                                                                              \
        DISPATCH_MLA_SPARSE_PREFILL_B_H(ELEM, 512);                                                          \
        break;                                                                                               \
      case 576:                                                                                              \
        DISPATCH_MLA_SPARSE_PREFILL_B_H(ELEM, 576);                                                          \
        break;                                                                                               \
      default:                                                                                               \
        TORCH_CHECK(false, "Unsupported d_qk for Sparse MLA prefill (must be 512 or 576), got ", q.size(2)); \
    }                                                                                                        \
  } while (0)

// bf16 only for now: the 2-stage Stage-2 QK DPAS is bf16 (K/V are the gathered bf16
// latent), so a half query has no compiling path here. The half TU is still generated
// but is a dead-code stub (guarded in runMlaSparsePrefill2Stage).
#define DISPATCH_MLA_SPARSE_PREFILL_DTYPE_2STAGE()                                               \
  do {                                                                                           \
    switch (in_dtype) {                                                                          \
      case at::ScalarType::BFloat16:                                                             \
        DISPATCH_MLA_SPARSE_PREFILL_D_QK(bf16);                                                  \
        break;                                                                                   \
      default:                                                                                   \
        TORCH_CHECK(false, "2-stage Sparse MLA prefill currently supports only bfloat16 query"); \
    }                                                                                            \
  } while (0)

}  // namespace

/// @brief Dispatch kernel implementation for V4 Sparse MLA prefill.
SGL_KERNEL_EXPORT void flash_mla_sparse_prefill(
    torch::Tensor& out,            // [s_q, h_q, d_v]
    torch::Tensor& max_logits,     // [s_q, h_q]
    torch::Tensor& lse,            // [s_q, h_q]
    const torch::Tensor& q,        // [s_q, h_q, d_qk=512]
    const torch::Tensor& kv,       // [s_kv, h_kv=1, d_qk=512]
    const torch::Tensor& indices,  // [s_q, h_kv=1, topk]
    double sm_scale,
    int64_t head_dim_v,
    const std::optional<torch::Tensor>& attn_sink = std::nullopt,    // [h_q] or nullopt
    const std::optional<torch::Tensor>& topk_length = std::nullopt)  // [s_q] or nullopt
{
  CHECK_INPUT(out);
  CHECK_INPUT(max_logits);
  CHECK_INPUT(lse);
  CHECK_INPUT(q);
  CHECK_INPUT(kv);
  CHECK_INPUT(indices);

  c10::DeviceGuard guard(q.device());

  auto in_dtype = q.scalar_type();
  TORCH_CHECK(q.dim() == 3, "q must have shape [s_q, h_q, d_qk]");
  TORCH_CHECK(kv.dim() == 3 && kv.size(1) == 1, "kv must have shape [s_kv, 1, d_qk]");
  TORCH_CHECK(
      indices.dim() == 3 && indices.size(0) == q.size(0) && indices.size(1) == 1,
      "indices must have shape [s_q, 1, topk]");
  TORCH_CHECK((q.size(1) % 8) == 0, "num_heads must be a multiple of 8 (kernel fuses 8 heads per workgroup)");
  TORCH_CHECK(
      in_dtype == at::ScalarType::Half || in_dtype == at::ScalarType::BFloat16,
      "Unsupported input data type for Sparse MLA prefill");
  TORCH_CHECK(head_dim_v == 512, "head_dim_v must be 512 for DeepSeek V4 MLA");

  DISPATCH_MLA_SPARSE_PREFILL_DTYPE_2STAGE();
}

#undef DISPATCH_MLA_SPARSE_PREFILL_DTYPE_2STAGE
#undef DISPATCH_MLA_SPARSE_PREFILL_D_QK
#undef DISPATCH_MLA_SPARSE_PREFILL_B_H
#undef DISPATCH_MLA_SPARSE_PREFILL_SINK
#undef DISPATCH_MLA_SPARSE_PREFILL_LAUNCH_2STAGE
#undef SYCL_INTEL_TARGET
