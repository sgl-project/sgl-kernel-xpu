/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#define SYCL_INTEL_TARGET 20
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include "kernels/flash_attention_v2/relative_attention.hpp"
#include "kernels/flash_attention_v2/xe_fmha_fwd_decode_dispatch.hpp"
#include "kernels/flash_attention_v2/xe_fmha_fwd_prefill_dispatch.hpp"
#include "sgl_kernel_export.h"

namespace {

const float*
get_per_tensor_descale_ptr(const at::Tensor& descale, const at::Tensor& ref, const char* name, const char* context) {
  TORCH_CHECK(descale.scalar_type() == at::ScalarType::Float, name, " must be float32");
  TORCH_CHECK(
      descale.device() == ref.device(),
      context,
      " reads ",
      name,
      " on-device: it must be on the same device as the tensor it descales");
  TORCH_CHECK(descale.numel() > 0, name, " must not be empty");
  bool is_scalar_or_expanded_scalar = descale.numel() == 1;
  if (!is_scalar_or_expanded_scalar) {
    is_scalar_or_expanded_scalar = true;
    for (int64_t dim = 0; dim < descale.dim(); ++dim) {
      if (descale.size(dim) > 1 && descale.stride(dim) != 0) {
        is_scalar_or_expanded_scalar = false;
        break;
      }
    }
  }
  TORCH_CHECK(
      is_scalar_or_expanded_scalar,
      context,
      " uses a per-tensor descale: ",
      name,
      " must be a scalar or a view expanded from a single scalar element");
  return descale.data_ptr<float>();
}

}  // namespace

namespace decode {

// Non-paged (contiguous ragged KV) decode entry. Dedicated decode path: it
// drives the decode kernel (FmhaDecodeRunner with PagedKV = false) rather than
// reusing the prefill kernel, so the single-query decode batches selected by the
// chunkprefill dispatcher run on the decode-optimized kernel. The non-paged
// decode kernel carries its own tile configuration (FMHA_DECODE_TILED_KV_NP_*)
// so it can be tuned independently of both the paged decode and prefill paths.
void mha_fwd_nopage(
    const at::Tensor& q,             // (total_q, h, d)
    const at::Tensor& k,             // (total_k, h_k, d)
    const at::Tensor& v,             // (total_k, h_k, dv)
    const at::Tensor& cu_seqlens_q,  // b+1
    const at::Tensor& cu_seqlens_k,  // b+1 (cumulative prefix sum of KV lengths)
    int max_seqlen_q,
    int max_seqlen_k,
    std::optional<const at::Tensor>& sinks_,
    const float softmax_scale_,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    float const softcap,
    at::Tensor& out,
    std::optional<at::Tensor>& softmax_lse,
    std::optional<at::Tensor> skip_batch_mask_opt) {
  auto q_type = q.scalar_type();
  TORCH_CHECK(
      q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16,
      "mha_fwd only supports Half and BFloat16, got",
      q_type);
  TORCH_CHECK(k.scalar_type() == q_type, "query and key must have the same dtype");
  TORCH_CHECK(v.scalar_type() == q_type, "query and value must have the same dtype");
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(v);

  TORCH_CHECK(q.dim() == 3, "query must be in ragged format (total_q, h, d)");
  TORCH_CHECK(k.dim() == 3, "key must be in ragged format (total_k, h_k, d)");
  TORCH_CHECK(v.dim() == 3, "value must be in ragged format (total_k, h_k, dv)");
  CHECK_INPUT(cu_seqlens_q);
  TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype torch.int32");
  CHECK_INPUT(cu_seqlens_k);
  TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype torch.int32");

  const int batch_size = cu_seqlens_q.size(0) - 1;
  int seqlen_q = max_seqlen_q;
  int total_q = q.size(0);
  int num_heads = q.size(-2);
  int const head_size = q.size(-1);
  int const head_size_v = v.size(-1);
  int const total_k = k.size(0);
  int const seqlen_k = max_seqlen_k;
  int const num_heads_k = k.size(-2);
  float softmax_scale = softmax_scale_;

  TORCH_CHECK(cu_seqlens_k.size(0) - 1 == batch_size, "cu_seqlens_q and cu_seqlens_k must describe the same batch");

  static constexpr int max_headdim = 512;
  TORCH_CHECK(head_size <= max_headdim, "FlashAttention forward only supports head dimension at most ", max_headdim);
  TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

  if (window_size_left >= seqlen_k - 1) {
    window_size_left = -1;
  }
  window_size_right = min(window_size_right, seqlen_q);
  if (is_causal) {
    window_size_right = 0;
  }

  CHECK_SHAPE(q, total_q, num_heads, head_size);
  CHECK_SHAPE(k, total_k, num_heads_k, head_size);
  CHECK_SHAPE(v, total_k, num_heads_k, head_size_v);

  static constexpr int alignment = 8;
  TORCH_CHECK(head_size % alignment == 0, "head_size should be a multiple of " + std::to_string(alignment));
  TORCH_CHECK(head_size_v % alignment == 0, "head_size_v should be a multiple of " + std::to_string(alignment));

  // ``out`` is caller-provided and written in place. Whether to compute the
  // softmax logsumexp is derived from the caller-provided ``softmax_lse``
  // optional (a present tensor requests it; std::nullopt skips it).
  bool const return_softmax_lse = softmax_lse.has_value();

  int const head_size_rounded = round_up_headdim(head_size);

  c10::DeviceGuard device_guard(q.device());

  Arguments params;
  params.is_bf16 = q.dtype() == torch::kBFloat16;
  params.is_fp16 = q.dtype() == torch::kHalf;

  // Q / O are in ragged (total, h, d) format; KV is a contiguous ragged
  // (total_k, h_k, d) cache addressed via cu_seqlens_k offsets.
  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  params.q_row_stride = q.stride(-3);
  params.k_row_stride = k.stride(-3);
  params.v_row_stride = v.stride(-3);
  params.q_head_stride = q.stride(-2);
  params.k_head_stride = k.stride(-2);
  params.v_head_stride = v.stride(-2);
  params.v_dim_stride = v.stride(-1);
  params.o_ptr = out.data_ptr();
  params.o_row_stride = out.stride(-3);
  params.o_head_stride = out.stride(-2);

  // Per-batch skip mask for the chunkprefill two-launch path (may be null).
  params.skip_batch_mask_ptr = skip_batch_mask_opt.has_value() ? skip_batch_mask_opt->data_ptr() : nullptr;

  params.cu_seqlens_q = cu_seqlens_q.data_ptr<int>();
  params.cu_seqlens_k = cu_seqlens_k.data_ptr<int>();
  // No "new" KV: the whole sequence lives in the contiguous cache buffer, so the
  // decode kernel reads everything from the K/V cache pointers (knew = 0).
  params.cu_seqlens_knew = nullptr;
  params.seqlen_knew = 0;
  params.total_knew = 0;

  params.softmax_lse_ptr = return_softmax_lse ? softmax_lse->data_ptr() : nullptr;
  params.return_softmax_lse = return_softmax_lse;

  params.b = batch_size;
  params.h = num_heads;
  params.h_k = num_heads_k;
  // GQA packing: the decode kernel folds q_group_size query heads into the Q
  // tile, matching the paged decode path.
  params.q_group_size = num_heads / num_heads_k;
  params.seqlen_q = seqlen_q;
  params.seqlen_k = seqlen_k;
  params.d = head_size;
  params.d_rounded = head_size_rounded;

  params.softmax_scale = softmax_scale;
  if (sinks_.has_value()) {
    TORCH_CHECK(head_size == 64, "sink is only supported for head_size == 64, got ", head_size);
    params.use_sink = true;
    params.softmax_sink_ptr = sinks_.value().data_ptr();
  } else {
    params.use_sink = false;
    params.softmax_sink_ptr = nullptr;
  }
  params.softcap = softcap;
  params.p_dropout = 1.f;

  // Decode never needs a causal mask (each selected batch has seqlen_q <= 1, so
  // a single query attends to the full cache); sliding-window/local masking is
  // still honored. Mirrors decode::mha_fwd.
  params.is_causal = false;
  params.is_local = (window_size_left >= 0 || window_size_right >= 0) && !params.is_causal;
  if (window_size_left < 0) {
    window_size_left = seqlen_k - 1;
  }
  if (window_size_right < 0) {
    window_size_right = seqlen_q - 1;
  }
  params.window_size_left = window_size_left;
  params.window_size_right = window_size_right;
  params.total_q = total_q;
  params.total_k = total_k;
  params.b_k = batch_size;
  params.dv = head_size_v;

  // Non-paged KV: no page table. The non-paged decode path is compiled into its
  // own runner type (FmhaDecodeNpRunner, no PAGE_SIZE) and dispatched via
  // DISPATCH_DECODE_NOPAGE. page_size is irrelevant here (the non-paged KV tile
  // is FMHA_DECODE_TILED_KV_NP_*, independent of any page size).
  params.page_table = nullptr;
  params.page_table_batch_stride = 0;
  params.max_num_pages_per_seq = 0;
  params.page_size = 128;
  params.num_pages = 0;

  // Split-KV is a paged-cache optimization; the non-paged path uses the
  // single-launch decode kernel.
  params.use_split_kv = false;
  params.num_kv_splits = -1;

  params.rotary_dim = 0;

  params.tensor_opts = torch::TensorOptions().dtype(torch::kUInt8).device(q.device());

  int qg_sz = nextPowerOf2(params.q_group_size);
  TORCH_CHECK(qg_sz >= 1 && qg_sz <= 16, "Unsupported q_group_size for decode attention: ", params.q_group_size);
  // Non-paged decode supports its own (independent) set of head dims; see
  // FMHA_DECODE_NP_HEAD_DIMS in FMHADecodeXe20.cmake.
  TORCH_CHECK(
      params.d == 64 || params.d == 72 || params.d == 80 || params.d == 96 || params.d == 128 || params.d == 192 ||
          params.d == 256 || params.d == 512,
      "Unsupported head size for non-paged decode attention: ",
      params.d);

  DISPATCH_DECODE_NOPAGE(qg_sz);
}

void mha_fwd(
    const at::Tensor& q,  // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
    const at::Tensor& k,  // (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size,
                          // h_k, d) if there is page_table.
    const at::Tensor& v,  // (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages,
                          // page_size, h_k, dv) if there is page_table.
    std::optional<const at::Tensor>& q_v_,  // (b, s_q, h, dv) or (total_q_new, h, dv) if there is cu_seqlens_q
    const at::Tensor& cu_seqlens_q,         // b+1
    const at::Tensor& cu_seqlens_k,         // b+1
    int max_seqlen_q,
    int max_seqlen_k,
    std::optional<const at::Tensor>& page_table,       // (b_k, max_num_pages_per_seq)
    std::optional<const at::Tensor>& kv_batch_idx_,    // b. indices to index into the KV cache
    std::optional<const at::Tensor>& leftpad_k_,       // b
    std::optional<const at::Tensor>& rotary_cos_,      // seqlen_ro x (rotary_dim / 2)
    std::optional<const at::Tensor>& rotary_sin_,      // seqlen_ro x (rotary_dim / 2)
    std::optional<const at::Tensor>& seqlens_rotary_,  // b
    std::optional<at::Tensor>& q_descale_,             // (b, h_k), not (b, h)
    std::optional<at::Tensor>& k_descale_,             // (b, h_k)
    std::optional<at::Tensor>& v_descale_,             // (b, h_k)
    const float softmax_scale_,
    std::optional<const at::Tensor>& sinks_,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    float const softcap,
    bool const is_rotary_interleaved,  // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
    std::optional<at::Tensor>& scheduler_metadata_,  // (b + 1)
    int num_kv_splits,
    std::optional<bool> pack_gqa_,
    int const sm_margin,
    // Caller-provided output buffers written in place: ``out`` receives the
    // attention result and ``softmax_lse`` the logsumexp when present. A
    // per-batch skip mask (length = batch, chunkprefill two-launch path) selects
    // which batches this launch processes.
    at::Tensor& out,
    std::optional<at::Tensor>& softmax_lse,
    std::optional<at::Tensor> skip_batch_mask_opt = std::nullopt,
    std::optional<const at::Tensor> rel_bias_ = std::nullopt,
    bool rel_bias_is_sheared = false) {
  auto q_type = q.scalar_type();
  TORCH_CHECK(
      q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16,
      "mha_fwd only supports Half and BFloat16, got",
      q_type);

  // FP8 KV cache: Q stays bf16/fp16 while K/V cache may be fp8 (e4m3 or e5m2).
  // The decode mainloop dequantizes K/V with per-tensor k_descale/v_descale.
  // When launched as the decode sub-kernel of a pure-prefill chunkprefill batch
  // the descale pointers are still forwarded; the kernel simply skips every
  // batch.
  bool const is_e4m3_kv = k.scalar_type() == at::ScalarType::Float8_e4m3fn;
  bool const is_e5m2_kv = k.scalar_type() == at::ScalarType::Float8_e5m2;
  bool const is_fp8_kv = is_e4m3_kv || is_e5m2_kv;
  if (is_fp8_kv) {
    TORCH_CHECK(
        v.scalar_type() == k.scalar_type(),
        "fp8 KV cache requires key and value to have the same fp8 dtype (both float8_e4m3fn or both float8_e5m2)");
  } else {
    TORCH_CHECK(k.scalar_type() == q_type, "query and key must have the same dtype");
    TORCH_CHECK(v.scalar_type() == q_type, "query and value must have the same dtype");
  }
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(v);

  // Non-paged (page_table == nullopt) decode sub-launch of chunkprefill. Uses
  // the decode-specific non-paged entry (decode::mha_fwd_nopage) so it can carry
  // its own parameter configuration independently of the prefill path.
  if (!page_table.has_value()) {
    TORCH_CHECK(!rel_bias_.has_value(), "relative attention requires paged KV cache");
    mha_fwd_nopage(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        sinks_,
        softmax_scale_,
        is_causal,
        window_size_left,
        window_size_right,
        softcap,
        out,
        softmax_lse,
        std::move(skip_batch_mask_opt));
    return;
  }

  TORCH_CHECK(page_table.value().dtype() == torch::kInt32, "page_table must have dtype torch.int32");
  TORCH_CHECK(page_table.value().stride(-1) == 1, "page_table must have contiguous last dimension");

  TORCH_CHECK(q.dim() == 3, "query must be in ragged format");
  CHECK_INPUT(cu_seqlens_q);
  TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype torch.int32");

  CHECK_INPUT(cu_seqlens_k);
  TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype torch.int32");

  auto const sizes = q.sizes();
  const int batch_size = cu_seqlens_q.size(0) - 1;
  int seqlen_q = max_seqlen_q;
  int total_q = q.size(0);
  int num_heads = q.size(-2);
  int const head_size = q.size(-1);
  int const head_size_v = v.size(-1);
  int const max_num_pages_per_seq = page_table.value().size(1);
  int const num_pages = k.size(0);
  int const page_size = k.size(1);
  int const seqlen_k = page_table.has_value() && max_seqlen_k <= 0 ? max_num_pages_per_seq * page_size : max_seqlen_k;
  int const total_k = num_pages * page_size;
  int const num_heads_k = k.size(-2);

  int const batch_size_k = page_table.value().size(0);
  float softmax_scale = softmax_scale_;

  if (!kv_batch_idx_.has_value()) {
    TORCH_CHECK(batch_size == batch_size_k, "batch_size must be equal to batch_size_k");
  }

  // Currently only support head dims <= 512
  static constexpr int max_headdim = 512;
  TORCH_CHECK(head_size <= max_headdim, "FlashAttention forward only supports head dimension at most ", max_headdim);
  TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
  if (rel_bias_.has_value()) {
    const at::Tensor& rel_bias = rel_bias_.value();
    TORCH_CHECK(
        (head_size == 128 || head_size == 512) && head_size_v == head_size,
        "relative attention requires head_dim=head_dim_v of 128 or 512");
    TORCH_CHECK(!is_fp8_kv, "relative attention does not support fp8 KV cache");
    TORCH_CHECK(rel_bias.scalar_type() == q.scalar_type(), "relative bias must have the same dtype as query");
    TORCH_CHECK(rel_bias.device() == q.device(), "relative bias must be on the same device as query");
    TORCH_CHECK(rel_bias.dim() == 3, "relative bias must have shape [total_q, num_heads, extent]");
    TORCH_CHECK(
        rel_bias.size(0) == total_q && rel_bias.size(1) == num_heads,
        "relative bias must have shape [total_q, num_heads, extent]");
    TORCH_CHECK(rel_bias.stride(-1) == 1, "relative bias must have a contiguous K dimension");
    TORCH_CHECK(rel_bias.size(-1) > 0, "relative bias extent must be positive");
    if (rel_bias_is_sheared) {
      TORCH_CHECK(
          rel_bias.size(-1) > page_size && rel_bias.size(-1) % page_size == 0,
          "pre-sheared relative bias must have a page-size-aligned decode surface width");
    }
  }

  // This needs to go before kBlockM & kBlockN since we rely on the correct window_size and is_causal to set kBlockM
  // TODO: check this

  if (window_size_left >= seqlen_k - 1) {
    window_size_left = -1;
  }
  window_size_right = min(window_size_right, seqlen_q);
  // causal=true is the same as causal=false in this case
  if (is_causal) {
    window_size_right = 0;
  }

  CHECK_SHAPE(k, num_pages, page_size, num_heads_k, head_size);
  CHECK_SHAPE(v, num_pages, page_size, num_heads_k, head_size_v);
  CHECK_SHAPE(page_table.value(), batch_size_k, max_num_pages_per_seq);

  if (leftpad_k_.has_value()) {
    auto leftpad_k = leftpad_k_.value();
    TORCH_CHECK(leftpad_k.dtype() == torch::kInt32, "leftpad_k must have dtype int32");
    CHECK_INPUT(leftpad_k);
    CHECK_SHAPE(leftpad_k, batch_size);
  }

  static constexpr int alignment = 8;
  TORCH_CHECK(head_size % alignment == 0, "head_size should be a multiple of " + std::to_string(alignment));
  TORCH_CHECK(head_size_v % alignment == 0, "head_size_v should be a multiple of " + std::to_string(alignment));

  // ``out`` is caller-provided and written in place. Whether to compute the
  // softmax logsumexp is derived from the caller-provided ``softmax_lse``
  // optional (a present tensor requests it; std::nullopt skips it).
  bool const return_softmax_lse = softmax_lse.has_value();
  at::Tensor temp_out;    // [batch, num_kv_splits, num_head_q, seq_q, head_size]
  at::Tensor exp_sums;    // [batch, num_head_q, seq_q, num_kv_splits]
  at::Tensor max_logits;  // [batch, num_head_q, seq_q, num_kv_splits]
  Arguments params;
  // num_kv_splits semantics (host-side scalar, no D2H sync):
  //   -1 or 1 -> split-KV disabled, use the non-split FmhaDecodeRunner
  //         0 -> auto: pick a split count from the device-occupancy heuristic
  //        >1 -> use the caller-provided split count with FmhaSplitDecodeRunner
  if (num_kv_splits == 0) {
    auto get_num_splits =
        [](int batch_size, int num_heads_q, int num_heads_kv, int head_size_v, int max_seqlen_k, int block_size) {
          auto stream = at::xpu::getCurrentXPUStream();
          auto queue = stream.queue();
          auto device = queue.get_device();
          int num_xe_cores = device.get_info<sycl::ext::intel::info::device::gpu_slices>() *
                             device.get_info<sycl::ext::intel::info::device::gpu_subslices_per_slice>();
          int parallel_ = num_xe_cores;
          int parallel_2 = num_xe_cores * 2;
          int cur_parallel_d = batch_size * num_heads_kv;
          int num_splits = (parallel_ + cur_parallel_d - 1) / cur_parallel_d;
          if (cur_parallel_d * num_splits > parallel_ && num_splits > 1) {
            num_splits = std::ceil(parallel_2 / static_cast<float>(cur_parallel_d)) - 1;
          }

          int total_blocks = (max_seqlen_k + block_size - 1) / block_size;
          constexpr int kMinBlocksToSplit = 64;
          constexpr int kAlwaysSplitHeadSizeV = 512;
          if (total_blocks <= kMinBlocksToSplit && head_size_v < kAlwaysSplitHeadSizeV) {
            return 1;
          }

          if (batch_size == 1 && num_heads_q == 8 && num_heads_kv == 1 && head_size_v == 512 && block_size == 64) {
            int const current_parallelism = 2;
            int const target_splits = (3 * num_xe_cores + current_parallelism - 1) / current_parallelism;
            int const split_limit = std::min({target_splits, total_blocks, 64});
            int best_splits = 1;
            int best_iters = total_blocks;
            for (int splits = 1; splits <= split_limit; ++splits) {
              int const iters = (total_blocks + splits - 1) / splits;
              if (iters < best_iters) {
                best_iters = iters;
                best_splits = splits;
              }
            }
            return best_splits;
          }

          int max_splits = std::min(total_blocks, parallel_);
          return std::min(num_splits, max_splits);
        };
    num_kv_splits = get_num_splits(batch_size, num_heads, num_heads_k, head_size_v, seqlen_k, page_size);
  }
  // Only split when the resolved count is > 1; -1 / 1 fall back to non-split.
  params.use_split_kv = num_kv_splits > 1;
  if (params.use_split_kv) {
    temp_out = torch::empty({total_q, num_kv_splits * num_heads, head_size_v}, q.options().device(q.device()));

    max_logits = torch::empty({total_q, num_heads, num_kv_splits}, q.options().dtype(at::kFloat).device(q.device()));
    exp_sums = torch::empty({total_q, num_heads, num_kv_splits}, q.options().dtype(at::kFloat).device(q.device()));

    params.temp_out_ptr = temp_out.data_ptr();
    params.exp_sums_ptr = exp_sums.data_ptr();
    params.max_logits_ptr = max_logits.data_ptr();
  }
  int const head_size_rounded = round_up_headdim(head_size);
  int const head_size_v_rounded = head_size_v == head_size ? head_size_rounded : round_up_headdim(head_size_v);

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  c10::DeviceGuard device_guard(q.device());

  // ``softmax_lse`` is caller-provided; empty (numel == 0) when the LSE was not
  // requested, in which case the kernel skips every LSE write.

  // align with FA3

  params.is_bf16 = q.dtype() == torch::kBFloat16;
  params.is_fp16 = q.dtype() == torch::kHalf;

  // Set the pointers and strides.
  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  // All stride are in elements, not bytes.
  params.q_row_stride = q.stride(-3);
  params.k_row_stride = k.stride(-3);
  params.v_row_stride = v.stride(-3);
  params.q_head_stride = q.stride(-2);

  params.k_head_stride = k.stride(-2);
  params.v_head_stride = v.stride(-2);

  params.k_stride_page = k.stride(0);
  params.k_stride_seq = k.stride(1);
  params.k_stride_heads = k.stride(2);
  params.v_stride_page = v.stride(0);
  params.v_stride_seq = v.stride(1);
  params.v_stride_heads = v.stride(2);

  params.v_dim_stride = v.stride(-1);
  params.o_ptr = out.data_ptr();
  params.o_row_stride = out.stride(-3);
  params.o_head_stride = out.stride(-2);
  at::Tensor sheared_rel_bias;
  if (rel_bias_.has_value()) {
    const at::Tensor& rel_bias = rel_bias_.value();
    if (rel_bias_is_sheared) {
      sheared_rel_bias = rel_bias;
    } else if (q.scalar_type() == at::ScalarType::BFloat16) {
      sheared_rel_bias = flash_attention_v2::relative_attention::prepare_decode_bias<at::BFloat16>(
          rel_bias, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, page_size);
    } else {
      sheared_rel_bias = flash_attention_v2::relative_attention::prepare_decode_bias<at::Half>(
          rel_bias, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, page_size);
    }
    params.rel_bias_ptr = sheared_rel_bias.data_ptr();
    params.rel_bias_token_stride = sheared_rel_bias.stride(0);
    params.rel_bias_head_stride = sheared_rel_bias.stride(1);
    params.rel_bias_extent = rel_bias_is_sheared ? rel_bias.size(-1) - page_size : rel_bias.size(-1);
  }

  // Per-batch skip mask for the chunkprefill two-launch path
  // (vllm-xpu-kernels#218). When provided, decode skips batches where
  // mask[idx_b] == true (i.e. the prefill rows).
  params.skip_batch_mask_ptr = skip_batch_mask_opt.has_value() ? skip_batch_mask_opt->data_ptr() : nullptr;

  params.cu_seqlens_q = cu_seqlens_q.data_ptr<int>();
  params.cu_seqlens_k = cu_seqlens_k.data_ptr<int>();
  params.num_kv_splits = num_kv_splits;

  // Softmax sum (null when the caller passes std::nullopt).
  params.softmax_lse_ptr = return_softmax_lse ? softmax_lse->data_ptr() : nullptr;
  params.return_softmax_lse = return_softmax_lse;

  // Set the dimensions.
  params.b = batch_size;
  params.h = num_heads;
  params.h_k = num_heads_k;
  params.q_group_size = num_heads / num_heads_k;
  params.seqlen_q = seqlen_q;
  params.seqlen_k = seqlen_k;
  params.d = head_size;
  params.d_rounded = head_size_rounded;

  // Set the different scale values.
  params.softmax_scale = softmax_scale;
  if (sinks_.has_value()) {
    TORCH_CHECK(head_size == 64, "sink is only supported for head_size == 64, got ", head_size);
    params.use_sink = true;
    params.softmax_sink_ptr = sinks_.value().data_ptr();
  } else {
    params.use_sink = false;
    params.softmax_sink_ptr = nullptr;
  }

  params.softcap = softcap;

  // FP8 KV cache descale wiring. K/V are stored as fp8 (e4m3 or e5m2) and
  // dequantized in the decode mainloop by a single per-tensor scale
  // (k_descale/v_descale, float32).
  params.is_e4m3 = is_e4m3_kv;
  params.is_e5m2 = is_e5m2_kv;
  if (is_fp8_kv) {
    TORCH_CHECK(
        k_descale_.has_value() && v_descale_.has_value(), "fp8 KV cache decode requires k_descale and v_descale");
    // Per-tensor dequant: the kernel only dereferences one float, so accept a
    // true scalar or a tensor whose elements are all the same repeated value.
    params.k_scale_ptr = get_per_tensor_descale_ptr(k_descale_.value(), k, "k_descale", "fp8 KV cache decode");
    params.v_scale_ptr = get_per_tensor_descale_ptr(v_descale_.value(), v, "v_descale", "fp8 KV cache decode");
  }

  // Set this to probability of keeping an element to simplify things.
  params.p_dropout = 1.f;

  // Causal is the special case where window_size_right == 0 and window_size_left < 0.
  // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
  params.is_causal = false;  // Decode don't need causal mask since we only compute attention for the current token, but
                             // this kernel can also be used for local attention in the future
  params.is_local = (window_size_left >= 0 || window_size_right >= 0) && !params.is_causal;

  // TODO: check this
  if (window_size_left < 0) {
    window_size_left = seqlen_k - 1;
  }
  if (window_size_right < 0) {
    window_size_right = seqlen_q - 1;
  }
  params.window_size_left = window_size_left;
  params.window_size_right = window_size_right;
  params.total_q = total_q;
  params.total_k = total_k;
  params.b_k = batch_size_k;
  params.dv = head_size_v;
  params.page_table = page_table.value().data_ptr<int>();
  params.page_table_batch_stride = page_table.value().stride(0);
  params.max_num_pages_per_seq = max_num_pages_per_seq;
  params.page_size = page_size;
  params.num_pages = num_pages;

  if (q_v_.has_value()) {
    TORCH_CHECK(head_size <= 64, "q_v is only supported for head_size <= 64");
    TORCH_CHECK(
        q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16,
        "q_v is only supported for fp16 and bf16 data type");
    TORCH_CHECK(false, "q_v is not supported yet");
    at::Tensor q_v = q_v_.value();
    TORCH_CHECK(q_v.dtype() == q_type, "q_v must have the same dtype as query");
    TORCH_CHECK(q_v.stride(-1) == 1, "q_v tensor must have contiguous last dimension");
    CHECK_SHAPE(q_v, total_q, num_heads, head_size_v);
    params.qv_ptr = q_v.data_ptr();
    // All stride are in elements, not bytes.
    params.qv_row_stride = q_v.stride(-3);
    params.qv_head_stride = q_v.stride(-2);
  }

  if (rotary_cos_.has_value()) {
    auto rotary_cos = rotary_cos_.value();
    CHECK_INPUT(rotary_cos);
    params.rotary_dim = rotary_cos.size(1) * 2;
    TORCH_CHECK(params.rotary_dim <= head_size, "rotary_dim must be <= headdim");
    TORCH_CHECK(params.rotary_dim % 16 == 0, "Only rotary dimensions divisible by 16 are currently supported");
    const int seqlen_ro = rotary_cos.size(0);
    TORCH_CHECK(seqlen_ro >= seqlen_k, "cos/sin seqlen must be at least the seqlen of KV cache");
    CHECK_SHAPE(rotary_cos, seqlen_ro, params.rotary_dim / 2);
    TORCH_CHECK(rotary_cos.scalar_type() == q_type, "rotary_cos must have the same dtype as query");

    TORCH_CHECK(rotary_sin_.has_value(), "If rotary cos is provided, rotary sin must also be provided");
    auto rotary_sin = rotary_sin_.value();
    CHECK_INPUT(rotary_sin);
    CHECK_SHAPE(rotary_sin, seqlen_ro, params.rotary_dim / 2);
    TORCH_CHECK(rotary_sin.scalar_type() == q_type, "rotary_cos must have the same dtype as query");
    params.rotary_cos_ptr = rotary_cos.data_ptr();
    params.rotary_sin_ptr = rotary_sin.data_ptr();
    params.is_rotary_interleaved = is_rotary_interleaved;
    if (seqlens_rotary_.has_value()) {
      at::Tensor seqlens_rotary = seqlens_rotary_.value();
      CHECK_INPUT(seqlens_rotary);
      TORCH_CHECK(seqlens_rotary.dtype() == torch::kInt32, "seqlens_rotary must have dtype torch.int32");
      CHECK_SHAPE(seqlens_rotary, batch_size);
      params.seqlens_rotary = seqlens_rotary.data_ptr<int>();
    }
  } else {
    params.rotary_dim = 0;
  }

  if (kv_batch_idx_.has_value()) {
    auto kv_batch_idx = kv_batch_idx_.value();
    CHECK_INPUT(kv_batch_idx);
    TORCH_CHECK(kv_batch_idx.scalar_type() == torch::kInt32, "kv_batch_idx must have dtype int32");
    params.kv_batch_idx = reinterpret_cast<int*>(kv_batch_idx.data_ptr());
  }

  params.tensor_opts = torch::TensorOptions().dtype(torch::kUInt8).device(q.device());

  int qg_sz = nextPowerOf2(params.q_group_size);
  TORCH_CHECK(qg_sz >= 1 && qg_sz <= 16, "Unsupported q_group_size for decode attention: ", params.q_group_size);
  // Paged decode supports its own (independent) set of head dims; see
  // FMHA_DECODE_PAGED_HEAD_DIMS in FMHADecodeXe20.cmake.
  TORCH_CHECK(
      params.d == 64 || params.d == 96 || params.d == 128 || params.d == 192 || params.d == 256 || params.d == 512,
      "Unsupported head size for paged decode attention: ",
      params.d);
  TORCH_CHECK(
      params.page_size == 64 || params.page_size == 128,
      "Unsupported page size for decode attention: ",
      params.page_size);

  DISPATCH_DECODE(qg_sz);
}

}  // namespace decode

namespace prefill {

// Non-paged (contiguous ragged KV) prefill entry. Drives both the prefill and
// the decode sub-launches of the no-page chunkprefill two-launch path: the
// caller passes shared ``out`` / ``softmax_lse`` buffers (written in place) and
// a per-batch skip mask (skip_batch_mask_opt) selecting which batches this
// launch processes.
void mha_fwd_nopage(
    const at::Tensor& q,             // (total_q, h, d)
    const at::Tensor& k,             // (total_k, h_k, d)
    const at::Tensor& v,             // (total_k, h_k, dv)
    const at::Tensor& cu_seqlens_q,  // b+1
    const at::Tensor& cu_seqlens_k,  // b+1 (cumulative prefix sum of KV lengths)
    int max_seqlen_q,
    int max_seqlen_k,
    std::optional<const at::Tensor>& sinks_,
    const float softmax_scale_,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    float const softcap,
    at::Tensor& out,
    std::optional<at::Tensor>& softmax_lse,
    std::optional<at::Tensor> skip_batch_mask_opt) {
  auto q_type = q.scalar_type();
  TORCH_CHECK(
      q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16,
      "mha_fwd only supports Half and BFloat16, got",
      q_type);
  TORCH_CHECK(k.scalar_type() == q_type, "query and key must have the same dtype");
  TORCH_CHECK(v.scalar_type() == q_type, "query and value must have the same dtype");
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(v);

  TORCH_CHECK(q.dim() == 3, "query must be in ragged format (total_q, h, d)");
  TORCH_CHECK(k.dim() == 3, "key must be in ragged format (total_k, h_k, d)");
  TORCH_CHECK(v.dim() == 3, "value must be in ragged format (total_k, h_k, dv)");
  CHECK_INPUT(cu_seqlens_q);
  TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype torch.int32");
  CHECK_INPUT(cu_seqlens_k);
  TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype torch.int32");

  const int batch_size = cu_seqlens_q.size(0) - 1;
  int seqlen_q = max_seqlen_q;
  int total_q = q.size(0);
  int num_heads = q.size(-2);
  int const head_size = q.size(-1);
  int const head_size_v = v.size(-1);
  int const total_k = k.size(0);
  int const seqlen_k = max_seqlen_k;
  int const num_heads_k = k.size(-2);
  float softmax_scale = softmax_scale_;

  TORCH_CHECK(cu_seqlens_k.size(0) - 1 == batch_size, "cu_seqlens_q and cu_seqlens_k must describe the same batch");

  static constexpr int max_headdim = 512;
  TORCH_CHECK(head_size <= max_headdim, "FlashAttention forward only supports head dimension at most ", max_headdim);
  TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

  if (window_size_left >= seqlen_k - 1) {
    window_size_left = -1;
  }
  window_size_right = min(window_size_right, seqlen_q);
  if (is_causal) {
    window_size_right = 0;
  }

  CHECK_SHAPE(q, total_q, num_heads, head_size);
  CHECK_SHAPE(k, total_k, num_heads_k, head_size);
  CHECK_SHAPE(v, total_k, num_heads_k, head_size_v);

  static constexpr int alignment = 8;
  TORCH_CHECK(head_size % alignment == 0, "head_size should be a multiple of " + std::to_string(alignment));
  TORCH_CHECK(head_size_v % alignment == 0, "head_size_v should be a multiple of " + std::to_string(alignment));

  // ``out`` is caller-provided and written in place. Non-paged prefill does not
  // currently compute the softmax logsumexp (``softmax_lse`` is left untouched).

  int const head_size_rounded = round_up_headdim(head_size);

  c10::DeviceGuard device_guard(q.device());

  Arguments params;
  params.is_bf16 = q.dtype() == torch::kBFloat16;
  params.is_fp16 = q.dtype() == torch::kHalf;

  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  params.q_row_stride = q.stride(-3);
  params.k_row_stride = k.stride(-3);
  params.v_row_stride = v.stride(-3);
  params.q_head_stride = q.stride(-2);
  params.k_head_stride = k.stride(-2);
  params.v_head_stride = v.stride(-2);
  params.v_dim_stride = v.stride(-1);
  params.o_ptr = out.data_ptr();
  params.o_row_stride = out.stride(-3);
  params.o_head_stride = out.stride(-2);

  // Per-batch skip mask for the chunkprefill two-launch path (may be null).
  params.skip_batch_mask_ptr = skip_batch_mask_opt.has_value() ? skip_batch_mask_opt->data_ptr() : nullptr;

  params.cu_seqlens_q = cu_seqlens_q.data_ptr<int>();
  params.cu_seqlens_k = cu_seqlens_k.data_ptr<int>();

  params.softmax_lse_ptr = softmax_lse.has_value() ? softmax_lse->data_ptr() : nullptr;

  params.b = batch_size;
  params.h = num_heads;
  params.h_k = num_heads_k;
  params.q_group_size = 1;
  params.seqlen_q = seqlen_q;
  params.seqlen_k = seqlen_k;
  params.d = head_size;
  params.d_rounded = head_size_rounded;

  params.softmax_scale = softmax_scale;
  if (sinks_.has_value()) {
    TORCH_CHECK(head_size == 64, "sink is only supported for head_size == 64, got ", head_size);
    params.softmax_sink_ptr = sinks_.value().data_ptr();
  } else {
    params.softmax_sink_ptr = nullptr;
  }
  params.softcap = softcap;
  params.p_dropout = 1.f;

  params.is_causal = window_size_left < 0 && window_size_right == 0;
  params.is_local = (window_size_left >= 0 || window_size_right >= 0) && !params.is_causal;
  if (window_size_left < 0) {
    window_size_left = seqlen_k - 1;
  }
  if (window_size_right < 0) {
    window_size_right = seqlen_q - 1;
  }
  params.window_size_left = window_size_left;
  params.window_size_right = window_size_right;
  params.total_q = total_q;
  params.total_k = total_k;
  params.b_k = batch_size;
  params.dv = head_size_v;

  // Non-paged KV: no page table. The kernel branches on page_table == nullptr.
  params.page_table = nullptr;
  params.page_table_batch_stride = 0;
  params.max_num_pages_per_seq = 0;
  params.page_size = 0;
  params.num_pages = 0;

  params.rotary_dim = 0;

  params.tensor_opts = torch::TensorOptions().dtype(torch::kUInt8).device(q.device());

  // Non-paged prefill supports its own (independent) set of head dims; see
  // FMHA_PREFILL_NP_HEAD_DIMS in FMHAPrefillXe20.cmake.
  TORCH_CHECK(
      params.d == 64 || params.d == 72 || params.d == 80 || params.d == 96 || params.d == 128 || params.d == 192 ||
          params.d == 256 || params.d == 512,
      "Unsupported head size for non-paged prefill attention: ",
      params.d);

  switch (params.d) {
    case 64:
      DISPATCH_PREFILL_NOPAGE_KERNEL(64);
      break;
    case 72:
      DISPATCH_PREFILL_NOPAGE_KERNEL(72);
      break;
    case 80:
      DISPATCH_PREFILL_NOPAGE_KERNEL(80);
      break;
    case 96:
      DISPATCH_PREFILL_NOPAGE_KERNEL(96);
      break;
    case 128:
      DISPATCH_PREFILL_NOPAGE_KERNEL(128);
      break;
    case 192:
      DISPATCH_PREFILL_NOPAGE_KERNEL(192);
      break;
    case 256:
      DISPATCH_PREFILL_NOPAGE_KERNEL(256);
      break;
    case 512:
      DISPATCH_PREFILL_NOPAGE_KERNEL(512);
      break;
    default:
      TORCH_CHECK(false, "Unsupported head size for non-paged prefill attention: ", params.d);
  }

  // TODO: Support prefill softmax_lse, now is 0
}

void mha_fwd(
    const at::Tensor& q,  // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
    const at::Tensor& k,  // (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size,
                          // h_k, d) if there is page_table.
    const at::Tensor& v,  // (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages,
                          // page_size, h_k, dv) if there is page_table.
    std::optional<const at::Tensor>& q_v_,  // (b, s_q, h, dv) or (total_q_new, h, dv) if there is cu_seqlens_q
    const at::Tensor& cu_seqlens_q,         // b+1
    const at::Tensor& cu_seqlens_k,         // b+1
    int max_seqlen_q,
    int max_seqlen_k,
    std::optional<const at::Tensor>& page_table,       // (b_k, max_num_pages_per_seq)
    std::optional<const at::Tensor>& kv_batch_idx_,    // b. indices to index into the KV cache
    std::optional<const at::Tensor>& leftpad_k_,       // b
    std::optional<const at::Tensor>& rotary_cos_,      // seqlen_ro x (rotary_dim / 2)
    std::optional<const at::Tensor>& rotary_sin_,      // seqlen_ro x (rotary_dim / 2)
    std::optional<const at::Tensor>& seqlens_rotary_,  // b
    std::optional<at::Tensor>& q_descale_,             // (b, h_k), not (b, h)
    std::optional<at::Tensor>& k_descale_,             // (b, h_k)
    std::optional<at::Tensor>& v_descale_,             // (b, h_k)
    const float softmax_scale_,
    std::optional<const at::Tensor>& sinks_,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    float const softcap,
    bool const is_rotary_interleaved,  // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
    std::optional<at::Tensor>& scheduler_metadata_,  // (b + 1)
    int num_splits,
    std::optional<bool> pack_gqa_,
    int const sm_margin,
    // Caller-provided output buffers written in place: ``out`` receives the
    // attention result and ``softmax_lse`` the logsumexp when present. A
    // per-batch skip mask (length = batch, chunkprefill two-launch path) selects
    // which batches this launch processes.
    at::Tensor& out,
    std::optional<at::Tensor>& softmax_lse,
    std::optional<at::Tensor> skip_batch_mask_opt = std::nullopt,
    std::optional<const at::Tensor> rel_bias_ = std::nullopt) {
  auto q_type = q.scalar_type();
  TORCH_CHECK(
      q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16,
      "mha_fwd only supports Half and BFloat16, got",
      q_type);

  // FP8 KV cache: Q stays bf16/fp16 while K/V cache may be fp8 (e4m3 or e5m2).
  // In that case per-tensor k_descale/v_descale must be supplied. Otherwise K/V
  // must match the Q dtype.
  bool const is_e4m3_kv = k.scalar_type() == at::ScalarType::Float8_e4m3fn;
  bool const is_e5m2_kv = k.scalar_type() == at::ScalarType::Float8_e5m2;
  bool const is_fp8_kv = is_e4m3_kv || is_e5m2_kv;
  if (is_fp8_kv) {
    TORCH_CHECK(
        v.scalar_type() == k.scalar_type(),
        "fp8 KV cache requires key and value to have the same fp8 dtype (both float8_e4m3fn or both float8_e5m2)");
    TORCH_CHECK(k_descale_.has_value() && v_descale_.has_value(), "fp8 KV cache requires k_descale and v_descale");
  } else {
    TORCH_CHECK(k.scalar_type() == q_type, "query and key must have the same dtype");
    TORCH_CHECK(v.scalar_type() == q_type, "query and value must have the same dtype");
  }
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(v);

  // Non-paged (page_table == nullopt) prefill: contiguous ragged KV cache.
  if (!page_table.has_value()) {
    TORCH_CHECK(!rel_bias_.has_value(), "relative attention requires paged KV cache");
    mha_fwd_nopage(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        sinks_,
        softmax_scale_,
        is_causal,
        window_size_left,
        window_size_right,
        softcap,
        out,
        softmax_lse,
        std::move(skip_batch_mask_opt));
    return;
  }

  TORCH_CHECK(page_table.value().dtype() == torch::kInt32, "page_table must have dtype torch.int32");
  TORCH_CHECK(page_table.value().stride(-1) == 1, "page_table must have contiguous last dimension");

  TORCH_CHECK(q.dim() == 3, "query must be in ragged format");
  CHECK_INPUT(cu_seqlens_q);
  TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype torch.int32");

  CHECK_INPUT(cu_seqlens_k);
  TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype torch.int32");

  auto const sizes = q.sizes();
  const int batch_size = cu_seqlens_q.size(0) - 1;
  int seqlen_q = max_seqlen_q;
  int total_q = q.size(0);
  int num_heads = q.size(-2);
  int const head_size = q.size(-1);
  int const head_size_v = v.size(-1);
  int const max_num_pages_per_seq = page_table.value().size(1);
  int const num_pages = k.size(0);
  int const page_size = k.size(1);
  int const seqlen_k = max_num_pages_per_seq * page_size;
  int const total_k = num_pages * page_size;
  int const num_heads_k = k.size(-2);

  int const batch_size_k = page_table.value().size(0);
  float softmax_scale = softmax_scale_;

  if (!kv_batch_idx_.has_value()) {
    TORCH_CHECK(batch_size == batch_size_k, "batch_size must be equal to batch_size_k");
  }

  // Currently only support head dims <= 512
  static constexpr int max_headdim = 512;
  TORCH_CHECK(head_size <= max_headdim, "FlashAttention forward only supports head dimension at most ", max_headdim);
  TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
  if (rel_bias_.has_value()) {
    const at::Tensor& rel_bias = rel_bias_.value();
    TORCH_CHECK(head_size == 128 && head_size_v == 128, "relative attention requires head_dim=head_dim_v=128");
    TORCH_CHECK(!is_fp8_kv, "relative attention does not support fp8 KV cache");
    TORCH_CHECK(rel_bias.scalar_type() == q.scalar_type(), "relative bias must have the same dtype as query");
    TORCH_CHECK(rel_bias.device() == q.device(), "relative bias must be on the same device as query");
    TORCH_CHECK(rel_bias.dim() == 3, "relative bias must have shape [total_q, num_heads, extent]");
    TORCH_CHECK(
        rel_bias.size(0) == total_q && rel_bias.size(1) == num_heads,
        "relative bias must have shape [total_q, num_heads, extent]");
    TORCH_CHECK(rel_bias.stride(-1) == 1, "relative bias must have a contiguous K dimension");
    TORCH_CHECK(rel_bias.size(-1) > 0, "relative bias extent must be positive");
  }

  // This needs to go before kBlockM & kBlockN since we rely on the correct window_size and is_causal to set kBlockM
  // TODO: check this

  if (window_size_left >= seqlen_k - 1) {
    window_size_left = -1;
  }
  window_size_right = min(window_size_right, seqlen_q);
  // causal=true is the same as causal=false in this case
  if (is_causal) {
    window_size_right = 0;
  }

  CHECK_SHAPE(k, num_pages, page_size, num_heads_k, head_size);
  CHECK_SHAPE(v, num_pages, page_size, num_heads_k, head_size_v);
  CHECK_SHAPE(page_table.value(), batch_size_k, max_num_pages_per_seq);

  if (leftpad_k_.has_value()) {
    auto leftpad_k = leftpad_k_.value();
    TORCH_CHECK(leftpad_k.dtype() == torch::kInt32, "leftpad_k must have dtype int32");
    CHECK_INPUT(leftpad_k);
    CHECK_SHAPE(leftpad_k, batch_size);
  }

  static constexpr int alignment = 8;
  TORCH_CHECK(head_size % alignment == 0, "head_size should be a multiple of " + std::to_string(alignment));
  TORCH_CHECK(head_size_v % alignment == 0, "head_size_v should be a multiple of " + std::to_string(alignment));

  // ``out`` is caller-provided and written in place. Whether to compute the
  // softmax logsumexp is derived from the caller-provided ``softmax_lse``
  // optional (a present tensor requests it; std::nullopt skips it).
  bool const return_softmax_lse = softmax_lse.has_value();

  int const head_size_rounded = round_up_headdim(head_size);
  int const head_size_v_rounded = head_size_v == head_size ? head_size_rounded : round_up_headdim(head_size_v);

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  c10::DeviceGuard device_guard(q.device());

  // ``softmax_lse`` is caller-provided; empty (numel == 0) when the LSE was not
  // requested, in which case the kernel skips every LSE write.

  // align with FA3
  Arguments params;
  params.is_bf16 = q.dtype() == torch::kBFloat16;
  params.is_fp16 = q.dtype() == torch::kHalf;

  // Set the pointers and strides.
  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  // All stride are in elements, not bytes.
  params.q_row_stride = q.stride(-3);
  params.k_row_stride = k.stride(-3);
  params.v_row_stride = v.stride(-3);
  params.q_head_stride = q.stride(-2);
  params.k_head_stride = k.stride(-2);
  params.v_head_stride = v.stride(-2);
  params.v_dim_stride = v.stride(-1);
  params.o_ptr = out.data_ptr();
  params.o_row_stride = out.stride(-3);
  params.o_head_stride = out.stride(-2);
  at::Tensor sheared_rel_bias;
  if (rel_bias_.has_value()) {
    const at::Tensor& rel_bias = rel_bias_.value();
    if (q.scalar_type() == at::ScalarType::BFloat16) {
      sheared_rel_bias = flash_attention_v2::relative_attention::prepare_bias<at::BFloat16>(
          rel_bias, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k);
    } else {
      sheared_rel_bias = flash_attention_v2::relative_attention::prepare_bias<at::Half>(
          rel_bias, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k);
    }
    params.rel_bias_ptr = sheared_rel_bias.data_ptr();
    params.rel_bias_token_stride = sheared_rel_bias.stride(0);
    params.rel_bias_head_stride = sheared_rel_bias.stride(1);
    params.rel_bias_extent = rel_bias.size(-1);
  }

  // Per-batch skip mask for the chunkprefill two-launch dispatcher.
  params.skip_batch_mask_ptr = skip_batch_mask_opt.has_value() ? skip_batch_mask_opt->data_ptr() : nullptr;

  params.cu_seqlens_q = cu_seqlens_q.data_ptr<int>();
  params.cu_seqlens_k = cu_seqlens_k.data_ptr<int>();

  // Softmax sum (null when the caller passes std::nullopt).
  params.softmax_lse_ptr = return_softmax_lse ? softmax_lse->data_ptr() : nullptr;
  params.return_softmax_lse = return_softmax_lse;

  // Set the dimensions.
  params.b = batch_size;
  params.h = num_heads;
  params.h_k = num_heads_k;
  params.q_group_size = 1;
  params.seqlen_q = seqlen_q;
  params.seqlen_k = seqlen_k;
  params.d = head_size;
  params.d_rounded = head_size_rounded;

  // Set the different scale values.
  params.softmax_scale = softmax_scale;
  if (sinks_.has_value()) {
    TORCH_CHECK(head_size == 64, "sink is only supported for head_size == 64, got ", head_size);
    params.softmax_sink_ptr = sinks_.value().data_ptr();
  } else {
    params.softmax_sink_ptr = nullptr;
  }

  params.softcap = softcap;

  // FP8 KV cache: flag the kernel dispatch so the fp8 mainloop is selected.
  params.is_e4m3 = is_e4m3_kv;
  params.is_e5m2 = is_e5m2_kv;
  if (is_fp8_kv) {
    TORCH_CHECK(
        k_descale_.has_value() && v_descale_.has_value(), "fp8 KV cache prefill requires k_descale and v_descale");
    // Per-tensor dequant: the kernel only dereferences one float, so accept a
    // true scalar or a tensor whose elements are all the same repeated value.
    params.k_scale_ptr = get_per_tensor_descale_ptr(k_descale_.value(), k, "k_descale", "fp8 KV cache prefill");
    params.v_scale_ptr = get_per_tensor_descale_ptr(v_descale_.value(), v, "v_descale", "fp8 KV cache prefill");
  }

  // Set this to probability of keeping an element to simplify things.
  params.p_dropout = 1.f;

  // Causal is the special case where window_size_right == 0 and window_size_left < 0.
  // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
  params.is_causal = window_size_left < 0 && window_size_right == 0;
  params.is_local = (window_size_left >= 0 || window_size_right >= 0) && !params.is_causal;

  // TODO: check this
  if (window_size_left < 0) {
    window_size_left = seqlen_k - 1;
  }
  if (window_size_right < 0) {
    window_size_right = seqlen_q - 1;
  }
  params.window_size_left = window_size_left;
  params.window_size_right = window_size_right;
  params.total_q = total_q;
  params.total_k = total_k;
  params.b_k = batch_size_k;
  params.dv = head_size_v;
  params.page_table = page_table.value().data_ptr<int>();
  params.page_table_batch_stride = page_table.value().stride(0);
  params.max_num_pages_per_seq = max_num_pages_per_seq;
  params.page_size = page_size;
  params.num_pages = num_pages;

  if (q_v_.has_value()) {
    TORCH_CHECK(head_size <= 64, "q_v is only supported for head_size <= 64");
    TORCH_CHECK(
        q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16,
        "q_v is only supported for fp16 and bf16 data type");
    TORCH_CHECK(false, "q_v is not supported yet");
    at::Tensor q_v = q_v_.value();
    TORCH_CHECK(q_v.dtype() == q_type, "q_v must have the same dtype as query");
    TORCH_CHECK(q_v.stride(-1) == 1, "q_v tensor must have contiguous last dimension");
    CHECK_SHAPE(q_v, total_q, num_heads, head_size_v);
    params.qv_ptr = q_v.data_ptr();
    // All stride are in elements, not bytes.
    params.qv_row_stride = q_v.stride(-3);
    params.qv_head_stride = q_v.stride(-2);
  }

  if (rotary_cos_.has_value()) {
    auto rotary_cos = rotary_cos_.value();
    CHECK_INPUT(rotary_cos);
    params.rotary_dim = rotary_cos.size(1) * 2;
    TORCH_CHECK(params.rotary_dim <= head_size, "rotary_dim must be <= headdim");
    TORCH_CHECK(params.rotary_dim % 16 == 0, "Only rotary dimensions divisible by 16 are currently supported");
    const int seqlen_ro = rotary_cos.size(0);
    TORCH_CHECK(seqlen_ro >= seqlen_k, "cos/sin seqlen must be at least the seqlen of KV cache");
    CHECK_SHAPE(rotary_cos, seqlen_ro, params.rotary_dim / 2);
    TORCH_CHECK(rotary_cos.scalar_type() == q_type, "rotary_cos must have the same dtype as query");

    TORCH_CHECK(rotary_sin_.has_value(), "If rotary cos is provided, rotary sin must also be provided");
    auto rotary_sin = rotary_sin_.value();
    CHECK_INPUT(rotary_sin);
    CHECK_SHAPE(rotary_sin, seqlen_ro, params.rotary_dim / 2);
    TORCH_CHECK(rotary_sin.scalar_type() == q_type, "rotary_cos must have the same dtype as query");
    params.rotary_cos_ptr = rotary_cos.data_ptr();
    params.rotary_sin_ptr = rotary_sin.data_ptr();
    params.is_rotary_interleaved = is_rotary_interleaved;
    if (seqlens_rotary_.has_value()) {
      at::Tensor seqlens_rotary = seqlens_rotary_.value();
      CHECK_INPUT(seqlens_rotary);
      TORCH_CHECK(seqlens_rotary.dtype() == torch::kInt32, "seqlens_rotary must have dtype torch.int32");
      CHECK_SHAPE(seqlens_rotary, batch_size);
      params.seqlens_rotary = seqlens_rotary.data_ptr<int>();
    }
  } else {
    params.rotary_dim = 0;
  }

  if (kv_batch_idx_.has_value()) {
    auto kv_batch_idx = kv_batch_idx_.value();
    CHECK_INPUT(kv_batch_idx);
    TORCH_CHECK(kv_batch_idx.scalar_type() == torch::kInt32, "kv_batch_idx must have dtype int32");
    params.kv_batch_idx = reinterpret_cast<int*>(kv_batch_idx.data_ptr());
  }

  params.tensor_opts = torch::TensorOptions().dtype(torch::kUInt8).device(q.device());

  // Paged prefill supports its own (independent) set of head dims; see
  // FMHA_PREFILL_PAGED_HEAD_DIMS in FMHAPrefillXe20.cmake.
  TORCH_CHECK(
      params.d == 64 || params.d == 96 || params.d == 128 || params.d == 192 || params.d == 256 || params.d == 512,
      "Unsupported head size for paged prefill attention: ",
      params.d);

  switch (params.d) {
    case 64:
      DISPATCH_PREFILL_KERNEL(64);
      break;
    case 96:
      DISPATCH_PREFILL_KERNEL(96);
      break;
    case 128:
      DISPATCH_PREFILL_KERNEL(128);
      break;
    case 192:
      DISPATCH_PREFILL_KERNEL(192);
      break;
    case 256:
      DISPATCH_PREFILL_KERNEL(256);
      break;
    case 512:
      DISPATCH_PREFILL_KERNEL(512);
      break;
    default:
      TORCH_CHECK(false, "Unsupported head size for paged prefill attention: ", params.d);
  }

  // TODO: Support prefill softmax_lse, now is 0
}

}  // namespace prefill

namespace chunkprefill {

// Two-launch mix-batch dispatcher (vllm-xpu-kernels#218).
//
// Build a per-batch ``is_prefill`` bool mask on device, then launch the
// decode kernel skipping prefill batches and the prefill kernel skipping
// decode batches. Both launches write into the same output tensor.
//
// Limitations: paged KV cache required; rotary / q_v / descale / scheduler
// metadata are not supported on this path. Sliding window and attention sinks
// are forwarded to both sub-kernels, which support them.
void mha_fwd(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    std::optional<const at::Tensor>& q_v_,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,  // per-batch cache_seqlens (size = batch) in paged mode
    int max_seqlen_q,
    int max_seqlen_k,
    std::optional<const at::Tensor>& page_table,
    std::optional<const at::Tensor>& kv_batch_idx_,
    std::optional<const at::Tensor>& leftpad_k_,
    std::optional<const at::Tensor>& rotary_cos_,
    std::optional<const at::Tensor>& rotary_sin_,
    std::optional<const at::Tensor>& seqlens_rotary_,
    std::optional<at::Tensor>& q_descale_,
    std::optional<at::Tensor>& k_descale_,
    std::optional<at::Tensor>& v_descale_,
    const float softmax_scale_,
    std::optional<const at::Tensor>& sinks_,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    float const softcap,
    bool const is_rotary_interleaved,
    std::optional<at::Tensor>& scheduler_metadata_,
    int num_kv_splits,
    std::optional<bool> pack_gqa_,
    int const sm_margin,
    at::Tensor& out,
    std::optional<at::Tensor>& softmax_lse) {
  // Supports both paged (page_table != None) and non-paged (contiguous ragged
  // KV, page_table == None) layouts.
  // ``seqlens_rotary_`` is intentionally not checked here: callers pass it
  // alongside ``cache_seqlens`` even when rotary is disabled, and the
  // sub-kernels only consume it inside the ``rotary_cos_.has_value()`` branch.
  TORCH_CHECK(
      !q_v_.has_value() && !rotary_cos_.has_value() && !rotary_sin_.has_value() && !q_descale_.has_value() &&
          !scheduler_metadata_.has_value(),
      "chunkprefill two-launch path does not yet support q_v / rotary / q_descale / scheduler_metadata.");
  TORCH_CHECK(cu_seqlens_q.scalar_type() == at::kInt, "cu_seqlens_q must be int32.");
  // chunkprefill is only reached for paged KV (the public dispatcher routes the
  // non-paged and single-batch cases to the pure prefill path), so the
  // caller-provided ``out`` buffer is always backed by a page table.
  TORCH_CHECK(page_table.has_value(), "chunkprefill: requires page_table (paged KV cache).");

  int64_t batch_size = cu_seqlens_q.size(0) - 1;
  TORCH_CHECK(batch_size >= 0, "cu_seqlens_q must have at least 1 element.");

  auto seqlens_q = cu_seqlens_q.slice(0, 1, batch_size + 1).sub(cu_seqlens_q.slice(0, 0, batch_size));
  auto is_prefill = seqlens_q.gt(1).contiguous();  // true for prefill batches

  // Forward every shared argument to a sub-kernel, overriding only the per-batch
  // skip mask. Both launches write into the same caller-provided ``out`` and
  // ``softmax_lse`` buffers.
  auto launch = [&](auto&& fn, std::optional<at::Tensor> skip_mask, auto&&... tail) {
    fn(q,
       k,
       v,
       q_v_,
       cu_seqlens_q,
       cu_seqlens_k,
       max_seqlen_q,
       max_seqlen_k,
       page_table,
       kv_batch_idx_,
       leftpad_k_,
       rotary_cos_,
       rotary_sin_,
       seqlens_rotary_,
       q_descale_,
       k_descale_,
       v_descale_,
       softmax_scale_,
       sinks_,
       is_causal,
       window_size_left,
       window_size_right,
       softcap,
       is_rotary_interleaved,
       scheduler_metadata_,
       num_kv_splits,
       pack_gqa_,
       sm_margin,
       out,
       softmax_lse,
       std::move(skip_mask),
       std::forward<decltype(tail)>(tail)...);
  };

  // Launch 1: decode writes the decode rows of ``out`` (and, when requested, of
  // ``softmax_lse``) and skips prefill batches, leaving their rows untouched.
  launch(decode::mha_fwd, is_prefill, std::nullopt, false);
  // Launch 2: prefill writes the prefill rows into the same buffers and skips
  // decode batches. The two complementary skip masks partition every query token
  // across the launches, so the shared buffers end up fully written with no
  // stitching or extra copies needed.
  launch(prefill::mha_fwd, is_prefill.logical_not(), std::nullopt);
}

}  // namespace chunkprefill

SGL_KERNEL_EXPORT void mha_fwd(
    const at::Tensor& q,  // (total_q, h, d) — ragged 3D
    const at::Tensor& k,  // (total_k, h_k, d) if non-paged, or (num_pages, page_size, h_k, d) if paged
    const at::Tensor& v,  // (total_k, h_k, dv) if non-paged, or (num_pages, page_size, h_k, dv) if paged
    std::optional<const at::Tensor>& q_v_,  // (total_q, h, dv) — not yet supported
    const at::Tensor& cu_seqlens_q,         // b+1
    const at::Tensor& cu_seqlens_k,         // b+1
    int max_seqlen_q,
    int max_seqlen_k,
    std::optional<const at::Tensor>& page_table,       // (b_k, max_num_pages_per_seq)
    std::optional<const at::Tensor>& kv_batch_idx_,    // b. indices to index into the KV cache
    std::optional<const at::Tensor>& leftpad_k_,       // b
    std::optional<const at::Tensor>& rotary_cos_,      // seqlen_ro x (rotary_dim / 2)
    std::optional<const at::Tensor>& rotary_sin_,      // seqlen_ro x (rotary_dim / 2)
    std::optional<const at::Tensor>& seqlens_rotary_,  // b
    std::optional<at::Tensor>& q_descale_,             // (b, h_k), not (b, h)
    std::optional<at::Tensor>& k_descale_,             // (b, h_k)
    std::optional<at::Tensor>& v_descale_,             // (b, h_k)
    const float softmax_scale_,
    std::optional<const at::Tensor>& sinks_,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    float const softcap,
    bool const is_rotary_interleaved,  // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
    std::optional<at::Tensor>& scheduler_metadata_,  // (b + 1)
    int num_kv_splits,
    std::optional<bool> pack_gqa_,
    int const sm_margin,
    // Caller-provided output buffers, written in place (no value is returned).
    // ``softmax_lse`` doubles as the "return LSE" flag: a present optional
    // requests the logsumexp be computed and written; std::nullopt skips the
    // LSE computation entirely.
    at::Tensor& out,
    std::optional<at::Tensor>& softmax_lse,
    std::optional<const at::Tensor>& rel_bias_,
    bool rel_bias_is_sheared) {
  TORCH_CHECK(q.dim() == 3, "query must be in ragged format (total_q, h, d)");
  // k and v may be 3D (total_k, h_k, d) for non-paged or 4D (num_pages, page_size, h_k, d)
  // for paged KV cache; sub-functions validate their own shapes.
  TORCH_CHECK(out.scalar_type() == q.scalar_type(), "out dtype must match q dtype");
  TORCH_CHECK(
      out.dim() == 3 && out.size(0) == q.size(0) && out.size(1) == q.size(1) && out.size(2) == v.size(-1),
      "out shape must be [total_q, num_heads, head_size_v]");
  TORCH_CHECK(out.device() == q.device(), "out must be on the same device as q");
  TORCH_CHECK(out.stride(-1) == 1, "out must have a contiguous last dimension");

  // Whether to compute the softmax logsumexp is derived from the caller-provided
  // ``softmax_lse`` optional: a present tensor requests it. The sub-kernels
  // derive the same flag and write into the buffer directly.
  if (softmax_lse.has_value()) {
    TORCH_CHECK(softmax_lse->scalar_type() == at::kFloat, "softmax_lse must be float32");
    TORCH_CHECK(
        softmax_lse->dim() == 2 && softmax_lse->size(0) == q.size(1) && softmax_lse->size(1) == q.size(0),
        "softmax_lse shape must be [num_heads, total_q]");
    TORCH_CHECK(softmax_lse->device() == q.device(), "softmax_lse must be on the same device as q");
  }

  int64_t batch_size = cu_seqlens_q.size(0) - 1;

  if (rel_bias_.has_value() && max_seqlen_q != 1) {
    TORCH_CHECK(!rel_bias_is_sheared, "pre-sheared relative bias is supported only for decode");
    // Relative prefill shares the same public bias contract. Single-token rows
    // use the decode-specific zero-drift surface below.
    prefill::mha_fwd(
        q,
        k,
        v,
        q_v_,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        page_table,
        kv_batch_idx_,
        leftpad_k_,
        rotary_cos_,
        rotary_sin_,
        seqlens_rotary_,
        q_descale_,
        k_descale_,
        v_descale_,
        softmax_scale_,
        sinks_,
        is_causal,
        window_size_left,
        window_size_right,
        softcap,
        is_rotary_interleaved,
        scheduler_metadata_,
        num_kv_splits,
        pack_gqa_,
        sm_margin,
        out,
        softmax_lse,
        std::nullopt,
        rel_bias_);
    return;
  }

  // decode / prefill / chunkprefill all take the same leading argument list;
  // only the trailing parameters differ. Bind the shared arguments once here so
  // each branch reduces to a single call. ``out`` and ``softmax_lse`` are
  // threaded by reference and written in place; ``tail`` carries the
  // callee-specific suffix: decode and prefill additionally accept a per-batch
  // skip mask (unused at this top level, so left as std::nullopt); chunkprefill
  // has no such slot.
  auto dispatch = [&](auto&& fn, auto&&... tail) {
    fn(q,
       k,
       v,
       q_v_,
       cu_seqlens_q,
       cu_seqlens_k,
       max_seqlen_q,
       max_seqlen_k,
       page_table,
       kv_batch_idx_,
       leftpad_k_,
       rotary_cos_,
       rotary_sin_,
       seqlens_rotary_,
       q_descale_,
       k_descale_,
       v_descale_,
       softmax_scale_,
       sinks_,
       is_causal,
       window_size_left,
       window_size_right,
       softcap,
       is_rotary_interleaved,
       scheduler_metadata_,
       num_kv_splits,
       pack_gqa_,
       sm_margin,
       out,
       softmax_lse,
       std::forward<decltype(tail)>(tail)...);
  };

  if (max_seqlen_q == 1) {
    // Pure decode path
    decode::mha_fwd(
        q,
        k,
        v,
        q_v_,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        page_table,
        kv_batch_idx_,
        leftpad_k_,
        rotary_cos_,
        rotary_sin_,
        seqlens_rotary_,
        q_descale_,
        k_descale_,
        v_descale_,
        softmax_scale_,
        sinks_,
        is_causal,
        window_size_left,
        window_size_right,
        softcap,
        is_rotary_interleaved,
        scheduler_metadata_,
        num_kv_splits,
        pack_gqa_,
        sm_margin,
        out,
        softmax_lse,
        std::nullopt,
        rel_bias_,
        rel_bias_is_sheared);
  } else if (!page_table.has_value() || batch_size == 1) {
    // Pure prefill path
    // Non-paged attn: assumption of all seqlen_q > 1;
    // Paged attn: Proving "all prefill" for batch_size > 1 would require
    // is_prefill.all() — a device reduction + D2H sync that costs more than it saves.
    // But batch_size == 1 makes it provable from host scalars:
    // a single sequence with max_seqlen_q > 1 is prefill
    dispatch(prefill::mha_fwd, std::nullopt, std::nullopt);
  } else {
    // Chunk prefill path
    // Paged attn with max_seqlen_q > 1 and batch_size > 1
    dispatch(chunkprefill::mha_fwd);
  }
}
#undef SYCL_INTEL_TARGET
