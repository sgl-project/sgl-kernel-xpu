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

#include "Utils.h"
#include "kernels/flash_attention_v2/xe_fmha_fwd_decode_dispatch.hpp"
#include "kernels/flash_attention_v2/xe_fmha_fwd_prefill_dispatch.hpp"

namespace {

template <typename scalar_t>
struct StorePagedAppendKVKernel {
  void operator()(sycl::nd_item<1> item) const {
    int const lane = item.get_local_id(0);
    int const group_size = item.get_local_range(0);
    int const token_begin = item.get_group(0) * kTokensPerGroup;
    int const token_end = sycl::min(token_begin + kTokensPerGroup, total_k_new);

    for (int token = token_begin; token < token_end; ++token) {
      int physical_page = 0;
      int page_offset = 0;
      if (lane == 0) {
        int batch_begin = 0;
        int batch_end = batch_size;
        while (batch_begin + 1 < batch_end) {
          int const batch_mid = (batch_begin + batch_end) / 2;
          if (token < cu_seqlens_k_new[batch_mid]) {
            batch_end = batch_mid;
          } else {
            batch_begin = batch_mid;
          }
        }
        int const batch = batch_begin;
        int const token_in_batch = token - cu_seqlens_k_new[batch];
        int const cache_pos = cache_seqlens_old[batch] + token_in_batch;
        int const logical_page = cache_pos / page_size;
        page_offset = cache_pos - logical_page * page_size;
        physical_page = page_table[batch * page_table_batch_stride + logical_page];
      }
      physical_page = sycl::group_broadcast(item.get_group(), physical_page, 0);
      page_offset = sycl::group_broadcast(item.get_group(), page_offset, 0);
      using pack_t = sycl::vec<uint32_t, 4>;
      constexpr int kVecWidth = sizeof(pack_t) / sizeof(scalar_t);
      auto* k_src = reinterpret_cast<uint32_t*>(const_cast<scalar_t*>(k_new + token * k_new_row_stride));
      auto* v_src = reinterpret_cast<uint32_t*>(const_cast<scalar_t*>(v_new + token * v_new_row_stride));
      auto* k_dst = reinterpret_cast<uint32_t*>(
          k_cache + physical_page * k_cache_page_stride + page_offset * k_cache_row_stride);
      auto* v_dst = reinterpret_cast<uint32_t*>(
          v_cache + physical_page * v_cache_page_stride + page_offset * v_cache_row_stride);

      int const k_pack_count = num_heads_kv * head_size / kVecWidth;
      int const v_pack_count = num_heads_kv * head_size_v / kVecWidth;
      int const common_pack_count = sycl::min(k_pack_count, v_pack_count);
      for (int i = lane; i < common_pack_count; i += group_size) {
        pack_t k_value;
        pack_t v_value;
        k_value.load(i, k_src);
        v_value.load(i, v_src);
        k_value.store(i, k_dst);
        v_value.store(i, v_dst);
      }
      for (int i = common_pack_count + lane; i < k_pack_count; i += group_size) {
        pack_t value;
        value.load(i, k_src);
        value.store(i, k_dst);
      }
      for (int i = common_pack_count + lane; i < v_pack_count; i += group_size) {
        pack_t value;
        value.load(i, v_src);
        value.store(i, v_dst);
      }
    }
  }

  static constexpr int kTokensPerGroup = 1;
  scalar_t const* k_new;
  scalar_t const* v_new;
  scalar_t* k_cache;
  scalar_t* v_cache;
  int const* cu_seqlens_k_new;
  int const* cache_seqlens_old;
  int const* page_table;
  int batch_size;
  int total_k_new;
  int page_size;
  int page_table_batch_stride;
  int num_heads_kv;
  int head_size;
  int head_size_v;
  int64_t k_new_row_stride;
  int64_t v_new_row_stride;
  int64_t k_cache_page_stride;
  int64_t k_cache_row_stride;
  int64_t v_cache_page_stride;
  int64_t v_cache_row_stride;
};

void store_paged_append_kv(
    const at::Tensor& k_new,
    const at::Tensor& v_new,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    const at::Tensor& cu_seqlens_k_new,
    const at::Tensor& cache_seqlens_old,
    const at::Tensor& page_table) {
  int const total_k_new = k_new.size(0);
  if (total_k_new == 0) {
    return;
  }
  int const batch_size = cu_seqlens_k_new.size(0) - 1;
  int const row_dim = k_cache.size(2) * std::max(k_cache.size(3), v_cache.size(3));
  int const group_size =
      std::min<int>(std::min<int>(row_dim, 512), dpcppMaxWorkGroupSize(dpcppGetDeviceIdOfCurrentQueue()));
  SYCL_DISPATCH_FLOATING_TYPES(
      at::ScalarType::Half, at::ScalarType::BFloat16, k_new.scalar_type(), "store_paged_append_kv", [&]() {
        StorePagedAppendKVKernel<scalar_t> kernel{
            .k_new = k_new.data_ptr<scalar_t>(),
            .v_new = v_new.data_ptr<scalar_t>(),
            .k_cache = const_cast<scalar_t*>(k_cache.data_ptr<scalar_t>()),
            .v_cache = const_cast<scalar_t*>(v_cache.data_ptr<scalar_t>()),
            .cu_seqlens_k_new = cu_seqlens_k_new.data_ptr<int>(),
            .cache_seqlens_old = cache_seqlens_old.data_ptr<int>(),
            .page_table = page_table.data_ptr<int>(),
            .batch_size = batch_size,
            .total_k_new = total_k_new,
            .page_size = static_cast<int>(k_cache.size(1)),
            .page_table_batch_stride = static_cast<int>(page_table.stride(0)),
            .num_heads_kv = static_cast<int>(k_cache.size(2)),
            .head_size = static_cast<int>(k_cache.size(3)),
            .head_size_v = static_cast<int>(v_cache.size(3)),
            .k_new_row_stride = k_new.stride(0),
            .v_new_row_stride = v_new.stride(0),
            .k_cache_page_stride = k_cache.stride(0),
            .k_cache_row_stride = k_cache.stride(1),
            .v_cache_page_stride = v_cache.stride(0),
            .v_cache_row_stride = v_cache.stride(1)};
        dpcppGetCurrentQueue().submit([&](sycl::handler& cgh) {
          cgh.parallel_for(
              sycl::nd_range<1>(
                  sycl::range<1>(
                      static_cast<size_t>((total_k_new + StorePagedAppendKVKernel<scalar_t>::kTokensPerGroup - 1) /
                                          StorePagedAppendKVKernel<scalar_t>::kTokensPerGroup) *
                      group_size),
                  sycl::range<1>(group_size)),
              kernel);
        });
      });
}

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
std::vector<at::Tensor> mha_fwd_nopage(
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
    std::optional<at::Tensor> out_opt,
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

  auto opts = q.options();
  // Use the caller-provided shared output when present (two-launch path); the
  // first launch zero-initializes so that rows of batches with zero KV length
  // (never written by the kernel) read back their correct value of 0.
  at::Tensor out = out_opt.has_value() ? *out_opt : torch::zeros({total_q, num_heads, head_size_v}, opts);

  int const head_size_rounded = round_up_headdim(head_size);

  c10::DeviceGuard device_guard(q.device());

  at::Tensor softmax_lse = torch::empty({num_heads, total_q}, opts.dtype(at::kFloat));

  Arguments params;
  params.is_bf16 = q.dtype() == torch::kBFloat16;

  // Q / O are in ragged (total, h, d) format; KV is a contiguous ragged
  // (total_k, h_k, d) cache addressed via cu_seqlens_k offsets.
  params.q_ptr = q.data_ptr();
  params.k_cache_ptr = k.data_ptr();
  params.v_cache_ptr = v.data_ptr();
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
  params.cu_seqlens_k_cache = cu_seqlens_k.data_ptr<int>();
  // No "new" KV: the whole sequence lives in the contiguous cache buffer, so the
  // decode kernel reads everything from the K/V cache pointers (knew = 0).
  params.k_ptr = nullptr;
  params.v_ptr = nullptr;
  params.cu_seqlens_k = nullptr;
  params.seqlen_k = 0;
  params.total_k = 0;

  params.softmax_lse_ptr = softmax_lse.data_ptr();

  params.b = batch_size;
  params.h = num_heads;
  params.h_k = num_heads_k;
  // GQA packing: the decode kernel folds q_group_size query heads into the Q
  // tile, matching the paged decode path.
  params.q_group_size = num_heads / num_heads_k;
  params.seqlen_q = seqlen_q;
  params.seqlen_k_cache = seqlen_k;
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
  params.is_e4m3 = false;
  params.is_e5m2 = false;

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
  params.total_k_cache = total_k;
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

  at::Tensor out_accum, softmax_lse_accum;

  int qg_sz = nextPowerOf2(params.q_group_size);
  TORCH_CHECK(qg_sz >= 1 && qg_sz <= 16, "Unsupported q_group_size for decode attention: ", params.q_group_size);
  // Non-paged decode supports its own (independent) set of head dims; see
  // FMHA_DECODE_NP_HEAD_DIMS in FMHADecodeXe20.cmake.
  TORCH_CHECK(
      params.d == 64 || params.d == 72 || params.d == 80 || params.d == 96 || params.d == 128 || params.d == 192,
      "Unsupported head size for non-paged decode attention: ",
      params.d);

  DISPATCH_DECODE_NOPAGE(qg_sz);

  return {out, softmax_lse, out_accum, softmax_lse_accum};
}

std::vector<at::Tensor> mha_fwd(
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
    // chunkprefill two-launch path: pre-allocated shared output, and a per-batch
    // bool mask (length = batch) whose true entries are skipped by the kernel.
    std::optional<at::Tensor> out_opt = std::nullopt,
    std::optional<at::Tensor> skip_batch_mask_opt = std::nullopt,
    std::optional<at::Tensor> softmax_lse_opt = std::nullopt,
    std::optional<const at::Tensor> cache_seqlens_delta_opt = std::nullopt) {
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
    return mha_fwd_nopage(
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
        std::move(out_opt),
        std::move(skip_batch_mask_opt));
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
  int const seqlen_k = page_table.has_value() ? max_num_pages_per_seq * page_size : max_seqlen_k;
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

  auto opts = q.options();
  at::Tensor out;
  at::Tensor temp_out;    // [batch, num_kv_splits, num_head_q, seq_q, head_size]
  at::Tensor exp_sums;    // [batch, num_head_q, seq_q, num_kv_splits]
  at::Tensor max_logits;  // [batch, num_head_q, seq_q, num_kv_splits]
  out = out_opt.has_value() ? *out_opt : torch::empty({total_q, num_heads, head_size_v}, opts);
  Arguments params;
  // num_kv_splits semantics (host-side scalar, no D2H sync):
  //   -1 or 1 -> split-KV disabled, use the non-split FmhaDecodeRunner
  //         0 -> auto: pick a split count from the device-occupancy heuristic
  //        >1 -> use the caller-provided split count with FmhaSplitDecodeRunner
  if (num_kv_splits == 0) {
    auto get_num_splits = [](int batch_size, int num_heads_kv, int max_seqlen_k, int block_size) {
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
      // Split-KV adds a separate reduction launch whose cost is roughly fixed.
      // Benchmarks (benchmark/bench_flash_attn_split_decode.py) show that on the
      // decode path splitting only pays off once the KV cache spans more than
      // ~64 pages; below that the occupancy-only heuristic over-splits short
      // sequences and the non-split runner is 20-40% faster. Gate on total work.
      constexpr int kMinBlocksToSplit = 64;
      if (total_blocks <= kMinBlocksToSplit) {
        return 1;
      }

      int max_splits = std::min(total_blocks, parallel_);
      return std::min(num_splits, max_splits);
    };
    num_kv_splits = get_num_splits(batch_size, num_heads_k, seqlen_k, page_size);
  }
  // Only split when the resolved count is > 1; -1 / 1 fall back to non-split.
  params.use_split_kv = num_kv_splits > 1;
  if (params.use_split_kv) {
    temp_out = torch::empty({total_q, num_kv_splits * num_heads, head_size_v}, q.options().device(q.device()));

    max_logits = torch::full(
        {total_q, num_heads, num_kv_splits},
        -std::numeric_limits<float>::infinity(),
        q.options().dtype(at::kFloat).device(q.device()));

    exp_sums = torch::zeros({total_q, num_heads, num_kv_splits}, q.options().dtype(at::kFloat).device(q.device()));

    params.temp_out_ptr = temp_out.data_ptr();
    params.exp_sums_ptr = exp_sums.data_ptr();
    params.max_logits_ptr = max_logits.data_ptr();
  }
  int const head_size_rounded = round_up_headdim(head_size);
  int const head_size_v_rounded = head_size_v == head_size ? head_size_rounded : round_up_headdim(head_size_v);

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  c10::DeviceGuard device_guard(q.device());

  at::Tensor softmax_lse =
      softmax_lse_opt.has_value() ? *softmax_lse_opt : torch::empty({num_heads, total_q}, opts.dtype(at::kFloat));

  // align with FA3

  params.is_bf16 = q.dtype() == torch::kBFloat16;

  // Set the pointers and strides.
  params.q_ptr = q.data_ptr();
  params.k_cache_ptr = k.data_ptr();
  params.v_cache_ptr = v.data_ptr();
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

  // Per-batch skip mask for the chunkprefill two-launch path
  // (vllm-xpu-kernels#218). When provided, decode skips batches where
  // mask[idx_b] == true (i.e. the prefill rows).
  params.skip_batch_mask_ptr = skip_batch_mask_opt.has_value() ? skip_batch_mask_opt->data_ptr() : nullptr;

  params.cu_seqlens_q = cu_seqlens_q.data_ptr<int>();
  params.cu_seqlens_k_cache = cu_seqlens_k.data_ptr<int>();
  params.k_ptr = nullptr;
  params.v_ptr = nullptr;
  params.cu_seqlens_k = nullptr;
  params.seqlen_k = 0;
  params.total_k = 0;
  if (cache_seqlens_delta_opt.has_value()) {
    auto const& cache_seqlens_delta = *cache_seqlens_delta_opt;
    CHECK_INPUT(cache_seqlens_delta);
    TORCH_CHECK(cache_seqlens_delta.dtype() == torch::kInt32, "cache_seqlens_delta must have dtype torch.int32");
    CHECK_SHAPE(cache_seqlens_delta, batch_size + 1);
    params.cu_seqlens_k = cache_seqlens_delta.data_ptr<int>();
  }
  params.num_kv_splits = num_kv_splits;

  // Softmax sum
  params.softmax_lse_ptr = softmax_lse.data_ptr();

  // Set the dimensions.
  params.b = batch_size;
  params.h = num_heads;
  params.h_k = num_heads_k;
  params.q_group_size = num_heads / num_heads_k;
  params.seqlen_q = seqlen_q;
  params.seqlen_k_cache = seqlen_k;
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
  params.total_k_cache = total_k;
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

  at::Tensor out_accum, softmax_lse_accum;

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

  return {out, softmax_lse, out_accum, softmax_lse_accum};
}

}  // namespace decode

namespace prefill {

// Non-paged (contiguous ragged KV) prefill entry. Drives both the prefill and
// the decode sub-launches of the no-page chunkprefill two-launch path: the
// caller passes a shared output (out_opt) and a per-batch skip mask
// (skip_batch_mask_opt) selecting which batches this launch processes.
std::vector<at::Tensor> mha_fwd_nopage(
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
    std::optional<at::Tensor> out_opt,
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

  auto opts = q.options();
  // Use the caller-provided shared output when present (two-launch path); the
  // first launch zero-initializes so that rows of batches with zero KV length
  // (never written by the kernel) read back their correct value of 0.
  at::Tensor out = out_opt.has_value() ? *out_opt : torch::zeros({total_q, num_heads, head_size_v}, opts);

  int const head_size_rounded = round_up_headdim(head_size);

  c10::DeviceGuard device_guard(q.device());

  at::Tensor softmax_lse = torch::empty({num_heads, total_q}, opts.dtype(at::kFloat));

  Arguments params;
  params.is_bf16 = q.dtype() == torch::kBFloat16;

  params.q_ptr = q.data_ptr();
  params.k_cache_ptr = k.data_ptr();
  params.v_cache_ptr = v.data_ptr();
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
  params.cu_seqlens_k_cache = cu_seqlens_k.data_ptr<int>();
  params.k_ptr = nullptr;
  params.v_ptr = nullptr;
  params.cu_seqlens_k = nullptr;
  params.cache_seqlens_old = nullptr;
  params.seqlen_k = 0;
  params.total_k = 0;

  params.softmax_lse_ptr = softmax_lse.data_ptr();

  params.b = batch_size;
  params.h = num_heads;
  params.h_k = num_heads_k;
  params.q_group_size = 1;
  params.seqlen_q = seqlen_q;
  params.seqlen_k_cache = seqlen_k;
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
  params.total_k_cache = total_k;
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

  at::Tensor out_accum, softmax_lse_accum;

  // Non-paged prefill supports its own (independent) set of head dims; see
  // FMHA_PREFILL_NP_HEAD_DIMS in FMHAPrefillXe20.cmake.
  TORCH_CHECK(
      params.d == 64 || params.d == 72 || params.d == 80 || params.d == 96 || params.d == 128 || params.d == 192,
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
    default:
      TORCH_CHECK(false, "Unsupported head size for non-paged prefill attention: ", params.d);
  }

  return {out, softmax_lse, out_accum, softmax_lse_accum};
}

std::vector<at::Tensor> mha_fwd_appendkv(
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
    // chunkprefill two-launch path: pre-allocated shared output, and a per-batch
    // bool mask (length = batch) whose true entries are skipped by the kernel.
    std::optional<at::Tensor> out_opt = std::nullopt,
    std::optional<at::Tensor> skip_batch_mask_opt = std::nullopt,
    std::optional<const at::Tensor> k_new_ = std::nullopt,
    std::optional<const at::Tensor> v_new_ = std::nullopt,
    std::optional<const at::Tensor> cu_seqlens_k_new_ = std::nullopt,
    std::optional<at::Tensor> softmax_lse_opt = std::nullopt,
    std::optional<const at::Tensor> cache_seqlens_delta_opt = std::nullopt) {
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

  bool const has_new_kv = k_new_.has_value() || v_new_.has_value() || cu_seqlens_k_new_.has_value();

  // Non-paged (page_table == nullopt) prefill: contiguous ragged KV cache.
  if (!page_table.has_value()) {
    TORCH_CHECK(!has_new_kv, "AppendKV requires paged KV cache");
    return mha_fwd_nopage(
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
        std::move(out_opt),
        std::move(skip_batch_mask_opt));
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

  auto opts = q.options();
  at::Tensor out;
  out = out_opt.has_value() ? *out_opt : torch::empty({total_q, num_heads, head_size_v}, opts);

  int const head_size_rounded = round_up_headdim(head_size);
  int const head_size_v_rounded = head_size_v == head_size ? head_size_rounded : round_up_headdim(head_size_v);

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  c10::DeviceGuard device_guard(q.device());

  at::Tensor softmax_lse =
      softmax_lse_opt.has_value() ? *softmax_lse_opt : torch::empty({num_heads, total_q}, opts.dtype(at::kFloat));

  // align with FA3
  Arguments params;
  params.is_bf16 = q.dtype() == torch::kBFloat16;

  // Set the pointers and strides.
  params.q_ptr = q.data_ptr();
  params.k_cache_ptr = k.data_ptr();
  params.v_cache_ptr = v.data_ptr();
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

  // Per-batch skip mask for the chunkprefill two-launch dispatcher.
  params.skip_batch_mask_ptr = skip_batch_mask_opt.has_value() ? skip_batch_mask_opt->data_ptr() : nullptr;

  params.cu_seqlens_q = cu_seqlens_q.data_ptr<int>();
  params.cu_seqlens_k_cache = cu_seqlens_k.data_ptr<int>();

  // Softmax sum
  params.softmax_lse_ptr = softmax_lse.data_ptr();

  // Set the dimensions.
  params.b = batch_size;
  params.h = num_heads;
  params.h_k = num_heads_k;
  params.q_group_size = 1;
  params.seqlen_q = seqlen_q;
  params.seqlen_k_cache = seqlen_k;
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
  params.total_k_cache = total_k;
  params.b_k = batch_size_k;
  params.dv = head_size_v;
  params.page_table = page_table.value().data_ptr<int>();
  params.page_table_batch_stride = page_table.value().stride(0);
  params.max_num_pages_per_seq = max_num_pages_per_seq;
  params.page_size = page_size;
  params.num_pages = num_pages;

  params.k_ptr = nullptr;
  params.v_ptr = nullptr;
  params.cu_seqlens_k = nullptr;
  params.cache_seqlens_old = nullptr;
  params.seqlen_k = 0;
  params.total_k = 0;
  if (has_new_kv) {
    TORCH_CHECK(k_new_.has_value() && v_new_.has_value(), "AppendKV requires both k_new and v_new");
    auto const& k_new = k_new_.value();
    auto const& v_new = v_new_.value();
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(k_new);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(v_new);
    TORCH_CHECK(k_new.scalar_type() == k.scalar_type(), "k_new dtype must match KV cache key dtype");
    TORCH_CHECK(v_new.scalar_type() == v.scalar_type(), "v_new dtype must match KV cache value dtype");
    TORCH_CHECK(k_new.dim() == 3 || k_new.dim() == 4, "k_new must be [total_k_new, h_k, d] or [b, s, h_k, d]");
    TORCH_CHECK(v_new.dim() == k_new.dim(), "v_new rank must match k_new rank");
    int total_knew = 0;
    int seqlen_knew = max_seqlen_k > 0 ? max_seqlen_k : max_seqlen_q;
    if (k_new.dim() == 3) {
      total_knew = k_new.size(0);
      CHECK_SHAPE(k_new, total_knew, num_heads_k, head_size);
      CHECK_SHAPE(v_new, total_knew, num_heads_k, head_size_v);
      TORCH_CHECK(
          cu_seqlens_k_new_.has_value() || seqlen_knew > 0,
          "ragged k_new requires cu_seqlens_k_new or positive max_seqlen_k");
    } else {
      TORCH_CHECK(k_new.size(0) == batch_size, "batched k_new first dimension must match batch size");
      int const k_new_seqlen = k_new.size(1);
      total_knew = batch_size * k_new_seqlen;
      seqlen_knew = max_seqlen_k > 0 ? max_seqlen_k : k_new_seqlen;
      CHECK_SHAPE(k_new, batch_size, k_new_seqlen, num_heads_k, head_size);
      CHECK_SHAPE(v_new, batch_size, k_new_seqlen, num_heads_k, head_size_v);
    }
    if (cu_seqlens_k_new_.has_value()) {
      auto const& cu_seqlens_k_new = cu_seqlens_k_new_.value();
      CHECK_INPUT(cu_seqlens_k_new);
      TORCH_CHECK(cu_seqlens_k_new.dtype() == torch::kInt32, "cu_seqlens_k_new must have dtype torch.int32");
      CHECK_SHAPE(cu_seqlens_k_new, batch_size + 1);
      params.cu_seqlens_k = cu_seqlens_k_new.data_ptr<int>();
    }
    params.k_ptr = k_new.data_ptr();
    params.v_ptr = v_new.data_ptr();
    params.cache_seqlens_old = cu_seqlens_k.data_ptr<int>();
    params.seqlen_k = seqlen_knew;
    params.total_k = total_knew;
  }
  if (cache_seqlens_delta_opt.has_value()) {
    TORCH_CHECK(!has_new_kv, "cache_seqlens_delta cannot be combined with fused AppendKV");
    auto const& cache_seqlens_delta = *cache_seqlens_delta_opt;
    CHECK_INPUT(cache_seqlens_delta);
    TORCH_CHECK(cache_seqlens_delta.dtype() == torch::kInt32, "cache_seqlens_delta must have dtype torch.int32");
    CHECK_SHAPE(cache_seqlens_delta, batch_size + 1);
    params.cu_seqlens_k = cache_seqlens_delta.data_ptr<int>();
  }

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

  at::Tensor out_accum, softmax_lse_accum;

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

  return {out, softmax_lse, out_accum, softmax_lse_accum};
}

std::vector<at::Tensor> mha_fwd(
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
    std::optional<at::Tensor> out_opt = std::nullopt,
    std::optional<at::Tensor> skip_batch_mask_opt = std::nullopt,
    std::optional<at::Tensor> softmax_lse_opt = std::nullopt,
    std::optional<const at::Tensor> cache_seqlens_delta_opt = std::nullopt) {
  return mha_fwd_appendkv(
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
      num_splits,
      pack_gqa_,
      sm_margin,
      std::move(out_opt),
      std::move(skip_batch_mask_opt),
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::move(softmax_lse_opt),
      std::move(cache_seqlens_delta_opt));
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
std::vector<at::Tensor> mha_fwd(
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
    std::optional<at::Tensor> out_ = std::nullopt,
    std::optional<const at::Tensor> cache_seqlens_delta_opt = std::nullopt) {
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
  // Pre-allocated out requires paged KV: on the non-paged path zero-KV-length
  // rows are never written by the kernel, so a caller buffer would retain stale
  // values on graph replay. SGLang always provides page_table (paged KV cache),
  // so this check should never fire in practice.
  TORCH_CHECK(
      !out_.has_value() || page_table.has_value(), "chunkprefill: out buffer requires page_table (paged KV cache).");

  int64_t batch_size = cu_seqlens_q.size(0) - 1;
  TORCH_CHECK(batch_size >= 0, "cu_seqlens_q must have at least 1 element.");

  auto seqlens_q = cu_seqlens_q.slice(0, 1, batch_size + 1).sub(cu_seqlens_q.slice(0, 0, batch_size));
  auto is_prefill = seqlens_q.gt(1).contiguous();  // true for prefill batches
  // Forward every shared argument to a sub-kernel, overriding only the output
  // tensor and the per-batch skip mask.
  auto launch = [&](auto&& fn, std::optional<at::Tensor> out_opt, std::optional<at::Tensor> skip_mask) {
    return fn(
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
        std::move(out_opt),
        std::move(skip_mask),
        std::nullopt,
        cache_seqlens_delta_opt);
  };

  // Launch 1: decode allocates the shared output (or reuses the caller-provided
  // out_ buffer) and skips prefill batches.
  auto out = launch(decode::mha_fwd, std::move(out_), is_prefill)[0];
  // Launch 2: prefill writes into the same output and skips decode batches.
  launch(prefill::mha_fwd, out, is_prefill.logical_not());

  // softmax_lse / accum tensors are not stitched here; return empty
  // placeholders to keep the Python ABI stable.
  auto empty_f = at::empty({0}, q.options().dtype(at::kFloat));
  return {out, empty_f, empty_f, empty_f};
}

}  // namespace chunkprefill

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mha_fwd(
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
    std::optional<at::Tensor>& out_) {
  TORCH_CHECK(q.dim() == 3, "query must be in ragged format (total_q, h, d)");
  // k and v may be 3D (total_k, h_k, d) for non-paged or 4D (num_pages, page_size, h_k, d)
  // for paged KV cache; sub-functions validate their own shapes.
  if (out_.has_value()) {
    const at::Tensor& out_val = out_.value();
    TORCH_CHECK(out_val.scalar_type() == q.scalar_type(), "out dtype must match q dtype");
    TORCH_CHECK(
        out_val.dim() == 3 && out_val.size(0) == q.size(0) && out_val.size(1) == q.size(1) &&
            out_val.size(2) == v.size(-1),
        "out shape must be [total_q, num_heads, head_size_v]");
    TORCH_CHECK(out_val.device() == q.device(), "out must be on the same device as q");
    TORCH_CHECK(out_val.stride(-1) == 1, "out must have a contiguous last dimension");
  }
  auto to_tuple = [](std::vector<at::Tensor> v) { return std::make_tuple(v[0], v[1], v[2], v[3]); };
  int const num_heads = q.size(-2);
  int const num_heads_k = k.size(-2);
  int64_t batch_size = cu_seqlens_q.size(0) - 1;

  // decode / prefill / chunkprefill all take the same leading argument list;
  // only the trailing parameters differ. Bind the shared arguments once here so
  // each branch reduces to a single call. ``tail`` carries the callee-specific
  // suffix: decode and prefill additionally accept a per-batch skip mask (unused
  // at this top level, so left as std::nullopt); chunkprefill has no such slot.
  auto dispatch = [&](auto&& fn, auto&&... tail) {
    return to_tuple(
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
           out_,
           std::forward<decltype(tail)>(tail)...));
  };

  if (max_seqlen_q == 1) {
    // Pure decode path
    return dispatch(decode::mha_fwd, std::nullopt, std::nullopt, std::nullopt);
  } else if (!page_table.has_value() || batch_size == 1) {
    // Pure prefill path
    // Non-paged attn: assumption of all seqlen_q > 1;
    // Paged attn: Proving "all prefill" for batch_size > 1 would require
    // is_prefill.all() — a device reduction + D2H sync that costs more than it saves.
    // But batch_size == 1 makes it provable from host scalars:
    // a single sequence with max_seqlen_q > 1 is prefill
    return dispatch(prefill::mha_fwd, std::nullopt, std::nullopt, std::nullopt);
  } else {
    // Chunk prefill path
    // Paged attn with max_seqlen_q > 1 and batch_size > 1
    return dispatch(chunkprefill::mha_fwd, std::nullopt);
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mha_fwd_appendkv(
    const at::Tensor& q,  // (total_q, h, d) — ragged 3D
    const at::Tensor& k,  // (total_k, h_k, d) if non-paged, or (num_pages, page_size, h_k, d) if paged
    const at::Tensor& v,  // (total_k, h_k, dv) if non-paged, or (num_pages, page_size, h_k, dv) if paged
    const at::Tensor& k_new,
    const at::Tensor& v_new,
    std::optional<const at::Tensor>& q_v_,  // (total_q, h, dv) — not yet supported
    const at::Tensor& cu_seqlens_q,         // b+1
    const at::Tensor& cu_seqlens_k,         // b+1
    const at::Tensor& cu_seqlens_k_new,
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
    std::optional<at::Tensor>& out_) {
  std::optional<const at::Tensor> k_new_ = k_new;
  std::optional<const at::Tensor> v_new_ = v_new;
  std::optional<const at::Tensor> cu_seqlens_k_new_ = cu_seqlens_k_new;
  TORCH_CHECK(q.dim() == 3, "query must be in ragged format (total_q, h, d)");
  if (out_.has_value()) {
    const at::Tensor& out_val = out_.value();
    TORCH_CHECK(out_val.scalar_type() == q.scalar_type(), "out dtype must match q dtype");
    TORCH_CHECK(
        out_val.dim() == 3 && out_val.size(0) == q.size(0) && out_val.size(1) == q.size(1) &&
            out_val.size(2) == v.size(-1),
        "out shape must be [total_q, num_heads, head_size_v]");
    TORCH_CHECK(out_val.device() == q.device(), "out must be on the same device as q");
    TORCH_CHECK(out_val.stride(-1) == 1, "out must have a contiguous last dimension");
  }
  auto to_tuple = [](std::vector<at::Tensor> v) { return std::make_tuple(v[0], v[1], v[2], v[3]); };
  int64_t const batch_size = cu_seqlens_q.size(0) - 1;
  bool const use_split_mixed_append =
      page_table.has_value() && batch_size > 1 && max_seqlen_q > 1 && num_kv_splits == 1 &&
      q.size(0) != batch_size * max_seqlen_q && q.size(-1) % 8 == 0 && q.size(-2) > k.size(-2) &&
      (q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16) && k.scalar_type() == q.scalar_type() &&
      v.scalar_type() == q.scalar_type() && !q_v_.has_value() && !kv_batch_idx_.has_value() &&
      !leftpad_k_.has_value() && !rotary_cos_.has_value() && !rotary_sin_.has_value() &&
      !q_descale_.has_value() && !k_descale_.has_value() && !v_descale_.has_value() &&
      !scheduler_metadata_.has_value() && k_new_.has_value() && v_new_.has_value() &&
      cu_seqlens_k_new_.has_value() && k.is_contiguous() && v.is_contiguous() &&
      k_new_->is_contiguous() && v_new_->is_contiguous() && v.size(-1) % 8 == 0 &&
      cu_seqlens_k_new_->data_ptr() == cu_seqlens_q.data_ptr();
  if (use_split_mixed_append) {
    auto const& k_new = *k_new_;
    auto const& v_new = *v_new_;
    auto const& cu_seqlens_k_new = *cu_seqlens_k_new_;
    auto const& page_table_value = *page_table;
    TORCH_CHECK(k_new.dim() == 3 && v_new.dim() == 3, "split mixed AppendKV requires packed 3D k_new/v_new");
    TORCH_CHECK(
        k_new.size(0) == v_new.size(0) && k_new.size(1) == k.size(-2) && v_new.size(1) == v.size(-2) &&
            k_new.size(2) == k.size(-1) && v_new.size(2) == v.size(-1),
        "split mixed AppendKV k_new/v_new shapes must match the KV cache");
    TORCH_CHECK(
        k_new.scalar_type() == k.scalar_type() && v_new.scalar_type() == v.scalar_type(),
        "split mixed AppendKV k_new/v_new dtypes must match the KV cache");
    TORCH_CHECK(
        k_new.stride(-1) == 1 && v_new.stride(-1) == 1,
        "split mixed AppendKV k_new/v_new must have contiguous last dimensions");
    TORCH_CHECK(
        cu_seqlens_k.scalar_type() == at::kInt && cu_seqlens_k.size(0) == batch_size,
        "split mixed AppendKV requires int32 cache lengths with one entry per batch");
    TORCH_CHECK(
        cu_seqlens_k_new.scalar_type() == at::kInt && cu_seqlens_k_new.size(0) == batch_size + 1,
        "split mixed AppendKV requires int32 cu_seqlens_k_new with batch + 1 entries");
    TORCH_CHECK(
        page_table_value.scalar_type() == at::kInt && page_table_value.dim() == 2 &&
            page_table_value.size(0) == batch_size && page_table_value.stride(1) == 1,
        "split mixed AppendKV requires a contiguous int32 page table");

    store_paged_append_kv(k_new, v_new, k, v, cu_seqlens_k_new, cu_seqlens_k, page_table_value);
    bool const use_unified_hd64 =
        q.size(-1) == 64 && window_size_left < 0 && window_size_right < 0 && !sinks_.has_value() &&
        (batch_size == 2 ||
         (max_seqlen_q <= 256 &&
          q.size(0) * 100 <= batch_size * static_cast<int64_t>(max_seqlen_q) * 13));
    if (use_unified_hd64) {
      return to_tuple(prefill::mha_fwd(
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
          out_,
          std::nullopt,
          std::nullopt,
          cu_seqlens_k_new));
    }
    return to_tuple(chunkprefill::mha_fwd(
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
        out_,
        cu_seqlens_k_new));
  }
  return to_tuple(prefill::mha_fwd_appendkv(
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
      out_,
      std::nullopt,
      k_new_,
      v_new_,
      cu_seqlens_k_new_));
}
#undef SYCL_INTEL_TARGET
