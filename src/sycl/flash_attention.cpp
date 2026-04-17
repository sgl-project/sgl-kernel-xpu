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

#include "kernels/chunk_prefill/chunk_prefill_runner.hpp"
#include "kernels/flash_attention_v2/xe_fmha_fwd_decode_dispatch.hpp"
#include "kernels/flash_attention_v2/xe_fmha_fwd_prefill_dispatch.hpp"

namespace decode {

// Dispatch macros following the GroupGemmXe20.cpp pattern.
// Directly call struct operator() - no function pointers.

#define DISPATCH_DECODE_KERNEL(QG, HD, PS)         \
  do {                                             \
    if (params.use_split_kv) {                     \
      FmhaSplitDecodeRunner<QG, HD, PS>{}(params); \
    } else {                                       \
      FmhaDecodeRunner<QG, HD, PS>{}(params);      \
    }                                              \
  } while (0)

#define DISPATCH_DECODE_PAGE_SIZE(QG, HD)                                                     \
  do {                                                                                        \
    switch (params.page_size) {                                                               \
      case 64:                                                                                \
        DISPATCH_DECODE_KERNEL(QG, HD, 64);                                                   \
        break;                                                                                \
      case 128:                                                                               \
        DISPATCH_DECODE_KERNEL(QG, HD, 128);                                                  \
        break;                                                                                \
      default:                                                                                \
        TORCH_CHECK(false, "Unsupported page_size for decode attention: ", params.page_size); \
    }                                                                                         \
  } while (0)

#define DISPATCH_DECODE_HEAD_DIM(QG)                                                  \
  do {                                                                                \
    switch (params.d) {                                                               \
      case 64:                                                                        \
        DISPATCH_DECODE_PAGE_SIZE(QG, 64);                                            \
        break;                                                                        \
      case 96:                                                                        \
        DISPATCH_DECODE_PAGE_SIZE(QG, 96);                                            \
        break;                                                                        \
      case 128:                                                                       \
        DISPATCH_DECODE_PAGE_SIZE(QG, 128);                                           \
        break;                                                                        \
      case 192:                                                                       \
        DISPATCH_DECODE_PAGE_SIZE(QG, 192);                                           \
        break;                                                                        \
      case 256:                                                                       \
        DISPATCH_DECODE_PAGE_SIZE(QG, 256);                                           \
        break;                                                                        \
      case 512:                                                                       \
        DISPATCH_DECODE_PAGE_SIZE(QG, 512);                                           \
        break;                                                                        \
      default:                                                                        \
        TORCH_CHECK(false, "Unsupported head size for decode attention: ", params.d); \
    }                                                                                 \
  } while (0)

#define DISPATCH_DECODE(qg_sz)                                                                      \
  do {                                                                                              \
    switch (qg_sz) {                                                                                \
      case 1:                                                                                       \
        DISPATCH_DECODE_HEAD_DIM(1);                                                                \
        break;                                                                                      \
      case 2:                                                                                       \
        DISPATCH_DECODE_HEAD_DIM(2);                                                                \
        break;                                                                                      \
      case 4:                                                                                       \
        DISPATCH_DECODE_HEAD_DIM(4);                                                                \
        break;                                                                                      \
      case 8:                                                                                       \
        DISPATCH_DECODE_HEAD_DIM(8);                                                                \
        break;                                                                                      \
      case 16:                                                                                      \
        DISPATCH_DECODE_HEAD_DIM(16);                                                               \
        break;                                                                                      \
      default:                                                                                      \
        TORCH_CHECK(false, "Unsupported q_group_size for decode attention: ", params.q_group_size); \
    }                                                                                               \
  } while (0)

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
    int const sm_margin) {
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
  num_kv_splits = -1;
  out = torch::empty({total_q, num_heads, head_size_v}, opts);
  Arguments params;
  params.use_split_kv = true;
  if (params.use_split_kv) {
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

      int max_splits = (max_seqlen_k + block_size - 1) / block_size;
      max_splits = std::min(max_splits, parallel_);
      return std::min(num_splits, max_splits);
    };
    num_kv_splits = get_num_splits(batch_size, num_heads_k, seqlen_k, page_size);
    temp_out = num_kv_splits == 1
                   ? out
                   : torch::empty({total_q, num_kv_splits * num_heads, head_size_v}, q.options().device(q.device()));

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

  at::Tensor softmax_lse;
  softmax_lse = torch::empty({num_heads, total_q}, opts.dtype(at::kFloat));

  // align with FA3

  params.is_bf16 = q.dtype() == torch::kBFloat16;

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

  params.cu_seqlens_q = cu_seqlens_q.data_ptr<int>();
  params.cu_seqlens_k = cu_seqlens_k.data_ptr<int>();
  params.num_kv_splits = num_kv_splits;

  // Softmax sum
  params.softmax_lse_ptr = softmax_lse.data_ptr();

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
  params.use_sink = sinks_.has_value();
  params.softmax_sink_ptr = params.use_sink ? sinks_.value().data_ptr() : nullptr;

  params.softcap = softcap;

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

  at::Tensor out_accum, softmax_lse_accum;

  int qg_sz = nextPowerOf2(params.q_group_size);
  TORCH_CHECK(qg_sz >= 1 && qg_sz <= 16, "Unsupported q_group_size for decode attention: ", params.q_group_size);
  TORCH_CHECK(
      params.d == 64 || params.d == 96 || params.d == 128 || params.d == 192 || params.d == 256 || params.d == 512,
      "Unsupported head size for decode attention: ",
      params.d);
  TORCH_CHECK(
      params.page_size == 64 || params.page_size == 128,
      "Unsupported page size for decode attention: ",
      params.page_size);

  DISPATCH_DECODE(qg_sz);

  return {out, softmax_lse, out_accum, softmax_lse_accum};
}

#undef DISPATCH_DECODE_KERNEL
#undef DISPATCH_DECODE_PAGE_SIZE
#undef DISPATCH_DECODE_HEAD_DIM
#undef DISPATCH_DECODE

}  // namespace decode

namespace prefill {

// Dispatch macro following the same pattern as decode.
// Directly call struct operator() - no function pointers.

#define DISPATCH_PREFILL_KERNEL(HD) FmhaPrefillRunner<HD>{}(params)

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
    int const sm_margin) {
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
  out = torch::empty({total_q, num_heads, head_size_v}, opts);

  int const head_size_rounded = round_up_headdim(head_size);
  int const head_size_v_rounded = head_size_v == head_size ? head_size_rounded : round_up_headdim(head_size_v);

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  c10::DeviceGuard device_guard(q.device());

  at::Tensor softmax_lse;
  softmax_lse = torch::empty({num_heads, total_q}, opts.dtype(at::kFloat));

  // align with FA3
  Arguments params;
  params.is_bf16 = q.dtype() == torch::kBFloat16;

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

  params.cu_seqlens_q = cu_seqlens_q.data_ptr<int>();
  params.cu_seqlens_k = cu_seqlens_k.data_ptr<int>();

  // Softmax sum
  params.softmax_lse_ptr = softmax_lse.data_ptr();

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
  params.softmax_sink_ptr = sinks_.has_value() ? sinks_.value().data_ptr() : nullptr;

  params.softcap = softcap;

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

  at::Tensor out_accum, softmax_lse_accum;

  TORCH_CHECK(
      params.d == 64 || params.d == 96 || params.d == 128 || params.d == 192 || params.d == 256 || params.d == 512,
      "Unsupported head size for prefill attention: ",
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
      TORCH_CHECK(false, "Unsupported head size for prefill attention: ", params.d);
  }

  return {out, softmax_lse, out_accum, softmax_lse_accum};
}

#undef DISPATCH_PREFILL_KERNEL

}  // namespace prefill

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
    int const sm_margin) {
  TORCH_CHECK(cu_seqlens_k.data_ptr<int>() != nullptr, "cu_seqlens_k is not valid.");
  int const num_heads = q.size(-2);
  int const num_heads_k = k.size(-2);
  if (max_seqlen_q == 1 && page_table.has_value()) {
    return decode::mha_fwd(
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
        sm_margin);
  } else if (!(window_size_left >= 0 || window_size_right >= 0) && !sinks_.has_value() && page_table.has_value()) {
    // TODO: support the cases for non-kv cache, sliding window and sink.
    return prefill::mha_fwd(
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
        sm_margin);
  } else {
    return chunkprefill::mha_fwd(
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
        sm_margin);
  }
}
#undef SYCL_INTEL_TARGET
