#pragma once

#include "namespace_config.h"

#include "cutlass/bfloat16.h"

#include <cstdint>

namespace FLASH_NAMESPACE {

#ifndef FLASH_MLA_PREFILL_B_TOPK
#define FLASH_MLA_PREFILL_B_TOPK 32
#endif

static constexpr float LOG_2_E = 1.4426950408889634f;
static constexpr float LOG_E_2 = 0.6931471805599453f;

// Below are for FlashMLA definitions

// Same as https://github.com/deepseek-ai/FlashMLA/blob/main/csrc/params.h#L145
struct SparseAttnFwdParams {
    int s_q, s_kv, h_q, h_kv, d_qk, d_v, topk;
    float sm_scale, sm_scale_div_log2;

    // Input tensors
    cutlass::bfloat16_t* __restrict__ q;    // [s_q, h_q, d_qk]
    cutlass::bfloat16_t* __restrict__ kv;   // [s_kv, h_kv, d_qk]
    int* __restrict__ indices;   // [s_q, h_kv, topk]
    float* __restrict__ attn_sink;   // [h_q], may be nullptr
    int* __restrict__ topk_length;    // [s_q], may be nullptr

    // Strides
    int stride_q_s_q; int stride_q_h_q;
    int stride_kv_s_kv; int stride_kv_h_kv;
    int stride_indices_s_q; int stride_indices_h_kv;

    // Internal gathered dense KV workspace.
    cutlass::bfloat16_t* __restrict__ gathered_k;
    int* __restrict__ gathered_valid_mask;

    // Gathered workspace strides.
    int stride_gathered_k_s_q; int stride_gathered_k_topk;
    int stride_gathered_mask_s_q;

    // Output tensors
    cutlass::bfloat16_t* __restrict__ out;   // [s_q, h_q, d_v]
    float* __restrict__ max_logits; // [s_q, h_q], may be nullptr
    float* __restrict__ lse; // [s_q, h_q], may be nullptr
    bool return_softmax_lse;

    int num_sm;

    // Chunking: gathered_k workspace is only [gathered_k_s_q_chunk, topk, d_qk].
    // The launch function internally loops over s_q in chunks of this size.
    int gathered_k_s_q_chunk;
};

// WA: avoid issue of not supported device-only copy
struct XPUSparseAttnFwdParams : public SparseAttnFwdParams {
    sycl::queue queue;
};

template<int D_QK, bool HAVE_TOPK_LENGTH>
void launch_sparse_mla_prefill_gather_kernel(const XPUSparseAttnFwdParams& params);

template<int D_QK, bool HAVE_TOPK_LENGTH, bool HAS_ATTN_SINK>
void launch_sparse_mla_prefill_fwd_kernel(const XPUSparseAttnFwdParams& params);

} // namespace FLASH_NAMESPACE
