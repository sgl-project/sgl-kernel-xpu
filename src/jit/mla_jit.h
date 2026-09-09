// Runtime-JIT dispatch layer for MLA (multi-head latent attention) kernels.
//
// The MLA *.cpp.in templates export functions taking at::Tensor&; here the
// dispatcher (mla_decode.cpp) passes tensor addresses as opaque void* and the
// JIT-compiled module casts them back to at::Tensor* (both sides share
// <torch/all.h>, so the layout is identical). Selected config is compiled on
// first use via the generic JIT engine and cached O(1).
#pragma once

#include <cstdint>
#include <string>

namespace sgl {
namespace mla_jit {

// Launch MLA decode for (query dtype, page size). Tensor args are at::Tensor*
// passed as void*. Returns true on success; fills *err on failure.
bool mla_decode_launch(
    bool is_fp16,
    int page_size,
    void* out,
    const void* q_nope,
    const void* q_pe,
    const void* kv_c_and_k_pe_cache,
    const void* seq_lens,
    const void* page_table,
    void* workspace,
    double sm_scale,
    int64_t num_kv_splits,
    int arch = 0,
    std::string* err = nullptr);

// Launch MLA prefill for (query dtype, page size). `bucket` selects the Q-tile
// variant (0=small, 1=medium, 2=large). Tensor args are at::Tensor* as void*.
bool mla_prefill_launch(
    bool is_fp16,
    int page_size,
    int bucket,
    void* out,
    const void* q_nope,
    const void* q_pe,
    const void* kv_c_and_k_pe_cache,
    const void* cu_seqlens_q,
    const void* seq_lens,
    int64_t max_seqlen_q,
    const void* page_table,
    void* workspace,
    double sm_scale,
    bool causal,
    int64_t num_kv_splits,
    int arch = 0,
    std::string* err = nullptr);

// Launch sparse MLA decode (2-stage) for (dtype, d_qk, b_h, has_attn_sink).
// Tensor / std::optional<at::Tensor> args are passed as pointers via void*.
bool sparse_decode_launch(
    bool is_fp16,
    int d_qk,
    int b_h,
    bool has_attn_sink,
    void* out,
    void* lse_out,
    const void* q,
    const void* k_cache,
    const void* indices,
    const void* topk_length,
    const void* extra_k_cache,
    const void* extra_indices,
    const void* extra_topk_length,
    const void* attn_sink,
    double sm_scale,
    int64_t head_dim_v,
    bool is_fp8_kvcache,
    int arch = 0,
    std::string* err = nullptr);

// Launch sparse MLA prefill (2-stage) for (dtype, d_qk, b_h, has_attn_sink).
bool sparse_prefill_launch(
    bool is_fp16,
    int d_qk,
    int b_h,
    bool has_attn_sink,
    void* out,
    void* max_logits,
    void* lse,
    const void* q,
    const void* kv,
    const void* indices,
    const void* attn_sink,
    const void* topk_length,
    double sm_scale,
    int64_t head_dim_v,
    int arch = 0,
    std::string* err = nullptr);

}  // namespace mla_jit
}  // namespace sgl
