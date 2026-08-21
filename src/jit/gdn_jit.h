// Runtime-JIT dispatch layer for the GDN (Gated DeltaNet) chunk delta-rule
// kernel. The dispatcher (chunk_gated_delta_rule_impl_xe20) keeps all torch
// marshalling and passes raw device pointers here; this layer renders +
// icpx-compiles the selected (scalar, state) instance on first use and caches
// the resolved entry in an O(1) front cache, mirroring the moe_jit pattern.
#pragma once

#include <string>

namespace sgl {
namespace gdn_jit {

// Launch the chunk delta-rule for (scalar dtype, state dtype). scalar is
// bf16/half (is_half selects); state_code is 0=fp32, 1=bf16, 2=half. All tensor
// arguments are raw device pointers (queue is a sycl::queue*). Returns true on
// success; on failure fills *err if non-null.
bool chunk_launch(
    bool is_half,
    int state_code,
    void* queue,
    void* core_attn_out,
    const void* q,
    const void* k,
    const void* v,
    void* A,
    void* w,
    void* u,
    const void* b,
    const void* a,
    const void* A_log,
    const void* dt_bias,
    void* ssm_state,
    int ssm_state_stride_0,
    const int* query_start_loc,
    const int* cache_indices,
    const bool* has_initial_state,
    const int* token_indx,
    int batch_size,
    int total_virtual_seqlen,
    int num_k_heads,
    int head_k_dim,
    int num_v_heads,
    int head_v_dim,
    std::string* err = nullptr);

}  // namespace gdn_jit
}  // namespace sgl
