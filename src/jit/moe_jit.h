// Runtime-JIT dispatch layer for the bf16 grouped GEMM (MoE) kernel.
//
// Selects a (tile, subgroup-layout, activation, fuse, bias) instantiation of
// GroupGemmXe20LauncherInstance.cpp.in at runtime, compiles it on first use via
// the generic JIT engine, and caches the resolved entry in an O(1) front cache.
// The caller (GroupGemmXe20.cpp) keeps all torch marshalling/validation; here we
// only render+compile+dispatch the heavy kernel, mirroring the compile-time
// dispatch tree in that file.
#pragma once

#include <string>

namespace sgl {
namespace moe_jit {

// Launch the grouped GEMM for the given runtime config. The tile/subgroup-layout
// are selected from (avg_m, gemm_k, gemm_n, fuse_act) exactly as the AOT
// dispatcher does. Returns true on success; on failure fills *err if non-null.
bool grouped_gemm_launch(
    int avg_m,
    int activation_type,  // 0=silu, 1=gelu, 2=swiglu_gpt_oss, 3=relu2
    bool fuse_act,
    bool with_bias,
    void* queue,
    const void* activations,
    const void* weights,
    const void* scales,
    const void* bias,
    void* outputs,
    int gemm_n,
    int gemm_k,
    const int* num_rows_per_expert,
    int num_experts,
    int* workspace,
    float gemm1_alpha,
    float gemm1_limit,
    int ld_b,
    int arch = 0,  // sgl::jit::Arch code (0=BMG/Xe20, 1=XE3P/Xe35)
    std::string* err = nullptr);

// Launch the W4A16 (int4 / mxfp4) grouped GEMM. The policy is selected from
// avg_m and (ElementS, ElementA) from (is_int4, is_fp16), mirroring
// GroupGemmW4A16Xe20.cpp. group_size is a runtime arg (not a template param).
bool w4a16_grouped_gemm_launch(
    int avg_m,
    bool is_int4,
    bool is_fp16,
    void* queue,
    const void* activations,
    const void* packed_weights,
    const void* scales,
    const void* zeros,
    const void* bias,
    void* outputs,
    int gemm_n,
    int gemm_k,
    const int* rows_per_expert,
    int num_experts,
    int group_size,
    int* atomic_buffer,
    int arch = 0,  // sgl::jit::Arch code (0=BMG/Xe20, 1=XE3P/Xe35)
    std::string* err = nullptr);

}  // namespace moe_jit
}  // namespace sgl
