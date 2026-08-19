// Runtime-JIT dispatch layer for the FP8 MQA-logits batched GEMM.
//
// The kernel has a single tile config (Shape<_32,_128,_32>) and a pure
// raw-pointer interface, so this renders/compiles
// Fp8MqaGemmXe20LauncherInstance.cpp.in once on first use and caches the
// resolved entry. The caller (Fp8MqaLogitsXe20.cpp) keeps all torch
// marshalling/validation and tile-alignment gating.
#pragma once

#include <cstdint>
#include <string>

namespace sgl {
namespace fp8_mqa_jit {

// Launch the batched FP8 GEMM D_b(M,N) = A_b(M,K) @ B_b(N,K)^T for all batches.
// Returns true on success; on failure fills *err if non-null.
bool gemm_launch(
    void* queue,
    const void* A_fp8,
    const void* B_fp8,
    void* D_f32,
    int batch,
    int M,
    int N,
    int K,
    int64_t A_batch_stride,
    int64_t B_batch_stride,
    int64_t D_batch_stride,
    std::string* err = nullptr);

}  // namespace fp8_mqa_jit
}  // namespace sgl
