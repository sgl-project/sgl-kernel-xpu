// Runtime-JIT resolver for the LoRA grouped-GEMM kernels (sgemm A/B, qkv B).
//
// Torch-free layer: builds a CompileSpec for one (dtype, tile) LoRA config,
// renders + compiles the matching *.cpp.in on first use via the generic JIT
// engine, and returns the resolved entry symbol. The caller (the LoRA
// dispatcher .cpp, which is torch-aware) casts the returned pointer to a
// function taking torch::Tensor args and invokes it. Keeping this layer
// torch-free lets it live in the torch-free sgl_jit static library.
#pragma once

#include <string>

namespace sgl {
namespace lora_jit {

// Render `template_file` (relative to the packaged sycl/ source root) with the
// given @KEY@ substitutions, compile it with -D<entry_macro>, and resolve
// `entry_symbol`. Returns nullptr and sets *err on failure. Thread-safe.
void* resolve(
    const char* template_file,
    const char* elem_tag,
    const char* tile_tag,
    const char* elem_torch_type,
    const char* tile_type,
    const char* entry_macro,
    const char* entry_symbol,
    const char* name,
    std::string* err);

}  // namespace lora_jit
}  // namespace sgl
