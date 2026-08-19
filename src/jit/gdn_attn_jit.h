// Runtime-JIT resolver for the GDN (Gated DeltaNet) non-chunk leaf kernels:
// causal_conv1d, gated_delta_rule (recurrent), chunk_causal_conv1d[_tiled], and
// l2norm. Each is a self-contained leaf launcher (no external kernel symbols),
// so it compiles into its own standalone JIT module on first use.
//
// Torch-free layer (lives in the torch-free sgl_jit static library): it only
// renders+compiles the requested launcher template and returns the resolved
// entry symbol. The caller (gdn_attn_interface_impl.hpp, torch-aware) casts the
// returned pointer to the matching gdn:: function-pointer type and invokes it.
#pragma once

#include <string>

namespace sgl {
namespace gdn_attn_jit {

// Render `template_file` (relative to the packaged sycl/ source root), compile
// it with -D<entry_macro>, and resolve `entry_symbol`. `name` is the cache id
// for the compiled .so. Returns nullptr and sets *err on failure. Thread-safe.
void* resolve(
    const char* template_file,
    const char* entry_macro,
    const char* entry_symbol,
    const char* name,
    std::string* err);

}  // namespace gdn_attn_jit
}  // namespace sgl
