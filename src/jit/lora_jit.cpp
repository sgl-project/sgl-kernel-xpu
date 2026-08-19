#include "jit/lora_jit.h"

#include <string>

#include "jit/sycl_template_jit.h"

namespace sgl {
namespace lora_jit {

void* resolve(
    const char* template_file,
    const char* elem_tag,
    const char* tile_tag,
    const char* elem_torch_type,
    const char* tile_type,
    const char* entry_macro,
    const char* entry_symbol,
    const char* name,
    std::string* err) {
  const jit::JitConfig& cfg = jit::default_config();
  if (!cfg.valid) {
    if (err) *err = "LoRA grouped GEMM JIT unavailable: " + cfg.error;
    return nullptr;
  }
  if (cfg.src_root.empty()) {
    if (err) *err = "LoRA grouped GEMM JIT: source template root not resolved";
    return nullptr;
  }

  jit::CompileSpec spec;
  spec.template_path = cfg.src_root + "/sycl/" + template_file;
  spec.subs["ELEM_TAG"] = elem_tag;
  spec.subs["TILE_TAG"] = tile_tag;
  spec.subs["ELEM_TORCH_TYPE"] = elem_torch_type;
  spec.subs["TILE_TYPE"] = tile_type;
  spec.extra_flags = {std::string("-D") + entry_macro};
  spec.entry_symbol = entry_symbol;
  spec.name = name;

  return jit::get_or_compile(spec, cfg, err);
}

}  // namespace lora_jit
}  // namespace sgl
