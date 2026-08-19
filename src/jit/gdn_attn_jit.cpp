#include "jit/gdn_attn_jit.h"

#include <string>

#include "jit/sycl_template_jit.h"

namespace sgl {
namespace gdn_attn_jit {

void* resolve(
    const char* template_file,
    const char* entry_macro,
    const char* entry_symbol,
    const char* name,
    std::string* err) {
  const jit::JitConfig& cfg = jit::default_config();
  if (!cfg.valid) {
    if (err) *err = "GDN attn JIT unavailable: " + cfg.error;
    return nullptr;
  }
  if (cfg.src_root.empty()) {
    if (err) *err = "GDN attn JIT: source template root not resolved";
    return nullptr;
  }

  jit::CompileSpec spec;
  spec.template_path = cfg.src_root + "/sycl/gdn_attn/" + template_file;
  spec.extra_flags = {std::string("-D") + entry_macro};
  spec.entry_symbol = entry_symbol;
  spec.name = name;

  return jit::get_or_compile(spec, cfg, err);
}

}  // namespace gdn_attn_jit
}  // namespace sgl
