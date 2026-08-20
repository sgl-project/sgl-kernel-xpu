// Runtime C++ template JIT engine (XPU / SYCL).
//
// Renders a `configure_file`-style template (`@KEY@` placeholders — the SAME
// *.cpp.in files the AOT build uses), compiles it on demand with `icpx` into a
// cached shared object, and resolves an exported `extern "C"` entry symbol.
// Resolved function pointers are cached in-memory so the hot path is a single
// map/array lookup; the icpx compile happens only on the first use of a config
// (and is persisted to disk across processes).
//
// This engine is pure infrastructure: it depends only on libdl + the C++
// standard library (no torch, no SYCL headers), so it can be unit-tested
// standalone and linked into any op library.
#pragma once

#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace sgl {
namespace jit {

// Compilation environment: include dirs, extra flags, link flags, cache dir and
// the compiler binary. `default_config()` populates this from the installed
// package layout (via dladdr) plus environment overrides.
struct JitConfig {
  std::vector<std::string> include_dirs;
  std::vector<std::string> extra_flags;  // e.g. -DFOO, -fsycl-targets=...
  std::vector<std::string> link_flags;   // e.g. -L.../torch/lib -ltorch
  std::string cache_dir;                 // where compiled .so are cached
  std::string src_root;                  // dir under which "sycl/..." templates live
  std::string compiler = "icpx";
  bool valid = false;  // false => environment incomplete
  std::string error;   // populated when !valid
};

// A single kernel instantiation request.
struct CompileSpec {
  std::string name;                         // short id for the .so filename
  std::string template_path;                // absolute path to a *.cpp.in
  std::map<std::string, std::string> subs;  // @KEY@ -> value substitutions
  std::vector<std::string> extra_flags;     // per-kernel flags (macros, ...)
  std::string entry_symbol;                 // extern "C" symbol to resolve
};

// Base flags every SYCL JIT compile needs (SYCL std, fp behavior, target, SPIRV
// extensions for CUTLASS Xe kernels). Appended to JitConfig::extra_flags.
const std::vector<std::string>& default_sycl_flags();

// Resolve (compiling + caching on first use) the entry symbol for `spec`.
// Returns nullptr and sets `err` on failure. Thread-safe.
void* get_or_compile(const CompileSpec& spec, const JitConfig& config, std::string* err);

// Auto-detected compilation environment for the installed sgl_kernel package.
// Cached after first call. Reads env overrides:
//   SGL_JIT_TORCH_INCLUDE  (os.pathsep-joined torch include dirs)  [required]
//   SGL_JIT_TORCH_LIB      (torch lib dir)                         [required]
//   SGL_JIT_INCLUDE_ROOT   (override the package include root)
//   SGL_JIT_CACHE_DIR      (override the compiled-.so cache dir)
//   SGL_JIT_CUTLASS_INCLUDE(os.pathsep-joined cutlass include dirs)
const JitConfig& default_config();

// Render a template string, replacing every `@KEY@` with subs[KEY] (mirrors
// CMake configure_file @ONLY). Exposed for testing.
std::string render_template(const std::string& tmpl, const std::map<std::string, std::string>& subs);

}  // namespace jit
}  // namespace sgl
