#include "jit/sycl_template_jit.h"

#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace sgl {
namespace jit {

namespace {

std::string read_file(const std::string& path, bool* ok) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    *ok = false;
    return {};
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  *ok = true;
  return ss.str();
}

// 64-bit FNV-1a over content. Sufficient for a per-user local .so cache keyed by
// the full rendered source + flags (content is regenerated, never trusted blind).
std::string content_hash(const std::string& s) {
  uint64_t h = 1469598103934665603ULL;
  for (unsigned char c : s) {
    h ^= c;
    h *= 1099511628211ULL;
  }
  char buf[17];
  std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
  return std::string(buf);
}

bool path_exists(const std::string& p) {
  struct stat st;
  return ::stat(p.c_str(), &st) == 0;
}

bool make_dirs(const std::string& dir) {
  std::string cur;
  for (size_t i = 0; i < dir.size(); ++i) {
    cur += dir[i];
    if (dir[i] == '/' || i + 1 == dir.size()) {
      if (!cur.empty() && cur != "/" && !path_exists(cur)) {
        if (::mkdir(cur.c_str(), 0755) != 0 && !path_exists(cur)) return false;
      }
    }
  }
  return true;
}

std::string shell_quote(const std::string& s) {
  std::string out = "'";
  for (char c : s) {
    if (c == '\'')
      out += "'\\''";
    else
      out += c;
  }
  out += "'";
  return out;
}

std::string env_or_empty(const char* name) {
  const char* v = std::getenv(name);
  return v ? std::string(v) : std::string();
}

void split_paths(const std::string& joined, std::vector<std::string>* out) {
  std::string cur;
  for (char c : joined) {
    if (c == ':') {
      if (!cur.empty()) out->push_back(cur);
      cur.clear();
    } else {
      cur += c;
    }
  }
  if (!cur.empty()) out->push_back(cur);
}

// Run a command (argv already quoted into one string), capture combined output.
int run_command(const std::string& cmd, std::string* output) {
  std::string full = cmd + " 2>&1";
  FILE* pipe = ::popen(full.c_str(), "r");
  if (!pipe) return -1;
  char buf[4096];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof(buf), pipe)) > 0) {
    output->append(buf, n);
  }
  int status = ::pclose(pipe);
  if (status == -1) return -1;
  if (WIFEXITED(status)) return WEXITSTATUS(status);
  return -2;
}

std::once_flag g_icpx_ver_once;
std::string g_icpx_ver;

std::string icpx_version(const std::string& compiler) {
  std::call_once(g_icpx_ver_once, [&] {
    std::string out;
    run_command(shell_quote(compiler) + " --version", &out);
    g_icpx_ver = out;
  });
  return g_icpx_ver;
}

struct Caches {
  std::mutex mu;
  std::unordered_map<std::string, void*> handles;  // so_path -> dlopen handle
  std::unordered_map<std::string, void*> fns;      // cache_key -> resolved symbol
};

Caches& caches() {
  static Caches c;
  return c;
}

}  // namespace

const std::vector<std::string>& default_sycl_flags() {
  static const std::vector<std::string> flags = {
      "-fsycl",
      "-sycl-std=2020",
      "-std=c++20",
      "-O2",
      "-fPIC",
      "-shared",
      "-ftemplate-backtrace-limit=0",
      "-fno-sycl-unnamed-lambda",
      "-fhonor-nans",
      "-fhonor-infinities",
      "-fno-associative-math",
      "-fno-approx-func",
      "-no-ftz",
      "-fno-sycl-instrument-device-code",
      "-D_GLIBCXX_USE_CXX11_ABI=1",
      "-DCUTLASS_ENABLE_SYCL",
      "-fsycl-targets=intel_gpu_bmg_g21",
      "-Xspirv-translator",
      "-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,"
      "+SPV_INTEL_subgroup_matrix_multiply_accumulate",
  };
  return flags;
}

std::string render_template(const std::string& tmpl,
                            const std::map<std::string, std::string>& subs) {
  std::string out;
  out.reserve(tmpl.size());
  for (size_t i = 0; i < tmpl.size();) {
    if (tmpl[i] == '@') {
      size_t end = tmpl.find('@', i + 1);
      if (end != std::string::npos) {
        std::string key = tmpl.substr(i + 1, end - i - 1);
        auto it = subs.find(key);
        if (it != subs.end()) {
          out += it->second;
          i = end + 1;
          continue;
        }
      }
    }
    out += tmpl[i];
    ++i;
  }
  return out;
}

void* get_or_compile(const CompileSpec& spec, const JitConfig& config, std::string* err) {
  auto set_err = [&](const std::string& m) {
    if (err) *err = m;
    return static_cast<void*>(nullptr);
  };

  if (!config.valid) {
    return set_err("JIT environment not available: " + config.error);
  }

  // Render the template.
  bool ok = false;
  std::string tmpl = read_file(spec.template_path, &ok);
  if (!ok) return set_err("cannot read JIT template: " + spec.template_path);
  std::string source = render_template(tmpl, spec.subs);

  // Assemble the compile flags (base SYCL + config + per-kernel).
  std::vector<std::string> flags;
  for (const auto& f : default_sycl_flags()) flags.push_back(f);
  for (const auto& f : config.extra_flags) flags.push_back(f);
  for (const auto& f : spec.extra_flags) flags.push_back(f);

  // Cache key: rendered source + flags + includes + entry + compiler version.
  std::string key_material = source;
  for (const auto& f : flags) key_material += "\n" + f;
  for (const auto& d : config.include_dirs) key_material += "\nI:" + d;
  key_material += "\nENTRY:" + spec.entry_symbol;
  key_material += "\nVER:" + icpx_version(config.compiler);
  std::string hash = content_hash(key_material);
  std::string cache_key = spec.name + "_" + hash + ":" + spec.entry_symbol;

  auto& c = caches();
  {
    std::lock_guard<std::mutex> lk(c.mu);
    auto it = c.fns.find(cache_key);
    if (it != c.fns.end()) return it->second;
  }

  if (!make_dirs(config.cache_dir)) {
    return set_err("cannot create JIT cache dir: " + config.cache_dir);
  }
  std::string so_path = config.cache_dir + "/" + spec.name + "_" + hash + ".so";

  // Compile if the .so is not already on disk (serialize compiles process-wide).
  std::lock_guard<std::mutex> lk(c.mu);
  // Re-check the fn cache under the lock (another thread may have resolved it).
  {
    auto it = c.fns.find(cache_key);
    if (it != c.fns.end()) return it->second;
  }

  if (!path_exists(so_path)) {
    std::string tmp_cpp = so_path + ".render.cpp";
    {
      std::ofstream out(tmp_cpp, std::ios::binary);
      if (!out) return set_err("cannot write rendered source: " + tmp_cpp);
      out << source;
    }
    std::string tmp_so = so_path + ".tmp";

    std::string cmd = shell_quote(config.compiler);
    for (const auto& f : flags) cmd += " " + shell_quote(f);
    for (const auto& d : config.include_dirs) cmd += " -I " + shell_quote(d);
    cmd += " " + shell_quote(tmp_cpp);
    cmd += " -o " + shell_quote(tmp_so);
    for (const auto& l : config.link_flags) cmd += " " + shell_quote(l);

    std::string out;
    int rc = run_command(cmd, &out);
    ::unlink(tmp_cpp.c_str());
    if (rc != 0) {
      ::unlink(tmp_so.c_str());
      return set_err("icpx JIT compile failed (rc=" + std::to_string(rc) + ") for " +
                     spec.name + "\nCMD: " + cmd + "\nOUTPUT:\n" + out);
    }
    if (::rename(tmp_so.c_str(), so_path.c_str()) != 0) {
      ::unlink(tmp_so.c_str());
      if (!path_exists(so_path)) {
        return set_err("failed to publish JIT .so: " + so_path);
      }
    }
  }

  // dlopen (cache handle by path) + dlsym.
  void* handle = nullptr;
  auto hit = c.handles.find(so_path);
  if (hit != c.handles.end()) {
    handle = hit->second;
  } else {
    handle = ::dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) return set_err(std::string("dlopen failed: ") + dlerror());
    c.handles[so_path] = handle;
  }

  ::dlerror();
  void* sym = ::dlsym(handle, spec.entry_symbol.c_str());
  const char* de = ::dlerror();
  if (de) return set_err("dlsym('" + spec.entry_symbol + "') failed: " + de);

  c.fns[cache_key] = sym;
  return sym;
}

const JitConfig& default_config() {
  static JitConfig cfg = [] {
    JitConfig c;

    // Locate the package include root: env override, else derive from this
    // shared object's path via dladdr (works when linked into common_ops.so
    // under <pkg>/, headers shipped under <pkg>/include).
    std::string include_root = env_or_empty("SGL_JIT_INCLUDE_ROOT");
    if (include_root.empty()) {
      Dl_info info;
      if (dladdr(reinterpret_cast<void*>(&default_config), &info) && info.dli_fname) {
        std::string self(info.dli_fname);
        size_t slash = self.find_last_of('/');
        std::string dir = (slash == std::string::npos) ? "." : self.substr(0, slash);
        // Prefer <dir>/include (installed) then <dir>/../include, else <dir>.
        if (path_exists(dir + "/include")) {
          include_root = dir + "/include";
        } else if (path_exists(dir + "/../include")) {
          include_root = dir + "/../include";
        } else {
          include_root = dir;
        }
      }
    }
    if (include_root.empty()) {
      c.error = "cannot resolve sgl_kernel include root (set SGL_JIT_INCLUDE_ROOT)";
      return c;
    }

    // The shipped sgl source-header tree (templates + kernel headers) lives
    // under <include_root>/sgl_src so quote-includes like "sycl/kernels/..."
    // resolve. cutlass headers under <include_root>/cutlass_sycl.
    c.include_dirs.push_back(include_root);
    if (path_exists(include_root + "/sgl_src")) {
      c.include_dirs.push_back(include_root + "/sgl_src");
      c.src_root = include_root + "/sgl_src";
    } else if (path_exists(include_root + "/sycl")) {
      // dev tree: include_root already contains sycl/ (e.g. repo/src).
      c.src_root = include_root;
    }

    std::vector<std::string> cutlass;
    split_paths(env_or_empty("SGL_JIT_CUTLASS_INCLUDE"), &cutlass);
    if (cutlass.empty()) {
      std::string base = include_root + "/cutlass_sycl";
      if (path_exists(base + "/include")) cutlass.push_back(base + "/include");
      if (path_exists(base + "/tools/util/include"))
        cutlass.push_back(base + "/tools/util/include");
    }
    for (auto& d : cutlass) c.include_dirs.push_back(d);

    // torch include + lib are supplied by Python at import (only Python knows
    // torch's install location).
    std::vector<std::string> torch_inc;
    split_paths(env_or_empty("SGL_JIT_TORCH_INCLUDE"), &torch_inc);
    if (torch_inc.empty()) {
      c.error = "SGL_JIT_TORCH_INCLUDE not set (Python must export torch include paths)";
      return c;
    }
    for (auto& d : torch_inc) c.include_dirs.push_back(d);

    std::string torch_lib = env_or_empty("SGL_JIT_TORCH_LIB");
    if (torch_lib.empty()) {
      c.error = "SGL_JIT_TORCH_LIB not set (Python must export torch lib dir)";
      return c;
    }
    c.link_flags.push_back("-L" + torch_lib);
    c.link_flags.push_back("-Wl,-rpath," + torch_lib);
    c.link_flags.push_back("-lc10");
    c.link_flags.push_back("-ltorch");
    c.link_flags.push_back("-ltorch_cpu");

    std::string cache_dir = env_or_empty("SGL_JIT_CACHE_DIR");
    if (cache_dir.empty()) {
      std::string home = env_or_empty("HOME");
      cache_dir = (home.empty() ? std::string("/tmp") : home) + "/.cache/sgl_kernel/cxx_jit";
    }
    c.cache_dir = cache_dir;

    c.valid = true;
    return c;
  }();
  return cfg;
}

}  // namespace jit
}  // namespace sgl
