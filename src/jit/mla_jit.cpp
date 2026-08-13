#include "jit/mla_jit.h"

#include <mutex>
#include <unordered_map>

#include "jit/sycl_template_jit.h"

namespace sgl {
namespace mla_jit {

namespace {

using DecodeFn = void (*)(
    void*, const void*, const void*, const void*, const void*, const void*, void*, double,
    int64_t);

uint64_t pack_decode_key(bool is_fp16, int page_size) {
  return (static_cast<uint64_t>(page_size) << 1) | (is_fp16 ? 1u : 0u);
}

std::mutex g_mu;
std::unordered_map<uint64_t, DecodeFn> g_decode_fns;

DecodeFn resolve_decode(bool is_fp16, int page_size, std::string* err) {
  const uint64_t key = pack_decode_key(is_fp16, page_size);
  {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_decode_fns.find(key);
    if (it != g_decode_fns.end()) return it->second;
  }

  const jit::JitConfig& cfg = jit::default_config();
  if (!cfg.valid) {
    if (err) *err = "MLA decode JIT unavailable: " + cfg.error;
    return nullptr;
  }
  if (cfg.src_root.empty()) {
    if (err) *err = "MLA decode JIT: source template root not resolved";
    return nullptr;
  }

  jit::CompileSpec spec;
  spec.template_path = cfg.src_root + "/sycl/mla_decode_kernel.cpp.in";
  spec.subs["ELEM_TAG"] = is_fp16 ? "half" : "bf16";
  spec.subs["ELEM_SYCL_TYPE"] = is_fp16 ? "sycl::half" : "sycl::ext::oneapi::bfloat16";
  spec.subs["PAGE_SIZE"] = std::to_string(page_size);
  spec.extra_flags = {"-DSGL_MLA_JIT_ENTRY"};
  spec.entry_symbol = "sgl_mla_decode_entry";
  spec.name = std::string("mla_decode_") + (is_fp16 ? "half" : "bf16") + "_" +
              std::to_string(page_size);

  void* sym = jit::get_or_compile(spec, cfg, err);
  if (!sym) return nullptr;

  DecodeFn fn = reinterpret_cast<DecodeFn>(sym);
  {
    std::lock_guard<std::mutex> lk(g_mu);
    g_decode_fns[key] = fn;
  }
  return fn;
}

}  // namespace

bool mla_decode_launch(
    bool is_fp16, int page_size, void* out, const void* q_nope, const void* q_pe,
    const void* kv_c_and_k_pe_cache, const void* seq_lens, const void* page_table, void* workspace,
    double sm_scale, int64_t num_kv_splits, std::string* err) {
  DecodeFn fn = resolve_decode(is_fp16, page_size, err);
  if (!fn) return false;
  fn(out, q_nope, q_pe, kv_c_and_k_pe_cache, seq_lens, page_table, workspace, sm_scale,
     num_kv_splits);
  return true;
}

// ---------------------------------------------------------------------------
// MLA prefill (bucket selected at runtime; one entry per (elem, page)).
// ---------------------------------------------------------------------------

namespace {

using PrefillFn = void (*)(
    int, void*, const void*, const void*, const void*, const void*, const void*, int64_t,
    const void*, void*, double, bool, int64_t);

std::mutex g_prefill_mu;
std::unordered_map<uint64_t, PrefillFn> g_prefill_fns;

PrefillFn resolve_prefill(bool is_fp16, int page_size, std::string* err) {
  const uint64_t key = pack_decode_key(is_fp16, page_size);
  {
    std::lock_guard<std::mutex> lk(g_prefill_mu);
    auto it = g_prefill_fns.find(key);
    if (it != g_prefill_fns.end()) return it->second;
  }

  const jit::JitConfig& cfg = jit::default_config();
  if (!cfg.valid) {
    if (err) *err = "MLA prefill JIT unavailable: " + cfg.error;
    return nullptr;
  }
  if (cfg.src_root.empty()) {
    if (err) *err = "MLA prefill JIT: source template root not resolved";
    return nullptr;
  }

  jit::CompileSpec spec;
  spec.template_path = cfg.src_root + "/sycl/mla_prefill_kernel.cpp.in";
  spec.subs["ELEM_TAG"] = is_fp16 ? "half" : "bf16";
  spec.subs["ELEM_SYCL_TYPE"] = is_fp16 ? "sycl::half" : "sycl::ext::oneapi::bfloat16";
  spec.subs["PAGE_SIZE"] = std::to_string(page_size);
  spec.extra_flags = {"-DSGL_MLA_JIT_ENTRY"};
  spec.entry_symbol = "sgl_mla_prefill_entry";
  spec.name = std::string("mla_prefill_") + (is_fp16 ? "half" : "bf16") + "_" +
              std::to_string(page_size);

  void* sym = jit::get_or_compile(spec, cfg, err);
  if (!sym) return nullptr;

  PrefillFn fn = reinterpret_cast<PrefillFn>(sym);
  {
    std::lock_guard<std::mutex> lk(g_prefill_mu);
    g_prefill_fns[key] = fn;
  }
  return fn;
}

}  // namespace

bool mla_prefill_launch(
    bool is_fp16, int page_size, int bucket, void* out, const void* q_nope, const void* q_pe,
    const void* kv_c_and_k_pe_cache, const void* cu_seqlens_q, const void* seq_lens,
    int64_t max_seqlen_q, const void* page_table, void* workspace, double sm_scale, bool causal,
    int64_t num_kv_splits, std::string* err) {
  PrefillFn fn = resolve_prefill(is_fp16, page_size, err);
  if (!fn) return false;
  fn(bucket, out, q_nope, q_pe, kv_c_and_k_pe_cache, cu_seqlens_q, seq_lens, max_seqlen_q,
     page_table, workspace, sm_scale, causal, num_kv_splits);
  return true;
}

}  // namespace mla_jit
}  // namespace sgl
