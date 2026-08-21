#include "jit/mla_jit.h"

#include <mutex>
#include <unordered_map>

#include "jit/sycl_template_jit.h"

namespace sgl {
namespace mla_jit {

namespace {

using DecodeFn =
    void (*)(void*, const void*, const void*, const void*, const void*, const void*, void*, double, int64_t);

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
  spec.name = std::string("mla_decode_") + (is_fp16 ? "half" : "bf16") + "_" + std::to_string(page_size);

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
    bool is_fp16,
    int page_size,
    void* out,
    const void* q_nope,
    const void* q_pe,
    const void* kv_c_and_k_pe_cache,
    const void* seq_lens,
    const void* page_table,
    void* workspace,
    double sm_scale,
    int64_t num_kv_splits,
    std::string* err) {
  DecodeFn fn = resolve_decode(is_fp16, page_size, err);
  if (!fn) return false;
  fn(out, q_nope, q_pe, kv_c_and_k_pe_cache, seq_lens, page_table, workspace, sm_scale, num_kv_splits);
  return true;
}

// ---------------------------------------------------------------------------
// MLA prefill (bucket selected at runtime; one entry per (elem, page)).
// ---------------------------------------------------------------------------

namespace {

using PrefillFn = void (*)(
    int,
    void*,
    const void*,
    const void*,
    const void*,
    const void*,
    const void*,
    int64_t,
    const void*,
    void*,
    double,
    bool,
    int64_t);

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
  spec.name = std::string("mla_prefill_") + (is_fp16 ? "half" : "bf16") + "_" + std::to_string(page_size);

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
    bool is_fp16,
    int page_size,
    int bucket,
    void* out,
    const void* q_nope,
    const void* q_pe,
    const void* kv_c_and_k_pe_cache,
    const void* cu_seqlens_q,
    const void* seq_lens,
    int64_t max_seqlen_q,
    const void* page_table,
    void* workspace,
    double sm_scale,
    bool causal,
    int64_t num_kv_splits,
    std::string* err) {
  PrefillFn fn = resolve_prefill(is_fp16, page_size, err);
  if (!fn) return false;
  fn(bucket,
     out,
     q_nope,
     q_pe,
     kv_c_and_k_pe_cache,
     cu_seqlens_q,
     seq_lens,
     max_seqlen_q,
     page_table,
     workspace,
     sm_scale,
     causal,
     num_kv_splits);
  return true;
}

// ---------------------------------------------------------------------------
// Sparse MLA decode / prefill (2-stage). Config = (elem, d_qk, b_h, sink).
// ---------------------------------------------------------------------------

namespace {

const char* elem_tag(bool is_fp16) {
  return is_fp16 ? "half" : "bf16";
}
const char* elem_sycl_type(bool is_fp16) {
  return is_fp16 ? "sycl::half" : "sycl::ext::oneapi::bfloat16";
}

uint64_t pack_sparse_key(bool is_fp16, int d_qk, int b_h, bool sink) {
  uint64_t k = is_fp16 ? 1u : 0u;
  k = (k << 16) | (static_cast<uint64_t>(d_qk) & 0xFFFF);
  k = (k << 8) | (static_cast<uint64_t>(b_h) & 0xFF);
  k = (k << 1) | (sink ? 1u : 0u);
  return k;
}

using SparseDecodeFn = void (*)(
    void*,
    void*,
    const void*,
    const void*,
    const void*,
    const void*,
    const void*,
    const void*,
    const void*,
    const void*,
    double,
    int64_t,
    bool);
using SparsePrefillFn =
    void (*)(void*, void*, void*, const void*, const void*, const void*, const void*, const void*, double, int64_t);

std::mutex g_sparse_dec_mu;
std::unordered_map<uint64_t, SparseDecodeFn> g_sparse_dec_fns;
std::mutex g_sparse_pre_mu;
std::unordered_map<uint64_t, SparsePrefillFn> g_sparse_pre_fns;

void* resolve_sparse(
    const char* template_rel,
    const char* entry,
    bool is_fp16,
    int d_qk,
    int b_h,
    bool sink,
    const char* name_prefix,
    std::string* err) {
  const jit::JitConfig& cfg = jit::default_config();
  if (!cfg.valid) {
    if (err) *err = std::string(name_prefix) + " JIT unavailable: " + cfg.error;
    return nullptr;
  }
  if (cfg.src_root.empty()) {
    if (err) *err = std::string(name_prefix) + " JIT: source template root not resolved";
    return nullptr;
  }
  jit::CompileSpec spec;
  spec.template_path = cfg.src_root + "/sycl/" + template_rel;
  spec.subs["ELEM_TAG"] = elem_tag(is_fp16);
  spec.subs["ELEM_SYCL_TYPE"] = elem_sycl_type(is_fp16);
  spec.subs["D_QK"] = std::to_string(d_qk);
  spec.subs["B_H"] = std::to_string(b_h);
  spec.subs["HAS_ATTN_SINK"] = sink ? "1" : "0";
  spec.extra_flags = {"-DSGL_MLA_JIT_ENTRY"};
  spec.entry_symbol = entry;
  spec.name = std::string(name_prefix) + "_" + elem_tag(is_fp16) + "_" + std::to_string(d_qk) + "_" +
              std::to_string(b_h) + "_" + (sink ? "1" : "0");
  return jit::get_or_compile(spec, cfg, err);
}

}  // namespace

bool sparse_decode_launch(
    bool is_fp16,
    int d_qk,
    int b_h,
    bool has_attn_sink,
    void* out,
    void* lse_out,
    const void* q,
    const void* k_cache,
    const void* indices,
    const void* topk_length,
    const void* extra_k_cache,
    const void* extra_indices,
    const void* extra_topk_length,
    const void* attn_sink,
    double sm_scale,
    int64_t head_dim_v,
    bool is_fp8_kvcache,
    std::string* err) {
  const uint64_t key = pack_sparse_key(is_fp16, d_qk, b_h, has_attn_sink);
  SparseDecodeFn fn = nullptr;
  {
    std::lock_guard<std::mutex> lk(g_sparse_dec_mu);
    auto it = g_sparse_dec_fns.find(key);
    if (it != g_sparse_dec_fns.end()) fn = it->second;
  }
  if (!fn) {
    void* sym = resolve_sparse(
        "mla_sparse_decode_2stage_kernel.cpp.in",
        "sgl_mla_sparse_decode_entry",
        is_fp16,
        d_qk,
        b_h,
        has_attn_sink,
        "mla_sparse_decode",
        err);
    if (!sym) return false;
    fn = reinterpret_cast<SparseDecodeFn>(sym);
    std::lock_guard<std::mutex> lk(g_sparse_dec_mu);
    g_sparse_dec_fns[key] = fn;
  }
  fn(out,
     lse_out,
     q,
     k_cache,
     indices,
     topk_length,
     extra_k_cache,
     extra_indices,
     extra_topk_length,
     attn_sink,
     sm_scale,
     head_dim_v,
     is_fp8_kvcache);
  return true;
}

bool sparse_prefill_launch(
    bool is_fp16,
    int d_qk,
    int b_h,
    bool has_attn_sink,
    void* out,
    void* max_logits,
    void* lse,
    const void* q,
    const void* kv,
    const void* indices,
    const void* attn_sink,
    const void* topk_length,
    double sm_scale,
    int64_t head_dim_v,
    std::string* err) {
  const uint64_t key = pack_sparse_key(is_fp16, d_qk, b_h, has_attn_sink);
  SparsePrefillFn fn = nullptr;
  {
    std::lock_guard<std::mutex> lk(g_sparse_pre_mu);
    auto it = g_sparse_pre_fns.find(key);
    if (it != g_sparse_pre_fns.end()) fn = it->second;
  }
  if (!fn) {
    void* sym = resolve_sparse(
        "mla_sparse_prefill_2stage_kernel.cpp.in",
        "sgl_mla_sparse_prefill_entry",
        is_fp16,
        d_qk,
        b_h,
        has_attn_sink,
        "mla_sparse_prefill",
        err);
    if (!sym) return false;
    fn = reinterpret_cast<SparsePrefillFn>(sym);
    std::lock_guard<std::mutex> lk(g_sparse_pre_mu);
    g_sparse_pre_fns[key] = fn;
  }
  fn(out, max_logits, lse, q, kv, indices, attn_sink, topk_length, sm_scale, head_dim_v);
  return true;
}

}  // namespace mla_jit
}  // namespace sgl
