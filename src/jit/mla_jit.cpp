#include "jit/mla_jit.h"

#include "jit/jit_arch.h"
#include "jit/sycl_template_jit.h"

namespace sgl {
namespace mla_jit {

namespace {

const char* elem_tag(bool is_fp16) {
  return is_fp16 ? "half" : "bf16";
}
const char* elem_sycl_type(bool is_fp16) {
  return is_fp16 ? "sycl::half" : "sycl::ext::oneapi::bfloat16";
}

// Shared config validation for all MLA JIT resolves.
bool check_config(const char* op_label, std::string* err) {
  const jit::JitConfig& cfg = jit::default_config();
  if (!cfg.valid) {
    if (err) *err = std::string(op_label) + " JIT unavailable: " + cfg.error;
    return false;
  }
  if (cfg.src_root.empty()) {
    if (err) *err = std::string(op_label) + " JIT: source template root not resolved";
    return false;
  }
  return true;
}

using DecodeFn =
    void (*)(void*, const void*, const void*, const void*, const void*, const void*, void*, double, int64_t);

uint64_t pack_decode_key(int arch, bool is_fp16, int page_size) {
  uint64_t k = static_cast<uint64_t>(arch) & 0xFF;
  k = (k << 16) | (static_cast<uint64_t>(page_size) & 0xFFFF);
  k = (k << 1) | (is_fp16 ? 1u : 0u);
  return k;
}

jit::JitFnCache<DecodeFn> g_decode_fns("MLA decode");

DecodeFn resolve_decode(bool is_fp16, int page_size, int arch, std::string* err) {
  const uint64_t key = pack_decode_key(arch, is_fp16, page_size);
  auto build = [&](std::string* berr) -> void* {
    if (!check_config("MLA decode", berr)) return nullptr;

    jit::CompileSpec spec;
    spec.template_path = jit::default_config().src_root + "/sycl/mla_decode_kernel.cpp.in";
    spec.subs["ELEM_TAG"] = elem_tag(is_fp16);
    spec.subs["ELEM_SYCL_TYPE"] = elem_sycl_type(is_fp16);
    spec.subs["PAGE_SIZE"] = std::to_string(page_size);
    const jit::ArchSpec as = jit::arch_spec(static_cast<jit::Arch>(arch), "-DSGL_MLA_JIT_ENTRY");
    spec.extra_flags = as.extra_flags;
    spec.target = as.target;
    spec.entry_symbol = "sgl_mla_decode_entry";
    spec.name = std::string("mla_decode_") + elem_tag(is_fp16) + "_" + std::to_string(page_size) + "_" + as.suffix;

    return jit::get_or_compile(spec, jit::default_config(), berr);
  };
  return g_decode_fns.get(key, build, err);
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
    int arch,
    std::string* err) {
  DecodeFn fn = resolve_decode(is_fp16, page_size, arch, err);
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

jit::JitFnCache<PrefillFn> g_prefill_fns("MLA prefill");

PrefillFn resolve_prefill(bool is_fp16, int page_size, int arch, std::string* err) {
  const uint64_t key = pack_decode_key(arch, is_fp16, page_size);
  auto build = [&](std::string* berr) -> void* {
    if (!check_config("MLA prefill", berr)) return nullptr;

    jit::CompileSpec spec;
    spec.template_path = jit::default_config().src_root + "/sycl/mla_prefill_kernel.cpp.in";
    spec.subs["ELEM_TAG"] = elem_tag(is_fp16);
    spec.subs["ELEM_SYCL_TYPE"] = elem_sycl_type(is_fp16);
    spec.subs["PAGE_SIZE"] = std::to_string(page_size);
    const jit::ArchSpec as = jit::arch_spec(static_cast<jit::Arch>(arch), "-DSGL_MLA_JIT_ENTRY");
    spec.extra_flags = as.extra_flags;
    spec.target = as.target;
    spec.entry_symbol = "sgl_mla_prefill_entry";
    spec.name = std::string("mla_prefill_") + elem_tag(is_fp16) + "_" + std::to_string(page_size) + "_" + as.suffix;

    return jit::get_or_compile(spec, jit::default_config(), berr);
  };
  return g_prefill_fns.get(key, build, err);
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
    int arch,
    std::string* err) {
  PrefillFn fn = resolve_prefill(is_fp16, page_size, arch, err);
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

uint64_t pack_sparse_key(int arch, bool is_fp16, int d_qk, int b_h, bool sink) {
  uint64_t k = static_cast<uint64_t>(arch) & 0xFF;
  k = (k << 1) | (is_fp16 ? 1u : 0u);
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

jit::JitFnCache<SparseDecodeFn> g_sparse_dec_fns("MLA sparse decode");
jit::JitFnCache<SparsePrefillFn> g_sparse_pre_fns("MLA sparse prefill");

// Build the CompileSpec for a sparse (2-stage) MLA kernel and resolve it.
void* resolve_sparse(
    const char* template_rel,
    const char* entry,
    bool is_fp16,
    int d_qk,
    int b_h,
    bool sink,
    int arch,
    const char* name_prefix,
    std::string* err) {
  if (!check_config(name_prefix, err)) return nullptr;
  jit::CompileSpec spec;
  spec.template_path = jit::default_config().src_root + "/sycl/" + template_rel;
  spec.subs["ELEM_TAG"] = elem_tag(is_fp16);
  spec.subs["ELEM_SYCL_TYPE"] = elem_sycl_type(is_fp16);
  spec.subs["D_QK"] = std::to_string(d_qk);
  spec.subs["B_H"] = std::to_string(b_h);
  spec.subs["HAS_ATTN_SINK"] = sink ? "1" : "0";
  const jit::ArchSpec as = jit::arch_spec(static_cast<jit::Arch>(arch), "-DSGL_MLA_JIT_ENTRY");
  spec.extra_flags = as.extra_flags;
  spec.target = as.target;
  spec.entry_symbol = entry;
  spec.name = std::string(name_prefix) + "_" + elem_tag(is_fp16) + "_" + std::to_string(d_qk) + "_" +
              std::to_string(b_h) + "_" + (sink ? "1" : "0") + "_" + as.suffix;
  return jit::get_or_compile(spec, jit::default_config(), err);
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
    int arch,
    std::string* err) {
  const uint64_t key = pack_sparse_key(arch, is_fp16, d_qk, b_h, has_attn_sink);
  auto build = [&](std::string* berr) -> void* {
    return resolve_sparse(
        "mla_sparse_decode_2stage_kernel.cpp.in",
        "sgl_mla_sparse_decode_entry",
        is_fp16,
        d_qk,
        b_h,
        has_attn_sink,
        arch,
        "mla_sparse_decode",
        berr);
  };
  SparseDecodeFn fn = g_sparse_dec_fns.get(key, build, err);
  if (!fn) return false;
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
    int arch,
    std::string* err) {
  const uint64_t key = pack_sparse_key(arch, is_fp16, d_qk, b_h, has_attn_sink);
  auto build = [&](std::string* berr) -> void* {
    return resolve_sparse(
        "mla_sparse_prefill_2stage_kernel.cpp.in",
        "sgl_mla_sparse_prefill_entry",
        is_fp16,
        d_qk,
        b_h,
        has_attn_sink,
        arch,
        "mla_sparse_prefill",
        berr);
  };
  SparsePrefillFn fn = g_sparse_pre_fns.get(key, build, err);
  if (!fn) return false;
  fn(out, max_logits, lse, q, kv, indices, attn_sink, topk_length, sm_scale, head_dim_v);
  return true;
}

}  // namespace mla_jit
}  // namespace sgl
