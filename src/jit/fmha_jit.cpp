#include "jit/fmha_jit.h"

#include <cstdint>
#include <mutex>
#include <unordered_map>

#include "jit/sycl_template_jit.h"

namespace sgl {
namespace fmha_jit {

namespace {

using KernelFn = void (*)(const void*);

// Per-op template file name (under <src_root>/sycl/kernels/flash_attention_v2/).
const char* template_file(DecodeOp op) {
  switch (op) {
    case DecodeOp::kDecode:
      return "xe_fmha_fwd_decode_kernel.cpp.in";
    case DecodeOp::kSplitDecode:
      return "xe_fmha_fwd_split_decode_kernel.cpp.in";
    case DecodeOp::kDecodeFp8:
      return "xe_fmha_fwd_decode_fp8_kernel.cpp.in";
    case DecodeOp::kSplitDecodeFp8:
      return "xe_fmha_fwd_split_decode_fp8_kernel.cpp.in";
    case DecodeOp::kDecodeNoPage:
      return "xe_fmha_fwd_decode_nopage_kernel.cpp.in";
  }
  return "";
}

const char* op_tag(DecodeOp op) {
  switch (op) {
    case DecodeOp::kDecode:
      return "decode";
    case DecodeOp::kSplitDecode:
      return "split_decode";
    case DecodeOp::kDecodeFp8:
      return "decode_fp8";
    case DecodeOp::kSplitDecodeFp8:
      return "split_decode_fp8";
    case DecodeOp::kDecodeNoPage:
      return "decode_nopage";
  }
  return "";
}

// Pack the config into a fast integer key for the front cache.
uint64_t pack_key(DecodeOp op, int qg, int hd, int ps, bool is_fp16) {
  uint64_t k = static_cast<uint64_t>(op) & 0xFF;
  k = (k << 12) | (static_cast<uint64_t>(qg) & 0xFFF);
  k = (k << 16) | (static_cast<uint64_t>(hd) & 0xFFFF);
  k = (k << 16) | (static_cast<uint64_t>(ps) & 0xFFFF);
  k = (k << 1) | (is_fp16 ? 1u : 0u);
  return k;
}

std::mutex g_mu;
std::unordered_map<uint64_t, KernelFn> g_fns;  // O(1) hot-path cache

// Resolve (compiling on first use) the kernel entry for a config.
KernelFn resolve(DecodeOp op, int qg, int hd, int ps, bool is_fp16, std::string* err) {
  const uint64_t key = pack_key(op, qg, hd, ps, is_fp16);
  {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_fns.find(key);
    if (it != g_fns.end()) return it->second;
  }

  const jit::JitConfig& cfg = jit::default_config();
  if (!cfg.valid) {
    if (err) *err = "FMHA JIT unavailable: " + cfg.error;
    return nullptr;
  }
  if (cfg.src_root.empty()) {
    if (err) *err = "FMHA JIT: source template root not resolved";
    return nullptr;
  }

  const bool no_page = (op == DecodeOp::kDecodeNoPage);

  jit::CompileSpec spec;
  spec.template_path =
      cfg.src_root + "/sycl/kernels/flash_attention_v2/" + template_file(op);
  spec.subs["QG_SZ"] = std::to_string(qg);
  spec.subs["HEAD_DIM"] = std::to_string(hd);
  if (!no_page) spec.subs["PAGE_SIZE"] = std::to_string(ps);
  spec.subs["ELEM_TYPE"] = is_fp16 ? "cutlass::half_t" : "cutlass::bfloat16_t";
  spec.subs["ELEM_TAG"] = is_fp16 ? "fp16" : "bf16";
  spec.extra_flags = {"-DSGL_FMHA_JIT_ENTRY"};
  spec.entry_symbol = "sgl_fmha_entry";

  spec.name = std::string("xe_fmha_fwd_") + op_tag(op) + "_" + std::to_string(qg) + "_" +
              std::to_string(hd) + (no_page ? "" : "_" + std::to_string(ps)) + "_" +
              (is_fp16 ? "fp16" : "bf16");

  void* sym = jit::get_or_compile(spec, cfg, err);
  if (!sym) return nullptr;

  KernelFn fn = reinterpret_cast<KernelFn>(sym);
  {
    std::lock_guard<std::mutex> lk(g_mu);
    g_fns[key] = fn;
  }
  return fn;
}

// ---------------------------------------------------------------------------
// Prefill: dispatch on head dim only; per-head-dim tile params mirror
// FMHAPrefillXe20.cmake (kept in sync by hand).
// ---------------------------------------------------------------------------

struct PrefillTile {
  int tiled_q;
  int tiled_kv;
  int num_sg;
};

int round32(int hd) { return ((hd + 31) / 32) * 32; }

// Paged / fp8 prefill tile params. Returns false for an unsupported head dim.
bool paged_prefill_tile(int hd, PrefillTile* t) {
  switch (hd) {
    case 64: *t = {128, 64, 8}; return true;
    case 96: *t = {128, 64, 8}; return true;
    case 128: *t = {256, 32, 16}; return true;
    case 192: *t = {256, 64, 32}; return true;
    case 256: *t = {256, 64, 32}; return true;
    case 512: *t = {256, 64, 32}; return true;
    default: return false;
  }
}

// Non-paged prefill tile params.
bool nopage_prefill_tile(int hd, PrefillTile* t) {
  switch (hd) {
    case 64: *t = {128, 64, 8}; return true;
    case 72: *t = {256, 64, 16}; return true;
    case 80: *t = {256, 64, 16}; return true;
    case 96: *t = {256, 64, 16}; return true;
    case 128: *t = {256, 32, 16}; return true;
    case 192: *t = {256, 32, 16}; return true;
    case 256: *t = {128, 64, 16}; return true;
    case 512: *t = {128, 128, 16}; return true;
    default: return false;
  }
}

const char* prefill_template_file(PrefillOp op) {
  switch (op) {
    case PrefillOp::kPrefill:
      return "xe_fmha_fwd_prefill_kernel.cpp.in";
    case PrefillOp::kPrefillFp8:
      return "xe_fmha_fwd_prefill_fp8_kernel.cpp.in";
    case PrefillOp::kPrefillNoPage:
      return "xe_fmha_fwd_prefill_nopage_kernel.cpp.in";
  }
  return "";
}

const char* prefill_op_tag(PrefillOp op) {
  switch (op) {
    case PrefillOp::kPrefill:
      return "prefill";
    case PrefillOp::kPrefillFp8:
      return "prefill_fp8";
    case PrefillOp::kPrefillNoPage:
      return "prefill_nopage";
  }
  return "";
}

uint64_t pack_prefill_key(PrefillOp op, int hd, bool is_fp16) {
  uint64_t k = static_cast<uint64_t>(op) & 0xFF;
  k = (k << 16) | (static_cast<uint64_t>(hd) & 0xFFFF);
  k = (k << 1) | (is_fp16 ? 1u : 0u);
  return k;
}

std::mutex g_prefill_mu;
std::unordered_map<uint64_t, KernelFn> g_prefill_fns;

KernelFn resolve_prefill(PrefillOp op, int hd, bool is_fp16, std::string* err) {
  const uint64_t key = pack_prefill_key(op, hd, is_fp16);
  {
    std::lock_guard<std::mutex> lk(g_prefill_mu);
    auto it = g_prefill_fns.find(key);
    if (it != g_prefill_fns.end()) return it->second;
  }

  const jit::JitConfig& cfg = jit::default_config();
  if (!cfg.valid) {
    if (err) *err = "FMHA prefill JIT unavailable: " + cfg.error;
    return nullptr;
  }
  if (cfg.src_root.empty()) {
    if (err) *err = "FMHA prefill JIT: source template root not resolved";
    return nullptr;
  }

  const bool nopage = (op == PrefillOp::kPrefillNoPage);
  // fp8 prefill is bf16-query only.
  const bool fp16 = (op == PrefillOp::kPrefillFp8) ? false : is_fp16;

  PrefillTile tile;
  const bool ok = nopage ? nopage_prefill_tile(hd, &tile) : paged_prefill_tile(hd, &tile);
  if (!ok) {
    if (err) *err = "FMHA prefill JIT: unsupported head dim " + std::to_string(hd);
    return nullptr;
  }
  const int tiled_out = (hd == 512) ? 256 : round32(hd);
  const int enable_score_block2d = (hd == 512) ? 1 : 0;

  jit::CompileSpec spec;
  spec.template_path =
      cfg.src_root + "/sycl/kernels/flash_attention_v2/" + prefill_template_file(op);
  spec.subs["HEAD_DIM"] = std::to_string(hd);
  spec.subs["ELEM_TYPE"] = fp16 ? "cutlass::half_t" : "cutlass::bfloat16_t";
  spec.subs["ELEM_TAG"] = fp16 ? "fp16" : "bf16";
  spec.subs["ENABLE_SCORE_BLOCK2D"] = std::to_string(enable_score_block2d);
  if (nopage) {
    spec.subs["TILED_Q_NP"] = std::to_string(tile.tiled_q);
    spec.subs["TILED_KV_NP"] = std::to_string(tile.tiled_kv);
    spec.subs["NUM_SG_NP"] = std::to_string(tile.num_sg);
    spec.subs["TILED_OUT_NP"] = std::to_string(tiled_out);
  } else {
    spec.subs["TILED_Q"] = std::to_string(tile.tiled_q);
    spec.subs["TILED_KV"] = std::to_string(tile.tiled_kv);
    spec.subs["NUM_SG"] = std::to_string(tile.num_sg);
    spec.subs["TILED_OUT"] = std::to_string(tiled_out);
  }
  spec.extra_flags = {"-DSGL_FMHA_JIT_ENTRY"};
  spec.entry_symbol = "sgl_fmha_entry";
  spec.name = std::string("xe_fmha_fwd_") + prefill_op_tag(op) + "_" + std::to_string(hd) + "_" +
              (fp16 ? "fp16" : "bf16");

  void* sym = jit::get_or_compile(spec, cfg, err);
  if (!sym) return nullptr;

  KernelFn fn = reinterpret_cast<KernelFn>(sym);
  {
    std::lock_guard<std::mutex> lk(g_prefill_mu);
    g_prefill_fns[key] = fn;
  }
  return fn;
}

}  // namespace

bool decode_launch(
    DecodeOp op, int qg, int hd, int ps, bool is_fp16, const void* params, std::string* err) {
  KernelFn fn = resolve(op, qg, hd, ps, is_fp16, err);
  if (!fn) return false;
  fn(params);
  return true;
}

bool decode_prewarm(DecodeOp op, int qg, int hd, int ps, bool is_fp16, std::string* err) {
  return resolve(op, qg, hd, ps, is_fp16, err) != nullptr;
}

bool prefill_launch(PrefillOp op, int hd, bool is_fp16, const void* params, std::string* err) {
  KernelFn fn = resolve_prefill(op, hd, is_fp16, err);
  if (!fn) return false;
  fn(params);
  return true;
}

bool prefill_prewarm(PrefillOp op, int hd, bool is_fp16, std::string* err) {
  return resolve_prefill(op, hd, is_fp16, err) != nullptr;
}

}  // namespace fmha_jit
}  // namespace sgl
