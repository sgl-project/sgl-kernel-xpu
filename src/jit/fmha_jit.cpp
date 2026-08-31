#include "jit/fmha_jit.h"

#include <cstdint>

#include "jit/jit_arch.h"
#include "jit/sycl_template_jit.h"
#include "sycl/kernels/flash_attention_v2/fmha_tile_dispatch.h"

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
    case DecodeOp::kDecodeMxfp4:
      return "xe_fmha_fwd_decode_mxfp4_kernel.cpp.in";
    case DecodeOp::kSplitDecodeMxfp4:
      return "xe_fmha_fwd_split_decode_mxfp4_kernel.cpp.in";
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
    case DecodeOp::kDecodeMxfp4:
      return "decode_mxfp4";
    case DecodeOp::kSplitDecodeMxfp4:
      return "split_decode_mxfp4";
    case DecodeOp::kDecodeNoPage:
      return "decode_nopage";
  }
  return "";
}

// Pack the config into a fast integer key for the front cache.
uint64_t pack_key(int arch, DecodeOp op, int qg, int hd, int ps, bool is_fp16) {
  uint64_t k = static_cast<uint64_t>(arch) & 0xFF;
  k = (k << 8) | (static_cast<uint64_t>(op) & 0xFF);
  k = (k << 12) | (static_cast<uint64_t>(qg) & 0xFFF);
  k = (k << 16) | (static_cast<uint64_t>(hd) & 0xFFFF);
  k = (k << 16) | (static_cast<uint64_t>(ps) & 0xFFFF);
  k = (k << 1) | (is_fp16 ? 1u : 0u);
  return k;
}

jit::JitFnCache<KernelFn> g_fns("FMHA decode");
jit::JitFnCache<KernelFn> g_prefill_fns("FMHA prefill");

// Shared config-validation + spec-assembly for both decode and prefill.
// Error messages are bare (no op prefix); JitFnCache::get adds the prefix.
void* resolve_spec(const std::string& template_rel, jit::CompileSpec& spec, int arch, std::string* err) {
  const jit::JitConfig& cfg = jit::default_config();
  if (!cfg.valid) {
    if (err) *err = std::string("unavailable: ") + cfg.error;
    return nullptr;
  }
  if (cfg.src_root.empty()) {
    if (err) *err = "source template root not resolved";
    return nullptr;
  }
  spec.template_path = cfg.src_root + "/sycl/kernels/flash_attention_v2/" + template_rel;
  const jit::ArchSpec as = jit::arch_spec(static_cast<jit::Arch>(arch), "-DSGL_FMHA_JIT_ENTRY");
  spec.extra_flags = as.extra_flags;
  spec.target = as.target;
  spec.entry_symbol = "sgl_fmha_entry";
  return jit::get_or_compile(spec, cfg, err);
}

KernelFn resolve(DecodeOp op, int qg, int hd, int ps, bool is_fp16, int arch, std::string* err) {
  const uint64_t key = pack_key(arch, op, qg, hd, ps, is_fp16);
  auto build = [&](std::string* berr) -> void* {
    const bool no_page = (op == DecodeOp::kDecodeNoPage);

    jit::CompileSpec spec;
    spec.subs["QG_SZ"] = std::to_string(qg);
    spec.subs["HEAD_DIM"] = std::to_string(hd);
    if (!no_page) {
      spec.subs["PAGE_SIZE"] = std::to_string(ps);
    } else {
      // Non-paged decode has no page size; it uses an independent KV tile.
      spec.subs["TILED_KV_NP"] = std::to_string(sgl::fmha::decode_tiled_kv_np(hd));
    }
    spec.subs["ELEM_TYPE"] = is_fp16 ? "cutlass::half_t" : "cutlass::bfloat16_t";
    spec.subs["ELEM_TAG"] = is_fp16 ? "fp16" : "bf16";

    spec.name = std::string("xe_fmha_fwd_") + op_tag(op) + "_" + std::to_string(qg) + "_" + std::to_string(hd) +
                (no_page ? "" : "_" + std::to_string(ps)) + "_" + (is_fp16 ? "fp16" : "bf16");

    return resolve_spec(template_file(op), spec, arch, berr);
  };
  return g_fns.get(key, build, err);
}

// ---------------------------------------------------------------------------
// Prefill: dispatch on head dim only; per-head-dim tile params come from the
// shared fmha_tile_dispatch.h (single source of truth with the AOT templates).
// ---------------------------------------------------------------------------

const char* prefill_template_file(PrefillOp op) {
  switch (op) {
    case PrefillOp::kPrefill:
      return "xe_fmha_fwd_prefill_kernel.cpp.in";
    case PrefillOp::kPrefillFp8:
      return "xe_fmha_fwd_prefill_fp8_kernel.cpp.in";
    case PrefillOp::kPrefillMxfp4:
      return "xe_fmha_fwd_prefill_mxfp4_kernel.cpp.in";
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
    case PrefillOp::kPrefillMxfp4:
      return "prefill_mxfp4";
    case PrefillOp::kPrefillNoPage:
      return "prefill_nopage";
  }
  return "";
}

uint64_t pack_prefill_key(int arch, PrefillOp op, int hd, bool is_fp16) {
  uint64_t k = static_cast<uint64_t>(arch) & 0xFF;
  k = (k << 8) | (static_cast<uint64_t>(op) & 0xFF);
  k = (k << 16) | (static_cast<uint64_t>(hd) & 0xFFFF);
  k = (k << 1) | (is_fp16 ? 1u : 0u);
  return k;
}

KernelFn resolve_prefill(PrefillOp op, int hd, bool is_fp16, int arch, std::string* err) {
  const uint64_t key = pack_prefill_key(arch, op, hd, is_fp16);
  auto build = [&](std::string* berr) -> void* {
    const bool nopage = (op == PrefillOp::kPrefillNoPage);
    // fp8 / mxfp4 prefill are bf16-query only.
    const bool fp16 = (op == PrefillOp::kPrefillFp8 || op == PrefillOp::kPrefillMxfp4) ? false : is_fp16;

    const sgl::fmha::PrefillTile tile = nopage ? sgl::fmha::prefill_nopage_tile(hd) : sgl::fmha::prefill_paged_tile(hd);
    if (!tile.ok) {
      *berr = "unsupported head dim " + std::to_string(hd);
      return nullptr;
    }
    const int tiled_out = sgl::fmha::prefill_tiled_out(hd);
    const int enable_score_block2d = (hd == 512) ? 1 : 0;

    jit::CompileSpec spec;
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
    spec.name =
        std::string("xe_fmha_fwd_") + prefill_op_tag(op) + "_" + std::to_string(hd) + "_" + (fp16 ? "fp16" : "bf16");

    return resolve_spec(prefill_template_file(op), spec, arch, berr);
  };
  return g_prefill_fns.get(key, build, err);
}

}  // namespace

bool decode_launch(DecodeOp op, int qg, int hd, int ps, bool is_fp16, const void* params, int arch, std::string* err) {
  KernelFn fn = resolve(op, qg, hd, ps, is_fp16, arch, err);
  if (!fn) return false;
  fn(params);
  return true;
}

bool decode_prewarm(DecodeOp op, int qg, int hd, int ps, bool is_fp16, int arch, std::string* err) {
  return resolve(op, qg, hd, ps, is_fp16, arch, err) != nullptr;
}

bool prefill_launch(PrefillOp op, int hd, bool is_fp16, const void* params, int arch, std::string* err) {
  KernelFn fn = resolve_prefill(op, hd, is_fp16, arch, err);
  if (!fn) return false;
  fn(params);
  return true;
}

bool prefill_prewarm(PrefillOp op, int hd, bool is_fp16, int arch, std::string* err) {
  return resolve_prefill(op, hd, is_fp16, arch, err) != nullptr;
}

}  // namespace fmha_jit
}  // namespace sgl
