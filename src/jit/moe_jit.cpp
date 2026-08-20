#include "jit/moe_jit.h"

#include <cstdint>
#include <mutex>
#include <unordered_map>

#include "jit/sycl_template_jit.h"
#include "jit/jit_arch.h"

namespace sgl {
namespace moe_jit {

namespace {

using KernelFn = void (*)(
    void*,
    const void*,
    const void*,
    const void*,
    const void*,
    void*,
    int,
    int,
    const int*,
    int,
    int*,
    float,
    float,
    int);

struct TileCfg {
  const char* tile;
  const char* sglayout;
};

// (tile, subgroup layout) pairs, matching src/sycl/GroupGemmXe20.cpp.
const TileCfg kTiles[7] = {
    {"Shape<_8, _64, _32>", "Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>"},
    {"Shape<_16, _64, _32>", "Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>"},
    {"Shape<_32, _64, _32>", "Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>"},
    {"Shape<_128, _64, _32>", "Layout<Shape<_4, _2, _1>, Stride<_2, _1, _0>>"},
    {"Shape<_128, _128, _32>", "Layout<Shape<_4, _2, _1>, Stride<_2, _1, _0>>"},
    {"Shape<_256, _64, _32>", "Layout<Shape<_8, _2, _1>, Stride<_2, _1, _0>>"},
    {"Shape<_256, _256, _32>", "Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>"},
};

// Mirror MOE_GROUPED_GEMM_SMALL_WEIGHT_THRESHOLD in src/sycl/Utils.h.
constexpr int64_t kSmallWeightThreshold = int64_t(4096) * 4096;

int select_tile(int avg_m, int gemm_k, int gemm_n, bool fuse_act) {
  const bool small_weight = static_cast<int64_t>(gemm_k) * gemm_n <= kSmallWeightThreshold;
  const bool narrow_k = gemm_k <= 256;
  const bool narrow_n_fused = fuse_act && (gemm_n <= 512);

  if (avg_m <= 8) return 0;
  if (avg_m <= 16 && small_weight) return 1;
  if (avg_m <= 32 && small_weight) return 2;
  if (avg_m <= 128 && small_weight) return fuse_act ? 3 : 4;
  if (narrow_k) return fuse_act ? 3 : 4;
  if (narrow_n_fused) return 3;
  return fuse_act ? 5 : 6;
}

uint64_t pack_key(int tile_id, int act, bool fuse, bool bias, int arch) {
  uint64_t k = static_cast<uint64_t>(arch) & 0xFF;
  k = (k << 8) | (static_cast<uint64_t>(tile_id) & 0xFF);
  k = (k << 8) | (static_cast<uint64_t>(act) & 0xFF);
  k = (k << 1) | (fuse ? 1u : 0u);
  k = (k << 1) | (bias ? 1u : 0u);
  return k;
}

std::mutex g_mu;
std::unordered_map<uint64_t, KernelFn> g_fns;

KernelFn resolve(int tile_id, int act, bool fuse, bool bias, int arch, std::string* err) {
  const uint64_t key = pack_key(tile_id, act, fuse, bias, arch);
  {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_fns.find(key);
    if (it != g_fns.end()) return it->second;
  }

  const jit::JitConfig& cfg = jit::default_config();
  if (!cfg.valid) {
    if (err) *err = "MoE grouped GEMM JIT unavailable: " + cfg.error;
    return nullptr;
  }
  if (cfg.src_root.empty()) {
    if (err) *err = "MoE grouped GEMM JIT: source template root not resolved";
    return nullptr;
  }

  jit::CompileSpec spec;
  spec.template_path = cfg.src_root + "/sycl/GroupGemmXe20LauncherInstance.cpp.in";
  spec.subs["TILE"] = kTiles[tile_id].tile;
  spec.subs["SGLAYOUT"] = kTiles[tile_id].sglayout;
  spec.subs["ACT_TYPE"] = std::to_string(act);
  spec.subs["FUSE_ACT"] = fuse ? "true" : "false";
  spec.subs["WITH_BIAS"] = bias ? "true" : "false";
  const jit::ArchProfile& prof = jit::arch_profile(static_cast<jit::Arch>(arch));
  spec.extra_flags = {"-DSGL_MOE_JIT_ENTRY"};
  if (!prof.macro.empty()) spec.extra_flags.push_back("-D" + prof.macro);
  spec.target = prof.target;
  spec.entry_symbol = "sgl_moe_gg_entry";
  spec.name = std::string("group_gemm_xe20_t") + std::to_string(tile_id) + "_a" + std::to_string(act) + "_f" +
              (fuse ? "1" : "0") + "_b" + (bias ? "1" : "0") + "_" + prof.suffix;

  void* sym = jit::get_or_compile(spec, cfg, err);
  if (!sym) return nullptr;

  KernelFn fn = reinterpret_cast<KernelFn>(sym);
  {
    std::lock_guard<std::mutex> lk(g_mu);
    g_fns[key] = fn;
  }
  return fn;
}

}  // namespace

bool grouped_gemm_launch(
    int avg_m,
    int activation_type,
    bool fuse_act,
    bool with_bias,
    void* queue,
    const void* activations,
    const void* weights,
    const void* scales,
    const void* bias,
    void* outputs,
    int gemm_n,
    int gemm_k,
    const int* num_rows_per_expert,
    int num_experts,
    int* workspace,
    float gemm1_alpha,
    float gemm1_limit,
    int ld_b,
    int arch,
    std::string* err) {
  const int tile_id = select_tile(avg_m, gemm_k, gemm_n, fuse_act);
  KernelFn fn = resolve(tile_id, activation_type, fuse_act, with_bias, arch, err);
  if (!fn) return false;
  fn(queue,
     activations,
     weights,
     scales,
     bias,
     outputs,
     gemm_n,
     gemm_k,
     num_rows_per_expert,
     num_experts,
     workspace,
     gemm1_alpha,
     gemm1_limit,
     ld_b);
  return true;
}

// ---------------------------------------------------------------------------
// W4A16 (int4 / mxfp4) grouped GEMM.
// ---------------------------------------------------------------------------

namespace {

using W4A16Fn = void (*)(
    void*,
    const void*,
    const void*,
    const void*,
    const void*,
    const void*,
    void*,
    int,
    int,
    const int*,
    int,
    int,
    int*);

// Policy name selected from avg_m (mirrors GroupGemmW4A16Xe20.cpp).
const char* w4a16_policy(int avg_m) {
  if (avg_m <= 4) return "w4a16_policy_m_8";
  if (avg_m <= 8) return "w4a16_policy_m_16";
  if (avg_m <= 128) return "w4a16_policy_m_32";
  return "w4a16_policy";
}

int w4a16_policy_id(int avg_m) {
  if (avg_m <= 4) return 0;
  if (avg_m <= 8) return 1;
  if (avg_m <= 128) return 2;
  return 3;
}

uint64_t pack_w4a16_key(int policy_id, bool is_int4, bool is_fp16, int arch) {
  uint64_t k = static_cast<uint64_t>(arch) & 0xFF;
  k = (k << 8) | (static_cast<uint64_t>(policy_id) & 0xFF);
  k = (k << 1) | (is_int4 ? 1u : 0u);
  k = (k << 1) | (is_fp16 ? 1u : 0u);
  return k;
}

std::mutex g_w4a16_mu;
std::unordered_map<uint64_t, W4A16Fn> g_w4a16_fns;

W4A16Fn resolve_w4a16(int avg_m, bool is_int4, bool is_fp16, int arch, std::string* err) {
  const int policy_id = w4a16_policy_id(avg_m);
  const uint64_t key = pack_w4a16_key(policy_id, is_int4, is_fp16, arch);
  {
    std::lock_guard<std::mutex> lk(g_w4a16_mu);
    auto it = g_w4a16_fns.find(key);
    if (it != g_w4a16_fns.end()) return it->second;
  }

  const jit::JitConfig& cfg = jit::default_config();
  if (!cfg.valid) {
    if (err) *err = "W4A16 grouped GEMM JIT unavailable: " + cfg.error;
    return nullptr;
  }
  if (cfg.src_root.empty()) {
    if (err) *err = "W4A16 grouped GEMM JIT: source template root not resolved";
    return nullptr;
  }

  const char* elem_a = is_fp16 ? "cutlass::half_t" : "cutlass::bfloat16_t";

  jit::CompileSpec spec;
  spec.template_path = cfg.src_root + "/sycl/GroupGemmW4A16Xe20LauncherInstance.cpp.in";
  spec.subs["POLICY"] = w4a16_policy(avg_m);
  spec.subs["ELEMENT_A"] = elem_a;
  spec.subs["ELEMENT_S"] = is_int4 ? elem_a : "uint8_t";
  const jit::ArchProfile& prof = jit::arch_profile(static_cast<jit::Arch>(arch));
  spec.extra_flags = {"-DSGL_W4A16_JIT_ENTRY"};
  if (!prof.macro.empty()) spec.extra_flags.push_back("-D" + prof.macro);
  spec.target = prof.target;
  spec.entry_symbol = "sgl_moe_w4a16_entry";
  spec.name = std::string("group_gemm_w4a16_p") + std::to_string(policy_id) + (is_int4 ? "_int4" : "_mxfp4") +
              (is_fp16 ? "_fp16" : "_bf16") + "_" + prof.suffix;

  void* sym = jit::get_or_compile(spec, cfg, err);
  if (!sym) return nullptr;

  W4A16Fn fn = reinterpret_cast<W4A16Fn>(sym);
  {
    std::lock_guard<std::mutex> lk(g_w4a16_mu);
    g_w4a16_fns[key] = fn;
  }
  return fn;
}

}  // namespace

bool w4a16_grouped_gemm_launch(
    int avg_m,
    bool is_int4,
    bool is_fp16,
    void* queue,
    const void* activations,
    const void* packed_weights,
    const void* scales,
    const void* zeros,
    const void* bias,
    void* outputs,
    int gemm_n,
    int gemm_k,
    const int* rows_per_expert,
    int num_experts,
    int group_size,
    int* atomic_buffer,
    int arch,
    std::string* err) {
  W4A16Fn fn = resolve_w4a16(avg_m, is_int4, is_fp16, arch, err);
  if (!fn) return false;
  fn(queue,
     activations,
     packed_weights,
     scales,
     zeros,
     bias,
     outputs,
     gemm_n,
     gemm_k,
     rows_per_expert,
     num_experts,
     group_size,
     atomic_buffer);
  return true;
}

}  // namespace moe_jit
}  // namespace sgl
