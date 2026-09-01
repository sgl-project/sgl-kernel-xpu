#include "jit/moe_jit.h"

#include <cstdint>

#include "jit/jit_arch.h"
#include "jit/sycl_template_jit.h"
#include "sycl/kernels/moe/xe20/bf16/grouped_gemm_dispatch.h"

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

// (tile, subgroup layout) strings generated from the SAME per-tile tokens the
// AOT dispatcher (GroupGemmXe20.cpp) pastes into cute Shape<>/Layout<>. Both
// sides consume sgl::moe::SGL_MOE_GG_{SHAPE,LAYOUT}_n, so the tile table is
// defined exactly once (grouped_gemm_dispatch.h).
#define SGL_MOE_STR(...) #__VA_ARGS__
#define SGL_MOE_XSTR(...) SGL_MOE_STR(__VA_ARGS__)
#define SGL_MOE_TILE_ROW(id) {SGL_MOE_XSTR(SGL_MOE_GG_SHAPE_##id), SGL_MOE_XSTR(SGL_MOE_GG_LAYOUT_##id)},
const TileCfg kTiles[sgl::moe::kGroupedGemmNumTiles] = {SGL_MOE_TILE_ROW(0) SGL_MOE_TILE_ROW(1) SGL_MOE_TILE_ROW(
    2) SGL_MOE_TILE_ROW(3) SGL_MOE_TILE_ROW(4) SGL_MOE_TILE_ROW(5) SGL_MOE_TILE_ROW(6)};
#undef SGL_MOE_TILE_ROW
#undef SGL_MOE_XSTR
#undef SGL_MOE_STR

uint64_t pack_key(int tile_id, int act, bool fuse, bool bias, int arch) {
  uint64_t k = static_cast<uint64_t>(arch) & 0xFF;
  k = (k << 8) | (static_cast<uint64_t>(tile_id) & 0xFF);
  k = (k << 8) | (static_cast<uint64_t>(act) & 0xFF);
  k = (k << 1) | (fuse ? 1u : 0u);
  k = (k << 1) | (bias ? 1u : 0u);
  return k;
}

jit::JitFnCache<KernelFn> g_fns("MoE grouped GEMM");

KernelFn resolve(int tile_id, int act, bool fuse, bool bias, int arch, std::string* err) {
  const uint64_t key = pack_key(tile_id, act, fuse, bias, arch);
  auto build = [&](std::string* berr) -> void* {
    const jit::JitConfig& cfg = jit::default_config();
    if (!cfg.valid) {
      *berr = "unavailable: " + cfg.error;
      return nullptr;
    }
    if (cfg.src_root.empty()) {
      *berr = "source template root not resolved";
      return nullptr;
    }

    jit::CompileSpec spec;
    spec.template_path = cfg.src_root + "/sycl/GroupGemmXe20LauncherInstance.cpp.in";
    spec.subs["TILE"] = kTiles[tile_id].tile;
    spec.subs["SGLAYOUT"] = kTiles[tile_id].sglayout;
    spec.subs["ACT_TYPE"] = std::to_string(act);
    spec.subs["FUSE_ACT"] = fuse ? "true" : "false";
    spec.subs["WITH_BIAS"] = bias ? "true" : "false";
    const jit::ArchSpec as = jit::arch_spec(static_cast<jit::Arch>(arch), "-DSGL_MOE_JIT_ENTRY");
    spec.extra_flags = as.extra_flags;
    spec.target = as.target;
    spec.entry_symbol = "sgl_moe_gg_entry";
    spec.name = std::string("group_gemm_xe20_t") + std::to_string(tile_id) + "_a" + std::to_string(act) + "_f" +
                (fuse ? "1" : "0") + "_b" + (bias ? "1" : "0") + "_" + as.suffix;

    return jit::get_or_compile(spec, cfg, berr);
  };
  return g_fns.get(key, build, err);
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
  const int tile_id = sgl::moe::grouped_gemm_select_tile(avg_m, gemm_k, gemm_n, fuse_act);
  // Honor the same per-tile fuse policy the AOT dispatcher uses (single source in
  // grouped_gemm_dispatch.h) so both paths select the identical fuse variant for
  // a given tile; tiles whose fuse is fixed compile only that variant.
  const bool eff_fuse = sgl::moe::grouped_gemm_effective_fuse(tile_id, fuse_act);
  KernelFn fn = resolve(tile_id, activation_type, eff_fuse, with_bias, arch, err);
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

// Policy id is selected by GroupGemmW4A16Xe20.cpp so AOT and JIT dispatch use
// the same avg_m- and gemm_n-dependent decision.
const char* w4a16_policy(int policy_id) {
  switch (policy_id) {
    case 0:
      return "w4a16_policy_m_8_n_64";
    case 1:
      return "w4a16_policy_m_16_n_64";
    case 2:
      return "w4a16_policy_m_32_n_64";
    case 3:
      return "w4a16_policy_m_64_n_128";
    case 4:
      return "w4a16_policy_m_128_n_128";
    case 5:
      return "w4a16_policy_m_64_n_256";
    default:
      return nullptr;
  }
}

uint64_t pack_w4a16_key(int policy_id, bool is_int4, bool is_fp16, int arch) {
  uint64_t k = static_cast<uint64_t>(arch) & 0xFF;
  k = (k << 8) | (static_cast<uint64_t>(policy_id) & 0xFF);
  k = (k << 1) | (is_int4 ? 1u : 0u);
  k = (k << 1) | (is_fp16 ? 1u : 0u);
  return k;
}

jit::JitFnCache<W4A16Fn> g_w4a16_fns("W4A16 grouped GEMM");

W4A16Fn resolve_w4a16(int policy_id, bool is_int4, bool is_fp16, int arch, std::string* err) {
  const char* policy = w4a16_policy(policy_id);
  if (policy == nullptr) {
    if (err != nullptr) *err = "invalid W4A16 policy id";
    return nullptr;
  }

  const uint64_t key = pack_w4a16_key(policy_id, is_int4, is_fp16, arch);
  auto build = [&](std::string* berr) -> void* {
    const jit::JitConfig& cfg = jit::default_config();
    if (!cfg.valid) {
      *berr = "unavailable: " + cfg.error;
      return nullptr;
    }
    if (cfg.src_root.empty()) {
      *berr = "source template root not resolved";
      return nullptr;
    }

    const char* elem_a = is_fp16 ? "cutlass::half_t" : "cutlass::bfloat16_t";

    jit::CompileSpec spec;
    spec.template_path = cfg.src_root + "/sycl/GroupGemmW4A16Xe20LauncherInstance.cpp.in";
    spec.subs["POLICY"] = policy;
    spec.subs["ELEMENT_A"] = elem_a;
    spec.subs["ELEMENT_S"] = is_int4 ? elem_a : "uint8_t";
    const jit::ArchSpec as = jit::arch_spec(static_cast<jit::Arch>(arch), "-DSGL_W4A16_JIT_ENTRY");
    spec.extra_flags = as.extra_flags;
    spec.target = as.target;
    spec.entry_symbol = "sgl_moe_w4a16_entry";
    spec.name = std::string("group_gemm_w4a16_p") + std::to_string(policy_id) + (is_int4 ? "_int4" : "_mxfp4") +
                (is_fp16 ? "_fp16" : "_bf16") + "_" + as.suffix;

    return jit::get_or_compile(spec, cfg, berr);
  };
  return g_w4a16_fns.get(key, build, err);
}

}  // namespace

bool w4a16_grouped_gemm_launch(
    int policy_id,
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
  W4A16Fn fn = resolve_w4a16(policy_id, is_int4, is_fp16, arch, err);
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
