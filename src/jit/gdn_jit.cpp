#include "jit/gdn_jit.h"

#include <cstdint>

#include "jit/jit_arch.h"
#include "jit/sycl_template_jit.h"

namespace sgl {
namespace gdn_jit {

namespace {

using ChunkFn = void (*)(
    void*,
    void*,
    const void*,
    const void*,
    const void*,
    void*,
    void*,
    void*,
    const void*,
    const void*,
    const void*,
    const void*,
    void*,
    int,
    const int*,
    const int*,
    const bool*,
    const int*,
    int,
    int,
    int,
    int,
    int,
    int);

// @SCALAR_T@ / @STATE_T@ template substitutions. Indexed by is_half and
// state_code (0=fp32, 1=bf16, 2=half).
const char* scalar_type(bool is_half) {
  return is_half ? "cutlass::half_t" : "cutlass::bfloat16_t";
}

const char* state_type(int state_code) {
  switch (state_code) {
    case 0:
      return "float";
    case 1:
      return "cutlass::bfloat16_t";
    default:
      return "cutlass::half_t";
  }
}

uint64_t pack_key(int arch, bool is_half, int state_code) {
  uint64_t k = static_cast<uint64_t>(arch) & 0xFF;
  k = (k << 8) | (static_cast<uint64_t>(is_half ? 1u : 0u));
  k = (k << 8) | (static_cast<uint64_t>(state_code) & 0xFF);
  return k;
}

jit::JitFnCache<ChunkFn> g_fns("GDN chunk");

ChunkFn resolve(bool is_half, int state_code, int arch, std::string* err) {
  const uint64_t key = pack_key(arch, is_half, state_code);
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
    spec.template_path = cfg.src_root + "/sycl/kernels/gdn_attn/chunk_gated_delta_rule_jit_instance.cpp.in";
    spec.subs["SCALAR_T"] = scalar_type(is_half);
    spec.subs["STATE_T"] = state_type(state_code);
    const jit::ArchSpec as = jit::arch_spec(static_cast<jit::Arch>(arch), "-DSGL_GDN_JIT_ENTRY");
    spec.extra_flags = as.extra_flags;
    spec.target = as.target;
    spec.entry_symbol = "sgl_gdn_chunk_entry";
    spec.name =
        std::string("gdn_chunk_") + (is_half ? "f16" : "bf16") + "_s" + std::to_string(state_code) + "_" + as.suffix;

    return jit::get_or_compile(spec, cfg, berr);
  };
  return g_fns.get(key, build, err);
}

}  // namespace

bool chunk_launch(
    bool is_half,
    int state_code,
    void* queue,
    void* core_attn_out,
    const void* q,
    const void* k,
    const void* v,
    void* A,
    void* w,
    void* u,
    const void* b,
    const void* a,
    const void* A_log,
    const void* dt_bias,
    void* ssm_state,
    int ssm_state_stride_0,
    const int* query_start_loc,
    const int* cache_indices,
    const bool* has_initial_state,
    const int* token_indx,
    int batch_size,
    int total_virtual_seqlen,
    int num_k_heads,
    int head_k_dim,
    int num_v_heads,
    int head_v_dim,
    int arch,
    std::string* err) {
  ChunkFn fn = resolve(is_half, state_code, arch, err);
  if (!fn) return false;
  fn(queue,
     core_attn_out,
     q,
     k,
     v,
     A,
     w,
     u,
     b,
     a,
     A_log,
     dt_bias,
     ssm_state,
     ssm_state_stride_0,
     query_start_loc,
     cache_indices,
     has_initial_state,
     token_indx,
     batch_size,
     total_virtual_seqlen,
     num_k_heads,
     head_k_dim,
     num_v_heads,
     head_v_dim);
  return true;
}

}  // namespace gdn_jit
}  // namespace sgl
