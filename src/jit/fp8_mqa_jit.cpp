#include "jit/fp8_mqa_jit.h"

#include <cstdint>
#include <mutex>

#include "jit/sycl_template_jit.h"

namespace sgl {
namespace fp8_mqa_jit {

namespace {

using KernelFn = void (*)(
    void*, const void*, const void*, void*, int, int, int, int, int64_t, int64_t, int64_t);

std::mutex g_mu;
KernelFn g_fn = nullptr;

KernelFn resolve(std::string* err) {
  {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_fn) return g_fn;
  }

  const jit::JitConfig& cfg = jit::default_config();
  if (!cfg.valid) {
    if (err) *err = "FP8 MQA GEMM JIT unavailable: " + cfg.error;
    return nullptr;
  }
  if (cfg.src_root.empty()) {
    if (err) *err = "FP8 MQA GEMM JIT: source template root not resolved";
    return nullptr;
  }

  jit::CompileSpec spec;
  spec.template_path = cfg.src_root + "/sycl/Fp8MqaGemmXe20LauncherInstance.cpp.in";
  // Single shipping tile (mirrors GemmTileShape in Fp8MqaLogitsXe20.cpp).
  spec.subs["GEMM_TILE"] = "cute::Shape<cute::_32, cute::_128, cute::_32>";
  spec.extra_flags = {"-DSGL_FP8_MQA_JIT_ENTRY"};
  spec.entry_symbol = "sgl_fp8_mqa_gemm_entry";
  spec.name = "fp8_mqa_gemm_xe20_32_128_32";

  void* sym = jit::get_or_compile(spec, cfg, err);
  if (!sym) return nullptr;

  KernelFn fn = reinterpret_cast<KernelFn>(sym);
  {
    std::lock_guard<std::mutex> lk(g_mu);
    g_fn = fn;
  }
  return fn;
}

}  // namespace

bool gemm_launch(
    void* queue, const void* A_fp8, const void* B_fp8, void* D_f32, int batch, int M, int N, int K,
    int64_t A_batch_stride, int64_t B_batch_stride, int64_t D_batch_stride, std::string* err) {
  KernelFn fn = resolve(err);
  if (!fn) return false;
  fn(queue, A_fp8, B_fp8, D_f32, batch, M, N, K, A_batch_stride, B_batch_stride, D_batch_stride);
  return true;
}

}  // namespace fp8_mqa_jit
}  // namespace sgl
