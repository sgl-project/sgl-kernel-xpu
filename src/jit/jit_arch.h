// Per-GPU-architecture JIT profile: maps an Intel Xe architecture to the
// compile knobs its kernels need. Xe20 (BMG) and Xe35 (XE3P) have DIFFERENT
// device kernel code (selected at compile time via -DXPU_ENABLED_XE3P and tuned
// tiles/GRF/subgroups), so a single SPIR-V cannot serve both -- each arch must
// be compiled separately. The caller (a SYCL-aware dispatcher) classifies the
// queue's device and threads the resulting Arch into each op's JIT resolve.
//
// This header is torch-free and SYCL-free so it can live in the sgl_jit static
// library alongside the generic compile engine.
#pragma once

#include <string>

namespace sgl {
namespace jit {

enum class Arch {
  BMG = 0,   // Xe20 / Battlemage (intel_gpu_bmg_g21 / g31)
  XE3P = 1,  // Xe35 / XE3P
};

struct ArchProfile {
  // -fsycl-targets value; empty means "use default_sycl_target()".
  std::string target;
  // Extra preprocessor macro that selects this arch's kernel code path
  // (without the -D prefix); empty for none.
  std::string macro;
  // Short tag appended to CompileSpec::name so cached .so files and logs are
  // legible per arch (e.g. "..._bmg" vs "..._xe3p").
  std::string suffix;
};

// Profile for `a`. Unknown/unspecified archs fall back to BMG.
const ArchProfile& arch_profile(Arch a);

}  // namespace jit
}  // namespace sgl
