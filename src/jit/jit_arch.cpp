#include "jit/jit_arch.h"

namespace sgl {
namespace jit {

const ArchProfile& arch_profile(Arch a) {
  // BMG keeps `target` empty so the existing SGLANG_SYCL_AOT_TARGETS / default
  // (intel_gpu_bmg_g21) behavior is preserved unchanged.
  static const ArchProfile kBmg{/*target=*/"", /*macro=*/"", /*suffix=*/"bmg"};
  // XE3P selects the Xe35 kernel code path via -DXPU_ENABLED_XE3P. Its concrete
  // -fsycl-targets id is filled in once Xe35 AOT targets are finalized; until
  // then it leaves `target` empty (falls back to the default/env target).
  static const ArchProfile kXe3p{/*target=*/"", /*macro=*/"XPU_ENABLED_XE3P", /*suffix=*/"xe3p"};

  switch (a) {
    case Arch::XE3P:
      return kXe3p;
    case Arch::BMG:
    default:
      return kBmg;
  }
}

}  // namespace jit
}  // namespace sgl
