#pragma once

#include "cutlass/gemm/dispatch_policy.hpp"

namespace cutlass::gemm {

struct KernelXeMoEGEMM {};
// partial specialization for KernelXeMoEGEMM
template <int Stages_, class KernelScheduler_>
struct MainloopIntelXeXMX16MoE : MainloopIntelXeXMX16<Stages_, KernelScheduler_> {};
}  // namespace cutlass::gemm
