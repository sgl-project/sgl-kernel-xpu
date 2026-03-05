#define SYCL_INTEL_TARGET 35

#include "kernels/moe/xe20/GroupGemm.hpp"

void moe_grouped_mm_nt_xe35(
    torch::Tensor& output,
    const torch::Tensor& activations,
    const torch::Tensor& weights,
    const std::optional<at::Tensor>& bias,
    const torch::Tensor& total_rows_for_experts,
    const int64_t n_experts,
    const int64_t activation_type,  // 0=silu, 1=gelu
    bool fuse_act) {
  moe_grouped_mm_nt_xe20(
      output, activations, weights, bias, total_rows_for_experts, n_experts, activation_type, fuse_act);
}

#undef SYCL_INTEL_TARGET
