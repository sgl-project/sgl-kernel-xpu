/* Copyright 2025 SGLang Team. All Rights Reserved.
 *
 * Op registration for the vendored vllm-xpu-kernels gdn_attn module.
 * Schema matches vllm-xpu-kernels/csrc/xpu/torch_bindings.cpp exactly so that
 * existing Python callers (e.g. Qwen3.5 / Qwen3-Next linear-attn prefill)
 * keep working unchanged. The only difference: we publish under the unified
 * `sgl_kernel` torch-library namespace, reachable as
 *     torch.ops.sgl_kernel.gdn_attention(...)
 */
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>
#include <torch/library.h>

// Forward declaration of the vendored implementation (gdn_attn_interface.cpp).
// Must match the signature there exactly.
void gdn_attention(
    torch::Tensor& core_attn_out,
    torch::Tensor& z,
    const torch::Tensor& projected_states_qkvz,
    const torch::Tensor& projected_states_ba,
    const int64_t num_k_heads,
    const int64_t num_v_heads,
    const int64_t head_k_dim,
    const int64_t head_v_dim,
    torch::Tensor& conv_state,
    torch::Tensor& ssm_state,
    const torch::Tensor& conv_weights,
    const std::optional<torch::Tensor>& conv_bias,
    const std::string& activation,
    const torch::Tensor& A_log,
    const torch::Tensor& dt_bias,
    const int64_t num_prefills,
    const int64_t num_decodes,
    const std::optional<torch::Tensor>& has_initial_state,
    const torch::Tensor& non_spec_query_start_loc,
    const torch::Tensor& non_spec_state_indices_tensor,
    const int64_t num_actual_tokens,
    const int64_t tp_size,
    // RADIX TRACK-BUFFER FIX: optional per-chunk intermediate ssm snapshot.
    const std::optional<torch::Tensor>& inter_ssm,
    const std::optional<torch::Tensor>& inter_ssm_indices,
    // RADIX TRACK-BUFFER FIX: optional aligned-boundary conv snapshot.
    const std::optional<torch::Tensor>& inter_conv,
    const std::optional<torch::Tensor>& inter_conv_indices);

// Plain-SYCL RMSNormGated for the GDN output norm (rms_norm_gated_sycl.cpp).
// Replaces the triton layernorm_gated fallback (the ESIMD variant's bucket is
// disabled). out = rmsnorm(x) * weight * silu(z). fp16-only.
namespace gdn {
at::Tensor rms_norm_gated(
    at::Tensor& output,
    const at::Tensor& x,
    const at::Tensor& z,
    const at::Tensor& weight,
    double eps);
}  // namespace gdn

// Plain-SYCL RMSNormGated for the GDN output norm (rms_norm_gated_sycl.cpp).
// Replaces the triton layernorm_gated fallback (the ESIMD variant's bucket is
// disabled). out = rmsnorm(x) * weight * silu(z). fp16-only.
namespace gdn {
at::Tensor rms_norm_gated(
    at::Tensor& output,
    const at::Tensor& x,
    const at::Tensor& z,
    const at::Tensor& weight,
    double eps);
}  // namespace gdn

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def(
      "gdn_rms_norm_gated(Tensor! output, Tensor x, Tensor z, Tensor weight, "
      "float eps) -> Tensor");
  m.impl("gdn_rms_norm_gated", torch::kXPU, &gdn::rms_norm_gated);
  m.def(
      "gdn_attention(Tensor! core_attn_out, Tensor! z, Tensor "
      "projected_states_qkvz, Tensor projected_states_ba,"
      "int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim,"
      "Tensor! conv_state, Tensor! ssm_state, Tensor conv_weights, Tensor? "
      "conv_bias, str activation, Tensor A_log, Tensor dt_bias,"
      "int num_prefills, int num_decodes, Tensor? has_initial_state, Tensor "
      "non_spec_query_start_loc,"
      "Tensor non_spec_state_indices_tensor, int num_actual_tokens, int "
      "tp_size, Tensor!? inter_ssm=None, Tensor? inter_ssm_indices=None, "
      "Tensor!? inter_conv=None, Tensor? inter_conv_indices=None) -> ()");
  m.impl("gdn_attention", torch::kXPU, &gdn_attention);
}
