#include <torch/all.h>
#include <torch/library.h>

#include <ATen/xpu/XPUContext.h>

#include "gdn_attn/gated_delta_rule.hpp"

// Side-library entry points. The base image's sgl_kernel wheel already
// registers gdn_attention and causal_conv1d_update. These two verify ops
// are newer than that wheel, so they are added here without replacing it.

at::Tensor causal_conv1d_update_verify(
    const at::Tensor& x,
    const at::Tensor& conv_state,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias_,
    bool silu_activation,
    const std::optional<at::Tensor>& cache_indices_,
    at::Tensor& intermediate_conv_window,
    const at::Tensor& intermediate_state_indices,
    const std::optional<at::Tensor>& retrieve_next_token,
    const std::optional<at::Tensor>& retrieve_next_sibling,
    const std::optional<at::Tensor>& retrieve_parent_token,
    int64_t pad_slot_id);

void gated_delta_rule_verify(
    torch::Tensor& core_attn_out,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& b,
    const torch::Tensor& a,
    const torch::Tensor& A_log,
    const torch::Tensor& dt_bias,
    const torch::Tensor& ssm_state,
    const torch::Tensor& cache_indices,
    torch::Tensor& intermediate_states_buffer,
    const torch::Tensor& intermediate_state_indices,
    int64_t cache_steps,
    const std::optional<torch::Tensor>& token_indx,
    const std::optional<torch::Tensor>& retrieve_parent_token) {
  auto& queue = at::xpu::getCurrentXPUStream().queue();
  gdn::gated_delta_rule_verify(
      queue,
      core_attn_out,
      q,
      k,
      v,
      b,
      a,
      A_log,
      dt_bias,
      ssm_state,
      cache_indices,
      intermediate_states_buffer,
      intermediate_state_indices,
      cache_steps,
      token_indx,
      retrieve_parent_token);
}

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def(
      "gated_delta_rule_verify(Tensor! core_attn_out, Tensor q, Tensor k, Tensor v, Tensor b, Tensor a, "
      "Tensor A_log, Tensor dt_bias, Tensor ssm_state, Tensor cache_indices, "
      "Tensor! intermediate_states_buffer, Tensor intermediate_state_indices, int cache_steps, "
      "Tensor? token_indx, Tensor? retrieve_parent_token) -> ()");
  m.impl("gated_delta_rule_verify", torch::kXPU, &gated_delta_rule_verify);

  m.def(
      "causal_conv1d_update_verify(Tensor x, Tensor conv_state, Tensor weight, Tensor? bias_, "
      "bool silu_activation, Tensor? cache_indices_, Tensor! intermediate_conv_window, "
      "Tensor intermediate_state_indices, Tensor? retrieve_next_token, Tensor? retrieve_next_sibling, "
      "Tensor(a!)? retrieve_parent_token, int pad_slot_id) -> Tensor");
  m.impl("causal_conv1d_update_verify", torch::kXPU, &causal_conv1d_update_verify);
}
