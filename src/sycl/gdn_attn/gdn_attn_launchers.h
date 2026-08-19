// JIT-mode forwarding definitions for the GDN non-chunk leaf kernels.
//
// When USE_GDN_ATTN_JIT is set, gdn_attn_interface_impl.hpp includes THIS header
// instead of the heavy causal_conv1d.hpp / chunk_causal_conv1d[_tiled].hpp /
// gated_delta_rule.hpp / l2norm_kernel.hpp, so those definitions are not
// compiled into the AOT op library. Each gdn:: entry below is a thin inline
// forwarder that resolves the matching standalone JIT module (one .so per leaf,
// compiled on first use) and invokes it; call sites in the interface stay
// unchanged.
#pragma once

#include <torch/all.h>

#include <optional>
#include <string>

#include <sycl/sycl.hpp>

#include "gdn_attn_utils.h"  // gdn::ActMode, gdn::gdn_workspace_sections
#include "jit/gdn_attn_jit.h"

namespace gdn {

// Mirrors the value in chunk_causal_conv1d_tiled.hpp (kept in sync). The heavy
// header is not included in JIT mode, so the interface reads it from here.
static constexpr int conv1d_tile_size = 8;

inline void causal_conv1d(
    sycl::queue& queue,
    torch::Tensor& q_out,
    torch::Tensor& k_out,
    torch::Tensor& v_out,
    torch::Tensor& z_out,
    torch::Tensor& b_out,
    torch::Tensor& a_out,
    const torch::Tensor& mixed_qkvz,
    const torch::Tensor& mixed_ba,
    const torch::Tensor& conv_weights,
    const std::optional<torch::Tensor>& conv_bias,
    torch::Tensor& conv_states,
    const std::optional<torch::Tensor>& query_start_loc,
    const std::optional<torch::Tensor>& token_indx,
    const std::optional<torch::Tensor>& cache_indices,
    const std::optional<torch::Tensor>& has_initial_state,
    const std::optional<torch::Tensor>& num_accepted_tokens,
    const ActMode& act_mode,
    const int& pad_slot_id,
    const int num_prefills,
    const int num_decodes,
    const int num_spec_decodes,
    const bool reorder_input) {
  using FnT = decltype(&causal_conv1d);
  static FnT fn = nullptr;
  if (!fn) {
    std::string err;
    fn = reinterpret_cast<FnT>(sgl::gdn_attn_jit::resolve(
        "gdn_causal_conv1d_launcher.cpp.in",
        "SGL_GDN_CAUSAL_CONV1D_JIT_ENTRY",
        "sgl_gdn_causal_conv1d_entry",
        "gdn_causal_conv1d_launcher",
        &err));
    TORCH_CHECK(fn, "GDN causal_conv1d JIT: ", err);
  }
  fn(queue, q_out, k_out, v_out, z_out, b_out, a_out, mixed_qkvz, mixed_ba, conv_weights, conv_bias, conv_states,
     query_start_loc, token_indx, cache_indices, has_initial_state, num_accepted_tokens, act_mode, pad_slot_id,
     num_prefills, num_decodes, num_spec_decodes, reorder_input);
}

inline void gated_delta_rule(
    sycl::queue& queue,
    torch::Tensor& core_attn_out,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& b,
    const torch::Tensor& a,
    const torch::Tensor& A_log,
    const torch::Tensor& dt_bias,
    torch::Tensor& ssm_state,
    const std::optional<torch::Tensor>& query_start_loc,
    const std::optional<torch::Tensor>& token_indx,
    const std::optional<torch::Tensor>& cache_indices,
    const std::optional<torch::Tensor>& has_initial_state,
    const std::optional<torch::Tensor>& num_accepted_tokens,
    const int num_prefills,
    const int num_decodes,
    const int num_spec_decodes) {
  using FnT = decltype(&gated_delta_rule);
  static FnT fn = nullptr;
  if (!fn) {
    std::string err;
    fn = reinterpret_cast<FnT>(sgl::gdn_attn_jit::resolve(
        "gdn_gated_delta_rule_launcher.cpp.in",
        "SGL_GDN_GATED_DELTA_RULE_JIT_ENTRY",
        "sgl_gdn_gated_delta_rule_entry",
        "gdn_gated_delta_rule_launcher",
        &err));
    TORCH_CHECK(fn, "GDN gated_delta_rule JIT: ", err);
  }
  fn(queue, core_attn_out, q, k, v, b, a, A_log, dt_bias, ssm_state, query_start_loc, token_indx, cache_indices,
     has_initial_state, num_accepted_tokens, num_prefills, num_decodes, num_spec_decodes);
}

inline void chunk_causal_conv1d(
    sycl::queue& queue,
    torch::Tensor& q_out,
    torch::Tensor& k_out,
    torch::Tensor& v_out,
    torch::Tensor& z_out,
    torch::Tensor& b_out,
    torch::Tensor& a_out,
    const torch::Tensor& mixed_qkvz,
    const torch::Tensor& mixed_ba,
    const torch::Tensor& conv_weights,
    const std::optional<torch::Tensor>& conv_bias,
    torch::Tensor& conv_states,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& cache_indices,
    const std::optional<torch::Tensor>& has_initial_state,
    const ActMode& act_mode,
    const int& pad_slot_id,
    const int num_prefills,
    const int num_decodes,
    const bool reorder_input,
    const int* token_indx = nullptr,
    int num_actual_tokens_override = -1,
    const bool fuse_l2norm = false) {
  using FnT = void (*)(
      sycl::queue&, torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&,
      const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const std::optional<torch::Tensor>&,
      torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const std::optional<torch::Tensor>&, const ActMode&,
      const int&, const int, const int, const bool, const int*, int, const bool);
  static FnT fn = nullptr;
  if (!fn) {
    std::string err;
    fn = reinterpret_cast<FnT>(sgl::gdn_attn_jit::resolve(
        "gdn_chunk_causal_conv1d_launcher.cpp.in",
        "SGL_GDN_CHUNK_CAUSAL_CONV1D_JIT_ENTRY",
        "sgl_gdn_chunk_causal_conv1d_entry",
        "gdn_chunk_causal_conv1d_launcher",
        &err));
    TORCH_CHECK(fn, "GDN chunk_causal_conv1d JIT: ", err);
  }
  fn(queue, q_out, k_out, v_out, z_out, b_out, a_out, mixed_qkvz, mixed_ba, conv_weights, conv_bias, conv_states,
     query_start_loc, cache_indices, has_initial_state, act_mode, pad_slot_id, num_prefills, num_decodes, reorder_input,
     token_indx, num_actual_tokens_override, fuse_l2norm);
}

inline void chunk_causal_conv1d_tiled(
    sycl::queue& queue,
    torch::Tensor& q_out,
    torch::Tensor& k_out,
    torch::Tensor& v_out,
    torch::Tensor& z_out,
    torch::Tensor& b_out,
    torch::Tensor& a_out,
    const torch::Tensor& mixed_qkvz,
    const torch::Tensor& mixed_ba,
    const torch::Tensor& conv_weights,
    const std::optional<torch::Tensor>& conv_bias,
    torch::Tensor& conv_states,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& cache_indices,
    const std::optional<torch::Tensor>& has_initial_state,
    const ActMode& act_mode,
    const int& pad_slot_id,
    const int num_prefills,
    const int num_decodes,
    const bool reorder_input,
    const int* token_indx = nullptr,
    int num_actual_tokens_override = -1,
    const bool fuse_l2norm = false) {
  using FnT = void (*)(
      sycl::queue&, torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&,
      const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const std::optional<torch::Tensor>&,
      torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const std::optional<torch::Tensor>&, const ActMode&,
      const int&, const int, const int, const bool, const int*, int, const bool);
  static FnT fn = nullptr;
  if (!fn) {
    std::string err;
    fn = reinterpret_cast<FnT>(sgl::gdn_attn_jit::resolve(
        "gdn_chunk_causal_conv1d_tiled_launcher.cpp.in",
        "SGL_GDN_CHUNK_CAUSAL_CONV1D_TILED_JIT_ENTRY",
        "sgl_gdn_chunk_causal_conv1d_tiled_entry",
        "gdn_chunk_causal_conv1d_tiled_launcher",
        &err));
    TORCH_CHECK(fn, "GDN chunk_causal_conv1d_tiled JIT: ", err);
  }
  fn(queue, q_out, k_out, v_out, z_out, b_out, a_out, mixed_qkvz, mixed_ba, conv_weights, conv_bias, conv_states,
     query_start_loc, cache_indices, has_initial_state, act_mode, pad_slot_id, num_prefills, num_decodes, reorder_input,
     token_indx, num_actual_tokens_override, fuse_l2norm);
}

}  // namespace gdn

// l2norm lives in the global namespace (see l2norm_kernel.hpp).
inline void l2norm(sycl::queue& queue, const torch::Tensor& q, const torch::Tensor& k) {
  using FnT = decltype(&l2norm);
  static FnT fn = nullptr;
  if (!fn) {
    std::string err;
    fn = reinterpret_cast<FnT>(sgl::gdn_attn_jit::resolve(
        "gdn_l2norm_launcher.cpp.in", "SGL_GDN_L2NORM_JIT_ENTRY", "sgl_gdn_l2norm_entry", "gdn_l2norm_launcher", &err));
    TORCH_CHECK(fn, "GDN l2norm JIT: ", err);
  }
  fn(queue, q, k);
}
