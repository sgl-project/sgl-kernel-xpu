/* Copyright 2026 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "sgl_kernel_ops.h"

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def(
      "inkling_attn_prologue_verify(Tensor qkvr, Tensor k_cache, Tensor v_cache, "
      "Tensor cache_indices, Tensor cache_mask, Tensor k_weight, Tensor v_weight, "
      "Tensor(a!) k_inter, Tensor(a!) v_inter, Tensor q_gamma, Tensor k_gamma, float eps, "
      "Tensor loc, Tensor(a!) k_buf, Tensor(a!) v_buf, int q_off, int k_off, int v_off, "
      "int dq, int dkv, int draft_token_num, bool silu_activation, bool use_residual, "
      "bool do_store, Tensor log_scaling_tau) -> (Tensor, Tensor, Tensor)");
  m.impl("inkling_attn_prologue_verify", torch::kXPU, &inkling_attn_prologue_verify);

  m.def(
      "inkling_attn_prologue_decode(Tensor qkvr, Tensor(a!) k_cache, Tensor(a!) v_cache, "
      "Tensor cache_indices, Tensor cache_mask, Tensor k_weight, Tensor v_weight, "
      "Tensor? track_mask, Tensor? track_indices, Tensor q_gamma, Tensor k_gamma, float eps, "
      "Tensor loc, Tensor(a!) k_buf, Tensor(a!) v_buf, int q_off, int k_off, int v_off, "
      "int dq, int dkv, bool silu_activation, bool use_residual, bool do_store, "
      "Tensor log_scaling_tau) "
      "-> (Tensor, Tensor, Tensor)");
  m.impl("inkling_attn_prologue_decode", torch::kXPU, &inkling_attn_prologue_decode);

  m.def(
      "inkling_attn_prologue_extend(Tensor qkvr, Tensor(a!) k_cache, Tensor(a!) v_cache, "
      "Tensor cache_indices, Tensor cache_mask, Tensor has_initial_state, Tensor cu, Tensor si, "
      "Tensor k_weight, Tensor v_weight, Tensor? track_rows, Tensor? track_mask, Tensor? track_dst, "
      "Tensor q_gamma, Tensor k_gamma, float eps, Tensor loc, Tensor(a!) k_buf, Tensor(a!) v_buf, "
      "int q_off, int k_off, int v_off, int dq, int dkv, bool silu_activation, "
      "bool use_residual, bool do_store, bool do_cache_update, Tensor log_scaling_tau) "
      "-> (Tensor, Tensor, Tensor)");
  m.impl("inkling_attn_prologue_extend", torch::kXPU, &inkling_attn_prologue_extend);

  m.def(
      "inkling_attn_prologue_verify_mxfp8(Tensor qkvr, Tensor k_cache, Tensor v_cache, "
      "Tensor cache_indices, Tensor cache_mask, Tensor k_weight, Tensor v_weight, "
      "Tensor(a!) k_inter, Tensor(a!) v_inter, Tensor q_gamma, Tensor k_gamma, float eps, "
      "Tensor loc, Tensor(b!) k_buf, Tensor(c!) v_buf, Tensor(d!) sfk, Tensor(e!) sfv, "
      "int q_off, int k_off, int v_off, int dq, int dkv, int draft_token_num, "
      "bool silu_activation, bool use_residual, bool do_store, int page_size, "
      "Tensor log_scaling_tau) -> (Tensor, Tensor, Tensor, Tensor)");
  m.impl("inkling_attn_prologue_verify_mxfp8", torch::kXPU, &inkling_attn_prologue_verify_mxfp8);

  m.def(
      "inkling_attn_prologue_decode_mxfp8(Tensor qkvr, Tensor(a!) k_cache, Tensor(a!) v_cache, "
      "Tensor cache_indices, Tensor cache_mask, Tensor k_weight, Tensor v_weight, "
      "Tensor? track_mask, Tensor? track_indices, Tensor q_gamma, Tensor k_gamma, float eps, "
      "Tensor loc, Tensor(b!) k_buf, Tensor(c!) v_buf, Tensor(d!) sfk, Tensor(e!) sfv, "
      "int q_off, int k_off, int v_off, int dq, int dkv, bool silu_activation, "
      "bool use_residual, bool do_store, int page_size, Tensor log_scaling_tau) "
      "-> (Tensor, Tensor, Tensor, Tensor)");
  m.impl("inkling_attn_prologue_decode_mxfp8", torch::kXPU, &inkling_attn_prologue_decode_mxfp8);

  m.def(
      "inkling_attn_prologue_extend_mxfp8(Tensor qkvr, Tensor(a!) k_cache, Tensor(a!) v_cache, "
      "Tensor cache_indices, Tensor cache_mask, Tensor has_initial_state, Tensor cu, Tensor si, "
      "Tensor k_weight, Tensor v_weight, Tensor? track_rows, Tensor? track_mask, Tensor? track_dst, "
      "Tensor q_gamma, Tensor k_gamma, float eps, Tensor loc, Tensor(b!) k_buf, Tensor(c!) v_buf, "
      "Tensor(d!) sfk, Tensor(e!) sfv, int q_off, int k_off, int v_off, int dq, int dkv, "
      "bool silu_activation, bool use_residual, bool do_store, bool do_cache_update, "
      "int page_size, Tensor log_scaling_tau) -> (Tensor, Tensor, Tensor, Tensor)");
  m.impl("inkling_attn_prologue_extend_mxfp8", torch::kXPU, &inkling_attn_prologue_extend_mxfp8);
}

REGISTER_EXTENSION(inkling_attn_prologue_ops)
