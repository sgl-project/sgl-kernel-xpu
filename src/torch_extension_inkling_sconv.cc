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
      "inkling_sconv_forward(Tensor x, Tensor weight, Tensor sconv_cache, Tensor cache_mask, Tensor safe_idx, "
      "Tensor cu, Tensor si, bool silu_activation, bool use_residual, bool is_decode) -> Tensor");
  m.impl("inkling_sconv_forward", torch::kXPU, &inkling_sconv_forward);

  m.def(
      "inkling_update_sconv_cache(Tensor x, Tensor(a!) sconv_cache, Tensor cache_indices, "
      "Tensor has_initial_state, Tensor query_start_loc) -> ()");
  m.impl("inkling_update_sconv_cache", torch::kXPU, &inkling_update_sconv_cache);

  m.def(
      "inkling_fused_decode_update_sconv(Tensor x, Tensor weight, Tensor(a!) sconv_cache, "
      "Tensor cache_indices, Tensor cache_mask, bool silu_activation, bool use_residual, "
      "Tensor? track_mask=None, Tensor? track_indices=None) -> Tensor");
  m.impl("inkling_fused_decode_update_sconv", torch::kXPU, &inkling_fused_decode_update_sconv);

  m.def(
      "inkling_gather_scatter_sconv_cache(Tensor hidden_states, Tensor(a!) sconv_cache, "
      "Tensor track_conv_indices, Tensor mask, Tensor dst_indices) -> ()");
  m.impl("inkling_gather_scatter_sconv_cache", torch::kXPU, &inkling_gather_scatter_sconv_cache);

  m.def(
      "inkling_draft_extend_sconv_cache(Tensor hidden_states, Tensor(a!) sconv_cache, Tensor cache_indices, "
      "Tensor num_accepted_tokens, int draft_token_num, bool do_tracking, Tensor? crossed=None, "
      "Tensor? track_step=None, Tensor? mamba_track_indices=None) -> ()");
  m.impl("inkling_draft_extend_sconv_cache", torch::kXPU, &inkling_draft_extend_sconv_cache);

  m.def(
      "inkling_fused_decode_sconv_metadata(int B, Tensor cache_indices) -> "
      "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.impl("inkling_fused_decode_sconv_metadata", torch::kXPU, &inkling_fused_decode_sconv_metadata);

  m.def(
      "inkling_fused_extend_sconv_metadata(int B, int T, Tensor cache_indices, int his_mode, "
      "Tensor? extend_seq_lens=None, Tensor? his_src=None, int draft_token_num=1) -> "
      "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.impl("inkling_fused_extend_sconv_metadata", torch::kXPU, &inkling_fused_extend_sconv_metadata);

  m.def(
      "inkling_track_conv_indices(Tensor query_start_loc, Tensor mamba_track_seqlens, Tensor extend_prefix_lens, "
      "int width_minus_one, int chunk_size, int total_tokens) -> Tensor");
  m.impl("inkling_track_conv_indices", torch::kXPU, &inkling_track_conv_indices);

  m.def(
      "inkling_save_intermediate_conv_windows(Tensor sconv_cache, Tensor hidden_states, Tensor cache_indices, "
      "Tensor(a!) intermediate_out, int batch_size, int draft_token_num) -> ()");
  m.impl("inkling_save_intermediate_conv_windows", torch::kXPU, &inkling_save_intermediate_conv_windows);
}

REGISTER_EXTENSION(inkling_sconv_ops)
