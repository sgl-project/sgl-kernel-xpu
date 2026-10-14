/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
/*!
  \file
  \brief Forward declarations of the generated per-(dtype, tile) launch
         functions of the chunked-SGMV LoRA expand kernel. The definitions are produced by
         chunked_sgmv_lora_expand_fwd_kernel.cpp.in via ChunkedSgmvLoraExpandFwdXe20.cmake,
         each in its own translation unit (parallel compilation; the entrypoint
         stays free of kernel code).

         Naming: launch_chunked_sgmv_lora_expand_fwd_<ELEM_TAG>_<TILE_TAG>
           ELEM_TAG  in {half, bf16}
           TILE_TAG  in {wide}                 -- see chunked_sgmv_lora_expand_fwd_types.hpp
*/

#pragma once

#include <ATen/ATen.h>
#include <torch/all.h>

#include <cstdint>
#include <optional>
#include <sycl/sycl.hpp>

namespace chunked_sgmv_lora_expand_fwd_impl {

#define DECLARE_CHUNKED_SGMV_LORA_EXPAND_FWD_LAUNCH(ELEM, TILE) \
  void launch_chunked_sgmv_lora_expand_fwd_##ELEM##_##TILE(     \
      const torch::Tensor& input_x,                             \
      const torch::Tensor& weights,                             \
      torch::Tensor& output,                                    \
      const std::optional<torch::Tensor>& base_output,          \
      const torch::Tensor& slice_offsets_i32,                   \
      const torch::Tensor& seg_indptr_i32,                      \
      const torch::Tensor& weight_indices_i32,                  \
      const torch::Tensor& scalings_f32,                        \
      const int64_t* permutation,                               \
      const int num_slices,                                     \
      const int num_segments,                                   \
      sycl::queue& queue);

// One declaration per registered tile. Extend as tiles are added.
#define DECLARE_CHUNKED_SGMV_LORA_EXPAND_FWD_ALL(ELEM) DECLARE_CHUNKED_SGMV_LORA_EXPAND_FWD_LAUNCH(ELEM, wide)

DECLARE_CHUNKED_SGMV_LORA_EXPAND_FWD_ALL(half)
DECLARE_CHUNKED_SGMV_LORA_EXPAND_FWD_ALL(bf16)

#undef DECLARE_CHUNKED_SGMV_LORA_EXPAND_FWD_LAUNCH
#undef DECLARE_CHUNKED_SGMV_LORA_EXPAND_FWD_ALL

}  // namespace chunked_sgmv_lora_expand_fwd_impl
