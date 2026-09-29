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
 **************************************************************************************************/
/*!
  \file
  \brief Forward declarations of the generated per-dtype launch functions for
         gate_up_lora_b_fwd. The definitions are produced by
         gate_up_lora_b_fwd_kernel.cpp.in via GateUpLoraBFwdXe20.cmake, each in
         its own translation unit for parallel compilation of the heavy CUTLASS
         template instantiation.

         Naming: launch_gate_up_lora_b_fwd_<ELEM_TAG>_<TILE_TAG>
         Parameters:
           ELEM_TAG in {half, bf16}       -- fp16/bf16.
           TILE_TAG in {tall}             -- extend via GateUpLoraBFwdXe20.cmake
                                             + a new option tag/type in
                                             gate_up_lora_b_fwd_types.hpp.
*/

#pragma once

#include <ATen/ATen.h>
#include <torch/all.h>

#include <optional>
#include <sycl/sycl.hpp>

namespace gate_up_lora_b_fwd_impl {

// Each function is defined in a separate generated .cpp file from
// gate_up_lora_b_fwd_kernel.cpp.in, compiled as its own library.
#define DECLARE_GATE_UP_LORA_B_FWD_LAUNCH(ELEM, TILE)  \
  void launch_gate_up_lora_b_fwd_##ELEM##_##TILE(      \
      const torch::Tensor& input_x,                    \
      const torch::Tensor& gate_up_lora_b,             \
      const torch::Tensor& seg_indptr_i32,             \
      const torch::Tensor& weight_indices_i32,         \
      const torch::Tensor& scalings,                   \
      torch::Tensor& output,                           \
      const std::optional<torch::Tensor>& base_output, \
      const int output_dim,                            \
      const int num_segments,                          \
      sycl::queue& queue);

// One declaration per registered tile. Extend as tiles are added.
#define DECLARE_GATE_UP_LORA_B_FWD_ALL_TILES(ELEM) DECLARE_GATE_UP_LORA_B_FWD_LAUNCH(ELEM, tall)

DECLARE_GATE_UP_LORA_B_FWD_ALL_TILES(half)
DECLARE_GATE_UP_LORA_B_FWD_ALL_TILES(bf16)

#undef DECLARE_GATE_UP_LORA_B_FWD_LAUNCH
#undef DECLARE_GATE_UP_LORA_B_FWD_ALL_TILES

}  // namespace gate_up_lora_b_fwd_impl
