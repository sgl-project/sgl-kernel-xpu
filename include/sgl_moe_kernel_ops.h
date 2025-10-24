/* Copyright 2025 SGLang Team. All Rights Reserved.

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

#pragma once

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <torch/torch.h>

/*
 * FusedMoE operations for Intel XPU
 * Clean interface with no duplicate declarations
 */

// Main fused MoE forward function
at::Tensor fused_moe_forward(
    const at::Tensor& hidden_states,
    const at::Tensor& gate_weights,
    const at::Tensor& up_weights,
    const at::Tensor& down_weights,
    const at::Tensor& topk_weights,
    const at::Tensor& topk_indices,
    int64_t top_k,
    bool renormalize = false,
    bool inplace = false,
    bool use_grouped_topk = false
);

// MoE alignment function for block-wise processing  
void moe_align_block_size_xpu(
    const at::Tensor& topk_ids,
    int64_t num_experts,
    int64_t block_size,
    at::Tensor& sorted_token_ids,
    at::Tensor& experts_ids,
    at::Tensor& num_tokens_post_pad
);

// Grouped GEMM for MoE expert computation
void moe_grouped_gemm_xpu(
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    at::Tensor& D,
    const at::Tensor& topk_weights,
    const at::Tensor& topk_indices,
    int64_t top_k,
    bool trans_b = false
);

// SiLU activation with multiplication for MoE gating
void silu_and_mul_moe_xpu(
    at::Tensor& gate_output,
    const at::Tensor& up_output
);