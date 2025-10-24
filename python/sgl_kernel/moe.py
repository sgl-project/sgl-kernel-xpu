"""
Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================
"""

from typing import Any, Dict, Optional, Tuple
import warnings

import torch

try:
    import sgl_kernel
except ImportError:
    sgl_kernel = None


def _pytorch_moe_forward(
    hidden_states: torch.Tensor,
    gate_weights: torch.Tensor,
    up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    top_k: int,
    renormalize: bool = True,
) -> torch.Tensor:
    """
    Reference MoE implementation using standard PyTorch operations.
    Used as fallback when fused kernels are not available.
    """
    num_tokens, hidden_size = hidden_states.shape
    num_experts, _, intermediate_size = gate_weights.shape
    
    # Initialize output
    output = torch.zeros_like(hidden_states)
    
    # Process each token
    for token_idx in range(num_tokens):
        token_hidden = hidden_states[token_idx:token_idx+1]  # [1, hidden_size]
        token_output = torch.zeros_like(token_hidden)
        
        # Get weights for this token
        token_weights = topk_weights[token_idx]
        token_indices = topk_indices[token_idx]
        
        # Renormalize weights if requested
        if renormalize:
            token_weights = token_weights / token_weights.sum()
        
        # Process each selected expert for this token
        for k in range(top_k):
            expert_idx = token_indices[k].item()
            expert_weight = token_weights[k].item()
            
            if expert_weight < 1e-8:  # Skip negligible weights
                continue
            
            # Gate projection
            gate_proj = torch.matmul(token_hidden, gate_weights[expert_idx])  # [1, intermediate_size]
            
            # Up projection
            up_proj = torch.matmul(token_hidden, up_weights[expert_idx])  # [1, intermediate_size]
            
            # SiLU activation and gating
            gate_activated = gate_proj * torch.sigmoid(gate_proj)  # SiLU
            intermediate = gate_activated * up_proj
            
            # Down projection
            expert_output = torch.matmul(intermediate, down_weights[expert_idx])  # [1, hidden_size]
            
            # Weighted accumulation
            token_output += expert_weight * expert_output
        
        output[token_idx] = token_output.squeeze(0)
    
    return output


def fused_moe_forward(
    hidden_states: torch.Tensor,
    gate_weights: torch.Tensor,
    up_weights: torch.Tensor, 
    down_weights: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    top_k: int = 2,
    renormalize: bool = True,
    inplace: bool = False,
    use_grouped_topk: bool = True,
) -> torch.Tensor:
    """
    Fused MoE forward pass optimized for Intel XPU.
    
    Args:
        hidden_states: Input tensor of shape [num_tokens, hidden_size]
        gate_weights: Gate projection weights [num_experts, hidden_size, intermediate_size]
        up_weights: Up projection weights [num_experts, hidden_size, intermediate_size]  
        down_weights: Down projection weights [num_experts, intermediate_size, hidden_size]
        topk_weights: TopK expert weights [num_tokens, top_k]
        topk_indices: TopK expert indices [num_tokens, top_k]
        top_k: Number of experts to select per token
        renormalize: Whether to renormalize expert weights
        inplace: Whether to perform operations in-place when possible
        use_grouped_topk: Whether to use grouped topk for better efficiency
        
    Returns:
        Output tensor of shape [num_tokens, hidden_size]
    """
    if sgl_kernel is None:
        # Fallback to PyTorch implementation when sgl_kernel is not available
        return _pytorch_moe_forward(hidden_states, gate_weights, up_weights, down_weights, 
                                   topk_weights, topk_indices, top_k, renormalize)
    
    # Input validation for XPU tensors only when sgl_kernel is available
    if hidden_states.device.type == "xpu":
        assert hidden_states.is_xpu, "hidden_states must be on XPU device"
        assert gate_weights.is_xpu, "gate_weights must be on XPU device"
        assert up_weights.is_xpu, "up_weights must be on XPU device"
        assert down_weights.is_xpu, "down_weights must be on XPU device"
        assert topk_weights.is_xpu, "topk_weights must be on XPU device"
        assert topk_indices.is_xpu, "topk_indices must be on XPU device"
    
    # Dtype validation - prefer bfloat16 for Intel XPU
    if hidden_states.dtype not in [torch.bfloat16, torch.float16, torch.float32]:
        warnings.warn(f"Unsupported dtype {hidden_states.dtype}, converting to bfloat16")
        hidden_states = hidden_states.to(torch.bfloat16)
    
    # Ensure all weight tensors have consistent dtype
    target_dtype = hidden_states.dtype
    if gate_weights.dtype != target_dtype:
        gate_weights = gate_weights.to(target_dtype)
    if up_weights.dtype != target_dtype:
        up_weights = up_weights.to(target_dtype)
    if down_weights.dtype != target_dtype:
        down_weights = down_weights.to(target_dtype)
    
    try:
        return torch.ops.sgl_kernel.fused_moe_forward.default(
            hidden_states,
            gate_weights,
            up_weights,
            down_weights,
            topk_weights,
            topk_indices,
            top_k,
            renormalize,
            inplace,
            use_grouped_topk,
        )
    except (RuntimeError, AttributeError):
        # Fallback to PyTorch implementation if kernel is not available
        warnings.warn("Fused MoE kernel not available, using PyTorch fallback")
        return _pytorch_moe_forward(hidden_states, gate_weights, up_weights, down_weights, 
                                   topk_weights, topk_indices, top_k, renormalize)


def grouped_gemm_moe(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    top_k: int,
    trans_b: bool = False,
) -> None:
    """
    Grouped GEMM operation optimized for MoE workloads.
    
    Args:
        A: Input tensor for matrix multiplication [num_tokens, K]
        B: Weight tensor for matrix multiplication [num_experts, K, N] or [num_experts, N, K] if trans_b
        C: Bias tensor (can be zero tensor) [num_experts, N]
        D: Output tensor [num_tokens, N]
        topk_weights: TopK expert weights [num_tokens, top_k]
        topk_indices: TopK expert indices [num_tokens, top_k]
        top_k: Number of experts per token
        trans_b: Whether to transpose B matrix
    """
    if sgl_kernel is None:
        # Fallback implementation
        warnings.warn("Using PyTorch fallback for grouped_gemm_moe")
        _pytorch_grouped_gemm_moe(A, B, C, D, topk_weights, topk_indices, top_k, trans_b)
        return
    
    # Input validation for XPU tensors only when sgl_kernel is available
    if A.device.type == "xpu":
        assert A.is_xpu, "A must be on XPU device"
        assert B.is_xpu, "B must be on XPU device"
        assert C.is_xpu, "C must be on XPU device"
        assert D.is_xpu, "D must be on XPU device"
        assert topk_weights.is_xpu, "topk_weights must be on XPU device"
        assert topk_indices.is_xpu, "topk_indices must be on XPU device"
    
    try:
        torch.ops.sgl_kernel.grouped_gemm_moe.default(
            A, B, C, D, topk_weights, topk_indices, top_k, trans_b
        )
    except (RuntimeError, AttributeError):
        warnings.warn("Fused grouped GEMM kernel not available, using PyTorch fallback")
        _pytorch_grouped_gemm_moe(A, B, C, D, topk_weights, topk_indices, top_k, trans_b)


def _pytorch_grouped_gemm_moe(A, B, C, D, topk_weights, topk_indices, top_k, trans_b):
    """PyTorch fallback for grouped GEMM MoE."""
    num_tokens = A.shape[0]
    D.zero_()  # Initialize output
    
    for token_idx in range(num_tokens):
        for k in range(top_k):
            expert_idx = topk_indices[token_idx, k].item()
            weight = topk_weights[token_idx, k].item()
            
            if weight < 1e-8:
                continue
            
            # Select expert weights
            if trans_b:
                expert_B = B[expert_idx].t()  # Transpose if needed
            else:
                expert_B = B[expert_idx]
            
            # Compute A @ B + C
            result = torch.matmul(A[token_idx:token_idx+1], expert_B) + C[expert_idx:expert_idx+1]
            D[token_idx] += weight * result.squeeze(0)


def silu_and_mul_moe(
    gate_output: torch.Tensor,
    up_output: torch.Tensor,
) -> None:
    """
    Fused SiLU activation and element-wise multiplication for MoE.
    Computes: gate_output = SiLU(gate_output) * up_output
    
    Args:
        gate_output: Gate projection output (modified in-place) [*, intermediate_size]
        up_output: Up projection output [*, intermediate_size]
    """
    if sgl_kernel is None:
        # Fallback implementation
        gate_output.copy_(gate_output * torch.sigmoid(gate_output) * up_output)
        return
    
    # Input validation for XPU tensors only when sgl_kernel is available
    if gate_output.device.type == "xpu":
        assert gate_output.is_xpu, "gate_output must be on XPU device"
        assert up_output.is_xpu, "up_output must be on XPU device"
        assert gate_output.shape == up_output.shape, "gate_output and up_output must have same shape"
    
    try:
        torch.ops.sgl_kernel.silu_and_mul_moe.default(gate_output, up_output)
    except (RuntimeError, AttributeError):
        # Fallback to PyTorch implementation
        gate_output.copy_(gate_output * torch.sigmoid(gate_output) * up_output)


def moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    experts_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    token_cnts_buffer: Optional[torch.Tensor] = None,
    cumsum_buffer: Optional[torch.Tensor] = None,
) -> None:
    """
    Align MoE token routing to block sizes for efficient processing.
    
    Args:
        topk_ids: TopK expert indices [num_tokens, top_k]
        num_experts: Total number of experts
        block_size: Desired block size for alignment
        sorted_token_ids: Output buffer for sorted token IDs 
        experts_ids: Output buffer for expert assignments
        num_tokens_post_pad: Output buffer for token counts after padding
        token_cnts_buffer: Optional buffer for token counts per expert
        cumsum_buffer: Optional buffer for cumulative sums
    """
    if sgl_kernel is None:
        # Fallback implementation
        warnings.warn("Using PyTorch fallback for moe_align_block_size")
        _pytorch_moe_align_block_size(topk_ids, num_experts, block_size, sorted_token_ids, 
                                     experts_ids, num_tokens_post_pad, token_cnts_buffer, cumsum_buffer)
        return
    
    # Input validation for XPU tensors only when sgl_kernel is available
    if topk_ids.device.type == "xpu":
        assert topk_ids.is_xpu, "topk_ids must be on XPU device"
        assert sorted_token_ids.is_xpu, "sorted_token_ids must be on XPU device"
        assert experts_ids.is_xpu, "experts_ids must be on XPU device"
        assert num_tokens_post_pad.is_xpu, "num_tokens_post_pad must be on XPU device"
    
    # Handle optional buffers for compatibility
    if token_cnts_buffer is None:
        token_cnts_buffer = torch.zeros(num_experts, dtype=torch.int32, device=topk_ids.device)
    if cumsum_buffer is None:
        cumsum_buffer = torch.zeros(num_experts + 1, dtype=torch.int32, device=topk_ids.device)
    
    try:
        torch.ops.sgl_kernel.moe_align_block_size.default(
            topk_ids,
            num_experts,
            block_size,
            sorted_token_ids,
            experts_ids,
            num_tokens_post_pad,
            token_cnts_buffer,
            cumsum_buffer,
        )
    except (RuntimeError, AttributeError):
        warnings.warn("Fused moe_align_block_size kernel not available, using PyTorch fallback")
        _pytorch_moe_align_block_size(topk_ids, num_experts, block_size, sorted_token_ids, 
                                     experts_ids, num_tokens_post_pad, token_cnts_buffer, cumsum_buffer)


def _pytorch_moe_align_block_size(topk_ids, num_experts, block_size, sorted_token_ids, 
                                 experts_ids, num_tokens_post_pad, token_cnts_buffer, cumsum_buffer):
    """PyTorch fallback for moe_align_block_size."""
    # Simple fallback - just copy the data without complex alignment
    num_tokens, top_k = topk_ids.shape
    total_assignments = num_tokens * top_k
    
    # Flatten the topk assignments
    flat_tokens = torch.arange(num_tokens, device=topk_ids.device).repeat_interleave(top_k)
    flat_experts = topk_ids.flatten()
    
    # Simple sorting by expert ID
    sorted_indices = torch.argsort(flat_experts)
    
    # Fill output buffers
    if len(sorted_token_ids) >= len(flat_tokens):
        sorted_token_ids[:len(flat_tokens)] = flat_tokens[sorted_indices]
    if len(experts_ids) >= len(flat_experts):
        experts_ids[:len(flat_experts)] = flat_experts[sorted_indices]
    
    # Set the number of tokens (simplified)
    num_tokens_post_pad[0] = total_assignments


def topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: float,
    renormalize: bool = False,
) -> None:
    torch.ops.sgl_kernel.topk_softmax.default(
        topk_weights, topk_ids, gating_output, renormalize
    )


def moe_fused_gate(
    input_tensor,
    bias,
    num_expert_group,
    topk_group,
    topk,
    num_fused_shared_experts=0,
    routed_scaling_factor=0,
):
    # This fused kernel function is used to select topk expert in a hierarchical 2-layer fashion
    # it split group of expert into num_expert_group, and use top2 expert weight sum in each group
    # as the group weight to select expert groups and then select topk experts within the selected groups
    # the #experts is decided by the input tensor shape and we currently only support power of 2 #experts
    # and #experts should be divisible by num_expert_group. #expert/num_expert_group <= 32 is limited for now.
    # for non-supported case, we suggest to use the biased_grouped_topk func in sglang.srt.layers.moe.topk
    # num_fused_shared_experts: if > 0, the last several experts will be replaced with shared experts
    # routed_scaling_factor: if > 0, the shared experts will be scaled by this factor
    return torch.ops.sgl_kernel.moe_fused_gate.default(
        input_tensor,
        bias,
        num_expert_group,
        topk_group,
        topk,
        num_fused_shared_experts,
        routed_scaling_factor,
    )


def ep_moe_pre_reorder(
    input_tensor,
    gateup_input,
    src2dst,
    topk_ids,
    a1_scales,
    start_expert_id,
    end_expert_id,
    topk,
    use_per_token_if_dynamic,
):
    return torch.ops.sgl_kernel.ep_moe_pre_reorder.default(
        input_tensor,
        gateup_input,
        src2dst,
        topk_ids,
        a1_scales,
        start_expert_id,
        end_expert_id,
        topk,
        use_per_token_if_dynamic,
    )


def ep_moe_silu_and_mul(
    gateup_output,
    down_input,
    reorder_topk_ids,
    scales,
    start_expert_id,
    end_expert_id,
):
    return torch.ops.sgl_kernel.ep_moe_silu_and_mul.default(
        gateup_output,
        down_input,
        reorder_topk_ids,
        scales,
        start_expert_id,
        end_expert_id,
    )


def ep_moe_post_reorder(
    down_output,
    output,
    src2dst,
    topk_ids,
    topk_weights,
    start_expert_id,
    end_expert_id,
    topk,
):
    return torch.ops.sgl_kernel.ep_moe_post_reorder.default(
        down_output,
        output,
        src2dst,
        topk_ids,
        topk_weights,
        start_expert_id,
        end_expert_id,
        topk,
    )


def fp8_blockwise_scaled_grouped_mm(
    output,
    a_ptrs,
    b_ptrs,
    out_ptrs,
    a_scales_ptrs,
    b_scales_ptrs,
    a,
    b,
    scales_a,
    scales_b,
    stride_a,
    stride_b,
    stride_c,
    layout_sfa,
    layout_sfb,
    problem_sizes,
    expert_offsets,
    workspace,
):
    torch.ops.sgl_kernel.fp8_blockwise_scaled_grouped_mm.default(
        output,
        a_ptrs,
        b_ptrs,
        out_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        a,
        b,
        scales_a,
        scales_b,
        stride_a,
        stride_b,
        stride_c,
        layout_sfa,
        layout_sfb,
        problem_sizes,
        expert_offsets,
        workspace,
    )


def prepare_moe_input(
    topk_ids,
    expert_offsets,
    problem_sizes1,
    problem_sizes2,
    input_permutation,
    output_permutation,
    num_experts,
    n,
    k,
    blockscale_offsets: Optional[torch.Tensor] = None,
):
    torch.ops.sgl_kernel.prepare_moe_input.default(
        topk_ids,
        expert_offsets,
        blockscale_offsets,
        problem_sizes1,
        problem_sizes2,
        input_permutation,
        output_permutation,
        num_experts,
        n,
        k,
    )


def apply_shuffle_mul_sum(
    input,
    output,
    permutation,
    factors,
):
    torch.ops.sgl_kernel.apply_shuffle_mul_sum.default(
        input, output, permutation, factors
    )


def cutlass_fp4_group_mm(
    a_fp4,
    b_fp4,
    a_blockscale,
    b_blockscale,
    alphas,
    out_dtype,
    device,
    params: Dict[str, Any],
):
    """
    An FP4 Blockscaled Group Gemm that takes in  a_tensors, b_tensors and runs
    the gemms for each combination based on the specified problem sizes.

    This is used as the MoE gemm during NVFP4 Quantized FusedMoE forward.
    - a/b_tensors: the NVFP4 a_ptrs and b_ptrs tensors which are quantized
                     input and expert weights.
    - a_/b_scales: The blockscales in FP8-E4M3 precision
    - ab_strides/c_strides: Strides for the a/b tensors between rows.
    - expert_offsets/sf_offsets: Indices that mark at which token index
                    each expert begins its computation. The number of tokens
                    computed with expert E is expert_offsets[E + 1] -
                    expert_offsets[E] And the sf_size per expert is
                    sf_offset[E+1] - sf_offset[E]
    - problem_sizes: MxNxK sizes of each expert's multiplication in two grouped
                     MMs used in the fused MoE operation.
    """
    m_topk = a_fp4.shape[0]
    n = b_fp4.shape[1]
    c_shape = (m_topk, n)
    c = torch.empty(c_shape, device=device, dtype=out_dtype)
    torch.ops.sgl_kernel.cutlass_fp4_group_mm.default(
        c,
        a_fp4,
        b_fp4,
        a_blockscale,
        b_blockscale,
        alphas,
        params["ab_strides"],
        params["c_strides"],
        params["problem_sizes"],
        params["expert_offsets"],
        params["blockscale_offsets"],
    )
    return c.to(dtype=out_dtype)
