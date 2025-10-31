from typing import Any, Dict, Optional

import torch


def moe_align_block_size_impl(
    topk_ids,
    num_experts,
    block_size,
    sorted_token_ids,
    experts_ids,
    num_tokens_post_pad,
    cumsum_buffer,
    pad_sorted_token_ids=False,
):
    torch.ops.sgl_kernel.moe_align_block_size.default(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        pad_sorted_token_ids,
    )


def moe_align_block_size(
    topk_ids,
    num_experts,
    block_size,
    pad_sorted_token_ids=False,
):
    max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)

    sorted_ids_xpu = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    if not pad_sorted_token_ids:
        sorted_ids_xpu.fill_(topk_ids.numel())
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids_xpu = torch.zeros(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad_xpu = torch.empty(
        (1), dtype=torch.int32, device=topk_ids.device
    )
    cumsum_buffer = torch.empty(
        num_experts + 2, dtype=torch.int32, device=topk_ids.device
    )
    moe_align_block_size_impl(
        topk_ids,
        num_experts + 1,
        block_size,
        sorted_ids_xpu,
        expert_ids_xpu,
        num_tokens_post_pad_xpu,
        cumsum_buffer,
        pad_sorted_token_ids,
    )

    return sorted_ids_xpu, expert_ids_xpu, num_tokens_post_pad_xpu


def topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: float,
    renormalize: bool = False,
) -> None:
    torch.ops.sgl_kernel.topk_softmax.default(
        topk_weights, topk_ids, gating_output, renormalize
    )


def moe_sum_reduce(
    input_tensor,
    output_tensor,
    routed_scaling_factor=0,
):
    torch.ops.sgl_kernel.moe_sum_reduce.default(
        input_tensor,
        output_tensor,
        routed_scaling_factor,
    )


def moe_sum(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
):
    torch.ops.sgl_kernel.moe_sum.default(
        input_tensor,
        output_tensor,
    )


def moe_fused_gate(
    input_tensor,
    bias,
    num_expert_group,
    topk_group,
    topk,
    num_fused_shared_experts=0,
    routed_scaling_factor=0,
    apply_routed_scaling_factor_on_output=False,
):
    # This fused kernel function is used to select topk expert in a hierarchical 2-layer fashion
    # it split group of expert into num_expert_group, and use top2 expert weight sum in each group
    # as the group weight to select expert groups and then select topk experts within the selected groups
    # the #experts is decided by the input tensor shape and we currently only support power of 2 #experts
    # and #experts should be divisible by num_expert_group. #expert/num_expert_group <= 32 is limited for now.
    # for non-supported case, we suggest to use the biased_grouped_topk func in sglang.srt.layers.moe.topk
    # num_fused_shared_experts: if > 0, the last several experts will be
    #   replaced with shared experts. the shared experts will be divided by the
    #   routed_scaling_factor - this is intended to cancel out later when routed+shared
    #   output is scaled so that shared experts are not scaled.
    # routed_scaling_factor: if > 0, the experts will be scaled by this factor
    # apply_routed_scaling_factor_on_output: if true, output will be
    #   scaled by the routed_scaling_factor
    return torch.ops.sgl_kernel.moe_fused_gate.default(
        input_tensor,
        bias,
        num_expert_group,
        topk_group,
        topk,
        num_fused_shared_experts,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output,
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


def moe_grouped_mm_nn(activations, weights, total_rows_for_experts, n_experts):
    """
    BF16/FP16 grouped GEMM for MoE with non-transposed weights.
    activations: (total_tokens, hidden_dim)
    weights: (total_expert_rows, hidden_dim, output_dim)
    total_rows_for_experts: (n_experts + 1,) prefix sum of rows for each expert
    n_experts: number of experts
    returns: (total_tokens, output_dim)
    """
    output = torch.empty(
        (activations.size(0), weights.size(2)),
        device=activations.device,
        dtype=activations.dtype,
    )
    torch.ops.sgl_kernel.moe_grouped_mm_nn(
        output, activations, weights, total_rows_for_experts, n_experts
    )
    return output


def fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    inplace: bool = False,
    activation: str = "silu",
    use_fp8_w8a8: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = None,
    gemm1_alpha: Optional[float] = None,
    gemm1_limit: Optional[float] = None,
) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states [num_tokens, hidden_dim] (torch.Tensor): The input tensor to the MoE layer.
    - w1 [num_experts, hidden_dim, output_channel] (torch.Tensor): The first set of expert weights.
    - w2 [num_experts, output_channel, hidden_dim] (torch.Tensor): The second set of expert weights.
    - topk_weights [num_tokens, topk] (torch.Tensor): The top-k output of the experts.
    - topk_ids [num_tokens, topk] (torch.Tensor): The top-k indices of the experts.
    - b1 (Optional[torch.Tensor]): Optional bias for w1.
    - b2 (Optional[torch.Tensor]): Optional bias for w2.
    - inplace (bool): If True, perform operations in-place to save memory. Defaults to False.
    - activation (str): The activation function to use ('silu' or 'gelu'). Defaults to 'silu'.
    - use_fp8_w8a8 (bool): If True, use fp8 arithmetic to compute the inner
        products for w1 and w2. Defaults to False.
    - w1_scale (Optional[torch.Tensor]): Optional scale to be used for
        w1.
    - w2_scale (Optional[torch.Tensor]): Optional scale to be used for
        w2.
    - a1_scale (Optional[torch.Tensor]): Optional scale to be used for
        a1.
    - a2_scale (Optional[torch.Tensor]): Optional scale to be used for
        a2.
    - block_shape: (Optional[List[int]]): Optional block size for block-wise
        quantization.
    - no_combine (bool): If True, skip the combine step. Defaults to False.
    - routed_scaling_factor (Optional[float]): Optional scaling factor for routed tokens, used by Llama4 only.
    - gemm1_alpha (Optional[float]): Optional gemm1_alpha for the activation
        function.
    - gemm1_limit (Optional[float]): Optional gemm1_limit for the swiglu activation
        function.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """

    assert use_fp8_w8a8 is False, "current MoE does not support use_fp8_w8a8"
    assert w1_scale is None, "current MoE does not support w1_scale"
    assert w2_scale is None, "current MoE does not support w2_scale"
    assert a1_scale is None, "current MoE does not support a1_scale"
    assert a2_scale is None, "current MoE does not support a2_scale"
    assert block_shape is None, "current MoE does not support block_shape"

    # type check
    assert hidden_states.dtype == torch.bfloat16, "hidden_states must be bfloat16"
    assert w1.dtype == torch.bfloat16, "w1 must be bfloat16"
    assert w2.dtype == torch.bfloat16, "w2 must be bfloat16"

    # Shape check
    assert hidden_states.ndim == 2, "hidden_states must be 2D"
    assert (
        hidden_states.shape[-1] == w1.shape[-2]
    ), f"hidden_states shape[-1] {hidden_states.shape} must be equal to w1 shape[-2] {w1.shape}"
    assert (
        2 * w2.shape[1] == w1.shape[2]
    ), f"w2 shape[1] {w2.shape[1]} must be half of w1 shape[2] {w1.shape[2]}"
    assert (topk_ids.shape == topk_weights.shape) and (
        topk_ids.shape[0] == hidden_states.shape[0]
    ), f"topk_ids shape {topk_ids.shape} and topk_weights shape {topk_weights.shape} must be equal and match hidden_states shape[0] {hidden_states.shape[0]}"

    num_tokens, _ = hidden_states.shape
    E, K, _ = w1.shape
    N, OutK = w2.shape[1]

    M = num_tokens
    TopK = topk_ids.shape[1]

    cache = torch.empty(
        M * TopK * max(2 * N, OutK),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache1 = cache[:, M * TopK * 2 * N].view((M, TopK, 2 * N))
    intermediate_cache2 = torch.empty(
        (M * TopK, N // 2),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache3 = cache[:, M * TopK * OutK].view((M, TopK, OutK))

    if no_combine:
        assert not inplace
        out_hidden_states = torch.empty(
            (num_tokens, OutK),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
    elif inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)

    idxs = topk_ids.argsort()
    counts = topk_ids.to(torch.long).bincount().cpu().numpy()
    tokens_per_expert = counts.cumsum()
    num_per_tok = TopK
    token_idxs = idxs // num_per_tok
    offset = []
    input_A = torch.empty(
        (num_tokens * TopK, K), device=hidden_states.device, dtype=hidden_states.dtype
    )
    for expert_id, end_idx in enumerate(tokens_per_expert):
        start_idx = 0 if expert_id == 0 else tokens_per_expert[expert_id - 1]
        offset.append(end_idx - start_idx)
        if start_idx == end_idx:
            continue
        exp_token_idxs = token_idxs[start_idx:end_idx]
        # expert_tokens = hidden_states[exp_token_idxs]
        # grouped_input_A.append(expert_tokens)
        input_A[start_idx:end_idx, :].copy_(hidden_states[exp_token_idxs])
    offset = torch.tensor(offset, device="cpu", dtype=torch.int32)

    torch.ops.sglang.moe_grouped_mm_nn(
        intermediate_cache1,
        input_A,
        w1,
        offset,
    )

    gate, up_ = torch.split(intermediate_cache1, N, dim=1)
    act = torch.nn.SiLU()
    intermediate_cache2 = act(gate) * up_

    torch.ops.sglang.moe_grouped_mm_nn(
        intermediate_cache3,
        intermediate_cache2.contiguous(),
        w2,
        offset,
    )
    for expert_id, end_idx in enumerate(tokens_per_expert):
        start_idx = 0 if expert_id == 0 else tokens_per_expert[expert_id - 1]
        if start_idx == end_idx:
            continue

        exp_token_idxs = token_idxs[start_idx:end_idx]
        expert_out = intermediate_cache3[start_idx:end_idx]
        expert_out.mul_(topk_weights[idxs[start_idx:end_idx]])
        out_hidden_states.scatter_reduce_(
            0, exp_token_idxs.view(-1, 1).repeat(1, OutK), expert_out, reduce="sum"
        )

    return out_hidden_states
