from typing import Any, Dict, Optional

import torch

from .utils import is_xe2_arch

_MOE_SCORING_FUNC_MAP = {
    "sigmoid": 0,
    "softmax": 1,
}


def moe_align_block_size(
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


def topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: float,
    renormalize: bool = False,
) -> None:
    torch.ops.sgl_kernel.topk_softmax.default(
        topk_weights, topk_ids, gating_output, renormalize
    )


def topk_sigmoid(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    correction_bias: Optional[torch.Tensor] = None,
    routed_scaling_factor: float = 1.0,
    num_fused_shared_experts: int = 0,
) -> None:
    torch.ops.sgl_kernel.topk_sigmoid.default(
        topk_weights,
        topk_ids,
        gating_output,
        renormalize,
        correction_bias,
        routed_scaling_factor,
        num_fused_shared_experts,
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


def swiglu_gpt_oss_sigmoid_alpha(x, gemm1_alpha, gemm1_limit):
    assert gemm1_limit > 0, f"gemm1_limit must be positive, got {gemm1_limit}"
    assert x.dim() == 2, f"x must be 2D [B, 2H], got {x.dim()}D"
    assert (
        x.size(1) % 2 == 0
    ), f"Last dim must be even for gate/up split, got {x.size(1)}"
    return torch.ops.sgl_kernel.swiglu_gpt_oss_sigmoid_alpha.default(
        x,
        gemm1_alpha,
        gemm1_limit,
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
    bias: Optional[torch.Tensor],
    num_expert_group,
    topk_group,
    topk,
    renormalize=True,
    scoring_func="sigmoid",
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
    # renormalize: if true, normalize selected topk weights by their sum
    scoring_func_int = _MOE_SCORING_FUNC_MAP.get(scoring_func.lower())
    if scoring_func_int is None:
        raise ValueError(
            f"Unknown scoring_func '{scoring_func}', must be one of {list(_MOE_SCORING_FUNC_MAP.keys())}"
        )
    return torch.ops.sgl_kernel.moe_fused_gate.default(
        input_tensor,
        bias,
        num_expert_group,
        topk_group,
        topk,
        num_fused_shared_experts,
        scoring_func_int,
        renormalize,
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
    routed_scaling_factor: Optional[float] = None,
):
    rsf = 1.0

    if routed_scaling_factor is not None:
        rsf = routed_scaling_factor

    torch.ops.sgl_kernel.apply_shuffle_mul_sum.default(
        input, output, permutation, rsf, factors
    )


def scatter_tokens_to_experts(input, src2dst_map, output):
    torch.ops.sgl_kernel.scatter_tokens_to_experts.default(input, src2dst_map, output)


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


def _quant_fp8_per_token(x: torch.Tensor):
    """Quantize a bf16/half [num_tokens, hidden] tensor to fp8 e4m3 with one
    fp32 scale per token (row). Thin wrapper around the existing
    sgl_per_token_quant_fp8 op, used to prepare activations for the fp8
    W8A8 MoE GEMM (moe_grouped_mm_nt_xe20_fp8_w8a8)."""
    output_q = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    output_s = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
    torch.ops.sgl_kernel.sgl_per_token_quant_fp8.default(x, output_q, output_s)
    return output_q, output_s


def _expand_fp8_block_scale_to_per_row(
    scale: torch.Tensor, n_full: int, k_full: int, group_size: int = 128
) -> torch.Tensor:
    """Expand a 2-D block-quant fp8 weight scale [E, ceil(n_full/group_size),
    ceil(k_full/group_size)] to the per-N-row granularity
    [E, n_full, ceil(k_full/group_size)] moe_grouped_mm_nt_xe20_fp8_w8a8
    expects (see its kernel's moe_mainloop.hpp: weight scale is per-N-row,
    per-group_size-K-group). This assumes group_size=128 (DeepSeek
    convention); a differently-blocked checkpoint would need this (and the
    kernel's compile-time FP8_GROUP_SIZE_K) updated together."""
    assert (
        scale.dim() == 3
    ), f"expected 3D block scale [E, N/{group_size}, K/{group_size}], got {scale.shape}"
    k_groups = (k_full + group_size - 1) // group_size
    assert scale.shape[2] == k_groups, (
        f"weight scale K-group dim {scale.shape[2]} does not match ceil(K/{group_size})={k_groups} "
        f"(K={k_full}); only group_size={group_size} block-quant is supported by the fp8 W8A8 kernel"
    )
    return scale.repeat_interleave(group_size, dim=1)[:, :n_full, :].contiguous()


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
    use_mxfp4_w4a16: bool = False,
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
    swiglu_limit: Optional[float] = None,
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
    - use_mxfp4_w4a16 (bool): If True, w1 and w2 are in MXFP4 packed format
        (int8, two E2M1 nibbles per byte) with corresponding float32 block
        scales (direct multiplier) supplied via w1_scale and w2_scale.
        Routes through moe_grouped_mm_nt_xe20_mxfp4_w4a16, which
        dequantizes B per-tile in registers and feeds BF16 × BF16 DPAS —
        no BF16 weight tensor is ever materialized on device. Activations
        stay in BF16 (W4A16). Defaults to False.
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
    - swiglu_limit (Optional[float]): Optional swiglu_limit for the swiglu activation
        function.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """

    assert not (
        use_mxfp4_w4a16 and use_fp8_w8a8
    ), "use_mxfp4_w4a16 and use_fp8_w8a8 are mutually exclusive"
    assert (
        a1_scale is None
    ), "current MoE does not support a1_scale (fp8 activation scale is computed internally)"
    assert (
        a2_scale is None
    ), "current MoE does not support a2_scale (fp8 activation scale is computed internally)"
    assert block_shape is None, "current MoE does not support block_shape"
    if use_fp8_w8a8:
        # v1: only silu is AOT-instantiated on the XPU fp8 W8A8 grouped GEMM
        # (see sgl-kernel-xpu/src/sycl/GroupGemmFp8W8A8Xe20.cpp). The mainloop
        # itself is not limited to silu; other activations are a matter of
        # adding instantiations once the target fp8 checkpoints are known.
        assert activation == "silu", (
            "use_fp8_w8a8=True currently only supports activation='silu'; see "
            "GroupGemmFp8W8A8Xe20.cpp/.cmake to add more activation instantiations"
        )
        assert gemm1_alpha is None and swiglu_limit is None, (
            "use_fp8_w8a8=True does not support gpt-oss/deepseek-v4 style swiglu "
            "clamping yet (only plain silu is AOT-instantiated for fp8)"
        )
    else:
        assert activation in (
            "silu",
            "gelu",
            "relu2",
        ), f"Only silu, gelu and relu2 are supported but got {activation}"

    # For MXFP4 W4A16: validate packed int8 inputs and float32 scales.
    # Scales must be None on all non-mxfp4/non-fp8 code paths.
    if use_mxfp4_w4a16:
        assert (
            w1.dtype == torch.int8
        ), "use_mxfp4_w4a16=True requires w1 to be int8 (packed MXFP4)"
        assert (
            w2.dtype == torch.int8
        ), "use_mxfp4_w4a16=True requires w2 to be int8 (packed MXFP4)"
        assert (
            w1_scale is not None
        ), "w1_scale (float32) must be provided when use_mxfp4_w4a16=True"
        assert (
            w2_scale is not None
        ), "w2_scale (float32) must be provided when use_mxfp4_w4a16=True"
        assert w1_scale.dtype == torch.float32, "w1_scale must be float32"
        assert w2_scale.dtype == torch.float32, "w2_scale must be float32"
    elif use_fp8_w8a8:
        assert (
            w1.dtype == torch.float8_e4m3fn
        ), "use_fp8_w8a8=True requires w1 to be float8_e4m3fn"
        assert (
            w2.dtype == torch.float8_e4m3fn
        ), "use_fp8_w8a8=True requires w2 to be float8_e4m3fn"
        assert (
            w1_scale is not None and w2_scale is not None
        ), "w1_scale/w2_scale (float32 block scale) must be provided when use_fp8_w8a8=True"
        assert w1_scale.dtype == torch.float32, "w1_scale must be float32"
        assert w2_scale.dtype == torch.float32, "w2_scale must be float32"
        # Only 2-D block-quant (e.g. DeepSeek-style 128x128) scale tensors are
        # supported in this first version; per-tensor (single scalar) weight
        # scale is intentionally not wired up yet (see
        # xpu_fp8_moe_minimal_plan.md - "do not try to unify every
        # quantization mode at first").
        assert (
            w1_scale.dim() == 3 and w2_scale.dim() == 3
        ), "use_fp8_w8a8=True currently only supports 2-D block-quant weight scales [E, N/128, K/128]"
    else:
        assert (
            w1_scale is None
        ), "w1_scale is only supported when use_mxfp4_w4a16/use_fp8_w8a8=True"
        assert (
            w2_scale is None
        ), "w2_scale is only supported when use_mxfp4_w4a16/use_fp8_w8a8=True"

    # type check
    assert hidden_states.dtype == torch.bfloat16, "hidden_states must be bfloat16"
    if not use_mxfp4_w4a16 and not use_fp8_w8a8:
        assert w1.dtype == torch.bfloat16, "w1 must be bfloat16"
        assert w2.dtype == torch.bfloat16, "w2 must be bfloat16"
    if b1 is not None:
        assert (
            b1.dtype == torch.bfloat16 or b1.dtype == torch.float32
        ), "b1 must be bfloat16 or float32"
        if is_xe2_arch() and b1.dtype == torch.bfloat16:
            # cast b1 to float32, since bias is accumulated in float32 in the kernel
            b1 = b1.float()
    if b2 is not None:
        assert (
            b2.dtype == torch.bfloat16 or b2.dtype == torch.float32
        ), "b2 must be bfloat16 or float32"
        if is_xe2_arch() and b2.dtype == torch.bfloat16:
            # cast b2 to float32, since bias is accumulated in float32 in the kernel
            b2 = b2.float()
    # Shape check
    # For packed MXFP4 the last dim of w1/w2 is halved (2 FP4 values per byte),
    # so compute the actual (unpacked) inner dimensions for validation.
    _w1_inner = w1.shape[-1] * 2 if use_mxfp4_w4a16 else w1.shape[-1]
    _w2_inner = w2.shape[-1] * 2 if use_mxfp4_w4a16 else w2.shape[-1]
    assert hidden_states.ndim == 2, "hidden_states must be 2D"
    assert (
        hidden_states.shape[-1] == _w1_inner
    ), f"hidden_states shape[-1] {hidden_states.shape} must equal w1 inner dim {_w1_inner} (w1.shape={w1.shape})"
    assert (2 * _w2_inner == w1.shape[1]) or (
        (_w2_inner == w1.shape[1]) and (activation == "relu2")
    ), f"w2 inner dim {_w2_inner} must be half of w1 shape[1] {w1.shape[1]} except non-gate"
    assert (topk_ids.shape == topk_weights.shape) and (
        topk_ids.shape[0] == hidden_states.shape[0]
    ), f"topk_ids shape {topk_ids.shape} and topk_weights shape {topk_weights.shape} must be equal and match hidden_states shape[0] {hidden_states.shape[0]}"

    num_tokens, hidden_dims = hidden_states.shape

    E, _, K = w1.shape
    E, OutK, N = w2.shape
    if use_mxfp4_w4a16:
        # w1/w2 last dims are packed (H//2, I//2); recover actual dims
        K = K * 2
        N = N * 2
    if b1 is not None:
        assert b1.shape == w1.shape[:2], "b1 shape must match w1 shape[:2]"
    if b2 is not None:
        assert b2.shape == w2.shape[:2], "b2 shape must match w2 shape[:2]"

    M = num_tokens
    TopK = topk_ids.shape[1]

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

    topk_ids = topk_ids.int() if topk_ids.dtype == torch.long else topk_ids
    expert_offsets = torch.empty((E), dtype=torch.int32, device=hidden_states.device)
    problem_sizes1 = torch.empty((E, 3), dtype=torch.int32, device=hidden_states.device)
    problem_sizes2 = torch.empty((E, 3), dtype=torch.int32, device=hidden_states.device)
    a_map = torch.empty(
        (topk_ids.numel()), dtype=torch.int32, device=hidden_states.device
    )
    c_map = torch.empty(
        (topk_ids.numel()), dtype=torch.int32, device=hidden_states.device
    )
    torch.ops.sgl_kernel.prepare_moe_input.default(
        topk_ids,
        expert_offsets,
        None,
        problem_sizes1,
        problem_sizes2,
        a_map,
        c_map,
        E,
        hidden_dims,
        TopK,
    )
    input_A_shuffle = torch.empty(
        (num_tokens * TopK, K), device=hidden_states.device, dtype=hidden_states.dtype
    )
    # Use scatter_tokens_to_experts (IPEX MoEScatter style):
    # 1 WG per source token, reads sequentially, scatters to TopK destinations,
    # with coalesced reads and data reuse.
    torch.ops.sgl_kernel.scatter_tokens_to_experts.default(
        hidden_states, c_map, input_A_shuffle
    )

    intermediate_cache3 = torch.empty(
        (M * TopK, OutK), device=hidden_states.device, dtype=hidden_states.dtype
    )

    if use_fp8_w8a8:
        # FP8 (E4M3) W8A8 path. Activations are quantized per-token (one
        # fp32 scale per row) right before each expert GEMM, matching the
        # CUTLASS/FlashInfer fp8 MoE contract (see xpu_fp8_moe_minimal_plan.md).
        # Weight scales are expected as 2-D block-quant [E, N/128, K/128]
        # (DeepSeek-style) and are expanded to per-N-row here, since the
        # kernel's block-scale granularity is (per-N-row, per-128-K-group) -
        # see sgl-kernel-xpu/src/sycl/kernels/moe/xe20/fp8_w8a8/moe_mainloop.hpp.
        #
        # v1 always fuses GEMM1's gate/up activation in-kernel - there is no
        # "unfused GEMM1 for huge-weight/small-M" heuristic yet (see
        # GroupGemmFp8W8A8Xe20.cpp header comment), unlike the bf16/MXFP4
        # paths below.
        w1_scale_expanded = _expand_fp8_block_scale_to_per_row(w1_scale, w1.shape[1], K)
        w2_scale_expanded = _expand_fp8_block_scale_to_per_row(w2_scale, OutK, N)

        input_A_shuffle_fp8, a1_scale = _quant_fp8_per_token(input_A_shuffle)

        intermediate_cache1 = torch.empty(
            (M * TopK, N), device=hidden_states.device, dtype=hidden_states.dtype
        )
        torch.ops.sgl_kernel.moe_grouped_mm_nt_xe20_fp8_w8a8(
            intermediate_cache1,
            input_A_shuffle_fp8,
            a1_scale,
            w1,
            w1_scale_expanded,
            b1,
            expert_offsets,
            E,
            0,  # activation_type: silu only in v1, see assert above
            True,  # fuse_act
            1.702,
            7.0,
        )

        intermediate_cache1_fp8, a2_scale = _quant_fp8_per_token(intermediate_cache1)
        torch.ops.sgl_kernel.moe_grouped_mm_nt_xe20_fp8_w8a8(
            intermediate_cache3,
            intermediate_cache1_fp8,
            a2_scale,
            w2,
            w2_scale_expanded,
            b2,
            expert_offsets,
            E,
            0,
            False,  # fuse_act
            1.702,
            7.0,
        )

        rsf = 1.0
        if routed_scaling_factor is not None:
            rsf = routed_scaling_factor
        torch.ops.sgl_kernel.apply_shuffle_mul_sum.default(
            intermediate_cache3, out_hidden_states, c_map, rsf, topk_weights
        )
        return out_hidden_states

    # 0=silu, 1=gelu, 2=swiglu (silu with alpha/limit clamping for gpt-oss),
    # 3=relu2, 4=swiglu_deepseek_v4 (clamp gate/up then plain silu * up).
    if activation == "silu":
        activation_type = 0
        if gemm1_alpha is not None:
            assert (
                gemm1_limit is not None
            ), "gemm1_limit must be provided when gemm1_alpha is set for swiglu for GPT-OSS"
            activation_type = 2
            activation = "swiglu_gpt_oss"
        elif swiglu_limit is not None:
            assert swiglu_limit == 10
            # The fused swiglu_deepseek_v4 epilogue (activation_type=4) is only
            # AOT-instantiated for the MXFP4 W4A16 grouped GEMM. The bf16
            # grouped GEMM caps activation_type at RELU2 (3); routing 4 there
            # would trip its dispatcher range check.
            assert (
                use_mxfp4_w4a16
            ), "swiglu_limit (swiglu_deepseek_v4) is only supported with use_mxfp4_w4a16=True"
            activation_type = 4
            activation = "swiglu_deepseek_v4"
            # The kernel ABI carries the clamp threshold in gemm1_limit (the
            # only limit slot). The fused epilogue
            # apply_fused_activation<SWIGLU_DEEPSEEK_V4> reads it from there;
            # gemm1_alpha is unused for this activation.
            gemm1_limit = float(swiglu_limit)
    elif activation == "gelu":
        activation_type = 1
    elif activation == "relu2":
        activation_type = 3
    else:
        raise ValueError(f"Unsupported activation {activation}")

    assert is_xe2_arch(), f"Current MoE is only supported on BMG"

    # Gated activations (silu/gelu/swiglu) split w1's output into gate+up, so
    # w1.shape[1] == 2*N; non-gated relu2 has w1.shape[1] == N. Compare against
    # the recovered (unpacked) N — w2.shape[2] is the packed I/2 under MXFP4,
    # which would mis-detect the gated case as non-gated (gate_factor=1).
    gate_factor = 2 if (2 * N == w1.shape[1]) else 1

    # Heuristic for choosing fused vs unfused activation. The K*N threshold
    # mirrors the small-weight cutoff in the C++ grouped-GEMM dispatchers
    # (MOE_GROUPED_GEMM_SMALL_WEIGHT_THRESHOLD in src/sycl/Utils.h). Keep
    # the two in sync if either side is re-tuned.
    _MOE_GROUPED_GEMM_SMALL_WEIGHT_THRESHOLD = 4096 * 4096
    avg_m = (M * TopK) // E
    big_weight = K * N > _MOE_GROUPED_GEMM_SMALL_WEIGHT_THRESHOLD
    use_unfused_act = avg_m <= 128 and big_weight
    if use_unfused_act:
        intermediate_cache1 = torch.empty(
            (M * TopK, gate_factor * N),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        intermediate_cache2 = torch.empty(
            (M * TopK, N), device=hidden_states.device, dtype=hidden_states.dtype
        )
        # GEMM1: B = w1 (gate+up).
        if use_mxfp4_w4a16:
            torch.ops.sgl_kernel.moe_grouped_mm_nt_xe20_mxfp4_w4a16(
                intermediate_cache1,
                input_A_shuffle,
                w1,
                w1_scale,
                b1,
                expert_offsets,
                E,
                activation_type,
                False,  # fuse_act
                float(gemm1_alpha) if gemm1_alpha is not None else 1.702,
                float(gemm1_limit) if gemm1_limit is not None else 7.0,
            )
        else:
            torch.ops.sgl_kernel.moe_grouped_mm_nt_xe20(
                intermediate_cache1,
                input_A_shuffle,
                w1,
                b1,
                expert_offsets,
                E,
                activation_type,
                fuse_act=False,
                gemm1_alpha=float(gemm1_alpha) if gemm1_alpha is not None else 1.702,
                gemm1_limit=float(gemm1_limit) if gemm1_limit is not None else 7.0,
            )
        if activation_type in (0, 4):
            if activation_type == 4:
                # DeepSeek-V4 swiglu clamp, applied here on the raw gate+up
                # projection because the unfused GEMM1 wrote it out without
                # activation. The fused path does the same clamp in-kernel
                # (apply_fused_activation<SWIGLU_DEEPSEEK_V4>).
                half = w1.shape[1] // 2
                intermediate_cache1[:, :half].clamp_(max=swiglu_limit)
                intermediate_cache1[:, half:].clamp_(
                    min=-swiglu_limit, max=swiglu_limit
                )
            torch.ops.sgl_kernel.silu_and_mul(intermediate_cache2, intermediate_cache1)
        elif activation_type == 1:
            torch.ops.sgl_kernel.gelu_tanh_and_mul(
                intermediate_cache2, intermediate_cache1
            )
        elif activation_type == 2:
            intermediate_cache2 = torch.ops.sgl_kernel.swiglu_gpt_oss_sigmoid_alpha(
                intermediate_cache1, gemm1_alpha, gemm1_limit
            )
        elif activation_type == 3:
            intermediate_cache2 = torch.square(torch.relu(intermediate_cache1))
        # GEMM2: B = w2 (down).
        if use_mxfp4_w4a16:
            torch.ops.sgl_kernel.moe_grouped_mm_nt_xe20_mxfp4_w4a16(
                intermediate_cache3,
                intermediate_cache2,
                w2,
                w2_scale,
                b2,
                expert_offsets,
                E,
                activation_type,
                False,  # fuse_act
                float(gemm1_alpha) if gemm1_alpha is not None else 1.702,
                float(gemm1_limit) if gemm1_limit is not None else 7.0,
            )
        else:
            torch.ops.sgl_kernel.moe_grouped_mm_nt_xe20(
                intermediate_cache3,
                intermediate_cache2,
                w2,
                b2,
                expert_offsets,
                E,
                activation_type,
                fuse_act=False,
                gemm1_alpha=float(gemm1_alpha) if gemm1_alpha is not None else 1.702,
                gemm1_limit=float(gemm1_limit) if gemm1_limit is not None else 7.0,
            )
    else:
        intermediate_cache1 = torch.empty(
            (M * TopK, N), device=hidden_states.device, dtype=hidden_states.dtype
        )
        # GEMM1 (fused act): B = w1 (gate+up).
        if use_mxfp4_w4a16:
            torch.ops.sgl_kernel.moe_grouped_mm_nt_xe20_mxfp4_w4a16(
                intermediate_cache1,
                input_A_shuffle,
                w1,
                w1_scale,
                b1,
                expert_offsets,
                E,
                activation_type,
                True,  # fuse_act
                float(gemm1_alpha) if gemm1_alpha is not None else 1.702,
                float(gemm1_limit) if gemm1_limit is not None else 7.0,
            )
        else:
            torch.ops.sgl_kernel.moe_grouped_mm_nt_xe20(
                intermediate_cache1,
                input_A_shuffle,
                w1,
                b1,
                expert_offsets,
                E,
                activation_type,
                fuse_act=True,
                gemm1_alpha=float(gemm1_alpha) if gemm1_alpha is not None else 1.702,
                gemm1_limit=float(gemm1_limit) if gemm1_limit is not None else 7.0,
            )
        # GEMM2: B = w2 (down). Always fuse_act=False on the second GEMM.
        if use_mxfp4_w4a16:
            torch.ops.sgl_kernel.moe_grouped_mm_nt_xe20_mxfp4_w4a16(
                intermediate_cache3,
                intermediate_cache1,
                w2,
                w2_scale,
                b2,
                expert_offsets,
                E,
                activation_type,
                False,  # fuse_act
                float(gemm1_alpha) if gemm1_alpha is not None else 1.702,
                float(gemm1_limit) if gemm1_limit is not None else 7.0,
            )
        else:
            torch.ops.sgl_kernel.moe_grouped_mm_nt_xe20(
                intermediate_cache3,
                intermediate_cache1,
                w2,
                b2,
                expert_offsets,
                E,
                activation_type,
                fuse_act=False,
                gemm1_alpha=float(gemm1_alpha) if gemm1_alpha is not None else 1.702,
                gemm1_limit=float(gemm1_limit) if gemm1_limit is not None else 7.0,
            )

    rsf = 1.0

    if routed_scaling_factor is not None:
        rsf = routed_scaling_factor

    torch.ops.sgl_kernel.apply_shuffle_mul_sum.default(
        intermediate_cache3, out_hidden_states, c_map, rsf, topk_weights
    )

    return out_hidden_states
