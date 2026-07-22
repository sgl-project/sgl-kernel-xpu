from typing import Any, Dict, Optional

import torch

from .utils import is_xe2_arch

_MOE_SCORING_FUNC_MAP = {
    "sigmoid": 0,
    "softmax": 1,
}


def _apply_per_expert_channel_gather(
    x: torch.Tensor,
    perm: torch.Tensor,
    rows_per_expert: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Gather channels of `x` per contiguous expert row-block according to a
    per-expert permutation.

    Used to implement GPTQ desc_act/g_idx support: weights are sorted by
    g_idx at weight-load time (so their K-dim is contiguous per quantization
    group), and the corresponding activation slice for each expert must be
    re-ordered the same way before the 4-bit GEMM. `x` rows are already
    grouped contiguously by expert (as produced by
    scatter_tokens_to_experts/prepare_moe_input), and `rows_per_expert` gives
    the row count for each expert (not cumulative offsets).

    - x: [total_rows, C] activation slice for one GEMM.
    - perm: [num_experts, C] int64/int32 per-expert channel permutation;
      out[:, c'] = x[:, perm[e, c']] for expert e's row block.
    - rows_per_expert: [num_experts] int32 row count per expert.
    """
    expert_ids = torch.repeat_interleave(
        torch.arange(num_experts, device=x.device),
        rows_per_expert.to(torch.int64),
        output_size=x.size(0),
    )
    return x.gather(1, perm.index_select(0, expert_ids))


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
    use_int4_w4a16: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    w1_g_idx_perm: Optional[torch.Tensor] = None,
    w2_g_idx_perm: Optional[torch.Tensor] = None,
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
        (int8, two E2M1 nibbles per byte) with corresponding uint8 E8M0
        block scales supplied via w1_scale and w2_scale.
        Routes through moe_grouped_mm_nt_xe20_w4a16, which dequantizes B
        per-tile in registers and feeds BF16 × BF16 DPAS — no BF16 weight
        tensor is ever materialized on device. Activations stay in BF16
        (W4A16). Defaults to False.
    - use_int4_w4a16 (bool): If True, w1 and w2 are in INT4 packed format
        (int8, two 4-bit values per byte) with bfloat16 block scales
        (direct multiplier) supplied via w1_scale and w2_scale. Zero-points
        are optional and, if the checkpoint has them, must be supplied raw
        (unfolded) via w1_zp/w2_zp -- see below. Shares the
        moe_grouped_mm_nt_xe20_w4a16 kernel with mxfp4. Mutually exclusive
        with use_mxfp4_w4a16. Defaults to False.
    - w1_scale (Optional[torch.Tensor]): Optional scale to be used for
        w1.
    - w2_scale (Optional[torch.Tensor]): Optional scale to be used for
        w2.
    - w1_zp (Optional[torch.Tensor]): Optional explicit per-group
        zero-point for w1, same [num_experts, output_channel,
        hidden_dim // group_size] shape and bfloat16 dtype as w1_scale,
        holding the raw zero-point in code units (i.e. weight dequants as
        `(code - zp) * scale`, not pre-folded into a signed 4-bit code).
        Only valid with use_int4_w4a16=True; None means the checkpoint has
        no zero-point (symmetric quantization).
    - w2_zp (Optional[torch.Tensor]): Optional explicit per-group
        zero-point for w2, analogous to w1_zp. Only valid with
        use_int4_w4a16=True.
    - w1_g_idx_perm (Optional[torch.Tensor]): [num_experts, hidden_dim]
        int64/int32 per-expert channel permutation for GPTQ desc_act/g_idx
        support. If provided, each expert's activation slice is gathered
        along the hidden dim to match the K-dim sort applied to w1 at
        weight-load time. Only valid with use_int4_w4a16=True.
    - w2_g_idx_perm (Optional[torch.Tensor]): [num_experts, output_channel]
        int64/int32 per-expert channel permutation for GPTQ desc_act/g_idx
        support, applied to the intermediate activation before GEMM2 to
        match the K-dim sort applied to w2 at weight-load time. Only valid
        with use_int4_w4a16=True.
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

    assert use_fp8_w8a8 is False, "current MoE does not support use_fp8_w8a8"
    assert a1_scale is None, "current MoE does not support a1_scale"
    assert a2_scale is None, "current MoE does not support a2_scale"
    assert block_shape is None, "current MoE does not support block_shape"
    assert activation in (
        "silu",
        "gelu",
        "relu2",
    ), f"Only silu, gelu and relu2 are supported but got {activation}"

    # Unified 4-bit W4A16 MoE (mxfp4 or int4). Weights are packed int8
    # [E, N, K/2]; scales are [E, N, K/group_size] N-outer. For mxfp4 the
    # scale is a uint8 E8M0 exponent; for int4 it is a direct multiplier
    # with the same dtype as hidden_states. int4 may optionally carry an explicit per-group
    # zero-point (w1_zp/w2_zp, same shape/dtype as the scale) applied as
    # `(code - zp) * scale` in-kernel -- this is NOT folded into the packed
    # weights, avoiding the signed 4-bit overflow that folding causes for
    # non-symmetric real-world AWQ checkpoints. Scales must be None on all
    # non-4bit code paths.
    use_4bit_w4a16 = use_mxfp4_w4a16 or use_int4_w4a16
    assert not (
        use_mxfp4_w4a16 and use_int4_w4a16
    ), "use_mxfp4_w4a16 and use_int4_w4a16 are mutually exclusive"
    if use_4bit_w4a16:
        assert (
            w1.dtype == torch.int8
        ), "4-bit W4A16 requires w1 to be int8 (packed [E, N, K/2])"
        assert (
            w2.dtype == torch.int8
        ), "4-bit W4A16 requires w2 to be int8 (packed [E, N, K/2])"
        assert w1_scale is not None, "w1_scale must be provided for 4-bit W4A16"
        assert w2_scale is not None, "w2_scale must be provided for 4-bit W4A16"
        if use_mxfp4_w4a16:
            assert (
                w1_scale.dtype == torch.uint8 and w2_scale.dtype == torch.uint8
            ), "mxfp4 scales must be uint8 (E8M0 exponent)"
            assert (
                w1_zp is None and w2_zp is None
            ), "w1_zp/w2_zp are not supported for use_mxfp4_w4a16 (mxfp4 has no zero-point)"
        else:
            assert (
                w1_scale.dtype == hidden_states.dtype
                and w2_scale.dtype == hidden_states.dtype
            ), "int4 scales dtype must match hidden_states dtype"
            if w1_zp is not None:
                assert w1_zp.dtype == w1_scale.dtype and w1_zp.shape == w1_scale.shape, (
                    "w1_zp must have the same dtype and shape as w1_scale"
                )
            if w2_zp is not None:
                assert w2_zp.dtype == w2_scale.dtype and w2_zp.shape == w2_scale.shape, (
                    "w2_zp must have the same dtype and shape as w2_scale"
                )
    else:
        assert w1_scale is None, "w1_scale is only supported for 4-bit W4A16 MoE"
        assert w2_scale is None, "w2_scale is only supported for 4-bit W4A16 MoE"
        assert w1_zp is None and w2_zp is None, "w1_zp/w2_zp are only supported for 4-bit W4A16 MoE"
    # GPTQ desc_act/g_idx support: the caller sorts each expert's weight
    # K-dim by g_idx at weight-load time, and passes the corresponding
    # per-expert channel permutation here so the activation can be reordered
    # to match before the 4-bit GEMM. Only meaningful for int4 (mxfp4 weights
    # are never g_idx-permuted).
    if w1_g_idx_perm is not None or w2_g_idx_perm is not None:
        assert use_int4_w4a16, "w1_g_idx_perm/w2_g_idx_perm only apply to use_int4_w4a16"
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
    # For packed 4-bit weights the last dim of w1/w2 is halved (2 values per
    # byte), so compute the actual (unpacked) inner dimensions for validation.
    _w1_inner = w1.shape[-1] * 2 if use_4bit_w4a16 else w1.shape[-1]
    _w2_inner = w2.shape[-1] * 2 if use_4bit_w4a16 else w2.shape[-1]
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
    w1_group_size = 0
    w2_group_size = 0
    if use_4bit_w4a16:
        # w1/w2 last dims are packed (H//2, I//2); recover actual dims
        K = K * 2
        N = N * 2
        # scales are [E, N, K/group_size] N-outer; recover group_size per GEMM
        # (GEMM1 contracts over K=H, GEMM2 over N=I).
        w1_group_size = K // w1_scale.shape[2]
        w2_group_size = N // w2_scale.shape[2]
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
    if w1_g_idx_perm is not None:
        # GPTQ desc_act/g_idx: reorder each expert's activation slice to
        # match the K-dim sort applied to w1 at weight-load time.
        input_A_shuffle = _apply_per_expert_channel_gather(
            input_A_shuffle, w1_g_idx_perm, expert_offsets, E
        )

    intermediate_cache3 = torch.empty(
        (M * TopK, OutK), device=hidden_states.device, dtype=hidden_states.dtype
    )

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
            # DeepSeek-V4 swiglu clamp. The 4-bit grouped GEMM no longer fuses
            # activation, so the clamp is applied in the unfused activation path
            # below (see activation_type == 4 handling).
            assert (
                use_4bit_w4a16
            ), (
                "swiglu_limit requires use_mxfp4_w4a16=True or use_int4_w4a16=True"
            )
            activation_type = 4
            activation = "swiglu_deepseek_v4"
            # Carry the clamp threshold in gemm1_limit (the only limit slot).
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
    # The 4-bit W4A16 grouped GEMM uses a two-GEMM path. Keep GEMM1 independent
    # and apply the gated activation with its dedicated elementwise kernel.
    # This preserves GEMM N-dimension parallelism.
    use_unfused_act = use_4bit_w4a16 or (avg_m <= 128 and big_weight)
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
        if use_4bit_w4a16:
            torch.ops.sgl_kernel.moe_grouped_mm_nt_xe20_w4a16(
                intermediate_cache1,
                input_A_shuffle,
                w1,
                w1_scale,
                w1_zp,
                b1,
                expert_offsets,
                E,
                use_int4_w4a16,
                w1_group_size,
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
        if activation_type == 0:
            torch.ops.sgl_kernel.silu_and_mul(intermediate_cache2, intermediate_cache1)
        elif activation_type == 4:
            torch.ops.sgl_kernel.silu_and_mul_clamp(
                intermediate_cache2, intermediate_cache1, swiglu_limit
            )
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
        if w2_g_idx_perm is not None:
            # GPTQ desc_act/g_idx: reorder each expert's activation slice to
            # match the K-dim sort applied to w2 at weight-load time.
            intermediate_cache2 = _apply_per_expert_channel_gather(
                intermediate_cache2, w2_g_idx_perm, expert_offsets, E
            )
        # GEMM2: B = w2 (down).
        if use_4bit_w4a16:
            torch.ops.sgl_kernel.moe_grouped_mm_nt_xe20_w4a16(
                intermediate_cache3,
                intermediate_cache2,
                w2,
                w2_scale,
                w2_zp,
                b2,
                expert_offsets,
                E,
                use_int4_w4a16,
                w2_group_size,
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
        # GEMM1 (fused act): B = w1 (gate+up). The 4-bit W4A16 paths always use the
        # separate GEMM1 -> activation -> GEMM2 sequence above, so this branch is
        # only for the non-4-bit grouped-GEMM path.
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
