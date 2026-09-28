from typing import Any, Dict, Optional, Tuple

import torch

from .elementwise import silu_and_mul
from .gemm import sgl_per_token_group_quant_8bit, sgl_per_token_group_quant_fp4
from .utils import is_xe2_arch, is_xe3_arch

_MOE_SCORING_FUNC_MAP = {
    "sigmoid": 0,
    "softmax": 1,
}

_MOE_BIASED_TOPK_SCORING_MAP = {
    "sigmoid": 0,
    "sqrtsoftplus": 1,
}


def _mxfp4_e8m0_to_fp32(scale: torch.Tensor) -> torch.Tensor:
    """Decode E8M0 exponent-byte MXFP4 block scales into fp32 direct
    multipliers: value = 2^(byte - 127) (OCP MX spec bias 127; byte 0xFF is
    reserved/NaN and is not special-cased here).

    moe_grouped_mm_nt_xe35_mxfp4_w4a16 (Xe3/CRI) requires pre-decoded fp32
    scales -- unlike moe_grouped_mm_nt_xe20_w4a16 (Xe2/BMG), which decodes
    E8M0 bytes on-device as part of a fused dequant trick. This helper
    bridges that gap so callers can keep passing raw E8M0 bytes/
    torch.float8_e8m0fnu regardless of target architecture.

    Uses exp2 rather than a (byte << 23) bit-cast: the bit-cast trick maps
    byte=0 (true value 2^-127, representable as an fp32 subnormal) to a hard
    zero, because an all-zero IEEE-754 exponent field means subnormal/zero,
    not "unbiased exponent -127". exp2 rounds to the correct subnormal
    instead.
    """
    byte = scale.view(torch.uint8) if scale.dtype == torch.float8_e8m0fnu else scale
    return torch.exp2(byte.to(torch.float32) - 127.0)


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


def hash_topk(
    router_logits: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    routed_scaling_factor: float = 1.0,
    scoring_func: str = "sqrtsoftplus",
) -> None:

    assert (
        scoring_func == "sqrtsoftplus"
    ), "hash_topk only supports scoring_func of 'sqrtsoftplus'"
    torch.ops.sgl_kernel.hash_topk.default(
        router_logits,
        input_ids,
        tid2eid,
        topk_weights,
        topk_ids,
        routed_scaling_factor,
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


def biased_topk(
    input_tensor,
    bias,
    output,
    indices,
    topk,
    scoring_func,
    num_fused_shared_experts=0,
    renormalize=False,
    routed_scaling_factor=1.0,
    apply_routed_scaling_factor_on_output=False,
):
    scoring_func_int = _MOE_BIASED_TOPK_SCORING_MAP.get(scoring_func.lower())
    if scoring_func_int is None:
        raise ValueError(
            f"Unknown scoring_func '{scoring_func}', must be one of {list(_MOE_BIASED_TOPK_SCORING_MAP.keys())}"
        )

    torch.ops.sgl_kernel.biased_topk.default(
        input_tensor,
        bias,
        output,
        indices,
        topk,
        scoring_func_int,
        num_fused_shared_experts,
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


def mxfp4_blockwise_scaled_grouped_mm(
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
    problem_sizes,
    expert_offsets,
    workspace,
):
    assert (
        is_xe3_arch()
    ), "mxfp4_blockwise_scaled_grouped_mm is only supported on CRI (Xe3P) devices"
    torch.ops.sgl_kernel.mxfp4_blockwise_scaled_grouped_mm.default(
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


_MOE_WS_HEADROOM = 1
_MOE_SMALL_PREPARE_MAX_ROUTES = 64
_MOE_SMALL_PREPARE_MAX_TOPK = 16
_MOE_SMALL_PREPARE_MAX_ELEMENTS = 163840
_moe_ws_cache: Dict[Tuple[str, torch.device], torch.Tensor] = {}
_moe_ws_view_cache: Dict[
    Tuple[str, torch.device], Tuple[torch.Tensor, tuple, torch.Tensor]
] = {}


def _get_moe_ws(
    name: str,
    shape: tuple,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return a tensor of `shape`/`dtype` on `device`, backed by a flat 1-D
    process-wide scratch buffer cached grow-only inside sgl-kernel-xpu.
    Reusing stable buffers across calls and MoE layers avoids the XPU caching
    allocator accumulating differently-shaped cached blocks."""
    numel = 1
    for d in shape:
        numel *= d
    key = (name, device)
    cur = _moe_ws_cache.get(key)
    if cur is None or cur.numel() < numel or cur.dtype != dtype:
        # Grow the buffer with headroom so reallocations are rare.
        # No explicit sync is needed: PyTorch's caching allocator inserts a
        # stream-ordered deallocation event when the old tensor is dropped, so
        # its memory cannot be reused until all in-flight kernels referencing
        # it have completed.
        new_numel = max(numel, int(numel * _MOE_WS_HEADROOM))
        cur = torch.empty(new_numel, dtype=dtype, device=device)
        _moe_ws_cache[key] = cur
    cached_view = _moe_ws_view_cache.get(key)
    shape = tuple(shape)
    if cached_view is None or cached_view[0] is not cur or cached_view[1] != shape:
        view = cur.narrow(0, 0, numel).view(shape)
        _moe_ws_view_cache[key] = (cur, shape, view)
        return view
    return cached_view[2]


def _should_use_small_moe_prepare(
    num_tokens: int, topk: int, hidden_dims: int, num_experts: int
) -> bool:
    routed_rows = num_tokens * topk
    return (
        1 <= topk <= _MOE_SMALL_PREPARE_MAX_TOPK
        and routed_rows <= _MOE_SMALL_PREPARE_MAX_ROUTES
        and routed_rows * hidden_dims <= _MOE_SMALL_PREPARE_MAX_ELEMENTS
    )


def _validate_fp8_weight_scale(
    scale: torch.Tensor,
    weights: torch.Tensor,
    name: str,
    allow_scalar: bool,
) -> None:
    """Validate an FP8 expert scale tensor against its physical weight shape."""
    is_mxfp8 = scale.dtype in (torch.uint8, torch.float8_e8m0fnu)
    if is_mxfp8:
        assert scale.ndim == 3, f"{name} MXFP8 scales must be 3D [E, N, K/32]"
        assert scale.shape[0] == weights.shape[0], (
            f"{name} expert dimension {scale.shape[0]} must match weights "
            f"expert dimension {weights.shape[0]}"
        )
        assert scale.shape[1] == weights.shape[1], (
            f"{name} MXFP8 scale N dimension {scale.shape[1]} must match weights "
            f"N dimension {weights.shape[1]}"
        )
        assert (
            weights.shape[2] % 32 == 0
        ), f"{name} MXFP8 scale requires K divisible by 32, got K={weights.shape[2]}"
        assert scale.shape[2] == weights.shape[2] // 32, (
            f"{name} MXFP8 scale K dimension {scale.shape[2]} must equal K/32 "
            f"({weights.shape[2] // 32})"
        )
        return

    assert (
        scale.dtype == torch.float32
    ), f"{name} must be float32 for scalar or 128x128 block scales"
    assert scale.ndim in (
        (1, 2, 3) if allow_scalar else (3,)
    ), f"{name} must be 3D block scales, 2D scalar scales, or 1D per-expert scales"
    assert scale.shape[0] == weights.shape[0], (
        f"{name} expert dimension {scale.shape[0]} must match weights "
        f"expert dimension {weights.shape[0]}"
    )
    if scale.ndim == 1:
        assert allow_scalar, f"{name} scalar scales are not supported for this FP8 path"
        return
    if scale.ndim == 2:
        assert allow_scalar, f"{name} scalar scales are not supported for this FP8 path"
        expected_columns = 2 if name == "w1_scale" else 1
        assert scale.shape[1] in (1, expected_columns), (
            f"{name} scalar scale shape must be [E, 1] or "
            f"[E, {expected_columns}], got {tuple(scale.shape)}"
        )
        return

    expected_shape = (
        weights.shape[0],
        (weights.shape[1] + 127) // 128,
        (weights.shape[2] + 127) // 128,
    )
    assert tuple(scale.shape) == expected_shape, (
        f"{name} block scales must have shape [E, ceil(N/128), ceil(K/128)] "
        f"={expected_shape}, got {tuple(scale.shape)}"
    )
    assert (
        weights.shape[2] % 128 == 0
    ), f"{name} block scales require K divisible by 128, got K={weights.shape[2]}"


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
    - use_fp8_w8a8 (bool): If True, use FP8 E4M3 expert weights from a W8A8
        checkpoint. Xe2 currently has no native FP8-A MoE kernel, so this
        path falls back to BF16 activations and the W8A16 operator. Defaults
        to False.
    - use_mxfp4_w4a16 (bool): If True, w1 and w2 are in MXFP4 packed format
        (int8 or uint8, two E2M1 nibbles per byte) with corresponding E8M0 block
        scales supplied via w1_scale and w2_scale. Scales may be represented
        as uint8 exponent bytes or torch.float8_e8m0fnu.
        On Xe2 (BMG), routes through moe_grouped_mm_nt_xe20_w4a16, which
        dequantizes B per-tile in registers and feeds W4A16 DPAS with BF16
        or FP16 activations -- no dequantized weight tensor is materialized
        on device. On Xe3 (CRI), routes through
        moe_grouped_mm_nt_xe35_mxfp4_w4a16 instead (mxfp4-only, no
        zero-point/g_idx support, group_size fixed at 32, activation
        limited to silu/swiglu_deepseek_v4, and bias unsupported -- see
        that op's AOT instantiation matrix in
        src/GroupGemmMxfp4W4A16Xe35.cmake); the E8M0 scale bytes are
        decoded to fp32 direct multipliers on the host before the call,
        since unlike the Xe2 kernel this one expects pre-decoded scales.
        use_int4_w4a16 is not supported on Xe3. Defaults to False.
    - use_int4_w4a16 (bool): If True, w1 and w2 are in INT4 packed format
        (int8 or uint8, two 4-bit values per byte) with BF16 or FP16 block scales
        (direct multiplier) matching hidden_states.dtype, supplied via
        w1_scale and w2_scale. Zero-points are optional and, if the checkpoint
        has them, must be supplied raw (unfolded) via w1_zp/w2_zp -- see below. Shares the
        moe_grouped_mm_nt_xe20_w4a16 kernel with mxfp4. Mutually exclusive
        with use_mxfp4_w4a16. Defaults to False.
    - w1_scale (Optional[torch.Tensor]): Optional scale to be used for
        w1.
    - w2_scale (Optional[torch.Tensor]): Optional scale to be used for
        w2.
    - w1_zp (Optional[torch.Tensor]): Optional explicit per-group
        zero-point for w1, same [num_experts, output_channel,
        hidden_dim // group_size] shape and BF16 or FP16 dtype as w1_scale
        (matching hidden_states.dtype), holding the raw zero-point in code units (i.e. weight dequants as
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
    - a1_scale (Optional[torch.Tensor]): Reserved for a future prequantized
        FP8 activation input. It is currently rejected because the Xe2
        fallback consumes BF16 activations.
    - a2_scale (Optional[torch.Tensor]): Reserved for a future prequantized
        FP8 activation input. It is currently rejected because the Xe2
        fallback consumes BF16 activations.
    - block_shape: (Optional[List[int]]): Weight block size metadata. Standard FP8
        block scales use [128, 128]; MXFP8 uses [1, 32] or [32]. The value is validated when supplied.
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

    assert (
        is_xe2_arch() or is_xe3_arch()
    ), "Current MoE is only supported on BMG (Xe2) or CRI (Xe3)"

    # Automatically detect FP8 expert weights from flag or tensor dtype.
    use_fp8_weight = use_fp8_w8a8 or (w1.dtype == torch.float8_e4m3fn)
    assert not (
        use_fp8_weight and is_xe3_arch()
    ), "the W8A16 grouped GEMM (moe_grouped_mm_nt_xe20_w8a16) is not yet ported to CRI (Xe3)"
    assert a1_scale is None, (
        "prequantized FP8 activation input is not supported: " "a1_scale must be None"
    )
    assert a2_scale is None, (
        "prequantized FP8 activation input is not supported: " "a2_scale must be None"
    )
    if block_shape is not None:
        assert use_fp8_weight, "block_shape is only supported for FP8 MoE paths"
        assert list(block_shape) in (
            [128, 128],
            [1, 32],
            [32],
        ), "FP8 MoE currently supports block_shape=[128, 128], [1, 32], or [32] (MXFP8)"
    if use_fp8_weight:
        assert activation in ("silu", "gelu", "relu2"), (
            "FP8 MoE supports silu, gelu, relu2, GPT-OSS SwiGLU, and "
            "DeepSeek-V4 clamped SwiGLU only"
        )
        assert (
            w1_g_idx_perm is None and w2_g_idx_perm is None
        ), "w1_g_idx_perm/w2_g_idx_perm are only supported by the INT4 W4A16 path"
        if activation == "gelu" or activation == "relu2":
            assert (
                gemm1_alpha is None and gemm1_limit is None and swiglu_limit is None
            ), f"{activation} cannot be combined with a SwiGLU alpha or clamp"
        elif gemm1_alpha is not None:
            assert gemm1_limit is not None and swiglu_limit is None, (
                "GPT-OSS SwiGLU requires gemm1_alpha and gemm1_limit, "
                "and cannot use swiglu_limit"
            )
        elif swiglu_limit is not None:
            assert (
                swiglu_limit == 10 and gemm1_limit is None and gemm1_alpha is None
            ), "FP8 DeepSeek-V4 SwiGLU currently requires swiglu_limit=10"
        else:
            assert (
                gemm1_limit is None
            ), "gemm1_limit requires gemm1_alpha for GPT-OSS SwiGLU"
    else:
        assert activation in (
            "silu",
            "gelu",
            "relu2",
        ), f"Only silu, gelu and relu2 are supported but got {activation}"

    # Unified 4-bit W4A16 MoE (mxfp4 or int4). Weights are packed int8/uint8
    # [E, N, K/2]; scales are [E, N, K/group_size] N-outer. For mxfp4 the
    # scale is an E8M0 byte (uint8 or float8_e8m0fnu); for int4 it is a direct multiplier
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
    assert not (
        use_4bit_w4a16 and use_fp8_weight
    ), "4-bit W4A16 and FP8 paths are mutually exclusive"
    assert not (
        use_int4_w4a16 and is_xe3_arch()
    ), "the int4 W4A16 grouped GEMM (moe_grouped_mm_nt_xe20_w4a16) is not ported to CRI (Xe3); only use_mxfp4_w4a16 is supported on Xe3"
    if use_4bit_w4a16:
        assert (
            w1.dtype == torch.int8 or w1.dtype == torch.uint8
        ), "4-bit W4A16 requires w1 to be int8 or uint8 (packed [E, N, K/2])"
        assert (
            w2.dtype == torch.int8 or w2.dtype == torch.uint8
        ), "4-bit W4A16 requires w2 to be int8 or uint8 (packed [E, N, K/2])"
        assert w1_scale is not None, "w1_scale must be provided for 4-bit W4A16"
        assert w2_scale is not None, "w2_scale must be provided for 4-bit W4A16"
        if use_mxfp4_w4a16:
            mxfp4_scale_dtypes = (torch.uint8, torch.float8_e8m0fnu)
            assert (
                w1_scale.dtype in mxfp4_scale_dtypes
                and w2_scale.dtype in mxfp4_scale_dtypes
            ), "mxfp4 scales must be uint8 or float8_e8m0fnu (E8M0 exponent)"
            assert (
                w1_zp is None and w2_zp is None
            ), "w1_zp/w2_zp are not supported for use_mxfp4_w4a16 (mxfp4 has no zero-point)"
        else:
            assert (
                w1_scale.dtype == hidden_states.dtype
                and w2_scale.dtype == hidden_states.dtype
            ), "int4 scales dtype must match hidden_states dtype"
            if w1_zp is not None:
                assert (
                    w1_zp.dtype == w1_scale.dtype and w1_zp.shape == w1_scale.shape
                ), "w1_zp must have the same dtype and shape as w1_scale"
            if w2_zp is not None:
                assert (
                    w2_zp.dtype == w2_scale.dtype and w2_zp.shape == w2_scale.shape
                ), "w2_zp must have the same dtype and shape as w2_scale"
    else:
        if not use_fp8_weight:
            assert w1_scale is None, "w1_scale is only supported for 4-bit W4A16 MoE"
            assert w2_scale is None, "w2_scale is only supported for 4-bit W4A16 MoE"
        assert (
            w1_zp is None and w2_zp is None
        ), "w1_zp/w2_zp are only supported for 4-bit W4A16 MoE"
    # GPTQ desc_act/g_idx support: the caller sorts each expert's weight
    # K-dim by g_idx at weight-load time, and passes the corresponding
    # per-expert channel permutation here so the activation can be reordered
    # to match before the 4-bit GEMM. Only meaningful for int4 (mxfp4 weights
    # are never g_idx-permuted).
    if w1_g_idx_perm is not None or w2_g_idx_perm is not None:
        assert (
            use_int4_w4a16
        ), "w1_g_idx_perm/w2_g_idx_perm only apply to use_int4_w4a16"
    elif use_fp8_weight:
        assert (
            w1.dtype == torch.float8_e4m3fn
        ), "FP8 weight-only MoE requires w1 to be float8_e4m3fn"
        assert (
            w2.dtype == torch.float8_e4m3fn
        ), "FP8 weight-only MoE requires w2 to be float8_e4m3fn"
        assert (
            w1_scale is not None and w2_scale is not None
        ), "w1_scale/w2_scale must be provided for FP8 weight-only MoE"
        assert (
            w1.is_contiguous() and w2.is_contiguous()
        ), "FP8 weight-only MoE requires contiguous expert weights"
        _validate_fp8_weight_scale(w1_scale, w1, "w1_scale", allow_scalar=True)
        _validate_fp8_weight_scale(w2_scale, w2, "w2_scale", allow_scalar=True)
        if w1_scale.ndim == 1:
            w1_scale = w1_scale.unsqueeze(1)
        if w2_scale.ndim == 1:
            w2_scale = w2_scale.unsqueeze(1)
        w1_scale_is_mxfp8 = w1_scale.ndim == 3 and w1_scale.dtype in (
            torch.uint8,
            torch.float8_e8m0fnu,
        )
        w2_scale_is_mxfp8 = w2_scale.ndim == 3 and w2_scale.dtype in (
            torch.uint8,
            torch.float8_e8m0fnu,
        )
        assert (
            w1_scale.ndim == w2_scale.ndim and w1_scale_is_mxfp8 == w2_scale_is_mxfp8
        ), "w1_scale and w2_scale must use the same scalar, MXFP8, or 128x128 block layout"
        if block_shape is not None:
            norm_block_shape = list(block_shape)
            if w1_scale_is_mxfp8:
                assert norm_block_shape in (
                    [1, 32],
                    [32],
                ), f"MXFP8 scales require block_shape=[1, 32] or [32], got {block_shape}"
            elif w1_scale.ndim == 3:
                assert norm_block_shape == [
                    128,
                    128,
                ], f"Standard FP8 block scales require block_shape=[128, 128], got {block_shape}"
            else:
                assert False, (
                    f"block_shape is only valid for 3D block scales, got {block_shape} "
                    f"with scalar scales (w1_scale.shape={w1_scale.shape})"
                )
        assert hidden_states.dtype == torch.bfloat16, "hidden_states must be bfloat16"
    if b1 is not None:
        assert (
            b1.dtype == torch.bfloat16 or b1.dtype == torch.float32
        ), "b1 must be bfloat16 or float32"
        if (is_xe2_arch() or is_xe3_arch()) and b1.dtype == torch.bfloat16:
            # cast b1 to float32, since bias is accumulated in float32 in the kernel
            b1 = b1.float()
    if b2 is not None:
        assert (
            b2.dtype == torch.bfloat16 or b2.dtype == torch.float32
        ), "b2 must be bfloat16 or float32"
        if (is_xe2_arch() or is_xe3_arch()) and b2.dtype == torch.bfloat16:
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
    use_mxfp4_xe3 = use_mxfp4_w4a16 and is_xe3_arch()
    w1_scale_xe35 = None
    w2_scale_xe35 = None
    if use_4bit_w4a16:
        # w1/w2 last dims are packed (H//2, I//2); recover actual dims
        K = K * 2
        N = N * 2
        # scales are [E, N, K/group_size] N-outer; recover group_size per GEMM
        # (GEMM1 contracts over K=H, GEMM2 over N=I).
        w1_group_size = K // w1_scale.shape[2]
        w2_group_size = N // w2_scale.shape[2]
        if use_mxfp4_xe3:
            assert w1_group_size == 32 and w2_group_size == 32, (
                "moe_grouped_mm_nt_xe35_mxfp4_w4a16 hardcodes MXFP4_GROUP_SIZE=32; "
                f"got w1_group_size={w1_group_size}, w2_group_size={w2_group_size}"
            )
            # moe_grouped_mm_nt_xe35_mxfp4_w4a16 expects pre-decoded fp32
            # direct-multiplier scales (unlike moe_grouped_mm_nt_xe20_w4a16,
            # which decodes E8M0 bytes on-device); decode once here.
            # TODO: Decodes E8M0 bytes on-device on Xe3, and remove this host-side decode.
            w1_scale_xe35 = _mxfp4_e8m0_to_fp32(w1_scale)
            w2_scale_xe35 = _mxfp4_e8m0_to_fp32(w2_scale)
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

    expert_offsets = _get_moe_ws(
        "expert_offsets", (E,), torch.int32, hidden_states.device
    )
    use_small_prepare = (
        _should_use_small_moe_prepare(M, TopK, hidden_dims, E) and use_fp8_weight
    )
    c_map = _get_moe_ws("c_map", (topk_ids.numel(),), torch.int32, hidden_states.device)
    input_A_shuffle = _get_moe_ws(
        "input_A_shuffle",
        (num_tokens * TopK, K),
        hidden_states.dtype,
        hidden_states.device,
    )
    if use_small_prepare:
        # TODO: Support strided topk_ids in the prepare kernels and remove this copy.
        topk_ids_small = topk_ids if topk_ids.is_contiguous() else topk_ids.contiguous()
        torch.ops.sgl_kernel.prepare_moe_input_small.default(
            hidden_states, topk_ids_small, expert_offsets, c_map, input_A_shuffle
        )
    else:
        if topk_ids.dtype == torch.long:
            topk_ids_int = _get_moe_ws(
                "topk_ids_int", topk_ids.shape, torch.int32, hidden_states.device
            )
            topk_ids_int.copy_(topk_ids)
        else:
            topk_ids_int = topk_ids
        problem_sizes1 = _get_moe_ws(
            "problem_sizes1", (E, 3), torch.int32, hidden_states.device
        )
        problem_sizes2 = _get_moe_ws(
            "problem_sizes2", (E, 3), torch.int32, hidden_states.device
        )
        a_map = _get_moe_ws(
            "a_map", (topk_ids.numel(),), torch.int32, hidden_states.device
        )
        torch.ops.sgl_kernel.prepare_moe_input.default(
            topk_ids_int,
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
        torch.ops.sgl_kernel.scatter_tokens_to_experts.default(
            hidden_states, c_map, input_A_shuffle
        )
    if w1_g_idx_perm is not None:
        # GPTQ desc_act/g_idx: reorder each expert's activation slice to
        # match the K-dim sort applied to w1 at weight-load time.
        input_A_shuffle = _apply_per_expert_channel_gather(
            input_A_shuffle, w1_g_idx_perm, expert_offsets, E
        )

    intermediate_cache3 = _get_moe_ws(
        "intermediate_cache3",
        (M * TopK, OutK),
        hidden_states.dtype,
        hidden_states.device,
    )

    if use_fp8_weight:
        if activation == "gelu":
            activation_type = 1
        elif activation == "relu2":
            activation_type = 3
        elif activation != "silu":
            raise ValueError(
                f"FP8 W8A16 Xe2 path does not support activation={activation!r}; "
                "supported activations are 'silu', 'gelu', and 'relu2'"
            )
        elif gemm1_alpha is not None:
            if gemm1_limit is None:
                raise AssertionError(
                    "gemm1_limit must be provided when gemm1_alpha is set for swiglu for GPT-OSS"
                )
            activation_type = 2
        elif swiglu_limit is not None:
            activation_type = 4
            gemm1_limit = float(swiglu_limit)
        else:
            activation_type = 0

        gemm1_output_width = N if activation_type == 3 else 2 * N
        intermediate_cache1 = _get_moe_ws(
            "intermediate_cache1",
            (M * TopK, gemm1_output_width),
            hidden_states.dtype,
            hidden_states.device,
        )
        torch.ops.sgl_kernel.moe_grouped_mm_nt_xe20_w8a16(
            intermediate_cache1,
            input_A_shuffle,
            w1,
            w1_scale,
            b1,
            expert_offsets,
            E,
        )

        if activation_type == 2:
            intermediate_cache2 = torch.ops.sgl_kernel.swiglu_gpt_oss_sigmoid_alpha(
                intermediate_cache1, gemm1_alpha, gemm1_limit
            )
        else:
            intermediate_cache2 = _get_moe_ws(
                "intermediate_cache2",
                (M * TopK, N),
                hidden_states.dtype,
                hidden_states.device,
            )
            if activation_type == 0:
                torch.ops.sgl_kernel.silu_and_mul(
                    intermediate_cache2, intermediate_cache1
                )
            elif activation_type == 4:
                torch.ops.sgl_kernel.silu_and_mul_clamp(
                    intermediate_cache2, intermediate_cache1, swiglu_limit
                )
            elif activation_type == 1:
                torch.ops.sgl_kernel.gelu_tanh_and_mul(
                    intermediate_cache2, intermediate_cache1
                )
            elif activation_type == 3:
                torch.clamp_min(intermediate_cache1, 0, out=intermediate_cache2)
                torch.square(intermediate_cache2, out=intermediate_cache2)
            else:
                raise AssertionError(
                    f"unsupported FP8 activation type: {activation_type}"
                )

        torch.ops.sgl_kernel.moe_grouped_mm_nt_xe20_w8a16(
            intermediate_cache3,
            intermediate_cache2,
            w2,
            w2_scale,
            b2,
            expert_offsets,
            E,
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
            # DeepSeek-V4 swiglu clamp. The 4-bit grouped GEMM no longer fuses
            # activation, so the clamp is applied in the unfused activation path
            # below (see activation_type == 4 handling).
            assert (
                use_4bit_w4a16
            ), "swiglu_limit requires use_mxfp4_w4a16=True or use_int4_w4a16=True"
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

    # Gated activations (silu/gelu/swiglu) split w1's output into gate+up, so
    # w1.shape[1] == 2*N; non-gated relu2 has w1.shape[1] == N. Compare against
    # the recovered (unpacked) N — w2.shape[2] is the packed I/2 under MXFP4,
    # which would mis-detect the gated case as non-gated (gate_factor=1).
    gate_factor = 2 if (2 * N == w1.shape[1]) else 1

    if use_mxfp4_xe3:
        # moe_grouped_mm_nt_xe35_mxfp4_w4a16's AOT instantiation matrix is
        # pruned to activation_type in {0 silu, 4 swiglu_deepseek_v4} and
        # WithBias=false (see src/GroupGemmMxfp4W4A16Xe35.cmake); fail fast
        # with a clear message instead of letting the op's own TORCH_CHECK
        # fire deeper in the call stack.
        assert activation_type in (0, 4), (
            "moe_grouped_mm_nt_xe35_mxfp4_w4a16 only supports activation_type "
            f"0 (silu) or 4 (swiglu_deepseek_v4); got {activation_type} "
            f"(activation={activation!r})"
        )
        assert (
            b1 is None and b2 is None
        ), "moe_grouped_mm_nt_xe35_mxfp4_w4a16 does not support bias (b1/b2 must be None)"

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
    # Plain bf16 grouped GEMM: dispatch to the Xe3 (CRI) op when running on
    # Xe3, otherwise the Xe2 (BMG) op. use_fp8_weight and
    # use_int4_w4a16 are asserted unsupported on Xe3 above; use_mxfp4_w4a16
    # on Xe3 is handled separately below via use_mxfp4_xe3, so this only
    # guards the plain bf16 path.
    moe_grouped_mm_nt = (
        torch.ops.sgl_kernel.moe_grouped_mm_nt_xe35
        if is_xe3_arch()
        else torch.ops.sgl_kernel.moe_grouped_mm_nt_xe20
    )
    if use_unfused_act:
        intermediate_cache1 = _get_moe_ws(
            "intermediate_cache1_unfused",
            (M * TopK, gate_factor * N),
            hidden_states.dtype,
            hidden_states.device,
        )
        # GEMM1: B = w1 (gate+up).
        if use_mxfp4_xe3:
            torch.ops.sgl_kernel.moe_grouped_mm_nt_xe35_mxfp4_w4a16(
                intermediate_cache1,
                input_A_shuffle,
                w1,
                w1_scale_xe35,
                b1,
                expert_offsets,
                E,
                activation_type,
                False,  # fuse_act
                float(gemm1_alpha) if gemm1_alpha is not None else 1.702,
                float(gemm1_limit) if gemm1_limit is not None else 7.0,
            )
        elif use_4bit_w4a16:
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
            moe_grouped_mm_nt(
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
        if activation_type == 2:
            intermediate_cache2 = torch.ops.sgl_kernel.swiglu_gpt_oss_sigmoid_alpha(
                intermediate_cache1, gemm1_alpha, gemm1_limit
            )
        else:
            intermediate_cache2 = _get_moe_ws(
                "intermediate_cache2",
                (M * TopK, N),
                hidden_states.dtype,
                hidden_states.device,
            )
            if activation_type == 0:
                torch.ops.sgl_kernel.silu_and_mul(
                    intermediate_cache2, intermediate_cache1
                )
            elif activation_type == 4:
                torch.ops.sgl_kernel.silu_and_mul_clamp(
                    intermediate_cache2, intermediate_cache1, swiglu_limit
                )
            elif activation_type == 1:
                torch.ops.sgl_kernel.gelu_tanh_and_mul(
                    intermediate_cache2, intermediate_cache1
                )
            elif activation_type == 3:
                torch.clamp_min(intermediate_cache1, 0, out=intermediate_cache2)
                torch.square(intermediate_cache2, out=intermediate_cache2)
        if w2_g_idx_perm is not None:
            # GPTQ desc_act/g_idx: reorder each expert's activation slice to
            # match the K-dim sort applied to w2 at weight-load time.
            intermediate_cache2 = _apply_per_expert_channel_gather(
                intermediate_cache2, w2_g_idx_perm, expert_offsets, E
            )
        # GEMM2: B = w2 (down).
        if use_mxfp4_xe3:
            torch.ops.sgl_kernel.moe_grouped_mm_nt_xe35_mxfp4_w4a16(
                intermediate_cache3,
                intermediate_cache2,
                w2,
                w2_scale_xe35,
                b2,
                expert_offsets,
                E,
                activation_type,
                False,  # fuse_act
                float(gemm1_alpha) if gemm1_alpha is not None else 1.702,
                float(gemm1_limit) if gemm1_limit is not None else 7.0,
            )
        elif use_4bit_w4a16:
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
            moe_grouped_mm_nt(
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
        intermediate_cache1 = _get_moe_ws(
            "intermediate_cache1_fused",
            (M * TopK, N),
            hidden_states.dtype,
            hidden_states.device,
        )
        # GEMM1 (fused act): B = w1 (gate+up). The 4-bit W4A16 paths always use the
        # separate GEMM1 -> activation -> GEMM2 sequence above, so this branch is
        # only for the non-4-bit grouped-GEMM path.
        moe_grouped_mm_nt(
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
        moe_grouped_mm_nt(
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


# ---------------------------------------------------------------------------
# CUTLASS-based fused MoE wrapper (sglang-compatible), merged from the
# formerly-separate cutlass_moe.py. Signature-compatible with sglang's CUDA
# cutlass_fused_experts_fp8. CUDA-only args (*_strides, *_ptrs, enable_es) are
# unused on XPU. `use_mxfp8` selects between DSV3-style FP8 (BS=128, fp32
# scales) and OCP MXFP8 (BS=32, UE8M0 uint8 scales); C++ dispatches on
# scales_a dtype.
# ---------------------------------------------------------------------------


_FP8_E4M3_MIN = -448.0
_FP8_E4M3_MAX = 448.0

_MXFP8_BLOCK_SIZE = 32


def _per_token_group_quant_fp8(x: torch.Tensor, group_size: int = 128):
    """Per-token group quant along last dim. Returns (x_q fp8_e4m3, x_s fp32)."""
    assert x.shape[-1] % group_size == 0
    out_q = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    out_s_shape = (*x.shape[:-1], x.shape[-1] // group_size)
    out_s = torch.empty(out_s_shape, device=x.device, dtype=torch.float32)
    sgl_per_token_group_quant_8bit(
        x,
        out_q,
        out_s,
        group_size,
        1e-10,
        _FP8_E4M3_MIN,
        _FP8_E4M3_MAX,
        False,
        False,
        None,
        False,
    )
    return out_q, out_s


def _per_token_group_quant_mxfp8(x: torch.Tensor, group_size: int = _MXFP8_BLOCK_SIZE):
    """MXFP8 per-token group quant. Returns (x_q float8_e4m3fn, x_s uint8 UE8M0).

    Uses v1 (enable_v2=False): v2 does not support row-major UE8M0 output.
    """
    assert x.shape[-1] % group_size == 0
    out_q = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    out_s_shape = (*x.shape[:-1], x.shape[-1] // group_size)
    out_s = torch.empty(out_s_shape, device=x.device, dtype=torch.uint8)
    sgl_per_token_group_quant_8bit(
        x,
        out_q,
        out_s,
        group_size,
        1e-10,
        _FP8_E4M3_MIN,
        _FP8_E4M3_MAX,
        True,  # scale_ue8m0
        False,
        None,
        False,  # enable_v2 (v1 supports row-major UE8M0; v2 does not)
    )
    return out_q, out_s


def cutlass_fused_experts_fp8(
    a: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    a1_strides: torch.Tensor,
    c1_strides: torch.Tensor,
    a2_strides: torch.Tensor,
    c2_strides: torch.Tensor,
    workspace: torch.Tensor,
    a_ptrs: torch.Tensor,
    b_ptrs: torch.Tensor,
    out_ptrs: torch.Tensor,
    a_scales_ptrs: torch.Tensor,
    b_scales_ptrs: torch.Tensor,
    expert_offsets: torch.Tensor,
    problem_sizes1: torch.Tensor,
    problem_sizes2: torch.Tensor,
    use_fp8_blockscale: bool = True,
    use_mxfp8: bool = False,
    output: Optional[torch.Tensor] = None,
    enable_es: Tuple[bool, bool] = (False, False),
) -> torch.Tensor:
    """Fused MoE on Intel XPU.

    Mirrors sglang's CUDA cutlass_fused_experts_fp8 signature; CUDA-only args
    (*_strides, *_ptrs, enable_es) are unused on XPU.

    Weights expected in (E, N, K) row-major: w1=(E, n*2, k), w2=(E, k, n).
    MXFP8 scales are (E, n*2, k/32) and (E, k, n/32) un-transposed row-major;
    the kernel transposes B-scales on device. Sglang's dispatcher applies
    .transpose(1, 2) before calling; auto-detected and undone below.
    """
    assert use_fp8_blockscale, "Only support fp8 blockscale on XPU"
    assert enable_es == (False, False), "enable_es is CUDA-only"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert w1_q.dtype == torch.float8_e4m3fn
    assert w2_q.dtype == torch.float8_e4m3fn
    assert w1_q.dim() == 3 and w2_q.dim() == 3, "Weights must be 3D"
    assert w1_q.shape[0] == w2_q.shape[0], "Expert count mismatch w1/w2"
    assert a.dtype in (torch.half, torch.bfloat16), "Invalid input dtype"
    if use_mxfp8:
        assert (
            w1_scale.dtype == torch.uint8 and w2_scale.dtype == torch.uint8
        ), "use_mxfp8=True requires uint8 UE8M0 weight scales"
    else:
        assert (
            w1_scale.dtype == torch.float32 and w2_scale.dtype == torch.float32
        ), "use_mxfp8=False requires fp32 weight scales"

    # Detect on w1 only — w2 is ambiguous when intermediate == k_hidden.
    # Sglang transposes both weights+scales together or not at all.
    hidden_size = a.shape[1]
    if w1_q.shape[2] == hidden_size:
        sglang_transposed = False
    elif w1_q.shape[1] == hidden_size:
        sglang_transposed = True
    else:
        raise AssertionError(
            f"w1_q shape {tuple(w1_q.shape)} incompatible with a.shape[1]={hidden_size}"
        )

    if sglang_transposed:
        w1_q = w1_q.transpose(1, 2).contiguous()
        w2_q = w2_q.transpose(1, 2).contiguous()
        w1_scale = w1_scale.transpose(1, 2).contiguous()
        w2_scale = w2_scale.transpose(1, 2).contiguous()

    assert w1_q.shape[2] == hidden_size
    assert w2_q.shape[1] == hidden_size
    assert w1_q.shape[1] == 2 * w2_q.shape[2]

    del a1_strides, c1_strides, a2_strides, c2_strides
    del a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs

    out_dtype = a.dtype
    num_experts = w1_q.size(0)
    m = a.size(0)
    k = a.size(1)
    n = w2_q.size(2)
    topk = topk_ids.size(1)
    device = a.device

    if use_mxfp8:
        # MXFP8 gating (mirrors CUDA use_mxfp8 checks).
        from sgl_kernel import is_xe3_arch

        assert is_xe3_arch(), "MXFP8 requires an Xe3P (CRI) XPU"
        assert (
            k % _MXFP8_BLOCK_SIZE == 0
        ), f"MXFP8 requires hidden size divisible by {_MXFP8_BLOCK_SIZE}, got k={k}"
        assert (
            n % _MXFP8_BLOCK_SIZE == 0
        ), f"MXFP8 requires intermediate size divisible by {_MXFP8_BLOCK_SIZE}, got n={n}"
        expected_w1_scale_shape = (
            num_experts,
            w1_q.shape[1],
            w1_q.shape[2] // _MXFP8_BLOCK_SIZE,
        )
        expected_w2_scale_shape = (
            num_experts,
            w2_q.shape[1],
            w2_q.shape[2] // _MXFP8_BLOCK_SIZE,
        )
        assert (
            w1_scale.shape == expected_w1_scale_shape
        ), f"MXFP8 w1_scale must be {expected_w1_scale_shape}, got {tuple(w1_scale.shape)}"
        assert (
            w2_scale.shape == expected_w2_scale_shape
        ), f"MXFP8 w2_scale must be {expected_w2_scale_shape}, got {tuple(w2_scale.shape)}"

    a_map = torch.empty((topk_ids.numel(),), dtype=torch.int32, device=device)
    c_map = torch.empty((topk_ids.numel(),), dtype=torch.int32, device=device)

    # sglang allocates expert_offsets as (E+1,) and slices [:-1]; XPU's
    # prepare_moe_input takes size E. Accept either.
    eo = (
        expert_offsets[:num_experts]
        if expert_offsets.numel() > num_experts
        else expert_offsets
    )

    # prepare_moe_input fills eo with per-expert M counts (not cumulative).
    prepare_moe_input(
        topk_ids,
        eo,
        problem_sizes1,
        problem_sizes2,
        a_map,
        c_map,
        num_experts,
        n,
        k,
    )

    # Flat-2D kernel wants cumulative start offsets; exclusive-scan eo.
    expert_starts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    if num_experts > 1:
        expert_starts[1:] = torch.cumsum(eo[:-1], dim=0).to(torch.int32)

    # Pass all experts (including zero-M) directly. The Xe group tile
    # scheduler skips zero-M groups at runtime via ceil_div(0, BLK_M) = 0
    # (xe_tile_scheduler_group.hpp), so the mainloop never runs for them.
    # The driver bypasses sycl-tla's spurious M==0 reject in
    # can_implement — see blockwise_moe_runner.hpp for details.

    # Scatter then quantize. scatter_tokens_to_experts uses c_map (the
    # src->dst permutation) to gather per-expert rows.
    rep_a = torch.empty((m * topk, k), dtype=a.dtype, device=device)
    scatter_tokens_to_experts(a, c_map, rep_a)
    if use_mxfp8:
        rep_a_q, rep_a1_scales = _per_token_group_quant_mxfp8(
            rep_a, group_size=_MXFP8_BLOCK_SIZE
        )
    else:
        rep_a_q, rep_a1_scales = _per_token_group_quant_fp8(rep_a, group_size=128)

    c1 = torch.zeros((m * topk, n * 2), dtype=torch.float32, device=device)

    # Empty int64 sentinels select on-device pointer-table + scale-transpose.
    empty_ptrs = torch.empty((0,), dtype=torch.int64, device=device)
    zeros_stride = torch.zeros((num_experts,), dtype=torch.int64, device=device)
    zeros_layout = torch.zeros((num_experts, 5), dtype=torch.int32, device=device)

    fp8_blockwise_scaled_grouped_mm(
        c1,
        empty_ptrs,
        empty_ptrs,
        empty_ptrs,
        empty_ptrs,
        empty_ptrs,
        rep_a_q,
        w1_q,
        rep_a1_scales,
        w1_scale,
        zeros_stride,
        zeros_stride,
        zeros_stride,
        zeros_layout,
        zeros_layout,
        problem_sizes1,
        expert_starts,
        workspace,
    )

    intermediate = torch.empty((m * topk, n), dtype=out_dtype, device=device)
    silu_and_mul(c1.to(out_dtype), intermediate)

    if use_mxfp8:
        intermediate_q, a2_scale = _per_token_group_quant_mxfp8(
            intermediate, group_size=_MXFP8_BLOCK_SIZE
        )
    else:
        intermediate_q, a2_scale = _per_token_group_quant_fp8(
            intermediate, group_size=128
        )

    c2 = torch.zeros((m * topk, k), dtype=torch.float32, device=device)
    fp8_blockwise_scaled_grouped_mm(
        c2,
        empty_ptrs,
        empty_ptrs,
        empty_ptrs,
        empty_ptrs,
        empty_ptrs,
        intermediate_q,
        w2_q,
        a2_scale,
        w2_scale,
        zeros_stride,
        zeros_stride,
        zeros_stride,
        zeros_layout,
        zeros_layout,
        problem_sizes2,
        expert_starts,
        workspace,
    )

    if output is None:
        output = torch.empty((m, k), dtype=out_dtype, device=device)
    apply_shuffle_mul_sum(c2.to(out_dtype), output, c_map, topk_weights.to(out_dtype))
    return output


_MXFP4_BLOCK_SIZE = 32


def cutlass_fused_experts_mxfp4(
    a: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused MoE on XPU using mxfp4_blockwise_scaled_grouped_mm. Slim signature
    mirrors sglang's NVFP4 cutlass_moe_fp4 (no CUDA-only placeholders).
    Weights in (E, N, K/2) uint8 packed E2M1 with UE8M0 block-32 scales:
      w1_q (E, n*2, k/2), w2_q (E, k, n/2),
      w1_scale (E, n*2, k/32), w2_scale (E, k, n/32) (un-transposed).
    """
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert w1_q.dtype == torch.uint8, "w1_q must be uint8 (packed MXFP4)"
    assert w2_q.dtype == torch.uint8, "w2_q must be uint8 (packed MXFP4)"
    assert w1_scale.dtype == torch.uint8, "w1_scale must be uint8 (UE8M0)"
    assert w2_scale.dtype == torch.uint8, "w2_scale must be uint8 (UE8M0)"
    assert w1_q.dim() == 3 and w2_q.dim() == 3, "Weights must be 3D (E, N, K/2)"
    assert w1_q.shape[0] == w2_q.shape[0], "Expert count mismatch w1/w2"
    assert a.dtype in (torch.half, torch.bfloat16), "Invalid input dtype"

    out_dtype = a.dtype
    num_experts = w1_q.size(0)
    m = a.size(0)
    k = a.size(1)
    n = w2_q.size(2) * 2  # w2_q is (E, k, n/2)
    topk = topk_ids.size(1)
    device = a.device

    assert w1_q.size(2) * 2 == k
    assert w2_q.size(1) == k
    assert w1_q.size(1) == 2 * n

    a_map = torch.empty((topk_ids.numel(),), dtype=torch.int32, device=device)
    c_map = torch.empty((topk_ids.numel(),), dtype=torch.int32, device=device)

    # prepare_moe_input writes per-expert M counts (not cumulative starts).
    expert_offsets = torch.zeros(num_experts, dtype=torch.int32, device=device)
    problem_sizes1 = torch.zeros((num_experts, 3), dtype=torch.int32, device=device)
    problem_sizes2 = torch.zeros((num_experts, 3), dtype=torch.int32, device=device)

    prepare_moe_input(
        topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        a_map,
        c_map,
        num_experts,
        n,
        k,
    )

    # Flat-2D kernel needs cumulative start offsets.
    expert_starts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    if num_experts > 1:
        expert_starts[1:] = torch.cumsum(expert_offsets[:-1], dim=0).to(torch.int32)

    # XPU scatter doesn't support uint8/fp4; scatter bf16 then quantize.
    rep_a = torch.empty((m * topk, k), dtype=a.dtype, device=device)
    scatter_tokens_to_experts(a, c_map, rep_a)
    rep_a_q, rep_a1_scales = sgl_per_token_group_quant_fp4(
        rep_a, group_size=_MXFP4_BLOCK_SIZE
    )

    c1 = torch.zeros((m * topk, n * 2), dtype=torch.float32, device=device)
    # Empty sentinels -> on-device ptr-table + scale-transpose.
    empty_ptrs = torch.empty((0,), dtype=torch.int64, device=device)
    workspace = torch.zeros((64 * 1024 * 1024,), dtype=torch.uint8, device=device)

    mxfp4_blockwise_scaled_grouped_mm(
        c1,
        empty_ptrs,
        empty_ptrs,
        empty_ptrs,
        empty_ptrs,
        empty_ptrs,
        rep_a_q,
        w1_q,
        rep_a1_scales,
        w1_scale,
        problem_sizes1,
        expert_starts,
        workspace,
    )

    intermediate = torch.empty((m * topk, n), dtype=out_dtype, device=device)
    silu_and_mul(c1.to(out_dtype), intermediate)

    intermediate_q, a2_scale = sgl_per_token_group_quant_fp4(
        intermediate, group_size=_MXFP4_BLOCK_SIZE
    )

    c2 = torch.zeros((m * topk, k), dtype=torch.float32, device=device)
    mxfp4_blockwise_scaled_grouped_mm(
        c2,
        empty_ptrs,
        empty_ptrs,
        empty_ptrs,
        empty_ptrs,
        empty_ptrs,
        intermediate_q,
        w2_q,
        a2_scale,
        w2_scale,
        problem_sizes2,
        expert_starts,
        workspace,
    )

    if output is None:
        output = torch.empty((m, k), dtype=out_dtype, device=device)
    apply_shuffle_mul_sum(c2.to(out_dtype), output, c_map, topk_weights.to(out_dtype))
    return output
