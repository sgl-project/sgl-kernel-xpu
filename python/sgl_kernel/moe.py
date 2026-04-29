from typing import Any, Dict, Optional

import torch

from .utils import is_xe2_arch

# E2M1 lookup table: nibble value 0x0–0xF → float.
# Encodes sign directly: 0b0xxx = positive, 0b1xxx = negative.
# Packing convention (matches sglang upstream / DeepSeek / sgl-kernel-xpu):
# low nibble = first element, high nibble = second element.
_MXFP4_E2M1_LUT_CPU = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,  # 0b0xxx (positive)
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,  # 0b1xxx (negative)
    ],
    dtype=torch.bfloat16,
)
# Cache of device-resident LUTs to avoid re-copying on every call.
_MXFP4_E2M1_LUT_CACHE: Dict[torch.device, torch.Tensor] = {}


def _get_e2m1_lut(device: torch.device) -> torch.Tensor:
    """Return a device-resident BF16 E2M1 lookup table, cached after first use."""
    if device not in _MXFP4_E2M1_LUT_CACHE:
        _MXFP4_E2M1_LUT_CACHE[device] = _MXFP4_E2M1_LUT_CPU.to(device=device)
    return _MXFP4_E2M1_LUT_CACHE[device]


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
) -> None:
    torch.ops.sgl_kernel.topk_sigmoid.default(
        topk_weights,
        topk_ids,
        gating_output,
        renormalize,
        correction_bias,
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


def dequantize_mxfp4_weights(
    packed: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    block_size: int = 32,
) -> torch.Tensor:
    """Dequantize MXFP4 (E2M1) packed expert weight tensor to BF16 (or other dtype).

    Packing convention (matches sglang upstream / DeepSeek / sgl-kernel-xpu):
        byte = (first_elem & 0xF) | (second_elem << 4)
    i.e. low nibble = first element, high nibble = second element.

    Parameters:
    - packed:     [E, rows, packed_cols] uint8 – two FP4 nibbles per byte,
                  where packed_cols = cols // 2.
    - scales:     [E, rows, num_blocks]  uint8 in UE8M0 format
                  (stored_byte = biased_exp + 127), where
                  num_blocks = cols // block_size.
    - dtype:      Output floating-point dtype (default: torch.bfloat16).
    - block_size: Number of FP4 elements sharing one scale factor (default: 32).

    Returns:
    - Tensor of shape [E, rows, cols] in ``dtype`` on the same device as
      ``packed``.
    """
    E, rows, packed_cols = packed.shape
    cols = packed_cols * 2
    num_blocks = cols // block_size

    # Validate inputs
    assert (
        cols % block_size == 0
    ), f"cols ({cols}) must be divisible by block_size ({block_size})"
    assert scales.shape == (
        E,
        rows,
        num_blocks,
    ), f"scales shape {scales.shape} must be ({E}, {rows}, {num_blocks})"
    assert (
        packed.device == scales.device
    ), f"packed and scales must be on the same device, got {packed.device} and {scales.device}"

    # --- 1. Unpack and dequantize via 16-entry signed LUT ---
    lut = _get_e2m1_lut(packed.device)

    # Allocate output directly and fill even/odd positions to avoid
    # torch.stack + reshape intermediaries.
    dequantized = torch.empty(E, rows, cols, dtype=torch.bfloat16, device=packed.device)
    dequantized[:, :, 0::2] = lut[(packed & 0x0F).int()]
    dequantized[:, :, 1::2] = lut[(packed >> 4).int()]

    # --- 2. Apply per-block UE8M0 scale factors ---
    dequantized = dequantized.view(E, rows, num_blocks, block_size)
    scale_exp = scales.to(torch.int32) - 127  # [E, rows, num_blocks]
    scale_values = torch.exp2(scale_exp).unsqueeze(-1).to(torch.bfloat16)
    dequantized = dequantized * scale_values

    result = dequantized.reshape(E, rows, cols)
    if dtype != torch.bfloat16:
        result = result.to(dtype)
    return result


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
        (uint8, two E2M1 nibbles per byte) with corresponding UE8M0 block
        scales supplied via w1_scale and w2_scale.  The weights are
        dequantized to BF16 before the grouped GeMM so the rest of the
        computation is unchanged (W4A16: activations stay in BF16).
        Defaults to False.
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
    assert a1_scale is None, "current MoE does not support a1_scale"
    assert a2_scale is None, "current MoE does not support a2_scale"
    assert block_shape is None, "current MoE does not support block_shape"
    assert activation in (
        "silu",
        "gelu",
    ), f"Only silu and gelu are supported but got {activation}"

    # For MXFP4 W4A16: validate packed uint8 inputs and scales.
    # Actual dequantization is deferred to just before each GeMM so that
    # at most one dequantized weight tensor lives in GPU memory at a time.
    # Scales must be None on all non-mxfp4 code paths.
    if use_mxfp4_w4a16:
        assert (
            w1.dtype == torch.uint8
        ), "use_mxfp4_w4a16=True requires w1 to be uint8 (packed MXFP4)"
        assert (
            w2.dtype == torch.uint8
        ), "use_mxfp4_w4a16=True requires w2 to be uint8 (packed MXFP4)"
        assert (
            w1_scale is not None
        ), "w1_scale (UE8M0 uint8) must be provided when use_mxfp4_w4a16=True"
        assert (
            w2_scale is not None
        ), "w2_scale (UE8M0 uint8) must be provided when use_mxfp4_w4a16=True"
        assert w1_scale.dtype == torch.uint8, "w1_scale must be uint8 (UE8M0 format)"
        assert w2_scale.dtype == torch.uint8, "w2_scale must be uint8 (UE8M0 format)"
    else:
        assert w1_scale is None, "w1_scale is only supported when use_mxfp4_w4a16=True"
        assert w2_scale is None, "w2_scale is only supported when use_mxfp4_w4a16=True"

    # type check
    assert hidden_states.dtype == torch.bfloat16, "hidden_states must be bfloat16"
    if not use_mxfp4_w4a16:
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
    assert (
        2 * _w2_inner == w1.shape[1]
    ), f"w2 inner dim {_w2_inner} must be half of w1 shape[1] {w1.shape[1]}"
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

    # 0=silu, 1=gelu, 2=swiglu (silu with alpha/limit clamping for gpt-oss)
    if activation == "silu":
        activation_type = 0
        if gemm1_alpha is not None:
            assert (
                gemm1_limit is not None
            ), "gemm1_limit must be provided when gemm1_alpha is set for swiglu for GPT-OSS"
            activation_type = 2
            activation = "swiglu_gpt_oss"
    elif activation == "gelu":
        activation_type = 1
    else:
        raise ValueError(f"Unsupported activation {activation}")

    assert is_xe2_arch(), f"Current MoE is only supported on BMG"

    # heuristic for choosing fused or unfused act, can be tuned
    avg_m = (M * TopK) // E
    big_weight = K * N > 4096 * 4096
    use_unfused_act = avg_m <= 128 and big_weight
    if use_unfused_act:
        intermediate_cache1 = torch.empty(
            (M * TopK, 2 * N), device=hidden_states.device, dtype=hidden_states.dtype
        )
        intermediate_cache2 = torch.empty(
            (M * TopK, N), device=hidden_states.device, dtype=hidden_states.dtype
        )
        # Dequantize w1 just before GEMM1, free immediately after
        if use_mxfp4_w4a16:
            w1 = dequantize_mxfp4_weights(w1, w1_scale)
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
        # Free dequantized w1 before allocating dequantized w2 — ensures at most
        # one dequantized weight tensor occupies GPU memory at a time.
        if use_mxfp4_w4a16:
            del w1
            w2 = dequantize_mxfp4_weights(w2, w2_scale)
        if activation_type == 0:
            torch.ops.sgl_kernel.silu_and_mul(intermediate_cache2, intermediate_cache1)
        elif activation_type == 1:
            torch.ops.sgl_kernel.gelu_tanh_and_mul(
                intermediate_cache2, intermediate_cache1
            )
        elif activation_type == 2:
            intermediate_cache2 = torch.ops.sgl_kernel.swiglu_gpt_oss_sigmoid_alpha(
                intermediate_cache1, gemm1_alpha, gemm1_limit
            )
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
        if use_mxfp4_w4a16:
            del w2
    else:
        intermediate_cache1 = torch.empty(
            (M * TopK, N), device=hidden_states.device, dtype=hidden_states.dtype
        )
        # Dequantize w1 just before GEMM1, free immediately after
        if use_mxfp4_w4a16:
            w1 = dequantize_mxfp4_weights(w1, w1_scale)
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
        # Free dequantized w1 before allocating dequantized w2 — ensures at most
        # one dequantized weight tensor occupies GPU memory at a time.
        if use_mxfp4_w4a16:
            del w1
            w2 = dequantize_mxfp4_weights(w2, w2_scale)
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
        if use_mxfp4_w4a16:
            del w2

    rsf = 1.0

    if routed_scaling_factor is not None:
        rsf = routed_scaling_factor

    torch.ops.sgl_kernel.apply_shuffle_mul_sum.default(
        intermediate_cache3, out_hidden_states, c_map, rsf, topk_weights
    )

    return out_hidden_states
