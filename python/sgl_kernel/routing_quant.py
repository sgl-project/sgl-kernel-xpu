from typing import Optional, Tuple

import torch

def sgl_per_token_group_quant_fp4_xe35(
    x: torch.Tensor,
    group_size: int = 32,
    eps: float = 1e-10,
    fuse_silu_and_mul: bool = False,
    scale_map: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to MXFP4 (E2M1) format with per-token group scaling.

    MXFP4 follows the OpenCompute MX (Microscaling) format specification:
    - Data type: E2M1 (4-bit float with 2-bit exponent, 1-bit mantissa)
    - Block size: 32 elements per scale factor (default)
    - Scale format: UE8M0 (unsigned 8-bit exponent-only, no mantissa)

    Args:
        x: Input tensor with shape (..., K) where K is divisible by group_size.
           Must be contiguous and dtype float16, bfloat16, or float32.
        group_size: Number of elements per quantization group. Must be 32 for MXFP4.
        eps: Small epsilon to avoid division by zero. Default is 1e-10.
        fuse_silu_and_mul: Quantize ``silu(x[..., :K // 2]) * x[..., K // 2:]``
            instead of ``x``, i.e. fold SwiGLU into this kernel. ``x`` must then
            be 2D with an even ``K``, and the outputs describe ``K // 2``
            values per row. Same halving convention as ``silu_and_mul``.
        scale_map: Optional contiguous int32 ``[M, 2]`` table giving, per
            row ``r``, the ``(start, rows)`` of the row block it belongs to.
            Its scale for group ``g`` is then stored at flat byte
            ``start * G + g * rows + (r - start)`` of ``output_s`` instead of
            ``r * G + g`` (with ``abs(rows)`` as the block size). This emits the
            per-expert MN-major surface of ``mxfp4_moe_scale_to_mn_major_xe35``
            directly. ``rows < 0`` marks a padding row whose GEMM output is
            discarded: it stores scale 0 (2^-127) and leaves its ``output_q``
            bytes unwritten. ``rows == 0`` marks slack: nothing is written.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: with ``KOUT = K // 2`` when
        ``fuse_silu_and_mul`` and ``KOUT = K`` otherwise,
            - output_q: Packed FP4 tensor with shape (..., KOUT // 2) and dtype uint8.
                        Two E2M1 values are packed into each byte.
            - output_s: Scale tensor with shape (..., KOUT // group_size) and dtype uint8.
                        Scales are stored in UE8M0 format (exponent + 127 bias).
    """
    assert (
        x.shape[-1] % group_size == 0
    ), f"the last dimension of `x` ({x.shape[-1]}) must be divisible by `group_size` ({group_size})"
    assert x.is_contiguous(), "`x` is not contiguous"
    assert group_size == 32, f"group_size must be 32 for MXFP4, got {group_size}"

    # Ensure input is 2D for the kernel
    original_shape = x.shape
    if x.dim() == 1:
        x = x.unsqueeze(0)
    elif x.dim() > 2:
        x = x.view(-1, x.shape[-1])

    m, k = x.shape
    if fuse_silu_and_mul:
        assert (
            k % (2 * group_size) == 0
        ), f"with `fuse_silu_and_mul` the last dimension of `x` ({k}) must be divisible by 2 x {group_size}"
        k //= 2
    num_groups_per_row = k // group_size

    # Output is packed FP4 (2 values per byte)
    output_q = torch.empty((m, k // 2), device=x.device, dtype=torch.uint8)

    # Scales in row-major layout: (m, num_groups_per_row)
    # Each row has the scales for that token's groups
    output_s = torch.empty((m, num_groups_per_row), device=x.device, dtype=torch.uint8)

    if x.shape[0] > 0:
        torch.ops.sgl_kernel.sgl_per_token_group_quant_fp4_xe35.default(
            x, output_q, output_s, group_size, eps, fuse_silu_and_mul, scale_map
        )

    # Reshape output to match input shape, in the activated width
    out_k = original_shape[-1] // 2 if fuse_silu_and_mul else original_shape[-1]
    output_shape_q = original_shape[:-1] + (out_k // 2,)
    output_shape_s = original_shape[:-1] + (out_k // group_size,)

    return output_q.view(output_shape_q), output_s.view(output_shape_s)


def sgl_padded_gather_per_token_group_quant_fp4_xe35(
    x: torch.Tensor,
    row_map: torch.Tensor,
    group_size: int = 32,
    eps: float = 1e-10,
    scale_map: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather source rows and quantize them directly to MXFP4.

    ``row_map[d]`` selects the source row written to padded destination row
    ``d``. This is equivalent to
    ``sgl_per_token_group_quant_fp4_xe35(x.index_select(0, row_map))`` without
    materializing the FP16/BF16 gathered tensor.

    Args:
        x: Contiguous 2D FP16 or BF16 tensor with shape ``[M, K]``. ``K`` must
            be divisible by 32.
        row_map: Contiguous 1D int32 or int64 XPU tensor. Every entry must be in
            ``[0, M)``; repeated entries are allowed and provide padding rows.
        group_size: MXFP4 group size. Only 32 is supported.
        eps: Positive lower bound for each group's absolute maximum.
        scale_map: Optional ``(start, rows)`` scale redirection table, as in
            ``sgl_per_token_group_quant_fp4_xe35``, with one row per ``row_map``
            entry.

    Returns:
        Packed E2M1 bytes with shape ``[row_map.numel(), K // 2]`` and row-major
        UE8M0 scales with shape ``[row_map.numel(), K // 32]``.
    """
    assert x.dim() == 2, f"`x` must be 2D, got {x.dim()} dimensions"
    assert x.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"`x` must be float16 or bfloat16, got {x.dtype}"
    assert x.is_contiguous(), "`x` is not contiguous"
    assert row_map.dim() == 1, f"`row_map` must be 1D, got {row_map.dim()} dimensions"
    assert row_map.dtype in (
        torch.int32,
        torch.int64,
    ), f"`row_map` must be int32 or int64, got {row_map.dtype}"
    assert row_map.is_contiguous(), "`row_map` is not contiguous"
    assert group_size == 32, f"group_size must be 32 for MXFP4, got {group_size}"
    assert (
        x.shape[1] % group_size == 0
    ), f"the last dimension of `x` ({x.shape[1]}) must be divisible by `group_size` ({group_size})"

    rows, k = row_map.numel(), x.shape[1]
    output_q = torch.empty((rows, k // 2), device=x.device, dtype=torch.uint8)
    output_s = torch.empty((rows, k // group_size), device=x.device, dtype=torch.uint8)
    if rows > 0:
        torch.ops.sgl_kernel.sgl_padded_gather_per_token_group_quant_fp4_xe35.default(
            x, row_map, output_q, output_s, group_size, eps, scale_map
        )
    return output_q, output_s


def muse_spark_moe_routing_xe35(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: Optional[torch.Tensor] = None,
    router_output_multiplier: float = 1.0,
    route_scale: float = 1.0,
    division_eps: float = 0.0,
    local_expert_mapping: Optional[torch.Tensor] = None,
    num_valid_tokens: Optional[torch.Tensor] = None,
) -> None:
    """Muse Spark's fused MoE routing, writing into ``topk_weights``/``topk_ids``.

    ``num_valid_tokens`` is a device int32 scalar so the padded row count never
    reaches the host: rows at or after it get expert id ``-1`` and weight 0.
    """
    torch.ops.sgl_kernel.muse_spark_moe_routing_xe35.default(
        topk_weights,
        topk_ids,
        gating_output,
        correction_bias,
        router_output_multiplier,
        route_scale,
        division_eps,
        local_expert_mapping,
        num_valid_tokens,
    )


def mxfp4_moe_scale_to_mn_major_xe35(out, input, rows_per_expert):
    """Relayout per-token MXFP4 scales for moe_grouped_mm_nt_xe35_mxfp4_w4a4.

    `input` is the [M, k // 32] row-major surface `sgl_per_token_group_quant_fp4_xe35`
    emits, with rows already grouped by expert; `out` is the per-expert MN-major,
    M-padded-to-4 surface the grouped GEMM reads, and needs
    `sum(round_up(rows_i, 4))` rows (zero it first if any row count is not a
    multiple of 4). `rows_per_expert` must be contiguous int32 on device.
    Row counts must be nonnegative, sum to at most the input row count, and
    fit in the padded output. The caller validates these device-resident counts;
    this op checks shapes without copying counts to the host or draining the stream.
    """
    torch.ops.sgl_kernel.mxfp4_moe_scale_to_mn_major_xe35.default(
        out, input, rows_per_expert
    )


def topk_sigmoid_xe35(topk_weights, topk_ids, gating_output, renormalize, correction_bias=None):
    torch.ops.sgl_kernel.topk_sigmoid_xe35.default(topk_weights, topk_ids, gating_output, renormalize, correction_bias)
