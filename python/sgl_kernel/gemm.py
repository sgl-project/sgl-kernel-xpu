from typing import List, Optional, Tuple

import torch
from sgl_kernel.utils import _get_cache_buf, get_xpu_stream


def awq_dequantize(
    qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor
) -> torch.ByteTensor:
    return torch.ops.sgl_kernel.awq_dequantize.default(qweight, scales, qzeros)


def int8_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=None):
    return torch.ops.sgl_kernel.int8_scaled_mm.default(
        mat_a,
        mat_b,
        scales_a,
        scales_b,
        out_dtype,
        bias,
    )


def fp8_blockwise_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype):
    return torch.ops.sgl_kernel.fp8_blockwise_scaled_mm.default(
        mat_a,
        mat_b,
        scales_a,
        scales_b,
        out_dtype,
    )


def fp8_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=None):
    return torch.ops.sgl_kernel.fp8_scaled_mm.default(
        mat_a,
        mat_b,
        scales_a,
        scales_b,
        out_dtype,
        bias,
    )


def mxfp8_w8a16_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dense MXFP8 W8A16 GEMM: out[M, N] = input[M, K] @ weight[N, K]^T.

    - input: bf16 activations, shape [M, K] (or [*, K]; flattened here).
    - weight: float8_e4m3fn MXFP8 weights, shape [N, K] (one byte per element).
    - weight_scale: fp32 per-32-K-block "direct multiplier" scales, shape
      [N, K/32] (UE8M0 already decoded to 2**(e-127) by the caller).
    - bias: optional fp32 bias, shape [N].

    Wraps the grouped kernel with a single logical expert (dense case).
    """
    assert input.dim() >= 2, "input must be at least 2D [.., K]"
    assert weight.dim() == 2, "weight must be 2D [N, K]"
    assert weight.dtype == torch.float8_e4m3fn, "weight must be float8_e4m3fn"
    assert weight_scale.dtype == torch.float32, "weight_scale must be float32 (direct multiplier)"

    out_shape = (*input.shape[:-1], weight.shape[0])
    input_2d = input.reshape(-1, input.shape[-1])
    input_2d = input_2d.contiguous() if not input_2d.is_contiguous() else input_2d
    m = input_2d.shape[0]
    n, k = weight.shape

    # Grouped kernel expects [E, N, K] weights, [E, N, K/32] scales and a
    # per-expert row count; the dense case is a single expert covering all M rows.
    packed_weights = weight.unsqueeze(0)
    scales = weight_scale.unsqueeze(0)
    total_rows_for_experts = torch.tensor([m], dtype=torch.int32, device=input.device)

    bias_arg = None
    if bias is not None:
        # Kernel expects fp32 bias shaped [E, N].
        bias_arg = bias.to(torch.float32).reshape(1, n).contiguous()

    output = torch.empty((m, n), dtype=torch.bfloat16, device=input.device)
    torch.ops.sgl_kernel.moe_grouped_mm_nt_xe20_mxfp8_w8a16.default(
        output,
        input_2d,
        packed_weights,
        scales,
        bias_arg,
        total_rows_for_experts,
        1,
    )
    return output.reshape(out_shape)


def _bmm_fp8_internal(
    workspace_buffer: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
) -> None:
    cublas_handle = torch.cuda.current_blas_handle()
    torch.ops.sgl_kernel.bmm_fp8.default(
        A,
        B,
        D,
        A_scale,
        B_scale,
        workspace_buffer,
        cublas_handle,
        get_xpu_stream(),
    )


def bmm_fp8(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty(
            (A.shape[0], A.shape[1], B.shape[2]),
            device=A.device,
            dtype=dtype,
        )
    workspace_buffer = _get_cache_buf("bmm_fp8_workspace", 32 * 1024 * 1024, A.device)
    _bmm_fp8_internal(workspace_buffer, A, B, out, A_scale, B_scale)
    return out


def sgl_per_token_group_quant_8bit(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    masked_m: Optional[torch.Tensor] = None,
    enable_v2: Optional[bool] = None,
) -> None:
    if enable_v2 is None:
        from sglang.srt.utils import get_bool_env_var

        enable_v2 = get_bool_env_var("SGLANG_PER_TOKEN_GROUP_QUANT_8BIT_V2")

    if enable_v2:
        return torch.ops.sgl_kernel.sgl_per_token_group_quant_8bit_v2.default(
            input,
            output_q,
            output_s,
            group_size,
            eps,
            fp8_min,
            fp8_max,
            scale_ue8m0,
            fuse_silu_and_mul,
            masked_m,
        )

    assert not fuse_silu_and_mul, "only v2 support fuse_silu_and_mul"
    assert masked_m is None, "only v2 support masked_m"
    torch.ops.sgl_kernel.sgl_per_token_group_quant_8bit.default(
        input, output_q, output_s, group_size, eps, fp8_min, fp8_max, scale_ue8m0
    )


# For legacy usage
sgl_per_token_group_quant_fp8 = sgl_per_token_group_quant_8bit
sgl_per_token_group_quant_int8 = sgl_per_token_group_quant_8bit


def sgl_per_tensor_quant_fp8(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    is_static: bool,
) -> None:
    torch.ops.sgl_kernel.sgl_per_tensor_quant_fp8.default(
        input, output_q, output_s, is_static
    )


def sgl_per_token_group_quant_fp4(
    x: torch.Tensor,
    group_size: int = 32,
    eps: float = 1e-10,
    x_secondary: Optional[torch.Tensor] = None,
    column_major_scales: bool = False,
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
           When x_secondary is provided, this is interpreted as the gate projection.
        group_size: Number of elements per quantization group. Must be 32 for MXFP4.
        eps: Small epsilon to avoid division by zero. Default is 1e-10.
        x_secondary: Optional secondary input tensor for SiLU+Mul fusion.
                     When provided, computes quantize(SiLU(x) * x_secondary).
                     Must have same shape, dtype, and device as x.
        column_major_scales: If True, store scales in column-major interleaved layout
                             for better cache locality in MoE workloads. Default is False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - output_q: Packed FP4 tensor with shape (..., K // 2) and dtype uint8.
                        Two E2M1 values are packed into each byte.
            - output_s: Scale tensor with shape (..., K // group_size) and dtype uint8.
                        Scales are stored in UE8M0 format (exponent + 127 bias).
                        If column_major_scales=True, scales are interleaved in memory.
    """
    assert (
        x.shape[-1] % group_size == 0
    ), f"the last dimension of `x` ({x.shape[-1]}) must be divisible by `group_size` ({group_size})"
    assert x.is_contiguous(), "`x` is not contiguous"
    assert group_size == 32, f"group_size must be 32 for MXFP4, got {group_size}"

    # Validate x_secondary if provided
    if x_secondary is not None:
        assert (
            x_secondary.shape == x.shape
        ), f"x_secondary shape {x_secondary.shape} must match x shape {x.shape}"
        assert (
            x_secondary.dtype == x.dtype
        ), f"x_secondary dtype {x_secondary.dtype} must match x dtype {x.dtype}"
        assert (
            x_secondary.device == x.device
        ), f"x_secondary device {x_secondary.device} must match x device {x.device}"
        assert x_secondary.is_contiguous(), "`x_secondary` is not contiguous"

    # Ensure input is 2D for the kernel
    original_shape = x.shape
    if x.dim() == 1:
        x = x.unsqueeze(0)
        if x_secondary is not None:
            x_secondary = x_secondary.unsqueeze(0)
    elif x.dim() > 2:
        x = x.view(-1, x.shape[-1])
        if x_secondary is not None:
            x_secondary = x_secondary.view(-1, x_secondary.shape[-1])

    m, k = x.shape
    num_groups_per_row = k // group_size

    # Output is packed FP4 (2 values per byte)
    output_q = torch.empty((m, k // 2), device=x.device, dtype=torch.uint8)

    # Scales: create with transposed strides if column-major requested
    if column_major_scales:
        # Create as (groups, tokens) then permute to (tokens, groups)
        # This gives stride(0) < stride(1), triggering column-major in kernel
        output_s = torch.empty(
            (num_groups_per_row, m), device=x.device, dtype=torch.uint8
        ).permute(1, 0)
    else:
        # Row-major layout: (m, num_groups_per_row)
        output_s = torch.empty(
            (m, num_groups_per_row), device=x.device, dtype=torch.uint8
        )

    if x.shape[0] > 0:
        torch.ops.sgl_kernel.sgl_per_token_group_quant_fp4.default(
            x, output_q, output_s, group_size, eps, x_secondary
        )

    # Reshape output to match input shape
    output_shape_q = original_shape[:-1] + (original_shape[-1] // 2,)
    output_shape_s = original_shape[:-1] + (original_shape[-1] // group_size,)

    return output_q.view(output_shape_q), output_s.view(output_shape_s)


def sgl_per_token_quant_fp8(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
) -> None:
    torch.ops.sgl_kernel.sgl_per_token_quant_fp8.default(input, output_q, output_s)


def cutlass_scaled_fp4_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2
    m, n = a.shape[0], b.shape[0]
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)
    torch.ops.sgl_kernel.cutlass_scaled_fp4_mm.default(
        out, a, b, block_scale_a, block_scale_b, alpha
    )
    return out


def scaled_fp4_quant(
    input: torch.Tensor, input_global_scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP4 and return quantized tensor and scale.

    This function quantizes the last dimension of the given tensor `input`. For
    every 16 consecutive elements, a single dynamically computed scaling factor
    is shared. This scaling factor is quantized using the `input_global_scale`
    and is stored in a swizzled layout (see
    https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-layout-4x).

    Args:
        input: The input tensor to be quantized to FP4
        input_global_scale: A scalar scaling factor for the entire tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The output tensor in FP4 but every
            two values are packed into a uint8 and float8_e4m3 scaling factors
            in a sizzled layout.
    """
    assert input.ndim >= 1, f"input.ndim needs to be >= 1, but got {input.ndim}."
    other_dims = 1 if input.ndim == 1 else -1
    input = input.reshape(other_dims, input.shape[-1])
    m, n = input.shape
    block_size = 16
    device = input.device

    assert n % block_size == 0, f"last dim has to be multiple of 16, but got {n}."
    assert input.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"input.dtype needs to be fp16 or bf16 but got {input.dtype}."

    # Two fp4 values will be packed into an uint8.
    output = torch.empty((m, n // 2), device=device, dtype=torch.uint8)

    # We use the rounded values to store the swizzled values. Then, the scaling
    # factors in float8_e4m3fn are packed into an int32 for every 4 values.
    rounded_m = ((m + 128 - 1) // 128) * 128
    scale_n = n // block_size
    rounded_n = ((scale_n + 4 - 1) // 4) * 4
    output_scale = torch.empty(
        (rounded_m, rounded_n // 4), device=device, dtype=torch.int32
    )

    torch.ops.sgl_kernel.scaled_fp4_quant.default(
        output, input, output_scale, input_global_scale
    )
    output_scale = output_scale.view(torch.float8_e4m3fn)
    return output, output_scale


def qserve_w4a8_per_chn_gemm(
    in_feats: torch.Tensor,
    kernel: torch.Tensor,
    wscales: torch.Tensor,
    ascales: torch.Tensor,
    w_szs: torch.Tensor,
    a_ssums: torch.Tensor,
    out_feats: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out_feats is None:
        # NOTE(HandH1998): qserve_w4a8_per_chn_gemm only supports out dtype=torch.float16 now
        out_feats = torch.empty(
            (in_feats.shape[0], kernel.shape[0]),
            device=in_feats.device,
            dtype=torch.float16,
        )
    torch.ops.sgl_kernel.qserve_w4a8_per_chn_gemm.default(
        in_feats, kernel, wscales, ascales, w_szs, a_ssums, out_feats
    )
    return out_feats


def qserve_w4a8_per_group_gemm(
    in_feats: torch.Tensor,
    kernel: torch.Tensor,
    zeros: torch.Tensor,
    scales_i8: torch.Tensor,
    wscales: torch.Tensor,
    ascales: torch.Tensor,
    out_feats: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out_feats is None:
        # NOTE(HandH1998): qserve_w4a8_per_group_gemm only supports out dtype=torch.float16 now
        out_feats = torch.empty(
            (in_feats.shape[0], kernel.shape[0]),
            device=in_feats.device,
            dtype=torch.float16,
        )
    torch.ops.sgl_kernel.qserve_w4a8_per_group_gemm.default(
        in_feats, kernel, zeros, scales_i8, wscales, ascales, out_feats
    )
    return out_feats


def scaled_fp4_experts_quant(
    input_tensor: torch.Tensor,
    input_global_scale: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    topk: int,
    expert_map: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP4 and return quantized tensor and scale, for
    packed MoE Inputs.
    Args:
        input: The input tensor to be quantized to FP4
        expert_map: The expert map tensor
        input_global_scale: A scalar scaling factor for the entire tensor.
        expert_offsets: The expert offsets tensor
        blockscale_offsets: The blockscale offsets tensor
    Outputs:
        output: The quantized tensor in FP4
        output_scales: The blockscale tensor in FP8-E4M3
    """
    assert (
        input_tensor.ndim == 2
    ), f"input.ndim needs to be == 2, but got {input_tensor.ndim}."
    if expert_map is not None:
        (m, k) = input_tensor.shape
        output_tensor_shape = (m * topk, k)
        input_tensor = input_tensor[expert_map]
    m_numtopk, k = input_tensor.shape
    # Control the maximum number of tokens per expert supported by the
    # NVFP4 MoE Expert Quantization. This is used to prevent the kernel
    # from running out of memory. This value can also be increased to support
    # larger models.
    import os

    MAX_TOKENS_PER_EXPERT = os.environ.get("MODELOPT_MAX_TOKENS_PER_EXPERT", 65536)
    assert m_numtopk <= MAX_TOKENS_PER_EXPERT * topk, (
        f"m_numtopk must be less than MAX_TOKENS_PER_EXPERT("
        f"{MAX_TOKENS_PER_EXPERT})"
        f" for cutlass_moe_fp4, observed m_numtopk = {m_numtopk}. Use"
        f" MODELOPT_MAX_TOKENS_PER_EXPERT to set this value."
    )
    scales_k = k // 16
    padded_k = (scales_k + (4 - 1)) // 4

    # output is uint8 and packed fp4 values
    output = torch.empty(
        m_numtopk, k // 2, device=input_tensor.device, dtype=torch.uint8
    )
    output_scales = torch.empty(
        MAX_TOKENS_PER_EXPERT * topk,
        padded_k,
        dtype=torch.int32,
        device=input_tensor.device,
    )
    torch.ops.sgl_kernel.scaled_fp4_experts_quant.default(
        output,
        output_scales,
        input_tensor,
        input_global_scale,
        expert_offsets,
        blockscale_offsets,
    )
    output_scales = output_scales.view(torch.float8_e4m3fn)
    return output, output_scales
