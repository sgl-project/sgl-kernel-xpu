# Test for fused QK weight-multiply and RoPE (no RMSNorm)
# Mirrors test_fused_qk_norm_rope.py but omits the RMS normalization step.

import math
import sys

import pytest
import sgl_kernel
import torch
import utils

precision = {
    torch.bfloat16: 1e-2,
    torch.float16: 1e-3,
    torch.float32: 1e-5,
}
device = utils.get_device()


def apply_rotary_emb_native(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    """
    Native PyTorch rotary embedding implementation.
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, rotary_dim // 2]
        sin: [num_tokens, rotary_dim // 2]
        is_neox_style: Whether to use Neox-style or interleaved style
    """
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)

    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin

    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)


def compute_inv_freq_yarn(
    rotary_dim: int,
    base: float,
    factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
):
    """Compute inverse frequencies for YARN RoPE."""
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device="cpu")
            / rotary_dim
        )
    )

    if factor != 1.0:
        dim_range = torch.arange(0, rotary_dim, 2, dtype=torch.float32, device="cpu")
        linear_func = (dim_range - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        ramp_func = torch.clamp(linear_func, 0.0, 1.0)
        inv_freq_extrapolation = inv_freq
        inv_freq_interpolation = inv_freq / factor
        inv_freq = (
            inv_freq_interpolation * (1.0 - ramp_func)
            + inv_freq_extrapolation * ramp_func
        )

    return inv_freq


def fused_qk_rope_reference(
    qkv: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    base: float,
    is_neox: bool,
    position_ids: torch.Tensor,
    factor: float = 1.0,
    low: float = 1.0,
    high: float = 1.0,
    attention_factor: float = 1.0,
    rotary_dim: int = None,
) -> torch.Tensor:
    """
    Reference implementation: per-dimension weight scaling + RoPE only (no RMSNorm).

    Args:
        qkv: [num_tokens, (num_heads_q + num_heads_k + num_heads_v) * head_dim]
        Other args match the kernel interface.
    """
    if rotary_dim is None:
        rotary_dim = head_dim

    num_tokens = qkv.shape[0]
    total_heads = num_heads_q + num_heads_k + num_heads_v

    qkv_reshaped = qkv.view(num_tokens, total_heads, head_dim)
    q = qkv_reshaped[:, :num_heads_q, :]
    k = qkv_reshaped[:, num_heads_q : num_heads_q + num_heads_k, :]
    v = qkv_reshaped[:, num_heads_q + num_heads_k :, :]

    # Per-dimension weight scaling only (no RMSNorm)
    q_scaled = q.float() * q_weight.float()
    k_scaled = k.float() * k_weight.float()

    # Compute RoPE frequencies
    inv_freq = compute_inv_freq_yarn(rotary_dim, base, factor, low, high)

    positions = position_ids.to(torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos = freqs.cos() * attention_factor
    sin = freqs.sin() * attention_factor

    # Apply RoPE to Q and K (only rotary_dim portion)
    q_rot = q_scaled[..., :rotary_dim]
    q_pass = q_scaled[..., rotary_dim:]
    q_rot = apply_rotary_emb_native(q_rot, cos, sin, is_neox)
    q_final = torch.cat([q_rot, q_pass], dim=-1)

    k_rot = k_scaled[..., :rotary_dim]
    k_pass = k_scaled[..., rotary_dim:]
    k_rot = apply_rotary_emb_native(k_rot, cos, sin, is_neox)
    k_final = torch.cat([k_rot, k_pass], dim=-1)

    # Concatenate Q, K, V back together
    result = torch.cat([q_final, k_final, v.float()], dim=1)
    result = result.view(num_tokens, total_heads * head_dim)
    return result


@pytest.mark.parametrize("num_tokens", [1, 7, 32, 128])
@pytest.mark.parametrize("num_heads_q", [8, 32])
@pytest.mark.parametrize("num_heads_k", [8])
@pytest.mark.parametrize("num_heads_v", [8])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("is_neox", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_qk_rope_basic(
    num_tokens, num_heads_q, num_heads_k, num_heads_v, head_dim, is_neox, dtype
):
    """Test basic fused QK weight-multiply + RoPE without YARN (no RMSNorm)."""
    base = 10000.0
    factor = 1.0
    low = 1.0
    high = 1.0
    attention_factor = 1.0
    rotary_dim = head_dim

    total_heads = num_heads_q + num_heads_k + num_heads_v

    qkv = torch.randn(num_tokens, total_heads * head_dim, dtype=dtype, device=device)
    q_weight = torch.randn(head_dim, dtype=dtype, device=device)
    k_weight = torch.randn(head_dim, dtype=dtype, device=device)
    position_ids = torch.arange(num_tokens, dtype=torch.int32, device=device)

    qkv_ref = qkv.clone().float().to("cpu")
    q_weight_ref = q_weight.clone().float().to("cpu")
    k_weight_ref = k_weight.clone().float().to("cpu")
    position_ids_ref = position_ids.clone().to("cpu")

    output_ref = fused_qk_rope_reference(
        qkv_ref,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        q_weight_ref,
        k_weight_ref,
        base,
        is_neox,
        position_ids_ref,
        factor,
        low,
        high,
        attention_factor,
        rotary_dim,
    ).to(dtype)

    sgl_kernel.fused_qk_rope(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        q_weight,
        k_weight,
        base,
        is_neox,
        position_ids,
        factor,
        low,
        high,
        attention_factor,
        rotary_dim,
    )

    torch.testing.assert_close(
        qkv.to("cpu"), output_ref, rtol=precision[dtype], atol=precision[dtype]
    )


@pytest.mark.parametrize("num_tokens", [32, 128])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("is_neox", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fused_qk_rope_yarn(num_tokens, head_dim, is_neox, dtype):
    """Test fused QK weight-multiply + RoPE with YARN scaling (no RMSNorm)."""
    num_heads_q = 32
    num_heads_k = 8
    num_heads_v = 8
    base = 10000.0
    factor = 2.0
    low = 8.0
    high = 1024.0
    attention_factor = 0.707
    rotary_dim = head_dim

    total_heads = num_heads_q + num_heads_k + num_heads_v

    qkv = torch.randn(num_tokens, total_heads * head_dim, dtype=dtype, device=device)
    q_weight = torch.randn(head_dim, dtype=dtype, device=device)
    k_weight = torch.randn(head_dim, dtype=dtype, device=device)
    position_ids = torch.arange(num_tokens, dtype=torch.int32, device=device)

    qkv_ref = qkv.clone().float().to("cpu")
    q_weight_ref = q_weight.clone().float().to("cpu")
    k_weight_ref = k_weight.clone().float().to("cpu")
    position_ids_ref = position_ids.clone().to("cpu")

    output_ref = fused_qk_rope_reference(
        qkv_ref,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        q_weight_ref,
        k_weight_ref,
        base,
        is_neox,
        position_ids_ref,
        factor,
        low,
        high,
        attention_factor,
        rotary_dim,
    ).to(dtype)

    sgl_kernel.fused_qk_rope(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        q_weight,
        k_weight,
        base,
        is_neox,
        position_ids,
        factor,
        low,
        high,
        attention_factor,
        rotary_dim,
    )

    torch.testing.assert_close(
        qkv.to("cpu"), output_ref, rtol=precision[dtype] * 2, atol=precision[dtype] * 2
    )


@pytest.mark.parametrize("num_tokens", [64])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("rotary_dim", [32, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fused_qk_rope_partial_rotary(num_tokens, head_dim, rotary_dim, dtype):
    """Test with partial rotary dimensions (rotary_dim < head_dim)."""
    num_heads_q = 16
    num_heads_k = 4
    num_heads_v = 4
    base = 10000.0
    is_neox = True
    factor = 1.0
    low = 1.0
    high = 1.0
    attention_factor = 1.0

    total_heads = num_heads_q + num_heads_k + num_heads_v

    qkv = torch.randn(num_tokens, total_heads * head_dim, dtype=dtype, device=device)
    q_weight = torch.randn(head_dim, dtype=dtype, device=device)
    k_weight = torch.randn(head_dim, dtype=dtype, device=device)
    position_ids = torch.arange(num_tokens, dtype=torch.int32, device=device)

    qkv_ref = qkv.clone().float().to("cpu")
    q_weight_ref = q_weight.clone().float().to("cpu")
    k_weight_ref = k_weight.clone().float().to("cpu")
    position_ids_ref = position_ids.clone().to("cpu")

    output_ref = fused_qk_rope_reference(
        qkv_ref,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        q_weight_ref,
        k_weight_ref,
        base,
        is_neox,
        position_ids_ref,
        factor,
        low,
        high,
        attention_factor,
        rotary_dim,
    ).to(dtype)

    sgl_kernel.fused_qk_rope(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        q_weight,
        k_weight,
        base,
        is_neox,
        position_ids,
        factor,
        low,
        high,
        attention_factor,
        rotary_dim,
    )

    torch.testing.assert_close(
        qkv.to("cpu"), output_ref, rtol=precision[dtype], atol=precision[dtype]
    )


@pytest.mark.parametrize("num_tokens", [1, 32, 128])
@pytest.mark.parametrize("num_heads_q", [8, 32])
@pytest.mark.parametrize("num_heads_k", [8])
@pytest.mark.parametrize("num_heads_v", [8])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("is_neox", [True, False])
def test_fused_qk_rope_fp8_basic(
    num_tokens, num_heads_q, num_heads_k, num_heads_v, head_dim, is_neox
):
    """Test basic fused QK weight-multiply + RoPE with FP8 (no RMSNorm).

    FP8 has limited dynamic range (~448) so inputs are drawn from a small
    uniform range and tolerances are relaxed accordingly.
    Note: FP8 scale factors (dequant/requant) are not yet implemented;
    this test validates the kernel path end-to-end at unit scale.
    """
    base = 10000.0
    factor = 1.0
    low = 1.0
    high = 1.0
    attention_factor = 1.0
    rotary_dim = head_dim
    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max
    input_bound = 2.0
    weight_bound = 0.5
    assert input_bound < fp8_max
    assert weight_bound < fp8_max

    total_heads = num_heads_q + num_heads_k + num_heads_v

    # Generate bounded float16 inputs, then cast to FP8
    qkv_fp32 = (
        torch.rand(num_tokens, total_heads * head_dim) * 2.0 - 1.0
    ) * input_bound
    q_weight_fp32 = (torch.rand(head_dim) * 2.0 - 1.0) * weight_bound
    k_weight_fp32 = (torch.rand(head_dim) * 2.0 - 1.0) * weight_bound

    qkv = qkv_fp32.to(fp8_dtype).to(device)
    q_weight = q_weight_fp32.to(fp8_dtype).to(device)
    k_weight = k_weight_fp32.to(fp8_dtype).to(device)
    position_ids = torch.arange(num_tokens, dtype=torch.int32, device=device)

    # Reference uses the FP8-quantized values (same precision as kernel sees)
    qkv_ref = qkv.to(torch.float32).to("cpu")
    q_weight_ref = q_weight.to(torch.float32).to("cpu")
    k_weight_ref = k_weight.to(torch.float32).to("cpu")
    position_ids_ref = position_ids.clone().to("cpu")

    output_ref = fused_qk_rope_reference(
        qkv_ref,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        q_weight_ref,
        k_weight_ref,
        base,
        is_neox,
        position_ids_ref,
        factor,
        low,
        high,
        attention_factor,
        rotary_dim,
    )

    sgl_kernel.fused_qk_rope(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        q_weight,
        k_weight,
        base,
        is_neox,
        position_ids,
        factor,
        low,
        high,
        attention_factor,
        rotary_dim,
    )

    # Compare in float32; FP8 quantization error dominates so use loose tolerance
    torch.testing.assert_close(
        qkv.to(torch.float32).to("cpu"),
        output_ref.to(torch.float32),
        rtol=0.1,
        atol=0.1,
    )


@pytest.mark.parametrize("num_tokens", [32, 128])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("is_neox", [True, False])
def test_fused_qk_rope_fp8_yarn(num_tokens, head_dim, is_neox):
    """Test fused QK weight-multiply + RoPE with FP8 and YARN scaling (no RMSNorm)."""
    num_heads_q = 32
    num_heads_k = 8
    num_heads_v = 8
    base = 10000.0
    factor = 2.0
    low = 8.0
    high = 1024.0
    attention_factor = 0.707
    rotary_dim = head_dim
    fp8_dtype = torch.float8_e4m3fn

    total_heads = num_heads_q + num_heads_k + num_heads_v

    qkv_fp32 = (torch.rand(num_tokens, total_heads * head_dim) * 2.0 - 1.0) * 2.0
    q_weight_fp32 = (torch.rand(head_dim) * 2.0 - 1.0) * 0.5
    k_weight_fp32 = (torch.rand(head_dim) * 2.0 - 1.0) * 0.5

    qkv = qkv_fp32.to(fp8_dtype).to(device)
    q_weight = q_weight_fp32.to(fp8_dtype).to(device)
    k_weight = k_weight_fp32.to(fp8_dtype).to(device)
    position_ids = torch.arange(num_tokens, dtype=torch.int32, device=device)

    qkv_ref = qkv.to(torch.float32).to("cpu")
    q_weight_ref = q_weight.to(torch.float32).to("cpu")
    k_weight_ref = k_weight.to(torch.float32).to("cpu")
    position_ids_ref = position_ids.clone().to("cpu")

    output_ref = fused_qk_rope_reference(
        qkv_ref,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        q_weight_ref,
        k_weight_ref,
        base,
        is_neox,
        position_ids_ref,
        factor,
        low,
        high,
        attention_factor,
        rotary_dim,
    )

    sgl_kernel.fused_qk_rope(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        q_weight,
        k_weight,
        base,
        is_neox,
        position_ids,
        factor,
        low,
        high,
        attention_factor,
        rotary_dim,
    )

    torch.testing.assert_close(
        qkv.to(torch.float32).to("cpu"),
        output_ref.to(torch.float32),
        rtol=0.1,
        atol=0.1,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
