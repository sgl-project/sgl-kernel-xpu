# Test for fused QK normalization and RoPE
# Adapted from the CUDA implementation in sglang

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


def llama_rms_norm(x, w, eps=1e-6):
    """PyTorch reference implementation of RMS normalization."""
    orig_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * w.float()
    x = x.to(orig_dtype)
    return x


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
        # Neox style: split in half along head dimension
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        # Interleaved style: even and odd indices
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin

    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)


def compute_inv_freq_yarn(
    head_dim: int,
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
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
            / rotary_dim
        )
    )

    if factor != 1.0:
        # YARN scaling
        dim_range = torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)

        # Compute linear interpolation factor
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


def fused_qk_norm_rope_reference(
    qkv: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
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
    Reference implementation in PyTorch for testing.

    Args:
        qkv: [num_tokens, (num_heads_q + num_heads_k + num_heads_v) * head_dim]
        Other args match the kernel interface
    """
    if rotary_dim is None:
        rotary_dim = head_dim

    num_tokens = qkv.shape[0]
    total_heads = num_heads_q + num_heads_k + num_heads_v

    # Reshape QKV to separate Q, K, V
    qkv_reshaped = qkv.view(num_tokens, total_heads, head_dim)

    q = qkv_reshaped[:, :num_heads_q, :]
    k = qkv_reshaped[:, num_heads_q : num_heads_q + num_heads_k, :]
    v = qkv_reshaped[:, num_heads_q + num_heads_k :, :]

    # Apply RMSNorm to Q and K
    q_normalized = llama_rms_norm(q, q_weight, eps)
    k_normalized = llama_rms_norm(k, k_weight, eps)

    # Compute RoPE frequencies
    inv_freq = compute_inv_freq_yarn(head_dim, rotary_dim, base, factor, low, high)

    # Compute cos and sin for each position
    positions = position_ids.to(torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()

    # Apply attention factor
    cos = cos * attention_factor
    sin = sin * attention_factor

    # Apply RoPE to Q and K (only to rotary_dim portion)
    q_rot = q_normalized[..., :rotary_dim]
    q_pass = q_normalized[..., rotary_dim:]
    q_rot = apply_rotary_emb_native(q_rot, cos, sin, is_neox)
    q_final = torch.cat([q_rot, q_pass], dim=-1)

    k_rot = k_normalized[..., :rotary_dim]
    k_pass = k_normalized[..., rotary_dim:]
    k_rot = apply_rotary_emb_native(k_rot, cos, sin, is_neox)
    k_final = torch.cat([k_rot, k_pass], dim=-1)

    # Concatenate Q, K, V back together
    result = torch.cat([q_final, k_final, v], dim=1)
    result = result.view(num_tokens, total_heads * head_dim)

    return result


@pytest.mark.parametrize("num_tokens", [1, 7, 32, 128])
@pytest.mark.parametrize("num_heads_q", [8, 32])
@pytest.mark.parametrize("num_heads_k", [8])
@pytest.mark.parametrize("num_heads_v", [8])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("is_neox", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_qk_norm_rope_basic(
    num_tokens, num_heads_q, num_heads_k, num_heads_v, head_dim, is_neox, dtype
):
    """Test basic fused QK norm + RoPE without YARN."""
    eps = 1e-6
    base = 10000.0
    factor = 1.0
    low = 1.0
    high = 1.0
    attention_factor = 1.0
    rotary_dim = head_dim

    total_heads = num_heads_q + num_heads_k + num_heads_v

    # Create input tensors
    qkv = torch.randn(num_tokens, total_heads * head_dim, dtype=dtype, device=device)
    q_weight = torch.randn(head_dim, dtype=dtype, device=device)
    k_weight = torch.randn(head_dim, dtype=dtype, device=device)
    position_ids = torch.arange(num_tokens, dtype=torch.int32, device=device)

    # Create a copy for reference
    qkv_ref = qkv.clone().float()
    q_weight_ref = q_weight.clone().float()
    k_weight_ref = k_weight.clone().float()
    position_ids_ref = position_ids.clone()

    # Compute reference output
    output_ref = fused_qk_norm_rope_reference(
        qkv_ref,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
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

    # Run kernel (in-place operation)
    sgl_kernel.fused_qk_norm_rope(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
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

    # Compare results
    torch.testing.assert_close(
        qkv, output_ref, rtol=precision[dtype], atol=precision[dtype]
    )


@pytest.mark.parametrize("num_tokens", [32, 128])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("is_neox", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fused_qk_norm_rope_yarn(num_tokens, head_dim, is_neox, dtype):
    """Test fused QK norm + RoPE with YARN scaling."""
    num_heads_q = 32
    num_heads_k = 8
    num_heads_v = 8
    eps = 1e-6
    base = 10000.0
    factor = 2.0  # YARN factor
    low = 8.0
    high = 1024.0
    attention_factor = 0.707  # sqrt(0.5)
    rotary_dim = head_dim

    total_heads = num_heads_q + num_heads_k + num_heads_v

    # Create input tensors
    qkv = torch.randn(num_tokens, total_heads * head_dim, dtype=dtype, device=device)
    q_weight = torch.randn(head_dim, dtype=dtype, device=device)
    k_weight = torch.randn(head_dim, dtype=dtype, device=device)
    position_ids = torch.arange(num_tokens, dtype=torch.int32, device=device)

    # Create a copy for reference
    qkv_ref = qkv.clone().float()
    q_weight_ref = q_weight.clone().float()
    k_weight_ref = k_weight.clone().float()
    position_ids_ref = position_ids.clone()

    # Compute reference output
    output_ref = fused_qk_norm_rope_reference(
        qkv_ref,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
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

    # Run kernel (in-place operation)
    sgl_kernel.fused_qk_norm_rope(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
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

    # Compare results - use slightly relaxed tolerance for YARN
    torch.testing.assert_close(
        qkv, output_ref, rtol=precision[dtype] * 2, atol=precision[dtype] * 2
    )


@pytest.mark.parametrize("num_tokens", [64])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("rotary_dim", [32, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fused_qk_norm_rope_partial_rotary(num_tokens, head_dim, rotary_dim, dtype):
    """Test with partial rotary dimensions (rotary_dim < head_dim)."""
    num_heads_q = 16
    num_heads_k = 4
    num_heads_v = 4
    eps = 1e-6
    base = 10000.0
    is_neox = True
    factor = 1.0
    low = 1.0
    high = 1.0
    attention_factor = 1.0

    total_heads = num_heads_q + num_heads_k + num_heads_v

    # Create input tensors
    qkv = torch.randn(num_tokens, total_heads * head_dim, dtype=dtype, device=device)
    q_weight = torch.randn(head_dim, dtype=dtype, device=device)
    k_weight = torch.randn(head_dim, dtype=dtype, device=device)
    position_ids = torch.arange(num_tokens, dtype=torch.int32, device=device)

    # Create a copy for reference
    qkv_ref = qkv.clone().float()
    q_weight_ref = q_weight.clone().float()
    k_weight_ref = k_weight.clone().float()
    position_ids_ref = position_ids.clone()

    # Compute reference output
    output_ref = fused_qk_norm_rope_reference(
        qkv_ref,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
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

    # Run kernel (in-place operation)
    sgl_kernel.fused_qk_norm_rope(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
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

    # Compare results
    torch.testing.assert_close(
        qkv, output_ref, rtol=precision[dtype], atol=precision[dtype]
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
