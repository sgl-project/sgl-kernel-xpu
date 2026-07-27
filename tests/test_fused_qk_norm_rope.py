# Test for fused QK normalization and RoPE
# Adapted from the CUDA implementation in sglang

import math
import sys

import pytest
import sgl_kernel
import torch
import utils
from sgl_kernel.fused_k_norm_rope_flashmla_torch import (
    fused_k_norm_rope_flashmla as fused_k_norm_rope_flashmla_ref,
)
from test_rope_utils import create_cos_sin_cache

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

    # Compute cos and sin for each position. Ensure both tensors are on the
    # same device to avoid cross-device ops (tests sometimes pass CPU tensors
    # as reference while inv_freq is constructed on `device`).
    positions = position_ids.to(torch.float32)
    inv_freq = inv_freq.to(positions.device)
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


def fused_qk_norm_rope_with_cache_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    is_neox: bool,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation for the cache-based fused QK norm + RoPE path."""
    head_dim = q.shape[-1]
    rope_dim = cos_sin_cache.shape[-1]
    positions = positions.flatten()
    flat_tokens = positions.numel()

    assert rope_dim % 2 == 0
    assert rope_dim <= head_dim

    cos_cache, sin_cache = cos_sin_cache.chunk(2, dim=-1)
    cos = cos_cache[positions].to(q.dtype)
    sin = sin_cache[positions].to(q.dtype)

    q_view = q.reshape(flat_tokens, -1, head_dim)
    k_view = k.reshape(flat_tokens, -1, head_dim)

    q_norm = llama_rms_norm(q_view, q_weight, eps)
    k_norm = llama_rms_norm(k_view, k_weight, eps)

    q_rot = q_norm[..., :rope_dim]
    q_pass = q_norm[..., rope_dim:]
    q_rot = apply_rotary_emb_native(q_rot, cos, sin, is_neox)
    q_out = torch.cat((q_rot, q_pass), dim=-1).reshape(q.shape)

    k_rot = k_norm[..., :rope_dim]
    k_pass = k_norm[..., rope_dim:]
    k_rot = apply_rotary_emb_native(k_rot, cos, sin, is_neox)
    k_out = torch.cat((k_rot, k_pass), dim=-1).reshape(k.shape)

    return q_out, k_out


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
    torch.random.manual_seed(42)
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
    torch.random.manual_seed(42)
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
    torch.random.manual_seed(42)
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


@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8])
@pytest.mark.parametrize(
    "num_heads_q,num_heads_k,num_heads_v",
    [
        (2, 2, 2),
        (1, 1, 1),
        (4, 2, 2),
        (8, 4, 2),
    ],
)
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("is_neox", [True, False])
def test_fused_qk_norm_rope_fp8_e4m3(
    num_tokens, num_heads_q, num_heads_k, num_heads_v, head_dim, is_neox
):
    """Test fused QK norm + RoPE with FP8_E4M3 dtype."""
    torch.random.manual_seed(42)
    dtype = torch.float8_e4m3fn
    eps = 1e-6
    base = 10000.0
    factor = 1.0
    low = 1.0
    high = 1.0
    attention_factor = 1.0
    rotary_dim = head_dim

    total_heads = num_heads_q + num_heads_k + num_heads_v

    # Create input tensors in float32 first, then convert to FP8
    qkv_f32 = torch.randn(
        num_tokens, total_heads * head_dim, dtype=torch.float32, device=device
    )
    # Clamp to FP8 representable range to avoid infinities/NaNs on conversion
    qkv_f32 = qkv_f32.clamp(-448.0, 448.0)
    qkv = qkv_f32.to(dtype)

    q_weight_f32 = torch.randn(head_dim, dtype=torch.float32, device=device)
    q_weight_f32 = q_weight_f32.clamp(-448.0, 448.0)
    q_weight = q_weight_f32.to(dtype)

    k_weight_f32 = torch.randn(head_dim, dtype=torch.float32, device=device)
    k_weight_f32 = k_weight_f32.clamp(-448.0, 448.0)
    k_weight = k_weight_f32.to(dtype)

    position_ids = torch.arange(num_tokens, dtype=torch.int32, device=device)

    # Create a copy for reference from FP8-dequantized values
    qkv_ref = qkv.to(torch.float32).clone().cpu()
    q_weight_ref = q_weight.to(torch.float32).clone().cpu()
    k_weight_ref = k_weight.to(torch.float32).clone().cpu()
    position_ids_ref = position_ids.clone().cpu()

    # Compute reference output on CPU
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
    ).to(device)

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

    # Compare results - use relaxed tolerance for FP8
    # FP8 has limited precision, so we need higher tolerance
    torch.testing.assert_close(qkv.to(torch.float32), output_ref, rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize(
    "use_4d,batch_size,seq_len,num_qo_heads,num_kv_heads,head_dim,rope_dim,is_neox,dtype,position_dtype",
    [
        (False, 3, None, 4, 2, 64, 32, False, torch.bfloat16, torch.int32),
        (False, 5, None, 8, 4, 128, 64, True, torch.float16, torch.int64),
        (True, 2, 4, 16, 4, 128, 128, False, torch.bfloat16, torch.int32),
        (True, 1, 8, 32, 8, 256, 128, True, torch.float16, torch.int64),
    ],
)
def test_fused_qk_norm_rope_with_cache(
    use_4d,
    batch_size,
    seq_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    rope_dim,
    is_neox,
    dtype,
    position_dtype,
):
    """Test fused QK norm + RoPE with a precomputed cos/sin cache."""
    torch.random.manual_seed(42)

    assert rope_dim <= head_dim

    if use_4d:
        assert seq_len is not None
        q = torch.randn(
            batch_size, seq_len, num_qo_heads, head_dim, dtype=dtype, device=device
        )
        k = torch.randn(
            batch_size, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device
        )
        num_tokens = batch_size * seq_len
    else:
        q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch_size, num_kv_heads, head_dim, dtype=dtype, device=device)
        num_tokens = batch_size

    q_weight = torch.randn(head_dim, dtype=dtype, device=device)
    k_weight = torch.randn(head_dim, dtype=dtype, device=device)
    positions = torch.arange(num_tokens, dtype=position_dtype, device=device)
    cos_sin_cache = create_cos_sin_cache(rope_dim, max_position=num_tokens + 1)

    q_ref, k_ref = fused_qk_norm_rope_with_cache_reference(
        q.clone().float(),
        k.clone().float(),
        q_weight.clone().float(),
        k_weight.clone().float(),
        cos_sin_cache,
        positions,
        is_neox,
    )

    q_test = q.clone()
    k_test = k.clone()
    sgl_kernel.fused_inplace_qknorm_rope(
        q_test,
        k_test,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        is_neox,
    )

    torch.testing.assert_close(
        q_test, q_ref.to(dtype), rtol=precision[dtype], atol=precision[dtype]
    )
    torch.testing.assert_close(
        k_test, k_ref.to(dtype), rtol=precision[dtype], atol=precision[dtype]
    )


@pytest.mark.parametrize(
    "use_4d,batch_size,seq_len,num_qo_heads,num_kv_heads,head_dim,rope_dim,is_neox,dtype,position_dtype,last_dim_padding",
    [
        (False, 3, None, 4, 2, 64, 32, False, torch.bfloat16, torch.int32, 16),
        (True, 2, 4, 16, 4, 128, 128, True, torch.float16, torch.int64, 32),
    ],
)
def test_fused_qk_norm_rope_with_cache_last_dim_strided(
    use_4d,
    batch_size,
    seq_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    rope_dim,
    is_neox,
    dtype,
    position_dtype,
    last_dim_padding,
):
    """Test fused QK norm + RoPE with non-contiguous Q/K views."""
    torch.random.manual_seed(42)

    assert rope_dim <= head_dim

    if use_4d:
        assert seq_len is not None
        q_storage = torch.randn(
            batch_size,
            seq_len,
            num_qo_heads,
            head_dim + last_dim_padding,
            dtype=dtype,
            device=device,
        )
        k_storage = torch.randn(
            batch_size,
            seq_len,
            num_kv_heads,
            head_dim + last_dim_padding,
            dtype=dtype,
            device=device,
        )
        num_tokens = batch_size * seq_len
    else:
        q_storage = torch.randn(
            batch_size,
            num_qo_heads,
            head_dim + last_dim_padding,
            dtype=dtype,
            device=device,
        )
        k_storage = torch.randn(
            batch_size,
            num_kv_heads,
            head_dim + last_dim_padding,
            dtype=dtype,
            device=device,
        )
        num_tokens = batch_size

    q = q_storage[..., :head_dim]
    k = k_storage[..., :head_dim]
    assert q.stride(-1) == 1
    assert k.stride(-1) == 1
    assert not q.is_contiguous()
    assert not k.is_contiguous()

    q_weight = torch.randn(head_dim, dtype=dtype, device=device)
    k_weight = torch.randn(head_dim, dtype=dtype, device=device)
    positions = torch.arange(num_tokens, dtype=position_dtype, device=device)
    cos_sin_cache = create_cos_sin_cache(rope_dim, max_position=num_tokens + 1)

    q_ref, k_ref = fused_qk_norm_rope_with_cache_reference(
        q.clone().float(),
        k.clone().float(),
        q_weight.clone().float(),
        k_weight.clone().float(),
        cos_sin_cache,
        positions,
        is_neox,
    )

    sgl_kernel.fused_inplace_qknorm_rope(
        q,
        k,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        is_neox,
    )

    torch.testing.assert_close(
        q, q_ref.to(dtype), rtol=precision[dtype], atol=precision[dtype]
    )
    torch.testing.assert_close(
        k, k_ref.to(dtype), rtol=precision[dtype], atol=precision[dtype]
    )


def fused_q_norm_rope_reference(
    q_input: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Pure-PyTorch reference for the DeepSeek-V4 Q path: unweighted
    RMSNorm-self over the full head_dim, then RoPE on the last `rope_dim`
    elements (interleaved re/im), matching `FusedQNormRopeKernel`
    (python/sglang/jit_kernel/csrc/deepseek_v4/main_norm_rope.cuh).

    Args:
        q_input: (B, H, head_dim), any float dtype.
        freqs_cis: (max_pos, rope_dim) fp32, interleaved
            [re0, im0, re1, im1, ...].
        positions: (B,) int32 or int64.
        eps: RMSNorm epsilon.

    Returns:
        (B, H, head_dim) tensor with q_input's dtype.
    """
    B, H, head_dim = q_input.shape
    rope_dim = freqs_cis.shape[-1]
    nope_dim = head_dim - rope_dim

    assert rope_dim % 2 == 0, "rope_dim must be even (interleaved re/im)"
    assert nope_dim >= 0

    # part 1: RMSNorm-self (no learned weight).
    x = q_input.float()
    rms = x.pow(2).mean(dim=-1, keepdim=True)
    norm_factor = torch.rsqrt(rms + eps)
    x = x * norm_factor

    # part 2: RoPE on the last rope_dim elements.
    freq_rows = freqs_cis[positions.long()].unsqueeze(1)  # (B, 1, rope_dim)
    freq_re = freq_rows[..., 0::2]
    freq_im = freq_rows[..., 1::2]

    x_rope = x[..., nope_dim:]
    x_re = x_rope[..., 0::2]
    x_im = x_rope[..., 1::2]

    rotated_re = x_re * freq_re - x_im * freq_im
    rotated_im = x_re * freq_im + x_im * freq_re
    rotated = torch.stack([rotated_re, rotated_im], dim=-1).flatten(-2)

    out = torch.cat([x[..., :nope_dim], rotated], dim=-1)
    return out.to(q_input.dtype)


@pytest.mark.parametrize("batch_size", [1, 4, 16, 32])
@pytest.mark.parametrize("num_heads", [1, 8, 64])
@pytest.mark.parametrize(
    "head_dim,rope_dim",
    [
        (64, 64),  # warp path, exact fit
        (128, 64),  # warp path
        (192, 84),  # warp path, different rope_dim to meet warp
        (256, 64),  # warp path
        (320, 64),  # head_dim not in {64,128,192,256} -> CTA path
        (512, 64),  # DeepSeek-V4 production shape -> CTA path
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_q_norm_rope(batch_size, num_heads, head_dim, rope_dim, dtype):
    """Test the Q-only fused RMSNorm-self + RoPE kernel (warp and CTA paths)."""
    torch.random.manual_seed(42)
    max_pos = 512
    eps = 1e-6

    q_input = torch.randn(batch_size, num_heads, head_dim, dtype=dtype, device=device)
    q_output = torch.empty_like(q_input)
    freqs_cis = torch.randn(
        max_pos, rope_dim // 2, dtype=torch.complex64, device=device
    )
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)
    positions = torch.randint(
        0, max_pos, (batch_size,), dtype=torch.int32, device=device
    )

    sgl_kernel.fused_q_norm_rope(q_input, q_output, freqs_real, positions, eps)

    expected = fused_q_norm_rope_reference(q_input, freqs_real, positions, eps)

    torch.testing.assert_close(
        q_output.float(), expected.float(), rtol=precision[dtype], atol=precision[dtype]
    )


@pytest.mark.parametrize("head_dim,rope_dim", [(128, 64), (192, 84), (512, 64)])
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
def test_fused_q_norm_rope_position_dtype(head_dim, rope_dim, position_dtype):
    """Test both supported `positions` dtypes for warp (128) and CTA (512) paths."""
    torch.random.manual_seed(0)
    batch_size, num_heads, max_pos, eps = 8, 4, 512, 1e-6
    dtype = torch.bfloat16

    q_input = torch.randn(batch_size, num_heads, head_dim, dtype=dtype, device=device)
    q_output = torch.empty_like(q_input)
    freqs_cis = torch.randn(
        max_pos, rope_dim // 2, dtype=torch.complex64, device=device
    )
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)
    positions = torch.randint(
        0, max_pos, (batch_size,), dtype=position_dtype, device=device
    )

    sgl_kernel.fused_q_norm_rope(q_input, q_output, freqs_real, positions, eps)
    expected = fused_q_norm_rope_reference(q_input, freqs_real, positions, eps)

    torch.testing.assert_close(
        q_output.float(), expected.float(), rtol=precision[dtype], atol=precision[dtype]
    )


def test_fused_q_norm_rope_zero_batch():
    """Empty batch should not crash, for both warp and CTA head_dims."""
    for head_dim in (128, 192, 512):
        q_input = torch.empty(0, 8, head_dim, dtype=torch.bfloat16, device=device)
        q_output = torch.empty(0, 8, head_dim, dtype=torch.bfloat16, device=device)
        freqs_cis = torch.randn(512, 32, dtype=torch.complex64, device=device)
        freqs_real = torch.view_as_real(freqs_cis).flatten(-2)
        positions = torch.empty(0, dtype=torch.int32, device=device)
        sgl_kernel.fused_q_norm_rope(q_input, q_output, freqs_real, positions, 1e-6)
        assert q_output.shape == q_input.shape


@pytest.mark.parametrize("batch_size", [1, 7, 32, 128])
@pytest.mark.parametrize("page_size", [16, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_fused_k_norm_rope_flashmla(batch_size, page_size, dtype):
    """Test fused K norm + RoPE + FlashMLA paged cache store kernel against reference."""
    torch.random.manual_seed(42)
    max_pos = 512
    head_dim = 512
    rope_dim = 64
    eps = 1e-6

    kv = torch.randn(batch_size, head_dim, dtype=dtype, device=device)
    kv_weight = torch.randn(head_dim, dtype=dtype, device=device)
    freqs_cis = torch.randn(
        max_pos, rope_dim // 2, dtype=torch.complex64, device=device
    )
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)
    positions = torch.randint(
        0, max_pos, (batch_size,), dtype=torch.int32, device=device
    )

    out_loc = torch.randperm(batch_size, dtype=torch.int32, device=device)
    if batch_size > 2:
        out_loc[0] = -1

    npages = (batch_size + page_size - 1) // page_size + 2
    k_page_bytes = page_size * 584
    kvcache_test = torch.zeros((npages, k_page_bytes), dtype=torch.uint8, device=device)
    kvcache_ref = torch.zeros_like(kvcache_test)

    sgl_kernel.fused_k_norm_rope_flashmla(
        kv, kv_weight, freqs_real, positions, out_loc, kvcache_test, eps, page_size
    )

    fused_k_norm_rope_flashmla_ref(
        kv, kv_weight, freqs_real, positions, out_loc, kvcache_ref, eps, page_size
    )

    # Note on tolerances for uint8 quantized byte comparisons:
    # kvcache stores FP8 quantized bytes and UE8M0 scale bytes.
    # Minor floating-point rounding differences between XPU C++ kernel and
    # PyTorch reference near quantization boundary thresholds can produce
    # 1-LSB uint8 integer offsets (e.g., 62 vs 63, delta = 1.0).
    torch.testing.assert_close(
        kvcache_test.float(),
        kvcache_ref.float(),
        rtol=1e-2,
        atol=1.0,
    )


@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
def test_fused_k_norm_rope_flashmla_position_dtype(position_dtype):
    """Test position tensor dtypes int32 and int64 for fused_k_norm_rope_flashmla."""
    torch.random.manual_seed(42)
    batch_size = 16
    page_size = 128
    max_pos = 512
    head_dim = 512
    rope_dim = 64
    eps = 1e-6
    dtype = torch.bfloat16

    kv = torch.randn(batch_size, head_dim, dtype=dtype, device=device)
    kv_weight = torch.randn(head_dim, dtype=dtype, device=device)
    freqs_cis = torch.randn(
        max_pos, rope_dim // 2, dtype=torch.complex64, device=device
    )
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)
    positions = torch.randint(
        0, max_pos, (batch_size,), dtype=position_dtype, device=device
    )

    out_loc = torch.randperm(batch_size, dtype=torch.int32, device=device)

    npages = (batch_size + page_size - 1) // page_size + 2
    k_page_bytes = page_size * 584
    kvcache_test = torch.zeros((npages, k_page_bytes), dtype=torch.uint8, device=device)
    kvcache_ref = torch.zeros_like(kvcache_test)

    sgl_kernel.fused_k_norm_rope_flashmla(
        kv, kv_weight, freqs_real, positions, out_loc, kvcache_test, eps, page_size
    )

    fused_k_norm_rope_flashmla_ref(
        kv, kv_weight, freqs_real, positions, out_loc, kvcache_ref, eps, page_size
    )

    # Note on tolerances for uint8 quantized byte comparisons:
    # kvcache stores FP8 quantized bytes and UE8M0 scale bytes.
    # Minor floating-point rounding differences between XPU C++ kernel and
    # PyTorch reference near quantization boundary thresholds can produce
    # 1-LSB uint8 integer offsets (e.g., 62 vs 63, delta = 1.0).
    torch.testing.assert_close(
        kvcache_test.float(),
        kvcache_ref.float(),
        rtol=1e-2,
        atol=1.0,
    )


def test_fused_k_norm_rope_flashmla_zero_batch():
    """Empty batch for fused_k_norm_rope_flashmla should not crash."""
    kv = torch.empty(0, 512, dtype=torch.bfloat16, device=device)
    kv_weight = torch.randn(512, dtype=torch.bfloat16, device=device)
    freqs_cis = torch.randn(512, 32, dtype=torch.complex64, device=device)
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)
    positions = torch.empty(0, dtype=torch.int32, device=device)
    out_loc = torch.empty(0, dtype=torch.int32, device=device)
    kvcache = torch.zeros((10, 584 * 16), dtype=torch.uint8, device=device)

    sgl_kernel.fused_k_norm_rope_flashmla(
        kv, kv_weight, freqs_real, positions, out_loc, kvcache, 1e-6, 16
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
