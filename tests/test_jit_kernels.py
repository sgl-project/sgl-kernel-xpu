"""
Tests for SYCL JIT kernel accuracy against AOT kernels.

These tests validate that JIT-compiled kernels produce the same results as
their AOT counterparts or PyTorch reference implementations.
"""

import pytest
import torch

try:
    import sgl_kernel

    HAS_SGL_KERNEL = True
except ImportError:
    HAS_SGL_KERNEL = False

try:
    from sgl_kernel.jit import apply_rope_inplace as jit_rope
    from sgl_kernel.jit import fused_inplace_qknorm as jit_qknorm
    from sgl_kernel.jit import (
        per_token_group_quant_8bit_v2 as jit_per_token_group_quant_8bit_v2,
    )
    from sgl_kernel.jit import rmsnorm as jit_rmsnorm
    from sgl_kernel.jit import timestep_embedding as jit_timestep_embedding

    HAS_SGLANG_JIT = True
except ImportError:
    HAS_SGLANG_JIT = False

HAS_XPU = hasattr(torch, "xpu") and torch.xpu.is_available()


# PyTorch reference implementations


def reference_qknorm(q, k, q_weight, k_weight, eps=1e-6):
    """PyTorch reference implementation of QKNorm."""
    q_rms = torch.sqrt(torch.mean(q**2, dim=-1, keepdim=True) + eps)
    k_rms = torch.sqrt(torch.mean(k**2, dim=-1, keepdim=True) + eps)
    q_out = (q / q_rms) * q_weight
    k_out = (k / k_rms) * k_weight
    return q_out, k_out


def reference_timestep_embedding(
    t: torch.Tensor,
    dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 0.0,
    scale: float = 1.0,
    max_period: int = 10000,
) -> torch.Tensor:
    """PyTorch reference implementation of Timestep Embedding."""
    half_dim = dim // 2

    # Compute frequency schedule
    freqs = torch.exp(
        -torch.log(torch.tensor(max_period, dtype=torch.float32, device=t.device))
        * torch.arange(0, half_dim, dtype=torch.float32, device=t.device)
        / (half_dim - downscale_freq_shift)
    )

    # Compute angles
    t_float = t.float().view(-1, 1)
    args = scale * t_float * freqs.view(1, -1)

    # Compute embeddings
    cos_emb = torch.cos(args)
    sin_emb = torch.sin(args)

    if flip_sin_to_cos:
        output = torch.cat([cos_emb, sin_emb], dim=-1)
    else:
        output = torch.cat([sin_emb, cos_emb], dim=-1)

    return output


@pytest.mark.skipif(not HAS_SGLANG_JIT, reason="Requires SGLang for JIT compilation")
@pytest.mark.skipif(not HAS_SGL_KERNEL, reason="Requires sgl_kernel for AOT comparison")
@pytest.mark.skipif(not HAS_XPU, reason="Requires XPU device")
def test_rmsnorm_jit_vs_aot():
    """Test RMSNorm JIT accuracy vs AOT for hidden_size=4096, dtype=float16."""
    device = "xpu"
    dtype = torch.float16
    hidden_size = 4096
    batch_size = 32
    eps = 1e-6

    x = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)

    # AOT kernel: sgl_kernel.rmsnorm(input, weight, eps, out=None) -> Tensor
    y_aot = sgl_kernel.rmsnorm(x.clone(), weight, eps)

    # JIT kernel (in-place operation, modifies out parameter)
    y_jit = torch.empty_like(x)
    jit_rmsnorm(x.clone(), weight, out=y_jit, eps=eps)

    # Compare accuracy
    torch.testing.assert_close(y_jit, y_aot, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not HAS_SGLANG_JIT, reason="Requires SGLang for JIT compilation")
@pytest.mark.skipif(not HAS_XPU, reason="Requires XPU device")
def test_qknorm_jit_vs_reference():
    """Test QKNorm JIT accuracy vs PyTorch reference for hidden_size=128, dtype=float16."""
    device = "xpu"
    dtype = torch.float16
    batch_size = 32
    seq_len = 64
    num_heads = 8
    head_dim = 128
    eps = 1e-6

    # Create input tensors (3D: batch_size, num_heads, head_dim)
    # Note: fused_inplace_qknorm expects 3D tensors
    q = torch.randn(
        batch_size * seq_len, num_heads, head_dim, dtype=dtype, device=device
    )
    k = torch.randn(
        batch_size * seq_len, num_heads, head_dim, dtype=dtype, device=device
    )
    q_weight = torch.randn(head_dim, dtype=dtype, device=device)
    k_weight = torch.randn(head_dim, dtype=dtype, device=device)

    # PyTorch reference (on 4D reshaped)
    q_4d = q.view(batch_size, seq_len, num_heads, head_dim)
    k_4d = k.view(batch_size, seq_len, num_heads, head_dim)
    q_ref, k_ref = reference_qknorm(q_4d, k_4d, q_weight, k_weight, eps)
    q_ref = q_ref.reshape(-1, num_heads, head_dim)
    k_ref = k_ref.reshape(-1, num_heads, head_dim)

    # JIT kernel (in-place operation)
    q_jit = q.clone().contiguous()
    k_jit = k.clone().contiguous()
    jit_qknorm(q_jit, k_jit, q_weight, k_weight, eps, head_dim=head_dim)

    # Compare accuracy
    torch.testing.assert_close(q_jit, q_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k_jit, k_ref, rtol=1e-2, atol=1e-2)


def reference_rope(q, k, cos_sin_cache, positions, is_neox, rope_dim):
    """PyTorch reference implementation of RoPE (both neox and GPT-J styles).

    cos_sin_cache: [max_pos, rope_dim] float32, first half cos, second half sin.
    q/k: [batch, num_heads, head_dim] - head_dim >= rope_dim.
    positions: [batch] int64.
    """
    half = rope_dim // 2
    cos = cos_sin_cache[positions, :half].float()  # [batch, half]
    sin = cos_sin_cache[positions, half:rope_dim].float()  # [batch, half]

    def apply(x):
        x_rot = x[..., :rope_dim].float()
        if is_neox:
            x1 = x_rot[..., :half]
            x2 = x_rot[..., half:]
            rotated = torch.cat([-x2, x1], dim=-1)
        else:
            x1 = x_rot[..., 0::2]
            x2 = x_rot[..., 1::2]
            rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
        cos_b = cos[:, None, :]  # [batch, 1, half]
        sin_b = sin[:, None, :]
        if is_neox:
            out = x_rot * torch.cat([cos_b, cos_b], dim=-1) + rotated * torch.cat(
                [sin_b, sin_b], dim=-1
            )
        else:
            cos_full = torch.stack([cos_b, cos_b], dim=-1).flatten(-2)
            sin_full = torch.stack([sin_b, sin_b], dim=-1).flatten(-2)
            out = x_rot * cos_full + rotated * sin_full
        return torch.cat([out.to(x.dtype), x[..., rope_dim:]], dim=-1)

    return apply(q), apply(k)


@pytest.mark.parametrize("is_neox", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(not HAS_SGLANG_JIT, reason="Requires SGLang for JIT compilation")
@pytest.mark.skipif(not HAS_XPU, reason="Requires XPU device")
def test_rope_jit_vs_reference(is_neox, dtype):
    """Test JIT RoPE accuracy vs PyTorch reference for both neox/GPT-J styles."""
    device = "xpu"
    batch_size = 16
    num_heads = 8
    head_dim = 128
    rope_dim = 128
    max_pos = 4096

    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, num_heads, head_dim, dtype=dtype, device=device)
    positions = torch.arange(batch_size, dtype=torch.int64, device=device)
    cos_sin_cache = torch.randn(max_pos, rope_dim, dtype=torch.float32, device=device)

    q_ref, k_ref = reference_rope(
        q.cpu().float(),
        k.cpu().float(),
        cos_sin_cache.cpu(),
        positions.cpu(),
        is_neox,
        rope_dim,
    )
    q_ref = q_ref.to(dtype=dtype, device=device)
    k_ref = k_ref.to(dtype=dtype, device=device)

    q_jit = q.clone()
    k_jit = k.clone()
    jit_rope(q_jit, k_jit, cos_sin_cache, positions, is_neox=is_neox, rope_dim=rope_dim)

    torch.testing.assert_close(q_jit, q_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k_jit, k_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not HAS_SGLANG_JIT, reason="Requires SGLang for JIT compilation")
@pytest.mark.skipif(not HAS_XPU, reason="Requires XPU device")
def test_timestep_embedding_jit_vs_reference():
    """Test Timestep Embedding JIT accuracy vs PyTorch reference for dim=256, dtype=float32."""
    device = "xpu"
    dtype = torch.float32
    batch_size = 128
    dim = 256

    timesteps = torch.randn(batch_size, dtype=dtype, device=device)

    # PyTorch reference with default parameters
    y_ref = reference_timestep_embedding(
        timesteps.clone(),
        dim,
        flip_sin_to_cos=False,
        downscale_freq_shift=0.0,
        scale=1.0,
        max_period=10000,
    )

    # JIT kernel with default parameters
    y_jit = jit_timestep_embedding(
        timesteps.clone(),
        dim,
        flip_sin_to_cos=False,
        downscale_freq_shift=0.0,
        scale=1.0,
        max_period=10000,
        dtype=dtype,
    )

    # Compare accuracy
    torch.testing.assert_close(y_jit, y_ref, rtol=1e-3, atol=1e-3)


def _make_ptgq_v2_scale(out_shape, group_size, column_major, device):
    ng = out_shape[-1] // group_size
    if column_major:
        return torch.empty(
            (ng,) + out_shape[:-1], device=device, dtype=torch.float32
        ).permute(-1, -2)
    return torch.empty(out_shape[:-1] + (ng,), device=device, dtype=torch.float32)


@pytest.mark.parametrize("group_size", [64, 128])
@pytest.mark.parametrize("column_major", [False, True])
@pytest.mark.parametrize("fuse_silu_and_mul", [False, True])
@pytest.mark.parametrize("out_dtype", [torch.float8_e4m3fn, torch.int8])
@pytest.mark.parametrize("in_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.skipif(not HAS_SGL_KERNEL, reason="Requires AOT sgl_kernel")
@pytest.mark.skipif(not HAS_SGLANG_JIT, reason="Requires SGLang for JIT compilation")
@pytest.mark.skipif(not HAS_XPU, reason="Requires XPU device")
def test_per_token_group_quant_8bit_v2_jit_vs_aot(
    group_size, column_major, fuse_silu_and_mul, out_dtype, in_dtype
):
    """JIT vs AOT per-token-group 8-bit v2 (vanilla / column-major / fused silu).

    Scales match closely; quant codes agree with AOT to within one ULP (int8) or
    one e4m3 step (fp8) -- both do the same reciprocal-multiply-then-clamp, so
    they differ only at quantization boundaries, as the AOT kernel differs from
    a pure-division reference.
    """
    # int8 output uses a row-major scale layout in this op.
    if column_major and out_dtype == torch.int8:
        pytest.skip("int8 path uses row-major scales")

    device = "xpu"
    hidden = 512
    num_tokens = 128
    eps = 1e-10
    min_8bit, max_8bit = (-128.0, 127.0) if out_dtype == torch.int8 else (-448.0, 448.0)

    torch.manual_seed(0)
    x = torch.randn(
        num_tokens,
        hidden * (2 if fuse_silu_and_mul else 1),
        dtype=in_dtype,
        device=device,
    )
    out_shape = (num_tokens, hidden)

    def _run(use_aot):
        q = torch.empty(out_shape, device=device, dtype=out_dtype)
        s = _make_ptgq_v2_scale(out_shape, group_size, column_major, device)
        if use_aot:
            torch.ops.sgl_kernel.sgl_per_token_group_quant_8bit_v2.default(
                x,
                q,
                s,
                group_size,
                eps,
                min_8bit,
                max_8bit,
                False,
                fuse_silu_and_mul,
                None,
            )
        else:
            jit_per_token_group_quant_8bit_v2(
                x,
                q,
                s,
                group_size,
                eps,
                min_8bit,
                max_8bit,
                scale_ue8m0=False,
                fuse_silu_and_mul=fuse_silu_and_mul,
                masked_m=None,
            )
        return q, s

    q_aot, s_aot = _run(True)
    q_jit, s_jit = _run(False)

    # Scales match tightly.
    torch.testing.assert_close(
        s_jit.float().cpu(), s_aot.float().cpu(), rtol=1e-3, atol=1e-5
    )

    # Quant codes: bounded by one ULP (int8) / one e4m3 step (fp8).
    if out_dtype == torch.int8:
        max_ulp = (q_jit.to(torch.int32) - q_aot.to(torch.int32)).abs().max().item()
        assert max_ulp <= 1, f"int8 quant differs from AOT by {max_ulp} > 1 ULP"
    else:
        ng = hidden // group_size
        dq = lambda q, s: (
            q.cpu().view(num_tokens, ng, group_size).to(torch.float32)
            * s.float().cpu().reshape(num_tokens, ng, 1)
        )
        torch.testing.assert_close(
            dq(q_jit, s_jit), dq(q_aot, s_aot), rtol=1e-1, atol=1e-1
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
