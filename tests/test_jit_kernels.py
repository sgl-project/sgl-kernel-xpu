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
    from sglang.jit_kernel.norm import rmsnorm as jit_rmsnorm
    from sglang.jit_kernel.norm import fused_inplace_qknorm as jit_qknorm
    from sglang.jit_kernel.timestep_embedding import timestep_embedding as jit_timestep_embedding
    HAS_SGLANG_JIT = True
except ImportError:
    HAS_SGLANG_JIT = False


# PyTorch reference implementations

def reference_qknorm(q, k, q_weight, k_weight, eps=1e-6):
    """PyTorch reference implementation of QKNorm."""
    q_rms = torch.sqrt(torch.mean(q ** 2, dim=-1, keepdim=True) + eps)
    k_rms = torch.sqrt(torch.mean(k ** 2, dim=-1, keepdim=True) + eps)
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
@pytest.mark.skipif(not torch.xpu.is_available(), reason="Requires XPU device")
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
@pytest.mark.skipif(not torch.xpu.is_available(), reason="Requires XPU device")
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
    q = torch.randn(batch_size * seq_len, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size * seq_len, num_heads, head_dim, dtype=dtype, device=device)
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


@pytest.mark.skipif(not HAS_SGLANG_JIT, reason="Requires SGLang for JIT compilation")
@pytest.mark.skipif(not torch.xpu.is_available(), reason="Requires XPU device")
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
