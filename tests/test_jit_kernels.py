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
    from sgl_kernel.jit import moe_align_block_size as jit_moe_align_block_size
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


def _run_moe_align(use_aot, topk_ids, num_experts, block_size, pad):
    """Allocate outputs and run the AOT or JIT moe_align_block_size op.

    num_experts here is the real expert count; callers pass num_experts + 1 to
    the op (the +1 offset bucket convention).
    """
    ne1 = num_experts + 1
    numel = topk_ids.numel()
    max_pad = numel + ne1 * (block_size - 1)
    sorted_ids = torch.empty(max_pad, dtype=torch.int32, device=topk_ids.device)
    if not pad:
        sorted_ids.fill_(numel)
    expert_ids = torch.zeros(
        max_pad // block_size, dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty(1, dtype=torch.int32, device=topk_ids.device)
    cumsum = torch.zeros(ne1 + 1, dtype=torch.int32, device=topk_ids.device)

    if use_aot:
        sgl_kernel.moe_align_block_size(
            topk_ids,
            ne1,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            cumsum,
            pad,
        )
    else:
        jit_moe_align_block_size(
            topk_ids,
            ne1,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            cumsum,
            pad,
        )
    return sorted_ids, expert_ids, num_tokens_post_pad, cumsum


@pytest.mark.parametrize("block_size", [32, 128])
@pytest.mark.parametrize(
    "num_tokens,num_experts",
    [
        (8, 64),  # small-batch path (numel < 1024, experts <= 64)
        (64, 64),  # small-batch path
        (512, 160),  # general Blelloch-scan path
        (2048, 256),  # general path, larger
    ],
)
@pytest.mark.parametrize("topk", [2, 8])
@pytest.mark.parametrize("pad_sorted_token_ids", [False, True])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.skipif(not HAS_SGL_KERNEL, reason="Requires AOT sgl_kernel")
@pytest.mark.skipif(not HAS_SGLANG_JIT, reason="Requires SGLang for JIT compilation")
@pytest.mark.skipif(not HAS_XPU, reason="Requires XPU device")
def test_moe_align_block_size_jit_vs_aot(
    block_size, num_tokens, num_experts, topk, pad_sorted_token_ids, dtype
):
    """Test JIT moe_align_block_size vs the AOT op across both code paths.

    expert_ids / num_tokens_post_pad / cumsum must match bit-exactly; the
    sorted_token_ids order within an expert bucket is nondeterministic (atomic
    scatter), so it is compared as the set of placed token ids.
    """
    device = "xpu"
    torch.manual_seed(num_tokens + num_experts + topk)
    topk_ids = (
        torch.argsort(torch.rand(num_tokens, num_experts, device=device), dim=1)[
            :, :topk
        ]
        .to(dtype)
        .contiguous()
    )
    numel = topk_ids.numel()

    s_aot, e_aot, n_aot, c_aot = _run_moe_align(
        True, topk_ids, num_experts, block_size, pad_sorted_token_ids
    )
    s_jit, e_jit, n_jit, c_jit = _run_moe_align(
        False, topk_ids, num_experts, block_size, pad_sorted_token_ids
    )

    assert torch.equal(e_jit, e_aot), "expert_ids mismatch"
    assert torch.equal(n_jit, n_aot), "num_tokens_post_pad mismatch"
    assert torch.equal(c_jit, c_aot), "cumsum mismatch"

    # Compare placed token ids as a set (order within a bucket is racy).
    def _placed(sorted_ids, ntpp):
        v = sorted_ids[: int(ntpp.item())]
        return torch.sort(v[v != numel]).values

    assert torch.equal(
        _placed(s_jit, n_jit), _placed(s_aot, n_aot)
    ), "sorted_token_ids placed-set mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
