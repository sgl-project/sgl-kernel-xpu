"""
Test suite for apply_rope_inplace_with_kvcache XPU kernel.
Tests the fused RoPE + KV-cache write operation.
"""

import sys

import pytest
import torch
import triton
import utils
from sgl_kernel import apply_rope_inplace_with_kvcache

DEVICE = utils.get_device()
MAX_SEQ_LEN = 131072
ROPE_BASE = 10000.0


def create_cos_sin_cache(
    rotary_dim: int,
    max_position: int = MAX_SEQ_LEN,
    base: float = ROPE_BASE,
) -> torch.Tensor:
    """Create cos/sin cache compatible with SGLang layout: [max_pos, rotary_dim]."""
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=DEVICE)
            / rotary_dim
        )
    )
    t = torch.arange(max_position, dtype=torch.float32, device=DEVICE)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    cache = torch.cat((cos, sin), dim=-1)  # [max_pos, rotary_dim]
    return cache


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    """
    Reference RoPE that mirrors the XPU kernel's fp32 accumulation: it loads cos/sin
    as fp32, casts the bf16/fp16 input up to fp32 for the multiply, and rounds back
    only at write-time. Downcasting cos/sin first would silently drop ~1 ulp on the
    odd element and make atol=1e-2 flaky.
    """
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    x1f, x2f = x1.float(), x2.float()
    o1 = (x1f * cos - x2f * sin).to(x.dtype)
    o2 = (x2f * cos + x1f * sin).to(x.dtype)
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)


def reference_rope_with_kvcache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    out_loc: torch.Tensor,
    rot_dim: int,
    is_neox: bool,
) -> tuple:
    """
    Reference implementation: separate RoPE + cache write.
    Returns: (q_out, k_out, k_cache_out, v_cache_out)
    """
    num_tokens = q.shape[0]
    head_dim = q.shape[-1]
    num_kv_heads = k.shape[1]

    positions = positions.flatten()
    cos_cache, sin_cache = cos_sin_cache.chunk(2, dim=-1)
    cos = cos_cache[positions]
    sin = sin_cache[positions]

    # Apply RoPE to Q
    q_out = q.clone()
    q_rot = q_out[..., :rot_dim]
    q_rot = apply_rotary_emb(q_rot, cos, sin, is_neox)
    q_out[..., :rot_dim] = q_rot

    # Apply RoPE to K
    k_out = k.clone()
    k_rot = k_out[..., :rot_dim]
    k_rot = apply_rotary_emb(k_rot, cos, sin, is_neox)
    k_out[..., :rot_dim] = k_rot

    # Write K and V to cache
    k_cache_out = k_cache.clone()
    v_cache_out = v_cache.clone()

    for i in range(num_tokens):
        idx = out_loc[i].item()
        if idx >= 0:  # Skip -1 indices (speculative decoding)
            k_flat = k_out[i].reshape(-1)  # [num_kv_heads * head_dim]
            v_flat = v[i].reshape(-1)
            k_cache_out[idx] = k_flat
            v_cache_out[idx] = v_flat

    return q_out, k_out, k_cache_out, v_cache_out


# Test parameters
BATCH_SIZE_LIST = [1, 16, 128, 2048]
NUM_Q_HEADS_LIST = [8, 16, 32]
NUM_KV_HEADS_LIST = [1, 2, 4, 8]
HEAD_DIM_LIST = [64, 96, 128, 256]
IS_NEOX_LIST = [False, True]
DTYPE_LIST = [torch.bfloat16, torch.float16]


@pytest.mark.parametrize("batch_size", BATCH_SIZE_LIST)
@pytest.mark.parametrize("num_q_heads", [8, 32])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("is_neox", IS_NEOX_LIST)
@pytest.mark.parametrize("dtype", DTYPE_LIST)
def test_fused_rope_kvcache_correctness(
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    is_neox: bool,
    dtype: torch.dtype,
) -> None:
    """Test correctness against reference implementation."""
    rot_dim = head_dim
    cache_size = 4096

    # Create inputs
    q = torch.randn(batch_size, num_q_heads, head_dim, device=DEVICE, dtype=dtype)
    k = torch.randn(batch_size, num_kv_heads, head_dim, device=DEVICE, dtype=dtype)
    v = torch.randn(batch_size, num_kv_heads, head_dim, device=DEVICE, dtype=dtype)

    # Create caches
    k_cache = torch.zeros(
        cache_size, num_kv_heads * head_dim, device=DEVICE, dtype=dtype
    )
    v_cache = torch.zeros(
        cache_size, num_kv_heads * head_dim, device=DEVICE, dtype=dtype
    )

    # Create cos/sin cache (must be float32)
    cos_sin_cache = create_cos_sin_cache(rot_dim).float()

    positions = torch.randint(0, MAX_SEQ_LEN, (batch_size,), device=DEVICE)
    # Unique slots: avoids races between the kernel's parallel writes and
    # the reference's serial scatter when batch_size > distinct-slots.
    out_loc = torch.randperm(cache_size, device=DEVICE, dtype=torch.int64)[:batch_size]

    # Clone for kernel test
    q_ker = q.clone()
    k_ker = k.clone()
    v_ker = v.clone()
    k_cache_ker = k_cache.clone()
    v_cache_ker = v_cache.clone()

    # Reference implementation
    q_ref, k_ref, k_cache_ref, v_cache_ref = reference_rope_with_kvcache(
        q, k, v, k_cache, v_cache, cos_sin_cache, positions, out_loc, rot_dim, is_neox
    )

    # Kernel implementation
    apply_rope_inplace_with_kvcache(
        q_ker,
        k_ker,
        v_ker,
        k_cache_ker,
        v_cache_ker,
        cos_sin_cache,
        positions,
        out_loc,
        is_neox,
    )

    # Verify results
    atol = rtol = 1e-2
    triton.testing.assert_close(q_ref, q_ker, atol=atol, rtol=rtol)
    triton.testing.assert_close(k_ref, k_ker, atol=atol, rtol=rtol)
    triton.testing.assert_close(v_ker, v, atol=atol, rtol=rtol)  # V should be unchanged
    triton.testing.assert_close(k_cache_ref, k_cache_ker, atol=atol, rtol=rtol)
    triton.testing.assert_close(v_cache_ref, v_cache_ker, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", DTYPE_LIST)
def test_partial_rotary_dim(dtype: torch.dtype) -> None:
    """Test when rot_dim < head_dim."""
    batch_size, num_q_heads, num_kv_heads = 16, 8, 2
    head_dim = 128
    rot_dim = 64  # Only rotate first half
    cache_size = 1024
    is_neox = True

    q = torch.randn(batch_size, num_q_heads, head_dim, device=DEVICE, dtype=dtype)
    k = torch.randn(batch_size, num_kv_heads, head_dim, device=DEVICE, dtype=dtype)
    v = torch.randn(batch_size, num_kv_heads, head_dim, device=DEVICE, dtype=dtype)

    k_cache = torch.zeros(
        cache_size, num_kv_heads * head_dim, device=DEVICE, dtype=dtype
    )
    v_cache = torch.zeros(
        cache_size, num_kv_heads * head_dim, device=DEVICE, dtype=dtype
    )

    cos_sin_cache = create_cos_sin_cache(rot_dim).float()
    positions = torch.randint(0, MAX_SEQ_LEN, (batch_size,), device=DEVICE)
    out_loc = torch.randperm(cache_size, device=DEVICE, dtype=torch.int64)[:batch_size]

    # Clone for comparison
    q_ker = q.clone()
    k_ker = k.clone()
    v_ker = v.clone()
    k_cache_ker = k_cache.clone()
    v_cache_ker = v_cache.clone()

    # Reference
    q_ref, k_ref, k_cache_ref, v_cache_ref = reference_rope_with_kvcache(
        q, k, v, k_cache, v_cache, cos_sin_cache, positions, out_loc, rot_dim, is_neox
    )

    # Kernel
    apply_rope_inplace_with_kvcache(
        q_ker,
        k_ker,
        v_ker,
        k_cache_ker,
        v_cache_ker,
        cos_sin_cache,
        positions,
        out_loc,
        is_neox,
    )

    atol = rtol = 1e-2
    triton.testing.assert_close(q_ref, q_ker, atol=atol, rtol=rtol)
    triton.testing.assert_close(k_ref, k_ker, atol=atol, rtol=rtol)
    triton.testing.assert_close(k_cache_ref, k_cache_ker, atol=atol, rtol=rtol)
    triton.testing.assert_close(v_cache_ref, v_cache_ker, atol=atol, rtol=rtol)


def test_speculative_decoding_skip() -> None:
    """Tokens with out_loc < 0 must skip both the cache write AND the in-place RoPE."""
    batch_size, num_q_heads, num_kv_heads, head_dim = 8, 8, 2, 64
    cache_size = 256
    is_neox = True
    dtype = torch.bfloat16

    q = torch.randn(batch_size, num_q_heads, head_dim, device=DEVICE, dtype=dtype)
    k = torch.randn(batch_size, num_kv_heads, head_dim, device=DEVICE, dtype=dtype)
    v = torch.randn(batch_size, num_kv_heads, head_dim, device=DEVICE, dtype=dtype)

    # Distinct slots so we can match writes to tokens unambiguously.
    out_loc = torch.arange(batch_size, device=DEVICE, dtype=torch.int64)
    skip = torch.zeros(batch_size, dtype=torch.bool, device=DEVICE)
    skip[::2] = True
    out_loc[skip] = -1

    cos_sin_cache = create_cos_sin_cache(head_dim).float()
    positions = torch.randint(0, MAX_SEQ_LEN, (batch_size,), device=DEVICE)

    # Sentinel: any slot the kernel touches will diverge from this.
    sentinel = 7.0
    k_cache = torch.full(
        (cache_size, num_kv_heads * head_dim), sentinel, device=DEVICE, dtype=dtype
    )
    v_cache = torch.full(
        (cache_size, num_kv_heads * head_dim), sentinel, device=DEVICE, dtype=dtype
    )

    q_in, k_in, v_in = q.clone(), k.clone(), v.clone()
    apply_rope_inplace_with_kvcache(
        q_in,
        k_in,
        v_in,
        k_cache,
        v_cache,
        cos_sin_cache,
        positions,
        out_loc,
        is_neox,
    )

    # In-place RoPE must NOT touch q/k for skipped tokens.
    assert torch.equal(q_in[skip], q[skip]), "Q rotated on a skipped token"
    assert torch.equal(k_in[skip], k[skip]), "K rotated on a skipped token"

    # Cache rows pointed to by valid out_loc must have been overwritten;
    # all other rows must still equal the sentinel.
    untouched = torch.ones(cache_size, dtype=torch.bool, device=DEVICE)
    untouched[out_loc[~skip]] = False
    assert torch.all(k_cache[untouched] == sentinel), "k_cache touched on skipped slot"
    assert torch.all(v_cache[untouched] == sentinel), "v_cache touched on skipped slot"
    assert torch.all(
        k_cache[out_loc[~skip]] != sentinel
    ), "k_cache not written for valid slot"
    assert torch.all(
        v_cache[out_loc[~skip]] != sentinel
    ), "v_cache not written for valid slot"


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_position_dtypes(dtype: torch.dtype) -> None:
    """Test both int32 and int64 position tensors."""
    batch_size, num_q_heads, num_kv_heads, head_dim = 64, 16, 2, 128
    cache_size = 512
    is_neox = True
    tensor_dtype = torch.bfloat16

    q = torch.randn(
        batch_size, num_q_heads, head_dim, device=DEVICE, dtype=tensor_dtype
    )
    k = torch.randn(
        batch_size, num_kv_heads, head_dim, device=DEVICE, dtype=tensor_dtype
    )
    v = torch.randn(
        batch_size, num_kv_heads, head_dim, device=DEVICE, dtype=tensor_dtype
    )

    k_cache = torch.zeros(
        cache_size, num_kv_heads * head_dim, device=DEVICE, dtype=tensor_dtype
    )
    v_cache = torch.zeros(
        cache_size, num_kv_heads * head_dim, device=DEVICE, dtype=tensor_dtype
    )

    cos_sin_cache = create_cos_sin_cache(head_dim).float()
    positions = torch.randint(0, MAX_SEQ_LEN, (batch_size,), device=DEVICE, dtype=dtype)
    out_loc = torch.randperm(cache_size, device=DEVICE, dtype=torch.int64)[
        :batch_size
    ].to(dtype)

    # Clone for reference
    q_ker = q.clone()
    k_ker = k.clone()
    v_ker = v.clone()
    k_cache_ker = k_cache.clone()
    v_cache_ker = v_cache.clone()

    # Reference
    q_ref, k_ref, k_cache_ref, v_cache_ref = reference_rope_with_kvcache(
        q, k, v, k_cache, v_cache, cos_sin_cache, positions, out_loc, head_dim, is_neox
    )

    # Kernel
    apply_rope_inplace_with_kvcache(
        q_ker,
        k_ker,
        v_ker,
        k_cache_ker,
        v_cache_ker,
        cos_sin_cache,
        positions,
        out_loc,
        is_neox,
    )

    atol = rtol = 1e-2
    triton.testing.assert_close(q_ref, q_ker, atol=atol, rtol=rtol)
    triton.testing.assert_close(k_ref, k_ker, atol=atol, rtol=rtol)
    triton.testing.assert_close(k_cache_ref, k_cache_ker, atol=atol, rtol=rtol)


def test_single_token() -> None:
    """Test with single token (batch_size=1)."""
    batch_size, num_q_heads, num_kv_heads, head_dim = 1, 8, 2, 64
    cache_size = 128
    is_neox = True
    dtype = torch.bfloat16

    q = torch.randn(batch_size, num_q_heads, head_dim, device=DEVICE, dtype=dtype)
    k = torch.randn(batch_size, num_kv_heads, head_dim, device=DEVICE, dtype=dtype)
    v = torch.randn(batch_size, num_kv_heads, head_dim, device=DEVICE, dtype=dtype)

    k_cache = torch.zeros(
        cache_size, num_kv_heads * head_dim, device=DEVICE, dtype=dtype
    )
    v_cache = torch.zeros(
        cache_size, num_kv_heads * head_dim, device=DEVICE, dtype=dtype
    )

    cos_sin_cache = create_cos_sin_cache(head_dim).float()
    positions = torch.tensor([42], device=DEVICE)
    out_loc = torch.tensor([10], device=DEVICE)

    # Should not crash
    apply_rope_inplace_with_kvcache(
        q, k, v, k_cache, v_cache, cos_sin_cache, positions, out_loc, is_neox
    )

    # Verify cache was written
    assert not torch.allclose(k_cache[10], torch.zeros_like(k_cache[10]))
    assert not torch.allclose(v_cache[10], torch.zeros_like(v_cache[10]))


def test_non_float32_cos_sin_cache() -> None:
    """Test that non-float32 cos_sin_cache raises an error."""
    batch_size, num_q_heads, num_kv_heads, head_dim = 4, 8, 2, 64
    cache_size = 128
    dtype = torch.bfloat16

    q = torch.randn(batch_size, num_q_heads, head_dim, device=DEVICE, dtype=dtype)
    k = torch.randn(batch_size, num_kv_heads, head_dim, device=DEVICE, dtype=dtype)
    v = torch.randn(batch_size, num_kv_heads, head_dim, device=DEVICE, dtype=dtype)

    k_cache = torch.zeros(
        cache_size, num_kv_heads * head_dim, device=DEVICE, dtype=dtype
    )
    v_cache = torch.zeros(
        cache_size, num_kv_heads * head_dim, device=DEVICE, dtype=dtype
    )

    # Create non-float32 cache (should fail)
    cos_sin_cache = create_cos_sin_cache(head_dim).to(dtype)
    positions = torch.randint(0, MAX_SEQ_LEN, (batch_size,), device=DEVICE)
    out_loc = torch.randint(0, cache_size, (batch_size,), device=DEVICE)

    with pytest.raises((ValueError, RuntimeError)):
        apply_rope_inplace_with_kvcache(
            q, k, v, k_cache, v_cache, cos_sin_cache, positions, out_loc, True
        )


def test_odd_rot_dim_rejected() -> None:
    """rot_dim must be even — kernel pairs (i, i+rot_dim/2) or (2i, 2i+1)."""
    batch_size, num_q_heads, num_kv_heads, head_dim = 4, 8, 2, 64
    cache_size = 128
    dtype = torch.bfloat16

    q = torch.randn(batch_size, num_q_heads, head_dim, device=DEVICE, dtype=dtype)
    k = torch.randn(batch_size, num_kv_heads, head_dim, device=DEVICE, dtype=dtype)
    v = torch.randn(batch_size, num_kv_heads, head_dim, device=DEVICE, dtype=dtype)
    k_cache = torch.zeros(
        cache_size, num_kv_heads * head_dim, device=DEVICE, dtype=dtype
    )
    v_cache = torch.zeros(
        cache_size, num_kv_heads * head_dim, device=DEVICE, dtype=dtype
    )
    # cos_sin_cache last dim is rot_dim; pick an odd one.
    cos_sin_cache = torch.zeros(MAX_SEQ_LEN, 17, device=DEVICE, dtype=torch.float32)
    positions = torch.zeros(batch_size, device=DEVICE, dtype=torch.int64)
    out_loc = torch.arange(batch_size, device=DEVICE, dtype=torch.int64)

    with pytest.raises(RuntimeError, match="rot_dim"):
        apply_rope_inplace_with_kvcache(
            q, k, v, k_cache, v_cache, cos_sin_cache, positions, out_loc, True
        )


def test_rot_dim_exceeds_head_dim_rejected() -> None:
    """rot_dim > head_dim must be rejected — would index past the row."""
    batch_size, num_q_heads, num_kv_heads, head_dim = 4, 8, 2, 64
    cache_size = 128
    dtype = torch.bfloat16

    q = torch.randn(batch_size, num_q_heads, head_dim, device=DEVICE, dtype=dtype)
    k = torch.randn(batch_size, num_kv_heads, head_dim, device=DEVICE, dtype=dtype)
    v = torch.randn(batch_size, num_kv_heads, head_dim, device=DEVICE, dtype=dtype)
    k_cache = torch.zeros(
        cache_size, num_kv_heads * head_dim, device=DEVICE, dtype=dtype
    )
    v_cache = torch.zeros(
        cache_size, num_kv_heads * head_dim, device=DEVICE, dtype=dtype
    )
    cos_sin_cache = create_cos_sin_cache(head_dim * 2).float()  # rot_dim = 128 > 64
    positions = torch.zeros(batch_size, device=DEVICE, dtype=torch.int64)
    out_loc = torch.arange(batch_size, device=DEVICE, dtype=torch.int64)

    with pytest.raises(RuntimeError, match="rot_dim"):
        apply_rope_inplace_with_kvcache(
            q, k, v, k_cache, v_cache, cos_sin_cache, positions, out_loc, True
        )


def test_mqa_configuration() -> None:
    """Test Multi-Query Attention (MQA) with num_kv_heads=1."""
    batch_size, num_q_heads, num_kv_heads, head_dim = 32, 32, 1, 128
    cache_size = 1024
    is_neox = True
    dtype = torch.bfloat16

    q = torch.randn(batch_size, num_q_heads, head_dim, device=DEVICE, dtype=dtype)
    k = torch.randn(batch_size, num_kv_heads, head_dim, device=DEVICE, dtype=dtype)
    v = torch.randn(batch_size, num_kv_heads, head_dim, device=DEVICE, dtype=dtype)

    k_cache = torch.zeros(
        cache_size, num_kv_heads * head_dim, device=DEVICE, dtype=dtype
    )
    v_cache = torch.zeros(
        cache_size, num_kv_heads * head_dim, device=DEVICE, dtype=dtype
    )

    cos_sin_cache = create_cos_sin_cache(head_dim).float()
    positions = torch.randint(0, MAX_SEQ_LEN, (batch_size,), device=DEVICE)
    out_loc = torch.randperm(cache_size, device=DEVICE, dtype=torch.int64)[:batch_size]

    # Clone for reference
    q_ker = q.clone()
    k_ker = k.clone()
    v_ker = v.clone()
    k_cache_ker = k_cache.clone()
    v_cache_ker = v_cache.clone()

    # Reference
    q_ref, k_ref, k_cache_ref, v_cache_ref = reference_rope_with_kvcache(
        q, k, v, k_cache, v_cache, cos_sin_cache, positions, out_loc, head_dim, is_neox
    )

    # Kernel
    apply_rope_inplace_with_kvcache(
        q_ker,
        k_ker,
        v_ker,
        k_cache_ker,
        v_cache_ker,
        cos_sin_cache,
        positions,
        out_loc,
        is_neox,
    )

    atol = rtol = 1e-2
    triton.testing.assert_close(q_ref, q_ker, atol=atol, rtol=rtol)
    triton.testing.assert_close(k_ref, k_ker, atol=atol, rtol=rtol)
    triton.testing.assert_close(k_cache_ref, k_cache_ker, atol=atol, rtol=rtol)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
