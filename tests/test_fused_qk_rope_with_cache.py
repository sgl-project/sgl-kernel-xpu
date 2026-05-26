import sys

import pytest
import torch
import triton
import utils
from sgl_kernel import fused_qk_rope_with_cos_sin_cache_inplace

DEVICE = utils.get_device()
DTYPE = torch.bfloat16
MAX_SEQ_LEN = 131072  # common seq length
ROPE_BASE = 10000.0
CACHE_SIZE = 1024 * 128


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
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
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


def torch_impl_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    rotary_dim: int,
    is_neox: bool,
):
    head_size = q.shape[-1]
    positions = positions.flatten()
    num_tokens = positions.shape[0]
    assert rotary_dim == cos_sin_cache.size(-1), (
        f"rotary_dim ({rotary_dim}) must match cos/sin cache rotary width "
        f"({cos_cache.size(-1)})"
    )
    cos_cache, sin_cache = cos_sin_cache.chunk(2, dim=-1)
    cos = cos_cache[positions]
    sin = sin_cache[positions]

    query_shape = q.shape
    query = q.view(num_tokens, -1, head_size)
    query_rot = query[..., :rotary_dim]
    query_pass = query[..., rotary_dim:]
    query_rot = apply_rotary_emb(query_rot, cos, sin, is_neox)
    query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

    key_shape = k.shape
    key = k.view(num_tokens, -1, head_size)
    key_rot = key[..., :rotary_dim]
    key_pass = key[..., rotary_dim:]
    key_rot = apply_rotary_emb(key_rot, cos, sin, is_neox)
    key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
    return query, key


def fused_qk_rope_with_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    rotary_dim: int,
    is_neox: bool,
):
    return fused_qk_rope_with_cos_sin_cache_inplace(
        q, k, cos_sin_cache, positions, rotary_dim, is_neox
    )


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

BS_LIST = [1, 128, 2048]
NUM_KV_HEADS_LIST = [1, 4]
GQA_RATIO = [1, 8]
ROPE_DIM_LIST = [64, 256]
IS_NEOX_LIST = [False, True]
DTYPE_LIST = [torch.bfloat16, torch.float16]
PARTIAL_ROPE_DIM_LIST = [64, 96]
HEAD_DIM_LIST = [64, 256]


@pytest.mark.parametrize("batch_size", BS_LIST)
@pytest.mark.parametrize("gqa_ratio", GQA_RATIO)
@pytest.mark.parametrize("num_kv_heads", NUM_KV_HEADS_LIST)
@pytest.mark.parametrize("rope_dim", ROPE_DIM_LIST)
@pytest.mark.parametrize("is_neox", IS_NEOX_LIST)
@pytest.mark.parametrize("dtype", DTYPE_LIST)
def test_rope(
    batch_size: int,
    gqa_ratio: int,
    num_kv_heads: int,
    rope_dim: int,
    is_neox: bool,
    dtype: torch.dtype,
) -> None:
    num_qo_heads = num_kv_heads * gqa_ratio
    q = torch.randn(batch_size, num_qo_heads, rope_dim, device=DEVICE, dtype=dtype)
    k = torch.randn(batch_size, num_kv_heads, rope_dim, device=DEVICE, dtype=dtype)
    positions = torch.randint(
        0, MAX_SEQ_LEN, (batch_size,), device=DEVICE, dtype=torch.int64
    )
    cos_sin_cache = create_cos_sin_cache(rope_dim).to(dtype)

    q_ker, k_ker = q.clone(), k.clone()
    q_na, k_na = torch_impl_rope(q, k, cos_sin_cache, positions, rope_dim, is_neox)
    fused_qk_rope_with_cos_sin_cache_inplace(
        q_ker, k_ker, cos_sin_cache, positions, rope_dim, is_neox
    )

    atol = rtol = 1e-2
    triton.testing.assert_close(q_na, q_ker, atol=atol, rtol=rtol)
    triton.testing.assert_close(k_na, k_ker, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_rope_position_dtypes(dtype: torch.dtype) -> None:
    """Ensure both int32 and int64 position tensors work correctly."""
    batch_size, num_qo_heads, num_kv_heads, rope_dim = 16384, 16, 2, 128
    is_neox = True

    q = torch.randn(batch_size, num_qo_heads, rope_dim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch_size, num_kv_heads, rope_dim, device=DEVICE, dtype=DTYPE)
    positions = torch.randint(0, MAX_SEQ_LEN, (batch_size,), device=DEVICE, dtype=dtype)
    cos_sin_cache = create_cos_sin_cache(rope_dim).to(DTYPE)

    q_ker, k_ker = q.clone(), k.clone()
    q_na, k_na = torch_impl_rope(q, k, cos_sin_cache, positions, rope_dim, is_neox)
    fused_qk_rope_with_cos_sin_cache_inplace(
        q_ker, k_ker, cos_sin_cache, positions, rope_dim, is_neox
    )
    atol = rtol = 1e-2
    triton.testing.assert_close(q_na, q_ker, atol=atol, rtol=rtol)
    triton.testing.assert_close(k_na, k_ker, atol=atol, rtol=rtol)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
