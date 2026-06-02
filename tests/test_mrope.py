import sys
from typing import List

import pytest
import triton
from sgl_kernel import multomodal_rotary_embedding
from test_rope_utils import *


def apply_interleaved_rope(x: torch.Tensor, mrope_section: list) -> torch.Tensor:
    x_t = x[0].clone()
    x_t[..., 1 : mrope_section[1] * 3 : 3] = x[1, ..., 1 : mrope_section[1] * 3 : 3]
    x_t[..., 2 : mrope_section[2] * 3 : 3] = x[2, ..., 2 : mrope_section[2] * 3 : 3]
    return x_t


def torch_impl_mrope(
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    mrope_section: List[int],
    head_size: int,
    rotary_dim: int,
    mrope_interleaved: bool,
    is_neox_style: bool,
):
    assert positions.ndim == 1 or positions.ndim == 2

    cos_sin = cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    if positions.ndim == 2:
        if mrope_interleaved:
            cos = apply_interleaved_rope(cos, mrope_section)
            sin = apply_interleaved_rope(sin, mrope_section)
        else:
            cos = torch.cat(
                [m[i] for i, m in enumerate(cos.split(mrope_section, dim=-1))],
                dim=-1,
            )
            sin = torch.cat(
                [m[i] for i, m in enumerate(sin.split(mrope_section, dim=-1))],
                dim=-1,
            )

    seq_len_q = query.shape[0]
    query_shape = query.shape
    query = query.view(seq_len_q, -1, head_size)
    query_rot = query[..., :rotary_dim]
    query_pass = query[..., rotary_dim:]
    query_rot = apply_rotary_emb(query_rot, cos, sin, is_neox_style)
    query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

    seq_len_k = key.shape[0]
    key_shape = key.shape

    key = key.view(seq_len_k, -1, head_size)
    key_rot = key[..., :rotary_dim]
    key_pass = key[..., rotary_dim:]
    key_rot = apply_rotary_emb(key_rot, cos, sin, is_neox_style)
    key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
    return query, key


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------
BS_LIST = [1, 128, 2048]
NUM_KV_HEADS_LIST = [1, 4]
GQA_RATIO = [4]
HEAD_DIM_LIST = [256]
PARTIAL_ROTARY_FACTOR = [0.25]
ROTARY_DIM_LIST = [64]
IS_NEOX_LIST = [False, True]
MROPE_IS_INTERLEAVED = [False, True]
MROPE_SECTION_LIST = [[11, 11, 10]]
DTYPE_LIST = [torch.bfloat16, torch.float16]


@pytest.mark.parametrize("batch_size", BS_LIST)
@pytest.mark.parametrize("gqa_ratio", GQA_RATIO)
@pytest.mark.parametrize("num_kv_heads", NUM_KV_HEADS_LIST)
@pytest.mark.parametrize("head_size", HEAD_DIM_LIST)
@pytest.mark.parametrize("partial_rotary_factor", PARTIAL_ROTARY_FACTOR)
@pytest.mark.parametrize("mrope_is_interleaved", MROPE_IS_INTERLEAVED)
@pytest.mark.parametrize("is_neox", IS_NEOX_LIST)
@pytest.mark.parametrize("mrope_section", MROPE_SECTION_LIST)
@pytest.mark.parametrize("dtype", DTYPE_LIST)
def test_multomodal_rotary_embedding(
    batch_size: int,
    gqa_ratio: int,
    num_kv_heads: int,
    head_size: int,
    partial_rotary_factor: float,
    mrope_is_interleaved: bool,
    is_neox: bool,
    mrope_section: List[int],
    dtype: torch.dtype,
):

    torch.manual_seed(1234)

    num_qo_heads = num_kv_heads * gqa_ratio
    rotary_dim = int(head_size * partial_rotary_factor)
    q = torch.randn(batch_size, (num_qo_heads * head_size), device=DEVICE, dtype=dtype)
    k = torch.randn(batch_size, (num_kv_heads * head_size), device=DEVICE, dtype=dtype)
    positions = torch.randint(
        0, MAX_SEQ_LEN, (3, batch_size), device=DEVICE, dtype=torch.int64
    )

    cos_sin_cache = create_cos_sin_cache(rotary_dim).to(dtype)

    q_ker, k_ker = q.clone(), k.clone()
    q_na, k_na = torch_impl_mrope(
        q,
        k,
        cos_sin_cache,
        positions,
        mrope_section,
        head_size,
        rotary_dim,
        mrope_is_interleaved,
        is_neox,
    )

    multomodal_rotary_embedding(
        q_ker,
        k_ker,
        cos_sin_cache,
        positions,
        mrope_section,
        head_size,
        rotary_dim,
        mrope_is_interleaved,
        False,
        is_neox,
        None,
    )

    # mrope_triton(q_ker, k_ker,
    #    cos_sin_cache, positions, mrope_section,  head_size, rotary_dim, mrope_is_interleaved, False, is_neox, None)

    atol = rtol = 1e-2
    triton.testing.assert_close(q_na, q_ker, atol=atol, rtol=rtol)
    triton.testing.assert_close(k_na, k_ker, atol=atol, rtol=rtol)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
