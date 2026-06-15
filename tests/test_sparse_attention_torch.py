"""Tests for the pure-PyTorch fallback implementations of sparse attention.

These tests target the _*_torch helper functions directly so they run on CPU
without requiring any native CUDA/SYCL kernels.
"""

import math

import pytest
import torch
from einops import repeat

from sgl_kernel.sparse_flash_attn import (
    _convert_vertical_slash_indexes_mergehead_torch,
    _convert_vertical_slash_indexes_torch,
    _sparse_attn_func_torch,
    _sparse_attn_varlen_func_torch,
)


# ---------------------------------------------------------------------------
# Reference dense attention (CPU, float32)
# ---------------------------------------------------------------------------


def _ref_attn(q, k, v, causal=False):
    """Scaled dot-product attention reference.

    Args:
        q: (batch, seqlen_q, nheads, headdim)
        k: (batch, seqlen_k, nheads_k, headdim)
        v: (batch, seqlen_k, nheads_k, headdim)
        causal: apply causal mask

    Returns:
        out: (batch, seqlen_q, nheads, headdim)  float32
        lse: (batch, nheads, seqlen_q)           float32
    """
    B, Sq, Hq, D = q.shape
    _, Sk, Hk, _ = k.shape
    scale = math.sqrt(D)
    # GQA expansion
    k = repeat(k, "b s h d -> b s (h g) d", g=Hq // Hk)
    v = repeat(v, "b s h d -> b s (h g) d", g=Hq // Hk)

    q_f = q.float().transpose(1, 2)  # (B, H, Sq, D)
    k_f = k.float().transpose(1, 2)  # (B, H, Sk, D)
    v_f = v.float().transpose(1, 2)

    scores = torch.matmul(q_f, k_f.transpose(-2, -1)) / scale  # (B, H, Sq, Sk)

    if causal:
        q_pos = torch.arange(Sq, device=q.device).unsqueeze(1)
        k_pos = torch.arange(Sk, device=q.device).unsqueeze(0)
        causal_mask = (q_pos + Sk - Sq) < k_pos
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    lse = scores.logsumexp(dim=-1)  # (B, H, Sq)
    attn = torch.softmax(scores, dim=-1).nan_to_num(0.0)
    out = torch.matmul(attn, v_f).transpose(1, 2).to(q.dtype)
    return out, lse


# ---------------------------------------------------------------------------
# convert_vertical_slash_indexes (PyTorch fallback)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("causal", [True, False])
def test_convert_vertical_slash_indexes_torch_shapes(causal):
    """Output tensors have the expected shapes."""
    q_seqlens = torch.tensor([4], dtype=torch.int32)
    kv_seqlens = torch.tensor([4], dtype=torch.int32)
    vertical_indexes = torch.tensor([[[1, 3]]], dtype=torch.int32)
    slash_indexes = torch.tensor([[[2]]], dtype=torch.int32)
    context_size, block_size_M, block_size_N = 4, 2, 2

    bc, bo, cc, ci = _convert_vertical_slash_indexes_torch(
        q_seqlens,
        kv_seqlens,
        vertical_indexes,
        slash_indexes,
        context_size,
        block_size_M,
        block_size_N,
        causal=causal,
    )

    num_rows = (context_size + block_size_M - 1) // block_size_M
    assert bc.shape == (1, 1, num_rows)
    assert bo.shape == (1, 1, num_rows, slash_indexes.shape[2])
    assert cc.shape == (1, 1, num_rows)
    assert ci.shape == (1, 1, num_rows, vertical_indexes.shape[2])


@pytest.mark.parametrize("causal", [True, False])
def test_convert_vertical_slash_indexes_torch_zero_init(causal):
    """Unused slots in output tensors are zero (deterministic)."""
    q_seqlens = torch.tensor([4], dtype=torch.int32)
    kv_seqlens = torch.tensor([4], dtype=torch.int32)
    vertical_indexes = torch.tensor([[[0, 1, 2, 3]]], dtype=torch.int32)
    slash_indexes = torch.tensor([[[0, 1, 2, 3]]], dtype=torch.int32)
    context_size, block_size_M, block_size_N = 4, 4, 4  # single row

    bc, bo, cc, ci = _convert_vertical_slash_indexes_torch(
        q_seqlens,
        kv_seqlens,
        vertical_indexes,
        slash_indexes,
        context_size,
        block_size_M,
        block_size_N,
        causal=causal,
    )

    # For each row, valid entries are <= bc / cc; remaining slots must be 0
    for row in range(bc.shape[2]):
        blk_cnt = bc[0, 0, row].item()
        col_cnt = cc[0, 0, row].item()
        if blk_cnt < bo.shape[3]:
            assert bo[0, 0, row, int(blk_cnt) :].eq(0).all(), (
                f"Non-zero padding in block_offset at row {row}"
            )
        if col_cnt < ci.shape[3]:
            assert ci[0, 0, row, int(col_cnt) :].eq(0).all(), (
                f"Non-zero padding in column_index at row {row}"
            )


# ---------------------------------------------------------------------------
# convert_vertical_slash_indexes_mergehead (PyTorch fallback)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("causal", [True, False])
def test_convert_vertical_slash_indexes_mergehead_torch_shapes(causal):
    q_seqlens = torch.tensor([4], dtype=torch.int32)
    kv_seqlens = torch.tensor([4], dtype=torch.int32)
    vertical_indexes = torch.tensor([[[1, 3], [2, 0]]], dtype=torch.int32)
    slash_indexes = torch.tensor([[[2, 0], [1, 3]]], dtype=torch.int32)
    vertical_indices_count = torch.tensor([2, 1], dtype=torch.int32)
    slash_indices_count = torch.tensor([1, 2], dtype=torch.int32)
    context_size, block_size_M, block_size_N = 4, 2, 2

    bc, bo, cc, ci = _convert_vertical_slash_indexes_mergehead_torch(
        q_seqlens,
        kv_seqlens,
        vertical_indexes,
        slash_indexes,
        vertical_indices_count,
        slash_indices_count,
        context_size,
        block_size_M,
        block_size_N,
        causal=causal,
    )

    num_rows = (context_size + block_size_M - 1) // block_size_M
    B, H = 1, 2
    assert bc.shape == (B, H, num_rows)
    assert bo.shape == (B, H, num_rows, slash_indexes.shape[2])
    assert cc.shape == (B, H, num_rows)
    assert ci.shape == (B, H, num_rows, vertical_indexes.shape[2])


# ---------------------------------------------------------------------------
# _sparse_attn_func_torch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("nheads", [1, 2])
def test_sparse_attn_func_torch_full_dense(causal, nheads):
    """Sparse attention == dense when all KV tokens are unmasked."""
    torch.manual_seed(0)
    B, Sq, Sk, D = 2, 8, 8, 32
    block_size_M = block_size_N = 4
    num_rows = (Sq + block_size_M - 1) // block_size_M
    NNZ_S = Sk // block_size_N

    q = torch.randn(B, Sq, nheads, D)
    k = torch.randn(B, Sk, nheads, D)
    v = torch.randn(B, Sk, nheads, D)

    # cover the full KV range with slash blocks
    block_count = torch.full((B, nheads, num_rows), NNZ_S, dtype=torch.int32)
    block_offset = torch.zeros(B, nheads, num_rows, NNZ_S, dtype=torch.int32)
    for i in range(NNZ_S):
        block_offset[:, :, :, i] = i * block_size_N
    column_count = torch.zeros(B, nheads, num_rows, dtype=torch.int32)
    column_index = torch.zeros(B, nheads, num_rows, 1, dtype=torch.int32)

    scale = 1.0 / math.sqrt(D)
    out, lse = _sparse_attn_func_torch(
        q,
        k,
        v,
        block_count,
        block_offset,
        column_count,
        column_index,
        softmax_scale=scale,
        causal=causal,
        block_size_M=block_size_M,
        block_size_N=block_size_N,
    )

    ref_out, ref_lse = _ref_attn(q, k, v, causal=causal)

    torch.testing.assert_close(out, ref_out, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(lse, ref_lse, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("causal", [True, False])
def test_sparse_attn_func_torch_output_shape(causal):
    torch.manual_seed(1)
    B, Sq, Sk, nheads, D = 1, 16, 16, 2, 64
    block_size_M = block_size_N = 8
    num_rows = (Sq + block_size_M - 1) // block_size_M
    NNZ_S = Sk // block_size_N

    q = torch.randn(B, Sq, nheads, D)
    k = torch.randn(B, Sk, nheads, D)
    v = torch.randn(B, Sk, nheads, D)

    block_count = torch.full((B, nheads, num_rows), NNZ_S, dtype=torch.int32)
    block_offset = torch.zeros(B, nheads, num_rows, NNZ_S, dtype=torch.int32)
    for i in range(NNZ_S):
        block_offset[:, :, :, i] = i * block_size_N
    column_count = torch.zeros(B, nheads, num_rows, dtype=torch.int32)
    column_index = torch.zeros(B, nheads, num_rows, 1, dtype=torch.int32)

    scale = 1.0 / math.sqrt(D)
    out, lse = _sparse_attn_func_torch(
        q,
        k,
        v,
        block_count,
        block_offset,
        column_count,
        column_index,
        softmax_scale=scale,
        causal=causal,
        block_size_M=block_size_M,
        block_size_N=block_size_N,
    )

    assert out.shape == (B, Sq, nheads, D)
    assert lse.shape == (B, nheads, Sq)


# ---------------------------------------------------------------------------
# _sparse_attn_varlen_func_torch
# ---------------------------------------------------------------------------


def test_sparse_attn_varlen_func_torch_output_shape():
    torch.manual_seed(2)
    query_lens = [8, 4]
    kv_lens = [8, 4]
    nheads, D = 2, 32
    block_size_M = block_size_N = 4

    total_q = sum(query_lens)
    total_k = sum(kv_lens)
    q = torch.randn(total_q, nheads, D)
    k = torch.randn(total_k, nheads, D)
    v = torch.randn(total_k, nheads, D)

    cu_q = torch.tensor([0, query_lens[0], total_q], dtype=torch.int32)
    cu_k = torch.tensor([0, kv_lens[0], total_k], dtype=torch.int32)
    max_sq = max(query_lens)
    max_sk = max(kv_lens)
    batch_size = len(query_lens)
    num_rows = (max_sq + block_size_M - 1) // block_size_M
    NNZ_S = max_sk // block_size_N

    block_count = torch.full((batch_size, nheads, num_rows), NNZ_S, dtype=torch.int32)
    block_offset = torch.zeros(batch_size, nheads, num_rows, NNZ_S, dtype=torch.int32)
    for i in range(NNZ_S):
        block_offset[:, :, :, i] = i * block_size_N
    column_count = torch.zeros(batch_size, nheads, num_rows, dtype=torch.int32)
    column_index = torch.zeros(batch_size, nheads, num_rows, 1, dtype=torch.int32)

    scale = 1.0 / math.sqrt(D)
    out, lse = _sparse_attn_varlen_func_torch(
        q,
        k,
        v,
        block_count,
        block_offset,
        column_count,
        column_index,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max_sq,
        max_seqlen_k=max_sk,
        softmax_scale=scale,
        causal=False,
        block_size_M=block_size_M,
        block_size_N=block_size_N,
    )

    assert out.shape == (total_q, nheads, D)
    assert lse.shape == (nheads, total_q)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
