"""Tests for the pure-PyTorch sparse attention implementation.

These tests validate the PyTorch fallback path of sparse_attn_func and
convert_vertical_slash_indexes. They run on CPU (or XPU/CUDA when available)
without requiring native CUDA/SYCL kernels.
"""

import importlib.util
import math
import os
import sys
from typing import Optional

import pytest
import torch
from einops import rearrange, repeat

# Import sparse_flash_attn module directly (bypassing sgl_kernel __init__.py
# which requires the C++ extension).
_module_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "python",
    "sgl_kernel",
    "sparse_flash_attn.py",
)
_spec = importlib.util.spec_from_file_location("sparse_flash_attn", _module_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_sparse_attn_func_torch = _mod._sparse_attn_func_torch
_sparse_attn_varlen_func_torch = _mod._sparse_attn_varlen_func_torch
convert_vertical_slash_indexes = _mod.convert_vertical_slash_indexes
convert_vertical_slash_indexes_mergehead = _mod.convert_vertical_slash_indexes_mergehead
sparse_attn_func = _mod.sparse_attn_func
sparse_attn_varlen_func = _mod.sparse_attn_varlen_func

# Use XPU if available, otherwise CPU
if torch.xpu.is_available():
    DEVICE = "xpu"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


def ref_attn(
    q,
    k,
    v,
    causal=False,
    softcap=0.0,
    upcast=True,
):
    """Reference standard attention for comparison.

    Args:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)

    Returns:
        output: (batch_size, seqlen_q, nheads, head_dim)
        lse: (batch_size, nheads, seqlen_q)
    """
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)

    if softcap > 0:
        scores = scores / softcap
        scores = scores.tanh()
        scores = scores * softcap

    if causal:
        # Causal mask aligned to bottom-right
        row = torch.arange(seqlen_q, device=q.device).unsqueeze(1)
        col = torch.arange(seqlen_k, device=q.device).unsqueeze(0)
        causal_mask = (row + seqlen_k - seqlen_q) >= col
        scores = scores.masked_fill(
            ~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )

    lse_ref = scores.logsumexp(dim=-1)
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    output = torch.einsum("bhts,bshd->bthd", attention, v)
    return output.to(dtype=dtype_og), lse_ref


# ===== Tests for sparse_attn_func (full-coverage pattern = standard attn) =====


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize(
    "seq_lens",
    [
        (32, 32),
        (65, 65),
        (128, 128),
        (129, 129),
    ],
)
@pytest.mark.parametrize("num_heads", [1, 2, 4])
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("NNZ_S", [0, 1, 2])
@torch.inference_mode()
def test_sparse_attention_torch_full_coverage(
    batch_size,
    seq_lens,
    num_heads,
    head_size,
    dtype,
    NNZ_S,
) -> None:
    """Test sparse attention with full KV coverage equals standard attention."""
    torch.manual_seed(42)
    device = DEVICE
    block_size_M = 64
    block_size_N = 64
    seqlen_q, seqlen_k = seq_lens

    q = torch.randn(
        batch_size,
        seqlen_q,
        num_heads,
        head_size,
        dtype=dtype,
        device=device,
        requires_grad=False,
    )
    k = torch.randn(
        batch_size,
        seqlen_k,
        num_heads,
        head_size,
        dtype=dtype,
        device=device,
        requires_grad=False,
    )
    v = torch.randn(
        batch_size,
        seqlen_k,
        num_heads,
        head_size,
        dtype=dtype,
        device=device,
        requires_grad=False,
    )

    NUM_ROWS = (seqlen_q + block_size_M - 1) // block_size_M
    if NNZ_S * block_size_N > seqlen_k:
        return

    NNZ_V = seqlen_k - NNZ_S * block_size_N
    block_count = torch.tensor(
        [NNZ_S] * batch_size * NUM_ROWS * num_heads, dtype=torch.int32, device=device
    ).reshape(batch_size, num_heads, NUM_ROWS)
    column_count = torch.tensor(
        [NNZ_V] * batch_size * NUM_ROWS * num_heads, dtype=torch.int32, device=device
    ).reshape(batch_size, num_heads, NUM_ROWS)
    block_offset = torch.tensor(
        [[i * block_size_N for i in range(NNZ_S)]] * batch_size * NUM_ROWS * num_heads,
        dtype=torch.int32,
        device=device,
    ).reshape(batch_size, num_heads, NUM_ROWS, NNZ_S)
    column_index = torch.tensor(
        [[NNZ_S * block_size_N + i for i in range(NNZ_V)]]
        * batch_size
        * NUM_ROWS
        * num_heads,
        dtype=torch.int32,
        device=device,
    ).reshape(batch_size, num_heads, NUM_ROWS, NNZ_V)

    out, lse = sparse_attn_func(
        q,
        k,
        v,
        block_count,
        block_offset,
        column_count,
        column_index,
        return_softmax_lse=True,
        block_size_M=block_size_M,
        block_size_N=block_size_N,
    )

    ref_out, ref_lse = ref_attn(q, k, v)

    torch.testing.assert_close(
        out, ref_out, atol=2e-2, rtol=1e-2
    ), f"Max diff: {torch.max(torch.abs(out - ref_out))}"
    torch.testing.assert_close(
        lse, ref_lse, atol=2e-2, rtol=1e-2
    ), f"Max diff: {torch.max(torch.abs(lse - ref_lse))}"


# ===== Tests for sparse attention with actual sparsity =====


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_sparse_attention_torch_actual_sparsity(batch_size, dtype) -> None:
    """Test that sparse attention computes correctly with partial KV coverage."""
    torch.manual_seed(42)
    device = DEVICE
    seqlen = 128
    num_heads = 2
    head_size = 64
    block_size_M = 64
    block_size_N = 64

    q = torch.randn(
        batch_size, seqlen, num_heads, head_size, dtype=dtype, device=device
    )
    k = torch.randn(
        batch_size, seqlen, num_heads, head_size, dtype=dtype, device=device
    )
    v = torch.randn(
        batch_size, seqlen, num_heads, head_size, dtype=dtype, device=device
    )

    NUM_ROWS = seqlen // block_size_M  # 2

    # Attend to first block (0-63) and column 100
    NNZ_S = 1
    NNZ_V = 1
    block_count = torch.full(
        (batch_size, num_heads, NUM_ROWS),
        NNZ_S,
        dtype=torch.int32,
        device=device,
    )
    block_offset = torch.zeros(
        batch_size, num_heads, NUM_ROWS, NNZ_S, dtype=torch.int32, device=device
    )  # block at position 0
    column_count = torch.full(
        (batch_size, num_heads, NUM_ROWS),
        NNZ_V,
        dtype=torch.int32,
        device=device,
    )
    column_index = torch.full(
        (batch_size, num_heads, NUM_ROWS, NNZ_V),
        100,
        dtype=torch.int32,
        device=device,
    )

    out = sparse_attn_func(
        q,
        k,
        v,
        block_count,
        block_offset,
        column_count,
        column_index,
        block_size_M=block_size_M,
        block_size_N=block_size_N,
    )

    # Verify output shape
    assert out.shape == q.shape

    # Manually compute reference: only attend to KV positions 0-63 and 100
    q_f = q.transpose(1, 2).float()
    k_f = k.transpose(1, 2).float()
    v_f = v.transpose(1, 2).float()
    scale = head_size**-0.5

    scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale
    # Mask out everything except positions 0-63 and 100
    mask = torch.zeros(seqlen, dtype=torch.bool, device=device)
    mask[:64] = True
    mask[100] = True
    scores = scores.masked_fill(
        ~mask.unsqueeze(0).unsqueeze(0).unsqueeze(0), float("-inf")
    )
    attn = torch.softmax(scores, dim=-1).nan_to_num(0.0)
    ref_out = torch.matmul(attn, v_f).transpose(1, 2).to(dtype)

    torch.testing.assert_close(out, ref_out, atol=2e-2, rtol=1e-2)


# ===== Tests for sparse attention with causal mask =====


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seqlen", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_sparse_attention_torch_causal(batch_size, seqlen, dtype) -> None:
    """Test sparse attention with causal masking and full coverage."""
    torch.manual_seed(42)
    device = DEVICE
    num_heads = 2
    head_size = 64
    block_size_M = 64
    block_size_N = 64

    q = torch.randn(
        batch_size, seqlen, num_heads, head_size, dtype=dtype, device=device
    )
    k = torch.randn(
        batch_size, seqlen, num_heads, head_size, dtype=dtype, device=device
    )
    v = torch.randn(
        batch_size, seqlen, num_heads, head_size, dtype=dtype, device=device
    )

    NUM_ROWS = (seqlen + block_size_M - 1) // block_size_M
    NNZ_V = seqlen  # Full column coverage (no slash blocks)

    block_count = torch.zeros(
        batch_size, num_heads, NUM_ROWS, dtype=torch.int32, device=device
    )
    block_offset = torch.zeros(
        batch_size, num_heads, NUM_ROWS, 1, dtype=torch.int32, device=device
    )
    column_count = torch.full(
        (batch_size, num_heads, NUM_ROWS),
        NNZ_V,
        dtype=torch.int32,
        device=device,
    )
    column_index = (
        torch.arange(NNZ_V, dtype=torch.int32, device=device)
        .unsqueeze(0)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(batch_size, num_heads, NUM_ROWS, NNZ_V)
        .contiguous()
    )

    out, lse = sparse_attn_func(
        q,
        k,
        v,
        block_count,
        block_offset,
        column_count,
        column_index,
        causal=True,
        return_softmax_lse=True,
        block_size_M=block_size_M,
        block_size_N=block_size_N,
    )

    ref_out, ref_lse = ref_attn(q, k, v, causal=True)

    torch.testing.assert_close(out, ref_out, atol=2e-2, rtol=1e-2)
    torch.testing.assert_close(lse, ref_lse, atol=2e-2, rtol=1e-2)


# ===== Tests for GQA (grouped-query attention) =====


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_sparse_attention_torch_gqa(dtype) -> None:
    """Test sparse attention with grouped-query attention (nheads > nheads_k)."""
    torch.manual_seed(42)
    device = DEVICE
    batch_size = 1
    seqlen = 64
    nheads = 4
    nheads_k = 2
    head_size = 64
    block_size_M = 64
    block_size_N = 64

    q = torch.randn(batch_size, seqlen, nheads, head_size, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen, nheads_k, head_size, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen, nheads_k, head_size, dtype=dtype, device=device)

    NUM_ROWS = 1
    NNZ_V = seqlen

    block_count = torch.zeros(
        batch_size, nheads, NUM_ROWS, dtype=torch.int32, device=device
    )
    block_offset = torch.zeros(
        batch_size, nheads, NUM_ROWS, 1, dtype=torch.int32, device=device
    )
    column_count = torch.full(
        (batch_size, nheads, NUM_ROWS), NNZ_V, dtype=torch.int32, device=device
    )
    column_index = (
        torch.arange(NNZ_V, dtype=torch.int32, device=device)
        .unsqueeze(0)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(batch_size, nheads, NUM_ROWS, NNZ_V)
        .contiguous()
    )

    out = sparse_attn_func(
        q,
        k,
        v,
        block_count,
        block_offset,
        column_count,
        column_index,
        block_size_M=block_size_M,
        block_size_N=block_size_N,
    )

    ref_out, _ = ref_attn(q, k, v)
    torch.testing.assert_close(out, ref_out, atol=2e-2, rtol=1e-2)


# ===== Tests for softcap =====


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_sparse_attention_torch_softcap(dtype) -> None:
    """Test sparse attention with attention score softcapping."""
    torch.manual_seed(42)
    device = DEVICE
    batch_size = 1
    seqlen = 64
    num_heads = 2
    head_size = 64
    block_size_M = 64
    block_size_N = 64
    softcap = 30.0

    q = torch.randn(
        batch_size, seqlen, num_heads, head_size, dtype=dtype, device=device
    )
    k = torch.randn(
        batch_size, seqlen, num_heads, head_size, dtype=dtype, device=device
    )
    v = torch.randn(
        batch_size, seqlen, num_heads, head_size, dtype=dtype, device=device
    )

    NUM_ROWS = 1
    NNZ_V = seqlen
    block_count = torch.zeros(
        batch_size, num_heads, NUM_ROWS, dtype=torch.int32, device=device
    )
    block_offset = torch.zeros(
        batch_size, num_heads, NUM_ROWS, 1, dtype=torch.int32, device=device
    )
    column_count = torch.full(
        (batch_size, num_heads, NUM_ROWS), NNZ_V, dtype=torch.int32, device=device
    )
    column_index = (
        torch.arange(NNZ_V, dtype=torch.int32, device=device)
        .unsqueeze(0)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(batch_size, num_heads, NUM_ROWS, NNZ_V)
        .contiguous()
    )

    out = sparse_attn_func(
        q,
        k,
        v,
        block_count,
        block_offset,
        column_count,
        column_index,
        softcap=softcap,
        block_size_M=block_size_M,
        block_size_N=block_size_N,
    )

    ref_out, _ = ref_attn(q, k, v, softcap=softcap)
    torch.testing.assert_close(out, ref_out, atol=2e-2, rtol=1e-2)


# ===== Tests for convert_vertical_slash_indexes =====


@pytest.mark.parametrize("causal", [True, False])
def test_convert_vertical_slash_indexes_torch(causal):
    """Test the PyTorch convert_vertical_slash_indexes with a small example."""
    device = DEVICE
    q_seqlens = torch.tensor([4], dtype=torch.int32, device=device)
    kv_seqlens = torch.tensor([4], dtype=torch.int32, device=device)
    vertical_indexes = torch.tensor([[[1, 3]]], dtype=torch.int32, device=device)
    slash_indexes = torch.tensor([[[2]]], dtype=torch.int32, device=device)
    context_size = 4
    block_size_M = 2
    block_size_N = 2

    block_count, block_offset, column_count, column_index = (
        convert_vertical_slash_indexes(
            q_seqlens,
            kv_seqlens,
            vertical_indexes,
            slash_indexes,
            context_size,
            block_size_M,
            block_size_N,
            causal=causal,
        )
    )

    # Verify shapes
    assert block_count.shape == (1, 1, 2)
    assert block_offset.shape == (1, 1, 2, 1)
    assert column_count.shape == (1, 1, 2)
    assert column_index.shape == (1, 1, 2, 2)

    if not causal:
        # Non-causal: vertical columns 1 and 3 should appear
        # Row 0: column 1 (column 3 is inside slash block range [2,4))
        # Row 1: columns 1 and 3 (slash block [4,6) is out of range)
        expected_column_index = torch.tensor(
            [[[[1, 0], [1, 3]]]], dtype=torch.int32, device=device
        )
        assert torch.equal(
            column_index, expected_column_index
        ), f"Got {column_index}, expected {expected_column_index}"
    else:
        # Causal: vertical columns are masked out by causal boundary
        expected_column_index = torch.tensor(
            [[[[0, 0], [0, 0]]]], dtype=torch.int32, device=device
        )
        assert torch.equal(
            column_index, expected_column_index
        ), f"Got {column_index}, expected {expected_column_index}"


# ===== Tests for variable-length sparse attention =====


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_sparse_attention_varlen_torch(dtype) -> None:
    """Test variable-length sparse attention with full coverage."""
    torch.manual_seed(42)
    device = DEVICE
    block_size_M = 64
    block_size_N = 64

    # Two sequences with same length (simplifies block index alignment)
    seq_lens = [(64, 64), (64, 64)]
    batch_size = len(seq_lens)
    query_lens = [s[0] for s in seq_lens]
    kv_lens = [s[1] for s in seq_lens]
    num_heads = 2
    head_size = 64

    q = torch.randn(sum(query_lens), num_heads, head_size, dtype=dtype, device=device)
    k = torch.randn(sum(kv_lens), num_heads, head_size, dtype=dtype, device=device)
    v = torch.randn_like(k)

    cu_seqlens_q = torch.tensor(
        [0] + query_lens, dtype=torch.int32, device=device
    ).cumsum(dim=0, dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0] + kv_lens, dtype=torch.int32, device=device).cumsum(
        dim=0, dtype=torch.int32
    )
    max_q = max(query_lens)
    max_k = max(kv_lens)

    NUM_ROWS = (max_q + block_size_M - 1) // block_size_M

    # Full column coverage (no slash blocks)
    NNZ_V = max_k
    block_count = torch.zeros(
        batch_size, num_heads, NUM_ROWS, dtype=torch.int32, device=device
    )
    block_offset = torch.zeros(
        batch_size, num_heads, NUM_ROWS, 1, dtype=torch.int32, device=device
    )

    # Build column indices per batch
    column_count_list = []
    column_index_list = []
    for b in range(batch_size):
        cc = torch.full(
            (num_heads, NUM_ROWS),
            kv_lens[b],
            dtype=torch.int32,
            device=device,
        )
        ci = (
            torch.arange(NNZ_V, dtype=torch.int32, device=device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(num_heads, NUM_ROWS, NNZ_V)
            .contiguous()
        )
        column_count_list.append(cc)
        column_index_list.append(ci)

    column_count = torch.stack(column_count_list)
    column_index = torch.stack(column_index_list)

    out = sparse_attn_varlen_func(
        q,
        k,
        v,
        block_count,
        block_offset,
        column_count,
        column_index,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        block_size_M=block_size_M,
        block_size_N=block_size_N,
    )

    assert out.shape == (sum(query_lens), num_heads, head_size)

    # Compare per-batch with standard attention
    for b in range(batch_size):
        q_s = cu_seqlens_q[b].item()
        q_e = cu_seqlens_q[b + 1].item()
        k_s = cu_seqlens_k[b].item()
        k_e = cu_seqlens_k[b + 1].item()

        q_b = q[q_s:q_e].unsqueeze(0)  # (1, sq, H, D)
        k_b = k[k_s:k_e].unsqueeze(0)
        v_b = v[k_s:k_e].unsqueeze(0)

        ref_out, _ = ref_attn(q_b, k_b, v_b)
        torch.testing.assert_close(
            out[q_s:q_e].unsqueeze(0), ref_out, atol=2e-2, rtol=1e-2
        )


# ===== Tests for DeepSeek V3 pattern =====


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_sparse_attention_deepseek_v3_pattern(dtype) -> None:
    """Test with a DeepSeek V3-style pattern: attention sinks + local window.

    DeepSeek V3 uses a vertical+slash sparse attention (NSA) pattern:
    - Vertical: a few "sink" tokens (e.g., first few positions)
    - Slash: local context window (recent tokens)
    """
    torch.manual_seed(42)
    device = DEVICE
    batch_size = 1
    seqlen = 256
    num_heads = 4
    head_size = 128
    block_size_M = 64
    block_size_N = 64
    NUM_ROWS = seqlen // block_size_M  # 4

    q = torch.randn(
        batch_size, seqlen, num_heads, head_size, dtype=dtype, device=device
    )
    k = torch.randn(
        batch_size, seqlen, num_heads, head_size, dtype=dtype, device=device
    )
    v = torch.randn(
        batch_size, seqlen, num_heads, head_size, dtype=dtype, device=device
    )

    # Pattern: attend to first block (sink tokens) + 2 "vertical" columns
    NNZ_S = 1  # 1 slash block
    NNZ_V = 2  # 2 vertical columns

    # Slash block: always attend to KV block starting at 0
    block_count = torch.full(
        (batch_size, num_heads, NUM_ROWS),
        NNZ_S,
        dtype=torch.int32,
        device=device,
    )
    block_offset = torch.zeros(
        batch_size, num_heads, NUM_ROWS, NNZ_S, dtype=torch.int32, device=device
    )  # block at 0

    # Vertical columns: attend to positions 64 and 128
    column_count = torch.full(
        (batch_size, num_heads, NUM_ROWS),
        NNZ_V,
        dtype=torch.int32,
        device=device,
    )
    ci = torch.tensor([64, 128], dtype=torch.int32, device=device)
    column_index = (
        ci.unsqueeze(0)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(batch_size, num_heads, NUM_ROWS, NNZ_V)
        .contiguous()
    )

    out = sparse_attn_func(
        q,
        k,
        v,
        block_count,
        block_offset,
        column_count,
        column_index,
        block_size_M=block_size_M,
        block_size_N=block_size_N,
    )

    assert out.shape == q.shape

    # Verify by manually computing attention with the same mask
    q_f = q.transpose(1, 2).float()
    k_f = k.transpose(1, 2).float()
    v_f = v.transpose(1, 2).float()
    scale = head_size**-0.5

    scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale
    # Build mask: attend to KV 0-63, 64, and 128
    mask = torch.zeros(seqlen, dtype=torch.bool, device=device)
    mask[:64] = True
    mask[64] = True
    mask[128] = True
    scores = scores.masked_fill(
        ~mask.unsqueeze(0).unsqueeze(0).unsqueeze(0), float("-inf")
    )
    attn = torch.softmax(scores, dim=-1).nan_to_num(0.0)
    ref_out = torch.matmul(attn, v_f).transpose(1, 2).to(dtype)

    torch.testing.assert_close(out, ref_out, atol=2e-2, rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
