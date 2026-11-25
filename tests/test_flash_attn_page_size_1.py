"""Tests for the page_size == 1 path of flash_attn_with_kvcache.

The paged kernel only supports page sizes 64/128, so page size 1 gathers
the per-token cache slots into a dense ragged buffer and runs the varlen
kernel. The single-request branch gathers a fixed-width row (for XPU
graph capture) with the padding indices clamped into range; these tests
check that path, the eager sliced path, and the multi-request masked
path against an fp32 reference over the valid prefix only.
"""

import pytest
import torch

import sgl_kernel  # noqa: F401 - registers torch.ops.sgl_kernel.*
from sgl_kernel.flash_attn import flash_attn_with_kvcache

DEVICE = "xpu"
pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available(), reason="requires an XPU device"
)

NUM_SLOTS = 256  # cache slots (page size 1: one token per slot)
TABLE_WIDTH = 160  # page_table row width, wider than any live length below
HEADS_Q = 4
HEADS_K = 2
HEAD_DIM = 64


def build_case(seqlen_k, sentinel=-1, seed=0):
    """One request whose tokens live in scattered slots, row padded."""
    gen = torch.Generator().manual_seed(seed)
    slots = torch.randperm(NUM_SLOTS, generator=gen)[:seqlen_k]
    page_table = torch.full((1, TABLE_WIDTH), sentinel, dtype=torch.int32)
    page_table[0, :seqlen_k] = slots.to(torch.int32)
    k_cache = torch.randn(NUM_SLOTS, 1, HEADS_K, HEAD_DIM, generator=gen)
    v_cache = torch.randn(NUM_SLOTS, 1, HEADS_K, HEAD_DIM, generator=gen)
    return slots, page_table, k_cache, v_cache


def reference(q, k_cache, v_cache, slots, causal):
    """fp32 attention of q over the valid slots only, GQA-expanded."""
    k = k_cache[slots, 0].float()  # [seqlen_k, HEADS_K, HEAD_DIM]
    v = v_cache[slots, 0].float()
    repeat = HEADS_Q // HEADS_K
    k = k.repeat_interleave(repeat, dim=1)
    v = v.repeat_interleave(repeat, dim=1)
    qf = q[0].float()  # [seqlen_q, HEADS_Q, HEAD_DIM]
    scale = 1.0 / (HEAD_DIM**0.5)
    scores = torch.einsum("qhd,khd->hqk", qf, k) * scale
    if causal:
        seqlen_q, seqlen_k = qf.shape[0], k.shape[0]
        rows = torch.arange(seqlen_q).unsqueeze(1)
        cols = torch.arange(seqlen_k).unsqueeze(0)
        scores = scores.masked_fill(
            cols > (seqlen_k - seqlen_q) + rows, float("-inf")
        )
    out = torch.einsum("hqk,khd->qhd", torch.softmax(scores, dim=-1), v)
    return out.unsqueeze(0)


def run_and_check(seqlen_q, seqlen_k, dtype, causal, seed=0):
    slots, page_table, k_cache, v_cache = build_case(seqlen_k, seed=seed)
    gen = torch.Generator().manual_seed(seed + 1)
    q = torch.randn(1, seqlen_q, HEADS_Q, HEAD_DIM, generator=gen)
    ref = reference(q, k_cache, v_cache, slots, causal)
    out = flash_attn_with_kvcache(
        q.to(device=DEVICE, dtype=dtype),
        k_cache.to(device=DEVICE, dtype=dtype),
        v_cache.to(device=DEVICE, dtype=dtype),
        cache_seqlens=torch.tensor([seqlen_k], dtype=torch.int32, device=DEVICE),
        page_table=page_table.to(DEVICE),
        causal=causal,
    )
    torch.xpu.synchronize()
    tol = dict(atol=2e-2, rtol=2e-2) if dtype == torch.float16 else dict(atol=5e-2, rtol=5e-2)
    assert torch.allclose(out.float().cpu(), ref.to(dtype).float(), **tol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_page_size_1_decode_eager(dtype):
    run_and_check(seqlen_q=1, seqlen_k=37, dtype=dtype, causal=True)


def test_page_size_1_verify_eager():
    run_and_check(seqlen_q=4, seqlen_k=64, dtype=torch.float16, causal=True)


def test_page_size_1_full_row_no_padding():
    run_and_check(seqlen_q=1, seqlen_k=TABLE_WIDTH, dtype=torch.float16, causal=False)


@pytest.mark.parametrize("sentinel", [-1, 1_000_000])
def test_page_size_1_capture_width_clamps_padding(monkeypatch, sentinel):
    # Force the capture branch: the gather spans the full table width, so
    # the padding reaches index_select. Without clamping, -1 (or any junk
    # sentinel) is an out-of-bounds read; with clamping the result must
    # still match the reference over the valid prefix.
    monkeypatch.setattr(
        torch.xpu, "is_current_stream_capturing", lambda: True
    )
    slots, page_table, k_cache, v_cache = build_case(37, sentinel=sentinel)
    gen = torch.Generator().manual_seed(3)
    q = torch.randn(1, 1, HEADS_Q, HEAD_DIM, generator=gen)
    ref = reference(q, k_cache, v_cache, slots, causal=True)
    out = flash_attn_with_kvcache(
        q.to(device=DEVICE, dtype=torch.float16),
        k_cache.to(device=DEVICE, dtype=torch.float16),
        v_cache.to(device=DEVICE, dtype=torch.float16),
        cache_seqlens=torch.tensor([37], dtype=torch.int32, device=DEVICE),
        page_table=page_table.to(DEVICE),
        causal=True,
    )
    torch.xpu.synchronize()
    assert torch.allclose(
        out.float().cpu(), ref.to(torch.float16).float(), atol=2e-2, rtol=2e-2
    )


def test_page_size_1_multi_request_masked_path():
    # Batch > 1 takes the masked gather; both rows must match their
    # references (this is the path the single-request result is compared
    # against for consistency).
    gen = torch.Generator().manual_seed(7)
    lengths = [37, 100]
    slots = torch.randperm(NUM_SLOTS, generator=gen)
    page_table = torch.full((2, TABLE_WIDTH), -1, dtype=torch.int32)
    page_table[0, : lengths[0]] = slots[: lengths[0]].to(torch.int32)
    page_table[1, : lengths[1]] = slots[lengths[0] : lengths[0] + lengths[1]].to(
        torch.int32
    )
    k_cache = torch.randn(NUM_SLOTS, 1, HEADS_K, HEAD_DIM, generator=gen)
    v_cache = torch.randn(NUM_SLOTS, 1, HEADS_K, HEAD_DIM, generator=gen)
    q = torch.randn(2, 1, HEADS_Q, HEAD_DIM, generator=gen)
    out = flash_attn_with_kvcache(
        q.to(device=DEVICE, dtype=torch.float16),
        k_cache.to(device=DEVICE, dtype=torch.float16),
        v_cache.to(device=DEVICE, dtype=torch.float16),
        cache_seqlens=torch.tensor(lengths, dtype=torch.int32, device=DEVICE),
        page_table=page_table.to(DEVICE),
        causal=True,
    )
    torch.xpu.synchronize()
    for row, length in enumerate(lengths):
        row_slots = page_table[row, :length].long()
        ref = reference(
            q[row : row + 1], k_cache, v_cache, row_slots, causal=True
        )
        assert torch.allclose(
            out[row : row + 1].float().cpu(),
            ref.to(torch.float16).float(),
            atol=2e-2,
            rtol=2e-2,
        )
