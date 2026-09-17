"""Bidirectional-block prefill attention (bidirectional_block_ids).

The prefill kernel takes a per-token block id (-1 = none) and, run non-causal,
re-imposes causal masking EXCEPT that keys sharing a query's non-negative block
id are attended both directions. Checked against an fp32 torch reference over a
multi-sequence ragged batch (which exercises the per-sequence q_token_offset),
plus the block-id builder and the host-side rejection guards.
"""

import itertools

import pytest
import torch

from sgl_kernel.flash_attn import (
    build_bidirectional_block_ids,
    flash_attn_with_kvcache,
)

P = 128  # page size (must be a multiple of the K-tile; page_size==1 is a separate path)
H, HKV, D = 8, 4, 128


def _paged_batch(seq_lens, dev):
    """Build a paged KV cache + ragged q for a full-prefill batch."""
    total = sum(seq_lens)
    q = torch.randn(total, H, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(total, HKV, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(total, HKV, D, device=dev, dtype=torch.bfloat16)
    npages = [(l + P - 1) // P for l in seq_lens]
    k_cache = torch.zeros(sum(npages), P, HKV, D, device=dev, dtype=torch.bfloat16)
    v_cache = torch.zeros(sum(npages), P, HKV, D, device=dev, dtype=torch.bfloat16)
    page_table = torch.zeros(len(seq_lens), max(npages), device=dev, dtype=torch.int32)
    tok = pg = 0
    for i, l in enumerate(seq_lens):
        for j in range(l):
            k_cache[pg + j // P, j % P] = k[tok + j]
            v_cache[pg + j // P, j % P] = v[tok + j]
        for pp in range(npages[i]):
            page_table[i, pp] = pg + pp
        tok += l
        pg += npages[i]
    cache_seqlens = torch.tensor(seq_lens, device=dev, dtype=torch.int32)
    cu_q = torch.tensor(
        [0, *itertools.accumulate(seq_lens)], device=dev, dtype=torch.int32
    )
    return q, k, v, k_cache, v_cache, page_table, cache_seqlens, cu_q


def _ref_bidir(q, k, v, seq_lens, block_spans, scale):
    """Per-sequence fp32 reference: causal, plus same-span tokens bidirectional."""
    out = torch.empty(q.shape[0], H, D, device=q.device, dtype=torch.float32)
    base = 0
    for l, spans in zip(seq_lens, block_spans):
        qi = q[base : base + l].float()
        ki = k[base : base + l].float().repeat_interleave(H // HKV, dim=1)
        vi = v[base : base + l].float().repeat_interleave(H // HKV, dim=1)
        rows = torch.arange(l, device=q.device).view(l, 1)
        cols = torch.arange(l, device=q.device).view(1, l)
        allow = cols <= rows
        for b, e in spans:
            allow[b : e + 1, b : e + 1] = True
        for h in range(H):
            sc = (qi[:, h] @ ki[:, h].T) * scale
            sc = sc.masked_fill(~allow, float("-inf"))
            out[base : base + l, h] = torch.softmax(sc, dim=-1) @ vi[:, h]
        base += l
    return out


@pytest.mark.parametrize(
    "seq_lens,block_spans",
    [
        # multi-batch ragged: different lengths (q_token_offset), multiple blocks,
        # a block at a sequence boundary, and a text-only sequence.
        ([256, 128, 200], [[(32, 96), (160, 224)], [(0, 63)], []]),
    ],
)
def test_bidirectional_block_mask(seq_lens, block_spans):
    if not torch.xpu.is_available():
        pytest.skip("XPU required")
    torch.manual_seed(0)
    dev = "xpu"
    scale = 1.0 / (D**0.5)
    q, k, v, kc, vc, pt, cs, cuq = _paged_batch(seq_lens, dev)
    block_ids = build_bidirectional_block_ids(seq_lens, block_spans, dev)

    out = flash_attn_with_kvcache(
        q, kc, vc, cache_seqlens=cs, page_table=pt, cu_seqlens_q=cuq,
        max_seqlen_q=max(seq_lens), max_seqlen_k=max(seq_lens),
        softmax_scale=scale, causal=False, bidirectional_block_ids=block_ids,
    ).float()
    ref = _ref_bidir(q, k, v, seq_lens, block_spans, scale)
    assert (out - ref).abs().max() < 2e-2
    # must differ from plain causal (mask has effect)
    ref_causal = _ref_bidir(q, k, v, seq_lens, [[] for _ in seq_lens], scale)
    assert (out - ref_causal).abs().max() > 0.1


def test_build_block_ids_none_and_validation():
    dev = "xpu" if torch.xpu.is_available() else "cpu"
    # No spans -> None (callers skip the mask).
    assert build_bidirectional_block_ids([4, 4], [[], []], dev) is None
    # Unique id per span, -1 elsewhere.
    ids = build_bidirectional_block_ids([5], [[(1, 2)]], dev)
    assert ids.tolist() == [-1, 0, 0, -1, -1]
    # Validation.
    with pytest.raises(ValueError):
        build_bidirectional_block_ids([4], [[], []], dev)  # length mismatch
    with pytest.raises(ValueError):
        build_bidirectional_block_ids([4], [[(2, 5)]], dev)  # end >= seq_len
    with pytest.raises(ValueError):
        build_bidirectional_block_ids([4], [[(3, 1)]], dev)  # begin > end


def test_rejections():
    if not torch.xpu.is_available():
        pytest.skip("XPU required")
    dev = "xpu"
    scale = 1.0 / (D**0.5)
    q, k, v, kc, vc, pt, cs, cuq = _paged_batch([128], dev)
    good = build_bidirectional_block_ids([128], [[(0, 63)]], dev)

    def run(block_ids, causal=False):
        return flash_attn_with_kvcache(
            q, kc, vc, cache_seqlens=cs, page_table=pt, cu_seqlens_q=cuq,
            max_seqlen_q=128, max_seqlen_k=128, softmax_scale=scale,
            causal=causal, bidirectional_block_ids=block_ids,
        )

    with pytest.raises(RuntimeError):
        run(good.float())  # wrong dtype
    with pytest.raises(RuntimeError):
        run(good[:64])  # wrong numel
    with pytest.raises(RuntimeError):
        run(good, causal=True)  # must be non-causal
