"""Bidirectional-block prefill attention (mm_block_id).

The prefill kernel takes a per-token block id (-1 = none) and, run non-causal,
re-imposes causal masking EXCEPT that keys sharing a query's non-negative block
id are attended both directions. This checks that against an fp32 torch
reference and self-checks the harness with the plain causal / full cases.
"""

import pytest
import torch

from sgl_kernel.flash_attn import flash_attn_with_kvcache


def _ref(q, k, v, scale, mask_allow, h_q, h_kv):
    # fp32 reference: mask_allow[i, j] True => query i attends key j.
    qf = q.float()
    kf = k.float().repeat_interleave(h_q // h_kv, dim=1)
    vf = v.float().repeat_interleave(h_q // h_kv, dim=1)
    out = torch.empty(q.shape[0], h_q, v.shape[-1], device=q.device, dtype=torch.float32)
    for h in range(h_q):
        sc = (qf[:, h] @ kf[:, h].T) * scale
        sc = sc.masked_fill(~mask_allow, float("-inf"))
        out[:, h] = torch.softmax(sc, dim=-1) @ vf[:, h]
    return out


@pytest.mark.parametrize("seqlen", [128, 256])
def test_bidirectional_block_mask(seqlen):
    if not torch.xpu.is_available():
        pytest.skip("XPU required")
    torch.manual_seed(0)
    dev = "xpu"
    S, h_q, h_kv, d = seqlen, 8, 4, 128  # GQA, head_dim 128
    blk_begin, blk_end = S // 4, S // 2  # one contiguous bidirectional block
    scale = 1.0 / (d**0.5)

    q = torch.randn(S, h_q, d, device=dev, dtype=torch.bfloat16)
    k = torch.randn(S, h_kv, d, device=dev, dtype=torch.bfloat16)
    v = torch.randn(S, h_kv, d, device=dev, dtype=torch.bfloat16)

    # Single page holds the whole sequence; page_size must be a multiple of the
    # K-tile (and page_size == 1 routes to the non-paged helper, which rejects
    # mm_block_id), so keep one page of length S.
    k_cache = k.view(1, S, h_kv, d).contiguous()
    v_cache = v.view(1, S, h_kv, d).contiguous()
    page_table = torch.zeros(1, 1, device=dev, dtype=torch.int32)
    cache_seqlens = torch.tensor([S], device=dev, dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, S], device=dev, dtype=torch.int32)

    block_id = torch.full((S,), -1, device=dev, dtype=torch.int32)
    block_id[blk_begin : blk_end + 1] = 0

    def run(causal, mm):
        return flash_attn_with_kvcache(
            q, k_cache, v_cache, cache_seqlens=cache_seqlens, page_table=page_table,
            cu_seqlens_q=cu_seqlens_q, max_seqlen_q=S, max_seqlen_k=S,
            softmax_scale=scale, causal=causal, mm_block_id=mm,
        ).float()

    rows = torch.arange(S, device=dev).view(S, 1)
    cols = torch.arange(S, device=dev).view(1, S)
    causal_allow = cols <= rows
    same_block = (block_id.view(S, 1) >= 0) & (block_id.view(S, 1) == block_id.view(1, S))
    bidir_allow = causal_allow | same_block

    tol = 2e-2
    # harness self-checks
    assert (run(True, None) - _ref(q, k, v, scale, causal_allow, h_q, h_kv)).abs().max() < tol
    assert (run(False, None) - _ref(q, k, v, scale, cols >= 0, h_q, h_kv)).abs().max() < tol
    # the bidirectional mask
    out = run(False, block_id)
    assert (out - _ref(q, k, v, scale, bidir_allow, h_q, h_kv)).abs().max() < tol
    # and it must actually differ from plain causal (mask has effect)
    assert (out - _ref(q, k, v, scale, causal_allow, h_q, h_kv)).abs().max() > 0.1
