"""Correctness tests for DeepSeek V4 sparse MLA prefill (ported from xattention).

Run: pytest tests/test_flash_mla_sparse_prefill.py
"""

import pytest
import torch
from sgl_kernel import flash_mla_sparse_prefill

default_params = {
    "has_attn_sink": [False, True],
    "has_topk_length": [False, True],
    "d_qk": [512],
    "topk": [6, 512],
    "h_q": [6, 22, 64, 96],
    "s_q": [1, 32, 128],
}


# https://github.com/deepseek-ai/FlashMLA/blob/main/tests/ref.py#L7
def _merge_two_lse(lse0, lse1, s_q, h_q):
    if lse1 is None:
        return lse0
    return torch.logsumexp(
        torch.stack([lse0.view(s_q, h_q), lse1.broadcast_to(s_q, h_q)], dim=0),
        dim=0,
    )


# Adapted from https://github.com/deepseek-ai/FlashMLA/blob/main/tests/ref.py#L19
def reference_mla_sparse_prefill(
    q,
    kv,
    indices,
    sm_scale,
    d_v,
    topk_length=None,
    attn_sink=None,
):
    """Returns (o, o_fp32, max_logits, lse)."""
    s_q, h_q, d_qk = q.shape
    s_kv, _, _ = kv.shape
    _, _, topk = indices.shape

    indices = indices.clone().squeeze(1)
    if topk_length is not None:
        mask = torch.arange(topk, device=topk_length.device).unsqueeze(0).broadcast_to(
            s_q, topk
        ) >= topk_length.unsqueeze(1)
        indices[mask] = -1
    invalid_mask = (indices < 0) | (indices >= s_kv)
    indices[invalid_mask] = 0

    q = q.float()
    gathered_kv = (
        kv.index_select(dim=0, index=indices.flatten()).reshape(s_q, topk, d_qk).float()
    )
    P = q @ gathered_kv.transpose(1, 2)
    P *= sm_scale
    P[invalid_mask.unsqueeze(1).broadcast_to(P.shape)] = float("-inf")

    orig_lse = torch.logsumexp(P, dim=-1)
    max_logits = P.max(dim=-1).values

    lse_for_o = _merge_two_lse(orig_lse, attn_sink, s_q, h_q)
    lse_for_o = lse_for_o.clone()
    lse_for_o[lse_for_o == float("-inf")] = float("+inf")
    s_for_o = torch.exp(P - lse_for_o.unsqueeze(-1))
    out = s_for_o @ gathered_kv[..., :d_v]

    lonely_q_mask = orig_lse == float("-inf")
    orig_lse[lonely_q_mask] = float("+inf")
    return (out.to(kv.dtype), out, max_logits, orig_lse)


@pytest.mark.parametrize("has_attn_sink", default_params["has_attn_sink"])
@pytest.mark.parametrize("has_topk_length", default_params["has_topk_length"])
@pytest.mark.parametrize("topk", default_params["topk"])
@pytest.mark.parametrize("d_qk", default_params["d_qk"])
@pytest.mark.parametrize("h_q", default_params["h_q"])
@pytest.mark.parametrize("s_q", default_params["s_q"])
def test_flash_mla_sparse_prefill_fwd(
    s_q, h_q, d_qk, topk, has_topk_length, has_attn_sink
):
    if not torch.xpu.is_available():
        pytest.skip("XPU not available")

    device = "xpu"
    dtype = torch.bfloat16

    torch.manual_seed(88)

    s_kv = 16384
    h_kv = 1
    d_v = 512

    q = torch.randn((s_q, h_q, d_qk), device=device, dtype=dtype)
    kv = torch.randn((s_kv, h_kv, d_qk), device=device, dtype=dtype)
    indices = torch.full((s_q, h_kv, topk), s_kv, dtype=torch.int32, device=device)

    for t in range(s_q):
        for h in range(h_kv):
            i_i = torch.randperm(max(1, t))[:topk]
            indices[t, h, : len(i_i)] = i_i

    sm_scale = d_qk**-0.5

    topk_length = None
    if has_topk_length:
        topk_length = torch.randint(
            1, topk + 1, (s_q,), device=device, dtype=torch.int32
        )

    attn_sink = None
    if has_attn_sink:
        attn_sink = torch.randn((h_q,), device=device, dtype=torch.float32)

    ref_out, _, ref_max_logits, ref_lse = reference_mla_sparse_prefill(
        q, kv, indices, sm_scale, d_v, topk_length=topk_length, attn_sink=attn_sink
    )

    out, max_logits, lse = flash_mla_sparse_prefill(
        q,
        kv,
        indices,
        sm_scale=sm_scale,
        d_v=d_v,
        attn_sink=attn_sink,
        topk_length=topk_length,
        return_softmax_lse=True,
    )

    assert out.shape == (s_q, h_q, d_v)
    assert max_logits.shape == (s_q, h_q)
    assert lse.shape == (s_q, h_q)

    torch.testing.assert_close(max_logits, ref_max_logits, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(lse, ref_lse, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)

    # No-lse path returns just the output.
    out_only = flash_mla_sparse_prefill(
        q,
        kv,
        indices,
        sm_scale=sm_scale,
        d_v=d_v,
        attn_sink=attn_sink,
        topk_length=topk_length,
        return_softmax_lse=False,
    )
    torch.testing.assert_close(out_only, ref_out, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
