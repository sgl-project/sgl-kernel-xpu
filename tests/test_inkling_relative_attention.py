import math

import pytest
import torch

from sgl_kernel import inkling_relative_attention

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "xpu") or not torch.xpu.is_available(),
    reason="Inkling relative attention tests require XPU",
)


def _lengths(batch, max_seq_len, *, varied=False, zero_last=False):
    if not varied:
        vals = [max_seq_len] * batch
    else:
        span = max(1, max_seq_len // 2)
        vals = [max(1, max_seq_len - ((b * 7 + 3) % span)) for b in range(batch)]
        if batch > 1:
            vals[1] = min(vals[1], max(1, max_seq_len // 4))
    if zero_last:
        vals[-1] = 0
    return vals


def _make_case(
    *,
    batch,
    max_seq_len,
    heads,
    kv_heads,
    d,
    dv,
    rel_len,
    decode_tail=False,
    varied=False,
    zero_last=False,
    dtype=torch.bfloat16,
    seed=0,
):
    torch.manual_seed(seed)
    lengths = _lengths(batch, max_seq_len, varied=varied, zero_last=zero_last)
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)

    q_to_seq = []
    q_pos = []
    for seq, length in enumerate(lengths):
        q_len = 1 if decode_tail and length > 0 else length
        q_start = length - q_len
        for q in range(q_len):
            q_to_seq.append(seq)
            q_pos.append(q_start + q)

    device = torch.device("xpu")
    total_q = len(q_to_seq)
    total_k = cu[-1]
    q = (torch.randn((total_q, heads, d), device=device) * 0.25).to(dtype)
    k = (torch.randn((total_k, kv_heads, d), device=device) * 0.25).to(dtype)
    v = (torch.randn((total_k, kv_heads, dv), device=device) * 0.25).to(dtype)
    rel_bias = (
        torch.randn((total_q, heads, rel_len), device=device, dtype=torch.float32)
        * 0.05
        if rel_len > 0
        else None
    )
    return (
        q,
        k,
        v,
        torch.tensor(q_to_seq, dtype=torch.int32, device=device),
        torch.tensor(q_pos, dtype=torch.int32, device=device),
        torch.tensor(cu, dtype=torch.int32, device=device),
        rel_bias,
    )


def _reference(
    q,
    k,
    v,
    q_to_seq,
    q_pos,
    cu_k,
    rel_bias,
    *,
    softmax_scale,
    causal,
    window_size,
    softcap=0.0,
):
    q_cpu = q.detach().cpu().float()
    k_cpu = k.detach().cpu().float()
    v_cpu = v.detach().cpu().float()
    q_to_seq_cpu = q_to_seq.detach().cpu().tolist()
    q_pos_cpu = q_pos.detach().cpu().tolist()
    cu_cpu = cu_k.detach().cpu().tolist()
    bias_cpu = None if rel_bias is None else rel_bias.detach().cpu().float()

    total_q, heads, _ = q_cpu.shape
    kv_heads = k_cpu.shape[1]
    dv = v_cpu.shape[2]
    out = torch.empty((total_q, heads, dv), dtype=torch.float32)
    lse = torch.empty((total_q, heads), dtype=torch.float32)
    kv_group = heads // kv_heads
    for q_row in range(total_q):
        seq = q_to_seq_cpu[q_row]
        begin, end = cu_cpu[seq], cu_cpu[seq + 1]
        kv_len = end - begin
        pos = q_pos_cpu[q_row]
        for head in range(heads):
            kv_head = head // kv_group
            scores = []
            valid_indices = []
            for k_local in range(kv_len):
                valid = True
                if causal:
                    valid = valid and k_local <= pos
                if window_size[0] >= 0:
                    valid = valid and k_local >= pos - window_size[0]
                if window_size[1] >= 0:
                    valid = valid and k_local <= pos + window_size[1]
                if not valid:
                    continue
                score = torch.dot(q_cpu[q_row, head], k_cpu[begin + k_local, kv_head])
                score = score * softmax_scale
                if bias_cpu is not None:
                    rel = pos - k_local
                    if 0 <= rel < bias_cpu.shape[2]:
                        score = score + bias_cpu[q_row, head, rel]
                if softcap > 0:
                    score = softcap * torch.tanh(score / softcap)
                scores.append(score)
                valid_indices.append(k_local)
            if not scores:
                out[q_row, head].zero_()
                lse[q_row, head] = -float("inf")
                continue
            scores = torch.stack(scores)
            probs = torch.softmax(scores, dim=0)
            acc = torch.zeros((dv,), dtype=torch.float32)
            for prob, k_local in zip(probs, valid_indices):
                acc += prob * v_cpu[begin + k_local, kv_head]
            out[q_row, head] = acc
            lse[q_row, head] = torch.logsumexp(scores, dim=0)
    return out.to(q.device), lse.to(q.device)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "case",
    [
        dict(batch=1, max_seq_len=5, heads=2, kv_heads=2, d=8, dv=7, rel_len=0),
        dict(
            batch=4,
            max_seq_len=17,
            heads=6,
            kv_heads=2,
            d=63,
            dv=19,
            rel_len=9,
            varied=True,
            decode_tail=True,
        ),
        dict(
            batch=3,
            max_seq_len=29,
            heads=8,
            kv_heads=2,
            d=64,
            dv=33,
            rel_len=0,
            varied=True,
        ),
        dict(
            batch=2,
            max_seq_len=64,
            heads=4,
            kv_heads=1,
            d=128,
            dv=128,
            rel_len=64,
            varied=True,
        ),
    ],
)
def test_relative_attention_matches_reference(dtype, case):
    q, k, v, q_to_seq, q_pos, cu_k, rel_bias = _make_case(
        dtype=dtype, seed=11, **case
    )
    softmax_scale = q.shape[-1] ** -0.5
    window = (7, 0) if case.get("rel_len", 0) == 0 and case["max_seq_len"] > 8 else (-1, -1)
    out, lse = inkling_relative_attention(
        q,
        k,
        v,
        q_to_seq,
        q_pos,
        cu_k,
        rel_bias=rel_bias,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=window,
        return_softmax_lse=True,
    )
    ref, ref_lse = _reference(
        q,
        k,
        v,
        q_to_seq,
        q_pos,
        cu_k,
        rel_bias,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=window,
    )
    atol = 6e-2 if dtype is torch.bfloat16 else 1e-2
    rtol = 6e-2 if dtype is torch.bfloat16 else 1e-2
    torch.testing.assert_close(out.float(), ref.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(lse, ref_lse, atol=atol, rtol=rtol)


def test_relative_attention_supports_noncausal_softcap_and_out():
    q, k, v, q_to_seq, q_pos, cu_k, rel_bias = _make_case(
        batch=2,
        max_seq_len=13,
        heads=4,
        kv_heads=4,
        d=24,
        dv=16,
        rel_len=11,
        varied=True,
        dtype=torch.bfloat16,
        seed=19,
    )
    out_buf = torch.empty_like(v.new_empty((q.shape[0], q.shape[1], v.shape[2])))
    out, lse = inkling_relative_attention(
        q,
        k,
        v,
        q_to_seq,
        q_pos,
        cu_k,
        rel_bias=rel_bias,
        softmax_scale=1.0 / math.sqrt(q.shape[-1]),
        causal=False,
        window_size=(3, 2),
        softcap=6.0,
        out=out_buf,
        return_softmax_lse=True,
    )
    ref, ref_lse = _reference(
        q,
        k,
        v,
        q_to_seq,
        q_pos,
        cu_k,
        rel_bias,
        softmax_scale=1.0 / math.sqrt(q.shape[-1]),
        causal=False,
        window_size=(3, 2),
        softcap=6.0,
    )
    assert out.data_ptr() == out_buf.data_ptr()
    torch.testing.assert_close(out.float(), ref.float(), atol=6e-2, rtol=6e-2)
    torch.testing.assert_close(lse, ref_lse, atol=6e-2, rtol=6e-2)
