"""inkling_gate_topk_renorm tests against an fp64 oracle.

Coverage:
- Production shape (256 routed + 2 sink, k=6) across token counts, including a
  non-multiple-of-BLOCK_M tail and the strided [T, 258]-of-[T, 264] layout
- Narrow rows (BLOCK_M > 1), bf16 logits, M == 0
- Ties (lowest expert id wins), a dominating sink, and all-underflow logits,
  where sigmoid(x) / sum(sigmoid(x)) divides 0/0 and returns NaN
- Packed mode is bitwise the manual (id << 16) | bf16_bits(weight) pack
"""

import sys

import pytest
import torch
import utils
from sgl_kernel import inkling_gate_topk_renorm

device = utils.get_device()

ROUTE_SCALE = 2.5


def reference(logits, k, n_shared, route_scale, global_scale, bias, idx=None):
    """fp64 oracle. Ranks unless `idx` is given, then renormalizes at `idx`."""
    x = logits.double().cpu()
    N = x.shape[1] - n_shared
    sel = torch.sigmoid(x[:, :N]) + bias.double().cpu()
    if idx is None:
        # Stable descending sort: equal scores keep ascending expert id.
        idx = torch.sort(sel, dim=1, descending=True, stable=True).indices[:, :k]
    idx = idx.cpu().long()
    active = torch.cat([torch.gather(x, 1, idx), x[:, N:]], dim=1)
    lp = torch.nn.functional.logsigmoid(active)
    w = torch.exp(lp - torch.logsumexp(lp, dim=1, keepdim=True))
    w = w * route_scale * global_scale.double().cpu()
    return w[:, :k], idx.to(torch.int32), w[:, k:], sel


def make_inputs(M, N, S, dtype=torch.float32, pad=0, scale=4.0):
    base = torch.randn(M, N + S + pad, device=device) * scale
    logits = base.to(dtype)[:, : N + S]  # column stride 1, row stride N + S + pad
    bias = torch.randn(N, device=device) * 0.1
    global_scale = torch.tensor([1.3], device=device)
    return logits, bias, global_scale


def check(logits, k, S, bias, global_scale, atol=2e-3):
    w, idx, sw, packed = inkling_gate_topk_renorm(
        logits, k, S, ROUTE_SCALE, global_scale, bias
    )
    assert packed is None
    assert w.dtype == logits.dtype and sw.dtype == logits.dtype
    _, ref_idx, _, sel = reference(logits, k, S, ROUTE_SCALE, global_scale, bias)
    # The kernel ranks in fp32, where scores that differ in fp64 can tie (e.g.
    # sigmoid rounding to exactly 1.0), so require a valid top-k up to fp32
    # resolution rather than the fp64 ranking bit for bit.
    idx_cpu = idx.cpu().long()
    assert all(len(set(r)) == k for r in idx_cpu.tolist()), "duplicate expert"
    picked = torch.gather(sel, 1, idx_cpu)
    best = torch.gather(sel, 1, ref_idx.long())
    torch.testing.assert_close(picked, best, rtol=0, atol=1e-6)
    ref_w, _, ref_sw, _ = reference(
        logits, k, S, ROUTE_SCALE, global_scale, bias, idx=idx
    )
    torch.testing.assert_close(w.cpu().double(), ref_w, rtol=atol, atol=atol)
    torch.testing.assert_close(sw.cpu().double(), ref_sw, rtol=atol, atol=atol)
    return w, sw


@pytest.mark.parametrize("M", [1, 8, 37, 64, 512, 4096])
@pytest.mark.parametrize("pad", [0, 6])
def test_production_shape(M, pad):
    logits, bias, gs = make_inputs(M, 256, 2, pad=pad)
    w, sw = check(logits, 6, 2, bias, gs)
    total = (w.sum(1) + sw.sum(1)).cpu().double()
    torch.testing.assert_close(
        total, torch.full_like(total, ROUTE_SCALE * 1.3), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("N,S,k", [(64, 1, 4), (32, 2, 2), (128, 3, 8)])
def test_other_shapes(N, S, k):
    logits, bias, gs = make_inputs(37, N, S)
    check(logits, k, S, bias, gs)


def test_bf16_logits():
    logits, bias, gs = make_inputs(64, 256, 2, dtype=torch.bfloat16)
    check(logits, 6, 2, bias, gs, atol=2e-2)


def test_ties_lowest_id_wins():
    logits, bias, gs = make_inputs(16, 256, 2)
    logits[:, :256] = 0.5
    bias.zero_()
    _, idx, _, _ = inkling_gate_topk_renorm(logits, 6, 2, ROUTE_SCALE, gs, bias)
    assert torch.equal(idx.cpu(), torch.arange(6, dtype=torch.int32).expand(16, 6))
    check(logits, 6, 2, bias, gs)


def test_dominating_sink():
    logits, bias, gs = make_inputs(16, 256, 2)
    logits[:, 256] = 60.0
    logits[:, :256] = -30.0
    check(logits, 6, 2, bias, gs)


def test_underflow_is_finite():
    # Every active logit below the fp32 sigmoid flush-to-zero point.
    logits, bias, gs = make_inputs(16, 256, 2, scale=1.0)
    logits.sub_(200.0)
    w, sw = check(logits, 6, 2, bias, gs)
    assert torch.isfinite(w).all() and torch.isfinite(sw).all()


def test_packed_matches_manual_pack():
    logits, bias, gs = make_inputs(37, 256, 2)
    w, idx, sw, _ = inkling_gate_topk_renorm(logits, 6, 2, ROUTE_SCALE, gs, bias)
    pw, pidx, psw, packed = inkling_gate_topk_renorm(
        logits, 6, 2, ROUTE_SCALE, gs, bias, return_packed_topk=True
    )
    assert pw is None and pidx is None
    bits = w.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
    assert torch.equal(packed, (idx << 16) | bits)
    assert torch.equal(psw, sw)


def test_empty():
    logits, bias, gs = make_inputs(0, 256, 2)
    w, idx, sw, packed = inkling_gate_topk_renorm(logits, 6, 2, ROUTE_SCALE, gs, bias)
    assert w.shape == (0, 6) and idx.shape == (0, 6) and sw.shape == (0, 2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
