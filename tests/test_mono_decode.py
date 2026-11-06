"""Single-card numerics of the mono decode layers (Qwen3.5 / Qwen3.6 MoE, one
TP2 rank; the all-reduces through a loopback mailbox, so they return twice the
rank's partial sums): the GatedDeltaNet and full-attention layer kernels (1, 2,
4, 8 rows) against a torch reference on SGLang's layouts (block-FP8 weights,
FusedMoE experts, separate K / V buffers with a page table, channel-major conv
state), in fp16 and bf16, plus the SGLang padding convention and the input
checks."""

import itertools

import pytest
import torch
from mono_decode_utils import (
    CONVD,
    HD,
    addr,
    attn_tensors,
    cos,
    cpu,
    gdn_caches,
    gdn_tensors,
    kv_caches,
    loopback_ar,
    loopback_ar_fp16,
    mailbox,
    ref_attn,
    ref_gdn,
    requires_mono,
    route_gap,
    rows,
)

DEVICE = "xpu:0"
F16, BF16 = torch.float16, torch.bfloat16
_ids = itertools.count(9000)


def _ops():
    return torch.ops.sgl_kernel


def _register(attn: bool, tensors: list, caches: list, page_size: int = 64):
    layer, mb = next(_ids), mailbox(DEVICE)
    _ops().mono_register_layer(layer, attn, addr(mb), addr(mb), tensors)
    _ops().mono_bind_caches(layer, caches, page_size)
    return layer, mb


def _run(layer, h, r, positions, m0, m1=None, m2=None):
    ho, ro = torch.empty_like(h), torch.empty_like(r)
    _ops().mono_decode_layer(layer, h, r, ho, ro, positions, m0, m1, m2)
    return ho, ro


def _check(got, ref, tol, what=""):
    c = cos(got, ref)
    assert torch.isfinite(got.float()).all() and c > tol, f"{what} cos {c:.7f}"


def _compare(ho, ro, ref, nb, tol_h=0.999, tol_r=0.9999):
    eh, er, info = ref
    gap = route_gap(info)
    for b in range(nb):
        _check(ro[b].cpu(), er[b], tol_r, f"res_out row {b}")
        if gap[b] > 0.02:  # a near tie of the 8th / 9th expert may pick either
            _check(ho[b].cpu(), eh[b], tol_h, f"hid_out row {b}")


@requires_mono
@pytest.mark.parametrize("dtype", [F16, BF16])
@pytest.mark.parametrize("nb", [1, 2, 4, 8])
def test_gdn_vs_reference(nb, dtype):
    gen = torch.Generator().manual_seed(nb + 10 * (dtype == BF16))
    t = gdn_tensors(gen, DEVICE, dtype)
    conv, ssm = gdn_caches(gen, DEVICE, 2 * nb + 1, dtype)
    h, r = rows(nb, gen, DEVICE, dtype)
    slots = torch.randperm(2 * nb, generator=gen)[:nb].to(torch.int32) + 1
    conv_ref, ssm_ref = conv.cpu(), ssm.cpu()
    layer, _mb = _register(False, t, [conv, ssm])
    ho, ro = _run(layer, h, r, torch.zeros(nb, dtype=torch.int64, device=DEVICE), slots.to(DEVICE))
    torch.xpu.synchronize()
    ar = loopback_ar if nb == 1 else loopback_ar_fp16
    ref = ref_gdn(cpu(t), conv_ref, ssm_ref, h.cpu(), r.cpu(), slots, dtype, ar)
    _compare(ho, ro, ref, nb)
    for sl in slots.tolist():
        _check(ssm[sl].cpu(), ssm_ref[sl], 0.99999, "ssm state")
        assert torch.equal(conv[sl, :, :2].cpu(), conv_ref[sl, :, :2])  # shifted window
        _check(conv[sl, :, 2].cpu(), conv_ref[sl, :, 2], 0.9999, "conv input")
    untouched = [s for s in range(2 * nb + 1) if s not in slots.tolist()]
    assert torch.equal(conv[untouched].cpu(), conv_ref[untouched])


def _attn_case(gen, lc, page_size):
    """Request b: lc[b] cached tokens in its own pages (shuffled page ids), one
    new token after them. Returns the page table, sequence lengths, the new
    tokens' slots, positions and the number of slots."""
    nreq = len(lc)
    npr = (max(lc) + 1 + page_size - 1) // page_size + 1
    perm = torch.randperm(nreq * npr, generator=gen).to(torch.int32) + 1  # page 0: SGLang's padding page
    pt = perm.view(nreq, npr)
    slot = [int(pt[b, lc[b] // page_size]) * page_size + lc[b] % page_size for b in range(nreq)]
    return (
        pt,
        torch.tensor([c + 1 for c in lc], dtype=torch.int32),
        torch.tensor(slot, dtype=torch.int64),
        torch.tensor(lc, dtype=torch.int64),
        (nreq * npr + 1) * page_size,
    )


ATTN_LENGTHS = [
    [0],
    [1],
    [63],
    [1000],
    [5000],
    [1000, 963],
    [1000, 963, 926, 889],
    [1000, 963, 926, 889, 852, 815, 778, 741],
    [0, 63, 64, 65, 1000, 4097, 9001, 40000],
    [50001, 47000, 0, 63],
]


@requires_mono
@pytest.mark.parametrize(
    "lc,page_size,dtype",
    [(lc, 64, F16) for lc in ATTN_LENGTHS]
    + [(lc, 64, BF16) for lc in ATTN_LENGTHS[3::2]]
    + [(ATTN_LENGTHS[i], 128, BF16) for i in (3, 7, 8)],
)
def test_attn_vs_reference(lc, page_size, dtype):
    nb = len(lc)
    gen = torch.Generator().manual_seed(100 + sum(lc) + page_size)
    pt, seq, slot, pos, nslots = _attn_case(gen, lc, page_size)
    t = attn_tensors(gen, DEVICE, dtype)
    kc, vc = kv_caches(gen, DEVICE, nslots, dtype)
    h, r = rows(nb, gen, DEVICE, dtype)
    kc_ref, vc_ref = kc.cpu(), vc.cpu()
    layer, _mb = _register(True, t, [kc, vc], page_size)
    ho, ro = _run(layer, h, r, pos.to(DEVICE), pt.to(DEVICE), seq.to(DEVICE), slot.to(DEVICE))
    torch.xpu.synchronize()
    ar = loopback_ar if nb == 1 else loopback_ar_fp16
    ref = ref_attn(cpu(t), kc_ref, vc_ref, h.cpu(), r.cpu(), pt, seq, slot, pos, page_size, dtype, ar)
    _compare(ho, ro, ref, nb)
    _check(kc[slot].cpu(), kc_ref[slot], 0.9999, "new k rows")
    _check(vc[slot].cpu(), vc_ref[slot], 0.9999, "new v rows")


@requires_mono
@pytest.mark.parametrize("ctx,dtype", [(c, F16) for c in (40959, 40960, 40961, 50001, 131109)] + [(50001, BF16)])
def test_attn_long_context(ctx, dtype):
    """The one-row kernel covers 40960 cached tokens a pass; small residual
    rows, so res_out - (h + r) is the attention's contribution."""
    gen = torch.Generator().manual_seed(ctx)
    for lc in ([ctx], [ctx, 63]):
        pt, seq, slot, pos, nslots = _attn_case(gen, lc, 64)
        t = attn_tensors(gen, DEVICE, dtype)
        kc, vc = kv_caches(gen, DEVICE, nslots, dtype)
        h, r = rows(len(lc), gen, DEVICE, dtype)
        h, r = h * 0.01, r * 0.01
        kc_ref, vc_ref = kc.cpu(), vc.cpu()
        layer, _mb = _register(True, t, [kc, vc])
        _, ro = _run(layer, h, r, pos.to(DEVICE), pt.to(DEVICE), seq.to(DEVICE), slot.to(DEVICE))
        torch.xpu.synchronize()
        ar = loopback_ar if len(lc) == 1 else loopback_ar_fp16
        _, er, _ = ref_attn(cpu(t), kc_ref, vc_ref, h.cpu(), r.cpu(), pt, seq, slot, pos, 64, dtype, ar)
        hr = h.cpu().float() + r.cpu().float()
        _check(ro[0].cpu().float() - hr[0], er[0] - hr[0], 0.999, f"mixer of {lc}")


@requires_mono
@pytest.mark.parametrize("dtype", [F16, BF16])
def test_padded_rows(dtype):
    """A batch of 3 requests run by the 4-row kernels, padded like SGLang's
    graphs: KV slot 0, sequence length 1, page table row 0, state slot 0. The
    real rows match the reference; the pads write only slot 0."""
    gen = torch.Generator().manual_seed(77)
    lc = [8, 1007, 70]
    pt, seq, slot, pos, nslots = _attn_case(gen, lc, 64)
    t = attn_tensors(gen, DEVICE, dtype)
    kc, vc = kv_caches(gen, DEVICE, nslots, dtype)
    h, r = rows(4, gen, DEVICE, dtype)
    kc_ref, vc_ref = kc.cpu(), vc.cpu()
    layer, _mb = _register(True, t, [kc, vc])
    pt4 = torch.cat([pt, torch.zeros_like(pt[:1])])
    seq4 = torch.cat([seq, torch.ones_like(seq[:1])])
    slot4 = torch.cat([slot, torch.zeros_like(slot[:1])])
    pos4 = torch.cat([pos, torch.zeros_like(pos[:1])])
    ho, ro = _run(layer, h, r, pos4.to(DEVICE), pt4.to(DEVICE), seq4.to(DEVICE), slot4.to(DEVICE))
    torch.xpu.synchronize()
    ref = ref_attn(cpu(t), kc_ref, vc_ref, h[:3].cpu(), r[:3].cpu(), pt, seq, slot, pos, 64, dtype, loopback_ar_fp16)
    _compare(ho[:3], ro[:3], ref, 3)
    changed = (kc.cpu() != kc_ref).any(-1).flatten().nonzero().flatten().tolist()
    assert set(changed) <= set(slot.tolist()) | {0}

    tg = gdn_tensors(gen, DEVICE, dtype)
    conv, ssm = gdn_caches(gen, DEVICE, 6, dtype)
    conv_ref, ssm_ref = conv.cpu(), ssm.cpu()
    layer, _mb = _register(False, tg, [conv, ssm])
    slots = torch.tensor([3, 1, 5, 0], dtype=torch.int32)
    ho, ro = _run(layer, h, r, pos4.to(DEVICE), slots.to(DEVICE))
    torch.xpu.synchronize()
    ref = ref_gdn(cpu(tg), conv_ref, ssm_ref, h[:3].cpu(), r[:3].cpu(), slots[:3], dtype, loopback_ar_fp16)
    _compare(ho[:3], ro[:3], ref, 3)
    for sl in (2, 4):
        assert torch.equal(conv[sl].cpu(), conv_ref[sl]) and torch.equal(ssm[sl].cpu(), ssm_ref[sl])


@requires_mono
def test_bf16_scales_accepted():
    """Block scales in the checkpoint dtype (bf16) are converted at
    registration: the same results as fp32 scales of the same values."""
    gen = torch.Generator().manual_seed(5)
    t = gdn_tensors(gen, DEVICE, BF16)
    for i in (5, 7, 9, 11, 13, 15):
        t[i] = t[i].to(BF16).float()
    tb = [x.to(BF16) if i in (5, 7, 9, 11, 13, 15) else x for i, x in enumerate(t)]
    h, r = rows(2, gen, DEVICE, BF16)
    out = []
    for tt in (t, tb):
        conv, ssm = gdn_caches(torch.Generator().manual_seed(6), DEVICE, 3, BF16)
        layer, _mb = _register(False, tt, [conv, ssm])
        out.append(_run(layer, h, r, torch.zeros(2, dtype=torch.int64, device=DEVICE), torch.tensor([1, 2], dtype=torch.int32, device=DEVICE)))
    torch.xpu.synchronize()
    assert torch.equal(out[0][0], out[1][0]) and torch.equal(out[0][1], out[1][1])


@requires_mono
def test_max_context_covers_the_model():
    # Qwen3.5 / 3.6: max_position_embeddings 262144
    assert _ops().mono_max_context() >= 262144


@requires_mono
def test_input_checks():
    gen = torch.Generator().manual_seed(8)
    for dtype, other in ((F16, BF16), (BF16, F16)):
        t = attn_tensors(gen, DEVICE, dtype)
        kc, vc = kv_caches(gen, DEVICE, 256, other)
        layer, mb = next(_ids), mailbox(DEVICE)
        _ops().mono_register_layer(layer, True, addr(mb), addr(mb), t)
        with pytest.raises(RuntimeError, match="mix of activations and KV"):
            _ops().mono_bind_caches(layer, [kc, vc], 64)
        tg = gdn_tensors(gen, DEVICE, dtype)
        conv, ssm = gdn_caches(gen, DEVICE, 2, other)
        layer, mb = next(_ids), mailbox(DEVICE)
        _ops().mono_register_layer(layer, False, addr(mb), addr(mb), tg)
        with pytest.raises(RuntimeError, match="conv state"):
            _ops().mono_bind_caches(layer, [conv, ssm], 64)
        tg[1] = tg[1].to(other)
        with pytest.raises(RuntimeError, match="the layer's dtype"):
            _ops().mono_register_layer(next(_ids), False, addr(mb), addr(mb), tg)
        tg = gdn_tensors(gen, DEVICE, dtype)
        tg[6] = tg[6].transpose(1, 2)  # an [E][K][N] expert layout
        with pytest.raises(RuntimeError, match="w13"):
            _ops().mono_register_layer(next(_ids), False, addr(mb), addr(mb), tg)
        conv, ssm = gdn_caches(gen, DEVICE, 2, dtype)
        layer, _mb = _register(False, gdn_tensors(gen, DEVICE, dtype), [conv, ssm])
        h, r = rows(3, gen, DEVICE, dtype)
        zero = torch.zeros(3, dtype=torch.int32, device=DEVICE)
        with pytest.raises(RuntimeError, match="supported: 1, 2, 4, 8"):
            _run(layer, h, r, zero.long(), zero)
        h, r = rows(1, gen, DEVICE, other)
        with pytest.raises(RuntimeError, match="the layer's dtype"):
            _run(layer, h, r, zero[:1].long(), zero[:1])
