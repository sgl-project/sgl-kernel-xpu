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
    ho, ro = _run(
        layer, h, r, torch.zeros(nb, dtype=torch.int64, device=DEVICE), slots.to(DEVICE)
    )
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
    perm = (
        torch.randperm(nreq * npr, generator=gen).to(torch.int32) + 1
    )  # page 0: SGLang's padding page
    pt = perm.view(nreq, npr)
    slot = [
        int(pt[b, lc[b] // page_size]) * page_size + lc[b] % page_size
        for b in range(nreq)
    ]
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
    ho, ro = _run(
        layer, h, r, pos.to(DEVICE), pt.to(DEVICE), seq.to(DEVICE), slot.to(DEVICE)
    )
    torch.xpu.synchronize()
    ar = loopback_ar if nb == 1 else loopback_ar_fp16
    ref = ref_attn(
        cpu(t),
        kc_ref,
        vc_ref,
        h.cpu(),
        r.cpu(),
        pt,
        seq,
        slot,
        pos,
        page_size,
        dtype,
        ar,
    )
    _compare(ho, ro, ref, nb)
    _check(kc[slot].cpu(), kc_ref[slot], 0.9999, "new k rows")
    _check(vc[slot].cpu(), vc_ref[slot], 0.9999, "new v rows")


@requires_mono
@pytest.mark.parametrize(
    "ctx,dtype",
    [(c, F16) for c in (40959, 40960, 40961, 50001, 131109)] + [(50001, BF16)],
)
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
        _, ro = _run(
            layer, h, r, pos.to(DEVICE), pt.to(DEVICE), seq.to(DEVICE), slot.to(DEVICE)
        )
        torch.xpu.synchronize()
        ar = loopback_ar if len(lc) == 1 else loopback_ar_fp16
        _, er, _ = ref_attn(
            cpu(t), kc_ref, vc_ref, h.cpu(), r.cpu(), pt, seq, slot, pos, 64, dtype, ar
        )
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
    ho, ro = _run(
        layer, h, r, pos4.to(DEVICE), pt4.to(DEVICE), seq4.to(DEVICE), slot4.to(DEVICE)
    )
    torch.xpu.synchronize()
    ref = ref_attn(
        cpu(t),
        kc_ref,
        vc_ref,
        h[:3].cpu(),
        r[:3].cpu(),
        pt,
        seq,
        slot,
        pos,
        64,
        dtype,
        loopback_ar_fp16,
    )
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
    ref = ref_gdn(
        cpu(tg),
        conv_ref,
        ssm_ref,
        h[:3].cpu(),
        r[:3].cpu(),
        slots[:3],
        dtype,
        loopback_ar_fp16,
    )
    _compare(ho[:3], ro[:3], ref, 3)
    for sl in (2, 4):
        assert torch.equal(conv[sl].cpu(), conv_ref[sl]) and torch.equal(
            ssm[sl].cpu(), ssm_ref[sl]
        )


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
        out.append(
            _run(
                layer,
                h,
                r,
                torch.zeros(2, dtype=torch.int64, device=DEVICE),
                torch.tensor([1, 2], dtype=torch.int32, device=DEVICE),
            )
        )
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
        h, r = rows(5, gen, DEVICE, dtype)
        zero = torch.zeros(5, dtype=torch.int32, device=DEVICE)
        with pytest.raises(RuntimeError, match="supported: 1, 2, 3, 4, 6, 8"):
            _run(layer, h, r, zero.long(), zero)
        ho, ro = torch.empty_like(h), torch.empty_like(r)
        with pytest.raises(RuntimeError, match="no verify scratch"):
            _ops().mono_decode_layer_verify(
                layer, 5, h, r, ho, ro, zero.long(), zero, None, None
            )
        h, r = rows(1, gen, DEVICE, other)
        with pytest.raises(RuntimeError, match="the layer's dtype"):
            _run(layer, h, r, zero[:1].long(), zero[:1])


VERIFY_SHAPES = [
    (2, 2),
    (4, 2),
    (6, 2),
    (8, 2),
    (3, 3),
    (6, 3),
    (4, 4),
    (8, 4),
    (6, 6),
    (8, 8),
]


def _scratch(nrows, tokens, dtype):
    """SGLang's verify scratch: intermediate ssm [rows, T, 16, 128, 128] and the
    windows, a [rows, 4096, T + 2] buffer viewed as [rows, T, 4096, 3]."""
    iss = torch.zeros((nrows, tokens, 16, 128, 128), dtype=torch.float32, device=DEVICE)
    buf = torch.zeros((nrows, CONVD, tokens + 2), dtype=dtype, device=DEVICE)
    win = buf.as_strided(
        (nrows, tokens, CONVD, 3), (CONVD * (tokens + 2), 1, tokens + 2, 1)
    )
    return iss, win


@requires_mono
@pytest.mark.parametrize("dtype", [F16, BF16])
@pytest.mark.parametrize("nb,tokens", VERIFY_SHAPES)
def test_gdn_verify_vs_reference(nb, tokens, dtype):
    if dtype == BF16 and (nb, tokens) not in ((4, 2), (6, 3), (8, 8)):
        pytest.skip("bf16: a subset of the shapes")
    nreq = nb // tokens
    gen = torch.Generator().manual_seed(200 + nb * 10 + tokens)
    t = gdn_tensors(gen, DEVICE, dtype)
    conv, ssm = gdn_caches(gen, DEVICE, nreq + 2, dtype)
    iss, win = _scratch(nreq + 1, tokens, dtype)
    h, r = rows(nb, gen, DEVICE, dtype)
    slots = torch.randperm(nreq + 1, generator=gen)[:nreq].to(torch.int32) + 1
    conv0, ssm0 = conv.cpu(), ssm.cpu()
    layer, mb = next(_ids), mailbox(DEVICE)
    _ops().mono_register_layer(layer, False, addr(mb), addr(mb), t)
    _ops().mono_bind_caches(layer, [conv, ssm, iss, win], 64)
    ho, ro = torch.empty_like(h), torch.empty_like(r)
    assert _ops().mono_decode_layer_verify(
        layer,
        tokens,
        h,
        r,
        ho,
        ro,
        torch.zeros(nb, dtype=torch.int64, device=DEVICE),
        slots.to(DEVICE),
        None,
        None,
    )
    torch.xpu.synchronize()
    tc = cpu(t)
    for b in range(nreq):
        cs, ss = (
            conv0[int(slots[b]) : int(slots[b]) + 1].clone(),
            ssm0[int(slots[b]) : int(slots[b]) + 1].clone(),
        )
        for i in range(tokens):
            row = b * tokens + i
            ref = ref_gdn(
                tc,
                cs,
                ss,
                h[row : row + 1].cpu(),
                r[row : row + 1].cpu(),
                [0],
                dtype,
                loopback_ar_fp16,
            )
            _compare(ho[row : row + 1], ro[row : row + 1], ref, 1)
            _check(iss[b, i].cpu(), ss[0], 0.99999, f"state after token {i}")
            _check(win[b, i].cpu(), cs[0], 0.9999, f"window after token {i}")
    assert torch.equal(conv.cpu(), conv0) and torch.equal(
        ssm.cpu(), ssm0
    )  # the committed state is not written


@requires_mono
@pytest.mark.parametrize(
    "nb,tokens,lc,dtype",
    [(nb, t, [1000 - 37 * b for b in range(nb // t)], F16) for nb, t in VERIFY_SHAPES]
    + [
        (8, 4, [63, 4097], BF16),
        (8, 2, [0, 65, 1000, 9001], BF16),
        (6, 3, [64, 40000], F16),
        (8, 8, [40001], F16),
    ],
)
def test_attn_verify_vs_reference(nb, tokens, lc, dtype):
    nreq = nb // tokens
    gen = torch.Generator().manual_seed(300 + nb * 10 + tokens + sum(lc))
    ps = 64
    npr = (max(lc) + tokens + ps - 1) // ps + 1
    pt = (torch.randperm(nreq * npr, generator=gen).to(torch.int32) + 1).view(nreq, npr)
    slot = torch.tensor(
        [
            int(pt[b, (lc[b] + i) // ps]) * ps + (lc[b] + i) % ps
            for b in range(nreq)
            for i in range(tokens)
        ]
    )
    pos = torch.tensor([lc[b] + i for b in range(nreq) for i in range(tokens)])
    t = attn_tensors(gen, DEVICE, dtype)
    kc, vc = kv_caches(gen, DEVICE, (nreq * npr + 1) * ps, dtype)
    h, r = rows(nb, gen, DEVICE, dtype)
    kc_ref, vc_ref = kc.cpu(), vc.cpu()
    layer, _mb = _register(True, t, [kc, vc], ps)
    ho, ro = torch.empty_like(h), torch.empty_like(r)
    seq = torch.tensor([c + tokens for c in lc], dtype=torch.int32)
    assert _ops().mono_decode_layer_verify(
        layer,
        tokens,
        h,
        r,
        ho,
        ro,
        pos.to(DEVICE),
        pt.to(DEVICE),
        seq.to(DEVICE),
        slot.to(DEVICE),
    )
    torch.xpu.synchronize()
    tc = cpu(t)
    for b in range(nreq):
        for i in range(tokens):
            row = b * tokens + i
            ref = ref_attn(
                tc,
                kc_ref,
                vc_ref,
                h[row : row + 1].cpu(),
                r[row : row + 1].cpu(),
                pt[b : b + 1],
                torch.tensor([lc[b] + i + 1]),
                slot[row : row + 1],
                pos[row : row + 1],
                ps,
                dtype,
                loopback_ar_fp16,
            )
            _compare(ho[row : row + 1], ro[row : row + 1], ref, 1)
    _check(kc[slot].cpu(), kc_ref[slot], 0.9999, "new k rows")
    _check(vc[slot].cpu(), vc_ref[slot], 0.9999, "new v rows")


@requires_mono
@pytest.mark.parametrize("nb,tokens", VERIFY_SHAPES)
@pytest.mark.parametrize("dtype", [F16, BF16])
def test_draft_extend_step(nb, tokens, dtype):
    """The MTP drafter's draft extend step: the prologue on requests x tokens
    rows, then the drafter layer's verify kernel (zero residual; it writes the
    drafter's K / V of every row and attends causally). The layer's rows are
    checked against the reference one row at a time, fed the prologue's output."""
    nreq = nb // tokens
    gen = torch.Generator().manual_seed(500 + nb * 10 + tokens)
    ps = 64
    lc = [70 + 4100 * (b % 2) + 13 * b for b in range(nreq)]
    npr = (max(lc) + tokens + ps - 1) // ps + 1
    pt = (torch.randperm(nreq * npr, generator=gen).to(torch.int32) + 1).view(nreq, npr)
    slot = torch.tensor(
        [
            int(pt[b, (lc[b] + i) // ps]) * ps + (lc[b] + i) % ps
            for b in range(nreq)
            for i in range(tokens)
        ]
    )
    pos = torch.tensor([lc[b] + i for b in range(nreq) for i in range(tokens)])
    t = attn_tensors(gen, DEVICE, dtype)
    kc, vc = kv_caches(gen, DEVICE, (nreq * npr + 1) * ps, dtype)
    kc_ref, vc_ref = kc.cpu(), vc.cpu()
    fc = (torch.randn(2048, 4096, generator=gen) * 0.02).to(dtype).to(DEVICE)
    ne = (torch.randn(2048, generator=gen) * 0.1).to(dtype).to(DEVICE)
    nh = (torch.randn(2048, generator=gen) * 0.1).to(dtype).to(DEVICE)
    e = torch.randn(nb, 2048, generator=gen).to(dtype).to(DEVICE)
    hs = (torch.randn(nb, 2048, generator=gen) * 3).to(dtype).to(DEVICE)
    pro, mb = next(_ids), mailbox(DEVICE)
    _ops().mono_register_mtp(pro, addr(mb), addr(mb), [fc, ne, nh], 0)
    layer, _mb = _register(True, t, [kc, vc], ps)
    x = torch.empty_like(e)
    assert _ops().mono_mtp_prologue(pro, e, hs, x)
    ho, ro = torch.empty_like(x), torch.empty_like(x)
    zeros = torch.zeros_like(x)
    seq = torch.tensor([c + tokens for c in lc], dtype=torch.int32)
    assert _ops().mono_decode_layer_verify(
        layer,
        tokens,
        x,
        zeros,
        ho,
        ro,
        pos.to(DEVICE),
        pt.to(DEVICE),
        seq.to(DEVICE),
        slot.to(DEVICE),
    )
    torch.xpu.synchronize()
    tc, xc = cpu(t), x.cpu()
    for b in range(nreq):
        for i in range(tokens):
            row = b * tokens + i
            ref = ref_attn(
                tc,
                kc_ref,
                vc_ref,
                xc[row : row + 1],
                torch.zeros_like(xc[row : row + 1]),
                pt[b : b + 1],
                torch.tensor([lc[b] + i + 1]),
                slot[row : row + 1],
                pos[row : row + 1],
                ps,
                dtype,
                loopback_ar_fp16,
            )
            _compare(ho[row : row + 1], ro[row : row + 1], ref, 1)
    _check(kc[slot].cpu(), kc_ref[slot], 0.9999, "new k rows")
    _check(vc[slot].cpu(), vc_ref[slot], 0.9999, "new v rows")


@requires_mono
@pytest.mark.parametrize("dtype", [F16, BF16])
@pytest.mark.parametrize("nb", [3, 6])
def test_decode_3_and_6_rows(nb, dtype):
    gen = torch.Generator().manual_seed(40 + nb)
    t = gdn_tensors(gen, DEVICE, dtype)
    conv, ssm = gdn_caches(gen, DEVICE, nb + 1, dtype)
    h, r = rows(nb, gen, DEVICE, dtype)
    slots = torch.arange(nb, dtype=torch.int32) + 1
    conv_ref, ssm_ref = conv.cpu(), ssm.cpu()
    layer, _mb = _register(False, t, [conv, ssm])
    ho, ro = _run(
        layer, h, r, torch.zeros(nb, dtype=torch.int64, device=DEVICE), slots.to(DEVICE)
    )
    torch.xpu.synchronize()
    _compare(
        ho,
        ro,
        ref_gdn(
            cpu(t), conv_ref, ssm_ref, h.cpu(), r.cpu(), slots, dtype, loopback_ar_fp16
        ),
        nb,
    )
    lc = [1000 - 37 * b for b in range(nb)]
    pt, seq, slot, pos, nslots = _attn_case(gen, lc, 64)
    t = attn_tensors(gen, DEVICE, dtype)
    kc, vc = kv_caches(gen, DEVICE, nslots, dtype)
    kc_ref, vc_ref = kc.cpu(), vc.cpu()
    layer, _mb = _register(True, t, [kc, vc])
    ho, ro = _run(
        layer, h, r, pos.to(DEVICE), pt.to(DEVICE), seq.to(DEVICE), slot.to(DEVICE)
    )
    torch.xpu.synchronize()
    ref = ref_attn(
        cpu(t),
        kc_ref,
        vc_ref,
        h.cpu(),
        r.cpu(),
        pt,
        seq,
        slot,
        pos,
        64,
        dtype,
        loopback_ar_fp16,
    )
    _compare(ho, ro, ref, nb)


def _mtp_ref(fc, ne, nh, e, h, dtype):
    """SGLang's MTP prologue: fc(cat(GemmaRMSNorm(e), GemmaRMSNorm(h)))."""
    from mono_decode_utils import _gemma_rms

    z = torch.cat(
        [
            _gemma_rms(e.float(), ne.float(), dtype),
            _gemma_rms(h.float(), nh.float(), dtype),
        ],
        dim=-1,
    )
    return (z.float() @ fc.float().t()).to(dtype)


@requires_mono
@pytest.mark.parametrize("dtype", [F16, BF16])
@pytest.mark.parametrize("nb", [1, 2, 3, 4, 6, 8])
@pytest.mark.parametrize("rank", [0, 1])
def test_mtp_prologue(nb, rank, dtype):
    """The MTP drafter prologue against a torch reference. With a loopback
    mailbox the gathered half is this rank's own half again."""
    gen = torch.Generator().manual_seed(11 + nb)
    fc = (torch.randn(2048, 4096, generator=gen) * 0.02).to(dtype).to(DEVICE)
    ne = (torch.randn(2048, generator=gen) * 0.1).to(dtype).to(DEVICE)
    nh = (torch.randn(2048, generator=gen) * 0.1).to(dtype).to(DEVICE)
    e = torch.randn(nb, 2048, generator=gen).to(dtype).to(DEVICE)
    h = (torch.randn(nb, 2048, generator=gen) * 3).to(dtype).to(DEVICE)
    layer, mb = next(_ids), mailbox(DEVICE)
    _ops().mono_register_mtp(layer, addr(mb), addr(mb), [fc, ne, nh], rank)
    out = torch.empty_like(e)
    assert _ops().mono_mtp_prologue(layer, e, h, out)
    torch.xpu.synchronize()
    ref = _mtp_ref(fc.cpu(), ne.cpu(), nh.cpu(), e.cpu(), h.cpu(), dtype)
    mine = slice(rank * 1024, (rank + 1) * 1024)
    got = out.cpu()
    for b in range(nb):
        _check(got[b, mine], ref[b, mine], 0.9999, f"row {b}")
        assert torch.equal(got[b, 1024 * (1 - rank) : 1024 * (2 - rank)], got[b, mine])
    e5, h5 = e[:1].expand(5, -1).contiguous(), h[:1].expand(5, -1).contiguous()
    assert not _ops().mono_mtp_prologue(
        layer, e5, h5, torch.empty_like(e5)
    )  # no 5-row kernel
