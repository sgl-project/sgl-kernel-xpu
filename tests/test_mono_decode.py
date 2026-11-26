"""Single-card numerics of the mono decode layers (Qwen3.5 / Qwen3.6 MoE, one
TP2 rank; the all-reduces through a loopback mailbox, so they return twice the
rank's partial sums): the GatedDeltaNet and full-attention layer kernels (1, 2,
4, 8 rows) against a torch reference on SGLang's layouts (block-FP8 weights,
FusedMoE experts, separate K / V buffers with a page table, channel-major conv
state), in fp16 and bf16, plus the SGLang padding convention, the input
checks and a multi-layer launch. The kernels run through the caller-owned
interface (torch.ops.sgl_kernel.qwen35_*): a CPU layer table, a loopback mailbox,
workspace and counters row a kernel (mono_decode_fixtures.LocalLayers)."""

import time

import pytest
import torch
from mono_decode_utils import (
    CONVD,
    LocalLayers,
    LocalState,
    _q35,
    attn_tensors,
    cos,
    cpu,
    gdn_caches,
    gdn_tensors,
    kv_caches,
    loopback_ar,
    loopback_ar_fp16,
    mailbox,
    mono_available,
    ref_attn,
    ref_gdn,
    requires_mono,
    route_gap,
    rows,
)

DEVICE = "xpu:0"
F16, BF16 = torch.float16, torch.bfloat16


def _register(attn: bool, tensors: list, caches: list, page_size=64, device=DEVICE):
    """One layer (kernel 0 of its own table, loopback mailbox, workspace and
    counters), its caches bound."""
    layer = LocalLayers(device, [("attn" if attn else "gdn", tensors)])
    layer.bind(0, caches, page_size)
    return layer


def _meta(layer, m0, m1, m2) -> dict:
    """The step's metadata: attention m0 page table, m1 seq_lens, m2 out_loc;
    GatedDeltaNet m0 state slots."""
    if layer.attn[0]:
        return dict(page_table=m0, seq_lens=m1, out_loc=m2)
    return dict(state_idx=m0)


def _run(layer, h, r, positions, m0, m1=None, m2=None):
    ho, ro = torch.empty_like(h), torch.empty_like(r)
    assert layer.step(0, 1, h, r, ho, ro, positions, **_meta(layer, m0, m1, m2))
    return ho, ro


def _verify(layer, tokens, h, r, ho, ro, positions, m0, m1=None, m2=None) -> bool:
    """A verify step (``tokens`` a request); False: the shape has no kernel."""
    return layer.step(
        0, 1, h, r, ho, ro, positions, tokens=tokens, **_meta(layer, m0, m1, m2)
    )


def _check(got, ref, tol, what=""):
    """Finite, the direction of the reference (cosine) and its size: a result
    scaled by a constant has the cosine of the reference, so the relative error
    is bounded too (twice the error a cosine of ``tol`` allows)."""
    assert torch.isfinite(got.float()).all(), f"{what} not finite"
    c = cos(got, ref)
    assert c > tol, f"{what} cos {c:.7f}"
    err = (got.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-12)
    bound = 2 * (2 * (1 - tol)) ** 0.5
    assert err < bound, f"{what} relative error {err:.4f} (bound {bound:.4f})"


def _compare(ho, ro, ref, nb, tol_h=0.999, tol_r=0.9999):
    eh, er, info = ref
    gap = route_gap(info)
    for b in range(nb):
        _check(ro[b].cpu(), er[b], tol_r, f"res_out row {b}")
        if gap[b] > 0.02:  # a near tie of the 8th / 9th expert may pick either
            _check(ho[b].cpu(), eh[b], tol_h, f"hid_out row {b}")
        else:  # either expert may be picked: the output is finite and of the reference's size
            got = ho[b].cpu().float()
            assert torch.isfinite(got).all(), f"hid_out row {b} not finite (near tie)"
            ratio = got.norm() / eh[b].float().norm().clamp_min(1e-12)
            assert (
                0.7 < ratio < 1.4
            ), f"hid_out row {b} size {ratio:.3f} x reference (near tie)"


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
    layer = _register(False, t, [conv, ssm])
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
    layer = _register(True, t, [kc, vc], page_size)
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
        layer = _register(True, t, [kc, vc])
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
    layer = _register(True, t, [kc, vc])
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
    layer = _register(False, tg, [conv, ssm])
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
    """Block scales in the checkpoint dtype (bf16) are converted by
    qwen35_prepare_layer: the same results as fp32 scales of the same values."""
    gen = torch.Generator().manual_seed(5)
    t = gdn_tensors(gen, DEVICE, BF16)
    for i in (5, 7, 9, 11, 13, 15):
        t[i] = t[i].to(BF16).float()
    tb = [x.to(BF16) if i in (5, 7, 9, 11, 13, 15) else x for i, x in enumerate(t)]
    h, r = rows(2, gen, DEVICE, BF16)
    out = []
    for tt in (t, tb):
        conv, ssm = gdn_caches(torch.Generator().manual_seed(6), DEVICE, 3, BF16)
        layer = _register(False, tt, [conv, ssm])
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
    assert _q35().max_context() >= 262144


@requires_mono
def test_input_checks():
    gen = torch.Generator().manual_seed(8)
    for dtype, other in ((F16, BF16), (BF16, F16)):
        t = attn_tensors(gen, DEVICE, dtype)
        kc, vc = kv_caches(gen, DEVICE, 256, other)
        layer = LocalLayers(DEVICE, [("attn", t)])
        with pytest.raises(RuntimeError, match="mix of activations and KV"):
            layer.bind(0, [kc, vc], 64)
        tg = gdn_tensors(gen, DEVICE, dtype)
        conv, ssm = gdn_caches(gen, DEVICE, 2, other)
        layer = LocalLayers(DEVICE, [("gdn", tg)])
        with pytest.raises(RuntimeError, match="conv state"):
            layer.bind(0, [conv, ssm], 64)
        tg[1] = tg[1].to(other)
        with pytest.raises(RuntimeError, match="the layer's dtype"):
            LocalLayers(DEVICE, [("gdn", tg)])
        tg = gdn_tensors(gen, DEVICE, dtype)
        tg[6] = tg[6].transpose(1, 2)  # an [E][K][N] expert layout
        with pytest.raises(RuntimeError, match="w13"):
            LocalLayers(DEVICE, [("gdn", tg)])
        conv, ssm = gdn_caches(gen, DEVICE, 2, dtype)
        layer = _register(False, gdn_tensors(gen, DEVICE, dtype), [conv, ssm])
        h, r = rows(5, gen, DEVICE, dtype)
        zero = torch.zeros(5, dtype=torch.int32, device=DEVICE)
        with pytest.raises(RuntimeError, match="supported: 1, 2, 3, 4, 6, 8"):
            _run(layer, h, r, zero.long(), zero)
        ho, ro = torch.empty_like(h), torch.empty_like(r)
        with pytest.raises(RuntimeError, match="no verify scratch"):
            _verify(layer, 5, h, r, ho, ro, zero.long(), zero)
        h, r = rows(1, gen, DEVICE, other)
        with pytest.raises(RuntimeError, match="the layer's dtype"):
            _run(layer, h, r, zero[:1].long(), zero[:1])


@requires_mono
def test_conv_state_must_have_three_taps():
    # the kernels keep the 3 history taps in entries 0..2; the stock conv update
    # keeps the trailing taps of a wider state, so a wider state is refused
    gen = torch.Generator().manual_seed(9)
    layer = LocalLayers(DEVICE, [("gdn", gdn_tensors(gen, DEVICE, F16))])
    conv, ssm = gdn_caches(gen, DEVICE, 2, F16, taps=4)
    with pytest.raises(RuntimeError, match="conv state"):
        layer.bind(0, [conv, ssm], 64)


@requires_mono
def test_prepare_layer_and_descriptor():
    """qwen35_prepare_layer: the block scales as contiguous fp32 of the same
    values, and for a GatedDeltaNet layer in_proj_ba in fp16 appended (the
    tensor itself in an fp16 layer); qwen35_layer_desc: a CPU uint8 descriptor
    of desc_bytes, from the prepared list only."""
    q35 = _q35()
    gen = torch.Generator().manual_seed(13)
    mb = mailbox(DEVICE)
    size = q35.qwen35_sizes()["desc_bytes"]
    for dtype in (F16, BF16):
        tg = gdn_tensors(gen, DEVICE, dtype)
        pg = q35.qwen35_prepare_layer(False, tg)
        assert len(pg) == 22
        assert pg[21].dtype == F16
        assert torch.equal(pg[21].cpu(), tg[16].to(F16).cpu())
        if dtype == F16:
            assert pg[21].data_ptr() == tg[16].data_ptr()  # used in place
        ta = attn_tensors(gen, DEVICE, dtype)
        pa = q35.qwen35_prepare_layer(True, ta)
        assert len(pa) == 18
        for t, p in ((tg, pg), (ta, pa)):
            for i in (5, 7, 9, 11, 13, 15):
                assert p[i].dtype == torch.float32 and p[i].is_contiguous(), i
                assert torch.equal(p[i].cpu(), t[i].float().cpu()), i
        for attn, p in ((False, pg), (True, pa)):
            d = q35.qwen35_layer_desc(attn, p, mb, mb)
            assert d.device.type == "cpu" and d.dtype == torch.uint8
            assert tuple(d.shape) == (size,)
    with pytest.raises(RuntimeError, match="22 tensors expected"):
        q35.qwen35_layer_desc(False, tg, mb, mb)  # not prepared
    with pytest.raises(RuntimeError, match="mailbox must be a contiguous int64"):
        q35.qwen35_layer_desc(False, pg, mb[:-1], mb)


@requires_mono
def test_layer_table_checks():
    """qwen35_decode_step reads the CPU layer table: a layer without bound
    caches, a table on the device, a range past the table or the counters and
    an MTP prologue row are refused; caches bound on a descriptor after it was
    stacked do not reach the table (torch.stack copies), bound on the table's
    row they do; a verify shape without a kernel returns False."""
    q35 = _q35()
    gen = torch.Generator().manual_seed(14)
    mb = mailbox(DEVICE)
    t = q35.qwen35_prepare_layer(False, gdn_tensors(gen, DEVICE, F16))
    conv, ssm = gdn_caches(gen, DEVICE, 2, F16)
    desc = q35.qwen35_layer_desc(False, t, mb, mb)
    table = q35.stack_descs([desc])
    q35.qwen35_bind_gdn_caches(desc, conv, ssm)  # the table's copy stays unbound
    state = LocalState(DEVICE, 2)
    idx = torch.ones(1, dtype=torch.int32, device=DEVICE)

    def step(tab, l0=0, l1=1, nb=1, **kw):
        h, r = rows(nb, gen, DEVICE, F16)
        pos = torch.zeros(nb, dtype=torch.int64, device=DEVICE)
        return q35.qwen35_decode_step(
            state,
            tab,
            l0,
            l1,
            h,
            r,
            torch.empty_like(h),
            torch.empty_like(r),
            pos,
            state_idx=idx,
            **kw,
        )

    with pytest.raises(RuntimeError, match="has no caches bound"):
        step(table)
    q35.qwen35_bind_gdn_caches(table[0], conv, ssm)  # on the table's row
    assert step(table)
    torch.xpu.synchronize()
    with pytest.raises(RuntimeError, match="layers must be a CPU uint8"):
        step(table.to(DEVICE))
    with pytest.raises(RuntimeError, match="bad layer range"):
        step(table, 0, 2)
    with pytest.raises(RuntimeError, match="a row for every layer"):
        step(table, counters=state.counters[:0])
    fc = (torch.randn(2048, 4096, generator=gen) * 0.02).to(F16).to(DEVICE)
    nrm = (torch.randn(2048, generator=gen) * 0.1).to(F16).to(DEVICE)
    both = q35.stack_descs([table[0], q35.qwen35_mtp_desc([fc, nrm, nrm], mb, mb, 0)])
    with pytest.raises(RuntimeError, match="is an MTP prologue, not a decoder layer"):
        step(both, 0, 2)
    assert not step(table, nb=10, tokens=2)  # more than 8 verify rows


def _strided(t):
    """The same values as a stride-2 view (base[::2]) of a twice as long tensor."""
    base = torch.zeros(2 * t.numel(), dtype=t.dtype, device=t.device)
    base[::2] = t
    return base[::2]


@requires_mono
@pytest.mark.parametrize("which", ["seq_lens", "out_cache_loc", "positions", "state"])
def test_metadata_must_be_unit_stride_xpu_vectors(which):
    gen = torch.Generator().manual_seed(10)
    attn = which != "state"
    if attn:
        kc, vc = kv_caches(gen, DEVICE, 256, F16)
        layer = _register(True, attn_tensors(gen, DEVICE, F16), [kc, vc])
        m0 = torch.ones(1, 4, dtype=torch.int32, device=DEVICE)
        m1 = torch.tensor([5], dtype=torch.int32, device=DEVICE)
        m2 = torch.tensor([64 + 4], dtype=torch.int64, device=DEVICE)
    else:
        conv, ssm = gdn_caches(gen, DEVICE, 2, F16)
        layer = _register(False, gdn_tensors(gen, DEVICE, F16), [conv, ssm])
        m0 = torch.ones(1, dtype=torch.int32, device=DEVICE)
        m1 = m2 = None
    pos = torch.tensor([4], dtype=torch.int64, device=DEVICE)
    h, r = rows(1, gen, DEVICE, F16)
    ok = _run(layer, h, r, pos, m0, m1, m2)  # the contiguous vectors run
    assert torch.isfinite(ok[0].float()).all()
    bad = dict(seq_lens=m1, out_cache_loc=m2, positions=pos, state=m0)[which]
    args = dict(positions=pos, m0=m0, m1=m1, m2=m2)
    key = dict(seq_lens="m1", out_cache_loc="m2", positions="positions", state="m0")
    args[key[which]] = _strided(bad)
    with pytest.raises(RuntimeError, match="unit stride|unit-stride"):
        _run(layer, h, r, **args)
    # and on the host: a CPU tensor is no device pointer
    args[key[which]] = bad.cpu()
    with pytest.raises(RuntimeError, match="XPU tensor"):
        _run(layer, h, r, **args)


@requires_mono
def test_the_layers_device_serves_calls_while_another_is_current():
    if torch.xpu.device_count() < 2 or not mono_available(1):
        pytest.skip("needs two Xe2 GPUs with at least 20 Xe cores")
    gen = torch.Generator().manual_seed(11)
    dev1 = torch.device("xpu:1")
    with torch.xpu.device(0):  # xpu:0 is current, the layer lives on xpu:1
        kc, vc = kv_caches(gen, dev1, 256, F16)
        layer = _register(True, attn_tensors(gen, dev1, F16), [kc, vc], device=dev1)
        h, r = rows(1, gen, dev1, F16)
        m0 = torch.ones(1, 4, dtype=torch.int32, device=dev1)
        m1 = torch.tensor([5], dtype=torch.int32, device=dev1)
        m2 = torch.tensor([64 + 4], dtype=torch.int64, device=dev1)
        pos = torch.tensor([4], dtype=torch.int64, device=dev1)
        ho, _ro = _run(layer, h, r, pos, m0, m1, m2)
        torch.xpu.synchronize(1)
        assert torch.isfinite(ho.float()).all()
        with pytest.raises(RuntimeError, match="XPU tensor on device 1"):
            _run(layer, h, r, pos.to(DEVICE), m0, m1, m2)
        with pytest.raises(RuntimeError, match="XPU tensor on device 1"):
            _run(layer, h, r, pos, m0, m1.to(DEVICE), m2)


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
    _gdn_verify_vs_reference(nb, tokens, dtype, tokens)


@requires_mono
@pytest.mark.parametrize("dtype", [F16, BF16])
@pytest.mark.parametrize("nb,tokens,scratch_tokens", [(4, 2, 8), (6, 3, 4), (8, 4, 8)])
def test_gdn_verify_in_a_wider_scratch(nb, tokens, scratch_tokens, dtype):
    # a step that verifies fewer tokens than the scratch was sized for (adaptive
    # speculation sizes the pool for the largest candidate): the windows of a
    # channel are T0 + 2 apart in the scratch, not T + 2
    _gdn_verify_vs_reference(nb, tokens, dtype, scratch_tokens)


def _gdn_verify_vs_reference(nb, tokens, dtype, scratch_tokens):
    nreq = nb // tokens
    gen = torch.Generator().manual_seed(200 + nb * 10 + tokens)
    t = gdn_tensors(gen, DEVICE, dtype)
    conv, ssm = gdn_caches(gen, DEVICE, nreq + 2, dtype)
    iss, win = _scratch(nreq + 1, scratch_tokens, dtype)
    h, r = rows(nb, gen, DEVICE, dtype)
    slots = torch.randperm(nreq + 1, generator=gen)[:nreq].to(torch.int32) + 1
    conv0, ssm0 = conv.cpu(), ssm.cpu()
    layer = _register(False, t, [conv, ssm, iss, win])
    ho, ro = torch.empty_like(h), torch.empty_like(r)
    assert _verify(
        layer,
        tokens,
        h,
        r,
        ho,
        ro,
        torch.zeros(nb, dtype=torch.int64, device=DEVICE),
        slots.to(DEVICE),
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
    layer = _register(True, t, [kc, vc], ps)
    ho, ro = torch.empty_like(h), torch.empty_like(r)
    seq = torch.tensor([c + tokens for c in lc], dtype=torch.int32)
    assert _verify(
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
    # the drafter's kernels as the runtime builds them: table row 0 the
    # prologue, row 1 the layer, sharing the workspace
    k = LocalLayers(DEVICE, [("mtp", [fc, ne, nh]), ("attn", t)], rank=0)
    k.bind(1, [kc, vc], ps)
    x = torch.empty_like(e)
    assert k.mtp(0, e, hs, x)
    ho, ro = torch.empty_like(x), torch.empty_like(x)
    zeros = torch.zeros_like(x)
    seq = torch.tensor([c + tokens for c in lc], dtype=torch.int32)
    assert k.step(
        1,
        2,
        x,
        zeros,
        ho,
        ro,
        pos.to(DEVICE),
        tokens=tokens,
        page_table=pt.to(DEVICE),
        seq_lens=seq.to(DEVICE),
        out_loc=slot.to(DEVICE),
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
    layer = _register(False, t, [conv, ssm])
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
    layer = _register(True, t, [kc, vc])
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
    pro = LocalLayers(DEVICE, [("mtp", [fc, ne, nh])], rank=rank)
    out = torch.empty_like(e)
    assert pro.mtp(0, e, h, out)
    torch.xpu.synchronize()
    ref = _mtp_ref(fc.cpu(), ne.cpu(), nh.cpu(), e.cpu(), h.cpu(), dtype)
    mine = slice(rank * 1024, (rank + 1) * 1024)
    got = out.cpu()
    for b in range(nb):
        _check(got[b, mine], ref[b, mine], 0.9999, f"row {b}")
        assert torch.equal(got[b, 1024 * (1 - rank) : 1024 * (2 - rank)], got[b, mine])
    e5, h5 = e[:1].expand(5, -1).contiguous(), h[:1].expand(5, -1).contiguous()
    assert not pro.mtp(0, e5, h5, torch.empty_like(e5))  # no 5-row kernel


# the MTP prologue's tag counters in a layer's counter words: EPOCH (6 stage
# counters x 16) + 18, one per row count in this order
MTP_TAGS = 6 * 16 + 18
MTP_WIDTHS = (1, 2, 3, 4, 6, 8)


@requires_mono
def test_mtp_prologue_tags_are_per_row_count():
    """Each row count of the MTP prologue has its own mailbox region and tag
    counter, as the decoder layers do: a call advances only its own counter.
    (With one counter and region for all row counts, the rows a narrow call does
    not write keep an old tag, which a wide call 65536 calls later takes for
    fresh.) The tags stay correct across the 16-bit wrap, where wire tag 0 also
    matches a word that was never written."""
    gen = torch.Generator().manual_seed(12)
    dtype = BF16
    fc = (torch.randn(2048, 4096, generator=gen) * 0.02).to(dtype).to(DEVICE)
    ne = (torch.randn(2048, generator=gen) * 0.1).to(dtype).to(DEVICE)
    nh = (torch.randn(2048, generator=gen) * 0.1).to(dtype).to(DEVICE)
    pro = LocalLayers(DEVICE, [("mtp", [fc, ne, nh])], rank=0)
    cnt = pro.counters(0)

    def tags():
        torch.xpu.synchronize()
        return [int(cnt[MTP_TAGS + i]) & 0xFFFFFFFF for i in range(6)]

    def call(nb):
        e = torch.randn(nb, 2048, generator=gen).to(dtype).to(DEVICE)
        h = (torch.randn(nb, 2048, generator=gen) * 3).to(dtype).to(DEVICE)
        out = torch.empty_like(e)
        assert pro.mtp(0, e, h, out)
        torch.xpu.synchronize()
        ref = _mtp_ref(fc.cpu(), ne.cpu(), nh.cpu(), e.cpu(), h.cpu(), dtype)
        for b in range(nb):
            _check(out.cpu()[b, :1024], ref[b, :1024], 0.9999, f"{nb} rows, row {b}")

    # a call advances the counter of its own row count only
    before = tags()
    for i, nb in enumerate(MTP_WIDTHS):
        call(nb)
        after = tags()
        assert [a - b for a, b in zip(after, before)] == [int(j == i) for j in range(6)]
        before = after
    # wide and narrow calls alternate across the wrap of every counter (tags
    # 65535, 65536 = wire tag 0, 65537 = wire tag 1)
    for i in range(6):
        cnt[MTP_TAGS + i] = 65533
    for _ in range(4):
        for nb in (8, 1, 6, 2, 4, 3):
            call(nb)
    assert tags() == [65533 + 4] * 6


@requires_mono
def test_mtp_prologue_on_20_work_groups():
    """The MTP prologue of a 20-core GPU (Arc Pro B60: the 32 work-groups'
    parts on 20 work-groups) gives bitwise the 32-work-group output, for every
    row count and both dtypes (needs a GPU with 32 Xe cores for the reference)."""
    q35 = _q35()
    if q35.work_groups(torch.device(DEVICE)) < 32:
        pytest.skip("needs a GPU with 32 Xe cores for the reference")
    for dtype in (F16, BF16):
        gen = torch.Generator().manual_seed(31)
        fc = (torch.randn(2048, 4096, generator=gen) * 0.02).to(dtype).to(DEVICE)
        ne = (torch.randn(2048, generator=gen) * 0.1).to(dtype).to(DEVICE)
        nh = (torch.randn(2048, generator=gen) * 0.1).to(dtype).to(DEVICE)
        pro = LocalLayers(DEVICE, [("mtp", [fc, ne, nh])], rank=1)
        for nb in MTP_WIDTHS:
            e = torch.randn(nb, 2048, generator=gen).to(dtype).to(DEVICE)
            h = (torch.randn(nb, 2048, generator=gen) * 3).to(dtype).to(DEVICE)
            outs = []
            for wgs in (0, 20):
                q35.force_work_groups(wgs)
                try:
                    out = torch.empty_like(e)
                    assert pro.mtp(0, e, h, out)
                    torch.xpu.synchronize()
                finally:
                    q35.force_work_groups(0)
                outs.append(out.cpu())
            assert outs[0].float().abs().max() > 0
            assert torch.equal(outs[0], outs[1]), f"{dtype} {nb} rows"


STACK = ("gdn", "attn", "gdn")


@requires_mono
@pytest.mark.parametrize("nb,tokens,dtype", [(1, 1, F16), (4, 1, BF16), (4, 2, BF16)])
def test_layer_stack_in_one_launch(nb, tokens, dtype):
    """qwen35_decode_step(0, 3) of a GatedDeltaNet, attention, GatedDeltaNet
    stack passes the streams between its layers through the workspace's
    ping-pong pairs: bitwise the outputs and caches of the three one-layer
    launches (0, 1), (1, 2), (2, 3), for a decode and a verify step."""
    nreq = nb // tokens
    gen = torch.Generator().manual_seed(600 + nb * 10 + tokens)
    ps = 64
    lc = [1000 - 37 * b for b in range(nreq)]
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
    seq = torch.tensor([c + tokens for c in lc], dtype=torch.int32)
    slots = torch.randperm(nreq + 1, generator=gen)[:nreq].to(torch.int32) + 1
    kernels, caches = [], []
    for kind in STACK:
        if kind == "attn":
            kernels.append((kind, attn_tensors(gen, DEVICE, dtype)))
            caches.append(kv_caches(gen, DEVICE, (nreq * npr + 1) * ps, dtype))
        else:
            kernels.append((kind, gdn_tensors(gen, DEVICE, dtype)))
            caches.append(gdn_caches(gen, DEVICE, nreq + 2, dtype))
    h, r = rows(nb, gen, DEVICE, dtype)
    meta = dict(
        tokens=tokens,
        state_idx=slots.to(DEVICE),
        page_table=pt.to(DEVICE),
        seq_lens=seq.to(DEVICE),
        out_loc=slot.to(DEVICE),
    )
    pos = pos.to(DEVICE)

    def run(split: bool):
        k = LocalLayers(DEVICE, kernels)
        mine = []
        for i, c in enumerate(caches):
            c = [x.clone() for x in c]
            if tokens > 1 and STACK[i] == "gdn":  # a fresh (zero) verify scratch
                c += list(_scratch(nreq + 1, tokens, dtype))
            k.bind(i, c, ps)
            mine.append(c)
        ho, ro = torch.empty_like(h), torch.empty_like(r)
        if split:
            x, y = h, r
            for i in range(len(STACK)):
                ho, ro = torch.empty_like(h), torch.empty_like(r)
                assert k.step(i, i + 1, x, y, ho, ro, pos, **meta)
                x, y = ho, ro
        else:
            assert k.step(0, len(STACK), h, r, ho, ro, pos, **meta)
        torch.xpu.synchronize()
        return ho.cpu(), ro.cpu(), [[x.cpu() for x in c] for c in mine]

    one, split = run(False), run(True)
    for x in one[:2]:
        assert torch.isfinite(x.float()).all()
        assert x.float().abs().max() > 0  # all-zero outputs are not a result
    assert torch.equal(one[0], split[0]), "hid_out"
    assert torch.equal(one[1], split[1]), "res_out"
    for i, (a, b) in enumerate(zip(one[2], split[2])):
        for j, (x, y) in enumerate(zip(a, b)):
            assert torch.equal(x, y), f"layer {i} ({STACK[i]}) cache {j}"


SPLIT_CASES = [
    (1, 1, F16),
    (1, 1, BF16),
    (2, 1, F16),
    (3, 1, BF16),
    (4, 1, BF16),
    (6, 1, F16),
    (8, 1, BF16),
    (4, 2, BF16),
    (6, 3, F16),
    (8, 4, BF16),
]


@requires_mono
@pytest.mark.parametrize("wgs", [32, 20])
@pytest.mark.parametrize("nb,tokens,dtype", SPLIT_CASES)
def test_split_modes_bitwise(nb, tokens, dtype, wgs):
    """Modes 1 (one kernel a segment, in-kernel all-reduces) and 2 (the
    all-reduces by the caller on ar, here the loopback sum ar + ar) give bitwise
    the outputs, caches and per-call counters of mode 0 for a GatedDeltaNet,
    attention, GatedDeltaNet stack, over two consecutive steps (the second one
    runs on the counters and states the first left).

    wgs 20: the kernels of a GPU with 20 Xe cores (Arc Pro B60), which run the
    32 work-groups' parts on 20 work-groups: on a GPU with 32 Xe cores, their
    modes 0, 1 and 2 against the 32-work-group mode 0; on a 20-core GPU (no
    32-work-group launch), modes 1 and 2 against its mode 0."""
    q35 = _q35()
    native = q35.work_groups(torch.device(DEVICE))
    if wgs > native:
        pytest.skip(f"the device runs {native} work-groups")
    nreq = nb // tokens
    gen = torch.Generator().manual_seed(700 + nb * 10 + tokens)
    ps = 64
    lc = [900 - 41 * b for b in range(nreq)]
    npr = (max(lc) + 2 * tokens + ps - 1) // ps + 1
    pt = (torch.randperm(nreq * npr, generator=gen).to(torch.int32) + 1).view(nreq, npr)
    slots = torch.randperm(nreq + 1, generator=gen)[:nreq].to(torch.int32) + 1
    kernels, caches = [], []
    for kind in STACK:
        if kind == "attn":
            kernels.append((kind, attn_tensors(gen, DEVICE, dtype)))
            caches.append(kv_caches(gen, DEVICE, (nreq * npr + 1) * ps, dtype))
        else:
            kernels.append((kind, gdn_tensors(gen, DEVICE, dtype)))
            caches.append(gdn_caches(gen, DEVICE, nreq + 2, dtype))
    h, r = rows(nb, gen, DEVICE, dtype)

    def meta(step):  # step 0, 1: the context grows by the step's tokens
        c = [x + step * tokens for x in lc]
        slot = torch.tensor(
            [
                int(pt[b, (c[b] + i) // ps]) * ps + (c[b] + i) % ps
                for b in range(nreq)
                for i in range(tokens)
            ]
        )
        pos = torch.tensor([c[b] + i for b in range(nreq) for i in range(tokens)])
        return pos.to(DEVICE), dict(
            tokens=tokens,
            state_idx=slots.to(DEVICE),
            page_table=pt.to(DEVICE),
            seq_lens=torch.tensor([x + tokens for x in c], dtype=torch.int32).to(
                DEVICE
            ),
            out_loc=slot.to(DEVICE),
        )

    def run(mode):
        k = LocalLayers(DEVICE, kernels)
        mine = []
        for i, c in enumerate(caches):
            c = [x.clone() for x in c]
            if tokens > 1 and STACK[i] == "gdn":  # a fresh (zero) verify scratch
                c += list(_scratch(nreq + 1, tokens, dtype))
            k.bind(i, c, ps)
            mine.append(c)
        outs = []
        for step in range(2):
            pos, m = meta(step)
            ho, ro = torch.empty_like(h), torch.empty_like(r)
            if mode == 2:
                ar = torch.zeros(nb, 2048, dtype=torch.float32, device=DEVICE)
                cuts = q35.qwen35_split_ar_slots(k.table, 0, len(STACK))
                i0 = 0
                for cut in cuts:
                    assert k.step(
                        0,
                        len(STACK),
                        h,
                        r,
                        ho,
                        ro,
                        pos,
                        mode=2,
                        i0=i0,
                        i1=cut,
                        ar=ar,
                        **m,
                    )
                    ar.add_(ar)  # the loopback all-reduce: own + "peer" (= own)
                    i0 = cut + 1
            else:
                assert k.step(0, len(STACK), h, r, ho, ro, pos, mode=mode, **m)
            torch.xpu.synchronize()
            outs += [ho.cpu(), ro.cpu()]
        torch.xpu.synchronize()
        return outs, [[x.cpu() for x in c] for c in mine], k.state.counters.cpu()

    ref = run(0)
    for x in ref[0]:
        assert torch.isfinite(x.float()).all()
        assert x.float().abs().max() > 0  # all-zero outputs are not a result
    q35.force_work_groups(0 if wgs == native else wgs)
    try:
        runs = [(mode, run(mode)) for mode in ((1, 2) if wgs == native else (0, 1, 2))]
    finally:
        q35.force_work_groups(0)
    for mode, got in runs:
        for j, (x, y) in enumerate(zip(ref[0], got[0])):
            assert torch.equal(
                x, y
            ), f"mode {mode}: step {j // 2} {'res_out' if j % 2 else 'hid_out'}"
        for i, (a, b) in enumerate(zip(ref[1], got[1])):
            for j, (x, y) in enumerate(zip(a, b)):
                assert torch.equal(
                    x, y
                ), f"mode {mode}: layer {i} ({STACK[i]}) cache {j}"
        assert torch.equal(ref[2], got[2]), f"mode {mode}: counters"


@requires_mono
def test_work_groups_never_exceed_the_device(monkeypatch):
    """The work-group override (qwen35_force_work_groups,
    SGLANG_XPU_MONO_WORK_GROUPS) never launches more work-groups than the
    device has Xe cores (they could not all be resident)."""
    from sgl_kernel.monokernel.qwen3_5 import runtime

    q35 = _q35()
    d = torch.device(DEVICE)
    native = q35.work_groups(d)
    try:
        q35.force_work_groups(32)
        assert q35.work_groups(d) == native
        q35.force_work_groups(20)
        assert q35.work_groups(d) == 20
        for bad in (19, 33):
            with pytest.raises(RuntimeError):
                q35.force_work_groups(bad)
        for bad in ("19", str(native + 1), "x"):
            monkeypatch.setenv(runtime.WORK_GROUPS_ENV, bad)
            with pytest.raises(ValueError, match=runtime.WORK_GROUPS_ENV):
                runtime.apply_work_groups(d)
        monkeypatch.setenv(runtime.WORK_GROUPS_ENV, "20")
        assert runtime.apply_work_groups(d) == 20
        monkeypatch.delenv(runtime.WORK_GROUPS_ENV)
        assert runtime.apply_work_groups(d) == native
    finally:
        q35.force_work_groups(0)


@requires_mono
@pytest.mark.parametrize("wgs", [32, 20])
@pytest.mark.parametrize("attn", [False, True])
def test_work_group_wait_gives_up(attn, wgs):
    """A wait for the other work-groups of the card (their arrivals on a stage
    counter) has the mailbox waits' deadline: when the arrivals never come
    (here the kernel's arrival counters are set back, as when not all
    work-groups are resident) it gives up instead of hanging, records which counter, and a
    later call exits at once."""
    from sgl_kernel.monokernel import failure

    q35 = _q35()
    native = q35.work_groups(torch.device(DEVICE))
    if wgs > native:
        pytest.skip(f"the device runs {native} work-groups")
    gen = torch.Generator().manual_seed(91 + attn)
    nb = 1
    if attn:
        pt, seq, slot, pos, nslots = _attn_case(gen, [100], 64)
        t = attn_tensors(gen, DEVICE, BF16)
        caches = list(kv_caches(gen, DEVICE, nslots, BF16))
        meta = (pos.to(DEVICE), pt.to(DEVICE), seq.to(DEVICE), slot.to(DEVICE))
    else:
        t = gdn_tensors(gen, DEVICE, BF16)
        caches = list(gdn_caches(gen, DEVICE, 2, BF16))
        z = torch.zeros(nb, dtype=torch.int64, device=DEVICE)
        meta = (z, torch.ones(nb, dtype=torch.int32, device=DEVICE))
    layer = LocalLayers(DEVICE, [("attn" if attn else "gdn", t)], deadline_s=0.5)
    layer.bind(0, caches, 64)
    h, r = rows(nb, gen, DEVICE, BF16)
    health = layer.state.health
    q35.force_work_groups(0 if wgs == native else wgs)
    try:
        _run(layer, h, r, *meta)
        torch.xpu.synchronize()
        assert int(health[0].cpu()) == 0
        torch.ops.sgl_kernel.qwen35_fault_inject(layer.counters(0), 1 << 20, True)
        times = []
        for _ in range(2):
            t0 = time.monotonic()
            _run(layer, h, r, *meta)
            torch.xpu.synchronize()
            times.append(time.monotonic() - t0)
    finally:
        q35.force_work_groups(0)
    w = health.cpu()
    assert int(w[0]) == 1 and int(w[1]) >= 1, w
    assert int(w[10]) == 1 and int(w[6]) == 0, w  # a counter wait, C_A (first stage)
    cnt = layer.counters(0).data_ptr() & 0xFFFFFFFF
    assert int(w[7]) & 0xFFFFFFFF == cnt
    host = layer.state.health_host
    assert int(host[0]) == 1 and int(host[10]) == 1
    assert times[0] < 15, f"the first call gave up after {times[0]:.1f} s"
    assert times[1] < 5, f"a call after the failure took {times[1]:.1f} s"
    message = failure.describe(w.tolist(), 0, 0.5, lambda a: "layer 0")
    assert "other work-groups" in message and "counter A" in message, message


MAX_CONTEXT_CASES = [  # (rows, tokens a request, cached tokens a request, dtype)
    (1, 1, [524287], BF16),
    (2, 1, [524287, 262144], F16),
    (4, 1, [524287, 300001, 131072, 63], BF16),
    (8, 1, [524287, 400000, 262145, 131109, 40961, 5000, 64, 0], BF16),
    (6, 3, [524285, 262143], F16),
    (8, 8, [524280], BF16),
]


@requires_mono
@pytest.mark.parametrize("wgs", [32, 20])
@pytest.mark.parametrize("nb,tokens,lc,dtype", MAX_CONTEXT_CASES)
def test_attn_at_the_max_context(nb, tokens, lc, dtype, wgs):
    """Attention at the longest supported context (max_context, 512K cached
    tokens with the new ones), decode and verify rows, against the torch
    reference; wgs 20: the kernels of a 20-core GPU (Arc Pro B60). Small
    residual rows, so res_out - (h + r) is the attention's contribution."""
    q35 = _q35()
    assert max(lc) + tokens <= q35.max_context()
    native = q35.work_groups(torch.device(DEVICE))
    if wgs > native:
        pytest.skip(f"the device runs {native} work-groups")
    nreq = nb // tokens
    gen = torch.Generator().manual_seed(800 + nb * 10 + tokens)
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
    seq = torch.tensor([c + tokens for c in lc], dtype=torch.int32)
    t = attn_tensors(gen, DEVICE, dtype)
    kc, vc = kv_caches(gen, DEVICE, (nreq * npr + 1) * ps, dtype)
    h, r = rows(nb, gen, DEVICE, dtype)
    h, r = h * 0.01, r * 0.01
    kc_ref, vc_ref = kc.cpu(), vc.cpu()
    layer = _register(True, t, [kc, vc], ps)
    ho, ro = torch.empty_like(h), torch.empty_like(r)
    args = (pos.to(DEVICE), pt.to(DEVICE), seq.to(DEVICE), slot.to(DEVICE))
    q35.force_work_groups(0 if wgs == native else wgs)
    try:
        if tokens == 1:
            ho, ro = _run(layer, h, r, *args)
        else:
            assert _verify(layer, tokens, h, r, ho, ro, *args)
        torch.xpu.synchronize()
    finally:
        q35.force_work_groups(0)
    tc = cpu(t)
    ar = loopback_ar if nb == 1 else loopback_ar_fp16
    hr = h.cpu().float() + r.cpu().float()
    for b in range(nreq):
        for i in range(tokens):
            row = b * tokens + i
            _, er, _ = ref_attn(
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
                ar,
            )
            _check(
                ro[row].cpu().float() - hr[row],
                er[0] - hr[row],
                0.999,
                f"mixer of row {row} ({lc[b] + i} cached tokens)",
            )
    _check(kc[slot].cpu(), kc_ref[slot], 0.9999, "new k rows")
    _check(vc[slot].cpu(), vc_ref[slot], 0.9999, "new v rows")
