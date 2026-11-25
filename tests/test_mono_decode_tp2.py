"""Two-card (TP2) checks of the mono decode layers, one process per card with
the mailboxes in torch symmetric memory as in SGLang (a Qwen35Workspace a
rank; each case is its own one-row layer table on mailbox and counters row i):

* the two ranks' outputs are bitwise equal, and swapping the ranks' weight
  shards gives bitwise the same outputs (the all-reduces add the two partial
  sums; a rank reading its own partial instead of the peer's would not be
  symmetric);
* fail-stop: a stalled peer fails both ranks within the deadline.
"""

import io
import os
import tempfile
import time

import pytest
import torch
import torch.multiprocessing as mp
from mono_decode_utils import (
    LayerTable,
    cos,
    gdn_caches,
    kv_caches,
    layer_tensors,
    recv_from,
    requires_two_cards,
    rows,
    tp2_workspace,
)

# (case, attention, rows, tokens a request, dtype) of the split-mode checks
SPLIT_TP2 = [
    ("sgdn1", False, 1, 1, torch.bfloat16),
    ("sattn1", True, 1, 1, torch.float16),
    ("sgdn8", False, 8, 1, torch.float16),
    ("sattn8", True, 8, 1, torch.bfloat16),
    ("sgdn_v82", False, 8, 2, torch.bfloat16),
    ("sattn_v63", True, 6, 3, torch.bfloat16),
]


def _worker(rank, path, conn):
    from sgl_kernel import qwen35_decode as q35

    torch.xpu.set_device(rank)
    dev = f"xpu:{rank}"
    nl = 96
    ws = tp2_workspace(rank, path, nl, dev)
    slot = iter(range(nl))
    out = {}

    def kernel(i, kind, t):
        """Kernel i of the workspace (mailbox i) as a one-row layer table."""
        return LayerTable([(kind, t)], [ws.mailbox(i)], rank)

    def step(i, k, *args, mode=0, **meta):
        """One launch of k's layer, on counters row i of the workspace; mode 1:
        one kernel a segment; mode 2: also the all-reduces on ar, by
        torch.distributed between the plan ranges (fp32 sum of the ranks)."""
        c = ws.counters[i : i + 1]
        if mode != 2:
            return q35.qwen35_decode_step(
                ws, k.table, 0, 1, *args, counters=c, mode=mode, **meta
            )
        ar = torch.zeros(args[0].shape[0], 2048, dtype=torch.float32, device=dev)
        cuts = q35.qwen35_split_ar_slots(k.table, 0, 1)
        i0 = 0
        for n, cut in enumerate(cuts):
            if not q35.qwen35_decode_step(
                ws,
                k.table,
                0,
                1,
                *args,
                counters=c,
                mode=2,
                i0=i0,
                i1=cut,
                ar=ar,
                **meta,
            ):
                return False
            if n + 1 < len(cuts):  # the last entry is the plan length
                torch.distributed.all_reduce(ar)
            i0 = cut + 1
        return True

    def layer_case(
        name, attn, nb, swap=False, dtype=torch.float16, reps=1, tokens=1, mode=0
    ):
        i = next(slot)
        seed = (2 if rank == 0 else 1) if swap else (1 if rank == 0 else 2)
        t = layer_tensors(attn, seed, dev, dtype)
        gen = torch.Generator().manual_seed(3)
        if attn:
            caches = kv_caches(gen, dev, nb * 4 * 64 + 64, dtype)
        else:
            caches = gdn_caches(gen, dev, nb + 1, dtype)
            if tokens > 1:  # the verify scratch
                iss = torch.zeros(
                    (nb + 1, tokens, 16, 128, 128), dtype=torch.float32, device=dev
                )
                buf = torch.zeros((nb + 1, 4096, tokens + 2), dtype=dtype, device=dev)
                caches = caches + (
                    iss,
                    buf.as_strided(
                        (nb + 1, tokens, 4096, 3),
                        (4096 * (tokens + 2), 1, tokens + 2, 1),
                    ),
                )
        k = kernel(i, "attn" if attn else "gdn", t)
        k.bind(0, caches, 64)
        gen = torch.Generator().manual_seed(5)
        hs, rs = rows(nb, gen, dev, dtype)
        ho, ro = torch.empty_like(hs), torch.empty_like(rs)
        for _ in range(reps):
            if tokens > 1:
                nrq = nb // tokens
                if attn:
                    ctx = 150
                    pt = (
                        torch.arange(nrq * 4, dtype=torch.int32, device=dev).view(
                            nrq, 4
                        )
                        + 1
                    )
                    pos = torch.tensor(
                        [ctx + j % tokens for j in range(nb)], device=dev
                    )
                    slots = torch.tensor(
                        [
                            (1 + b * 4) * 64 + ctx + j
                            for b in range(nrq)
                            for j in range(tokens)
                        ],
                        device=dev,
                    )
                    seq = torch.full(
                        (nrq,), ctx + tokens, dtype=torch.int32, device=dev
                    )
                    assert step(
                        i,
                        k,
                        hs,
                        rs,
                        ho,
                        ro,
                        pos,
                        tokens=tokens,
                        page_table=pt,
                        seq_lens=seq,
                        out_loc=slots,
                        mode=mode,
                    )
                else:
                    zero = torch.zeros(nb, dtype=torch.int64, device=dev)
                    idx = torch.arange(nrq, dtype=torch.int32, device=dev) + 1
                    assert step(
                        i,
                        k,
                        hs,
                        rs,
                        ho,
                        ro,
                        zero,
                        tokens=tokens,
                        state_idx=idx,
                        mode=mode,
                    )
            elif attn:
                ctx = 150
                pt = torch.arange(nb * 4, dtype=torch.int32, device=dev).view(nb, 4) + 1
                pos = torch.full((nb,), ctx, dtype=torch.int64, device=dev)
                slots = torch.tensor(
                    [(1 + b * 4) * 64 + ctx for b in range(nb)], device=dev
                )
                seq = torch.full((nb,), ctx + 1, dtype=torch.int32, device=dev)
                assert step(
                    i,
                    k,
                    hs,
                    rs,
                    ho,
                    ro,
                    pos,
                    page_table=pt,
                    seq_lens=seq,
                    out_loc=slots,
                    mode=mode,
                )
            else:
                zero = torch.zeros(nb, dtype=torch.int64, device=dev)
                idx = torch.arange(nb, dtype=torch.int32, device=dev) + 1
                assert step(i, k, hs, rs, ho, ro, zero, state_idx=idx, mode=mode)
        torch.xpu.synchronize()
        out[name] = (ho.cpu(), ro.cpu())

    for swap in (False, True):
        s = "swap" if swap else "base"
        for dtype, sfx in ((torch.float16, ""), (torch.bfloat16, "_bf16")):
            layer_case(f"gdn1{sfx}_{s}", False, 1, swap=swap, dtype=dtype)
            layer_case(f"gdn8{sfx}_{s}", False, 8, swap=swap, dtype=dtype)
            layer_case(f"attn1{sfx}_{s}", True, 1, swap=swap, dtype=dtype)
            layer_case(f"attn4{sfx}_{s}", True, 4, swap=swap, dtype=dtype)
        layer_case(f"gdn_v44_{s}", False, 4, swap=swap, tokens=4)
        layer_case(f"attn_v44_{s}", True, 4, swap=swap, tokens=4)
        layer_case(f"gdn_v82_{s}", False, 8, swap=swap, tokens=2, dtype=torch.bfloat16)
        layer_case(f"attn_v63_{s}", True, 6, swap=swap, tokens=3, dtype=torch.bfloat16)
        layer_case(f"gdn3_{s}", False, 3, swap=swap)
        layer_case(f"attn6_{s}", True, 6, swap=swap)
    # modes 1 / 2 (one kernel a segment; mode 2 with the all-reduces by
    # torch.distributed): two calls each (the second on the counters and states
    # the first left), compared bitwise with mode 0
    for case, attn, nb, tokens, dtype in SPLIT_TP2:
        for mode in (0, 1, 2):
            layer_case(
                f"{case}_m{mode}",
                attn,
                nb,
                tokens=tokens,
                dtype=dtype,
                reps=2,
                mode=mode,
            )
    # many calls on one mailbox region (tags, torn words): the last outputs
    layer_case("gdn2_repeat", False, 2, reps=300)
    layer_case("gdn_v44_repeat", False, 4, reps=100, tokens=4)

    # MTP prologue: same weights and inputs on both ranks, each computes half
    # of the fc rows; the gathered output is the full fc
    def mtp_case(name, nb, dtype):
        i = next(slot)
        gen = torch.Generator().manual_seed(21 + nb)
        fc = (torch.randn(2048, 4096, generator=gen) * 0.02).to(dtype).to(dev)
        ne = (torch.randn(2048, generator=gen) * 0.1).to(dtype).to(dev)
        nh = (torch.randn(2048, generator=gen) * 0.1).to(dtype).to(dev)
        e = torch.randn(nb, 2048, generator=gen).to(dtype).to(dev)
        hh = (torch.randn(nb, 2048, generator=gen) * 3).to(dtype).to(dev)
        k = kernel(i, "mtp", [fc, ne, nh])
        o = torch.empty_like(e)
        for _ in range(3):
            assert q35.qwen35_mtp_prologue(ws, k.table[0], e, hh, o, i)
        torch.xpu.synchronize()
        out[name] = (o.cpu(), fc.cpu(), ne.cpu(), nh.cpu(), e.cpu(), hh.cpu())

    for nb in (1, 3, 8):
        mtp_case(f"mtp{nb}", nb, torch.bfloat16)
    mtp_case("mtp2_fp16", 2, torch.float16)

    # fail-stop: no wait gave up so far; then rank 1's tags skip ahead, so
    # neither rank ever sees the words it waits for: both give up within the
    # deadline, record it, and a later call exits at once (sticky)
    # (the deadline change re-initialises every counters row: between steps, on
    # both ranks; the fault case's mailbox was never used)
    health = ws.health
    out["health_ok"] = health.cpu().clone()
    ws.set_deadline(3.0)
    i = next(slot)
    t = layer_tensors(False, 1 if rank == 0 else 2, dev, torch.float16)
    conv, ssm = gdn_caches(torch.Generator().manual_seed(3), dev, 2, torch.float16)
    k = kernel(i, "gdn", t)
    k.bind(0, [conv, ssm], 64)
    hs, rs = rows(1, torch.Generator().manual_seed(5), dev)
    ho, ro = torch.empty_like(hs), torch.empty_like(rs)
    zero = torch.zeros(1, dtype=torch.int32, device=dev)
    if rank == 1:
        q35.qwen35_fault_inject(ws, i, 7)
    times = []
    for _ in range(2):
        t0 = time.monotonic()
        assert step(i, k, hs, rs, ho, ro, zero.long(), state_idx=zero)
        torch.xpu.synchronize()
        times.append(time.monotonic() - t0)
    out["fault"] = (health.cpu().clone(), times)
    out["fault_host"] = ws.health_host.clone()
    out["fault_out"] = ho.cpu()
    # the record's word 6 (the mailbox word's address when no region is set)
    # and word 7 (the kernel's counters row), low 32 bits
    own, _ = ws.mailbox(i)
    out["fault_where"] = (
        own.data_ptr(),
        own.numel() * own.element_size(),
        ws.counters[i].data_ptr(),
    )
    buf = io.BytesIO()  # bytes: the process may exit before the tensors are
    torch.save(out, buf)  # read back from shared memory
    conn.send(buf.getvalue())
    conn.close()


def _run_pair():
    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "mono_tp2.pg")
        c0, c1 = ctx.Pipe(), ctx.Pipe()
        procs = [
            ctx.Process(target=_worker, args=(0, path, c0[1])),
            ctx.Process(target=_worker, args=(1, path, c1[1])),
        ]
        try:
            for p in procs:
                p.start()
            c0[1].close()  # the parent's copies: a worker that dies is an EOF
            c1[1].close()
            r0, r1 = (
                torch.load(io.BytesIO(recv_from(c, p)))
                for c, p in ((c0[0], procs[0]), (c1[0], procs[1]))
            )
            for p in procs:
                p.join(timeout=300)
                assert p.exitcode == 0
        finally:  # a failed worker must not leave the other one blocked
            for p in procs:
                if p.is_alive():
                    p.terminate()
                p.join(timeout=10)
    return r0, r1


@pytest.fixture(scope="module")
def pair():
    return _run_pair()


CASES = [
    "gdn1",
    "gdn8",
    "attn1",
    "attn4",
    "gdn1_bf16",
    "gdn8_bf16",
    "attn1_bf16",
    "attn4_bf16",
    "gdn_v44",
    "attn_v44",
    "gdn_v82",
    "attn_v63",
    "gdn3",
    "attn6",
]


@requires_two_cards
@pytest.mark.parametrize("case", CASES)
def test_layer_ranks_bitwise_equal(pair, case):
    r0, r1 = pair
    for s in ("base", "swap"):
        ho0, ro0 = r0[f"{case}_{s}"]
        ho1, ro1 = r1[f"{case}_{s}"]
        for x in (ho0, ro0, ho1, ro1):
            assert torch.isfinite(x.float()).all()
            assert x.float().abs().max() > 0  # all-zero outputs are not a result
        assert torch.equal(ho0, ho1)
        assert torch.equal(ro0, ro1)  # the residual after the mixer's all-reduce


@requires_two_cards
@pytest.mark.parametrize("case", CASES)
def test_layer_all_reduce_symmetric(pair, case):
    r0, _ = pair
    assert torch.equal(r0[f"{case}_base"][0], r0[f"{case}_swap"][0])


@requires_two_cards
def test_repeated_calls(pair):
    r0, r1 = pair
    assert torch.isfinite(r0["gdn2_repeat"][0].float()).all()
    assert torch.equal(r0["gdn2_repeat"][0], r1["gdn2_repeat"][0])
    assert torch.equal(r0["gdn_v44_repeat"][0], r1["gdn_v44_repeat"][0])


@requires_two_cards
def test_fail_stop(pair):
    """A stalled peer fails both ranks instead of hanging them: every wait is
    bounded, the failure is recorded in the device's health words (sticky), and
    healthy calls never give up a wait."""
    for r in pair:
        assert int(r["health_ok"][0]) == 0 and int(r["health_ok"][1]) == 0
        health, (first, second) = r["fault"]
        assert int(health[0]) == 1 and int(health[1]) >= 1
        assert first < 30, f"the first call gave up after {first:.1f} s"
        assert second < 5, f"a call after the failure took {second:.1f} s"
        # the host copy (for the fail-stop thread) and NaN outputs
        host = r["fault_host"]
        assert int(host[0]) == 1 and int(host[1]) >= 1
        assert torch.isnan(r["fault_out"].float()).all()
        # which wait: a word of this kernel's own mailbox, its counters row
        own, nbytes, cnt = r["fault_where"]
        if not int(health[2]):  # it timed out itself (not told by the peer)
            word = int(health[6]) & 0xFFFFFFFF
            assert (word - own) & 0xFFFFFFFF < nbytes, hex(word)
            assert int(health[7]) & 0xFFFFFFFF == cnt & 0xFFFFFFFF


@requires_two_cards
@pytest.mark.parametrize("case", ["mtp1", "mtp3", "mtp8", "mtp2_fp16"])
def test_mtp_prologue_tp2(pair, case):
    """The MTP prologue across two cards: both ranks hold the full fc output,
    bitwise equal, and it matches the torch reference."""
    from mono_decode_utils import _gemma_rms

    r0, r1 = pair
    o0, fc, ne, nh, e, h = r0[case]
    assert torch.equal(o0, r1[case][0])
    dtype = o0.dtype
    z = torch.cat([_gemma_rms(e, ne, dtype), _gemma_rms(h, nh, dtype)], dim=-1)
    ref = (z @ fc.float().t()).to(dtype)
    for b in range(o0.shape[0]):
        assert cos(o0[b], ref[b]) > 0.9999


@requires_two_cards
@pytest.mark.parametrize("case", [c[0] for c in SPLIT_TP2])
def test_split_modes_tp2(pair, case):
    """Modes 1 and 2 (the all-reduces by torch.distributed on ar) give bitwise
    the outputs of mode 0 on both ranks."""
    for r in pair:
        ho, ro = r[f"{case}_m0"]
        assert torch.isfinite(ho.float()).all() and ho.float().abs().max() > 0
        for mode in (1, 2):
            h2, r2 = r[f"{case}_m{mode}"]
            assert torch.equal(ho, h2), f"mode {mode}: hid_out"
            assert torch.equal(ro, r2), f"mode {mode}: res_out"
