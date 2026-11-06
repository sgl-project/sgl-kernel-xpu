"""Two-card (TP2) checks of the mono decode layers, one process per card with
the mailboxes shared through Level Zero IPC as in SGLang:

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
    attn_tensors,
    exchange_fd,
    gdn_caches,
    gdn_tensors,
    kv_caches,
    requires_two_cards,
    rows,
)

# tensors replicated on both ranks (norms, router, shared gate; GDN gated norm;
# attention q / k norms); the rest are the rank's shards
_REPLICATED = {0, 1, 2, 3}
_REPLICATED_GDN = _REPLICATED | {20}
_REPLICATED_ATTN = _REPLICATED | {16, 17}


def layer_tensors(attn, shard_seed, dev, dtype):
    def build(seed):
        gen = torch.Generator().manual_seed(seed)
        return attn_tensors(gen, dev, dtype) if attn else gdn_tensors(gen, dev, dtype)

    rep, own = build(0), build(shard_seed)
    keep = _REPLICATED_ATTN if attn else _REPLICATED_GDN
    return [rep[i] if i in keep else own[i] for i in range(len(rep))]


def _worker(rank, path, conn):
    import sgl_kernel  # noqa: F401

    ops = torch.ops.sgl_kernel
    torch.xpu.set_device(rank)
    dev = f"xpu:{rank}"
    nl = 32
    h = ops.mono_ipc_create(nl * ops.mono_mbox_bytes())
    info = ops.mono_ipc_export(h)
    conn.send(info)
    peer_info = conn.recv()
    peer_fd = exchange_fd(rank, path, int(info[1]))
    ops.mono_ipc_open(h, peer_info, peer_fd)
    own, peer = ops.mono_ipc_ptr(h, False), ops.mono_ipc_ptr(h, True)
    mb = ops.mono_mbox_bytes()
    slot = iter(range(nl))
    out = {}

    def layer_case(name, attn, nb, swap=False, dtype=torch.float16, reps=1):
        i = next(slot)
        seed = (2 if rank == 0 else 1) if swap else (1 if rank == 0 else 2)
        t = layer_tensors(attn, seed, dev, dtype)
        gen = torch.Generator().manual_seed(3)
        if attn:
            caches = kv_caches(gen, dev, nb * 4 * 64 + 64, dtype)
        else:
            caches = gdn_caches(gen, dev, nb + 1, dtype)
        ops.mono_register_layer(100 + i, attn, own + i * mb, peer + i * mb, t)
        ops.mono_bind_caches(100 + i, list(caches), 64)
        gen = torch.Generator().manual_seed(5)
        hs, rs = rows(nb, gen, dev, dtype)
        ho, ro = torch.empty_like(hs), torch.empty_like(rs)
        for _ in range(reps):
            if attn:
                ctx = 150
                pt = torch.arange(nb * 4, dtype=torch.int32, device=dev).view(nb, 4) + 1
                pos = torch.full((nb,), ctx, dtype=torch.int64, device=dev)
                slots = torch.tensor([(1 + b * 4) * 64 + ctx for b in range(nb)], device=dev)
                seq = torch.full((nb,), ctx + 1, dtype=torch.int32, device=dev)
                ops.mono_decode_layer(100 + i, hs, rs, ho, ro, pos, pt, seq, slots)
            else:
                zero = torch.zeros(nb, dtype=torch.int64, device=dev)
                idx = torch.arange(nb, dtype=torch.int32, device=dev) + 1
                ops.mono_decode_layer(100 + i, hs, rs, ho, ro, zero, idx, None, None)
        torch.xpu.synchronize()
        out[name] = (ho.cpu(), ro.cpu())

    for swap in (False, True):
        s = "swap" if swap else "base"
        for dtype, sfx in ((torch.float16, ""), (torch.bfloat16, "_bf16")):
            layer_case(f"gdn1{sfx}_{s}", False, 1, swap=swap, dtype=dtype)
            layer_case(f"gdn8{sfx}_{s}", False, 8, swap=swap, dtype=dtype)
            layer_case(f"attn1{sfx}_{s}", True, 1, swap=swap, dtype=dtype)
            layer_case(f"attn4{sfx}_{s}", True, 4, swap=swap, dtype=dtype)
    # many calls on one mailbox region (tags, torn words): the last outputs
    layer_case("gdn2_repeat", False, 2, reps=300)

    # fail-stop: no wait gave up so far; then rank 1's tags skip ahead, so
    # neither rank ever sees the words it waits for: both give up within the
    # deadline, record it, and a later call exits at once (sticky)
    health = ops.mono_health(rank)
    out["health_ok"] = health.cpu().clone()
    ops.mono_set_deadline(3.0)
    i = next(slot)
    t = layer_tensors(False, 1 if rank == 0 else 2, dev, torch.float16)
    conv, ssm = gdn_caches(torch.Generator().manual_seed(3), dev, 2, torch.float16)
    ops.mono_register_layer(100 + i, False, own + i * mb, peer + i * mb, t)
    ops.mono_bind_caches(100 + i, [conv, ssm], 64)
    hs, rs = rows(1, torch.Generator().manual_seed(5), dev)
    ho, ro = torch.empty_like(hs), torch.empty_like(rs)
    zero = torch.zeros(1, dtype=torch.int32, device=dev)
    if rank == 1:
        ops.mono_fault_inject(100 + i, 7)
    times = []
    for _ in range(2):
        t0 = time.monotonic()
        ops.mono_decode_layer(100 + i, hs, rs, ho, ro, zero.long(), zero, None, None)
        torch.xpu.synchronize()
        times.append(time.monotonic() - t0)
    out["fault"] = (health.cpu().clone(), times)
    buf = io.BytesIO()  # bytes: the process may exit before the tensors are
    torch.save(out, buf)  # read back from shared memory
    conn.send(buf.getvalue())
    conn.close()


def _run_pair():
    ctx = mp.get_context("spawn")
    path = os.path.join(tempfile.mkdtemp(), "mono_tp2.sock")
    c0, c1 = ctx.Pipe(), ctx.Pipe()
    procs = [
        ctx.Process(target=_worker, args=(0, path, c0[1])),
        ctx.Process(target=_worker, args=(1, path, c1[1])),
    ]
    for p in procs:
        p.start()
    i0, i1 = c0[0].recv(), c1[0].recv()
    c0[0].send(i1)
    c1[0].send(i0)
    r0, r1 = (torch.load(io.BytesIO(c[0].recv())) for c in (c0, c1))
    for p in procs:
        p.join(timeout=300)
        assert p.exitcode == 0
    return r0, r1


@pytest.fixture(scope="module")
def pair():
    return _run_pair()


CASES = ["gdn1", "gdn8", "attn1", "attn4", "gdn1_bf16", "gdn8_bf16", "attn1_bf16", "attn4_bf16"]


@requires_two_cards
@pytest.mark.parametrize("case", CASES)
def test_layer_ranks_bitwise_equal(pair, case):
    r0, r1 = pair
    for s in ("base", "swap"):
        ho0, _ = r0[f"{case}_{s}"]
        ho1, _ = r1[f"{case}_{s}"]
        assert torch.isfinite(ho0.float()).all()
        assert torch.equal(ho0, ho1)


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

