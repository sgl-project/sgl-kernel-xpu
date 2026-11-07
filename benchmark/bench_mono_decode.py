"""Benchmark the mono decode layer kernels (Qwen3.5 / Qwen3.6 MoE 35B-A3B, TP2)
on two Intel GPUs: device time per layer call, one process per card, the
mailboxes shared through Level Zero IPC as in SGLang.

Each case registers NCP copies of a layer (distinct random weights and
caches, so the weights stream from memory as in a 40-layer step rather than
from L2) and times back-to-back calls over the copies with XPU events.

Run (two cards):
  ZE_AFFINITY_MASK=2,3 python benchmark/bench_mono_decode.py
  ZE_AFFINITY_MASK=2,3 python benchmark/bench_mono_decode.py --dtype bf16 --ctx 1024 32768
"""

import argparse
import os
import sys
import tempfile

import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from mono_decode_utils import (  # noqa: E402
    attn_tensors,
    exchange_fd,
    gdn_caches,
    gdn_tensors,
    kv_caches,
    rows,
)


def _worker(rank, path, conn, args):
    import sgl_kernel  # noqa: F401

    ops = torch.ops.sgl_kernel
    torch.xpu.set_device(rank)
    dev = f"xpu:{rank}"
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    nl = args.ncp * len(args.rows) * (1 + len(args.ctx))  # one mailbox per layer copy
    h = ops.mono_ipc_create(nl * ops.mono_mbox_bytes())
    info = ops.mono_ipc_export(h)
    conn.send(info)
    ops.mono_ipc_open(h, conn.recv(), exchange_fd(rank, path, int(info[1])))
    own, peer = ops.mono_ipc_ptr(h, False), ops.mono_ipc_ptr(h, True)
    mb = ops.mono_mbox_bytes()
    slot = iter(range(nl))
    lid = iter(range(1000, 3000))
    out = []
    weights = {}  # kind -> NCP weight sets, shared by the cases

    def case(attn, nb, ctx):
        gen = torch.Generator().manual_seed(rank + 1)
        if attn not in weights:
            weights[attn] = [
                attn_tensors(gen, dev, dtype) if attn else gdn_tensors(gen, dev, dtype)
                for _ in range(args.ncp)
            ]
        layers = []
        for c in range(args.ncp):
            t = weights[attn][c]
            i, layer = next(slot), next(lid)
            ops.mono_register_layer(layer, attn, own + i * mb, peer + i * mb, t)
            if attn:
                ps = 64
                npr = (ctx + 1 + ps - 1) // ps + 1
                kc, vc = kv_caches(gen, dev, (nb * npr + 1) * ps, dtype)
                ops.mono_bind_caches(layer, [kc, vc], ps)
                pt = (torch.arange(nb * npr, dtype=torch.int32) + 1).view(nb, npr)
                loc = torch.tensor(
                    [int(pt[b, ctx // ps]) * ps + ctx % ps for b in range(nb)]
                )
                m = (
                    pt.to(dev),
                    torch.full((nb,), ctx + 1, dtype=torch.int32, device=dev),
                    loc.to(dev),
                )
                keep = (t, kc, vc)
            else:
                conv, ssm = gdn_caches(gen, dev, nb + 1, dtype)
                ops.mono_bind_caches(layer, [conv, ssm], 64)
                m = (torch.arange(nb, dtype=torch.int32, device=dev) + 1, None, None)
                keep = (t, conv, ssm)
            hs, rs = rows(nb, gen, dev, dtype)
            layers.append(
                (layer, keep, m, hs, rs, torch.empty_like(hs), torch.empty_like(rs))
            )
        pos = torch.full((nb,), ctx, dtype=torch.int64, device=dev)

        def run(n):
            for k in range(n):
                layer, _, m, hs, rs, ho, ro = layers[k % args.ncp]
                ops.mono_decode_layer(layer, hs, rs, ho, ro, pos, m[0], m[1], m[2])

        run(2 * args.ncp)
        torch.xpu.synchronize()
        times = []
        for _ in range(args.rounds):
            e0, e1 = torch.xpu.Event(enable_timing=True), torch.xpu.Event(
                enable_timing=True
            )
            e0.record()
            run(args.iters)
            e1.record()
            torch.xpu.synchronize()
            times.append(e0.elapsed_time(e1) * 1000.0 / args.iters)
        times.sort()
        out.append(
            (
                "attn" if attn else "gdn",
                nb,
                ctx if attn else 0,
                times[len(times) // 2],
                times[0],
            )
        )

    for nb in args.rows:
        case(False, nb, 0)
        for ctx in args.ctx:
            case(True, nb, ctx)
    conn.send(out)
    conn.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--ctx", type=int, nargs="+", default=[1024, 32768])
    p.add_argument("--ncp", type=int, default=8, help="layer copies (cold weights)")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--rounds", type=int, default=5)
    args = p.parse_args()
    ctx = mp.get_context("spawn")
    path = os.path.join(tempfile.mkdtemp(), "mono_bench.sock")
    c0, c1 = ctx.Pipe(), ctx.Pipe()
    procs = [
        ctx.Process(target=_worker, args=(r, path, c[1], args))
        for r, c in ((0, c0), (1, c1))
    ]
    for pr in procs:
        pr.start()
    i0, i1 = c0[0].recv(), c1[0].recv()
    c0[0].send(i1)
    c1[0].send(i0)
    r = []
    for c, pr in ((c0, procs[0]), (c1, procs[1])):
        while not c[0].poll(1.0):
            if not pr.is_alive():
                raise SystemExit(f"worker {pr.name} exited with {pr.exitcode}")
        r.append(c[0].recv())
    r0, r1 = r
    for pr in procs:
        pr.join()
    print(
        f"mono decode layer, {args.dtype}, device us per call (median / min of {args.rounds} rounds), rank0 | rank1"
    )
    for a, b in zip(r0, r1):
        kind, nb, c, med, mn = a
        print(
            f"{kind:5s} rows {nb}  ctx {c:6d}: {med:8.1f} / {mn:8.1f} | {b[3]:8.1f} / {b[4]:8.1f}"
        )


if __name__ == "__main__":
    main()
