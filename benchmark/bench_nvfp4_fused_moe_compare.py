# SPDX-License-Identifier: Apache-2.0
"""End-to-end fused MoE layer: NVFP4 vs BF16 / MXFP4-W4A16 / MXFP8 on Xe3P (CRI).

Providers (whole MoE layer: routing prep, scatter, activation quant, both
grouped GEMMs, activation, combine):

  bf16         sgl_kernel.fused_experts                       bf16 weights
  mxfp4_w4a16  sgl_kernel.fused_experts(use_mxfp4_w4a16=True) MXFP4 weights
  nvfp4_bf16   cutlass_fused_experts_nvfp4_woq, SGLANG_NVFP4_WOQ_COMPUTE=bf16
  mxfp8        cutlass_fused_experts_fp8(use_mxfp8=True)      MXFP8 weights
  nvfp4_mxfp8  cutlass_fused_experts_nvfp4_woq, SGLANG_NVFP4_WOQ_COMPUTE=mxfp8

(The MXFP4 W4A4 fused path asserts on ragged per-expert M, so it is not part of
this layer-level comparison; see bench_nvfp4_moe_compare.py for its GEMM.)

Timing: sum of device kernel time per layer call from the torch profiler, plus
host wall-clock per call.

Usage:
    python benchmark/bench_nvfp4_fused_moe_compare.py --model qwen3-30b --tokens 1 16 64 256 1024 4096
"""

import argparse
import json
import os
import time
from typing import Dict, List

import torch

MODELS = {
    "qwen3-30b": (128, 8, 2048, 768),
    "qwen3-235b": (128, 8, 4096, 1536),
    "gpt-oss-120b": (128, 4, 2880, 2880),
    "dsv3-ep8": (32, 8, 7168, 2048),
}
ALL_PROVIDERS = ["bf16", "mxfp4_w4a16", "nvfp4_bf16", "mxfp8", "nvfp4_mxfp8"]


def rand_u8(shape, device, lo=0, hi=256):
    return torch.randint(lo, hi, shape, dtype=torch.uint8, device=device)


def make_routing(tokens, e, topk, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    scores = torch.rand((tokens, e), generator=g)
    w, ids = scores.topk(topk, dim=-1)
    w = w / w.sum(dim=-1, keepdim=True)
    return w.to(device=device, dtype=torch.float32), ids.to(device=device, dtype=torch.int32)


def build(name, e, hidden, inter, dev):
    """Returns fn(a, topk_w, topk_ids) -> out."""
    import sgl_kernel
    from sgl_kernel import (
        cutlass_fused_experts_fp8,
        cutlass_fused_experts_nvfp4_woq,
        fused_experts,
    )

    if name == "bf16":
        w1 = (torch.randn((e, 2 * inter, hidden), device=dev) * 0.05).to(torch.bfloat16)
        w2 = (torch.randn((e, hidden, inter), device=dev) * 0.05).to(torch.bfloat16)
        return lambda a, tw, ti: fused_experts(a, w1, w2, tw.to(a.dtype), ti)
    if name == "mxfp4_w4a16":
        w1 = rand_u8((e, 2 * inter, hidden // 2), dev).view(torch.int8)
        w2 = rand_u8((e, hidden, inter // 2), dev).view(torch.int8)
        s1 = torch.exp2(torch.randint(-8, 0, (e, 2 * inter, hidden // 32), device=dev).float())
        s2 = torch.exp2(torch.randint(-8, 0, (e, hidden, inter // 32), device=dev).float())
        return lambda a, tw, ti: fused_experts(
            a, w1, w2, tw.to(a.dtype), ti, use_mxfp4_w4a16=True, w1_scale=s1, w2_scale=s2
        )
    if name == "mxfp8":
        w1 = (torch.randn((e, 2 * inter, hidden), device=dev) * 0.5).to(torch.float8_e4m3fn)
        w2 = (torch.randn((e, hidden, inter), device=dev) * 0.5).to(torch.float8_e4m3fn)
        s1 = rand_u8((e, 2 * inter, hidden // 32), dev, 118, 124)
        s2 = rand_u8((e, hidden, inter // 32), dev, 118, 124)
        ws = torch.zeros((64 * 1024 * 1024,), dtype=torch.uint8, device=dev)
        z = torch.empty((0,), dtype=torch.int64, device=dev)
        eo = torch.zeros((e,), dtype=torch.int32, device=dev)
        p1 = torch.zeros((e, 3), dtype=torch.int32, device=dev)
        p2 = torch.zeros((e, 3), dtype=torch.int32, device=dev)
        return lambda a, tw, ti: cutlass_fused_experts_fp8(
            a, w1, w2, s1, s2, tw, ti, z, z, z, z, ws, z, z, z, z, z, eo, p1, p2,
            use_fp8_blockscale=True, use_mxfp8=True,
        )
    if name in ("nvfp4_bf16", "nvfp4_mxfp8"):
        from sgl_kernel.nvfp4 import nvfp4_woq_prep

        w1 = rand_u8((e, 2 * inter, hidden // 2), dev)
        w2 = rand_u8((e, hidden, inter // 2), dev)
        sf1 = (torch.rand((e, 2 * inter, hidden // 16), device=dev) * 64 + 1).to(torch.float8_e4m3fn)
        sf2 = (torch.rand((e, hidden, inter // 16), device=dev) * 64 + 1).to(torch.float8_e4m3fn)
        gs = torch.full((e,), 1e-3, dtype=torch.float32, device=dev)
        c1, x1 = nvfp4_woq_prep(sf1, gs)
        c2, x2 = nvfp4_woq_prep(sf2, gs)
        prepared = {"w1_c": c1, "w1_sfmx": x1, "w2_c": c2, "w2_sfmx": x2}
        ws = torch.zeros((64 * 1024 * 1024,), dtype=torch.uint8, device=dev)
        mode = "bf16" if name == "nvfp4_bf16" else "mxfp8"

        def fn(a, tw, ti):
            os.environ["SGLANG_NVFP4_WOQ_COMPUTE"] = mode
            return cutlass_fused_experts_nvfp4_woq(a, w1, w2, prepared, tw, ti, workspace=ws)

        return fn
    raise ValueError(name)


def device_us(fn, iters):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
        for _ in range(iters):
            fn()
        torch.xpu.synchronize()
    per: Dict[str, float] = {}
    for evt in prof.events():
        if str(evt.device_type).endswith("XPU"):
            per[evt.name] = per.get(evt.name, 0.0) + evt.device_time_total
    total = sum(per.values()) / iters
    gemm = sum(t for n, t in per.items() if "Gemm" in n) / iters
    return total, gemm


def wall_us(fn, iters):
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-30b", choices=sorted(MODELS))
    ap.add_argument("--tokens", type=int, nargs="+", default=[1, 16, 64, 256, 1024, 4096])
    ap.add_argument("--providers", default=",".join(ALL_PROVIDERS))
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    providers = [p for p in args.providers.split(",") if p]
    e, topk, hidden, inter = MODELS[args.model]
    dev = "xpu"
    print(f"device={torch.xpu.get_device_name()} model={args.model} E={e} topk={topk} H={hidden} I={inter}")
    fns = {}
    for p in providers:
        try:
            fns[p] = build(p, e, hidden, inter, dev)
        except Exception as exc:  # noqa: BLE001
            print(f"{p}: build failed {type(exc).__name__}: {exc}")
    rows: List[Dict] = []
    for t in args.tokens:
        torch.manual_seed(t)
        a = (torch.randn((t, hidden), device=dev) * 0.5).to(torch.bfloat16)
        tw, ti = make_routing(t, e, topk, dev)
        row = {"tokens": t}
        parts = []
        for p, f in fns.items():
            call = lambda: f(a, tw, ti)  # noqa: E731
            try:
                for _ in range(args.warmup):
                    call()
                torch.xpu.synchronize()
                tot, gemm = device_us(call, args.iters)
                wall = wall_us(call, args.iters)
            except Exception as exc:  # noqa: BLE001
                parts.append(f"{p}=ERR({type(exc).__name__})")
                row[p] = None
                continue
            row[p] = {"dev_us": tot, "gemm_us": gemm, "wall_us": wall}
            parts.append(f"{p}={tot:8.1f}")
        rows.append(row)
        print(f"T={t:5d} layer device us: " + "  ".join(parts), flush=True)

    def r(row, a, b):
        return row[a]["dev_us"] / row[b]["dev_us"] if row.get(a) and row.get(b) else None

    print("\nratios (layer device time, <1 means NVFP4 faster)")
    print(f"{'T':>5s} {'nvbf16/bf16':>11s} {'nvbf16/mx4a16':>13s} {'nvmx8/mxfp8':>11s}")
    for row in rows:
        vals = [r(row, "nvfp4_bf16", "bf16"), r(row, "nvfp4_bf16", "mxfp4_w4a16"),
                r(row, "nvfp4_mxfp8", "mxfp8")]
        print(f"{row['tokens']:5d} " + " ".join(
            f"{v:11.2f}" if v is not None else f"{'-':>11s}" for v in vals))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
