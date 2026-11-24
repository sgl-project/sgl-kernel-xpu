# SPDX-License-Identifier: Apache-2.0
"""Compare NVFP4 MoE grouped GEMMs against BF16 / MXFP8 / MXFP4 on Xe3P (CRI).

Two activation paths, each with three weight formats:

  BF16 activations
    bf16          moe_grouped_mm_nt_xe35              bf16 weights
    mxfp4_w4a16   moe_grouped_mm_nt_xe35_mxfp4_w4a16  MXFP4 weights
    nvfp4_bf16    nvfp4_woq_bf16_grouped_mm           NVFP4 weights

  MXFP8 activations
    mxfp8         fp8_blockwise_scaled_grouped_mm     MXFP8 weights
    mxfp4         mxfp4_blockwise_scaled_grouped_mm   MXFP4 weights, MXFP4 acts (W4A4)
    nvfp4_mxfp8   nvfp4_woq_mxfp8_scaled_grouped_mm   NVFP4 weights

Timing is device time from the torch profiler (sum of every device kernel the op
submits, metadata prep included) averaged over ``--iters`` launches, plus host
wall-clock per call.  Inputs are random bit patterns: only the shapes and data
volume matter for performance.

Usage:
    python benchmark/bench_nvfp4_moe_compare.py --model qwen3-30b --tokens 1 16 128 1024 4096
    python benchmark/bench_nvfp4_moe_compare.py --providers nvfp4_bf16,mxfp4_w4a16 --gemm up
"""

import argparse
import json
import os
import time
from typing import Dict, List

import torch

MODELS = {
    # name: (num_experts, topk, hidden, moe_intermediate)
    "qwen3-30b": (128, 8, 2048, 768),
    "qwen3-235b": (128, 8, 4096, 1536),
    "gpt-oss-120b": (128, 4, 2880, 2880),
    "dsv3-ep8": (32, 8, 7168, 2048),  # DeepSeek-V3, 256 experts / EP8
}

ALL_PROVIDERS = ["bf16", "mxfp4_w4a16", "nvfp4_bf16", "mxfp8", "mxfp4", "nvfp4_mxfp8"]


def route_counts(
    tokens: int, num_experts: int, topk: int, align: int = 4, seed: int = 0
) -> List[int]:
    """Random top-k routing; per-expert counts rounded up to ``align``.

    The MXFP4 W4A4 kernel is built with CUTE_ENABLE_XE_BLOCK_2D_ASSERT and its
    A-scale surface asserts (aborting the process) unless every expert's M is a
    multiple of 4, so all providers get the same rounded counts.
    """
    g = torch.Generator().manual_seed(seed)
    scores = torch.rand((tokens, num_experts), generator=g)
    ids = scores.topk(topk, dim=-1).indices.flatten()
    counts = torch.bincount(ids, minlength=num_experts).tolist()
    return [(c + align - 1) // align * align for c in counts]


def rand_u8(shape, device, lo=0, hi=256):
    return torch.randint(lo, hi, shape, dtype=torch.uint8, device=device)


class Case:
    def __init__(self, counts: List[int], n: int, k: int, device: str):
        self.counts = counts
        self.e = len(counts)
        self.n = n
        self.k = k
        self.m = sum(counts)
        self.device = device
        m_alloc = max(self.m, 1)
        starts, acc = [], 0
        for c in counts:
            starts.append(acc)
            acc += c
        self.problem_sizes = torch.tensor(
            [[c, n, k] for c in counts], dtype=torch.int32, device=device
        )
        self.expert_starts = torch.tensor(starts, dtype=torch.int32, device=device)
        self.total_rows = torch.tensor(counts, dtype=torch.int32, device=device)
        self.workspace = torch.zeros((64 * 1024 * 1024,), dtype=torch.uint8, device=device)
        self.empty = torch.empty((0,), dtype=torch.int64, device=device)
        self.m_alloc = m_alloc

    def flops(self):
        return 2.0 * self.m * self.n * self.k


def build_provider(name: str, cs: Case):
    import sgl_kernel  # noqa: F401
    from sgl_kernel import (
        fp8_blockwise_scaled_grouped_mm,
        mxfp4_blockwise_scaled_grouped_mm,
        nvfp4_woq_bf16_grouped_mm,
        nvfp4_woq_mxfp8_scaled_grouped_mm,
    )

    dev, e, n, k, m = cs.device, cs.e, cs.n, cs.k, cs.m_alloc
    if name == "bf16":
        a = (torch.randn((m, k), device=dev) * 0.5).to(torch.bfloat16)
        w = (torch.randn((e, n, k), device=dev) * 0.05).to(torch.bfloat16)
        out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)

        def fn():
            torch.ops.sgl_kernel.moe_grouped_mm_nt_xe35(
                out, a, w, None, cs.total_rows, e, 0, False, 1.702, 7.0
            )

        return fn, e * n * k * 2
    if name == "mxfp4_w4a16":
        a = (torch.randn((m, k), device=dev) * 0.5).to(torch.bfloat16)
        w = rand_u8((e, n, k // 2), dev).view(torch.int8)
        s = torch.exp2(torch.randint(-8, 0, (e, n, k // 32), device=dev).float())
        out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)

        def fn():
            torch.ops.sgl_kernel.moe_grouped_mm_nt_xe35_mxfp4_w4a16(
                out, a, w, s, None, cs.total_rows, e, 0, False, 1.702, 7.0
            )

        return fn, e * n * k // 2 + e * n * (k // 32) * 4
    if name == "nvfp4_bf16":
        a = (torch.randn((m, k), device=dev) * 0.5).to(torch.bfloat16)
        w = rand_u8((e, n, k // 2), dev)
        sfmx = rand_u8((e, k // 32, n), dev, 120, 130)
        c = (torch.rand((e, n, k // 16), device=dev) + 0.5).to(torch.bfloat16)
        out = torch.empty((m, n), dtype=torch.float32, device=dev)

        def fn():
            nvfp4_woq_bf16_grouped_mm(
                out, a, w, sfmx, c, cs.problem_sizes, cs.expert_starts, cs.workspace
            )

        return fn, e * n * k // 2 + e * n * (k // 16) * 2 + e * n * (k // 32)
    if name == "mxfp8":
        a = (torch.randn((m, k), device=dev) * 0.5).to(torch.float8_e4m3fn)
        sa = rand_u8((m, k // 32), dev, 120, 130)
        w = (torch.randn((e, n, k), device=dev) * 0.5).to(torch.float8_e4m3fn)
        sw = rand_u8((e, n, k // 32), dev, 120, 130)
        out = torch.empty((m, n), dtype=torch.float32, device=dev)
        zs = torch.zeros((e,), dtype=torch.int64, device=dev)
        zl = torch.zeros((e, 5), dtype=torch.int32, device=dev)

        def fn():
            fp8_blockwise_scaled_grouped_mm(
                out, cs.empty, cs.empty, cs.empty, cs.empty, cs.empty,
                a, w, sa, sw, zs, zs, zs, zl, zl,
                cs.problem_sizes, cs.expert_starts, cs.workspace,
            )

        return fn, e * n * k + e * n * (k // 32)
    if name == "mxfp4":
        # Its scale-transpose scratch is not padded per expert, so the Xe
        # block-2D asserts (which abort the process) need 64-aligned expert row
        # starts. Pad this provider's counts; the 512-row tile makes the extra
        # rows free at decode sizes and negligible at prefill sizes.
        cs = Case([(c + 63) // 64 * 64 for c in cs.counts], n, k, dev)
        m = cs.m_alloc
        a = rand_u8((m, k // 2), dev)
        sa = rand_u8((m, k // 32), dev, 120, 130)
        w = rand_u8((e, n, k // 2), dev)
        sw = rand_u8((e, n, k // 32), dev, 120, 130)
        out = torch.empty((m, n), dtype=torch.float32, device=dev)

        def fn():
            mxfp4_blockwise_scaled_grouped_mm(
                out, cs.empty, cs.empty, cs.empty, cs.empty, cs.empty,
                a, w, sa, sw, cs.problem_sizes, cs.expert_starts, cs.workspace,
            )

        return fn, e * n * k // 2 + e * n * (k // 32)
    if name == "nvfp4_mxfp8":
        a = (torch.randn((m, k), device=dev) * 0.5).to(torch.float8_e4m3fn)
        sa = rand_u8((m, k // 32), dev, 120, 130)
        w = rand_u8((e, n, k // 2), dev)
        sfmx = rand_u8((e, k // 32, n), dev, 120, 130)
        c = (torch.rand((e, n, k // 16), device=dev) + 0.5).to(torch.bfloat16)
        out = torch.empty((m, n), dtype=torch.float32, device=dev)

        def fn():
            nvfp4_woq_mxfp8_scaled_grouped_mm(
                out, a, w, sa, sfmx, c, cs.problem_sizes, cs.expert_starts, cs.workspace
            )

        return fn, e * n * k // 2 + e * n * (k // 16) * 2 + e * n * (k // 32)
    raise ValueError(name)


def device_times_us(fn, iters: int):
    """Per-call device time of every kernel ``fn`` submits, and of its GEMM.

    Returns ``(op_us, gemm_us)``: the sum over all device kernels (metadata
    prep, scale transposes, ...) and the time of the single most expensive
    kernel name, which is the grouped GEMM itself for every provider here.
    """
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
        for _ in range(iters):
            fn()
        torch.xpu.synchronize()
    per_name: Dict[str, float] = {}
    for evt in prof.events():
        if str(evt.device_type).endswith("XPU"):
            per_name[evt.name] = per_name.get(evt.name, 0.0) + evt.device_time_total
    if not per_name:
        return float("nan"), float("nan")
    gemm = [t for name, t in per_name.items() if "Gemm" in name]
    gemm_t = max(gemm) if gemm else max(per_name.values())
    return sum(per_name.values()) / iters, gemm_t / iters


def wall_us(fn, iters: int) -> float:
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
    ap.add_argument("--gemm", choices=("up", "down", "both"), default="both")
    ap.add_argument("--providers", default=",".join(ALL_PROVIDERS))
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--kloop", action="store_true", help="SGLANG_NVFP4_WOQ_QUANTIZE_IN_K_LOOP=1")
    ap.add_argument("--align-m", type=int, default=4)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if args.kloop:
        os.environ["SGLANG_NVFP4_WOQ_QUANTIZE_IN_K_LOOP"] = "1"

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    e, topk, hidden, inter = MODELS[args.model]
    gemms = []
    if args.gemm in ("up", "both"):
        gemms.append(("up", 2 * inter, hidden))
    if args.gemm in ("down", "both"):
        gemms.append(("down", hidden, inter))

    print(f"device={torch.xpu.get_device_name()} model={args.model} E={e} topk={topk}")
    rows: List[Dict] = []
    for tokens in args.tokens:
        counts = route_counts(tokens, e, topk, args.align_m)
        for gname, n, k in gemms:
            cs = Case(counts, n, k, "xpu")
            line = {"tokens": tokens, "gemm": gname, "m": cs.m, "n": n, "k": k, "e": e}
            parts = []
            for p in providers:
                try:
                    fn, wbytes = build_provider(p, cs)
                    for _ in range(args.warmup):
                        fn()
                    torch.xpu.synchronize()
                    op_us, gemm_us = device_times_us(fn, args.iters)
                    w_us = wall_us(fn, args.iters)
                except Exception as exc:  # noqa: BLE001
                    line[p] = None
                    parts.append(f"{p}=ERR({type(exc).__name__}: {str(exc)[:80]})")
                    continue
                line[p] = {
                    "op_us": op_us,
                    "gemm_us": gemm_us,
                    "wall_us": w_us,
                    "weight_mb": wbytes / 1e6,
                }
                parts.append(f"{p}={gemm_us:7.1f}/{op_us:7.1f}")
                del fn
            rows.append(line)
            print(
                f"T={tokens:5d} {gname:4s} M={cs.m:6d} N={n:5d} K={k:5d}  gemm/op us: "
                + "  ".join(parts),
                flush=True,
            )
            torch.xpu.empty_cache()

    # Summary ratios.
    def ratio(r, a, b):
        if r.get(a) and r.get(b):
            return r[a]["gemm_us"] / r[b]["gemm_us"]
        return None

    print("\nratios (GEMM-kernel device time, <1 means NVFP4 is faster)")
    print(
        f"{'T':>5s} {'gemm':4s}  {'nvbf16/bf16':>11s} {'nvbf16/mx4a16':>13s}  "
        f"{'nvmx8/mxfp8':>11s} {'nvmx8/mxfp4':>11s}"
    )
    for r in rows:
        vals = [
            ratio(r, "nvfp4_bf16", "bf16"),
            ratio(r, "nvfp4_bf16", "mxfp4_w4a16"),
            ratio(r, "nvfp4_mxfp8", "mxfp8"),
            ratio(r, "nvfp4_mxfp8", "mxfp4"),
        ]
        fmt = lambda v: f"{v:11.2f}" if v is not None else f"{'-':>11s}"  # noqa: E731
        print(
            f"{r['tokens']:5d} {r['gemm']:4s}  {fmt(vals[0])} {fmt(vals[1]):>13s}  "
            f"{fmt(vals[2])} {fmt(vals[3])}"
        )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
