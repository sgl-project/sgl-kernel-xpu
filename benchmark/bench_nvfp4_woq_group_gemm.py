# SPDX-License-Identifier: Apache-2.0
"""Benchmark the NVFP4 weight-only-quantized MXFP8 grouped GEMM on Intel XPU.

Xe3P has no native NVFP4 MMA, so NVFP4 weights are requantized to MXFP8 inside
the mainloop. Two configurations are timed:

  mxfp8  Upper bound: the plain MXFP8 grouped GEMM over weights that are already
         MXFP8. Not an NVFP4 path — the reference to beat.
  woq    NVFP4 requantized to MXFP8 inside the mainloop. Weight traffic stays at
         4 bits/element; the cost is the in-register conversion.

The gap between them is the price of the emulation, and the weight-byte columns
show what it buys.

Usage:
    python bench_nvfp4_woq_group_gemm.py                 # default sweep
    python bench_nvfp4_woq_group_gemm.py --m 64 --n 512 --k 512 --groups 4
"""

import argparse
import time
from typing import List, Tuple

import torch

MXFP8_BLOCK_SIZE = 32
NVFP4_BLOCK_SIZE = 16
FP8_E4M3_MAX = 448.0
E2M1_MAX = 6.0
UE8M0_BIAS = 127


def is_cri_device() -> bool:
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        return False
    try:
        from sgl_kernel import is_xe3_arch

        return is_xe3_arch()
    except ImportError:
        return False


# --- minimal quantizers (self-contained so the benchmark has no test deps) ---

_E2M1_MAGS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def quantize_mxfp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xf = x.float()
    blocks = xf.reshape(*xf.shape[:-1], xf.shape[-1] // MXFP8_BLOCK_SIZE, MXFP8_BLOCK_SIZE)
    amax = blocks.abs().amax(dim=-1)
    m, e = torch.frexp(amax)
    kexp = torch.where(m <= 0.875, e - 9, e - 8).float()
    kexp = torch.where(amax > 0, kexp, torch.zeros_like(kexp)).clamp(-UE8M0_BIAS, 127)
    q = (blocks / torch.exp2(kexp).unsqueeze(-1)).reshape(*xf.shape)
    return q.to(torch.float8_e4m3fn), (kexp + UE8M0_BIAS).to(torch.uint8)


def quantize_nvfp4(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, k = x.shape
    xf = x.float()
    amax = xf.abs().amax().clamp_min(1e-12)
    gs = ((FP8_E4M3_MAX * E2M1_MAX) / amax).to(torch.float32)
    blocks = xf.reshape(n, k // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE)
    sf = ((blocks.abs().amax(dim=-1) / E2M1_MAX) * gs).clamp(max=FP8_E4M3_MAX)
    sf16 = sf.to(torch.float8_e4m3fn)
    inv = (sf16.float() / gs).clamp_min(1e-30).unsqueeze(-1)
    scaled = (blocks / inv).reshape(n, k)

    mags = torch.tensor(_E2M1_MAGS)
    mid = (mags[:-1] + mags[1:]) / 2
    a = scaled.abs().clamp(max=E2M1_MAX)
    bucket = torch.bucketize(a, mid, right=False).to(torch.int64)
    at_mid = (a.unsqueeze(-1) == mid).any(dim=-1)
    idx = torch.where(at_mid, torch.where(bucket % 2 == 0, bucket, bucket + 1), bucket)
    nib = (((scaled < 0).to(torch.uint8)) << 3) | idx.to(torch.uint8)
    packed = nib[:, 0::2] | (nib[:, 1::2] << 4)
    return packed.to(torch.uint8).contiguous(), sf16, gs


def build_inputs(groups: int, m_per_group: int, n: int, k: int, device: str):
    torch.manual_seed(0)
    w = torch.randn(groups, n, k) * 0.05
    qs, sfs, gss = [], [], []
    for e in range(groups):
        q, sf, gs = quantize_nvfp4(w[e])
        qs.append(q)
        sfs.append(sf)
        gss.append(1.0 / gs)
    b_fp4 = torch.stack(qs).to(device)
    b_sf = torch.stack(sfs).to(device)
    b_gs = torch.stack(gss).float().to(device)

    # MXFP8 weights for the upper-bound configuration, quantized straight from
    # the same FP32 source rather than requantized from NVFP4.
    b_mx, b_smx = quantize_mxfp8(w)
    b_mx, b_smx = b_mx.to(device), b_smx.to(device)

    m_total = groups * m_per_group
    a_q, a_s = quantize_mxfp8(torch.randn(m_total, k) * 0.5)
    a_q, a_s = a_q.to(device), a_s.to(device)

    problem_sizes = torch.tensor(
        [[m_per_group, n, k]] * groups, dtype=torch.int32, device=device
    )
    expert_offsets = torch.tensor(
        [e * m_per_group for e in range(groups)], dtype=torch.int32, device=device
    )
    return a_q, a_s, b_fp4, b_sf, b_gs, b_mx, b_smx, problem_sizes, expert_offsets, m_total


def timeit(fn, num_warmup: int, num_run: int) -> float:
    for _ in range(num_warmup):
        fn()
    torch.xpu.synchronize()
    start = time.perf_counter()
    for _ in range(num_run):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - start) / num_run * 1e6  # us


def bench_one(groups: int, m_per_group: int, n: int, k: int, num_warmup: int, num_run: int):
    from sgl_kernel import (
        fp8_blockwise_scaled_grouped_mm,
        nvfp4_woq_mxfp8_scaled_grouped_mm,
        nvfp4_woq_prep,
    )

    device = "xpu"
    (a_q, a_s, b_fp4, b_sf, b_gs, b_mx, b_smx,
     problem_sizes, expert_offsets, m_total) = build_inputs(
        groups, m_per_group, n, k, device
    )
    workspace = torch.zeros((64 * 1024 * 1024,), dtype=torch.uint8, device=device)
    empty = torch.empty((0,), dtype=torch.int64, device=device)
    zeros_stride = torch.zeros((groups,), dtype=torch.int64, device=device)
    zeros_layout = torch.zeros((groups, 5), dtype=torch.int32, device=device)
    out = torch.zeros((m_total, n), dtype=torch.float32, device=device)

    c_b, sfmx_b = nvfp4_woq_prep(b_sf, b_gs)

    def mxfp8_gemm():
        fp8_blockwise_scaled_grouped_mm(
            out, empty, empty, empty, empty, empty,
            a_q, b_mx, a_s, b_smx,
            zeros_stride, zeros_stride, zeros_stride, zeros_layout, zeros_layout,
            problem_sizes, expert_offsets, workspace,
        )

    def woq_gemm():
        nvfp4_woq_mxfp8_scaled_grouped_mm(
            out, a_q, b_fp4, a_s, sfmx_b, c_b, problem_sizes, expert_offsets, workspace
        )

    flops = 2.0 * m_total * n * k
    res = {"groups": groups, "m_per_group": m_per_group, "n": n, "k": k}
    for name, fn in (("mxfp8", mxfp8_gemm), ("woq", woq_gemm)):
        us = timeit(fn, num_warmup, num_run)
        res[f"{name}_us"] = us
        res[f"{name}_tflops"] = flops / (us * 1e-6) / 1e12
    res["woq_vs_mxfp8"] = res["woq_us"] / res["mxfp8_us"]
    # NVFP4 weight bytes vs MXFP8 weight bytes actually streamed per GEMM.
    res["weight_mb_nvfp4"] = groups * n * k * 0.5 / 1e6
    res["weight_mb_mxfp8"] = groups * n * k * 1.0 / 1e6
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=None)
    ap.add_argument("--m", type=int, default=None, help="M per group")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    if not is_cri_device():
        print("NVFP4 WoQ kernels require an Xe3P (CRI) XPU; skipping.")
        return

    if args.n is not None:
        shapes = [(args.groups or 2, args.m or 64, args.n, args.k or args.n)]
    else:
        shapes = [
            (2, 64, 128, 256),
            (2, 64, 256, 512),
            (4, 64, 256, 512),
            (2, 256, 512, 512),
        ]

    rows: List[dict] = []
    for groups, m, n, k in shapes:
        try:
            rows.append(bench_one(groups, m, n, k, args.warmup, args.runs))
        except Exception as exc:  # noqa: BLE001
            print(f"groups={groups} m={m} n={n} k={k}: FAILED {type(exc).__name__}: {exc}")
            continue
        r = rows[-1]
        print(
            f"groups={r['groups']} m/g={r['m_per_group']} n={r['n']} k={r['k']}  "
            f"mxfp8={r['mxfp8_us']:.1f}us ({r['mxfp8_tflops']:.2f} TF)  "
            f"woq={r['woq_us']:.1f}us ({r['woq_tflops']:.2f} TF)  "
            f"woq/mxfp8={r['woq_vs_mxfp8']:.2f}x  "
            f"weights {r['weight_mb_nvfp4']:.1f}MB vs {r['weight_mb_mxfp8']:.1f}MB",
            flush=True,
        )


if __name__ == "__main__":
    main()
