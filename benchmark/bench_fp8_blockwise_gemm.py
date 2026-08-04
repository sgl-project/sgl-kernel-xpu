"""
Copyright (C) 2026 Intel Corporation, All rights reserved.
SPDX-License-Identifier: BSD-3-Clause

Benchmark for fp8_blockwise_scaled_mm on XPU.

Measurement is a single warmup + a single timed launch with xpu.synchronize().
We deliberately avoid triton.do_bench because it auto-scales iteration counts
to hit ~100 ms of measurement, which is intractable on the CRI simulator where
a single 512^3 GEMM takes ~16 s.
"""

import time

import pandas as pd
import torch
from sgl_kernel import fp8_blockwise_scaled_mm

# Static shape list. N and K must be multiples of 128 for the blockwise kernel.
# Chosen to cover a range of M (thin -> square -> tall) and N/K sizes without
# taking forever on the CRI simulator.
SHAPES = [
    # (M,    N,    K) — kept small so the CI simulator finishes in reasonable time.
    (1, 128, 128),
    (1, 256, 256),
    (128, 128, 128),
    (128, 256, 256),
    (256, 256, 256),
    (512, 512, 512),
]

OUT_DTYPES = [torch.bfloat16, torch.float16]


def cdiv(a: int, b: int) -> int:
    return -(a // -b)


def scale_shape(shape, group_shape):
    return tuple(cdiv(shape[i], group_shape[i]) for i in range(len(group_shape)))


def _make_inputs(M, N, K, device):
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    a_fp32 = (torch.rand(M, K, dtype=torch.float32, device=device) - 0.5) * 2 * fp8_max
    A = a_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    b_fp32 = (torch.rand(N, K, dtype=torch.float32, device=device) - 0.5) * 2 * fp8_max
    B = b_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn).t()

    scale_a_shape = scale_shape(A.shape, (1, 128))
    scale_b_shape = scale_shape(B.shape, (128, 128))
    scale_a = torch.randn(scale_a_shape, device=device, dtype=torch.float32) * 0.001
    scale_b = torch.randn(scale_b_shape, device=device, dtype=torch.float32) * 0.001
    scale_a = scale_a.t().contiguous().t()
    scale_b = scale_b.t().contiguous().t()
    return A, B, scale_a, scale_b


def calc_bandwidth(M, N, K, out_dtype, time_ms):
    out_elem_bytes = torch.finfo(out_dtype).bits // 8
    bytes_a = M * K * 1
    bytes_b = N * K * 1
    bytes_sa = M * (K // 128) * 4
    bytes_sb = (K // 128) * (N // 128) * 4
    bytes_d = M * N * out_elem_bytes
    total_bytes = bytes_a + bytes_b + bytes_sa + bytes_sb + bytes_d
    time_s = time_ms / 1e3
    return {
        "total_bytes_mb": total_bytes / 1e6,
        "bandwidth_gbs": (total_bytes / 1e9) / time_s,
        "gflops": (2.0 * M * N * K / 1e9) / time_s,
    }


def time_once(M, N, K, out_dtype):
    device = torch.device("xpu")
    A, B, sa, sb = _make_inputs(M, N, K, device)

    _ = fp8_blockwise_scaled_mm(A, B, sa, sb, out_dtype)
    torch.xpu.synchronize()

    t0 = time.perf_counter()
    _ = fp8_blockwise_scaled_mm(A, B, sa, sb, out_dtype)
    torch.xpu.synchronize()
    t1 = time.perf_counter()

    ms = (t1 - t0) * 1e3
    bw = calc_bandwidth(M, N, K, out_dtype, ms)
    return {
        "M": M,
        "N": N,
        "K": K,
        "out_dtype": str(out_dtype).replace("torch.", ""),
        "time_ms": ms,
        "total_bytes_mb": bw["total_bytes_mb"],
        "bandwidth_gbs": bw["bandwidth_gbs"],
        "gflops": bw["gflops"],
    }


if __name__ == "__main__":
    rows = []
    for M, N, K in SHAPES:
        for dt in OUT_DTYPES:
            print(f"[start] M={M} N={N} K={K} dtype={dt}", flush=True)
            row = time_once(M, N, K, dt)
            print(
                f"[done ] M={M} N={N} K={K} dtype={dt}: "
                f"{row['time_ms']:.2f} ms  "
                f"({row['bandwidth_gbs']:.3f} GB/s, {row['gflops']:.3f} GFLOP/s)",
                flush=True,
            )
            rows.append(row)

    df = pd.DataFrame(rows)
    for col in ("time_ms", "total_bytes_mb", "bandwidth_gbs", "gflops"):
        df[col] = df[col].round(3)
    print("\n" + "=" * 80)
    print(df.to_markdown(index=False))
