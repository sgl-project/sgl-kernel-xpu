"""Benchmark: silu_and_mul_clamp  —  SYCL kernel vs pure-PyTorch reference.

Kernel signature (matches unit test):
    silu_and_mul_clamp(input: Tensor[M, 2H], output: Tensor[M, H], swiglu_limit: float)

Run:
    python benchmark/bench_silu_and_mul_clamp.py

Optional flags:
    --limit FLOAT   clamping bound (default: 10.0)
    --bf16-only     benchmark bf16 only
    --fp16-only     benchmark fp16 only
"""

import argparse
import itertools

import pandas as pd
import torch
import triton
import triton.testing
from sgl_kernel import silu_and_mul_clamp


# ---------------------------------------------------------------------------
# Pure-PyTorch reference  (same logic as tests/test_silu_and_mul_clamp.py)
# ---------------------------------------------------------------------------
def silu_and_mul_clamp_torch(
    input: torch.Tensor,
    output: torch.Tensor,
    swiglu_limit: float,
) -> None:
    M, D = input.shape
    H = D // 2
    gate = input[:, :H]
    up = input[:, H:]

    gate_bf16 = gate.to(torch.bfloat16)
    up_bf16 = up.to(torch.bfloat16)
    gate_clamped = gate_bf16.clamp(max=swiglu_limit)
    up_clamped = up_bf16.clamp(min=-swiglu_limit, max=swiglu_limit)

    gate_fp32 = gate_clamped.float()
    up_fp32 = up_clamped.float()
    silu_gate = gate_fp32 * torch.sigmoid(gate_fp32)
    output.copy_((silu_gate * up_fp32).to(input.dtype))


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------
def calculate_diff(M: int, H: int, swiglu_limit: float, dtype: torch.dtype) -> None:
    device = torch.device("xpu")
    inp = torch.randn(M, 2 * H, dtype=dtype, device=device)
    out_sycl = torch.empty(M, H, dtype=dtype, device=device)
    out_ref = torch.empty(M, H, dtype=dtype, device=device)

    silu_and_mul_clamp(inp, out_sycl, swiglu_limit)
    silu_and_mul_clamp_torch(inp, out_ref, swiglu_limit)
    torch.xpu.synchronize()

    diff = torch.abs(out_sycl - out_ref).mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        out_sycl.reshape(-1).float(), out_ref.reshape(-1).float(), dim=0
    ).item()

    tag = f"M={M}, H={H}, dtype={dtype}, limit={swiglu_limit}"
    print(f"  {tag}")
    print(f"    mean |diff|={diff:.6f}  cos_sim={cos_sim:.6f}")
    if cos_sim > 0.99:
        print(f"    ✅ SYCL and reference match")
    else:
        print(f"    ❌ SYCL and reference DIFFER")


# ---------------------------------------------------------------------------
# Bandwidth / FLOPS helper
# ---------------------------------------------------------------------------
def _bw_metrics(M: int, H: int, dtype: torch.dtype, time_ms: float) -> dict:
    elem_bytes = dtype.itemsize if hasattr(dtype, "itemsize") else 2  # bf16/fp16 = 2 B
    # read M*2H, write M*H
    total_bytes = M * (2 * H + H) * elem_bytes
    time_s = time_ms / 1000.0
    bw_gbs = (total_bytes / 1e9) / time_s if time_s > 0 else 0.0
    # ~5 FP ops per output element (2 clamps, sigmoid, 2 muls)
    flops = M * H * 5
    gflops = (flops / 1e9) / time_s if time_s > 0 else 0.0
    return {
        "total_bytes_mb": total_bytes / 1e6,
        "bandwidth_gbs": bw_gbs,
        "total_flops_m": flops / 1e6,
        "gflops": gflops,
    }


# ---------------------------------------------------------------------------
# Benchmark configs
# ---------------------------------------------------------------------------
# Realistic DeepSeek-V4 MoE token counts and intermediate hidden dims
M_range = [1, 16, 64, 128, 256, 512, 1024, 2048, 4096]
H_range = [512, 1024, 2048, 4096, 7168]

all_results = []


def _make_benchmark(dtype: torch.dtype, swiglu_limit: float):
    configs = list(itertools.product(M_range, H_range))

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["M", "H"],
            x_vals=configs,
            line_arg="provider",
            line_vals=["torch", "sycl"],
            line_names=["PyTorch Reference", "SYCL Kernel"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="us",
            plot_name=f"silu-and-mul-clamp-{str(dtype).split('.')[-1]}",
            args={"dtype": dtype, "swiglu_limit": swiglu_limit},
        )
    )
    def bench(M, H, dtype, swiglu_limit, provider):
        device = torch.device("xpu")
        inp = torch.randn(M, 2 * H, dtype=dtype, device=device)
        out = torch.empty(M, H, dtype=dtype, device=device)
        quantiles = [0.5, 0.2, 0.8]

        if provider == "torch":
            fn = lambda: silu_and_mul_clamp_torch(inp, out, swiglu_limit)
        else:
            fn = lambda: silu_and_mul_clamp(inp, out, swiglu_limit)

        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
        bw = _bw_metrics(M, H, dtype, ms)
        all_results.append(
            {
                "M": M,
                "H": H,
                "dtype": str(dtype).split(".")[-1],
                "provider": provider,
                "time_us": round(1000 * ms, 3),
                "bandwidth_gbs": round(bw["bandwidth_gbs"], 2),
                "total_bytes_mb": round(bw["total_bytes_mb"], 2),
                "gflops": round(bw["gflops"], 2),
            }
        )
        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return bench


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark silu_and_mul_clamp")
    parser.add_argument(
        "--limit", type=float, default=10.0, help="swiglu clamping bound"
    )
    parser.add_argument("--bf16-only", action="store_true")
    parser.add_argument("--fp16-only", action="store_true")
    args = parser.parse_args()

    dtypes = []
    if args.bf16_only:
        dtypes = [torch.bfloat16]
    elif args.fp16_only:
        dtypes = [torch.float16]
    else:
        dtypes = [torch.bfloat16, torch.float16]

    # --- correctness spot checks ---
    print("=" * 70)
    print("Correctness check (SYCL vs PyTorch reference)")
    print("=" * 70)
    for dtype in dtypes:
        for M, H in [(16, 1024), (128, 4096), (1024, 7168)]:
            calculate_diff(M, H, args.limit, dtype)
    print()

    # --- performance benchmark ---
    for dtype in dtypes:
        print("=" * 70)
        print(f"Benchmarking dtype={dtype}, swiglu_limit={args.limit}")
        print("=" * 70)
        bench_fn = _make_benchmark(dtype, args.limit)
        bench_fn.run(print_data=True)

    # --- summary table ---
    if all_results:
        print("\n" + "=" * 70)
        print("Full results (all shapes & dtypes)")
        print("=" * 70)
        df = pd.DataFrame(all_results)
        print(df.to_markdown(index=False))

        print("\n" + "=" * 70)
        print("Summary by provider")
        print("=" * 70)
        summary = (
            df.groupby(["dtype", "provider"])
            .agg(
                time_us_mean=("time_us", "mean"),
                time_us_min=("time_us", "min"),
                time_us_max=("time_us", "max"),
                bw_gbs_mean=("bandwidth_gbs", "mean"),
                gflops_mean=("gflops", "mean"),
            )
            .round(2)
        )
        print(summary.to_markdown())
