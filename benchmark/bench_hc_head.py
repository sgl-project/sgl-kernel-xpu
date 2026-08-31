from itertools import product

import pandas as pd
import torch
import triton
from sgl_kernel import fused_hc_head

batch_size_range = [1, 4, 16, 64]
seq_len_range = [1, 16, 128, 1024]
hidden_size_range = [4096, 7168]
dtype_range = [torch.bfloat16]

hc_mult = 4
norm_eps = 1e-6
hc_eps = 1e-6

MAX_TOKENS = 1_000_000

configs = [
    (b, s, h, dt)
    for b, s, h, dt in product(
        batch_size_range, seq_len_range, hidden_size_range, dtype_range
    )
    if b * s <= MAX_TOKENS
]

all_results = []


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "seq_len", "hidden_size", "dtype"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["sgl_kernel"],
        line_names=["SGL Kernel fused_hc_head"],
        styles=[("green", "-")],
        ylabel="ms",
        plot_name="fused-hc-head-performance",
        args={},
    )
)
def benchmark(batch_size, seq_len, hidden_size, dtype, provider):
    device = torch.device("xpu")

    torch.manual_seed(42)
    torch.xpu.manual_seed_all(42)

    tokens = batch_size * seq_len
    x = torch.randn(
        tokens,
        hc_mult,
        hidden_size,
        dtype=dtype,
        device=device,
    )
    hc_fn = torch.randn(
        hc_mult,
        hc_mult * hidden_size,
        dtype=torch.float32,
        device=device,
    )
    hc_scale = torch.randn(1, dtype=torch.float32, device=device)
    hc_base = torch.randn(hc_mult, dtype=torch.float32, device=device)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "sgl_kernel":
        fn = lambda: fused_hc_head(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            norm_eps=norm_eps,
            hc_eps=hc_eps,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    # Approximate HBM traffic: read x + read hc_fn + read scales/bases + write y
    read_bytes = (
        tokens * hc_mult * hidden_size * x.element_size()
        + hc_mult * hc_mult * hidden_size * 4
        + hc_scale.numel() * 4
        + hc_base.numel() * 4
    )
    write_bytes = tokens * hidden_size * x.element_size()
    total_bytes = read_bytes + write_bytes
    bandwidth_gbs = total_bytes / (ms / 1e3) / 1e9

    all_results.append(
        {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "hidden_size": hidden_size,
            "dtype": str(dtype).replace("torch.", ""),
            "provider": provider,
            "time_ms": ms,
            "tok_per_sec": tokens / (ms / 1e3),
            "Mtok_per_sec": tokens / (ms / 1e3) / 1e6,
            "bandwidth_gbs": bandwidth_gbs,
        }
    )

    return ms, max_ms, min_ms


if __name__ == "__main__":
    benchmark.run(print_data=True)

    print("\n" + "=" * 80)
    print("Performance Results")
    print("=" * 80)

    df = pd.DataFrame(all_results)
    df["time_ms"] = df["time_ms"].round(6)
    df["tok_per_sec"] = df["tok_per_sec"].round(2)
    df["Mtok_per_sec"] = df["Mtok_per_sec"].round(4)
    df["bandwidth_gbs"] = df["bandwidth_gbs"].round(4)

    print(df.to_markdown(index=False))

    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    summary = df.groupby("provider").agg(
        {
            "time_ms": ["mean", "min", "max"],
            "Mtok_per_sec": ["mean", "min", "max"],
            "bandwidth_gbs": ["mean", "min", "max"],
        }
    )
    print(summary)
