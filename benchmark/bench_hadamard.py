import argparse
import math
from itertools import product

import pandas as pd
import torch
import triton
from sgl_kernel import hadamard_transform

batch_size_range = [1, 16, 64, 256]
dim_range = [64, 256, 1024, 4096, 8192, 16384, 32768]
dtype_range = [torch.float16, torch.bfloat16, torch.float32]

configs = [
    (bs, dim, dtype)
    for bs, dim, dtype in product(batch_size_range, dim_range, dtype_range)
]

all_results = []


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["bs", "dim", "dtype_name"],
        x_vals=[
            (bs, dim, str(dtype).removeprefix("torch.")) for bs, dim, dtype in configs
        ],
        line_arg="provider",
        line_vals=["sgl_kernel"],
        line_names=["SGL Kernel"],
        styles=[("blue", "-")],
        ylabel="Time (ms)",
        plot_name="hadamard-performance",
        args={},
    )
)
def benchmark_hadamard(bs, dim, dtype_name, provider):
    if not torch.xpu.is_available():
        raise RuntimeError("XPU is required for hadamard benchmark")

    dtype = getattr(torch, dtype_name)
    torch.manual_seed(0)
    x = torch.randn(bs, dim, device="xpu", dtype=dtype)
    scale = 1.0 / math.sqrt(dim)

    if provider != "sgl_kernel":
        raise ValueError(f"Unsupported provider: {provider}")

    fn = lambda: hadamard_transform(x, scale=scale)

    for _ in range(5):
        fn()
    torch.xpu.synchronize()

    quantiles = [0.5, 0.25, 0.75]
    ms, min_ms, max_ms = triton.testing.do_bench(
        fn, quantiles=quantiles, return_mode="median"
    )

    total_bytes = x.numel() * x.element_size()
    bandwidth_gb_s = total_bytes / (ms / 1e3) / 1e9

    all_results.append(
        {
            "op": "hadamard_transform",
            "bs": bs,
            "dim": dim,
            "dtype": dtype_name,
            "provider": provider,
            "ms": ms,
            "min_ms": min_ms,
            "max_ms": max_ms,
            "bandwidth_gb_s": bandwidth_gb_s,
        }
    )
    return ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=batch_size_range,
        help="Batch sizes to sweep",
    )
    parser.add_argument(
        "--dims", nargs="+", type=int, default=dim_range, help="Hadamard dims to sweep"
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=["float16", "bfloat16", "float32"],
        default=["float16", "bfloat16", "float32"],
        help="Input dtypes to sweep",
    )
    args = parser.parse_args()

    benchmark_hadamard.benchmarks.x_vals = [
        (bs, dim, dtype_name)
        for bs, dim, dtype_name in product(args.batch_sizes, args.dims, args.dtypes)
    ]
    benchmark_hadamard.run(print_data=False)
    print("Benchmark finished!")

    df = pd.DataFrame(all_results)
    df = df.sort_values(["provider", "dtype", "bs", "dim"]).reset_index(drop=True)
    print(df.to_markdown(index=False))
