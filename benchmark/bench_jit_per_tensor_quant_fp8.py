"""Benchmark JIT per-tensor FP8 quantization vs AOT implementation."""

import argparse
from itertools import product

import pandas as pd
import torch
import triton

try:
    import sgl_kernel  # noqa: F401
    from sgl_kernel import sgl_per_tensor_quant_fp8

    HAS_AOT = True
except ImportError:
    HAS_AOT = False
    print("Warning: sgl_kernel not available, AOT comparison will be skipped")

all_results = []


def aot_quant(x, scale, is_static):
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    sgl_per_tensor_quant_fp8(x, out, scale, is_static)
    return out


def jit_quant(x, scale, is_static):
    from sgl_kernel.jit import per_tensor_quant_fp8 as _jit

    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    _jit(x, out, scale, is_static)
    return out


batch_size_range = [16, 32, 64, 128]
seq_len_range = [64, 256, 1024, 2048]
hidden_size = 1024

configs = [
    (bs, seq_len, is_static)
    for is_static in [False, True]
    for bs, seq_len in product(batch_size_range, seq_len_range)
]


def benchmark(batch_size, seq_len, is_static, provider, hidden_size=hidden_size):
    device = torch.device("xpu")
    dtype = torch.float16
    num_tokens = batch_size * seq_len

    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    scale = (
        torch.tensor([0.01], dtype=torch.float32, device=device)
        if is_static
        else torch.zeros(1, dtype=torch.float32, device=device)
    )
    quantiles = [0.5, 0.2, 0.8]

    if provider == "aot":
        if not HAS_AOT:
            print("Warning: sgl_kernel AOT not available, skipping")
            return 0, 0, 0
        fn = lambda: aot_quant(x, scale, is_static)
    elif provider == "jit":
        try:
            fn = lambda: jit_quant(x, scale, is_static)
        except ImportError:
            print("Warning: sgl_kernel JIT module not available, skipping")
            return 0, 0, 0

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    num_elements = num_tokens * hidden_size
    bytes_per_elem = 3 if is_static else 5  # read (+read) input + write fp8
    memory = num_elements * bytes_per_elem + (4 if is_static else 12)
    bandwidth_gbs = memory / (ms / 1e3) / 1e9

    all_results.append(
        {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "hidden_size": hidden_size,
            "is_static": is_static,
            "provider": provider,
            "time_us": 1000 * ms,
            "bandwidth_gbs": bandwidth_gbs,
        }
    )
    return 1000 * ms, 1000 * min_ms, 1000 * max_ms


def get_benchmark(configs=configs, hidden_size=hidden_size):
    return triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["batch_size", "seq_len", "is_static"],
            x_vals=configs,
            line_arg="provider",
            line_vals=["aot", "jit"],
            line_names=["AOT (sgl_kernel)", "JIT (sglang)"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="us",
            plot_name="aot-vs-jit-per-tensor-quant-fp8-performance",
            args={"hidden_size": hidden_size},
        )
    )(benchmark)


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
        "--seq-lens",
        nargs="+",
        type=int,
        default=seq_len_range,
        help="Sequence lengths to sweep",
    )
    parser.add_argument(
        "--hidden-size", type=int, default=hidden_size, help="Hidden size"
    )
    args = parser.parse_args()

    if not HAS_AOT:
        print("ERROR: sgl_kernel (AOT) not available. Please install sgl-kernel-xpu.")
        exit(1)

    print("Running AOT vs JIT per-tensor FP8 quant benchmarks...")
    print("AOT: sgl_kernel.sgl_per_tensor_quant_fp8 (compiled SYCL kernels)")
    print("JIT: sgl_kernel.jit.per_tensor_quant_fp8 (runtime JIT compilation)")
    print("\n" + "=" * 80 + "\n")

    configs = [
        (bs, seq_len, is_static)
        for is_static in [False, True]
        for bs, seq_len in product(args.batch_sizes, args.seq_lens)
    ]
    get_benchmark(configs, args.hidden_size).run(print_data=True)

    print("Benchmark finished!")
    df = pd.DataFrame(all_results)
    for col in ("time_us", "bandwidth_gbs"):
        df[col] = df[col].round(2)
    print(df.to_markdown(index=False))
