"""Benchmark JIT activation-and-mul kernels vs AOT implementation."""

import itertools

import pandas as pd
import torch
import triton

try:
    import sgl_kernel

    HAS_AOT = True
except ImportError:
    HAS_AOT = False
    print("Warning: sgl_kernel not available, AOT comparison will be skipped")

all_results = []

# AOT ops only support fp16/bf16 (see TripleOps.cpp dispatch).
_AOT_OPS = {
    "silu": (lambda inp, out: sgl_kernel.silu_and_mul(inp, out)) if HAS_AOT else None,
    "gelu": (lambda inp, out: sgl_kernel.gelu_and_mul(inp, out)) if HAS_AOT else None,
    "gelu_tanh": (
        (lambda inp, out: sgl_kernel.gelu_tanh_and_mul(inp, out)) if HAS_AOT else None
    ),
}


def calculate_effective_bandwidth(num_tokens, dim, dtype, time_ms):
    """Read [num_tokens, 2*dim] input, write [num_tokens, dim] output."""
    bytes_per_element = torch.finfo(dtype).bits // 8
    total_bytes = num_tokens * dim * bytes_per_element * 3  # 2*dim read + dim write
    bandwidth_gbs = (total_bytes / 1e9) / (time_ms / 1000.0)

    # act(gate)*up: ~8 FLOPs/element (activation transcendental + mul).
    total_flops = num_tokens * dim * 8
    gflops = (total_flops / 1e9) / (time_ms / 1000.0)
    return {
        "total_bytes": total_bytes,
        "bandwidth_gbs": bandwidth_gbs,
        "total_flops": total_flops,
        "gflops": gflops,
    }


# (num_tokens, dim) where input is [num_tokens, 2*dim].
configs = list(
    itertools.product(
        [1, 128, 512, 4096],  # num_tokens
        [1024, 4096, 14336],  # dim (SwiGLU hidden)
        ["silu", "gelu", "gelu_tanh"],  # activation
    )
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens", "dim", "op_name"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["aot", "jit"],
        line_names=["AOT (sgl_kernel)", "JIT (sglang)"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="aot-vs-jit-activation-performance",
        args={},
    )
)
def benchmark(num_tokens, dim, op_name, provider):
    device = torch.device("xpu")
    dtype = torch.bfloat16

    x = torch.randn(num_tokens, 2 * dim, device=device, dtype=dtype)
    out = torch.empty(num_tokens, dim, device=device, dtype=dtype)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "aot":
        if not HAS_AOT:
            print("Warning: sgl_kernel AOT not available, skipping")
            return 0, 0, 0
        aot_fn = _AOT_OPS[op_name]
        fn = lambda: aot_fn(x, out)
    elif provider == "jit":
        try:
            from sgl_kernel.jit import gelu_and_mul, gelu_tanh_and_mul, silu_and_mul

            jit_fn = {
                "silu": silu_and_mul,
                "gelu": gelu_and_mul,
                "gelu_tanh": gelu_tanh_and_mul,
            }[op_name]
            fn = lambda: jit_fn(x, out)
        except ImportError:
            print("Warning: sgl_kernel JIT module not available, skipping")
            return 0, 0, 0

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    bw = calculate_effective_bandwidth(num_tokens, dim, dtype, ms)
    all_results.append(
        {
            "num_tokens": num_tokens,
            "dim": dim,
            "op_name": op_name,
            "provider": provider,
            "time_us": 1000 * ms,
            "bandwidth_gbs": bw["bandwidth_gbs"],
            "total_bytes_mb": bw["total_bytes"] / 1e6,
            "gflops": bw["gflops"],
        }
    )

    return 1000 * ms, 1000 * min_ms, 1000 * max_ms


if __name__ == "__main__":
    if not HAS_AOT:
        print("ERROR: sgl_kernel (AOT) not available. Please install sgl-kernel-xpu.")
        exit(1)

    print("Running AOT vs JIT activation-and-mul benchmarks...")
    print("AOT: sgl_kernel.{silu,gelu,gelu_tanh}_and_mul (compiled SYCL kernels)")
    print("JIT: sgl_kernel.jit.{silu,gelu,gelu_tanh}_and_mul (runtime JIT compilation)")
    print("\n" + "=" * 80 + "\n")

    benchmark.run(print_data=True)

    print("Benchmark finished!")
    df = pd.DataFrame(all_results)
    for col in ("bandwidth_gbs", "total_bytes_mb", "time_us", "gflops"):
        df[col] = df[col].round(2)
    print(df.to_markdown(index=False))
