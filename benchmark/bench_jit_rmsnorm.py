"""Benchmark JIT RMSNorm kernel vs AOT implementation."""
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

# Storage for bandwidth/performance results
all_results = []


def calculate_effective_bandwidth(batch_size, hidden_size, time_ms):
    """Calculate memory bandwidth metrics."""
    # Bytes: input (read) + weight (read) + output (write)
    bytes_per_element = 2  # bfloat16
    total_bytes = batch_size * hidden_size * bytes_per_element * 3  # input + weight + output
    
    bandwidth_gbs = (total_bytes / 1e9) / (time_ms / 1000.0)
    
    # FLOPs: variance computation + rsqrt + multiply
    # variance: N multiplies + N-1 adds + 1 divide = ~2N ops
    # rsqrt + multiply: 2 ops per element
    total_flops = batch_size * hidden_size * 4
    gflops = (total_flops / 1e9) / (time_ms / 1000.0)
    
    return {
        "total_bytes": total_bytes,
        "bandwidth_gbs": bandwidth_gbs,
        "total_flops": total_flops,
        "gflops": gflops,
    }


# Test configurations
# JIT RMSNorm supported hidden_sizes: [64, 128, 256, 512, 1024, 1536, 2048, 2304, 2560, 3072, 4096, 5120, 6144, 7168, 8192, 12288, 16384]
configs = list(
    itertools.product(
        [1, 128, 256, 512, 1024, 4096],  # batch_size
        [2048, 4096, 8192, 12288],  # hidden_size (using supported sizes only)
    )
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "hidden_size"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["aot", "jit"],
        line_names=["AOT (sgl_kernel)", "JIT (sglang)"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="aot-vs-jit-rmsnorm-performance",
        args={},
    )
)
def benchmark(batch_size, hidden_size, provider):
    device = torch.device("xpu")
    dtype = torch.bfloat16
    eps = 1e-6
    
    input_tensor = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    weight = torch.randn(hidden_size, device=device, dtype=dtype)
    
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == "aot":
        if not HAS_AOT:
            print("Warning: sgl_kernel AOT not available, skipping")
            return 0, 0, 0
        fn = lambda: sgl_kernel.rmsnorm(input_tensor.clone(), weight, eps)
    elif provider == "jit":
        # Import here to allow optional dependency
        try:
            from sglang.jit_kernel.norm import rmsnorm as jit_rmsnorm
            output = torch.empty_like(input_tensor)
            fn = lambda: jit_rmsnorm(input_tensor.clone(), weight, output, eps)
        except ImportError:
            print("Warning: sglang JIT kernel not available, skipping")
            return 0, 0, 0
    
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    
    # Calculate metrics
    bw_metrics = calculate_effective_bandwidth(batch_size, hidden_size, ms)
    
    all_results.append(
        {
            "batch_size": batch_size,
            "hidden_size": hidden_size,
            "provider": provider,
            "time_us": 1000 * ms,
            "bandwidth_gbs": bw_metrics["bandwidth_gbs"],
            "total_bytes_mb": bw_metrics["total_bytes"] / 1e6,
            "total_flops_m": bw_metrics["total_flops"] / 1e6,
            "gflops": bw_metrics["gflops"],
        }
    )
    
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    if not HAS_AOT:
        print("ERROR: sgl_kernel (AOT) not available. Please install sgl-kernel-xpu.")
        exit(1)
    
    print("Running AOT vs JIT RMSNorm benchmarks...")
    print("AOT: sgl_kernel.rmsnorm (compiled SYCL kernels)")
    print("JIT: sglang.jit_kernel.norm.rmsnorm (runtime JIT compilation)")
    print("\n" + "=" * 80 + "\n")
    
    benchmark.run(print_data=True)
    
    # Print bandwidth results
    print("\n" + "=" * 80)
    print("Effective Bandwidth Results")
    print("=" * 80)
    
    df = pd.DataFrame(all_results)
    df["bandwidth_gbs"] = df["bandwidth_gbs"].round(2)
    df["total_bytes_mb"] = df["total_bytes_mb"].round(2)
    df["time_us"] = df["time_us"].round(2)
    df["total_flops_m"] = df["total_flops_m"].round(2)
    df["gflops"] = df["gflops"].round(2)
    
    print(df.to_markdown(index=False))
    
    # Print summary statistics per provider
    print("\n" + "=" * 80)
    print("Summary Statistics by Provider")
    print("=" * 80)
    summary = df.groupby("provider").agg(
        {
            "bandwidth_gbs": ["mean", "min", "max"],
            "time_us": ["mean", "min", "max"],
            "gflops": ["mean", "min", "max"],
        }
    )
    print(summary.to_markdown())
    
    # Print speedup analysis
    print("\n" + "=" * 80)
    print("Speedup Analysis")
    print("=" * 80)
    
    pivot = df.pivot_table(
        index=["batch_size", "hidden_size"],
        columns="provider",
        values="time_us",
    )
    
    if "aot" in pivot.columns and "jit" in pivot.columns:
        pivot["speedup"] = pivot["aot"] / pivot["jit"]
        print(f"\nJIT Speedup vs AOT (>1.0 means JIT is faster):")
        print(f"Average speedup: {pivot['speedup'].mean():.2f}x")
        print(f"Max speedup: {pivot['speedup'].max():.2f}x")
        print(f"Min speedup: {pivot['speedup'].min():.2f}x")
        print("\nPer-configuration comparison:")
        print(pivot[["aot", "jit", "speedup"]].to_markdown())
