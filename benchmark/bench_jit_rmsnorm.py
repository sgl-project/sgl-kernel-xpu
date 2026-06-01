"""Benchmark JIT RMSNorm kernel vs PyTorch eager implementation."""
import itertools

import pandas as pd
import torch
import triton

# Storage for bandwidth/performance results
all_results = []


def pytorch_rmsnorm(input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    """PyTorch reference implementation of RMS normalization."""
    orig_dtype = input.dtype
    input_float = input.float()
    variance = input_float.pow(2).mean(dim=-1, keepdim=True)
    output = input_float * torch.rsqrt(variance + eps) * weight.float()
    return output.to(orig_dtype)


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
configs = list(
    itertools.product(
        [1, 128, 256, 512, 1024, 4096],  # batch_size
        [2048, 4096, 8192, 11008],  # hidden_size
    )
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "hidden_size"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["torch", "jit_xpu"],
        line_names=["PyTorch Eager", "JIT XPU"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="jit-rmsnorm-performance",
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
    
    if provider == "torch":
        fn = lambda: pytorch_rmsnorm(input_tensor.clone(), weight, eps)
    elif provider == "jit_xpu":
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
    print("Running JIT RMSNorm benchmarks...")
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
    
    if "torch" in pivot.columns and "jit_xpu" in pivot.columns:
        pivot["speedup"] = pivot["torch"] / pivot["jit_xpu"]
        print(f"\nAverage speedup: {pivot['speedup'].mean():.2f}x")
        print(f"Max speedup: {pivot['speedup'].max():.2f}x")
        print(f"Min speedup: {pivot['speedup'].min():.2f}x")
        print("\nPer-configuration speedup:")
        print(pivot[["torch", "jit_xpu", "speedup"]].to_markdown())
