"""Benchmark JIT QKNorm kernel vs PyTorch eager implementation."""
import itertools

import pandas as pd
import torch
import triton

# Storage for bandwidth/performance results
all_results = []


def pytorch_qknorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float = 1e-6,
):
    """PyTorch reference implementation of QKNorm."""
    # q, k: [batch_size, num_heads, head_dim]
    orig_dtype = q.dtype
    
    # Normalize q
    q_float = q.float()
    q_var = q_float.pow(2).mean(dim=-1, keepdim=True)
    q_norm = q_float * torch.rsqrt(q_var + eps) * q_weight.float()
    q[:] = q_norm.to(orig_dtype)
    
    # Normalize k
    k_float = k.float()
    k_var = k_float.pow(2).mean(dim=-1, keepdim=True)
    k_norm = k_float * torch.rsqrt(k_var + eps) * k_weight.float()
    k[:] = k_norm.to(orig_dtype)


def calculate_effective_bandwidth(batch_size, num_heads, head_dim, time_ms):
    """Calculate memory bandwidth metrics."""
    # Bytes: q (read+write) + k (read+write) + q_weight (read) + k_weight (read)
    bytes_per_element = 2  # bfloat16
    total_bytes = (
        batch_size * num_heads * head_dim * bytes_per_element * 4  # q, k read+write
        + head_dim * bytes_per_element * 2  # weights
    )
    
    bandwidth_gbs = (total_bytes / 1e9) / (time_ms / 1000.0)
    
    # FLOPs: 2 * (variance + rsqrt + multiply) for q and k
    # variance: ~2*head_dim ops per token
    # rsqrt + multiply: 2 ops per element
    total_elements = batch_size * num_heads * head_dim
    total_flops = 2 * total_elements * 4  # 4 ops per element, for both q and k
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
        [1, 8, 16, 32, 64, 128],  # batch_size
        [8, 16, 32],  # num_heads
        [64, 128, 256],  # head_dim
    )
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "num_heads", "head_dim"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["torch", "jit_xpu"],
        line_names=["PyTorch Eager", "JIT XPU"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="jit-qknorm-performance",
        args={},
    )
)
def benchmark(batch_size, num_heads, head_dim, provider):
    device = torch.device("xpu")
    dtype = torch.bfloat16
    eps = 1e-6
    
    q = torch.randn(batch_size, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, head_dim, device=device, dtype=dtype)
    q_weight = torch.randn(head_dim, device=device, dtype=dtype)
    k_weight = torch.randn(head_dim, device=device, dtype=dtype)
    
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == "torch":
        def fn():
            q_clone = q.clone()
            k_clone = k.clone()
            pytorch_qknorm(q_clone, k_clone, q_weight, k_weight, eps)
            return q_clone, k_clone
    elif provider == "jit_xpu":
        # Import here to allow optional dependency
        try:
            from sglang.jit_kernel.norm import fused_inplace_qknorm as jit_qknorm
            def fn():
                q_clone = q.clone()
                k_clone = k.clone()
                jit_qknorm(q_clone, k_clone, q_weight, k_weight, eps, head_dim=head_dim)
                return q_clone, k_clone
        except ImportError:
            print("Warning: sglang JIT kernel not available, skipping")
            return 0, 0, 0
    
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    
    # Calculate metrics
    bw_metrics = calculate_effective_bandwidth(batch_size, num_heads, head_dim, ms)
    
    all_results.append(
        {
            "batch_size": batch_size,
            "num_heads": num_heads,
            "head_dim": head_dim,
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
    print("Running JIT QKNorm benchmarks...")
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
        index=["batch_size", "num_heads", "head_dim"],
        columns="provider",
        values="time_us",
    )
    
    if "torch" in pivot.columns and "jit_xpu" in pivot.columns:
        pivot["speedup"] = pivot["torch"] / pivot["jit_xpu"]
        print(f"\nAverage speedup: {pivot['speedup'].mean():.2f}x")
        print(f"Max speedup: {pivot['speedup'].max():.2f}x")
        print(f"Min speedup: {pivot['speedup'].min():.2f}x")
        print("\nTop 10 speedups:")
        print(pivot[["torch", "jit_xpu", "speedup"]].nlargest(10, "speedup").to_markdown())
