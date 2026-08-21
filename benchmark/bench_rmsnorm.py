import itertools
import os

import pandas as pd
import sgl_kernel
import torch
import triton

# Supported dtypes for benchmarking
DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
DTYPE_BYTES = {
    "fp16": 2,
    "bf16": 2,
}


def make_3d_input(batch_size, seq_len, hidden_size, dtype):
    """Create a 3D input tensor for row-wise RMSNorm benchmarks."""
    return torch.randn(
        batch_size,
        seq_len,
        hidden_size,
        device=torch.device("xpu"),
        dtype=DTYPE_MAP[dtype],
    )


def make_non_flattenable_3d(num_tokens, num_heads, head_dim, dtype):
    """Create a non-flattenable 3D tensor mimicking a QKV slice pattern."""
    assert num_tokens > 1
    total_heads = num_heads + 4
    full = torch.randn(
        num_tokens,
        total_heads * head_dim,
        device=torch.device("xpu"),
        dtype=DTYPE_MAP[dtype],
    )
    q_flat = full[:, : num_heads * head_dim]
    x = q_flat.unflatten(-1, (num_heads, head_dim))
    assert x.stride(0) != x.size(1) * x.stride(1)
    return x


def rms_norm(x, w, eps=1e-6):
    """PyTorch reference implementation of RMSNorm."""
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * w.to(torch.float32)
    x = x.to(orig_dtype)
    return x


def fused_add_rms_norm(x, residual, w, eps=1e-6):
    """PyTorch reference implementation of RMSNorm fused with residual add."""
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x = x + residual.to(torch.float32)
    residual = x.to(orig_dtype)

    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = (x * w.to(torch.float32)).to(orig_dtype)
    return x, residual


def gemma_rms_norm(x, w, eps=1e-6):
    """PyTorch reference implementation of Gemma-style RMSNorm."""
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * (1.0 + w.to(torch.float32))
    x = x.to(orig_dtype)
    return x


def gemma_fused_add_rms_norm(x, residual, w, eps=1e-6):
    """PyTorch reference implementation of Gemma-style RMSNorm fused with residual add."""
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x = x + residual.to(torch.float32)
    residual = x.to(orig_dtype)
    x = x.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * (1.0 + w.to(torch.float32))
    x = x.to(orig_dtype)
    return x, residual


# Benchmark configurations
batch_size_range = [1, 19, 99, 989, 1989]
hidden_size_range = [16, 32, 111, 500, 1024, 4096, 8192]
dtype_range = ["fp16", "bf16"]

norm_configs = list(itertools.product(batch_size_range, hidden_size_range, dtype_range))
three_d_norm_configs = list(
    itertools.product([1, 4, 19], [1, 7, 32], [111, 1024, 4096], dtype_range)
)
non_flattenable_3d_norm_configs = list(
    itertools.product([7, 32], [4, 8], [64, 128], dtype_range)
)

rmsnorm_results = []
fused_add_rmsnorm_results = []
gemma_rmsnorm_results = []
gemma_fused_add_rmsnorm_results = []
rmsnorm_3d_results = []
gemma_rmsnorm_non_flattenable_3d_results = []


def calculate_norm_flops(M, N, fused_add=False):
    """FLOPs per RMSNorm call: 4*M*N (square, mean, rsqrt, weight-mul), plus
    M*N for the residual add in fused-add variants."""
    flops = 4 * M * N
    if fused_add:
        flops += M * N
    return flops


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "hidden_size", "dtype"],
        x_vals=norm_configs,
        line_arg="provider",
        line_vals=["torch", "sglang"],
        line_names=["PyTorch", "SGL Kernel"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="rmsnorm-performance",
        args={},
    )
)
def benchmark_rmsnorm(batch_size, hidden_size, dtype, provider):
    device = torch.device("xpu")
    torch_dtype = DTYPE_MAP[dtype]

    x = torch.randn(batch_size, hidden_size, device=device, dtype=torch_dtype)
    w = torch.randn(hidden_size, device=device, dtype=torch_dtype)
    eps = 1e-6

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        fn = lambda: rms_norm(x, w, eps)
    elif provider == "sglang":
        fn = lambda: sgl_kernel.rmsnorm(x, w, eps)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    ms, min_ms, max_ms = triton.testing.do_bench(
        fn, warmup=50, rep=200, quantiles=quantiles
    )

    # GB/s = logical_tensor_size / time
    total_bytes = (2 * batch_size * hidden_size + hidden_size) * DTYPE_BYTES[dtype]
    bandwidth_gbs = (total_bytes / 1e9) / (ms / 1000.0)
    total_flops = calculate_norm_flops(batch_size, hidden_size, fused_add=False)
    gflops = (total_flops / 1e9) / (ms / 1000.0)

    rmsnorm_results.append(
        {
            "batch_size": batch_size,
            "hidden_size": hidden_size,
            "dtype": dtype,
            "provider": provider,
            "time_us": 1000 * ms,
            "bandwidth_gbs": bandwidth_gbs,
            "total_bytes": total_bytes,
            "total_flops": total_flops,
            "gflops": gflops,
        }
    )

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "hidden_size", "dtype"],
        x_vals=norm_configs,
        line_arg="provider",
        line_vals=["torch", "sglang"],
        line_names=["PyTorch", "SGL Kernel"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="fused-add-rmsnorm-performance",
        args={},
    )
)
def benchmark_fused_add_rmsnorm(batch_size, hidden_size, dtype, provider):
    device = torch.device("xpu")
    torch_dtype = DTYPE_MAP[dtype]

    x = torch.randn(batch_size, hidden_size, device=device, dtype=torch_dtype)
    residual = torch.randn_like(x)
    w = torch.randn(hidden_size, device=device, dtype=torch_dtype)
    eps = 1e-6

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        fn = lambda: fused_add_rms_norm(x.clone(), residual.clone(), w, eps)
    elif provider == "sglang":

        def fn():
            x_fused = x.clone()
            residual_fused = residual.clone()
            sgl_kernel.fused_add_rmsnorm(x_fused, residual_fused, w, eps)

    else:
        raise ValueError(f"Unknown provider: {provider}")

    ms, min_ms, max_ms = triton.testing.do_bench(
        fn, warmup=50, rep=200, quantiles=quantiles
    )

    # GB/s = logical_tensor_size / time
    total_bytes = (4 * batch_size * hidden_size + hidden_size) * DTYPE_BYTES[dtype]
    bandwidth_gbs = (total_bytes / 1e9) / (ms / 1000.0)
    total_flops = calculate_norm_flops(batch_size, hidden_size, fused_add=True)
    gflops = (total_flops / 1e9) / (ms / 1000.0)

    fused_add_rmsnorm_results.append(
        {
            "batch_size": batch_size,
            "hidden_size": hidden_size,
            "dtype": dtype,
            "provider": provider,
            "time_us": 1000 * ms,
            "bandwidth_gbs": bandwidth_gbs,
            "total_bytes": total_bytes,
            "total_flops": total_flops,
            "gflops": gflops,
        }
    )

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "hidden_size", "dtype"],
        x_vals=norm_configs,
        line_arg="provider",
        line_vals=["torch", "sglang"],
        line_names=["PyTorch", "SGL Kernel"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="gemma-rmsnorm-performance",
        args={},
    )
)
def benchmark_gemma_rmsnorm(batch_size, hidden_size, dtype, provider):
    device = torch.device("xpu")
    torch_dtype = DTYPE_MAP[dtype]

    x = torch.randn(batch_size, hidden_size, device=device, dtype=torch_dtype)
    w = torch.randn(hidden_size, device=device, dtype=torch_dtype)
    eps = 1e-6

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        fn = lambda: gemma_rms_norm(x, w, eps)
    elif provider == "sglang":
        fn = lambda: sgl_kernel.gemma_rmsnorm(x, w, eps)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    ms, min_ms, max_ms = triton.testing.do_bench(
        fn, warmup=50, rep=200, quantiles=quantiles
    )

    # GB/s = logical_tensor_size / time
    total_bytes = (2 * batch_size * hidden_size + hidden_size) * DTYPE_BYTES[dtype]
    bandwidth_gbs = (total_bytes / 1e9) / (ms / 1000.0)
    total_flops = calculate_norm_flops(batch_size, hidden_size, fused_add=False)
    gflops = (total_flops / 1e9) / (ms / 1000.0)

    gemma_rmsnorm_results.append(
        {
            "batch_size": batch_size,
            "hidden_size": hidden_size,
            "dtype": dtype,
            "provider": provider,
            "time_us": 1000 * ms,
            "bandwidth_gbs": bandwidth_gbs,
            "total_bytes": total_bytes,
            "total_flops": total_flops,
            "gflops": gflops,
        }
    )

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "hidden_size", "dtype"],
        x_vals=norm_configs,
        line_arg="provider",
        line_vals=["torch", "sglang"],
        line_names=["PyTorch", "SGL Kernel"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="gemma-fused-add-rmsnorm-performance",
        args={},
    )
)
def benchmark_gemma_fused_add_rmsnorm(batch_size, hidden_size, dtype, provider):
    device = torch.device("xpu")
    torch_dtype = DTYPE_MAP[dtype]

    x = torch.randn(batch_size, hidden_size, device=device, dtype=torch_dtype)
    residual = torch.randn_like(x)
    w = torch.randn(hidden_size, device=device, dtype=torch_dtype)
    eps = 1e-6

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        fn = lambda: gemma_fused_add_rms_norm(x.clone(), residual.clone(), w, eps)
    elif provider == "sglang":

        def fn():
            x_fused = x.clone()
            residual_fused = residual.clone()
            sgl_kernel.gemma_fused_add_rmsnorm(x_fused, residual_fused, w, eps)

    else:
        raise ValueError(f"Unknown provider: {provider}")

    ms, min_ms, max_ms = triton.testing.do_bench(
        fn, warmup=50, rep=200, quantiles=quantiles
    )

    # GB/s = logical_tensor_size / time
    total_bytes = (4 * batch_size * hidden_size + hidden_size) * DTYPE_BYTES[dtype]
    bandwidth_gbs = (total_bytes / 1e9) / (ms / 1000.0)
    total_flops = calculate_norm_flops(batch_size, hidden_size, fused_add=True)
    gflops = (total_flops / 1e9) / (ms / 1000.0)

    gemma_fused_add_rmsnorm_results.append(
        {
            "batch_size": batch_size,
            "hidden_size": hidden_size,
            "dtype": dtype,
            "provider": provider,
            "time_us": 1000 * ms,
            "bandwidth_gbs": bandwidth_gbs,
            "total_bytes": total_bytes,
            "total_flops": total_flops,
            "gflops": gflops,
        }
    )

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "seq_len", "hidden_size", "dtype"],
        x_vals=three_d_norm_configs,
        line_arg="provider",
        line_vals=["torch", "sglang"],
        line_names=["PyTorch", "SGL Kernel"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="rmsnorm-3d-performance",
        args={},
    )
)
def benchmark_rmsnorm_3d(batch_size, seq_len, hidden_size, dtype, provider):
    device = torch.device("xpu")
    torch_dtype = DTYPE_MAP[dtype]

    x = make_3d_input(batch_size, seq_len, hidden_size, dtype)
    w = torch.randn(hidden_size, device=device, dtype=torch_dtype)
    eps = 1e-6

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        fn = lambda: rms_norm(x, w, eps)
    elif provider == "sglang":
        fn = lambda: sgl_kernel.rmsnorm(x, w, eps)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    ms, min_ms, max_ms = triton.testing.do_bench(
        fn, warmup=50, rep=200, quantiles=quantiles
    )
    rows = batch_size * seq_len
    total_bytes = (2 * rows * hidden_size + hidden_size) * DTYPE_BYTES[dtype]
    bandwidth_gbs = (total_bytes / 1e9) / (ms / 1000.0)
    total_flops = calculate_norm_flops(rows, hidden_size, fused_add=False)
    gflops = (total_flops / 1e9) / (ms / 1000.0)

    rmsnorm_3d_results.append(
        {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "hidden_size": hidden_size,
            "dtype": dtype,
            "provider": provider,
            "time_us": 1000 * ms,
            "bandwidth_gbs": bandwidth_gbs,
            "total_bytes": total_bytes,
            "total_flops": total_flops,
            "gflops": gflops,
        }
    )

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens", "num_heads", "head_dim", "dtype"],
        x_vals=non_flattenable_3d_norm_configs,
        line_arg="provider",
        line_vals=["torch", "sglang"],
        line_names=["PyTorch", "SGL Kernel"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="gemma-rmsnorm-non-flattenable-3d-performance",
        args={},
    )
)
def benchmark_gemma_rmsnorm_non_flattenable_3d(
    num_tokens, num_heads, head_dim, dtype, provider
):
    device = torch.device("xpu")
    torch_dtype = DTYPE_MAP[dtype]

    x = make_non_flattenable_3d(num_tokens, num_heads, head_dim, dtype)
    w = torch.randn(head_dim, device=device, dtype=torch_dtype)
    eps = 1e-6

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        fn = lambda: gemma_rms_norm(x, w, eps)
    elif provider == "sglang":
        fn = lambda: sgl_kernel.gemma_rmsnorm(x, w, eps)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    ms, min_ms, max_ms = triton.testing.do_bench(
        fn, warmup=50, rep=200, quantiles=quantiles
    )
    total_bytes = (2 * num_tokens * num_heads * head_dim + head_dim) * DTYPE_BYTES[
        dtype
    ]
    bandwidth_gbs = (total_bytes / 1e9) / (ms / 1000.0)
    total_flops = calculate_norm_flops(
        num_tokens * num_heads, head_dim, fused_add=False
    )
    gflops = (total_flops / 1e9) / (ms / 1000.0)

    gemma_rmsnorm_non_flattenable_3d_results.append(
        {
            "num_tokens": num_tokens,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "dtype": dtype,
            "provider": provider,
            "time_us": 1000 * ms,
            "bandwidth_gbs": bandwidth_gbs,
            "total_bytes": total_bytes,
            "total_flops": total_flops,
            "gflops": gflops,
        }
    )

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


def _save_and_report(results, name, out_dir):
    df = pd.DataFrame(results)
    if df.empty:
        print(f"{name}: no benchmark results collected")
        return

    out_csv = os.path.join(out_dir, f"{name}.csv")
    df.to_csv(out_csv, index=False)
    print(f"Wrote results CSV: {out_csv}")

    df["time_us"] = df["time_us"].round(2)
    df["bandwidth_gbs"] = df["bandwidth_gbs"].round(2)
    df["total_bytes"] = df["total_bytes"].round(2)
    df["total_flops"] = df["total_flops"].round(2)
    df["gflops"] = df["gflops"].round(2)
    print(df.to_markdown(index=False))

    index_cols = [
        col for col in ["batch_size", "hidden_size", "dtype"] if col in df.columns
    ]
    if not index_cols:
        index_cols = [
            col
            for col in ["batch_size", "seq_len", "hidden_size", "dtype"]
            if col in df.columns
        ]
    if not index_cols:
        index_cols = [
            col
            for col in ["num_tokens", "num_heads", "head_dim", "dtype"]
            if col in df.columns
        ]
    if not index_cols:
        return

    speed_pivot = df.pivot_table(index=index_cols, columns="provider", values="time_us")
    if "torch" in speed_pivot.columns and "sglang" in speed_pivot.columns:
        speed_pivot["speedup"] = speed_pivot["torch"] / speed_pivot["sglang"]

        avg_speedup = speed_pivot["speedup"].mean()

        print(f"\n{name} avg speedup: {avg_speedup:.2f}x")
        print(f"{name} median speedup: {speed_pivot['speedup'].median():.2f}x")
        print(f"{name} max speedup: {speed_pivot['speedup'].max():.2f}x")
        print(f"{name} min speedup: {speed_pivot['speedup'].min():.2f}x")

        above_avg_count = (speed_pivot["speedup"] > avg_speedup).sum()
        print(f"{name} speedups above avg: {above_avg_count}/{len(speed_pivot)}")


if __name__ == "__main__":
    print("Running RMSNorm benchmarks...")
    benchmark_rmsnorm.run(print_data=True)
    benchmark_fused_add_rmsnorm.run(print_data=True)
    benchmark_gemma_rmsnorm.run(print_data=True)
    benchmark_gemma_fused_add_rmsnorm.run(print_data=True)
    benchmark_rmsnorm_3d.run(print_data=True)
    benchmark_gemma_rmsnorm_non_flattenable_3d.run(print_data=True)

    out_dir = "benchmark/results"
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("RMSNorm Benchmark Results")
    print("=" * 80)
    _save_and_report(rmsnorm_results, "rmsnorm", out_dir)

    print("\n" + "=" * 80)
    print("Fused-Add RMSNorm Benchmark Results")
    print("=" * 80)
    _save_and_report(fused_add_rmsnorm_results, "fused_add_rmsnorm", out_dir)
    print("\n" + "=" * 80)
    print("3D RMSNorm Benchmark Results")
    print("=" * 80)
    _save_and_report(rmsnorm_3d_results, "rmsnorm_3d", out_dir)

    print("\n" + "=" * 80)
    print("Non-Flattenable 3D Gemma RMSNorm Benchmark Results")
    print("=" * 80)
    _save_and_report(
        gemma_rmsnorm_non_flattenable_3d_results,
        "gemma_rmsnorm_non_flattenable_3d",
        out_dir,
    )
    print("\n" + "=" * 80)
    print("Gemma RMSNorm Benchmark Results")
    print("=" * 80)
    _save_and_report(gemma_rmsnorm_results, "gemma_rmsnorm", out_dir)

    print("\n" + "=" * 80)
    print("Gemma Fused-Add RMSNorm Benchmark Results")
    print("=" * 80)
    _save_and_report(
        gemma_fused_add_rmsnorm_results, "gemma_fused_add_rmsnorm", out_dir
    )
