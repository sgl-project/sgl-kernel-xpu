import itertools
import pandas as pd
import torch
import triton
import triton.testing
from sgl_kernel import swiglu_with_alpha_and_limit

def reference_swiglu_with_alpha_and_limit(x, alpha, limit):
    gate, up = x[..., ::2], x[..., 1::2]
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)
    return gate * torch.sigmoid(gate * alpha) * (up + 1)

def sglang_swiglu_with_alpha_and_limit(x, alpha, limit):
    return swiglu_with_alpha_and_limit(x, alpha, limit)

def calculate_diff(batch_size, hidden_dim, alpha, limit, dtype):
    device = torch.device("xpu")
    x = torch.randn(batch_size, hidden_dim * 2, device=device, dtype=dtype)
    torch_out = reference_swiglu_with_alpha_and_limit(x, alpha, limit)
    sglang_out = sglang_swiglu_with_alpha_and_limit(x, alpha, limit)
    output_diff = torch.abs(torch_out - sglang_out).mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        torch_out.reshape(-1), sglang_out.reshape(-1), dim=0
    ).item()

    print(f"Mean absolute difference: {output_diff:.6f}")
    print(f"Cosine similarity: {cos_sim:.6f}")

    if cos_sim > 0.99:
        print(f"✅ kernel and reference match for alpha={alpha}, limit={limit}")
    else:
        print(f"❌ kernel and reference differ for alpha={alpha}, limit={limit}")

batch_size_range = [1, 4, 8, 16, 32]
hidden_dim_range = [512, 1024, 2048, 4096]
configs = list(itertools.product(batch_size_range, hidden_dim_range))
all_results = []

def calculate_flops(batch_size, hidden_dim):
    # [B, H], 5 ops per output entry
    return batch_size * hidden_dim * 5

def calculate_effective_bandwidth(
    batch_size: int,
    hidden_dim: int,
    dtype: torch.dtype,
    time_ms: float,
) -> dict:
    input_bytes = batch_size * hidden_dim * 2 * 4  # [B, 2H] float32
    output_bytes = batch_size * hidden_dim * 4     # [B, H] float32
    total_bytes = input_bytes + output_bytes
    time_s = time_ms / 1000.0
    bandwidth_gbs = (total_bytes / 1e9) / time_s if time_s > 0 else 0
    total_flops = calculate_flops(batch_size, hidden_dim)
    gflops = (total_flops / 1e9) / time_s if time_s > 0 else 0
    return {
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "total_bytes": total_bytes,
        "bandwidth_gbs": bandwidth_gbs,
        "total_flops": total_flops,
        "gflops": gflops,
    }

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "hidden_dim"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["reference", "sglang"],
        line_names=["Reference", "SGL Kernel"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="swiglu-with-alpha-limit-performance-v2",
        args={"alpha": 1.0, "limit": 6.0, "dtype": torch.float32},
    )
)
def benchmark_swiglu_alpha_limit(batch_size, hidden_dim, alpha, limit, dtype, provider):
    device = torch.device("xpu")
    x = torch.randn(batch_size, hidden_dim * 2, device=device, dtype=dtype)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "reference":
        fn = lambda: reference_swiglu_with_alpha_and_limit(x, alpha, limit)
    elif provider == "sglang":
        fn = lambda: sglang_swiglu_with_alpha_and_limit(x, alpha, limit)

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    bw_metrics = calculate_effective_bandwidth(
        batch_size, hidden_dim, dtype, ms
    )

    all_results.append(
        {
            "batch_size": batch_size,
            "hidden_dim": hidden_dim,
            "alpha": alpha,
            "limit": limit,
            "dtype": str(dtype),
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
    # Test correctness kernel vs reference
    calculate_diff(
        batch_size=16,
        hidden_dim=1024,
        alpha=1.0,
        limit=6.0,
        dtype=torch.float32,
    )

    calculate_diff(
        batch_size=8,
        hidden_dim=4096,
        alpha=1.0,
        limit=6.0,
        dtype=torch.float32,
    )

    benchmark_swiglu_alpha_limit.run(print_data=True)

    # Print bandwidth and FLOPS results
    print("\n" + "=" * 80)
    print("Effective Bandwidth and FLOPS Results")
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