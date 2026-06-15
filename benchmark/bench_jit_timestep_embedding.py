"""Benchmark JIT Timestep Embedding kernel vs PyTorch eager implementation."""

import itertools

import pandas as pd
import torch
import triton

# Storage for bandwidth/performance results
all_results = []


def pytorch_timestep_embedding(
    t: torch.Tensor,
    dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 0.0,
    scale: float = 1.0,
    max_period: int = 10000,
):
    """PyTorch reference implementation of timestep embedding."""
    batch_size = t.shape[0]
    half_dim = dim // 2

    # Compute frequency schedule
    freqs = torch.exp(
        -torch.log(torch.tensor(max_period, dtype=torch.float32, device=t.device))
        * torch.arange(0, half_dim, dtype=torch.float32, device=t.device)
        / (half_dim - downscale_freq_shift)
    )

    # Compute angles
    t_float = t.float().view(-1, 1)
    args = scale * t_float * freqs.view(1, -1)

    # Compute embeddings
    cos_emb = torch.cos(args)
    sin_emb = torch.sin(args)

    if flip_sin_to_cos:
        output = torch.cat([cos_emb, sin_emb], dim=-1)
    else:
        output = torch.cat([sin_emb, cos_emb], dim=-1)

    return output


def calculate_effective_bandwidth(batch_size, dim, time_ms):
    """Calculate memory bandwidth metrics."""
    # Bytes: input timesteps (read) + output embeddings (write)
    bytes_per_element = 4  # float32
    total_bytes = batch_size * bytes_per_element + batch_size * dim * bytes_per_element

    bandwidth_gbs = (total_bytes / 1e9) / (time_ms / 1000.0)

    # FLOPs: exp, log, cos, sin operations
    # Roughly: half_dim * (exp + multiply) + batch * half_dim * (cos + sin + multiply)
    half_dim = dim // 2
    total_flops = half_dim * 3 + batch_size * half_dim * 5
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
        [1, 8, 16, 32, 128, 256, 512, 1024],  # batch_size
        [256, 512, 1024, 2048],  # dim
    )
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "dim"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["torch_ref", "jit"],
        line_names=["PyTorch Reference", "JIT (sglang)"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="jit-timestep-embedding-performance",
        args={},
    )
)
def benchmark(batch_size, dim, provider):
    device = torch.device("xpu")
    dtype = torch.float32

    timesteps = torch.randn(batch_size, dtype=dtype, device=device)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch_ref":
        fn = lambda: pytorch_timestep_embedding(
            timesteps.clone(),
            dim,
            flip_sin_to_cos=False,
            downscale_freq_shift=0.0,
            scale=1.0,
            max_period=10000,
        )
    elif provider == "jit":
        # Import here to allow optional dependency
        try:
            from sglang.jit_kernel.timestep_embedding import (
                timestep_embedding as jit_timestep_embedding,
            )

            fn = lambda: jit_timestep_embedding(
                timesteps.clone(),
                dim,
                flip_sin_to_cos=False,
                downscale_freq_shift=0.0,
                scale=1.0,
                max_period=10000,
                dtype=dtype,
            )
        except ImportError:
            print("Warning: sglang JIT kernel not available, skipping")
            return 0, 0, 0

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    # Calculate metrics
    bw_metrics = calculate_effective_bandwidth(batch_size, dim, ms)

    all_results.append(
        {
            "batch_size": batch_size,
            "dim": dim,
            "provider": provider,
            "time_us": 1000 * ms,
            "bandwidth_gbs": bw_metrics["bandwidth_gbs"],
            "total_bytes_mb": bw_metrics["total_bytes"] / 1e6,
            "total_flops_m": bw_metrics["total_flops"] / 1e6,
            "gflops": bw_metrics["gflops"],
        }
    )

    return 1000 * ms, 1000 * min_ms, 1000 * max_ms


if __name__ == "__main__":
    print("Running Timestep Embedding benchmarks...")
    print("Reference: PyTorch eager implementation")
    print("JIT: sglang.jit_kernel.timestep_embedding (runtime JIT compilation)")
    print(
        "Note: No AOT Timestep Embedding available; comparing against PyTorch reference"
    )
    print("\n" + "=" * 80 + "\n")
    benchmark.run(print_data=True)

    print("Benchmark finished!")

    import pandas as pd

    df = pd.DataFrame(all_results)
    print(df.to_markdown(index=False))
