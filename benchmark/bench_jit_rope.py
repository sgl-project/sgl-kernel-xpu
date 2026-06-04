"""Benchmark AOT vs JIT RoPE kernel implementations."""

import importlib.util
import itertools

import pandas as pd
import torch
import triton

# Check if AOT kernel is available
HAS_AOT = importlib.util.find_spec("sgl_kernel") is not None
if not HAS_AOT:
    print("Warning: sgl_kernel (AOT) not available")

# Storage for results
all_results = []

# Benchmark configurations
batch_sizes = [1, 8, 16, 32, 128, 256, 512, 1024]
head_dims = [64, 128, 256]
num_heads_configs = [8, 16, 32]

configs = list(itertools.product(batch_sizes, head_dims, num_heads_configs))

MAX_SEQ_LEN = 131072
ROPE_BASE = 10000.0


def create_cos_sin_cache(
    rotary_dim: int,
    max_position: int = MAX_SEQ_LEN,
    base: float = ROPE_BASE,
) -> torch.Tensor:
    """Create cos/sin cache compatible with SGLang layout: [max_pos, rotary_dim]."""
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
    t = torch.arange(max_position).float()
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    cache = torch.cat([cos, sin], dim=-1)
    return cache


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "head_dim", "num_heads"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["aot", "jit"] if HAS_AOT else ["jit"],
        line_names=(
            ["AOT (sgl_kernel)", "JIT (sglang)"] if HAS_AOT else ["JIT (sglang)"]
        ),
        styles=[("blue", "-"), ("green", "-")],
        ylabel="Time (us)",
        plot_name="rope-aot-vs-jit-performance",
        args={},
    )
)
def benchmark(batch_size, head_dim, num_heads, provider):
    device = torch.device("xpu")
    dtype = torch.bfloat16
    is_neox = True  # Test with Neox style

    # Create inputs
    q = torch.randn(batch_size, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, num_heads, head_dim, dtype=dtype, device=device)
    positions = torch.arange(
        batch_size, dtype=torch.int64, device=device
    )  # AOT kernel needs int64

    # Different kernels have different requirements for cos_sin_cache dtype
    # AOT kernel: must match q/k dtype (bfloat16)
    # JIT kernel: must be float32
    cos_sin_cache_aot = create_cos_sin_cache(head_dim).to(device=device, dtype=dtype)
    cos_sin_cache_jit = create_cos_sin_cache(head_dim).to(
        device=device, dtype=torch.float32
    )

    # Select implementation
    if provider == "aot":

        def fn():
            # AOT kernel returns new tensors, not in-place
            q_out, k_out = torch.ops.sgl_kernel.rotary_embedding(
                positions,
                q.view(batch_size, -1),
                k.view(batch_size, -1),
                head_dim,
                cos_sin_cache_aot,
                is_neox,
            )
            # Copy back to simulate in-place behavior for fair comparison
            q.copy_(q_out.view(q.shape))
            k.copy_(k_out.view(k.shape))

    else:  # jit
        from sglang.jit_kernel.rope import apply_rope_inplace

        def fn():
            apply_rope_inplace(
                q, k, cos_sin_cache_jit, positions, is_neox=is_neox, rope_dim=head_dim
            )

    # Warmup
    for _ in range(10):
        fn()
    torch.xpu.synchronize()

    # Benchmark
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    # Calculate bandwidth metrics
    # Memory: read q, k, cos_sin_cache and write q, k
    bytes_per_element = q.element_size()
    total_bytes = (
        batch_size * num_heads * head_dim * bytes_per_element * 4  # read/write q, k
        + batch_size * head_dim * bytes_per_element  # read cos_sin_cache
    )
    bandwidth_gbs = (total_bytes / (ms * 1e-3)) / 1e9

    # Operations: 4 ops per element (2 multiply, 1 add, 1 subtract) for rotary
    total_ops = batch_size * num_heads * head_dim * 4
    gflops = (total_ops / (ms * 1e-3)) / 1e9

    all_results.append(
        {
            "batch_size": batch_size,
            "head_dim": head_dim,
            "num_heads": num_heads,
            "provider": provider,
            "time_us": 1000 * ms,
            "bandwidth_gbs": bandwidth_gbs,
            "gflops": gflops,
        }
    )

    return 1000 * ms, 1000 * min_ms, 1000 * max_ms


if __name__ == "__main__":
    if not HAS_AOT:
        print("ERROR: sgl_kernel (AOT) not available. Please install sgl-kernel-xpu.")
        exit(1)

    print("Running AOT vs JIT RoPE benchmarks...")
    print(
        "AOT: sgl_kernel.apply_rope_with_cos_sin_cache_inplace (compiled SYCL kernels)"
    )
    print("JIT: sglang.jit_kernel.rope.apply_rope_inplace (runtime JIT compilation)")
    print("\n" + "=" * 80 + "\n")

    benchmark.run(print_data=True)

    print("Benchmark finished!")

    import pandas as pd

    df = pd.DataFrame(all_results)
    print(df.to_markdown(index=False))
