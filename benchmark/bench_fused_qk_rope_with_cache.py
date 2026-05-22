import itertools

import pandas as pd
import torch
import triton
from sgl_kernel import fused_qk_rope_with_cos_sin_cache_inplace

DEVICE = "xpu"
MAX_SEQ_LEN = 131072  # common seq length

ROPE_BASE = 10000.0
CACHE_SIZE = 1024 * 128


def create_cos_sin_cache(
    rotary_dim: int,
    max_position: int = MAX_SEQ_LEN,
    base: float = ROPE_BASE,
) -> torch.Tensor:
    """Create cos/sin cache compatible with SGLang layout: [max_pos, rotary_dim]."""
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=DEVICE)
            / rotary_dim
        )
    )
    t = torch.arange(max_position, dtype=torch.float32, device=DEVICE)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    cache = torch.cat((cos, sin), dim=-1)  # [max_pos, rotary_dim]
    return cache


def sglang_fused_qk_rope_with_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    rotary_dim: int,
    is_neox: bool,
):
    fused_qk_rope_with_cos_sin_cache_inplace(
        q, k, cos_sin_cache, positions, rotary_dim, is_neox
    )


def calculate_bandwidth(
    num_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    rotary_dim: int,
    time_ms: float,
    dtype: torch.dtype,
):
    """Estimate effective memory bandwidth (GB/s) for the fused QK RoPE kernel.

    This is an *effective* bandwidth model: it approximates how many bytes are
    moved for one invocation and divides by measured kernel time.

    Byte model:
    - Q/K read + write for rotary part:
        num_tokens * (num_heads + num_kv_heads) * rotary_dim * dtype_size * 2
        (*2 accounts for one read and one write of Q/K rotary slices).
    - Cos/sin cache read:
        num_tokens * rotary_dim * dtype_size
        (counted as one read per token; cache effects can make measured bandwidth
        differ from peak/theoretical device bandwidth).

    Effective bandwidth = total_bytes / elapsed_time.
    """
    dtype_size = 2 if dtype in [torch.float16, torch.bfloat16] else 4
    qkv_read_write_bytes = (
        num_tokens * (num_heads + num_kv_heads) * rotary_dim * dtype_size * 2
    )
    # cos_sin_cache is fetched once per token for the current position.
    cache_read_bytes = num_tokens * rotary_dim * dtype_size
    total_bytes = qkv_read_write_bytes + cache_read_bytes

    time_s = time_ms / 1000.0
    # Return effective GB/s (decimal GB).
    return (total_bytes / 1e9) / time_s


all_results = []


def get_benchmark(device="xpu"):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=[
                "num_tokens",
                "num_heads",
                "num_kv_heads",
                "rope_dim",
                "is_neox",
                "dtype",
            ],
            x_vals=configs,
            line_arg="provider",
            line_vals=["sglang"],
            line_names=["SGLang"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="Latency (us)",
            plot_name="fused-qk-rope-with-cache-performance",
            args={},
        )
    )
    def benchmark(
        num_tokens, num_heads, num_kv_heads, rope_dim, is_neox, dtype, provider
    ):

        q = torch.randn(num_tokens, num_heads, rope_dim, device=device, dtype=dtype)
        k = torch.randn(num_tokens, num_kv_heads, rope_dim, device=device, dtype=dtype)

        positions = torch.randint(
            0, MAX_SEQ_LEN, (num_tokens,), device=device, dtype=torch.int64
        )
        cos_sin_cache = create_cos_sin_cache(rope_dim).to(dtype)

        if provider == "sglang":
            fn = lambda: sglang_fused_qk_rope_with_cache(
                q, k, cos_sin_cache, positions, rope_dim, is_neox
            )

        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
        bandwidth = calculate_bandwidth(
            num_tokens, num_heads, num_kv_heads, rope_dim, ms, dtype
        )
        all_results.append(
            {
                "num_tokens": num_tokens,
                "num_heads": num_heads,
                "num_kv_heads": num_kv_heads,
                "rope_dim": rope_dim,
                "is_neox": is_neox,
                "dtype": dtype,
                "provider": provider,
                "bandwidth": bandwidth,
                "ms": ms,
            }
        )

        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


if __name__ == "__main__":
    bs = [1, 2, 4, 8]
    seq_len = 1024
    num_tokens = [seq_len * b for b in bs]
    # configs from llama3.1 8B an Qwen-235B
    num_heads = [32, 64]
    num_kv_heads = [4, 8]
    rope_dim = [128]
    is_neox = [True]
    dtype = [torch.bfloat16]

    sweep_params = {
        "num_tokens": num_tokens,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "rope_dim": rope_dim,
        "is_neox": is_neox,
        "dtype": dtype,
    }

    keys = sweep_params.keys()
    configs = list(itertools.product(*sweep_params.values()))

    global benchmark_configs
    benchmark_configs = configs

    # Run benchmark
    print("Starting performance benchmark...")
    benchmark = get_benchmark()
    benchmark.run(print_data=False)
    print("Benchmark finished!")

    df = pd.DataFrame(all_results)
    print(df.to_markdown())
