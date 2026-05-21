import itertools

import torch
import triton
from sgl_kernel import fused_qk_rope_with_cos_sin_cache

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


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    """
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    """
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)


def native_fused_qk_rope_with_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    rotary_dim: int,
    is_neox: bool,
):
    head_size = q.shape[-1]
    positions = positions.flatten()
    num_tokens = positions.shape[0]
    cos_cache, sin_cache = cos_sin_cache.chunk(2, dim=-1)
    cos = cos_cache[positions]
    sin = sin_cache[positions]

    query_shape = q.shape
    query = q.view(num_tokens, -1, head_size)
    query_rot = query[..., :rotary_dim]
    query_pass = query[..., rotary_dim:]
    query_rot = apply_rotary_emb(query_rot, cos, sin, is_neox)
    query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

    key_shape = k.shape
    key = k.view(num_tokens, -1, head_size)
    key_rot = key[..., :rotary_dim]
    key_pass = key[..., rotary_dim:]
    key_rot = apply_rotary_emb(key_rot, cos, sin, is_neox)
    key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
    return query, key


def sglang_fused_qk_rope_with_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    rotary_dim: int,
    is_neox: bool,
):
    return fused_qk_rope_with_cos_sin_cache(
        q, k, cos_sin_cache, positions, rotary_dim, is_neox
    )


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
            line_vals=["sglang", "native"],
            line_names=["SGLang", "native"],
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

        if provider == "sglang" or provider == "sglang1":
            fn = lambda: sglang_fused_qk_rope_with_cache(
                q, k, cos_sin_cache, positions, rope_dim, is_neox
            )
        elif provider == "native":
            fn = lambda: native_fused_qk_rope_with_cache(
                q, k, cos_sin_cache, positions, rope_dim, is_neox
            )

        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

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
    print(f"Testing {len(configs)} configurations...")
    for config in configs:
        num_tokens, num_heads, num_kv_heads, rope_dim, is_neox, dtype = config
        print(
            f"Config: num_tokens={num_tokens}, num_heads={num_heads}, num_kv_heads={num_kv_heads}, rope_dim: {rope_dim}, is_neox: {is_neox}, dtype={dtype}"
        )

    global benchmark_configs
    benchmark_configs = configs

    # Run benchmark
    print("Starting performance benchmark...")
    benchmark = get_benchmark()
    benchmark.run(print_data=True, show_plots=False, save_path=".")
