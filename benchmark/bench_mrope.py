from __future__ import annotations

import itertools
from typing import List, Optional

import pandas as pd
import torch
import triton
from sgl_kernel import multomodal_rotary_embedding


def apply_interleaved_rope(x: torch.Tensor, mrope_section: list) -> torch.Tensor:
    x_t = x[0].clone()
    x_t[..., 1 : mrope_section[1] * 3 : 3] = x[1, ..., 1 : mrope_section[1] * 3 : 3]
    x_t[..., 2 : mrope_section[2] * 3 : 3] = x[2, ..., 2 : mrope_section[2] * 3 : 3]
    return x_t


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


def mrope_sglang(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    mrope_section: List[int],
    head_size: int,
    rotary_dim: int,
    mrope_interleaved: bool,
    mrope_interleaved_glm: bool,
    is_neox_style: bool,
    axis_map: Optional[torch.Tensor],
):
    multomodal_rotary_embedding(
        q,
        k,
        cos_sin_cache,
        positions,
        mrope_section,
        head_size,
        rotary_dim,
        mrope_interleaved,
        mrope_interleaved_glm,
        is_neox_style,
        axis_map,
    )


def calculate_bandwidth(
    num_tokens: int,
    q_heads: int,
    kv_heads: int,
    rotary_dim: int,
    time_ms: float,
    dtype: torch.dtype,
) -> float:
    """
    Calculate the effective memory bandwidth of the MRoPE kernel in GB/s.

    Only the rotary_dim portion of each head is read and written;
    the non-rotary tail is never touched by the kernel.

    Args:
        num_tokens: Number of tokens in the batch.
        q_heads:    Number of query heads.
        kv_heads:   Number of key/value heads (k and v share the same head count).
        rotary_dim: Number of elements per head that are rotated (head_size * partial_rotary_factor).
        time_ms:    Kernel elapsed time in milliseconds (e.g. from triton.testing.do_bench).
        dtype:      Torch dtype of q and k tensors (used to get element size in bytes).

    Returns:
        Bandwidth in GB/s.
    """
    elem_bytes = (
        torch.finfo(dtype).bits // 8
    )  # e.g. 2 for bfloat16/float16, 4 for float32

    # Q and K are read then written in-place → multiply by 2
    bytes_q = num_tokens * q_heads * rotary_dim * elem_bytes * 2
    bytes_k = num_tokens * kv_heads * rotary_dim * elem_bytes * 2

    # cos_sin_cache: one effective row of `rotary_dim` elements read per token
    bytes_cos_sin = 3 * num_tokens * rotary_dim * elem_bytes

    # positions: 3 int64 values per token (t, h, w) — small but included for correctness
    bytes_positions = num_tokens * 3 * torch.iinfo(torch.int64).bits // 8
    # bytes_positions = 0
    total_bytes = bytes_q + bytes_k + bytes_cos_sin + bytes_positions

    bandwidth_gb_s = total_bytes / (time_ms * 1e-3) / 1e9
    return bandwidth_gb_s


all_results = []


def get_benchmark(device="xpu"):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=[
                "num_tokens",
                "num_head_pairs",
                "head_dim",
                "partial_rotary_factor",
                "max_position_embeddings",
            ],
            x_vals=configs,
            line_arg="provider",
            line_vals=["sglang"],
            line_names=["Sglang"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="Latency (us)",
            plot_name="mrope-performance",
            args={},
        )
    )
    def benchmark(
        num_tokens,
        num_head_pairs,
        head_dim,
        partial_rotary_factor,
        max_position_embeddings,
        provider,
    ):
        dtype = torch.bfloat16
        q_heads, kv_heads = num_head_pairs[0], num_head_pairs[1]
        q = torch.randn((num_tokens, q_heads * head_dim), device=device, dtype=dtype)
        k = torch.randn((num_tokens, kv_heads * head_dim), device=device, dtype=dtype)
        rotary_dim = int(head_dim * partial_rotary_factor)
        cos_sin_cache = torch.randn(
            (max_position_embeddings, rotary_dim), dtype=dtype, device=device
        )
        positions = torch.randint(
            0, max_position_embeddings, (3, num_tokens), device=device
        )
        mrope_section = [11, 11, 10]

        if provider == "sglang":
            fn = lambda: mrope_sglang(
                q,
                k,
                cos_sin_cache,
                positions,
                mrope_section,
                head_dim,
                rotary_dim,
                True,
                False,
                True,
                None,
            )

        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
        bandwidth = calculate_bandwidth(
            num_tokens, q_heads, kv_heads, rotary_dim, ms, dtype
        )

        all_results.append(
            {
                "num_tokens": num_tokens,
                "q_heads": q_heads,
                "kv_heads": kv_heads,
                "rotary_dim": rotary_dim,
                "dtype": dtype,
                "provider": provider,
                "bandwidth": bandwidth,
                "ms": ms,
            }
        )

        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


if __name__ == "__main__":
    # Run correctness test on small configs if not using a real model

    bs = [1, 2, 4, 8]
    # bs = [16, 32, 64]
    seq_len = 1024
    num_tokens = [seq_len * b for b in bs]
    # configs from qwen3.5-9b(16-4) and qwen3.5-35b-a3b(4-1 with tp=4)
    num_head_pairs = [[4, 1], [16, 4]]
    head_dim = [256]
    is_neox = [True]
    dtype = [torch.bfloat16]

    sweep_params = {
        "num_tokens": num_tokens,
        "num_head_pairs": num_head_pairs,
        "head_dim": head_dim,
        "partial_rotary_factor": [0.25],
        "max_position_embeddings": [262400],
    }

    keys = sweep_params.keys()
    configs = list(itertools.product(*sweep_params.values()))

    global benchmark_configs
    benchmark_configs = configs

    # Run benchmark
    print("Starting performance benchmark...")
    benchmark = get_benchmark()
    benchmark.run(print_data=False)

    df = pd.DataFrame(all_results)
    print(df.to_markdown())
