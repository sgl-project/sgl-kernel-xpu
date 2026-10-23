"""Benchmark the DSV4.1 XPU indexer query and key packing kernels.

Run inside the XPU container:
  ZE_AFFINITY_MASK=0,1,2,3 python benchmark/bench_indexer_rope_pack.py
"""

from itertools import product

import torch
import triton
from sgl_kernel import index_k_norm_rope_pack_store, index_q_rope_pack_weights

HEAD_DIM = 128
ROPE_DIM = 64
PAGE_SIZE = 64
SLOT_BYTES = 68
Q_CONFIGS = list(product([1, 4, 16, 64], [8, 64]))
K_CONFIGS = list(product([1, 4, 16, 64], [1, 2, 4]))


def _make_rope_cache(max_pos: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(max_pos, device=device, dtype=torch.float32)
    frequency = torch.arange(ROPE_DIM // 2, device=device, dtype=torch.float32)
    angle = position[:, None] / (10000.0 ** (frequency[None, :] / (ROPE_DIM // 2)))
    complex_freq = torch.complex(torch.cos(angle), torch.sin(angle))
    return torch.view_as_real(complex_freq).flatten(-2)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["tokens", "heads"],
        x_vals=Q_CONFIGS,
        line_arg="provider",
        line_vals=["sycl"],
        line_names=["SYCL indexer Q pack"],
        styles=[("blue", "-")],
        ylabel="us",
        plot_name="dsv41-indexer-q-rope-pack-xpu",
        args={},
    )
)
def benchmark_q(tokens: int, heads: int, provider: str) -> float:
    del provider
    device = torch.device("xpu")
    torch.manual_seed(tokens * 100 + heads)
    q = torch.randn(tokens, heads, HEAD_DIM, device=device, dtype=torch.bfloat16)
    head_weights = torch.randn(tokens, heads, device=device, dtype=torch.bfloat16)
    positions = torch.randint(0, 512, (tokens,), device=device, dtype=torch.int32)
    freqs_cis = _make_rope_cache(512, device)
    weight_scale = HEAD_DIM**-0.5 * heads**-0.5

    def fn():
        index_q_rope_pack_weights(q, freqs_cis, positions, head_weights, weight_scale)

    return triton.testing.do_bench(fn, warmup=10, rep=100) * 1000.0


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["tokens", "ratio"],
        x_vals=K_CONFIGS,
        line_arg="provider",
        line_vals=["sycl"],
        line_names=["SYCL indexer K store"],
        styles=[("green", "-")],
        ylabel="us",
        plot_name="dsv41-indexer-k-norm-rope-pack-xpu",
        args={},
    )
)
def benchmark_k(tokens: int, ratio: int, provider: str) -> float:
    del provider
    device = torch.device("xpu")
    torch.manual_seed(tokens * 100 + ratio)
    input = torch.randn(tokens, HEAD_DIM, device=device, dtype=torch.bfloat16)
    norm_weight = torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16)
    positions = torch.randint(0, 512, (tokens,), device=device, dtype=torch.int64)
    loc = torch.arange(1, tokens + 1, device=device, dtype=torch.int64)
    freqs_cis = _make_rope_cache(512, device)
    num_pages = (tokens + PAGE_SIZE) // PAGE_SIZE
    cache = torch.empty(
        (num_pages, PAGE_SIZE * SLOT_BYTES), device=device, dtype=torch.uint8
    )
    eps = 1.0e-6

    def fn():
        index_k_norm_rope_pack_store(
            input,
            norm_weight,
            eps,
            freqs_cis,
            positions,
            loc,
            cache,
            ratio=ratio,
        )

    return triton.testing.do_bench(fn, warmup=10, rep=100) * 1000.0


if __name__ == "__main__":
    benchmark_q.run(print_data=True)
    benchmark_k.run(print_data=True)
