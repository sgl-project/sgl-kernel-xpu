"""Benchmark the SYCL DeepSeek-V4 MXFP4 paged-MQA logits kernel.

Run inside the XPU container:
  ZE_AFFINITY_MASK=4,5,6,7 python benchmark/bench_fp4_paged_mqa_logits.py
"""

from itertools import product

import torch
import triton
from sgl_kernel import fp4_paged_mqa_logits

PACKED_DIM = 64
GROUPS = 4
PAGE_SIZE = 64
CONFIGS = list(product([1, 2, 4], [8], [256, 512]))


def _make_fp4_cache(
    batch: int, seq_len: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    pages_per_batch = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    num_pages = batch * pages_per_batch
    cache = torch.randint(
        0,
        128,
        (num_pages, PAGE_SIZE, 1, PACKED_DIM + GROUPS),
        device=device,
        dtype=torch.uint8,
    )
    cache[..., PACKED_DIM:] = torch.randint(
        120,
        135,
        cache[..., PACKED_DIM:].shape,
        device=device,
        dtype=torch.uint8,
    )
    page_table = torch.arange(num_pages, device=device, dtype=torch.int32).view(
        batch, pages_per_batch
    )
    return cache, page_table


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch", "heads", "seq_len"],
        x_vals=CONFIGS,
        line_arg="provider",
        line_vals=["sycl"],
        line_names=["SYCL FP4 paged MQA"],
        styles=[("green", "-")],
        ylabel="us",
        plot_name="dsv4-fp4-paged-mqa-xpu",
        args={},
    )
)
def benchmark(batch: int, heads: int, seq_len: int, provider: str):
    del provider
    device = torch.device("xpu")
    torch.manual_seed(batch * 1000 + heads * 10 + seq_len)
    q_fp4 = torch.randint(
        0, 256, (batch, heads, PACKED_DIM), device=device, dtype=torch.uint8
    )
    q_scale = torch.randint(
        120, 135, (batch, heads, GROUPS), device=device, dtype=torch.uint8
    )
    kv_cache, page_table = _make_fp4_cache(batch, seq_len, device)
    weights = torch.randn(batch, heads, device=device, dtype=torch.float32)
    seq_lens = torch.full((batch,), seq_len, device=device, dtype=torch.int32)
    max_seq_len = ((seq_len + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE

    def fn():
        fp4_paged_mqa_logits(
            (q_fp4, q_scale),
            kv_cache,
            weights,
            seq_lens,
            page_table,
            None,
            max_seq_len,
        )

    return triton.testing.do_bench(fn, warmup=10, rep=100) * 1000.0


if __name__ == "__main__":
    benchmark.run(print_data=True)
