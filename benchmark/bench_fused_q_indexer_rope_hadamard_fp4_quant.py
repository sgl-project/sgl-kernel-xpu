"""Benchmark the SYCL DeepSeek-V4 MXFP4 Indexer Q quantization kernel.

Run inside the XPU container:
  ZE_AFFINITY_MASK=4,5,6,7 python benchmark/bench_fused_q_indexer_rope_hadamard_fp4_quant.py
"""

from itertools import product

import torch
import triton
from sgl_kernel import fused_q_indexer_rope_hadamard_fp4_quant

HEAD_DIM = 128
PACKED_DIM = HEAD_DIM // 2
GROUPS = HEAD_DIM // 32
CONFIGS = list(product([1, 5, 17], [8]))


def _make_rope_cache(max_pos: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(max_pos, device=device, dtype=torch.float32)
    frequency = torch.arange(32, device=device, dtype=torch.float32)
    angle = position[:, None] / (10000.0 ** (frequency[None, :] / 32.0))
    complex_freq = torch.complex(torch.cos(angle), torch.sin(angle))
    return torch.view_as_real(complex_freq).flatten(-2)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch", "heads"],
        x_vals=CONFIGS,
        line_arg="provider",
        line_vals=["sycl"],
        line_names=["SYCL FP4 Q quant"],
        styles=[("blue", "-")],
        ylabel="us",
        plot_name="dsv4-fp4-indexer-q-quant-xpu",
        args={},
    )
)
def benchmark(batch: int, heads: int, provider: str):
    del provider
    device = torch.device("xpu")
    torch.manual_seed(batch * 100 + heads)
    q_input = torch.randn(batch, heads, HEAD_DIM, device=device, dtype=torch.bfloat16)
    weight = torch.randn(batch, heads, device=device, dtype=torch.bfloat16)
    positions = torch.randint(0, 512, (batch,), device=device, dtype=torch.int32)
    rope_cache = _make_rope_cache(512, device)
    q_fp4 = torch.empty(batch, heads, PACKED_DIM, device=device, dtype=torch.uint8)
    q_scale = torch.empty(batch, heads, GROUPS, device=device, dtype=torch.uint8)
    weights_out = torch.empty(batch, heads, 1, device=device, dtype=torch.float32)

    def fn():
        fused_q_indexer_rope_hadamard_fp4_quant(
            q_input,
            q_fp4,
            q_scale,
            weight,
            weights_out,
            HEAD_DIM**-0.5 * heads**-0.5,
            rope_cache,
            positions,
        )

    return triton.testing.do_bench(fn, warmup=10, rep=100) * 1000.0


if __name__ == "__main__":
    benchmark.run(print_data=True)
