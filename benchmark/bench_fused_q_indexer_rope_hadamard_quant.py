"""Benchmark for fused_q_indexer_rope_hadamard_quant SYCL kernel performance."""

import argparse
from itertools import product

import pandas as pd
import torch
import triton
from sgl_kernel import fused_q_indexer_rope_hadamard_quant

batch_size_range = [1, 4, 8, 16]
num_heads_range = [4, 8, 16]
max_pos_range = [256, 512, 1024]

configs = [
    (bs, num_heads, max_pos)
    for bs, num_heads, max_pos in product(
        batch_size_range, num_heads_range, max_pos_range
    )
]

all_results = []


def benchmark_fused_q_indexer_rope_hadamard_quant(
    batch_size, num_heads, max_pos, provider
):
    print(
        f"benchmark fused_q_indexer_rope_hadamard_quant {provider} "
        f"batch_size={batch_size} num_heads={num_heads} max_pos={max_pos}"
    )
    torch.xpu.manual_seed_all(42)

    head_dim = 128
    rope_dim = 64
    weight_scale = 0.5

    q_input = torch.randn(
        batch_size, num_heads, head_dim, dtype=torch.bfloat16, device="xpu"
    )
    weight = torch.randn(batch_size, num_heads, dtype=torch.bfloat16, device="xpu")
    freqs_cis = torch.randn(max_pos, rope_dim // 2, dtype=torch.complex64, device="xpu")
    positions = torch.randint(
        0, max_pos, (batch_size,), dtype=torch.int32, device="xpu"
    )

    q_fp8 = torch.empty(q_input.shape, dtype=torch.float8_e4m3fn, device="xpu")
    weights_out = torch.empty(
        (*q_input.shape[:-1], 1), dtype=torch.float32, device="xpu"
    )

    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)

    def bench_fn():
        fused_q_indexer_rope_hadamard_quant(
            q_input, q_fp8, weight, weights_out, weight_scale, freqs_real, positions
        )

    ms = triton.testing.do_bench(bench_fn, warmup=10, rep=100)
    all_results.append(
        {
            "batch_size": batch_size,
            "num_heads": num_heads,
            "max_pos": max_pos,
            "time_ms": ms,
        }
    )
    return ms


def get_benchmark(configs=configs):
    return triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["batch_size", "num_heads", "max_pos"],
            x_vals=configs,
            line_arg="provider",
            line_vals=["sgl_kernel"],
            line_names=["SGL Kernel (SYCL)"],
            styles=[("blue", "-")],
            ylabel="Time (ms)",
            plot_name="fused-q-indexer-rope-hadamard-quant-performance",
            args={},
        )
    )(benchmark_fused_q_indexer_rope_hadamard_quant)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=batch_size_range,
        help="Batch sizes to sweep",
    )
    parser.add_argument(
        "--num-heads",
        nargs="+",
        type=int,
        default=num_heads_range,
        help="Head counts to sweep",
    )
    parser.add_argument(
        "--max-pos",
        nargs="+",
        type=int,
        default=max_pos_range,
        help="RoPE cache lengths to sweep",
    )
    args = parser.parse_args()

    configs = list(product(args.batch_sizes, args.num_heads, args.max_pos))
    get_benchmark(configs).run(print_data=True)
    df = pd.DataFrame(all_results)
    print("\n" + "=" * 80)
    print("SYCL Kernel Performance Summary:")
    print("=" * 80)
    print(df.to_string(index=False))
