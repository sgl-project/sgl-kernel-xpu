import argparse
from itertools import product

import pandas as pd
import torch
import triton
from sgl_kernel import top_p_renorm_prob

batch_size_range = [1, 99, 989]
vocab_size_range = [1024, 32000, 128256]
top_p_range = [0.1, 0.5, 0.9]


def get_configs(batch_sizes, vocab_sizes, top_ps):
    return [
        (bs, vocab_size, top_p)
        for bs, vocab_size in product(batch_sizes, vocab_sizes)
        for top_p in top_ps
    ]


configs = get_configs(batch_size_range, vocab_size_range, top_p_range)

all_results = []


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "vocab_size", "top_p"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["sgl_kernel"],
        line_names=["SGL Kernel"],
        styles=[("blue", "-")],
        ylabel="Time (ms)",
        plot_name="top-p-renorm-probs-performance",
        args={},
    )
)
def benchmark(batch_size, vocab_size, top_p, provider):
    print(
        f"benchmark {provider} with batch_size={batch_size} vocab_size={vocab_size} top_p={top_p}"
    )
    dtype = torch.float32
    torch.set_default_device("xpu")
    torch.xpu.manual_seed_all(42)

    # Create input probabilities
    probs = torch.rand(
        batch_size, vocab_size, dtype=dtype, device="xpu", requires_grad=False
    )
    # Normalize to get valid probabilities
    probs = probs / probs.sum(dim=-1, keepdim=True)

    # Warmup
    for _ in range(10):
        _ = top_p_renorm_prob(probs, top_p)
    torch.xpu.synchronize()

    bench_lambda = lambda: top_p_renorm_prob(probs, top_p)

    quantiles = [0.5, 0.25, 0.75]
    ms, _, _ = triton.testing.do_bench(
        bench_lambda, quantiles=quantiles, return_mode="median"
    )

    torch.xpu.empty_cache()

    # Calculate memory bandwidth
    total_bytes = 2 * (probs.numel() * probs.element_size())  # read and write
    bandwidth_gb_s = total_bytes / (ms / 1e3) / 1e9

    del probs

    all_results.append(
        {
            "batch_size": batch_size,
            "vocab_size": vocab_size,
            "top_p": top_p,
            "provider": provider,
            "bandwidth_gb_s": bandwidth_gb_s,
            "ms": ms,
        }
    )
    return ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=batch_size_range,
        help="Batch sizes",
    )
    parser.add_argument(
        "--vocab-sizes",
        nargs="+",
        type=int,
        default=vocab_size_range,
        help="Vocab sizes",
    )
    parser.add_argument(
        "--top-ps", nargs="+", type=float, default=top_p_range, help="top_p values"
    )
    args = parser.parse_args()
    benchmark.benchmarks.x_vals = get_configs(
        args.batch_sizes, args.vocab_sizes, args.top_ps
    )

    benchmark.run(print_data=False)
    print("Benchmark finished!")

    df = pd.DataFrame(all_results)
    print(df.to_markdown())
