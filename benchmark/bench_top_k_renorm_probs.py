import argparse
from itertools import product

import pandas as pd
import torch
import triton
from sgl_kernel import top_k_renorm_prob

batch_size_range = [1, 99, 989]
vocab_size_range = [1024, 32000, 128256]
top_k_range = [10, 100, 500]


def get_configs(batch_size_range, vocab_size_range, top_k_range):
    return [
        (bs, vocab_size, top_k)
        for bs, vocab_size in product(batch_size_range, vocab_size_range)
        for top_k in top_k_range
        if top_k < vocab_size
    ]


configs = get_configs(batch_size_range, vocab_size_range, top_k_range)

all_results = []


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "vocab_size", "top_k"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["sgl_kernel"],
        line_names=["SGL Kernel"],
        styles=[("blue", "-")],
        ylabel="Time (ms)",
        plot_name="top-k-renorm-probs-performance",
        args={},
    )
)
def benchmark(batch_size, vocab_size, top_k, provider):
    print(
        f"benchmark {provider} with batch_size={batch_size} vocab_size={vocab_size} top_k={top_k}"
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
        _ = top_k_renorm_prob(probs, top_k)
    torch.xpu.synchronize()

    bench_lambda = lambda: top_k_renorm_prob(probs, top_k)

    quantiles = [0.5, 0.25, 0.75]
    ms, _, _ = triton.testing.do_bench(
        bench_lambda, quantiles=quantiles, return_mode="median"
    )

    torch.xpu.empty_cache()

    # Calculate memory bandwidth
    num_elements = batch_size * vocab_size

    total_bytes = 2 * (probs.numel() * probs.element_size())  # read and write
    bandwidth_gb_s = total_bytes / (ms / 1e3) / 1e9

    del probs

    all_results.append(
        {
            "batch_size": batch_size,
            "vocab_size": vocab_size,
            "top_k": top_k,
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
        help="List of batch sizes",
    )
    parser.add_argument(
        "--vocab-sizes",
        nargs="+",
        type=int,
        default=vocab_size_range,
        help="List of vocab sizes",
    )
    parser.add_argument(
        "--top-ks", nargs="+", type=int, default=top_k_range, help="List of top_k"
    )
    args = parser.parse_args()
    benchmark.benchmarks.x_vals = get_configs(
        args.batch_sizes, args.vocab_sizes, args.top_ks
    )

    benchmark.run(print_data=False)
    print("Benchmark finished!")

    df = pd.DataFrame(all_results)
    print(df.to_markdown())
