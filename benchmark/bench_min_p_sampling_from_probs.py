from itertools import product

import pandas as pd
import torch
import triton
from sgl_kernel import min_p_sampling_from_probs

batch_size_range = [989, 99, 1]
vocab_size_range = [128256, 32000, 1024]
min_p_range = [0.05, 0.1, 0.5]
dtype_range = [torch.float32]

configs = list(product(batch_size_range, vocab_size_range, min_p_range, dtype_range))

all_results = []


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "vocab_size", "min_p", "dtype"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["sgl_kernel"],
        line_names=["SGL Kernel"],
        styles=[("blue", "-")],
        ylabel="Time (ms)",
        plot_name="min-p-sampling-from-probs-performance",
        args={},
    )
)
def benchmark(batch_size, vocab_size, min_p, dtype, provider):
    print(
        f"benchmark {provider} with batch_size={batch_size} vocab_size={vocab_size} min_p={min_p} dtype={dtype}"
    )
    torch.set_default_device("xpu")
    torch.xpu.manual_seed_all(42)

    # Create input probabilities
    probs = torch.rand(
        batch_size, vocab_size, dtype=dtype, device="xpu", requires_grad=False
    )
    # Normalize to get valid probabilities
    probs = probs / probs.sum(dim=-1, keepdim=True)

    # Warmup
    output = None
    for _ in range(10):
        output = min_p_sampling_from_probs(probs, min_p)
    torch.xpu.synchronize()

    bench_lambda = lambda: min_p_sampling_from_probs(probs, min_p)

    quantiles = [0.5, 0.25, 0.75]
    ms, _, _ = triton.testing.do_bench(
        bench_lambda, quantiles=quantiles, return_mode="median"
    )

    torch.xpu.empty_cache()

    total_bytes = (
        probs.numel() * probs.element_size() + output.numel() * output.element_size()
    )  # read probs, write output
    bandwidth_gb_s = total_bytes / (ms / 1e3) / 1e9

    del probs

    all_results.append(
        {
            "batch_size": batch_size,
            "vocab_size": vocab_size,
            "min_p": min_p,
            "dtype": str(dtype),
            "provider": provider,
            "bandwidth_gb_s": bandwidth_gb_s,
            "ms": ms,
        }
    )
    return ms


if __name__ == "__main__":
    benchmark.run(print_data=False)
    print("Benchmark finished!")

    df = pd.DataFrame(all_results)
    print(df.to_markdown())
