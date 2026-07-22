from itertools import product

import pandas as pd
import torch
import triton
from sgl_kernel import top_k_top_p_sampling_from_probs

batch_size_range = [1, 99, 989]
vocab_size_range = [1024, 32000, 128256]
top_p_range = [0.1, 0.5]

configs = [
    (bs, vocab_size, top_p)
    for bs, vocab_size in product(batch_size_range, vocab_size_range)
    for top_p in top_p_range
]

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
        plot_name="top-k-top-p-joint-sampling-from-probs-performance",
        args={},
    )
)
def benchmark(batch_size, vocab_size, top_p, provider):
    # top_k mirrors the test: 50% of vocab for p=0.1, 10% of vocab for p=0.5.
    top_k = int(vocab_size * 0.5) if top_p == 0.1 else int(vocab_size * 0.1)
    print(
        f"benchmark {provider} with batch_size={batch_size} vocab_size={vocab_size} "
        f"top_k={top_k} top_p={top_p}"
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

    top_k_tensor = torch.full((batch_size,), top_k, device="xpu")
    top_p_tensor = torch.full((batch_size,), top_p, device="xpu")

    sample = lambda: top_k_top_p_sampling_from_probs(
        probs,
        top_k_tensor,
        top_p_tensor,
        filter_apply_order="joint",
    )

    # Warmup
    for _ in range(10):
        _ = sample()
    torch.xpu.synchronize()

    quantiles = [0.5, 0.25, 0.75]
    ms, _, _ = triton.testing.do_bench(
        sample, quantiles=quantiles, return_mode="median"
    )

    torch.xpu.empty_cache()

    # Rough memory-traffic estimate: probs is streamed multiple times per sample.
    total_bytes = 2 * (probs.numel() * probs.element_size())  # read and write
    bandwidth_gb_s = total_bytes / (ms / 1e3) / 1e9

    del probs

    all_results.append(
        {
            "batch_size": batch_size,
            "vocab_size": vocab_size,
            "top_k": top_k,
            "top_p": top_p,
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
