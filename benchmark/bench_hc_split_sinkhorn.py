from itertools import product

import pandas as pd
import torch
import triton
from sgl_kernel import hc_split_sinkhorn

batch_size_range = [1, 7, 32, 64, 384, 512]
seq_len_range = [1, 512, 4096, 131072]
sinkhorn_iters_range = [20]
hc = 4
col_size = (2 + hc) * hc  # 24 floats per token

MAX_TOKENS = 1_000_000

configs = [
    (b, s, it)
    for b, s, it in product(batch_size_range, seq_len_range, sinkhorn_iters_range)
    if b * s <= MAX_TOKENS
]

all_results = []


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "seq_len", "sinkhorn_iters"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["sgl_kernel"],
        line_names=["SGL Kernel"],
        styles=[("green", "-")],
        ylabel="Time (ms)",
        plot_name="hc-split-sinkhorn-performance",
        args={},
    )
)
def benchmark(batch_size, seq_len, sinkhorn_iters, provider):
    print(
        f"benchmark {provider} with batch_size={batch_size} seq_len={seq_len} sinkhorn_iters={sinkhorn_iters}"
    )
    torch.set_default_device("xpu")
    torch.xpu.manual_seed_all(42)

    mixes = torch.randn(
        batch_size, seq_len, col_size, dtype=torch.float32, device="xpu"
    )
    hc_scale = torch.rand(3, dtype=torch.float32, device="xpu") * 0.5 + 0.5
    hc_base = torch.randn(col_size, dtype=torch.float32, device="xpu") * 0.1

    for _ in range(10):
        _ = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc, sinkhorn_iters)
    torch.xpu.synchronize()

    bench_lambda = lambda: hc_split_sinkhorn(
        mixes, hc_scale, hc_base, hc, sinkhorn_iters
    )

    quantiles = [0.5, 0.25, 0.75]
    ms, _, _ = triton.testing.do_bench(
        bench_lambda, quantiles=quantiles, return_mode="median"
    )

    torch.xpu.empty_cache()

    T = batch_size * seq_len
    read_numel = T * col_size
    write_numel = T * (hc + hc + hc * hc)
    total_bytes = (read_numel + write_numel) * 4
    bandwidth_gb_s = total_bytes / (ms / 1e3) / 1e9

    all_results.append(
        {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "T": T,
            "sinkhorn_iters": sinkhorn_iters,
            "ms": ms,
            "Mtok_per_sec": T / (ms / 1e3) / 1e6,
            "bandwidth_gb_s": bandwidth_gb_s,
        }
    )
    return ms


if __name__ == "__main__":
    benchmark.run(print_data=False)
    print("Benchmark finished!")

    df = pd.DataFrame(all_results)
    print(df.to_markdown())
