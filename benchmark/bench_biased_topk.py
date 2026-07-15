import itertools

import pandas as pd
import torch
import triton
from sgl_kernel import biased_topk

all_results = []


def calculate_bandwidth_gbps(
    num_tokens: int,
    num_experts: int,
    topk: int,
    input_dtype: torch.dtype,
    latency_ms: float,
) -> float:
    # Effective bandwidth model: input + bias reads, output + indices writes.
    in_bytes = torch.tensor([], dtype=input_dtype).element_size()
    out_bytes = 4  # float32
    idx_bytes = 4  # int32

    bytes_read = num_tokens * num_experts * in_bytes + num_experts * in_bytes
    bytes_write = num_tokens * topk * out_bytes + num_tokens * topk * idx_bytes
    total_bytes = bytes_read + bytes_write

    return total_bytes / (latency_ms / 1e3) / 1e9


def get_benchmark(device: str = "xpu"):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=[
                "num_tokens",
                "num_experts",
                "topk",
                "dtype",
                "scoring_func",
                "renormalize",
                "num_shared",
                "scale",
                "apply_scale",
            ],
            x_vals=configs,
            line_arg="provider",
            line_vals=["biased_topk"],
            line_names=["biased_topk"],
            styles=[("blue", "-")],
            ylabel="Latency (us)",
            plot_name="biased-topk-performance",
            args={},
        )
    )
    def benchmark(
        num_tokens,
        num_experts,
        topk,
        dtype,
        scoring_func,
        renormalize,
        num_shared,
        scale,
        apply_scale,
        provider,
    ):
        del provider

        input_tensor = torch.randn(
            (num_tokens, num_experts), dtype=dtype, device=device
        )
        bias = torch.randn((num_experts,), dtype=dtype, device=device)
        output = torch.empty((num_tokens, topk), dtype=torch.float32, device=device)
        indices = torch.empty((num_tokens, topk), dtype=torch.int32, device=device)

        def run_op():
            biased_topk(
                input_tensor,
                bias,
                output,
                indices,
                topk,
                scoring_func,
                num_shared,
                renormalize,
                scale,
                apply_scale,
            )

        # Warmup
        for _ in range(10):
            run_op()
        torch.xpu.synchronize()

        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(run_op, quantiles=quantiles)

        bandwidth = calculate_bandwidth_gbps(num_tokens, num_experts, topk, dtype, ms)
        all_results.append(
            {
                "num_tokens": num_tokens,
                "num_experts": num_experts,
                "topk": topk,
                "dtype": str(dtype),
                "scoring_func": scoring_func,
                "renormalize": renormalize,
                "num_shared": num_shared,
                "scale": scale,
                "apply_scale": apply_scale,
                "ms": ms,
                "bandwidth": bandwidth,
            }
        )

        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


if __name__ == "__main__":
    sweep_params = {
        "num_tokens": [1, 32, 256, 1024, 8192],
        "num_experts": [256, 384],
        "topk": [6],
        "dtype": [torch.float32],
        "scoring_func": ["sqrtsoftplus"],
        "renormalize": [True],
        "num_shared": [0],
        "scale": [2.5],
        "apply_scale": [True],
    }

    configs = list(itertools.product(*sweep_params.values()))
    print(f"Running {len(configs)} biased_topk benchmark configs")

    benchmark = get_benchmark(device="xpu")
    benchmark.run(print_data=False, show_plots=False, save_path=".")

    df = pd.DataFrame(all_results)
    print("Detailed results (including bandwidth):")
    print(df.to_markdown(index=False))
