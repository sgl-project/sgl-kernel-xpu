import torch
import triton
from sgl_kernel import moe_sum_reduce as moe_sum_reduce_sycl
from triton.testing import do_bench

all_results = []


def compute_sum_scaled_baseline(
    x: torch.Tensor, out: torch.Tensor, routed_scaling_factor: float
) -> torch.Tensor:
    torch.sum(x, dim=1, out=out)
    out.mul_(routed_scaling_factor)
    return out


@torch.compile
def compute_sum_scaled_compiled(
    x: torch.Tensor, out: torch.Tensor, routed_scaling_factor: float
) -> torch.Tensor:
    torch.sum(x * routed_scaling_factor, dim=1, out=out)
    return out


def get_benchmark():
    num_tokens_range = [2**i for i in range(0, 13)]

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["num_tokens"],
            x_vals=num_tokens_range,
            line_arg="version",
            line_vals=["baseline", "compiled", "xpu"],
            line_names=["Original", "TorchCompile", "XPUKernel"],
            styles=[("blue", "-"), ("green", "-"), ("red", "-")],
            ylabel="us",
            plot_name="sum_scaled_performance",
            args={},
        )
    )
    def benchmark(num_tokens, version):
        topk = 9
        hidden_size = 4096
        dtype = torch.bfloat16
        scaling_factor = 0.3

        x = torch.randn(num_tokens, topk, hidden_size, dtype=dtype, device="xpu")
        out = torch.empty(num_tokens, hidden_size, dtype=dtype, device="xpu")

        # Warmup
        for _ in range(3):
            if version == "baseline":
                compute_sum_scaled_baseline(x, out, scaling_factor)
            elif version == "compiled":
                compute_sum_scaled_compiled(x, out, scaling_factor)
            else:
                moe_sum_reduce_sycl(x, out, scaling_factor)

        # Benchmark
        quantiles = [0.5, 0.2, 0.8]
        if version == "baseline":
            ms, min_ms, max_ms = do_bench(
                lambda: compute_sum_scaled_baseline(x, out, scaling_factor),
                quantiles=quantiles,
            )
        elif version == "compiled":
            ms, min_ms, max_ms = do_bench(
                lambda: compute_sum_scaled_compiled(x, out, scaling_factor),
                quantiles=quantiles,
            )
        else:
            ms, min_ms, max_ms = do_bench(
                lambda: moe_sum_reduce_sycl(x, out, scaling_factor),
                quantiles=quantiles,
            )

        flop = num_tokens * topk * hidden_size
        memory = (topk + 1) * num_tokens * hidden_size * torch.finfo(dtype).bits // 8
        tflops = flop / (ms / 1e3) / 1e12
        bandwidth = memory / (ms / 1e3) / 1e9
        all_results.append(
            {
                "num_tokens": num_tokens,
                "topk": topk,
                "hidden_size": hidden_size,
                "dtype": dtype,
                "tflops": tflops,
                "bandwidth": bandwidth,
                "us": 1000 * ms,
            }
        )

        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


def verify_correctness(num_tokens=1024):
    x = torch.randn(num_tokens, 9, 4096, device="xpu", dtype=torch.bfloat16)
    scaling_factor = 0.3

    out_baseline = torch.empty_like(x[:, 0])
    compute_sum_scaled_baseline(x, out_baseline, scaling_factor)

    out_compiled = torch.empty_like(out_baseline)
    compute_sum_scaled_compiled(x, out_compiled, scaling_factor)

    out_xpu = torch.empty_like(out_baseline)
    moe_sum_reduce_sycl(x, out_xpu, scaling_factor)

    if torch.allclose(
        out_baseline, out_compiled, atol=1e-2, rtol=1e-2
    ) and torch.allclose(out_baseline, out_xpu, atol=1e-2, rtol=1e-2):
        print("✅ All implementations match")
    else:
        print("❌ Implementations differ")
        print(
            f"Baseline vs Compiled: {(out_baseline - out_compiled).abs().max().item()}"
        )
        print(f"Baseline vs xpu: {(out_baseline - out_xpu).abs().max().item()}")


if __name__ == "__main__":
    print("Running correctness verification...")
    verify_correctness()

    print("\nRunning performance benchmark...")
    benchmark = get_benchmark()
    benchmark.run(
        print_data=True,
        # save_path="./configs/benchmark_ops/sum_scaled/"
    )

    print("\n ✅ sum_scaled_performance: ")
    import pandas as pd
    df = pd.DataFrame(all_results)
    print(df.to_markdown())
