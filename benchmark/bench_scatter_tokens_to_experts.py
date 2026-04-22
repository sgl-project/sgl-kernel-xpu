import torch
import triton
from sgl_kernel import scatter_tokens_to_experts
from triton.testing import do_bench

all_results = []


def scatter_tokens_baseline(input_tensor, src2dst_map, output_tensor, topk):
    """Baseline PyTorch implementation of scatter operation."""
    num_tokens = input_tensor.size(0)
    for token_id in range(num_tokens):
        for k in range(topk):
            dst_row = src2dst_map[token_id * topk + k].item()
            output_tensor[dst_row] = input_tensor[token_id]
    return output_tensor


@torch.compile
def scatter_tokens_compiled(input_tensor, src2dst_map, output_tensor, topk):
    """Torch-compiled version using index_copy_."""
    num_tokens = input_tensor.size(0)
    # Reshape to flatten topk dimension: [num_tokens, topk, hidden_dim]
    input_repeated = (
        input_tensor.unsqueeze(1)
        .expand(-1, topk, -1)
        .reshape(-1, input_tensor.size(-1))
    )
    # Use index_copy for scatter
    output_tensor.index_copy_(0, src2dst_map, input_repeated)
    return output_tensor


def get_benchmark():
    num_tokens_range = [2**i for i in range(4, 13)]  # 16 to 4096 tokens
    topk_vals = [2, 4, 8]
    hidden_dim_vals = [512, 1024, 2048, 4096, 8192]
    dtype_vals = [torch.bfloat16, torch.float16, torch.float8_e4m3fn]

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["num_tokens"],
            x_vals=num_tokens_range,
            line_arg="config",
            line_vals=[
                f"{dtype_str}-{hidden_dim}-{topk}-{version}"
                for dtype_str, dtype in [
                    ("bf16", torch.bfloat16),
                    ("fp16", torch.float16),
                    ("fp8", torch.float8_e4m3fn),
                ]
                for hidden_dim in [4096]  # Focus on common hidden_dim
                for topk in [8]  # Common topk
                for version in ["xpu"]  # Only test XPU kernel (baseline is too slow)
            ],
            line_names=[
                f"{dtype_str}-{hidden_dim}-{topk}-{version}"
                for dtype_str, dtype in [
                    ("bf16", torch.bfloat16),
                    ("fp16", torch.float16),
                    ("fp8", torch.float8_e4m3fn),
                ]
                for hidden_dim in [4096]
                for topk in [8]
                for version in ["xpu"]
            ],
            styles=[("blue", "-"), ("green", "-"), ("red", "-")],
            ylabel="GB/s",
            plot_name="scatter_tokens_bandwidth",
            args={},
        )
    )
    def benchmark(num_tokens, config):
        parts = config.split("-")
        dtype_str = parts[0]
        hidden_dim = int(parts[1])
        topk = int(parts[2])
        version = parts[3]

        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp8": torch.float8_e4m3fn,
        }
        dtype = dtype_map[dtype_str]

        # Create tensors
        if dtype == torch.float8_e4m3fn:
            input_tensor = torch.randn(
                num_tokens, hidden_dim, dtype=torch.float32, device="xpu"
            ).to(dtype)
        else:
            input_tensor = torch.randn(
                num_tokens, hidden_dim, dtype=dtype, device="xpu"
            )

        num_output_tokens = num_tokens * topk
        src2dst_map = torch.arange(num_output_tokens, dtype=torch.int32, device="xpu")
        output_tensor = torch.empty(
            num_output_tokens, hidden_dim, dtype=dtype, device="xpu"
        )

        # Warmup
        for _ in range(3):
            if version == "baseline":
                scatter_tokens_baseline(input_tensor, src2dst_map, output_tensor, topk)
            elif version == "compiled":
                scatter_tokens_compiled(input_tensor, src2dst_map, output_tensor, topk)
            else:  # xpu
                scatter_tokens_to_experts(input_tensor, src2dst_map, output_tensor)

        # Benchmark
        quantiles = [0.5, 0.2, 0.8]
        if version == "baseline":
            ms, min_ms, max_ms = do_bench(
                lambda: scatter_tokens_baseline(
                    input_tensor, src2dst_map, output_tensor, topk
                ),
                quantiles=quantiles,
            )
        elif version == "compiled":
            ms, min_ms, max_ms = do_bench(
                lambda: scatter_tokens_compiled(
                    input_tensor, src2dst_map, output_tensor, topk
                ),
                quantiles=quantiles,
            )
        else:  # xpu
            ms, min_ms, max_ms = do_bench(
                lambda: scatter_tokens_to_experts(
                    input_tensor, src2dst_map, output_tensor
                ),
                quantiles=quantiles,
            )

        # Calculate metrics
        dtype_size = torch.finfo(dtype).bits // 8 if dtype != torch.float8_e4m3fn else 1
        # Memory: read num_tokens * hidden_dim, write num_output_tokens * hidden_dim
        memory_bytes = (num_tokens + num_output_tokens) * hidden_dim * dtype_size
        bandwidth = memory_bytes / (ms / 1e3) / 1e9  # GB/s

        all_results.append(
            {
                "num_tokens": num_tokens,
                "topk": topk,
                "hidden_dim": hidden_dim,
                "dtype": dtype_str,
                "version": version,
                "bandwidth_gb_s": bandwidth,
                "us": 1000 * ms,
            }
        )

        return bandwidth, bandwidth * (max_ms / ms), bandwidth * (min_ms / ms)

    return benchmark


def verify_correctness():
    """Verify correctness for all data types."""
    print("Running correctness verification...")

    num_tokens = 128
    hidden_dim = 1024
    topk = 4
    num_output_tokens = num_tokens * topk

    for dtype in [torch.bfloat16, torch.float16, torch.float8_e4m3fn]:
        # Create input
        if dtype == torch.float8_e4m3fn:
            input_tensor = torch.randn(
                num_tokens, hidden_dim, dtype=torch.float32, device="xpu"
            ).to(dtype)
        else:
            input_tensor = torch.randn(
                num_tokens, hidden_dim, dtype=dtype, device="xpu"
            )

        src2dst_map = torch.arange(num_output_tokens, dtype=torch.int32, device="xpu")

        # XPU kernel output
        output_xpu = torch.empty(
            num_output_tokens, hidden_dim, dtype=dtype, device="xpu"
        )
        scatter_tokens_to_experts(input_tensor, src2dst_map, output_xpu)

        # Reference output (CPU)
        input_cpu = input_tensor.cpu()
        src2dst_map_cpu = src2dst_map.cpu()
        output_ref = torch.zeros(
            num_output_tokens, hidden_dim, dtype=dtype, device="cpu"
        )
        for token_id in range(num_tokens):
            for k in range(topk):
                dst_row = src2dst_map_cpu[token_id * topk + k].item()
                output_ref[dst_row] = input_cpu[token_id]

        # Compare
        if torch.equal(output_xpu.cpu(), output_ref):
            print(f"✅ {dtype} - Bit-exact match")
        else:
            print(f"❌ {dtype} - Mismatch detected")
            print(
                f"   Max diff: {(output_xpu.cpu().to(torch.float32) - output_ref.to(torch.float32)).abs().max().item()}"
            )


def benchmark_all_configs():
    """Benchmark all dtype x hidden_dim x topk combinations."""
    print("\nBenchmarking all configurations...")

    configs = [
        # (num_tokens, hidden_dim, topk, dtype)
        (128, 512, 2, torch.bfloat16),
        (128, 512, 2, torch.float16),
        (128, 512, 2, torch.float8_e4m3fn),
        (128, 1024, 4, torch.bfloat16),
        (128, 1024, 4, torch.float16),
        (128, 1024, 4, torch.float8_e4m3fn),
        (256, 4096, 8, torch.bfloat16),
        (256, 4096, 8, torch.float16),
        (256, 4096, 8, torch.float8_e4m3fn),
        (1024, 8192, 8, torch.bfloat16),
        (1024, 8192, 8, torch.float16),
        (1024, 8192, 8, torch.float8_e4m3fn),
    ]

    results = []
    for num_tokens, hidden_dim, topk, dtype in configs:
        # Create tensors
        if dtype == torch.float8_e4m3fn:
            input_tensor = torch.randn(
                num_tokens, hidden_dim, dtype=torch.float32, device="xpu"
            ).to(dtype)
        else:
            input_tensor = torch.randn(
                num_tokens, hidden_dim, dtype=dtype, device="xpu"
            )

        num_output_tokens = num_tokens * topk
        src2dst_map = torch.arange(num_output_tokens, dtype=torch.int32, device="xpu")
        output_tensor = torch.empty(
            num_output_tokens, hidden_dim, dtype=dtype, device="xpu"
        )

        # Warmup
        for _ in range(10):
            scatter_tokens_to_experts(input_tensor, src2dst_map, output_tensor)

        # Benchmark
        ms, _, _ = do_bench(
            lambda: scatter_tokens_to_experts(input_tensor, src2dst_map, output_tensor),
            quantiles=[0.5, 0.2, 0.8],
        )

        # Calculate bandwidth
        dtype_size = torch.finfo(dtype).bits // 8 if dtype != torch.float8_e4m3fn else 1
        memory_bytes = (num_tokens + num_output_tokens) * hidden_dim * dtype_size
        bandwidth = memory_bytes / (ms / 1e3) / 1e9

        dtype_str = {
            torch.bfloat16: "bf16",
            torch.float16: "fp16",
            torch.float8_e4m3fn: "fp8",
        }[dtype]
        results.append(
            {
                "num_tokens": num_tokens,
                "hidden_dim": hidden_dim,
                "topk": topk,
                "dtype": dtype_str,
                "bandwidth_gb_s": f"{bandwidth:.2f}",
                "latency_us": f"{ms * 1000:.2f}",
            }
        )

    return results


if __name__ == "__main__":
    verify_correctness()

    print("\n" + "=" * 80)
    print("Benchmarking all configurations...")
    print("=" * 80)

    results = benchmark_all_configs()

    import pandas as pd

    df = pd.DataFrame(results)
    print("\n" + df.to_markdown(index=False))

    print("\n" + "=" * 80)
    print("Running Triton benchmark report...")
    print("=" * 80)

    benchmark = get_benchmark()
    benchmark.run(print_data=True)
