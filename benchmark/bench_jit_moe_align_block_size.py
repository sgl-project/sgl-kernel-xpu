"""Benchmark JIT moe_align_block_size vs AOT implementation."""

import itertools

import pandas as pd
import torch
import triton

try:
    import sgl_kernel  # noqa: F401
    from sgl_kernel import moe_align_block_size as aot_moe_align_block_size

    HAS_AOT = True
except ImportError:
    HAS_AOT = False
    print("Warning: sgl_kernel not available, AOT comparison will be skipped")

all_results = []


def _make_inputs(num_tokens, num_experts, topk, block_size, device):
    topk_ids = torch.randint(
        0, num_experts, (num_tokens, topk), dtype=torch.int32, device=device
    )
    max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
    sorted_ids = torch.empty(max_num_tokens_padded, dtype=torch.int32, device=device)
    sorted_ids.fill_(topk_ids.numel())
    expert_ids = torch.zeros(
        max_num_tokens_padded // block_size, dtype=torch.int32, device=device
    )
    num_tokens_post_pad = torch.empty(1, dtype=torch.int32, device=device)
    cumsum = torch.zeros(num_experts + 2, dtype=torch.int32, device=device)
    return topk_ids, sorted_ids, expert_ids, num_tokens_post_pad, cumsum


def run_aot(topk_ids, num_experts, block_size, sti, eids, ntpp, cumsum):
    aot_moe_align_block_size(
        topk_ids, num_experts + 1, block_size, sti, eids, ntpp, cumsum, False
    )


def run_jit(topk_ids, num_experts, block_size, sti, eids, ntpp, cumsum):
    from sgl_kernel.jit import moe_align_block_size as _jit

    _jit(topk_ids, num_experts + 1, block_size, sti, eids, ntpp, cumsum, False)


# (num_tokens, num_experts, topk). Includes small-batch shapes (numel < 1024,
# experts <= 64) and general-path shapes.
configs = list(
    itertools.product(
        [8, 64, 512, 2048, 8192],  # num_tokens
        [64, 128, 256],  # num_experts
        [8],  # topk
    )
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens", "num_experts", "topk"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["aot", "jit"],
        line_names=["AOT (sgl_kernel)", "JIT (sglang)"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="aot-vs-jit-moe-align-block-size-performance",
        args={},
    )
)
def benchmark(num_tokens, num_experts, topk, provider):
    device = torch.device("xpu")
    block_size = 128
    topk_ids, sti, eids, ntpp, cumsum = _make_inputs(
        num_tokens, num_experts, topk, block_size, device
    )
    quantiles = [0.5, 0.2, 0.8]

    if provider == "aot":
        if not HAS_AOT:
            print("Warning: sgl_kernel AOT not available, skipping")
            return 0, 0, 0
        fn = lambda: run_aot(topk_ids, num_experts, block_size, sti, eids, ntpp, cumsum)
    elif provider == "jit":
        try:
            fn = lambda: run_jit(
                topk_ids, num_experts, block_size, sti, eids, ntpp, cumsum
            )
        except ImportError:
            print("Warning: sgl_kernel JIT module not available, skipping")
            return 0, 0, 0

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    all_results.append(
        {
            "num_tokens": num_tokens,
            "num_experts": num_experts,
            "topk": topk,
            "provider": provider,
            "time_us": 1000 * ms,
        }
    )

    return 1000 * ms, 1000 * min_ms, 1000 * max_ms


if __name__ == "__main__":
    if not HAS_AOT:
        print("ERROR: sgl_kernel (AOT) not available. Please install sgl-kernel-xpu.")
        exit(1)

    print("Running AOT vs JIT moe_align_block_size benchmarks...")
    print("AOT: sgl_kernel.moe_align_block_size (compiled SYCL kernels)")
    print("JIT: sgl_kernel.jit.moe_align_block_size (runtime JIT compilation)")
    print("\n" + "=" * 80 + "\n")

    benchmark.run(print_data=True)

    print("Benchmark finished!")
    df = pd.DataFrame(all_results)
    df["time_us"] = df["time_us"].round(2)
    print(df.to_markdown(index=False))
