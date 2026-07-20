"""Benchmark JIT MoE fused-gate kernel vs AOT implementation."""

import itertools

import pandas as pd
import torch
import triton

try:
    import sgl_kernel

    HAS_AOT = True
except ImportError:
    HAS_AOT = False
    print("Warning: sgl_kernel not available, AOT comparison will be skipped")

# Storage for bandwidth/performance results
all_results = []

# Fixed gate config (must be a config supported by the JIT kernel; see
# _SUPPORTED_MOE_GATE_CONFIGS in python/sgl_kernel/jit/moe_fused_gate.py).
NUM_EXPERTS = 256
NUM_EXPERT_GROUP = 8
TOPK_GROUP = 4
TOPK = 8
ROUTED_SCALING_FACTOR = 2.5


def calculate_effective_bandwidth(num_tokens, num_experts, dtype, time_ms):
    """Calculate memory bandwidth / throughput metrics for the gate kernel."""
    bytes_per_element = torch.finfo(dtype).bits // 8
    # Dominant traffic: read the [num_tokens, num_experts] gate logits.
    total_bytes = 2 * num_tokens * num_experts * bytes_per_element
    bandwidth_gbs = (total_bytes / 1e9) / (time_ms / 1000.0)

    # add bias + negate + exp + add + reciprocal + compare and shuffle ~50 ops
    total_flops = num_tokens * 50 * num_experts
    gflops = (total_flops / 1e9) / (time_ms / 1000.0)

    return {
        "total_bytes": total_bytes,
        "bandwidth_gbs": bandwidth_gbs,
        "total_flops": total_flops,
        "gflops": gflops,
    }


# Test configurations: number of tokens (rows).
configs = list(
    itertools.product(
        [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000],  # num_tokens
    )
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["aot", "jit"],
        line_names=["AOT (sgl_kernel)", "JIT (sglang)"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="aot-vs-jit-moe-fused-gate-performance",
        args={},
    )
)
def benchmark(num_tokens, provider):
    device = torch.device("xpu")
    dtype = torch.bfloat16

    scores = torch.randn(num_tokens, NUM_EXPERTS, device=device, dtype=dtype)
    bias = torch.rand(NUM_EXPERTS, device=device, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "aot":
        if not HAS_AOT:
            print("Warning: sgl_kernel AOT not available, skipping")
            return 0, 0, 0
        fn = lambda: sgl_kernel.moe_fused_gate(
            scores.clone(),
            bias.clone(),
            NUM_EXPERT_GROUP,
            TOPK_GROUP,
            TOPK,
            0,
            ROUTED_SCALING_FACTOR,
            False,
        )
    elif provider == "jit":
        # Import here to allow optional dependency
        try:
            from sgl_kernel.jit import moe_fused_gate as jit_moe_fused_gate

            fn = lambda: jit_moe_fused_gate(
                scores.clone(),
                bias.clone(),
                NUM_EXPERT_GROUP,
                TOPK_GROUP,
                TOPK,
                num_fused_shared_experts=0,
                routed_scaling_factor=ROUTED_SCALING_FACTOR,
                apply_routed_scaling_factor_on_output=False,
            )
        except ImportError:
            print("Warning: sgl_kernel JIT module not available, skipping")
            return 0, 0, 0

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    # Calculate metrics
    bw_metrics = calculate_effective_bandwidth(num_tokens, NUM_EXPERTS, dtype, ms)

    all_results.append(
        {
            "num_tokens": num_tokens,
            "num_experts": NUM_EXPERTS,
            "topk": TOPK,
            "provider": provider,
            "time_us": 1000 * ms,
            "bandwidth_gbs": bw_metrics["bandwidth_gbs"],
            "total_bytes_mb": bw_metrics["total_bytes"] / 1e6,
            "total_flops_m": bw_metrics["total_flops"] / 1e6,
            "gflops": bw_metrics["gflops"],
        }
    )

    return 1000 * ms, 1000 * min_ms, 1000 * max_ms


if __name__ == "__main__":
    if not HAS_AOT:
        print("ERROR: sgl_kernel (AOT) not available. Please install sgl-kernel-xpu.")
        exit(1)

    print("Running AOT vs JIT MoE fused-gate benchmarks...")
    print("AOT: sgl_kernel.moe_fused_gate (compiled SYCL kernels)")
    print("JIT: sgl_kernel.jit.moe_fused_gate (runtime JIT compilation)")
    print(
        f"config: num_experts={NUM_EXPERTS}, num_expert_group={NUM_EXPERT_GROUP}, "
        f"topk_group={TOPK_GROUP}, topk={TOPK}"
    )
    print("\n" + "=" * 80 + "\n")

    benchmark.run(print_data=True)

    print("Benchmark finished!")

    df = pd.DataFrame(all_results)
    print(df.to_markdown(index=False))
