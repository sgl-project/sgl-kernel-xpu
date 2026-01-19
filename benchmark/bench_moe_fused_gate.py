import torch
import triton
from sgl_kernel import moe_fused_gate

from sglang.srt.layers.moe.topk import biased_grouped_topk

all_results = []


def biased_grouped_topk_org(scores, bias, num_expert_group, topk_group, topk):
    return biased_grouped_topk(
        scores,
        scores,
        bias,
        topk=topk,
        renormalize=True,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        num_fused_shared_experts=0,
        routed_scaling_factor=2.5,
        apply_routed_scaling_factor_on_output=False,
    )


def biased_grouped_topk_org_kernel(scores, bias, num_expert_group, topk_group, topk):
    return moe_fused_gate(
        scores, bias, num_expert_group, topk_group, topk, 0, 2.5, False
    )


seq_length_range = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000]
configs = [(sq,) for sq in seq_length_range]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_length"],
        x_vals=[list(_) for _ in configs],
        line_arg="provider",
        line_vals=["original", "kernel"],
        line_names=["Original", "SGL Kernel"],
        styles=[("blue", "-"), ("red", "-")],
        ylabel="us",
        plot_name="moe-fused-gate-performance",
        args={},
    )
)
def benchmark(seq_length, provider):
    dtype = torch.bfloat16
    device = torch.device("xpu")
    num_experts, num_expert_group, topk_group, topk = 512, 16, 8, 16

    scores = torch.randn((seq_length, num_experts), device=device, dtype=dtype)
    bias = torch.rand(num_experts, device=device, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "original":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: biased_grouped_topk_org(
                scores.clone(), bias.clone(), num_expert_group, topk_group, topk
            ),
            quantiles=quantiles,
        )
    elif provider == "kernel":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: biased_grouped_topk_org_kernel(
                scores.clone(), bias.clone(), num_expert_group, topk_group, topk
            ),
            quantiles=quantiles,
        )
    # add bias + negate + exp + add + reciprocal + compare and shuffle ~50Ops
    flop = seq_length * 50 * num_experts
    memory = 2 * seq_length * num_experts * torch.finfo(dtype).bits // 8
    gflops = flop / (ms / 1e3) / 1e9
    bandwidth = memory / (ms / 1e3) / 1e9
    all_results.append(
        {
            "num_tokens": seq_length,
            "topk": topk,
            "gflops": gflops,
            "bandwidth": bandwidth,
            "ms": ms,
        }
    )

    return ms, max_ms, min_ms


if __name__ == "__main__":
    benchmark.run(print_data=True)

    print("\n ✅ moe_fused_gate_performance: ")
    import pandas as pd

    df = pd.DataFrame(all_results)
    print(df.to_markdown())
