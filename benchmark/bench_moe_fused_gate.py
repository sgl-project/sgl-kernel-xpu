from typing import Optional

import torch
import triton
from sgl_kernel import moe_fused_gate

all_results = []


def biased_grouped_topk_native(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: Optional[torch.Tensor],
    topk: int,
    renormalize: bool,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
    scoring_func: str = "sigmoid",
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    if scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    elif scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    else:
        raise ValueError(f"Unknown scoring_func: {scoring_func}")
    num_token = scores.shape[0]
    num_experts = scores.shape[1]
    scores_for_choice = scores.view(num_token, -1)
    if correction_bias is not None:
        scores_for_choice = scores_for_choice + correction_bias.unsqueeze(0)
    group_sum_count = 1 if scoring_func == "softmax" else 2
    group_scores = (
        scores_for_choice.view(num_token, num_expert_group, -1)
        .topk(group_sum_count, dim=-1)[0]
        .sum(dim=-1)
    )  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
        1
    ]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores_for_choice.masked_fill(
        ~score_mask.bool(), float("-inf")
    )  # [n, e]
    # TODO: NPU can't support directly evaluating a comparison for now
    _, topk_ids = torch.topk(
        tmp_scores,
        k=topk,
        dim=-1,
        sorted=(True if num_fused_shared_experts > 0 else False),
    )
    topk_weights = scores.gather(1, topk_ids)

    if num_fused_shared_experts:
        topk_ids[:, -1] = torch.randint(
            low=num_experts,
            high=num_experts + num_fused_shared_experts,
            size=(topk_ids.size(0),),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )
        topk_weights[:, -1] = topk_weights[:, :-1].sum(dim=-1) / routed_scaling_factor

    if renormalize:
        topk_weights_sum = (
            topk_weights.sum(dim=-1, keepdim=True)
            if num_fused_shared_experts == 0
            else topk_weights[:, :-1].sum(dim=-1, keepdim=True)
        )
        topk_weights = topk_weights / topk_weights_sum
        if apply_routed_scaling_factor_on_output:
            topk_weights *= routed_scaling_factor

    topk_weights, topk_ids = topk_weights.to(torch.float32), topk_ids.to(torch.int32)
    return topk_weights, topk_ids


def biased_grouped_topk_org(
    scores,
    bias: Optional[torch.Tensor],
    num_expert_group,
    topk_group,
    topk,
    scoring_func,
):
    return biased_grouped_topk_native(
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
        scoring_func=scoring_func,
    )


def biased_grouped_topk_org_kernel(
    scores,
    bias: Optional[torch.Tensor],
    num_expert_group,
    topk_group,
    topk,
    scoring_func,
):
    return moe_fused_gate(
        input_tensor=scores,
        bias=bias,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        topk=topk,
        renormalize=True,
        scoring_func=scoring_func,
        num_fused_shared_experts=0,
        routed_scaling_factor=2.5,
        apply_routed_scaling_factor_on_output=False,
    )


seq_length_range = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000]
configs = [(sq,) for sq in seq_length_range]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_length"],
        x_vals=[list(_) for _ in configs],
        line_arg="provider",
        line_vals=[
            "original_sigmoid",
            "kernel_sigmoid",
            "original_softmax",
            "kernel_softmax",
        ],
        line_names=[
            "Original (sigmoid)",
            "SGL Kernel (sigmoid)",
            "Original (softmax)",
            "SGL Kernel (softmax)",
        ],
        styles=[
            ("blue", "-"),
            ("red", "-"),
            ("green", "--"),
            ("orange", "--"),
        ],
        ylabel="us",
        plot_name="moe-fused-gate-performance",
        args={},
    )
)
def benchmark(seq_length, provider):
    dtype = torch.bfloat16
    device = torch.device("xpu")
    num_experts, num_expert_group, topk_group, topk = 512, 16, 8, 16

    impl, scoring_func = provider.split("_", 1)

    scores = torch.randn((seq_length, num_experts), device=device, dtype=dtype)
    if scoring_func == "softmax":
        # grouped topk with softmax does not use correction bias
        bias = None
    else:
        bias = torch.rand(num_experts, device=device, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]

    if impl == "original":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: biased_grouped_topk_org(
                scores.clone(),
                None if bias is None else bias.clone(),
                num_expert_group,
                topk_group,
                topk,
                scoring_func,
            ),
            quantiles=quantiles,
        )
    elif impl == "kernel":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: biased_grouped_topk_org_kernel(
                scores.clone(),
                None if bias is None else bias.clone(),
                num_expert_group,
                topk_group,
                topk,
                scoring_func,
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
            "implementation": impl,
            "scoring_func": scoring_func,
        }
    )

    return ms, max_ms, min_ms


if __name__ == "__main__":
    benchmark.run(print_data=True)

    print("\n ✅ moe_fused_gate_performance: ")
    import pandas as pd

    df = pd.DataFrame(all_results)
    print(df.to_markdown())
