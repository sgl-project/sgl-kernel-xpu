import itertools
from typing import Optional

import pandas as pd
import torch
import triton
from sgl_kernel import biased_topk

all_results = []


@torch.compile(dynamic=True)
def triton_biased_topk_impl(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    scoring_func: str = "sigmoid",
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    if scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    elif scoring_func == "sqrtsoftplus":
        scores = torch.nn.functional.softplus(gating_output).sqrt()

    num_token = scores.shape[0]
    num_experts = scores.shape[1]

    scores_for_choice = scores.view(num_token, -1) + correction_bias.unsqueeze(0)
    _, topk_ids = torch.topk(
        scores_for_choice,
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
        if routed_scaling_factor is not None:
            topk_weights[:, -1] = (
                topk_weights[:, :-1].sum(dim=-1) / routed_scaling_factor
            )

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
            line_vals=["kernel", "triton"],
            line_names=["biased_topk_kernel", "biased_topk_triton"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="Latency (us)",
            plot_name="biased-topk-kernel-vs-triton",
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
        input_tensor = torch.randn(
            (num_tokens, num_experts), dtype=dtype, device=device
        )
        bias = torch.randn((num_experts,), dtype=torch.float32, device=device)
        output = torch.empty((num_tokens, topk), dtype=torch.float32, device=device)
        indices = torch.empty((num_tokens, topk), dtype=torch.int32, device=device)

        if provider == "kernel":

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

        elif provider == "triton":

            hidden_states = torch.empty((num_tokens, 1), dtype=dtype, device=device)

            def run_op():
                triton_biased_topk_impl(
                    hidden_states,
                    input_tensor,
                    bias,
                    topk,
                    renormalize,
                    scoring_func,
                    num_shared,
                    scale,
                    apply_scale,
                )

        else:
            raise ValueError(f"Unknown provider: {provider}")

        for _ in range(10):
            run_op()
        torch.xpu.synchronize()

        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(run_op, quantiles=quantiles)

        all_results.append(
            {
                "provider": provider,
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
            }
        )

        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


if __name__ == "__main__":
    # configs from deepseek-v4
    sweep_params = {
        "num_tokens": [1, 32, 256, 1024, 8192],
        "num_experts": [256, 384],
        "topk": [6],
        "dtype": [torch.float32],
        "scoring_func": ["sqrtsoftplus"],
        "renormalize": [True],
        "num_shared": [1],
        "scale": [2.5],
        "apply_scale": [True],
    }

    configs = list(itertools.product(*sweep_params.values()))
    print(f"Running {len(configs)} biased_topk benchmark configs")

    benchmark = get_benchmark(device="xpu")
    benchmark.run(print_data=False, show_plots=False, save_path=".")

    df = pd.DataFrame(all_results)
    summary_key_cols = [
        "num_tokens",
        "num_experts",
        "topk",
        "dtype",
        "scoring_func",
        "renormalize",
        "num_shared",
        "scale",
        "apply_scale",
    ]
    summary_df = (
        df.pivot_table(
            index=summary_key_cols,
            columns="provider",
            values="ms",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={"kernel": "sglang_kernel_ms", "triton": "triton_ms"})
    )
    summary_df["speedup"] = (
        summary_df["triton_ms"] / summary_df["sglang_kernel_ms"]
    ).map(lambda x: f"{x:.2f}")

    print("Kernel vs Triton implementation latency summary:")
    print(summary_df.to_markdown(index=False))
