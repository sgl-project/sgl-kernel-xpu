import itertools
from typing import Optional

import torch
import torch.nn.functional as F
import triton
from sgl_kernel import topk_sigmoid
from utils import get_model_config, parse_args


def native_topk_sigmoid(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    correction_bias: Optional[torch.Tensor],
):
    if correction_bias is not None:
        n_routed_experts = gating_output.shape[-1]
        scores = F.sigmoid(gating_output)
        scores_for_choice = scores.view(
            -1, n_routed_experts
        ) + correction_bias.unsqueeze(0)
        topk_ids = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_ids)
    else:
        M, _ = gating_output.shape
        topk_weights = torch.empty(
            M,
            topk,
            dtype=torch.float32,
            device=gating_output.device,
        )
        topk_ids = torch.empty(M, topk, dtype=torch.int32, device=gating_output.device)
        topk_weights = F.sigmoid(gating_output.float())
        topk_weights, topk_ids = torch.topk(topk_weights, topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids


def sglang_topk_sigmoid(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    correction_bias: Optional[torch.Tensor],
):
    num_tokens, _ = gating_output.shape

    topk_weights = torch.empty(
        (num_tokens, topk), device=gating_output.device, dtype=torch.float32
    )
    topk_indices = torch.empty(
        (num_tokens, topk), dtype=torch.int32, device=gating_output.device
    )

    topk_sigmoid(
        topk_weights,
        topk_indices,
        gating_output,
        renormalize,
        correction_bias,
    )

    return topk_weights, topk_indices


def get_benchmark(device="xpu"):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=[
                "num_tokens",
                "num_experts",
                "topk",
                "dtype",
                "renormalize",
                "correction_bias",
            ],
            x_vals=configs,
            line_arg="provider",
            line_vals=["sglang", "native"],
            line_names=["SGLang", "native"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="Latency (us)",
            plot_name="topk-sigmoid-performance",
            args={},
        )
    )
    def benchmark(
        num_tokens, num_experts, topk, dtype, renormalize, correction_bias, provider
    ):

        gating_output = torch.randn(
            (num_tokens, num_experts), device=device, dtype=dtype
        )

        if provider == "sglang" or provider == "sglang1":
            fn = lambda: sglang_topk_sigmoid(
                gating_output, topk, renormalize, correction_bias
            )
        elif provider == "native":
            fn = lambda: native_topk_sigmoid(
                gating_output, topk, renormalize, correction_bias
            )

        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


if __name__ == "__main__":
    # Run correctness test on small configs if not using a real model
    args = parse_args()
    params = get_model_config(args)

    sweep_params = {
        "num_tokens": args.num_tokens,
        "num_experts": params["num_experts"] or [64],
        "top_k": params["top_k"] or [2, 4],
        "dtype": [torch.bfloat16],
        "renormalize": [False],
        "correction_bias": [None],
    }

    keys = sweep_params.keys()
    configs = list(itertools.product(*sweep_params.values()))
    print(f"Testing {len(configs)} configurations...")
    for config in configs:
        num_tokens, num_experts, topk, dtype, renormalize, correction_bias = config
        print(
            f"Config: num_tokens={num_tokens}, num_experts={num_experts}, topk={topk}, dtype={dtype}, renormalize={renormalize}, correction_bias={correction_bias}"
        )

    global benchmark_configs
    benchmark_configs = configs

    # Run benchmark
    print("Starting performance benchmark...")
    benchmark = get_benchmark()
    benchmark.run(print_data=True, show_plots=False, save_path=".")
