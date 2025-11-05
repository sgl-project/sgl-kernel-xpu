import itertools

import torch
import triton
from sgl_kernel import topk_softmax
from utils import get_model_config, parse_args


def vllm_topk_softmax(gating_output, topk):
    num_tokens, num_experts = gating_output.shape

    topk_weights = torch.empty(
        (num_tokens, topk), device=gating_output.device, dtype=torch.float32
    )
    topk_indices = torch.empty(
        (num_tokens, topk), dtype=torch.int32, device=gating_output.device
    )
    token_expert_indices = torch.empty(
        (num_tokens, topk), dtype=torch.int32, device=gating_output.device
    )
    torch.ops._moe_C.topk_softmax(
        topk_weights, topk_indices, token_expert_indices, gating_output
    )
    return topk_weights, topk_indices


def navtive_topk_softmax(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    num_tokens, num_experts = gating_output.shape

    import torch.nn.functional as F

    topk_weights = torch.empty(
        (num_tokens, topk), device=gating_output.device, dtype=torch.float32
    )
    topk_indices = torch.empty(
        (num_tokens, topk), dtype=torch.int32, device=gating_output.device
    )
    topk_weights = F.softmax(gating_output.float(), dim=-1)
    topk_weights, topk_indices = torch.topk(topk_weights, topk, dim=-1)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_indices


def sglang_topk_softmax(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    num_tokens, num_experts = gating_output.shape

    topk_weights = torch.empty(
        (num_tokens, topk), device=gating_output.device, dtype=torch.float32
    )
    topk_indices = torch.empty(
        (num_tokens, topk), dtype=torch.int32, device=gating_output.device
    )
    token_expert_indices = torch.empty(
        (num_tokens, topk), dtype=torch.int32, device=gating_output.device
    )

    topk_softmax(
        topk_weights,
        topk_indices,
        gating_output,
        renormalize=renormalize,
    )

    return topk_weights, topk_indices


def calculate_diff(num_tokens, num_experts, topk):
    gating_output = torch.randn(
        (num_tokens, num_experts), device=gating_output.device, dtype=torch.float32
    )
    weights_vllm, indices_vllm = vllm_topk_softmax(gating_output.clone(), topk)
    weights_sglang, indices_sglang = sglang_topk_softmax(gating_output.clone(), topk)

    weights_diff = torch.abs(weights_vllm - weights_sglang).mean().item()
    indices_match = torch.equal(indices_vllm, indices_sglang)

    if (
        torch.allclose(weights_vllm, weights_sglang, atol=1e-3, rtol=1e-3)
        and indices_match
    ):
        print("✅ VLLM and SGLang topk_softmax implementations match")
    else:
        print(
            f"❌ Implementations differ: Weights diff={weights_diff}, Indices match={indices_match}"
        )


def get_benchmark(device="xpu"):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["num_tokens", "num_experts", "topk", "dtype", "renormalize"],
            x_vals=configs,
            line_arg="provider",
            line_vals=["sglang", "native"],
            line_names=["SGLang", "native"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="Latency (us)",
            plot_name="topk-softmax-performance",
            args={},
        )
    )
    def benchmark(num_tokens, num_experts, topk, dtype, renormalize, provider):

        gating_output = torch.randn(
            (num_tokens, num_experts), device=device, dtype=dtype
        )

        if provider == "sglang" or provider == "sglang1":
            fn = lambda: sglang_topk_softmax(gating_output, topk, renormalize)
        elif provider == "native":
            fn = lambda: navtive_topk_softmax(gating_output, topk, renormalize)

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
    }

    keys = sweep_params.keys()
    configs = list(itertools.product(*sweep_params.values()))
    print(f"Testing {len(configs)} configurations...")
    for config in configs:
        num_tokens, num_experts, topk, dtype, renormalize = config
        print(
            f"Config: num_tokens={num_tokens}, num_experts={num_experts}, topk={topk}, dtype={dtype}, renormalize={renormalize}"
        )

        # calculate_diff(num_tokens, num_experts, topk)

    global benchmark_configs
    benchmark_configs = configs

    # Run benchmark
    print("Starting performance benchmark...")
    benchmark = get_benchmark()
    benchmark.run(print_data=True, show_plots=False, save_path=".")
