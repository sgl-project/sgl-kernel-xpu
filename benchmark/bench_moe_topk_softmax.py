import itertools

import torch
import triton
from sgl_kernel import topk_softmax
from utils import parse_args


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


def sglang_topk_softmax(gating_output, topk):
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
        renormalize=False,
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
            x_names=["num_tokens", "num_experts", "topk", "dtype"],
            x_vals=configs,
            line_arg="provider",
            line_vals=["sglang", "vllm"],
            line_names=["SGLang", "VLLM"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="Latency (us)",
            plot_name="topk-softmax-performance",
            args={},
        )
    )
    def benchmark(num_tokens, num_experts, topk, dtype, provider):

        gating_output = torch.randn(
            (num_tokens, num_experts), device=device, dtype=dtype
        )

        if provider == "vllm" or provider == "vllm1":
            fn = lambda: vllm_topk_softmax(gating_output, topk)
        elif provider == "sglang" or provider == "sglang1":
            fn = lambda: sglang_topk_softmax(gating_output, topk)

        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


if __name__ == "__main__":
    # Run correctness test on small configs if not using a real model
    args = parse_args()
    sweep_params = {
        "num_tokens": [1, 32, 128, 512],
        "num_experts": args.num_experts or [64],
        "top_k": args.top_k or [2, 4],
        "dtype": [torch.float16, torch.bfloat16],
    }
    keys = sweep_params.keys()
    configs = list(itertools.product(*sweep_params.values()))
    print(f"Testing {len(configs)} configurations...")
    for config in configs:
        num_tokens, num_experts, topk, dtype = config
        print(
            f"Config: num_tokens={num_tokens}, num_experts={num_experts}, topk={topk}, dtype={dtype}"
        )

        calculate_diff(num_tokens, num_experts, topk)

    global benchmark_configs
    benchmark_configs = configs

    # Run benchmark
    print("Starting performance benchmark...")
    benchmark = get_benchmark()
    benchmark.run(print_data=True, show_plots=False, save_path=".")
