# benchmark_topk_softmax.py
# Benchmark script for topk_softmax operator: compares VLLM vs SGLang implementations
# Supports two modes:
#   1. --model-name provided → load config from HF model
#   2. No --model-name → use default hardcoded test configurations

import itertools
import torch
import triton
from utils import parse_args, get_model_config
from sgl_kernel import topk_softmax


def vllm_topk_softmax(gating_output, topk):
    """
    Simulate vLLM's topk_softmax using torch.ops._moe_C (mock if not available).
    Output: topk_weights, topk_indices
    """
    num_tokens, num_experts = gating_output.shape

    topk_weights = torch.empty((num_tokens, topk), device=gating_output.device, dtype=torch.float32)
    topk_indices = torch.empty((num_tokens, topk), dtype=torch.int32, device=gating_output.device)
    token_expert_indices = torch.empty((num_tokens, topk), dtype=torch.int32, device=gating_output.device)

    try:
        torch.ops._moe_C.topk_softmax(topk_weights, topk_indices, token_expert_indices, gating_output)
    except (AttributeError, ImportError):
        # Mock behavior if vLLM ops not available
        scores = torch.softmax(gating_output, dim=-1)
        topk_vals, topk_idx = torch.topk(scores, topk, dim=-1)
        topk_weights.copy_(topk_vals)
        topk_indices.copy_(topk_idx)

    return topk_weights, topk_indices


def sglang_topk_softmax(gating_output, topk):
    """
    Call SGLang's custom topk_softmax kernel.
    Output: topk_weights, topk_indices
    """
    num_tokens, num_experts = gating_output.shape

    topk_weights = torch.empty((num_tokens, topk), device=gating_output.device, dtype=torch.float32)
    topk_indices = torch.empty((num_tokens, topk), dtype=torch.int32, device=gating_output.device)

    # Call the actual SGLang kernel
    topk_softmax(
        topk_weights=topk_weights,
        topk_ids=topk_indices,
        gating_output=gating_output,
        renormalize=True,
    )

    return topk_weights, topk_indices


def calculate_diff(num_tokens, num_experts, topk):
    """
    Compare output difference between VLLM and SGLang implementations.
    """
    gating_output = torch.randn((num_tokens, num_experts), device="cuda", dtype=torch.float32)

    weights_vllm, indices_vllm = vllm_topk_softmax(gating_output.clone(), topk)
    weights_sglang, indices_sglang = sglang_topk_softmax(gating_output.clone(), topk)

    weights_diff = torch.abs(weights_vllm - weights_sglang).mean().item()
    indices_match = torch.equal(indices_vllm, indices_sglang)

    if torch.allclose(weights_vllm, weights_sglang, atol=1e-3, rtol=1e-3) and indices_match:
        print(f"✅ Match | Tokens={num_tokens}, Experts={num_experts}, TopK={topk}")
    else:
        print(f"❌ Diff    | Tokens={num_tokens}, Δ={weights_diff:.6f}, Indices={indices_match}")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens", "num_experts", "topk"],
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
def benchmark(num_tokens, num_experts, topk, provider):

    gating_output = torch.randn(
        (num_tokens, num_experts), device="cuda", dtype=torch.float32
    )

    if provider == "vllm" or provider == "vllm1":
        fn = lambda: vllm_topk_softmax(gating_output, topk)
    elif provider == "sglang" or provider == "sglang1":
        fn = lambda: sglang_topk_softmax(gating_output, topk)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    # Run correctness test on small configs if not using a real model
    args = parse_args()
    config = get_model_config(args)
    if args.model_name is None:
        print("🧪 Running correctness tests on default configs...")
        test_configs = [
            (20, 256, 4),
            (20, 256, 8),
            (20, 12, 4),
            (20, 12, 1),
            (20, 512, 4),
            (20, 512, 1),
        ]
        for n, e, k in test_configs:
            calculate_diff(n, e, k)

    # Run benchmark
    print("🚀 Starting performance benchmark...")
    benchmark.run(print_data=True, show_plots=False, save_path=".")