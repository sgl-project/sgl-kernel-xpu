# python3 benchmark/bench_fused_moe.py
from itertools import product

import torch
import triton
from sgl_kernel import fused_experts, topk_softmax
from torch.nn import functional as F

shape_configs = [
    # # Qwen/Qwen2-57B-A14B-Instruct, tp = 1
    # {
    #     "num_experts": 64,
    #     "topk": 8,
    #     "hidden_size": 3584,
    #     "shard_intermediate_size": 5120,
    #     "dtype": torch.bfloat16,
    #     "block_shape": None,
    # },
    # # Qwen/Qwen2-57B-A14B-Instruct, tp = 2
    # {
    #     "num_experts": 64,
    #     "topk": 8,
    #     "hidden_size": 3584,
    #     "shard_intermediate_size": 2560,
    #     "dtype": torch.bfloat16,
    #     "block_shape": None,
    # },
    # Qwen/Qwen2-57B-A14B-Instruct, tp = 4
    {
        "num_experts": 64,
        "topk": 8,
        "hidden_size": 3584,
        "shard_intermediate_size": 1280,
        "dtype": torch.bfloat16,
        "block_shape": None,
    },
    # Qwen/Qwen2-57B-A14B-Instruct, tp = 8
    {
        "num_experts": 64,
        "topk": 8,
        "hidden_size": 3584,
        "shard_intermediate_size": 640,
        "dtype": torch.bfloat16,
        "block_shape": None,
    },
    # # DeepSeek-V3-0324, tp = 1
    # {
    #     "num_experts": 257,
    #     "topk": 8,
    #     "hidden_size": 7168,
    #     "shard_intermediate_size": 4096,
    #     "dtype": torch.bfloat16,
    #     "block_shape": [128, 128],
    # },
    # # DeepSeek-V3-0324, tp = 2
    # {
    #     "num_experts": 257,
    #     "topk": 8,
    #     "hidden_size": 7168,
    #     "shard_intermediate_size": 2048,
    #     "dtype": torch.bfloat16,
    #     "block_shape": [128, 128],
    # },
    # # DeepSeek-V3-0324, tp = 4
    # {
    #     "num_experts": 257,
    #     "topk": 8,
    #     "hidden_size": 7168,
    #     "shard_intermediate_size": 1024,
    #     "dtype": torch.bfloat16,
    #     "block_shape": [128, 128],
    # },
    # # DeepSeek-V3-0324, tp = 8
    # {
    #     "num_experts": 257,
    #     "topk": 8,
    #     "hidden_size": 7168,
    #     "shard_intermediate_size": 512,
    #     "dtype": torch.bfloat16,
    #     "block_shape": [128, 128],
    # },
    # # Mixtral-8x7B-Instruct-v0.1, tp = 1
    # {
    #     "num_experts": 8,
    #     "topk": 2,
    #     "hidden_size": 4096,
    #     "shard_intermediate_size": 28672,
    #     "dtype": torch.bfloat16,
    #     "block_shape": None,
    # },
    # # Mixtral-8x7B-Instruct-v0.1, tp = 2
    # {
    #     "num_experts": 8,
    #     "topk": 2,
    #     "hidden_size": 4096,
    #     "shard_intermediate_size": 14336,
    #     "dtype": torch.bfloat16,
    #     "block_shape": None,
    # },
    # Mixtral-8x7B-Instruct-v0.1, tp = 4
    {
        "num_experts": 8,
        "topk": 2,
        "hidden_size": 4096,
        "shard_intermediate_size": 7168,
        "dtype": torch.bfloat16,
        "block_shape": None,
    },
    # Mixtral-8x7B-Instruct-v0.1, tp = 8
    {
        "num_experts": 8,
        "topk": 2,
        "hidden_size": 4096,
        "shard_intermediate_size": 3584,
        "dtype": torch.bfloat16,
        "block_shape": None,
    },
]

shape_values = [list(d.values()) for d in shape_configs]
bs = [1, 16, 32]  # 128, 256, 512, 1024, 2048, 4096, 8192]
configs = [(k, *v) for k, v in product(bs, shape_values)]


def fused_topk_native(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
    M, _ = hidden_states.shape
    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
    topk_weights = F.softmax(gating_output.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(topk_weights, topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids


@torch.compile(dynamic=False)
def fused_moe_torch(
    x,
    w1,
    w2,
    input_gating,
    topk,
) -> torch.Tensor:

    topk_weights, topk_ids = fused_topk_native(
        hidden_states=x,
        gating_output=input_gating,
        topk=topk,
        renormalize=True,
    )
    w13_weights = w1[topk_ids]
    w1_weights, w3_weights = torch.chunk(w13_weights, 2, dim=2)
    w2_weights = w2[topk_ids]
    x1 = torch.einsum("ti,taoi -> tao", x, w1_weights)
    x1 = F.silu(x1)
    x3 = torch.einsum("ti, taoi -> tao", x, w3_weights)
    expert_outs = torch.einsum("tao, taio -> tai", (x1 * x3), w2_weights)
    return torch.einsum("tai,ta -> ti", expert_outs, topk_weights.to(expert_outs.dtype))


def fused_moe_torch_compile(
    x,
    w1,
    w2,
    input_gating,
    topk,
    use_fp8_w8a8=False,
    w1_scale=None,
    w2_scale=None,
    a1_scale=None,
    a2_scale=None,
):
    return fused_moe_torch(
        x,
        w1,
        w2,
        input_gating,
        topk,
    )


def fused_moe_sglang_api(
    x,
    w1,
    w2,
    input_gating,
    topk,
):
    num_tokens = x.shape[0]
    topk_weights = torch.empty(num_tokens, topk, dtype=torch.float32, device=x.device)
    topk_indices = torch.empty(num_tokens, topk, dtype=torch.int32, device=x.device)

    topk_softmax(
        topk_weights,
        topk_indices,
        input_gating,
        renormalize=True,
    )
    return fused_experts(
        x,
        w1,
        w2,
        topk_weights,
        topk_indices,
    )


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=[
            "num_tokens",
            "num_experts",
            "topk",
            "hidden_size",
            "shard_intermediate_size",
            "dtype",
            "block_shape",
        ],
        x_vals=configs,
        line_arg="provider",
        line_vals=[
            "torch_compile",
            "sgl_kernel",
        ],
        line_names=[
            "torch_compile",
            "sgl_kernel",
        ],
        styles=[
            ("blue", "-"),
            ("green", "-"),
        ],
        ylabel="Time (ms)",
        plot_name="fused-moe-performance",
        args={},
    )
)
def benchmark(
    num_tokens,
    num_experts,
    topk,
    hidden_size,
    shard_intermediate_size,
    dtype,
    block_shape,
    provider,
):
    print(
        f"benchmark {provider} with batch_size={num_tokens} hidden_size={hidden_size} shard_intermediate_size={shard_intermediate_size}"
    )
    torch.set_default_device("xpu")
    torch.xpu.manual_seed_all(0)

    x = torch.randn(num_tokens, hidden_size, dtype=dtype)

    w1 = torch.randn(num_experts, shard_intermediate_size, hidden_size, dtype=dtype)
    w2 = torch.randn(
        num_experts, hidden_size, shard_intermediate_size // 2, dtype=dtype
    )

    input_gating = torch.randn(num_tokens, num_experts, dtype=dtype)

    if provider == "torch_compile":
        api_func = fused_moe_torch_compile
    else:
        api_func = fused_moe_sglang_api

    api_kwargs = {
        "x": x,
        "w1": w1,
        "w2": w2,
        "input_gating": input_gating,
        "topk": topk,
    }

    # Warmup
    for _ in range(10):
        _ = api_func(**api_kwargs)
    torch.xpu.synchronize()

    bench_lambda = lambda: api_func(**api_kwargs)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(bench_lambda, quantiles=quantiles)
    torch.xpu.empty_cache()
    del x, w1, w2, input_gating
    return ms, min_ms, max_ms


if __name__ == "__main__":
    benchmark.run(print_data=True)
    print("Benchmark finished!")
