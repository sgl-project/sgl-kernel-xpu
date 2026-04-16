# python3 benchmark/bench_fused_moe.py
from itertools import product

import torch
import triton
from sgl_kernel import fused_experts, topk_softmax
from torch.nn import functional as F

# GPT-OSS SwiGLU parameters
SWIGLU_GPT_OSS_ALPHA = 1.702
SWIGLU_GPT_OSS_LIMIT = 7.0

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
    # deepseek-OCR, tp=1
    {
        "num_experts": 64,
        "topk": 6,
        "hidden_size": 1280,
        "shard_intermediate_size": 1792,
        "dtype": torch.bfloat16,
        "block_shape": None,
    },
]

shape_configs_gelu = [
    # grok, tp=1
    {
        "num_experts": 8,
        "topk": 2,
        "hidden_size": 8192,
        "shard_intermediate_size": 16384,
        "dtype": torch.bfloat16,
        "block_shape": None,
    },
]

shape_configs_swiglu_gpt_oss = [
    # lmsys/gpt-oss-20b-bf16, tp = 4
    {
        "num_experts": 32,
        "topk": 4,
        "hidden_size": 2880,
        "shard_intermediate_size": 2880,
        "dtype": torch.bfloat16,
        "block_shape": None,
        "gemm1_alpha": SWIGLU_GPT_OSS_ALPHA,
        "gemm1_limit": SWIGLU_GPT_OSS_LIMIT,
    },
]


def _cfg_vals(d):
    return [
        d["num_experts"],
        d["topk"],
        d["hidden_size"],
        d["shard_intermediate_size"],
        d["dtype"],
        d["block_shape"],
        d.get("gemm1_alpha"),
        d.get("gemm1_limit"),
    ]


shape_values = [_cfg_vals(d) for d in shape_configs]
bs = [1, 128, 512, 1024, 2048, 4096, 8192]
with_bias = [False, True]
configs = [(k, *v, b, "silu") for k, v, b in product(bs, shape_values, with_bias)]
shape_values_gelu = [_cfg_vals(d) for d in shape_configs_gelu]
configs += [(k, *v, b, "gelu") for k, v, b in product(bs, shape_values_gelu, with_bias)]
shape_values_swiglu_gpt_oss = [_cfg_vals(d) for d in shape_configs_swiglu_gpt_oss]
configs += [
    (k, *v, b, "silu")
    for k, v, b in product(bs, shape_values_swiglu_gpt_oss, with_bias)
]
all_results = []


@torch.compile
def swiglu_gpt_oss_sigmoid_alpha(x, gemm1_alpha, gemm1_limit):
    # At present, only GPT-OSS uses this variant.
    gate, up = x[..., ::2], x[..., 1::2]
    gate = gate.clamp(min=None, max=gemm1_limit)
    up = up.clamp(min=-gemm1_limit, max=gemm1_limit)
    return gate * torch.sigmoid(gate * gemm1_alpha) * (up + 1)


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
    b1,
    b2,
    act_type,
    gemm1_alpha=None,
    gemm1_limit=None,
    routed_scaling_factor=None,
) -> torch.Tensor:

    topk_weights, topk_ids = fused_topk_native(
        hidden_states=x,
        gating_output=input_gating,
        topk=topk,
        renormalize=True,
    )
    w2_weights = w2[topk_ids]
    is_swiglu_gpt_oss = (
        act_type == "silu" and gemm1_alpha is not None and gemm1_limit is not None
    )

    if is_swiglu_gpt_oss:
        # GPT-OSS: w1 rows are interleaved [g0, u0, g1, u1, ...].
        # Run a single GEMM to get the full interleaved output [T, topk, 2N],
        # add the interleaved bias, then let swiglu_gpt_oss_sigmoid_alpha
        # split gate/up internally.
        w1_weights = w1[topk_ids]
        x1 = torch.einsum("ti,taoi -> tao", x, w1_weights)
        if b1 is not None:
            b1_weights = b1[topk_ids]
            x1 = (x1.float() + b1_weights.float()).to(x.dtype)
        x1 = swiglu_gpt_oss_sigmoid_alpha(x1, gemm1_alpha, gemm1_limit)
        expert_outs = torch.einsum("tao, taio -> tai", x1, w2_weights)
    else:
        # silu/gelu: w1 is block-split [gate_rows | up_rows].
        w13_weights = w1[topk_ids]
        w1_weights, w3_weights = torch.chunk(w13_weights, 2, dim=2)
        x1 = torch.einsum("ti,taoi -> tao", x, w1_weights)
        x3 = torch.einsum("ti, taoi -> tao", x, w3_weights)
        if b1 is not None:
            b1_weights = b1[topk_ids]
            b1_gate, b1_up = torch.chunk(b1_weights, 2, dim=2)
            x1 = (x1.float() + b1_gate.float()).to(x.dtype)
            x3 = (x3.float() + b1_up.float()).to(x.dtype)
        if act_type == "silu":
            x1 = F.silu(x1)
        elif act_type == "gelu":
            x1 = F.gelu(x1, approximate="tanh")
        expert_outs = torch.einsum("tao, taio -> tai", x1 * x3, w2_weights)
    if b2 is not None:
        b2_weights = b2[topk_ids]
        expert_outs = expert_outs.float() + b2_weights.float()
        expert_outs = expert_outs.to(x.dtype)
    result = torch.einsum(
        "tai,ta -> ti", expert_outs, topk_weights.to(expert_outs.dtype)
    )
    if routed_scaling_factor is not None:
        result = result * routed_scaling_factor
    return result


def fused_moe_torch_compile(
    x,
    w1,
    w2,
    input_gating,
    topk,
    b1,
    b2,
    use_fp8_w8a8=False,
    w1_scale=None,
    w2_scale=None,
    a1_scale=None,
    a2_scale=None,
    act_type=None,
    gemm1_alpha=None,
    gemm1_limit=None,
    routed_scaling_factor=None,
):
    return fused_moe_torch(
        x,
        w1,
        w2,
        input_gating,
        topk,
        b1,
        b2,
        act_type=act_type,
        gemm1_alpha=gemm1_alpha,
        gemm1_limit=gemm1_limit,
        routed_scaling_factor=routed_scaling_factor,
    )


def fused_moe_sglang_api(
    x,
    w1,
    w2,
    input_gating,
    topk,
    b1,
    b2,
    act_type,
    gemm1_alpha=None,
    gemm1_limit=None,
    routed_scaling_factor=None,
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
    return (
        fused_experts(
            x,
            w1,
            w2,
            topk_weights,
            topk_indices,
            b1,
            b2,
            activation=act_type,
            gemm1_alpha=gemm1_alpha,
            gemm1_limit=gemm1_limit,
            routed_scaling_factor=routed_scaling_factor,
        ),
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
            "gemm1_alpha",
            "gemm1_limit",
            "with_bias",
            "act_type",
        ],
        x_vals=configs,
        line_arg="provider",
        line_vals=[
            "sgl_kernel",
        ],
        line_names=[
            "sgl_kernel",
        ],
        styles=[
            ("blue", "-"),
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
    with_bias,
    act_type,
    provider,
    gemm1_alpha,
    gemm1_limit,
):
    routed_scaling_factor = 1.0

    print(
        f"benchmark {provider} with {num_tokens=} {hidden_size=} {shard_intermediate_size=} {with_bias=} {act_type=}"
    )
    torch.set_default_device("xpu")
    torch.xpu.manual_seed_all(0)

    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    w1 = torch.randn(num_experts, shard_intermediate_size, hidden_size, dtype=dtype)
    w2 = torch.randn(
        num_experts, hidden_size, shard_intermediate_size // 2, dtype=dtype
    )
    b1, b2 = None, None
    if with_bias:
        b1 = torch.randn(w1.shape[:2], dtype=dtype)
        b2 = torch.randn(w2.shape[:2], dtype=dtype)

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
        "b1": b1,
        "b2": b2,
        "act_type": act_type,
        "gemm1_alpha": gemm1_alpha,
        "gemm1_limit": gemm1_limit,
        "routed_scaling_factor": routed_scaling_factor,
    }

    # Warmup
    for _ in range(10):
        _, topk_ids = api_func(**api_kwargs)
    torch.xpu.synchronize()

    bench_lambda = lambda: api_func(**api_kwargs)

    quantiles = [0.5, 0.2, 0.8]
    ms, _, _ = triton.testing.do_bench(
        bench_lambda,
        warmup=200,
        rep=300,
        quantiles=quantiles,
    )

    torch.xpu.empty_cache()
    del x, w1, w2, input_gating
    flop = (
        num_tokens
        * topk
        * (
            hidden_size * shard_intermediate_size * 2
            + shard_intermediate_size * hidden_size
        )
    )
    if with_bias:
        flop += num_tokens * topk * (shard_intermediate_size + hidden_size)
    num_act_experts = torch.unique(topk_ids).numel()
    memory = (
        num_act_experts
        * (
            hidden_size * shard_intermediate_size
            + hidden_size * shard_intermediate_size // 2
        )
        * torch.finfo(dtype).bits
        // 8
    )
    if with_bias:
        memory += (
            num_act_experts
            * (shard_intermediate_size + hidden_size)
            * torch.finfo(dtype).bits
            // 8
        )
    tflops = flop / (ms / 1e3) / 1e12
    bandwidth = memory / (ms / 1e3) / 1e9

    all_results.append(
        {
            "num_tokens": num_tokens,
            "num_experts": num_experts,
            "topk": topk,
            "hidden_size": hidden_size,
            "shard_intermediate_size": shard_intermediate_size,
            "dtype": dtype,
            # "block_shape": block_shape,  # Always None now. disabled to reduce the number of columns
            "with_bias": with_bias,
            "act_type": act_type,
            "provider": provider,
            "tflops": tflops,
            "bandwidth": bandwidth,
            "ms": ms,
        }
    )
    return ms


if __name__ == "__main__":
    benchmark.run(print_data=False)
    print("Benchmark finished!")

    import pandas as pd

    df = pd.DataFrame(all_results)
    print(df.to_markdown())
