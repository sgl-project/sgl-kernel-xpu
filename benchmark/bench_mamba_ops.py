import argparse
from dataclasses import dataclass
from typing import Callable

import pandas as pd
import sgl_kernel
import torch
import triton
from torch.nn.functional import softplus


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, _ = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(
        value
    )
    last_recurrent_state = initial_state.to(value)

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(
            -1
        ) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def torch_fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool,
):
    num_heads = q.shape[2]
    num_value_heads = v.shape[2]
    q_ref = q.transpose(0, 1)
    k_ref = k.transpose(0, 1)
    v_ref = v.transpose(0, 1)

    if num_value_heads // num_heads > 1:
        repeat = num_value_heads // num_heads
        q_ref = q_ref.repeat_interleave(repeat, dim=2)
        k_ref = k_ref.repeat_interleave(repeat, dim=2)

    beta = b.sigmoid()
    g = -A_log.float().exp() * softplus(a.float() + dt_bias)
    out, final_state = torch_recurrent_gated_delta_rule(
        q_ref,
        k_ref,
        v_ref,
        g.unsqueeze(0),
        beta.unsqueeze(0),
        initial_state_source[initial_state_indices],
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    return out, final_state


def torch_fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
):
    g = (-A_log.float().exp() * softplus(a.float() + dt_bias)).unsqueeze(0)
    beta = b.sigmoid().unsqueeze(0)
    return g, beta


@dataclass
class BenchResult:
    op: str
    provider: str
    config: str
    ms: float
    tflops: float
    bw_gbs: float


def tensor_bytes(*tensors: torch.Tensor) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def do_bench(fn: Callable[[], None]) -> float:
    ms, _, _ = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
    return ms


# Derived from sglang benchmark/bench_linear_attention/bench_gdn_decode.py
DECODE_BENCH_CONFIGS = [
    # (batch, q_heads, v_heads, head_dim)
    (1, 8, 16, 128),
    (4, 8, 16, 128),
    (16, 8, 16, 128),
    (32, 8, 16, 128),
    (64, 8, 16, 128),
    (128, 8, 16, 128),
    (1, 16, 32, 128),
    (32, 16, 32, 128),
    (64, 16, 32, 128),
]


# Derived from sglang benchmark/bench_linear_attention/bench_gdn_prefill.py
PREFILL_BENCH_CONFIGS = [
    # (batch(num_seqs), seq_per_seq, heads, head_dim)
    (4, 64, 16, 128),
    (4, 256, 16, 128),
    (8, 128, 16, 128),
    (16, 64, 16, 128),
    (32, 32, 16, 128),
    (4, 128, 32, 128),
]


def benchmark_fused_gdn(device: torch.device, quick: bool) -> list[BenchResult]:
    results: list[BenchResult] = []
    # More configs from decode references (heads = v_heads).
    cfgs = [
        (1024, 32),
        (4096, 64),
        (8192, 16),
    ]
    cfgs.extend((b, vh) for b, _, vh, _ in DECODE_BENCH_CONFIGS)
    if quick:
        cfgs = cfgs[:4]

    for batch, heads in cfgs:
        A_log = torch.rand(heads, dtype=torch.float32, device=device)
        a = torch.rand(batch, heads, dtype=torch.bfloat16, device=device)
        b = torch.rand(batch, heads, dtype=torch.bfloat16, device=device)
        dt_bias = torch.rand(heads, dtype=torch.bfloat16, device=device)

        def fn_torch():
            torch_fused_gdn_gating(A_log, a, b, dt_bias)

        def fn_kernel():
            sgl_kernel.fused_gdn_gating(A_log, a, b, dt_bias)

        for provider, fn in [("torch", fn_torch), ("sgl", fn_kernel)]:
            ms = do_bench(fn)
            n = batch * heads
            flops = n * 20
            # Read A_log/a/b/dt_bias and write g/beta.
            bytes_total = tensor_bytes(A_log, a, b, dt_bias) + (
                4 * n + a.element_size() * n
            )
            tflops = flops / (ms * 1e-3) / 1e12
            bw = bytes_total / (ms * 1e-3) / 1e9
            results.append(
                BenchResult(
                    op="fused_gdn_gating",
                    provider=provider,
                    config=f"batch={batch},heads={heads}",
                    ms=ms,
                    tflops=tflops,
                    bw_gbs=bw,
                )
            )
    return results


def benchmark_fused_sigmoid_update(
    device: torch.device, quick: bool
) -> list[BenchResult]:
    results: list[BenchResult] = []
    # Pull decode-like shapes and a couple of longer decode steps.
    cfgs = [
        (1, 16, 32, 128),
        (8, 16, 32, 128),
        (16, 16, 32, 128),
    ]
    cfgs = DECODE_BENCH_CONFIGS

    for seq_len, q_heads, v_heads, d in cfgs:
        batch_size = 1
        q = torch.rand(
            batch_size, seq_len, q_heads, d, dtype=torch.bfloat16, device=device
        )
        k = torch.rand(
            batch_size, seq_len, q_heads, d, dtype=torch.bfloat16, device=device
        )
        v = torch.rand(
            batch_size, seq_len, v_heads, d, dtype=torch.bfloat16, device=device
        )
        A_log = torch.rand(v_heads, dtype=torch.float32, device=device)
        a = torch.rand(batch_size, v_heads, dtype=torch.bfloat16, device=device)
        b = torch.rand(batch_size, v_heads, dtype=torch.bfloat16, device=device)
        dt_bias = torch.rand(v_heads, dtype=torch.bfloat16, device=device)
        ssm_states = torch.rand(513, v_heads, d, d, dtype=torch.float32, device=device)
        cache_indices = torch.randint(
            0, 513, (batch_size,), dtype=torch.int32, device=device
        )
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

        def fn_torch():
            torch_fused_sigmoid_gating_delta_rule_update(
                A_log,
                dt_bias,
                q,
                k,
                v,
                a,
                b,
                ssm_states,
                cache_indices,
                use_qk_l2norm_in_kernel=True,
            )

        def fn_kernel():
            sgl_kernel.fused_sigmoid_gating_delta_rule_update(
                A_log,
                dt_bias,
                q,
                k,
                v,
                a,
                b,
                ssm_states,
                cache_indices,
                cu_seqlens,
                True,
                1.0,
                20.0,
            )

        for provider, fn in [("torch", fn_torch), ("sgl", fn_kernel)]:
            ms = do_bench(fn)
            n = batch_size * seq_len * v_heads * d * d
            flops = n * 4
            bytes_total = tensor_bytes(
                q, k, v, A_log, a, b, dt_bias
            ) + 2 * tensor_bytes(ssm_states)
            tflops = flops / (ms * 1e-3) / 1e12
            bw = bytes_total / (ms * 1e-3) / 1e9
            results.append(
                BenchResult(
                    op="fused_sigmoid_gating_delta_rule_update",
                    provider=provider,
                    config=f"seq={seq_len},qh={q_heads},vh={v_heads},d={d}",
                    ms=ms,
                    tflops=tflops,
                    bw_gbs=bw,
                )
            )
    return results


def benchmark_chunk_gated_delta_rule(
    device: torch.device, quick: bool
) -> list[BenchResult]:
    results: list[BenchResult] = []
    # Prefill-like shapes from reference benchmark.
    cfgs = []
    for b, t_per_seq, h, d in PREFILL_BENCH_CONFIGS:
        cfgs.append((b * t_per_seq, h, h, d, b))
    # Keep one legacy config.
    cfgs.append((512, 16, 32, 64, 8))
    if quick:
        cfgs = cfgs[:3]

    for t, q_heads, v_heads, d, nseq in cfgs:
        batch_size = 1
        seqlens = torch.randint(1, max(2, t // nseq), (nseq + 1,), device=device)
        seqlens[0] = 0
        cu_seqlens = torch.cumsum(seqlens, dim=0).to(torch.int32)
        total_t = int(cu_seqlens[-1].item())

        q = torch.rand(
            (batch_size, total_t, q_heads, d), dtype=torch.bfloat16, device=device
        )
        k = torch.rand(
            (batch_size, total_t, q_heads, d), dtype=torch.bfloat16, device=device
        )
        v = torch.rand(
            (batch_size, total_t, v_heads, d), dtype=torch.bfloat16, device=device
        )
        g = torch.rand(
            (batch_size, total_t, v_heads), dtype=torch.float32, device=device
        )
        beta = torch.rand(
            (batch_size, total_t, v_heads), dtype=torch.bfloat16, device=device
        )
        initial_state = torch.rand(
            (nseq, v_heads, d, d), dtype=torch.float32, device=device
        )

        def fn_kernel():
            sgl_kernel.chunk_gated_delta_rule(
                q,
                k,
                v,
                g,
                beta,
                initial_state,
                cu_seqlens,
                False,
                True,
            )

        ms = do_bench(fn_kernel)
        n = batch_size * total_t * v_heads * d * d
        flops = n * 4
        bytes_total = tensor_bytes(q, k, v, g, beta, initial_state)
        tflops = flops / (ms * 1e-3) / 1e12
        bw = bytes_total / (ms * 1e-3) / 1e9
        results.append(
            BenchResult(
                op="chunk_gated_delta_rule",
                provider="sgl",
                config=f"T={total_t},qh={q_heads},vh={v_heads},d={d},n={nseq}",
                ms=ms,
                tflops=tflops,
                bw_gbs=bw,
            )
        )
    return results


def print_results(results: list[BenchResult]) -> None:
    rows = [
        {
            "op": r.op,
            "provider": r.provider,
            "config": r.config,
            "latency_ms": round(r.ms, 4),
            "TFLOPS": round(r.tflops, 6),
            "BW_GBps": round(r.bw_gbs, 4),
        }
        for r in results
    ]
    df = pd.DataFrame(rows)

    print("\n## Mamba Benchmark Results")
    try:
        print(df.to_markdown(index=False))
    except ImportError:
        print(df.to_string(index=False))

    print("\n## Per-Op Summary (best latency)")
    summary = (
        df.groupby(["op", "provider"], as_index=False)["latency_ms"]
        .min()
        .sort_values(["op", "provider"])
    )
    try:
        print(summary.to_markdown(index=False))
    except ImportError:
        print(summary.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark mamba ops and report BW/TFLOPS"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a reduced configuration set for quick smoke validation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.set_default_device("xpu")
    device = torch.device("xpu")

    all_results: list[BenchResult] = []
    all_results.extend(benchmark_fused_gdn(device, args.quick))
    all_results.extend(benchmark_fused_sigmoid_update(device, args.quick))
    all_results.extend(benchmark_chunk_gated_delta_rule(device, args.quick))

    print_results(all_results)
