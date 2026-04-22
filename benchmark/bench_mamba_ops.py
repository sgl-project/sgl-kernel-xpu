import argparse
from dataclasses import dataclass
from typing import Callable

import pandas as pd
import sgl_kernel
import torch
import triton


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

        def fn_kernel():
            sgl_kernel.fused_gdn_gating(A_log, a, b, dt_bias)

        for provider, fn in [("sgl", fn_kernel)]:
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
    cfgs = DECODE_BENCH_CONFIGS

    for batch_size, q_heads, v_heads, d in cfgs:
        q_d = v_d = d
        key_dim = q_d * q_heads
        value_dim = v_d * v_heads
        mixed_qkv_dim = q_heads * 2 * q_d + v_heads * v_d
        mixed_qkv = torch.rand(
            batch_size, mixed_qkv_dim, dtype=torch.bfloat16, device=device
        )
        query, key, value = torch.split(
            mixed_qkv,
            [
                key_dim,
                key_dim,
                value_dim,
            ],
            dim=-1,
        )
        q = query.view(1, batch_size, q_heads, q_d)
        k = key.view(1, batch_size, q_heads, q_d)
        v = value.view(1, batch_size, v_heads, v_d)
        A_log = torch.rand(v_heads, dtype=torch.float32, device=device)
        a = torch.rand(batch_size, v_heads, dtype=torch.bfloat16, device=device)
        b = torch.rand(batch_size, v_heads, dtype=torch.bfloat16, device=device)
        dt_bias = torch.rand(v_heads, dtype=torch.bfloat16, device=device)
        ssm_states = torch.rand(
            513, v_heads, v_d, q_d, dtype=torch.float32, device=device
        )
        cache_indices = torch.arange(batch_size, device=device, dtype=torch.int32)

        cu_seqlens = torch.arange(batch_size + 1, device=device, dtype=torch.int32)
        torch.xpu.synchronize()

        def fn_kernel():
            torch.ops.sgl_kernel.fused_sigmoid_gating_delta_rule_update(
                A_log=A_log,
                dt_bias=dt_bias,
                q=q,
                k=k,
                v=v,
                a=a,
                b=b,
                initial_state_source=ssm_states,
                initial_state_indices=cache_indices,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
                softplus_beta=1.0,
                softplus_threshold=20.0,
            )

        for provider, fn in [("sgl", fn_kernel)]:
            ms = do_bench(fn)
            n = batch_size * v_heads * d * d
            flops = n * 4
            actual_state_bytes = (
                batch_size * v_heads * d * d * ssm_states.element_size()
            )
            bytes_total = (
                tensor_bytes(q, k, v, A_log, a, b, dt_bias) + 2 * actual_state_bytes
            )
            tflops = flops / (ms * 1e-3) / 1e12
            bw = bytes_total / (ms * 1e-3) / 1e9
            results.append(
                BenchResult(
                    op="fused_sigmoid_gating_delta_rule_update",
                    provider=provider,
                    config=f"bs={batch_size},qh={q_heads},vh={v_heads},d={d}",
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
        torch.xpu.empty_cache()
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
    device = torch.device("xpu")

    all_results: list[BenchResult] = []
    all_results.extend(benchmark_fused_gdn(device, args.quick))
    all_results.extend(benchmark_fused_sigmoid_update(device, args.quick))
    all_results.extend(benchmark_chunk_gated_delta_rule(device, args.quick))

    print_results(all_results)
