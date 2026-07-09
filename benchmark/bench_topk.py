from itertools import product

import pandas as pd
import torch
import triton
from sgl_kernel import (
    fast_topk_transform_fused,
    fast_topk_transform_ragged_fused,
    fast_topk_v2,
)

# Sweep configuration (skip bs=1 per request).
batch_size_range = [132, 256, 4096]
k_range = [2048]  # only 2048 supported
seq_len_range = [2048, 4096, 16384, 65536]
has_row_starts_range = [True, False]

configs = [
    (bs, k, seq_len, has_row_starts)
    for bs, k, seq_len, has_row_starts in product(
        batch_size_range, k_range, seq_len_range, has_row_starts_range
    )
]

all_results = []


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["bs", "k", "seq_len", "has_row_starts"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["sgl_kernel"],
        line_names=["SGL Kernel"],
        styles=[("blue", "-")],
        ylabel="Time (ms)",
        plot_name="fast-topk-v2-performance",
        args={},
    )
)
def benchmark_fast_topk_v2(bs, k, seq_len, has_row_starts, provider):
    print(
        f"benchmark fast_topk_v2 {provider} bs={bs} k={k} seq_len={seq_len} "
        f"has_row_starts={has_row_starts}"
    )
    torch.xpu.manual_seed_all(42)

    score = torch.randn(
        bs,
        seq_len + (2048 if has_row_starts else 0),
        dtype=torch.float32,
        device="xpu",
    )
    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device="xpu")
    if has_row_starts:
        row_starts = torch.randint(0, 2048, (bs,), dtype=torch.int32, device="xpu")
    else:
        row_starts = None

    fn = lambda: fast_topk_v2(score, lengths, k, row_starts=row_starts)

    # Warmup
    for _ in range(5):
        fn()
    torch.xpu.synchronize()

    quantiles = [0.5, 0.25, 0.75]
    ms, _, _ = triton.testing.do_bench(fn, quantiles=quantiles, return_mode="median")

    torch.xpu.empty_cache()

    total_bytes = score.numel() * score.element_size()
    bandwidth_gb_s = total_bytes / (ms / 1e3) / 1e9

    all_results.append(
        {
            "op": "fast_topk_v2",
            "bs": bs,
            "k": k,
            "seq_len": seq_len,
            "has_row_starts": has_row_starts,
            "provider": provider,
            "bandwidth_gb_s": bandwidth_gb_s,
            "ms": ms,
        }
    )
    return ms


fused_configs = [
    (bs, k, seq_len, mode)
    for bs, k, seq_len, mode in product(
        batch_size_range, k_range, seq_len_range, ["extend", "decode", "target_verify"]
    )
]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["bs", "k", "seq_len", "mode"],
        x_vals=fused_configs,
        line_arg="provider",
        line_vals=["sgl_kernel"],
        line_names=["SGL Kernel"],
        styles=[("blue", "-")],
        ylabel="Time (ms)",
        plot_name="fast-topk-transform-fused-performance",
        args={},
    )
)
def benchmark_fast_topk_transform_fused(bs, k, seq_len, mode, provider):
    print(
        f"benchmark fast_topk_transform_fused {provider} bs={bs} k={k} "
        f"seq_len={seq_len} mode={mode}"
    )
    torch.xpu.manual_seed_all(42)

    if mode == "decode":
        step = 1
    else:
        step = 4 if bs % 4 == 0 else 1
    num_tokens = bs
    bs_eff = bs // step

    if mode == "extend":
        row_starts = torch.randint(0, 2048, (bs_eff,), dtype=torch.int32, device="xpu")
    else:
        row_starts = None

    score = torch.randn(
        bs_eff,
        seq_len + (2048 if row_starts is not None else 0),
        dtype=torch.float32,
        device="xpu",
    )
    lengths = torch.full((bs_eff,), seq_len, dtype=torch.int32, device="xpu")
    cu_seqlens_q = torch.arange(
        0, num_tokens + 1, step=step, dtype=torch.int32, device="xpu"
    )
    src_page_table = torch.arange(0, seq_len, dtype=torch.int32, device="xpu")
    src_page_table = src_page_table.unsqueeze(0).expand(bs_eff, -1).contiguous()

    fn = lambda: fast_topk_transform_fused(
        score=score,
        lengths=lengths,
        page_table_size_1=src_page_table,
        cu_seqlens_q=cu_seqlens_q,
        topk=k,
        row_starts=row_starts,
    )

    for _ in range(5):
        fn()
    torch.xpu.synchronize()

    quantiles = [0.5, 0.25, 0.75]
    ms, _, _ = triton.testing.do_bench(fn, quantiles=quantiles, return_mode="median")

    torch.xpu.empty_cache()

    total_bytes = score.numel() * score.element_size()
    bandwidth_gb_s = total_bytes / (ms / 1e3) / 1e9

    all_results.append(
        {
            "op": "fast_topk_transform_fused",
            "bs": bs,
            "k": k,
            "seq_len": seq_len,
            "mode": mode,
            "provider": provider,
            "bandwidth_gb_s": bandwidth_gb_s,
            "ms": ms,
        }
    )
    return ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["bs", "k", "seq_len", "has_row_starts"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["sgl_kernel"],
        line_names=["SGL Kernel"],
        styles=[("blue", "-")],
        ylabel="Time (ms)",
        plot_name="fast-topk-transform-ragged-fused-performance",
        args={},
    )
)
def benchmark_fast_topk_transform_ragged_fused(
    bs, k, seq_len, has_row_starts, provider
):
    print(
        f"benchmark fast_topk_transform_ragged_fused {provider} bs={bs} k={k} "
        f"seq_len={seq_len} has_row_starts={has_row_starts}"
    )
    torch.xpu.manual_seed_all(42)

    score = torch.randn(
        bs,
        seq_len + (2048 if has_row_starts else 0),
        dtype=torch.float32,
        device="xpu",
    )
    if has_row_starts:
        row_starts = torch.randint(0, 2048, (bs,), dtype=torch.int32, device="xpu")
    else:
        row_starts = None
    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device="xpu")
    topk_indices_offset = torch.randint(0, 1024, (bs,), dtype=torch.int32, device="xpu")

    fn = lambda: fast_topk_transform_ragged_fused(
        score=score,
        lengths=lengths,
        topk_indices_offset=topk_indices_offset,
        topk=k,
        row_starts=row_starts,
    )

    for _ in range(5):
        fn()
    torch.xpu.synchronize()

    quantiles = [0.5, 0.25, 0.75]
    ms, _, _ = triton.testing.do_bench(fn, quantiles=quantiles, return_mode="median")

    torch.xpu.empty_cache()

    total_bytes = score.numel() * score.element_size()
    bandwidth_gb_s = total_bytes / (ms / 1e3) / 1e9

    all_results.append(
        {
            "op": "fast_topk_transform_ragged_fused",
            "bs": bs,
            "k": k,
            "seq_len": seq_len,
            "has_row_starts": has_row_starts,
            "provider": provider,
            "bandwidth_gb_s": bandwidth_gb_s,
            "ms": ms,
        }
    )
    return ms


if __name__ == "__main__":
    benchmark_fast_topk_v2.run(print_data=False)
    benchmark_fast_topk_transform_fused.run(print_data=False)
    benchmark_fast_topk_transform_ragged_fused.run(print_data=False)
    print("Benchmark finished!")

    df = pd.DataFrame(all_results)
    for op, sub in df.groupby("op", sort=False):
        sub = sub.dropna(axis=1, how="all").reset_index(drop=True)
        print(f"\n### {op}\n")
        print(sub.to_markdown(index=False))
