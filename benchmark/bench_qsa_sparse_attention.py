"""Compare Xe2 BF16 QSA attention to the SGLang torch reference path."""

import argparse
import itertools

import pandas as pd
import sgl_kernel  # noqa: F401
import torch
import triton

all_results = []


def reference(q, k, v, slots, scale):
    result = []
    repeats = q.shape[1] // k.shape[1]
    for row in range(q.shape[0]):
        valid = slots[row] >= 0
        indices = slots[row, valid].long()
        if indices.numel() == 0:
            result.append(torch.zeros_like(q[row]))
            continue
        keys = k.index_select(0, indices).repeat_interleave(repeats, dim=1)
        values = v.index_select(0, indices).repeat_interleave(repeats, dim=1)
        scores = torch.einsum("hd,khd->hk", q[row].float(), keys.float()) * scale
        result.append(
            torch.einsum("hk,khd->hd", scores.softmax(dim=-1), values.float()).to(
                q.dtype
            )
        )
    return torch.stack(result)


def get_benchmark(providers, warmup_iters: int, device: str = "xpu"):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["rows", "topk"],
            x_vals=configs,
            line_arg="provider",
            line_vals=providers,
            line_names=providers,
            styles=[("blue", "-"), ("green", "-")],
            ylabel="Latency (ms)",
            plot_name="qsa-sparse-attention-reference-vs-xe2",
            args={},
        )
    )
    def benchmark(rows, topk, provider):
        torch.manual_seed(42)
        k = torch.randn(4096, 2, 256, device=device, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        q = torch.randn(rows, 24, 256, device=device, dtype=torch.bfloat16)
        slots = torch.randint(
            0, k.size(0), (rows, topk), device=device, dtype=torch.int32
        )

        if provider == "reference":

            def run_op():
                return reference(q, k, v, slots, 0.0625)

        else:

            def run_op():
                return sgl_kernel.qsa_sparse_attention(q, k, v, slots, 0.0625)

        for _ in range(warmup_iters):
            run_op()
        torch.xpu.synchronize()

        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(run_op, quantiles=quantiles)

        flops = rows * q.size(1) * topk * q.size(2) * 4
        logical_kv_bytes = rows * q.size(1) * topk * q.size(2) * 2 * 2
        all_results.append(
            {
                "provider": provider,
                "rows": rows,
                "topk": topk,
                "ms": ms,
                "useful_tflops": flops / (ms * 1e9),
                "logical_kv_gbps": logical_kv_bytes / (ms * 1e6),
            }
        )
        print(
            f"rows={rows:5} topk={topk:5} provider={provider:9} ms={ms:.3f} "
            f"useful_tflops={flops / (ms * 1e9):.3f} "
            f"logical_kv_gbps={logical_kv_bytes / (ms * 1e6):.1f}"
        )

        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 16, 128, 1024, 2048])
    parser.add_argument("--topk", type=int, nargs="+", default=[64, 512, 2051])
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--skip-reference", action="store_true")
    args = parser.parse_args()
    if any(rows <= 0 for rows in args.rows):
        parser.error("--rows must contain positive integers")
    if args.warmup_iters < 0:
        parser.error("--warmup-iters must be non-negative")

    if not torch.xpu.is_available():
        raise SystemExit("XPU required")

    providers = ["xe2"] if args.skip_reference else ["reference", "xe2"]
    configs = list(itertools.product(args.rows, args.topk))
    print(f"Running {len(configs)} qsa-sparse-attention benchmark configs")

    benchmark = get_benchmark(providers, args.warmup_iters, device="xpu")
    benchmark.run(print_data=False, show_plots=False, save_path=".")

    df = pd.DataFrame(all_results)
    summary_df = df.pivot_table(
        index=["rows", "topk"],
        columns="provider",
        values=["ms", "useful_tflops", "logical_kv_gbps"],
        aggfunc="first",
    )
    summary_df.columns = [
        f"{metric}_{provider}" for metric, provider in summary_df.columns
    ]
    summary_df = summary_df.reset_index()
    summary_df["rows"] = summary_df["rows"].astype(int)
    summary_df["topk"] = summary_df["topk"].astype(int)
    if not args.skip_reference:
        summary_df["speedup"] = summary_df["ms_reference"] / summary_df["ms_xe2"]
    for column in summary_df.columns:
        if column not in ("rows", "topk") and summary_df[column].dtype == float:
            summary_df[column] = summary_df[column].map(lambda x: f"{x:.3f}")

    print("Reference vs Xe2 QSA sparse attention latency summary:")
    print(summary_df.to_markdown(index=False))
