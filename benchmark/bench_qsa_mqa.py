"""Compare Xe2 QSA indexer scoring with the Torch reference."""

import argparse
import math

import pandas as pd
import sgl_kernel  # noqa: F401
import torch
import triton

all_results = []


def prefill_reference(q, k, starts, ends):
    scores = torch.einsum("rhd,cd->rch", q.float(), k[:, 0].float())
    logits = scores.relu().sum(-1) / math.sqrt(128)
    positions = torch.arange(k.shape[0], device=q.device)[None, :]
    return logits.masked_fill(
        (positions < starts[:, None]) | (positions >= ends[:, None]), -float("inf")
    )


def decode_reference(q, cache, pages, lengths, width):
    rows = q.shape[0]
    keys = cache[pages.long().reshape(-1), :, 0].reshape(rows, -1, 128)
    scores = torch.einsum("rhd,rcd->rch", q.float(), keys.float())
    logits = scores.relu().sum(-1)[:, :width] / math.sqrt(128)
    logits.masked_fill_(
        torch.arange(width, device=q.device)[None, :] >= lengths[:, None], -float("inf")
    )
    return logits


def get_benchmark(page_size: int, device: str = "xpu"):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["mode", "rows", "columns"],
            x_vals=configs,
            line_arg="provider",
            line_vals=["torch", "xe2"],
            line_names=["torch", "xe2"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="Latency (ms)",
            plot_name="qsa-mqa-torch-vs-xe2",
            args={},
        )
    )
    def benchmark(mode, rows, columns, provider):
        torch.manual_seed(73)
        q = torch.randn(rows, 4, 128, device=device, dtype=torch.bfloat16)

        if mode == "prefill":
            k = torch.randn(columns, 1, 128, device=device, dtype=torch.bfloat16)
            starts = torch.zeros(rows, device=device, dtype=torch.int32)
            ends = torch.full_like(starts, columns)

            if provider == "torch":

                def run_op():
                    return prefill_reference(q, k, starts, ends)

            else:

                def run_op():
                    return torch.ops.sgl_kernel.qsa_mqa_prefill(
                        q, k, starts, ends, None
                    )

        else:
            pages_per_row = (columns + page_size - 1) // page_size
            cache = torch.randn(
                pages_per_row, page_size, 1, 128, device=device, dtype=torch.bfloat16
            )
            pages = (
                torch.arange(pages_per_row, device=device, dtype=torch.int32)
                .expand(rows, -1)
                .contiguous()
            )
            lengths = torch.full((rows,), columns, device=device, dtype=torch.int32)

            if provider == "torch":

                def run_op():
                    return decode_reference(q, cache, pages, lengths, columns)

            else:

                def run_op():
                    return torch.ops.sgl_kernel.qsa_mqa_decode(
                        q, cache, pages, lengths, columns, None
                    )

        for _ in range(3):
            run_op()
        torch.xpu.synchronize()

        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(run_op, quantiles=quantiles)

        all_results.append(
            {
                "mode": mode,
                "rows": rows,
                "columns": columns,
                "provider": provider,
                "ms": ms,
            }
        )
        print(
            f"mode={mode:7} rows={rows:5} columns={columns:5} "
            f"provider={provider:5} ms={ms:.3f}"
        )

        return ms, max_ms, min_ms

    return benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill-rows", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--decode-rows", type=int, nargs="+", default=[1, 16, 64])
    parser.add_argument("--columns", type=int, nargs="+", default=[512, 1024, 4096])
    parser.add_argument("--page-size", type=int, default=32)
    args = parser.parse_args()
    if (
        any(x <= 0 for x in args.prefill_rows + args.decode_rows + args.columns)
        or args.page_size <= 0
    ):
        parser.error(
            "prefill rows, decode rows, columns, and page size must be positive"
        )

    if not torch.xpu.is_available():
        raise SystemExit("XPU required")

    configs = [
        ("prefill", rows, columns)
        for rows in args.prefill_rows
        for columns in args.columns
    ] + [
        ("decode", rows, columns)
        for rows in args.decode_rows
        for columns in args.columns
    ]
    print(f"Running {len(configs)} qsa-mqa benchmark configs")

    benchmark = get_benchmark(args.page_size, device="xpu")
    benchmark.run(print_data=False, show_plots=False, save_path=".")

    df = pd.DataFrame(all_results)
    summary_df = (
        df.pivot_table(
            index=["mode", "rows", "columns"],
            columns="provider",
            values="ms",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={"torch": "torch_ms", "xe2": "xe2_ms"})
    )
    summary_df["speedup"] = summary_df["torch_ms"] / summary_df["xe2_ms"]
    summary_df["torch_ms"] = summary_df["torch_ms"].map(lambda x: f"{x:.3f}")
    summary_df["xe2_ms"] = summary_df["xe2_ms"].map(lambda x: f"{x:.3f}")
    summary_df["speedup"] = summary_df["speedup"].map(lambda x: f"{x:.2f}x")

    print("Torch vs Xe2 QSA indexer latency summary:")
    print(summary_df.to_markdown(index=False))
