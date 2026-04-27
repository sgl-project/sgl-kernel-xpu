import argparse
import itertools
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import triton
from sgl_kernel import flash_mla_prefill, flash_mla_prefill_get_workspace_size

bs_range = [1, 4, 16]
seq_len_range = [128, 256, 512, 1024]

configs = list(itertools.product(bs_range, seq_len_range))

all_results = []


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "seq_len"],
        x_vals=configs,
        x_log=False,
        line_arg="provider",
        line_vals=[
            "128 heads",
            "64 heads",
            "32 heads",
            "16 heads",
        ],
        line_names=[
            "128 heads",
            "64 heads",
            "32 heads",
            "16 heads",
        ],
        styles=[("red", "-"), ("green", "-"), ("blue", "-"), ("violet", "-")],
        ylabel="GB/s",
        plot_name="cutlass mla prefill",
        args={},
    )
)
def benchmark(batch_size, seq_len, provider, block_size, mode):
    D_latent = 512
    D_rope = 64
    D_ckv = D_latent + D_rope

    h_q_map = {
        "128": 128,
        "64": 64,
        "32": 32,
        "16": 16,
    }
    parsed_h_q = next(
        (value for key, value in h_q_map.items() if key in provider), None
    )

    if parsed_h_q is None:
        raise ValueError(f"Unknown head configuration in provider: {provider}")
    h_q = parsed_h_q

    # Build per-batch sequence lengths
    if mode == "full":
        # Full prefill: seqlen_q == seqlen_k
        seqlens_q = [seq_len] * batch_size
        seqlens_k = [seq_len] * batch_size
    else:
        # Incremental prefill: seqlen_q = 16, seqlen_k = seq_len
        seqlens_q = [16] * batch_size
        seqlens_k = [seq_len] * batch_size

    total_q = sum(seqlens_q)
    max_seqlen_q = max(seqlens_q)

    cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int32, device="xpu")
    torch.cumsum(
        torch.tensor(seqlens_q, dtype=torch.int32, device="xpu"),
        dim=0,
        out=cu_seqlens_q[1:],
    )
    seq_lens_k = torch.tensor(seqlens_k, dtype=torch.int32, device="xpu")

    block_num = (max(seqlens_k) + block_size - 1) // block_size
    pack_factor = 128 // block_size
    block_num = ((block_num + pack_factor - 1) // pack_factor) * pack_factor

    q_nope = torch.randn(
        total_q, h_q, D_latent, dtype=torch.bfloat16, device="xpu"
    )
    q_pe = torch.randn(
        total_q, h_q, D_rope, dtype=torch.bfloat16, device="xpu"
    )
    block_table = torch.randint(
        0,
        batch_size * block_num,
        (batch_size, block_num),
        dtype=torch.int32,
        device="xpu",
    )
    kv_cache = torch.randn(
        block_table.max().item() + 1,
        block_size,
        D_ckv,
        dtype=torch.bfloat16,
        device="xpu",
    )

    scale = (128 + D_rope) ** (-0.5)

    ws_size = flash_mla_prefill_get_workspace_size(
        block_num * block_size, batch_size
    )
    workspace = torch.empty(ws_size, device="xpu", dtype=torch.uint8)

    quantiles = [0.5, 0.25, 0.75]
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: flash_mla_prefill(
            q_nope,
            q_pe,
            kv_cache,
            cu_seqlens_q,
            seq_lens_k,
            max_seqlen_q,
            block_table,
            workspace,
            scale,
            causal=True,
            num_kv_splits=1,
        ),
        quantiles=quantiles,
    )

    # Calculate total bytes transferred for bandwidth calculation
    total_bytes = (
        q_nope.numel() * q_nope.element_size()  # read q_nope
        + q_pe.numel() * q_pe.element_size()  # read q_pe
        + kv_cache.numel() * kv_cache.element_size()  # read kv_cache
        + block_table.numel() * block_table.element_size()  # read page_table
        + seq_lens_k.numel() * seq_lens_k.element_size()  # read seq_lens_k
        + cu_seqlens_q.numel() * cu_seqlens_q.element_size()  # read cu_seqlens_q
        + q_nope.numel() * q_nope.element_size()  # write output (same shape as q_nope)
    )

    gbps = lambda ms: total_bytes * 1e-9 / (ms * 1e-3)

    # Collect results for final summary table
    all_results.append(
        {
            "batch_size": batch_size,
            "seq_len_q": max_seqlen_q,
            "seq_len_k": max(seqlens_k),
            "mode": mode,
            "num_heads": h_q,
            "block_size": block_size,
            "time_ms": f"{ms:.3f}",
            "GB/s (median)": f"{gbps(ms):.2f}",
            "GB/s (min)": f"{gbps(max_ms):.2f}",
            "GB/s (max)": f"{gbps(min_ms):.2f}",
        }
    )

    result = (gbps(ms), gbps(max_ms), gbps(min_ms))
    del q_nope, q_pe, kv_cache, block_table, workspace, cu_seqlens_q, seq_lens_k
    return result


def plot_data(df):

    os.makedirs("bench_bmg_mla_prefill_res", exist_ok=True)

    previous_csv_path = "bench_bmg_mla_prefill_res/previous.csv"
    current_csv_path = "bench_bmg_mla_prefill_res/current.csv"

    df.to_csv(current_csv_path, index=False)
    print(f"Current results saved to: {current_csv_path}")

    all_dataframes = []
    has_previous = os.path.exists(previous_csv_path)

    if has_previous:
        try:
            df_prev = pd.read_csv(previous_csv_path)
            df_prev["GB/s"] = df_prev["GB/s (median)"].astype(float)
            df_prev["run"] = "previous"
            all_dataframes.append(df_prev)
            print(f"Loaded previous results from: {previous_csv_path}")
        except Exception as e:
            print(f"Error loading {previous_csv_path}: {e}")
            has_previous = False

    df_current = df.copy()
    df_current["run"] = "current"
    all_dataframes.append(df_current)

    if has_previous:
        print("Comparing current run with previous run")
    else:
        print("No previous.csv found, showing current data only")
        print(
            f"Tip: To compare runs, rename {current_csv_path} to {previous_csv_path} before next run"
        )

    combined_df = pd.concat(all_dataframes, ignore_index=True)

    block_sizes = combined_df["block_size"].unique()
    seq_lens = combined_df["seq_len_k"].unique()
    runs = ["previous", "current"] if has_previous else ["current"]

    n_cols = len(seq_lens)
    n_rows = len(block_sizes)

    plt.figure(figsize=(8 * n_cols, 5 * n_rows))

    colors = {"128": "red", "64": "green", "32": "blue", "16": "violet"}
    markers = {"128": "o", "64": "s", "32": "^", "16": "d"}

    subplot_idx = 1

    for block_size in sorted(block_sizes):
        df_config = combined_df[combined_df["block_size"] == block_size]

        if df_config.empty:
            continue

        for seq_len in sorted(seq_lens):
            df_seq = df_config[df_config["seq_len_k"] == seq_len]

            if df_seq.empty:
                continue

            plt.subplot(n_rows, n_cols, subplot_idx)

            all_batch_sizes_in_subplot = set()

            for run_name in runs:
                df_run = df_seq[df_seq["run"] == run_name]

                if df_run.empty:
                    continue

                if has_previous:
                    alpha = 1.0 if run_name == "current" else 0.6
                    linestyle = "-" if run_name == "current" else "--"
                else:
                    alpha = 1.0
                    linestyle = "-"

                for num_heads in [128, 64, 32, 16]:
                    df_heads = df_run[df_run["num_heads"] == num_heads]
                    if not df_heads.empty:
                        batch_sizes = df_heads["batch_size"].values
                        gbps_values = df_heads["GB/s"].values

                        all_batch_sizes_in_subplot.update(batch_sizes)

                        label = f"{num_heads} heads"
                        if has_previous:
                            label += f" ({run_name})"

                        plt.plot(
                            range(len(batch_sizes)),
                            gbps_values,
                            color=colors[str(num_heads)],
                            marker=markers[str(num_heads)],
                            label=label,
                            linewidth=2 if linestyle == "-" else 1,
                            linestyle=linestyle,
                            alpha=alpha,
                            markersize=6,
                        )

            plt.xlabel("Batch Size")
            plt.ylabel("GB/s")
            plt.title(f"block_size={block_size}, seq_len_k={seq_len}")

            actual_batch_sizes = sorted(list(all_batch_sizes_in_subplot))
            plt.xticks(range(len(actual_batch_sizes)), actual_batch_sizes)

            if subplot_idx == 1:
                plt.legend(loc="best", fontsize=8)
            plt.grid(True, alpha=0.3)

            subplot_idx += 1

    plt.tight_layout()

    if has_previous:
        plot_path = "bench_bmg_mla_prefill_res/flash_mla_prefill_current_vs_previous.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Comparison plot saved: {plot_path}")
    else:
        plot_path = "bench_bmg_mla_prefill_res/flash_mla_prefill_current_results.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Current results plot saved: {plot_path}")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--block-sizes",
        nargs="+",
        type=int,
        default=[16, 32, 64, 128],
        help="List of block sizes",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "incremental"],
        default="full",
        help="Prefill mode: full (seqlen_q==seqlen_k) or incremental (seqlen_q=16)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate CSV files and plots (default: only print results to console)",
    )
    args = parser.parse_args()

    all_results.clear()

    for block_size in args.block_sizes:
        print(f"\n{'='*60}")
        print(f"Running: block_size={block_size}, mode={args.mode}")
        print(f"{'='*60}")
        benchmark.run(
            print_data=False,
            show_plots=False,
            save_path=None,
            block_size=block_size,
            mode=args.mode,
        )

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80 + "\n")

    df = pd.DataFrame(all_results)
    df["GB/s"] = df["GB/s (median)"].astype(float)
    print(df.to_markdown(index=False))

    if args.plot:
        plot_data(df)
    else:
        print("\nBenchmark finished! (Use --plot flag to generate CSV files and plots)")
