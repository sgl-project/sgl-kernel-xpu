import argparse
import itertools
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import triton
from sgl_kernel import cutlass_mla_decode, cutlass_mla_get_workspace_size

bs_range = [1, 4, 16]
qlen_range = [1024, 2048, 4096, 8192]

configs = list(itertools.product(bs_range, qlen_range))

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
        plot_name="cutlass mla",
        args={},
    )
)
def benchmark(batch_size, seq_len, provider, block_size, num_kv_splits):
    d = 576
    dv = 512

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

    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device="xpu")
    max_seq_len = seq_lens.max().item()
    block_num = (max_seq_len + block_size - 1) // block_size

    # Pad block_num so that small blocks can be packed into full 128-sized CUTLASS tiles.
    # One 128-wide tile can hold (128 // block_size) small blocks.
    pack_factor = 128 // block_size
    block_num = ((block_num + pack_factor - 1) // pack_factor) * pack_factor

    q = torch.randn(batch_size, h_q, d, dtype=torch.bfloat16, device="xpu") * 100.0
    block_table = torch.randint(
        0,
        batch_size * block_num,
        (batch_size, block_num),
        dtype=torch.int32,
        device="xpu",
    )

    kv_cache = torch.randn(
        block_table.numel(), block_size, d, dtype=torch.bfloat16, device="xpu"
    )
    q_nope = torch.empty(
        (h_q, batch_size, dv), device="xpu", dtype=torch.bfloat16
    ).transpose(0, 1)
    q_nope.copy_(q[:, :, :dv])
    q_pe = q[:, :, dv:].clone()

    workspace_size = cutlass_mla_get_workspace_size(
        block_num * block_size, batch_size, num_kv_splits=num_kv_splits
    )
    workspace = torch.empty(workspace_size, device="xpu", dtype=torch.uint8)
    scale = (512 + 64) ** (-0.5)
    quantiles = [0.5, 0.25, 0.75]
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: cutlass_mla_decode(
            q_nope,
            q_pe,
            kv_cache,
            seq_lens,
            block_table,
            workspace,
            scale,
            num_kv_splits,
        ),
        quantiles=quantiles,
    )

    # Calculate total bytes transferred for bandwidth calculation
    total_bytes = (
        q.numel() * q.element_size()  # read q_nope+q_pe
        + kv_cache.numel() * kv_cache.element_size()  # read kv_cache
        + block_table.numel() * block_table.element_size()  # read page_table
        + seq_lens.numel() * seq_lens.element_size()  # read seq_lens
        + q_nope.numel() * q_nope.element_size()  # write output
    )

    gbps = lambda ms: total_bytes * 1e-9 / (ms * 1e-3)

    # Collect results for final summary table
    all_results.append(
        {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_heads": h_q,
            "block_size": block_size,
            "num_kv_splits": num_kv_splits,
            "time_ms": f"{ms:.3f}",
            "GB/s (median)": f"{gbps(ms):.2f}",
            "GB/s (min)": f"{gbps(max_ms):.2f}",
            "GB/s (max)": f"{gbps(min_ms):.2f}",
        }
    )

    result = (gbps(ms), gbps(max_ms), gbps(min_ms))
    del q, q_nope, q_pe, kv_cache, block_table, workspace, seq_lens
    return result


def plot_data(df):

    os.makedirs("bench_bmg_mla_res", exist_ok=True)

    # Define file paths
    previous_csv_path = "bench_bmg_mla_res/previous.csv"
    current_csv_path = "bench_bmg_mla_res/current.csv"

    # Save current results
    df.to_csv(current_csv_path, index=False)
    print(f"Current results saved to: {current_csv_path}")

    # Check if previous.csv exists for comparison
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

    # Add current data
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

    # Combine all data
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    # Create custom plots with comparison data (current vs previous)
    if has_previous:
        print("\nGenerating comparison plot (current vs previous)...")
    else:
        print("\nGenerating single-run plot...")

    # Get unique configurations
    block_sizes = combined_df["block_size"].unique()
    seq_lens = combined_df["seq_len"].unique()
    runs = ["previous", "current"] if has_previous else ["current"]

    # Calculate subplot grid size
    n_block_sizes = len(block_sizes)
    n_seq_lens = len(seq_lens)
    n_cols = n_seq_lens
    n_rows = n_block_sizes

    plt.figure(figsize=(8 * n_cols, 5 * n_rows))

    colors = {"128": "red", "64": "green", "32": "blue", "16": "violet"}
    markers = {"128": "o", "64": "s", "32": "^", "16": "d"}

    subplot_idx = 1

    for block_size in sorted(block_sizes):
        kv_split = -1  # Assuming single kv_split value
        df_config = combined_df[
            (combined_df["block_size"] == block_size)
            & (combined_df["num_kv_splits"] == kv_split)
        ]

        if df_config.empty:
            continue

        for seq_len in sorted(seq_lens):
            df_seq = df_config[df_config["seq_len"] == seq_len]

            if df_seq.empty:
                continue

            plt.subplot(n_rows, n_cols, subplot_idx)

            # Get all unique batch sizes for this specific subplot to set proper x-axis
            all_batch_sizes_in_subplot = set()

            # Plot each run with different styling
            for run_name in runs:
                df_run = df_seq[df_seq["run"] == run_name]

                if df_run.empty:
                    continue

                # Use different alpha and line styles for different runs
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

                        # Collect all batch sizes for this subplot
                        all_batch_sizes_in_subplot.update(batch_sizes)

                        # Create label
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
            plt.title(f"block_size={block_size}, seq_len={seq_len}")

            # Use only the actual batch sizes present in this subplot for x-axis labels
            actual_batch_sizes = sorted(list(all_batch_sizes_in_subplot))
            plt.xticks(range(len(actual_batch_sizes)), actual_batch_sizes)

            if subplot_idx == 1:  # Only show legend on first subplot
                plt.legend(loc="best", fontsize=8)
            plt.grid(True, alpha=0.3)

            subplot_idx += 1

    plt.tight_layout()

    # Use appropriate filename based on whether we have comparison data
    if has_previous:
        plot_path = "bench_bmg_mla_res/cutlass_mla_current_vs_previous.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Comparison plot saved: {plot_path}")
        print("\nPlot compares:")
        print(f"  - Previous: {previous_csv_path}")
        print(f"  - Current:  {current_csv_path}")
    else:
        plot_path = "bench_bmg_mla_res/cutlass_mla_current_results.png"
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
        "--num-kv-splits",
        nargs="+",
        type=int,
        default=[-1],
        help="List of num_kv_splits",
    )
    # How it works:
    # First run (no previous.csv): Creates current.csv and plots the current results only
    # Subsequent runs (with previous.csv): Compares against previous.csv and plots both
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate CSV files and plots (default: only print results to console)",
    )
    args = parser.parse_args()

    # Clear previous results
    all_results.clear()

    for block_size in args.block_sizes:
        for kv_split in args.num_kv_splits:
            print(f"\n{'='*60}")
            print(f"Running: block_size={block_size}, num_kv_splits={kv_split}")
            print(f"{'='*60}")
            benchmark.run(
                print_data=False,
                show_plots=False,
                save_path=None,  # Disable triton's plotting, we'll create custom plots
                block_size=block_size,
                num_kv_splits=kv_split,
            )

    # Print final summary table
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80 + "\n")

    df = pd.DataFrame(all_results)
    # Convert string columns to float for plotting
    df["GB/s"] = df["GB/s (median)"].astype(float)
    print(df.to_markdown(index=False))

    if args.plot:
        plot_data(df)
    else:
        print("\nBenchmark finished! (Use --plot flag to generate CSV files and plots)")
