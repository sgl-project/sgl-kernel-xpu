# SPDX-License-Identifier: Apache-2.0
"""Benchmark script for MXFP4 (E2M1) quantization with SiLU+Mul fusion on Intel XPU.

This benchmark compares three approaches:
1. Two-pass: SiLU+Mul (separate) → MXFP4 quantization
2. Fused: SiLU+Mul+MXFP4 quantization in single kernel
3. Unfused baseline: Direct MXFP4 quantization (no activation)

The fused kernel demonstrates bandwidth savings and latency reduction by eliminating
intermediate tensor materialization.
"""

import itertools
import os

import pandas as pd
import torch
import triton

MXFP4_BLOCK_SIZE = 32

IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)


def is_xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def silu_ref(x: torch.Tensor) -> torch.Tensor:
    """Reference SiLU implementation."""
    return x / (1.0 + torch.exp(-x))


def two_pass_silu_mul_quantize(
    gate: torch.Tensor, up: torch.Tensor, group_size: int, eps: float = 1e-10
) -> tuple:
    """Two-pass baseline: SiLU+Mul → MXFP4 quantization."""
    from sgl_kernel import sgl_per_token_group_quant_fp4

    # Pass 1: SiLU+Mul (materializes intermediate tensor)
    intermediate = torch.nn.functional.silu(gate) * up

    # Pass 2: MXFP4 quantization
    output_q, output_s = sgl_per_token_group_quant_fp4(
        intermediate, group_size=group_size, eps=eps
    )

    return output_q, output_s


def fused_silu_mul_quantize(
    gate: torch.Tensor, up: torch.Tensor, group_size: int, eps: float = 1e-10
) -> tuple:
    """Fused kernel: SiLU+Mul+MXFP4 quantization in single pass."""
    from sgl_kernel import sgl_per_token_group_quant_fp4

    output_q, output_s = sgl_per_token_group_quant_fp4(
        gate, group_size=group_size, eps=eps, x_secondary=up
    )

    return output_q, output_s


def unfused_baseline_quantize(
    x: torch.Tensor, group_size: int, eps: float = 1e-10
) -> tuple:
    """Unfused baseline: Direct MXFP4 quantization (no activation)."""
    from sgl_kernel import sgl_per_token_group_quant_fp4

    output_q, output_s = sgl_per_token_group_quant_fp4(
        x, group_size=group_size, eps=eps
    )

    return output_q, output_s


def calculate_bandwidth_metrics(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    src_dtype: torch.dtype,
    time_ms: float,
    mode: str = "two_pass",
) -> dict:
    """Calculate bandwidth metrics for different modes.

    Args:
        mode: "two_pass", "fused", or "unfused"
    """
    num_tokens = batch_size * seq_len
    num_elements = num_tokens * hidden_dim
    dtype_size = 2 if src_dtype in (torch.float16, torch.bfloat16) else 4

    if mode == "two_pass":
        # Read: gate, up (2x input)
        # Write: intermediate, output_q, output_s
        # Read again: intermediate for quantization
        input_bytes = 2 * num_elements * dtype_size  # gate + up
        intermediate_bytes = num_elements * dtype_size  # write + read
        output_bytes = (num_elements // 2) + (num_elements // 32)  # q + s
        total_bytes = input_bytes + 2 * intermediate_bytes + output_bytes

    elif mode == "fused":
        # Read: gate, up (2x input)
        # Write: output_q, output_s
        # No intermediate materialization
        input_bytes = 2 * num_elements * dtype_size  # gate + up
        output_bytes = (num_elements // 2) + (num_elements // 32)  # q + s
        total_bytes = input_bytes + output_bytes

    else:  # unfused
        # Read: input
        # Write: output_q, output_s
        input_bytes = num_elements * dtype_size
        output_bytes = (num_elements // 2) + (num_elements // 32)  # q + s
        total_bytes = input_bytes + output_bytes

    time_s = time_ms / 1000.0
    bandwidth_gbs = (total_bytes / 1e9) / time_s

    return {
        "num_tokens": num_tokens,
        "num_elements": num_elements,
        "total_bytes": total_bytes,
        "bandwidth_gbs": bandwidth_gbs,
        "time_ms": time_ms,
    }


def verify_correctness(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    group_size: int,
    src_dtype: torch.dtype,
    eps: float = 1e-10,
):
    """Verify that fused and two-pass produce similar results."""
    device = torch.device("xpu")

    gate = torch.randn(batch_size * seq_len, hidden_dim, device=device, dtype=src_dtype)
    up = torch.randn(batch_size * seq_len, hidden_dim, device=device, dtype=src_dtype)

    # Two-pass baseline
    q_2pass, s_2pass = two_pass_silu_mul_quantize(
        gate.clone(), up.clone(), group_size, eps
    )

    # Fused kernel
    q_fused, s_fused = fused_silu_mul_quantize(
        gate.clone(), up.clone(), group_size, eps
    )

    # Compare quantized values (allow small tolerance due to numerical differences)
    q_diff = (q_2pass.cpu() != q_fused.cpu()).sum().item()
    q_total = q_2pass.numel()
    q_match_ratio = 1.0 - (q_diff / q_total)

    # Compare scales (allow small tolerance)
    s_diff = (s_2pass.cpu() != s_fused.cpu()).sum().item()
    s_total = s_2pass.numel()
    s_match_ratio = 1.0 - (s_diff / s_total)

    # Allow up to 2% difference in quantized values and 5% in scales
    # due to numerical differences in SiLU computation (tanh-based vs exp-based)
    if q_match_ratio >= 0.98 and s_match_ratio >= 0.95:
        print(
            f"  ✅ Fused vs Two-pass match "
            f"(batch={batch_size}, seq={seq_len}, hidden={hidden_dim}, "
            f"q_match={q_match_ratio:.2%}, s_match={s_match_ratio:.2%})"
        )
    else:
        print(
            f"  ⚠️  Fused vs Two-pass differ "
            f"(q_match={q_match_ratio:.2%}, s_match={s_match_ratio:.2%})"
        )


# Configuration ranges
# Conservative settings to avoid device memory issues
batch_size_range = [1, 4, 16] if not IS_CI else [1, 8]
seq_len_range = [128, 512, 1024] if not IS_CI else [128, 512]
hidden_dim_range = [2048, 4096] if not IS_CI else [2048]
group_size_range = [32]  # Only 32 supported for MXFP4
src_dtype_range = [torch.bfloat16]

configs = list(
    itertools.product(
        batch_size_range,
        seq_len_range,
        hidden_dim_range,
        group_size_range,
        src_dtype_range,
    )
)

all_results = []


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "seq_len", "hidden_dim", "group_size", "src_dtype"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["two_pass", "fused", "unfused"],
        line_names=[
            "Two-Pass (SiLU+Mul → Quant)",
            "Fused (SiLU+Mul+Quant)",
            "Unfused (Quant only)",
        ],
        styles=[("red", "--"), ("green", "-"), ("blue", ":")],
        ylabel="us",
        plot_name="mxfp4-silu-mul-fusion-performance",
        args={},
    )
)
def benchmark(batch_size, seq_len, hidden_dim, group_size, src_dtype, provider):
    device = torch.device("xpu")

    try:
        # Create tensors only for the mode we need
        if provider in ["two_pass", "fused"]:
            gate = torch.randn(
                batch_size * seq_len, hidden_dim, device=device, dtype=src_dtype
            )
            up = torch.randn(
                batch_size * seq_len, hidden_dim, device=device, dtype=src_dtype
            )
            x = None
        else:  # unfused
            gate = None
            up = None
            x = torch.randn(
                batch_size * seq_len, hidden_dim, device=device, dtype=src_dtype
            )

        quantiles = [0.5, 0.2, 0.8]

        if provider == "two_pass":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: two_pass_silu_mul_quantize(gate, up, group_size),
                quantiles=quantiles,
            )
            mode = "two_pass"
        elif provider == "fused":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: fused_silu_mul_quantize(gate, up, group_size),
                quantiles=quantiles,
            )
            mode = "fused"
        else:  # unfused
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: unfused_baseline_quantize(x, group_size),
                quantiles=quantiles,
            )
            mode = "unfused"

    except RuntimeError as e:
        if "DEVICE_LOST" in str(e) or "out of memory" in str(e).lower():
            print(
                f"  ⚠️ Skipping config (batch={batch_size}, seq={seq_len}, hidden={hidden_dim}): {e}"
            )
            # Return a large time to indicate failure
            return 999999.0, 999999.0, 999999.0
        raise
    finally:
        # Clean up
        if gate is not None:
            del gate
        if up is not None:
            del up
        if x is not None:
            del x
        torch.xpu.empty_cache()

    metrics = calculate_bandwidth_metrics(
        batch_size, seq_len, hidden_dim, src_dtype, ms, mode
    )

    all_results.append(
        {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_tokens": metrics["num_tokens"],
            "hidden_dim": hidden_dim,
            "group_size": group_size,
            "src_dtype": str(src_dtype),
            "provider": provider,
            "time_us": 1000 * ms,
            "bandwidth_gbs": metrics["bandwidth_gbs"],
            "total_bytes_mb": metrics["total_bytes"] / 1e6,
        }
    )

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


def print_summary(results: list):
    """Print summary statistics from benchmark results."""
    print("\n" + "=" * 120)
    print("MXFP4 SiLU+Mul Fusion Benchmark Results")
    print("=" * 120)

    df = pd.DataFrame(results)
    df["bandwidth_gbs"] = df["bandwidth_gbs"].round(2)
    df["total_bytes_mb"] = df["total_bytes_mb"].round(2)
    df["time_us"] = df["time_us"].round(2)

    print("\nDetailed Results:")
    print(df.to_markdown(index=False))

    print("\n" + "=" * 120)
    print("Summary Statistics by Provider")
    print("=" * 120)
    summary = df.groupby("provider").agg(
        {
            "bandwidth_gbs": ["mean", "min", "max"],
            "time_us": ["mean", "min", "max"],
        }
    )
    print(summary.to_markdown())

    # Calculate speedup statistics
    print("\n" + "=" * 120)
    print("Speedup Analysis (Fused vs Two-Pass)")
    print("=" * 120)

    # Group by configuration and calculate speedup
    df_pivot = df.pivot_table(
        index=["batch_size", "seq_len", "hidden_dim"],
        columns="provider",
        values="time_us",
    )

    if "two_pass" in df_pivot.columns and "fused" in df_pivot.columns:
        df_pivot["speedup"] = df_pivot["two_pass"] / df_pivot["fused"]
        df_pivot["bandwidth_saved_pct"] = (
            (df_pivot["two_pass"] - df_pivot["fused"]) / df_pivot["two_pass"] * 100
        )

        print("\nSpeedup by Configuration:")
        print(
            df_pivot[["two_pass", "fused", "speedup", "bandwidth_saved_pct"]]
            .round(2)
            .to_markdown()
        )

        avg_speedup = df_pivot["speedup"].mean()
        max_speedup = df_pivot["speedup"].max()
        min_speedup = df_pivot["speedup"].min()

        print(f"\nAverage Speedup: {avg_speedup:.2f}x")
        print(f"Max Speedup: {max_speedup:.2f}x")
        print(f"Min Speedup: {min_speedup:.2f}x")

        avg_bw_saved = df_pivot["bandwidth_saved_pct"].mean()
        print(f"Average Bandwidth Saved: {avg_bw_saved:.1f}%")


def main():
    if not is_xpu_available():
        print("Error: Intel XPU not available")
        return

    try:
        from sgl_kernel import sgl_per_token_group_quant_fp4

        assert callable(sgl_per_token_group_quant_fp4)
    except ImportError:
        print("Error: sgl_per_token_group_quant_fp4 kernel not available")
        return

    print("Running MXFP4 SiLU+Mul Fusion Benchmark")
    print("  Device: Intel XPU")
    print(f"  MXFP4 block size: {MXFP4_BLOCK_SIZE}")
    print("\nBenchmarking three approaches:")
    print("  1. Two-Pass: SiLU+Mul (separate kernel) → MXFP4 quantization")
    print("  2. Fused: SiLU+Mul+MXFP4 quantization (single kernel)")
    print("  3. Unfused: Direct MXFP4 quantization (baseline, no activation)")

    print("\n" + "=" * 80)
    print("Correctness Verification (Fused vs Two-Pass)")
    print("=" * 80)

    verify_correctness(
        batch_size=8,
        seq_len=128,
        hidden_dim=2048,
        group_size=32,
        src_dtype=torch.bfloat16,
    )

    verify_correctness(
        batch_size=16,
        seq_len=512,
        hidden_dim=4096,
        group_size=32,
        src_dtype=torch.bfloat16,
    )

    print("\n" + "=" * 80)
    print("Performance Benchmark")
    print("=" * 80)
    benchmark.run(print_data=True, save_path="./")

    print_summary(all_results)

    # Print key insights
    print("\n" + "=" * 120)
    print("Key Insights")
    print("=" * 120)
    print(
        "1. Fused kernel eliminates intermediate tensor materialization (~4-8 MB per 1024 tokens)"
    )
    print("2. Memory bandwidth savings: ~40-50% compared to two-pass approach")
    print(
        "3. Expected speedup: 1.3-1.8x for typical MoE expert sizes (1024-4096 hidden dim)"
    )
    print("4. Unfused baseline shows pure quantization cost (no SiLU overhead)")
    print(
        "\nUse fused kernel for MoE models (DeepSeek-V3, Mixtral) to reduce expert activation overhead."
    )


if __name__ == "__main__":
    main()
