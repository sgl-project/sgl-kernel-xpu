import itertools
from typing import Optional, Tuple

import pandas as pd
import torch
import triton
import triton.language as tl
from sgl_kernel import merge_state_v2

all_results = []


@triton.jit
def merge_state_kernel(
    output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE] v_merged
    output_lse,  # [NUM_TOKENS, NUM_HEADS] s_merged
    prefix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE] v_a
    prefix_lse,  # [NUM_TOKENS, NUM_HEADS] s_a
    suffix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE] v_b
    suffix_lse,  # [NUM_TOKENS, NUM_HEADS] s_b
    HEAD_SIZE: tl.constexpr,
    PADDED_HEAD_SIZE: tl.constexpr,
    OUTPUT_LSE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    num_tokens = tl.num_programs(0)
    head_idx = tl.program_id(1)
    num_heads = tl.num_programs(1)

    p_lse = tl.load(prefix_lse + token_idx * num_heads + head_idx)
    s_lse = tl.load(suffix_lse + token_idx * num_heads + head_idx)
    p_lse = float("-inf") if p_lse == float("inf") else p_lse
    s_lse = float("-inf") if s_lse == float("inf") else s_lse

    max_lse = tl.maximum(p_lse, s_lse)
    p_lse = p_lse - max_lse
    s_lse = s_lse - max_lse
    out_se = tl.exp(p_lse) + tl.exp(s_lse)

    if OUTPUT_LSE:
        out_lse = tl.log(out_se) + max_lse
        tl.store(output_lse + token_idx * num_heads + head_idx, out_lse)

    head_arange = tl.arange(0, PADDED_HEAD_SIZE)
    head_mask = head_arange < HEAD_SIZE
    p_out = tl.load(
        prefix_output
        + token_idx * num_heads * HEAD_SIZE
        + head_idx * HEAD_SIZE
        + head_arange,
        mask=head_mask,
    )
    s_out = tl.load(
        suffix_output
        + token_idx * num_heads * HEAD_SIZE
        + head_idx * HEAD_SIZE
        + head_arange,
        mask=head_mask,
    )

    p_scale = tl.exp(p_lse) / out_se
    s_scale = tl.exp(s_lse) / out_se
    out = p_out * p_scale + s_out * s_scale
    tl.store(
        output + token_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE + head_arange,
        out,
        mask=head_mask,
    )


def triton_merge_state(
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    output_lse: Optional[torch.Tensor] = None,
) -> None:
    num_tokens = output.shape[0]
    num_query_heads = output.shape[1]
    head_size = output.shape[2]
    padded_head_size = triton.next_power_of_2(head_size)
    # Avoid creating new tensors if they are already provided
    if output is None:
        output = torch.empty_like(prefix_output)
    if output_lse is None:
        output_lse = torch.empty_like(prefix_lse)

    merge_state_kernel[(num_tokens, num_query_heads)](
        output,
        output_lse,
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
        head_size,
        padded_head_size,
        output_lse is not None,
    )
    return output, output_lse


def sglang_merge_state(
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    output_lse: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Avoid creating new tensors if they are already provided
    if output is None:
        output = torch.empty_like(prefix_output)
    if output_lse is None:
        output_lse = torch.empty_like(prefix_lse)

    merge_state_v2(
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
        output,
        output_lse,
    )
    return output, output_lse


def calculate_diff(num_tokens, num_heads, head_size, dtype):
    device = torch.device("xpu")

    # Create test tensors with some inf values
    prefix_lse = torch.randn(num_tokens, num_heads, dtype=torch.float32, device=device)
    suffix_lse = torch.randn(num_tokens, num_heads, dtype=torch.float32, device=device)

    # Generate boolean masks to add some inf values
    mask_prefix = torch.rand(num_tokens, num_heads, device=device) < 0.1
    mask_suffix = torch.rand(num_tokens, num_heads, device=device) < 0.1
    # Ensure that the same position is not True at the same time
    combined_mask = torch.logical_and(mask_prefix, mask_suffix)
    mask_prefix = torch.logical_and(mask_prefix, ~combined_mask)
    mask_suffix = torch.logical_and(mask_suffix, ~combined_mask)

    prefix_lse[mask_prefix] = float("inf")
    suffix_lse[mask_suffix] = float("inf")

    # Create output tensors
    prefix_output = torch.randn(
        num_tokens, num_heads, head_size, dtype=dtype, device=device
    )
    suffix_output = torch.randn(
        num_tokens, num_heads, head_size, dtype=dtype, device=device
    )

    output_triton = torch.empty_like(prefix_output)
    output_lse_triton = torch.empty_like(prefix_lse)

    output_sglang = torch.empty_like(prefix_output)
    output_lse_sglang = torch.empty_like(prefix_lse)

    output_triton, output_lse_triton = triton_merge_state(
        prefix_output.clone(),
        prefix_lse.clone(),
        suffix_output.clone(),
        suffix_lse.clone(),
        output_triton,
        output_lse_triton,
    )

    output_sglang, output_lse_sglang = sglang_merge_state(
        prefix_output.clone(),
        prefix_lse.clone(),
        suffix_output.clone(),
        suffix_lse.clone(),
        output_sglang,
        output_lse_sglang,
    )

    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-3
    if torch.allclose(
        output_triton.to(torch.float32),
        output_sglang.to(torch.float32),
        rtol=rtol,
        atol=1e-3,
    ) and torch.allclose(output_lse_triton, output_lse_sglang, rtol=rtol, atol=1e-3):
        print(f"✅ {dtype} implementations match")
    else:
        print(f"❌ {dtype} implementations differ")
        max_diff = torch.max(torch.abs(output_triton.float() - output_sglang.float()))
        max_lse_diff = torch.max(torch.abs(output_lse_triton - output_lse_sglang))
        print(f"   Max output diff: {max_diff:.6f}")
        print(f"   Max LSE diff: {max_lse_diff:.6f}")


num_tokens_range = [256, 512, 1024]
num_heads_range = [8, 16, 32]
head_size_range = [32, 48, 64, 128]
dtype_range = [torch.float16, torch.bfloat16]

configs = list(
    itertools.product(num_tokens_range, num_heads_range, head_size_range, dtype_range)
)


def calculate_bandwidth(num_tokens, num_heads, head_size, dtype, time_ms):
    """
    Calculate approximate effective bandwidth for merge_state_v2.

    Memory access pattern:
    - Read: prefix_output, suffix_output (2 tensors of [num_tokens, num_heads, head_size])
    - Read: prefix_lse, suffix_lse (2 tensors of [num_tokens, num_heads])
    - Write: output (1 tensor of [num_tokens, num_heads, head_size])
    - Write: output_lse (1 tensor of [num_tokens, num_heads])
    """
    dtype_size = 2 if dtype in [torch.float16, torch.bfloat16] else 4
    lse_size = 4  # float32

    # Read 2 outputs + 2 lse tensors
    bytes_read = 2 * (num_tokens * num_heads * head_size * dtype_size) + 2 * (
        num_tokens * num_heads * lse_size
    )
    # Write 1 output + 1 lse tensor
    bytes_write = (num_tokens * num_heads * head_size * dtype_size) + (
        num_tokens * num_heads * lse_size
    )

    total_bytes = bytes_read + bytes_write
    time_s = time_ms / 1000.0
    return (total_bytes / 1e9) / time_s


def benchmark_kernel(num_tokens, num_heads, head_size, dtype, provider):
    """Benchmark a single kernel configuration using XPU events for accurate timing."""
    device = torch.device("xpu")
    warmup_times = 2
    repeat_times = 20

    # Create test tensors
    prefix_lse = torch.randn(num_tokens, num_heads, dtype=torch.float32, device=device)
    suffix_lse = torch.randn(num_tokens, num_heads, dtype=torch.float32, device=device)

    # Add some inf values
    mask_prefix = torch.rand(num_tokens, num_heads, device=device) < 0.1
    mask_suffix = torch.rand(num_tokens, num_heads, device=device) < 0.1
    combined_mask = torch.logical_and(mask_prefix, mask_suffix)
    mask_prefix = torch.logical_and(mask_prefix, ~combined_mask)
    mask_suffix = torch.logical_and(mask_suffix, ~combined_mask)
    prefix_lse[mask_prefix] = float("inf")
    suffix_lse[mask_suffix] = float("inf")

    prefix_output = torch.randn(
        num_tokens, num_heads, head_size, dtype=dtype, device=device
    )
    suffix_output = torch.randn(
        num_tokens, num_heads, head_size, dtype=dtype, device=device
    )

    output = torch.empty_like(prefix_output)
    output_lse = torch.empty_like(prefix_lse)

    if provider == "triton":
        kernel_fn = lambda: triton_merge_state(
            prefix_output, prefix_lse, suffix_output, suffix_lse, output, output_lse
        )
    elif provider == "sglang":
        kernel_fn = lambda: sglang_merge_state(
            prefix_output, prefix_lse, suffix_output, suffix_lse, output, output_lse
        )

    # Warmup
    for _ in range(warmup_times):
        kernel_fn()
    torch.xpu.synchronize()

    # Benchmark
    total_time = 0
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)

    for _ in range(repeat_times):
        start.record()
        kernel_fn()
        end.record()
        torch.xpu.synchronize()
        total_time += start.elapsed_time(end)

    avg_time_ms = total_time / repeat_times

    # Calculate bandwidth
    bandwidth_gbs = calculate_bandwidth(
        num_tokens, num_heads, head_size, dtype, avg_time_ms
    )

    return avg_time_ms, bandwidth_gbs


if __name__ == "__main__":
    # Test correctness first
    print("Testing correctness...")
    calculate_diff(num_tokens=512, num_heads=16, head_size=128, dtype=torch.float16)
    calculate_diff(num_tokens=512, num_heads=16, head_size=128, dtype=torch.bfloat16)
    print()

    # Run benchmarks
    print("Running benchmarks...")
    print("=" * 100)

    for config in configs:
        num_tokens, num_heads, head_size, dtype = config

        # Benchmark both providers
        for provider in ["triton", "sglang"]:
            time_ms, bandwidth_gbs = benchmark_kernel(
                num_tokens, num_heads, head_size, dtype, provider
            )

            all_results.append(
                {
                    "num_tokens": num_tokens,
                    "num_heads": num_heads,
                    "head_size": head_size,
                    "dtype": str(dtype).replace("torch.", ""),
                    "provider": provider,
                    "time_ms": time_ms,
                    "bandwidth_gbs": bandwidth_gbs,
                }
            )

            print(
                f"{provider:8s} | tokens={num_tokens:4d} heads={num_heads:2d} "
                f"head_size={head_size:3d} dtype={str(dtype).replace('torch.', ''):8s} | "
                f"{time_ms:7.4f}ms | {bandwidth_gbs:6.2f} GB/s"
            )

    print("=" * 100)

    # Print bandwidth results
    print("\n" + "=" * 80)
    print("Effective Bandwidth Results")
    print("=" * 80)
    df = pd.DataFrame(all_results)
    df["bandwidth_gbs"] = df["bandwidth_gbs"].round(2)
    df["time_ms"] = df["time_ms"].round(4)
    print(df.to_markdown(index=False))

    # Summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics by Provider")
    print("=" * 80)
    summary = df.groupby("provider").agg(
        {
            "time_ms": ["mean", "min", "max"],
            "bandwidth_gbs": ["mean", "min", "max"],
        }
    )
    print(summary.to_markdown())

    # Speedup comparison
    print("\n" + "=" * 80)
    print("Speedup: SGL Kernel vs Triton")
    print("=" * 80)

    # Pivot to compare providers side by side
    df_pivot = df.pivot_table(
        index=["num_tokens", "num_heads", "head_size", "dtype"],
        columns="provider",
        values="time_ms",
    )
    df_pivot["speedup"] = df_pivot["triton"] / df_pivot["sglang"]
    df_pivot = df_pivot.round(4)
    print(df_pivot.to_markdown())

    print(f"\nAverage speedup: {df_pivot['speedup'].mean():.4f}x")
    print(f"Min speedup: {df_pivot['speedup'].min():.4f}x")
    print(f"Max speedup: {df_pivot['speedup'].max():.4f}x")

    # Summary by dtype
    print("\n" + "=" * 80)
    print("Summary Statistics by Data Type")
    print("=" * 80)
    dtype_summary = df.groupby(["dtype", "provider"]).agg(
        {
            "time_ms": "mean",
            "bandwidth_gbs": "mean",
        }
    )
    print(dtype_summary.to_markdown())
