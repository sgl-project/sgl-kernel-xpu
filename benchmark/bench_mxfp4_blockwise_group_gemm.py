# SPDX-License-Identifier: Apache-2.0
"""Benchmark script for MXFP4 (E2M1) Block-Scaled Grouped GEMM for MoE on Intel XPU."""

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
import torch

IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)

MXFP4_BLOCK_SIZE = 32


def is_xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def quantize_to_e2m1(tensor: torch.Tensor) -> torch.Tensor:
    """Quantize tensor values to E2M1 format (4-bit indices)."""
    e2m1_values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=tensor.dtype,
        device=tensor.device,
    )
    sign = (tensor < 0).to(torch.uint8)
    abs_val = torch.clamp(tensor.abs(), max=6.0)
    abs_val_expanded = abs_val.unsqueeze(-1)
    e2m1_expanded = e2m1_values.view(*([1] * abs_val.dim()), -1)
    distances = (abs_val_expanded - e2m1_expanded).abs()
    indices = distances.argmin(dim=-1).to(torch.uint8)
    return (sign << 3) | indices


def pack_fp4(tensor: torch.Tensor) -> torch.Tensor:
    """Pack two 4-bit values into one uint8."""
    assert tensor.shape[-1] % 2 == 0
    shape = tensor.shape[:-1] + (tensor.shape[-1] // 2, 2)
    paired = tensor.reshape(shape)
    packed = (paired[..., 0] & 0x0F) | ((paired[..., 1] & 0x0F) << 4)
    return packed.to(torch.uint8)


def quantize_to_mxfp4(
    tensor: torch.Tensor, block_size: int = MXFP4_BLOCK_SIZE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to MXFP4 format."""
    assert tensor.dim() == 2
    m, k = tensor.shape
    assert k % block_size == 0
    assert k % 2 == 0

    tensor_fp32 = tensor.float()
    num_blocks = k // block_size
    tensor_blocks = tensor_fp32.reshape(m, num_blocks, block_size)

    block_max = tensor_blocks.abs().max(dim=-1, keepdim=True).values
    block_max = torch.clamp(block_max, min=1e-12)

    log2_max = torch.log2(block_max / 6.0)
    exponent = torch.ceil(log2_max).clamp(min=-127, max=127).to(torch.int32)
    scales_ue8m0 = (exponent + 127).to(torch.uint8).squeeze(-1)

    scale_values = torch.pow(2.0, exponent.float())
    scaled_tensor = tensor_blocks / scale_values
    quantized_blocks = quantize_to_e2m1(scaled_tensor)
    quantized = quantized_blocks.reshape(m, k)
    packed = pack_fp4(quantized)

    return packed, scales_ue8m0


def ensure_contiguous_layout(tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.is_contiguous():
        return tensor.contiguous()
    return tensor


def create_random_mxfp4_data(m: int, k: int, device: str, seed: int = 42):
    """Create random MXFP4 quantized data."""
    torch.manual_seed(seed)
    original = torch.randn(m, k, dtype=torch.float32, device="cpu") * 2.0
    packed, scales = quantize_to_mxfp4(original)
    return packed.to(device), scales.to(device)


def prepare_kernel_inputs(
    a_list: list,
    b_list: list,
    scales_a_list: list,
    scales_b_list: list,
    device: str,
):
    """Prepare inputs for the mxfp4_blockwise_scaled_grouped_mm kernel."""
    num_experts = len(a_list)
    m, packed_k = a_list[0].shape
    k = packed_k * 2
    n = b_list[0].shape[0]

    a_stack = torch.stack(
        [ensure_contiguous_layout(a) for a in a_list], dim=0
    ).contiguous()
    b_stack = torch.stack(
        [ensure_contiguous_layout(b) for b in b_list], dim=0
    ).contiguous()
    scales_a_stack = torch.stack(
        [ensure_contiguous_layout(s.t().contiguous()) for s in scales_a_list], dim=0
    ).contiguous()
    scales_b_stack = torch.stack(
        [ensure_contiguous_layout(s.t().contiguous()) for s in scales_b_list], dim=0
    ).contiguous()

    output = torch.zeros((num_experts, m, n), dtype=torch.float32, device=device)

    a_ptrs = torch.tensor(
        [a_stack[i].data_ptr() for i in range(num_experts)],
        dtype=torch.uint64,
        device=device,
    )
    b_ptrs = torch.tensor(
        [b_stack[i].data_ptr() for i in range(num_experts)],
        dtype=torch.uint64,
        device=device,
    )
    out_ptrs = torch.tensor(
        [output[i].data_ptr() for i in range(num_experts)],
        dtype=torch.uint64,
        device=device,
    )
    a_scales_ptrs = torch.tensor(
        [scales_a_stack[i].data_ptr() for i in range(num_experts)],
        dtype=torch.uint64,
        device=device,
    )
    b_scales_ptrs = torch.tensor(
        [scales_b_stack[i].data_ptr() for i in range(num_experts)],
        dtype=torch.uint64,
        device=device,
    )

    stride_a = torch.tensor(
        [[k, 1, 0] for _ in range(num_experts)], dtype=torch.int64, device=device
    )
    stride_b = torch.tensor(
        [[k, 1, 0] for _ in range(num_experts)], dtype=torch.int64, device=device
    )
    stride_c = torch.tensor(
        [[n, 1, 0] for _ in range(num_experts)], dtype=torch.int64, device=device
    )
    layout_sfa = torch.tensor(
        [[1, m, 1] for _ in range(num_experts)], dtype=torch.int64, device=device
    )
    layout_sfb = torch.tensor(
        [[1, n, 1] for _ in range(num_experts)], dtype=torch.int64, device=device
    )

    problem_sizes = torch.tensor(
        [[m, n, k] for _ in range(num_experts)], dtype=torch.int32, device=device
    )
    expert_offsets = torch.arange(num_experts, dtype=torch.int32, device=device)
    workspace = torch.empty((1024 * 1024 * 1024,), dtype=torch.uint8, device=device)

    return {
        "output": output,
        "a_ptrs": a_ptrs,
        "b_ptrs": b_ptrs,
        "out_ptrs": out_ptrs,
        "a_scales_ptrs": a_scales_ptrs,
        "b_scales_ptrs": b_scales_ptrs,
        "a_stack": a_stack,
        "b_stack": b_stack,
        "scales_a_stack": scales_a_stack,
        "scales_b_stack": scales_b_stack,
        "stride_a": stride_a,
        "stride_b": stride_b,
        "stride_c": stride_c,
        "layout_sfa": layout_sfa,
        "layout_sfb": layout_sfb,
        "problem_sizes": problem_sizes,
        "expert_offsets": expert_offsets,
        "workspace": workspace,
        "m": m,
        "n": n,
        "k": k,
    }


def calculate_flops(m: int, n: int, k: int, num_groups: int) -> int:
    """Calculate FLOPs for grouped GEMM: num_groups * 2 * M * N * K."""
    return num_groups * 2 * m * n * k


def calculate_memory_bytes(
    m: int, n: int, k: int, num_groups: int, block_size: int = MXFP4_BLOCK_SIZE
) -> dict:
    """Calculate memory bytes for MXFP4 grouped GEMM."""
    k_packed = k // 2
    scale_k = k // block_size

    a_bytes = num_groups * m * k_packed
    b_bytes = num_groups * n * k_packed
    scales_a_bytes = num_groups * scale_k * m
    scales_b_bytes = num_groups * scale_k * n
    output_bytes = num_groups * m * n * 4

    total_read_bytes = a_bytes + b_bytes + scales_a_bytes + scales_b_bytes
    total_write_bytes = output_bytes

    return {
        "total_read_bytes": total_read_bytes,
        "total_write_bytes": total_write_bytes,
        "total_bytes": total_read_bytes + total_write_bytes,
    }


def calculate_metrics(m: int, n: int, k: int, num_groups: int, time_us: float) -> dict:
    """Calculate effective bandwidth and FLOPS metrics."""
    time_s = time_us / 1e6

    total_flops = calculate_flops(m, n, k, num_groups)
    gflops = (total_flops / 1e9) / time_s
    tflops = (total_flops / 1e12) / time_s

    mem_bytes = calculate_memory_bytes(m, n, k, num_groups)
    bandwidth_gbs = (mem_bytes["total_bytes"] / 1e9) / time_s

    return {
        "total_flops": total_flops,
        "gflops": gflops,
        "tflops": tflops,
        "total_bytes_mb": mem_bytes["total_bytes"] / 1e6,
        "bandwidth_gbs": bandwidth_gbs,
    }


@dataclass
class ShapeArg:
    """Shape configuration for benchmark."""

    expected_m_per_group: int
    n: int
    k: int
    num_groups: int


def construct_mxfp4_grouped_data(
    num_groups: int, m: int, k: int, n: int, device: str
) -> Tuple[list, list, list, list]:
    """Construct MXFP4 quantized data for grouped GEMM benchmark."""
    a_list, b_list, scales_a_list, scales_b_list = [], [], [], []

    for i in range(num_groups):
        a_packed, scales_a = create_random_mxfp4_data(m, k, device, seed=42 + i)
        b_packed, scales_b = create_random_mxfp4_data(n, k, device, seed=100 + i)

        a_list.append(ensure_contiguous_layout(a_packed))
        b_list.append(ensure_contiguous_layout(b_packed))
        scales_a_list.append(ensure_contiguous_layout(scales_a))
        scales_b_list.append(ensure_contiguous_layout(scales_b))

    return a_list, b_list, scales_a_list, scales_b_list


def bench_mxfp4_cutlass(
    expected_m_per_group: int,
    n: int,
    k: int,
    num_groups: int,
    num_warmup: int,
    num_run: int,
) -> Tuple[float, int, int]:
    """Benchmark the MXFP4 blockwise scaled grouped MM kernel."""
    from sgl_kernel import mxfp4_blockwise_scaled_grouped_mm

    device = "xpu"
    alignment = 64
    m = ceil_div(expected_m_per_group, alignment) * alignment
    k_aligned = ceil_div(k, MXFP4_BLOCK_SIZE) * MXFP4_BLOCK_SIZE

    a_list, b_list, scales_a_list, scales_b_list = construct_mxfp4_grouped_data(
        num_groups, m, k_aligned, n, device
    )
    inputs = prepare_kernel_inputs(a_list, b_list, scales_a_list, scales_b_list, device)

    def run_kernel():
        mxfp4_blockwise_scaled_grouped_mm(
            inputs["output"],
            inputs["a_ptrs"],
            inputs["b_ptrs"],
            inputs["out_ptrs"],
            inputs["a_scales_ptrs"],
            inputs["b_scales_ptrs"],
            inputs["a_stack"],
            inputs["b_stack"],
            inputs["scales_a_stack"],
            inputs["scales_b_stack"],
            inputs["stride_a"],
            inputs["stride_b"],
            inputs["stride_c"],
            inputs["layout_sfa"],
            inputs["layout_sfb"],
            inputs["problem_sizes"],
            inputs["expert_offsets"],
            inputs["workspace"],
        )

    for _ in range(num_warmup):
        run_kernel()
    torch.xpu.synchronize()

    start_event = torch.xpu.Event(enable_timing=True)
    end_event = torch.xpu.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_run):
        run_kernel()
    end_event.record()
    end_event.synchronize()
    torch.xpu.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    avg_us = (elapsed_ms / num_run) * 1000

    return avg_us, m, k_aligned


def benchmark_one_shape(
    shape_args: List[ShapeArg], num_warmup: int, num_run: int
) -> List[dict]:
    """Run benchmark for a list of shapes and collect results."""
    all_results = []

    for shape in shape_args:
        print(
            f"\nBenchmark: expected_m_per_group={shape.expected_m_per_group}, "
            f"n={shape.n}, k={shape.k}, num_groups={shape.num_groups}"
        )

        try:
            avg_time_us, actual_m, actual_k = bench_mxfp4_cutlass(
                shape.expected_m_per_group,
                shape.n,
                shape.k,
                shape.num_groups,
                num_warmup,
                num_run,
            )

            metrics = calculate_metrics(
                actual_m, shape.n, actual_k, shape.num_groups, avg_time_us
            )

            result = {
                "expected_m": shape.expected_m_per_group,
                "actual_m": actual_m,
                "n": shape.n,
                "k": shape.k,
                "actual_k": actual_k,
                "num_groups": shape.num_groups,
                "time_us": avg_time_us,
                "bandwidth_gbs": metrics["bandwidth_gbs"],
                "total_bytes_mb": metrics["total_bytes_mb"],
                "gflops": metrics["gflops"],
                "tflops": metrics["tflops"],
                "total_flops_g": metrics["total_flops"] / 1e9,
            }
            all_results.append(result)

            print(f"  MXFP4 CUTLASS: {avg_time_us:.2f} us")
            print(f"    Effective bandwidth: {metrics['bandwidth_gbs']:.2f} GB/s")
            print(
                f"    Performance: {metrics['gflops']:.2f} GFLOPS ({metrics['tflops']:.4f} TFLOPS)"
            )
            print(f"    Total memory: {metrics['total_bytes_mb']:.2f} MB")

        except Exception as e:
            print(f"  MXFP4 CUTLASS: FAILED - {e}")
            all_results.append(
                {
                    "expected_m": shape.expected_m_per_group,
                    "actual_m": None,
                    "n": shape.n,
                    "k": shape.k,
                    "actual_k": None,
                    "num_groups": shape.num_groups,
                    "time_us": None,
                    "bandwidth_gbs": None,
                    "total_bytes_mb": None,
                    "gflops": None,
                    "tflops": None,
                    "total_flops_g": None,
                    "error": str(e),
                }
            )

    return all_results


def print_summary(results: List[dict]):
    """Print summary statistics from benchmark results."""
    print("\n" + "=" * 100)
    print("MXFP4 Blockwise Group GEMM Benchmark Results")
    print("=" * 100)

    df = pd.DataFrame(results)
    df_valid = df[df["time_us"].notna()].copy()

    if df_valid.empty:
        print("No successful benchmark runs!")
        return

    df_valid["time_us"] = df_valid["time_us"].round(2)
    df_valid["bandwidth_gbs"] = df_valid["bandwidth_gbs"].round(2)
    df_valid["total_bytes_mb"] = df_valid["total_bytes_mb"].round(2)
    df_valid["gflops"] = df_valid["gflops"].round(2)
    df_valid["tflops"] = df_valid["tflops"].round(4)
    df_valid["total_flops_g"] = df_valid["total_flops_g"].round(2)

    display_cols = [
        "expected_m",
        "actual_m",
        "n",
        "k",
        "num_groups",
        "time_us",
        "bandwidth_gbs",
        "gflops",
        "tflops",
        "total_bytes_mb",
    ]
    print("\nDetailed Results:")
    print(df_valid[display_cols].to_markdown(index=False))

    print("\n" + "=" * 100)
    print("Summary Statistics")
    print("=" * 100)
    summary = df_valid.agg(
        {
            "time_us": ["mean", "min", "max", "std"],
            "bandwidth_gbs": ["mean", "min", "max"],
            "gflops": ["mean", "min", "max"],
            "tflops": ["mean", "min", "max"],
        }
    )
    print(summary.to_markdown())


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark MXFP4 blockwise group GEMM kernel"
    )
    parser.add_argument(
        "--num-warmup", type=int, default=3, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--num-run", type=int, default=10, help="Number of benchmark iterations"
    )
    args = parser.parse_args()

    if not is_xpu_available():
        print("Error: Intel XPU not available")
        return

    try:
        from sgl_kernel import mxfp4_blockwise_scaled_grouped_mm

        assert callable(mxfp4_blockwise_scaled_grouped_mm)
    except ImportError:
        print("Error: mxfp4_blockwise_scaled_grouped_mm kernel not available")
        return

    print("Running MXFP4 Blockwise Group GEMM Benchmark")
    print(f"  Device: Intel XPU")
    print(f"  Warmup iterations: {args.num_warmup}")
    print(f"  Benchmark iterations: {args.num_run}")
    print(f"  MXFP4 block size: {MXFP4_BLOCK_SIZE}")

    if IS_CI:
        shape_args = [
            ShapeArg(expected_m_per_group=64, n=64, k=64, num_groups=2),
            ShapeArg(expected_m_per_group=128, n=128, k=128, num_groups=4),
        ]
    else:
        shape_args = [
            # Small shapes for validation
            ShapeArg(expected_m_per_group=64, n=64, k=64, num_groups=2),
            ShapeArg(expected_m_per_group=64, n=128, k=128, num_groups=4),
            ShapeArg(expected_m_per_group=128, n=256, k=256, num_groups=8),
            # GPT-OSS-120B gate_up projection (N=5760, K=2880), 128 experts
            ShapeArg(expected_m_per_group=128, n=5760, k=2880, num_groups=128),
            ShapeArg(expected_m_per_group=256, n=5760, k=2880, num_groups=128),
            ShapeArg(expected_m_per_group=512, n=5760, k=2880, num_groups=128),
            ShapeArg(expected_m_per_group=1024, n=5760, k=2880, num_groups=128),
            # GPT-OSS-120B down projection (N=2880, K=2880), 128 experts
            ShapeArg(expected_m_per_group=128, n=2880, k=2880, num_groups=128),
            ShapeArg(expected_m_per_group=256, n=2880, k=2880, num_groups=128),
            ShapeArg(expected_m_per_group=512, n=2880, k=2880, num_groups=128),
            ShapeArg(expected_m_per_group=1024, n=2880, k=2880, num_groups=128),
            # GPT-OSS-120B decode shapes (small M)
            ShapeArg(expected_m_per_group=1, n=5760, k=2880, num_groups=128),
            ShapeArg(expected_m_per_group=2, n=5760, k=2880, num_groups=128),
            ShapeArg(expected_m_per_group=4, n=5760, k=2880, num_groups=128),
            ShapeArg(expected_m_per_group=8, n=5760, k=2880, num_groups=128),
            ShapeArg(expected_m_per_group=1, n=2880, k=2880, num_groups=128),
            ShapeArg(expected_m_per_group=2, n=2880, k=2880, num_groups=128),
            ShapeArg(expected_m_per_group=4, n=2880, k=2880, num_groups=128),
            ShapeArg(expected_m_per_group=8, n=2880, k=2880, num_groups=128),
            # GPT-OSS-20B gate_up projection (N=5760, K=2880), 32 experts
            ShapeArg(expected_m_per_group=128, n=5760, k=2880, num_groups=32),
            ShapeArg(expected_m_per_group=256, n=5760, k=2880, num_groups=32),
            ShapeArg(expected_m_per_group=512, n=5760, k=2880, num_groups=32),
            # GPT-OSS-20B down projection (N=2880, K=2880), 32 experts
            ShapeArg(expected_m_per_group=128, n=2880, k=2880, num_groups=32),
            ShapeArg(expected_m_per_group=256, n=2880, k=2880, num_groups=32),
            ShapeArg(expected_m_per_group=512, n=2880, k=2880, num_groups=32),
            # GPT-OSS-20B decode shapes (small M)
            ShapeArg(expected_m_per_group=1, n=5760, k=2880, num_groups=32),
            ShapeArg(expected_m_per_group=4, n=5760, k=2880, num_groups=32),
            ShapeArg(expected_m_per_group=1, n=2880, k=2880, num_groups=32),
            ShapeArg(expected_m_per_group=4, n=2880, k=2880, num_groups=32),
        ]

    results = benchmark_one_shape(shape_args, args.num_warmup, args.num_run)
    print_summary(results)


if __name__ == "__main__":
    main()
