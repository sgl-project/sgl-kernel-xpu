# SPDX-License-Identifier: Apache-2.0
"""Benchmark script for MXFP4 (E2M1) Block-Scaled Grouped GEMM for MoE on Intel XPU."""

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import torch
from utils import print_summary

IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)

MXFP4_BLOCK_SIZE = 32


def is_xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def is_cri_device() -> bool:
    """Check whether the current XPU device is CRI (Xe3P)."""
    if not is_xpu_available():
        return False
    try:
        from sgl_kernel import is_xe3_arch

        return is_xe3_arch()
    except ImportError:
        return False


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


def _common_meta(num_experts, m, n, k, device, workspace_bytes=64 * 1024 * 1024):
    return {
        "problem_sizes": torch.tensor(
            [[m, n, k]] * num_experts, dtype=torch.int32, device=device
        ),
        "workspace": torch.empty((workspace_bytes,), dtype=torch.uint8, device=device),
        "m": m,
        "n": n,
        "k": k,
    }


def prepare_kernel_inputs_legacy(
    a_list, b_list, scales_a_list, scales_b_list, device: str
) -> dict:
    """Legacy path: build pointer arrays + transpose A/B-scales in Python."""
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

    def _ptrs(t):
        return torch.tensor(
            [t[i].data_ptr() for i in range(num_experts)],
            dtype=torch.uint64,
            device=device,
        )

    inputs = {
        "output": output,
        "a_ptrs": _ptrs(a_stack),
        "b_ptrs": _ptrs(b_stack),
        "out_ptrs": _ptrs(output),
        "a_scales_ptrs": _ptrs(scales_a_stack),
        "b_scales_ptrs": _ptrs(scales_b_stack),
        "a_stack": a_stack,
        "b_stack": b_stack,
        "scales_a_stack": scales_a_stack,
        "scales_b_stack": scales_b_stack,
        "expert_offsets": torch.arange(num_experts, dtype=torch.int32, device=device),
    }
    inputs.update(_common_meta(num_experts, m, n, k, device))
    return inputs


def prepare_kernel_inputs_ondevice(
    a_list, b_list, sa_list, sb_list, device: str
) -> dict:
    """Empty sentinel ptrs + flat-2D inputs; kernel does prep on device."""
    num_experts = len(a_list)
    m, packed_k = a_list[0].shape
    k = packed_k * 2
    n = b_list[0].shape[0]
    total_m = num_experts * m

    a_flat = torch.cat(
        [ensure_contiguous_layout(a) for a in a_list], dim=0
    ).contiguous()
    sa_flat = torch.cat(
        [ensure_contiguous_layout(s) for s in sa_list], dim=0
    ).contiguous()
    b_stack = torch.stack(
        [ensure_contiguous_layout(b) for b in b_list], dim=0
    ).contiguous()
    sb_stack = torch.stack(
        [ensure_contiguous_layout(s) for s in sb_list], dim=0
    ).contiguous()

    output = torch.zeros((total_m, n), dtype=torch.float32, device=device)
    empty_ptrs = torch.empty((0,), dtype=torch.int64, device=device)

    inputs = {
        "output": output,
        "a_ptrs": empty_ptrs,
        "b_ptrs": empty_ptrs,
        "out_ptrs": empty_ptrs,
        "a_scales_ptrs": empty_ptrs,
        "b_scales_ptrs": empty_ptrs,
        "a_stack": a_flat,
        "b_stack": b_stack,
        "scales_a_stack": sa_flat,
        "scales_b_stack": sb_stack,
        "expert_offsets": torch.arange(0, total_m, m, dtype=torch.int32, device=device),
    }
    inputs.update(_common_meta(num_experts, m, n, k, device))
    return inputs


# Default = legacy (CI-backed).
prepare_kernel_inputs = prepare_kernel_inputs_legacy


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


def _call(inputs: dict) -> None:
    from sgl_kernel import mxfp4_blockwise_scaled_grouped_mm

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
        inputs["problem_sizes"],
        inputs["expert_offsets"],
        inputs["workspace"],
    )


def _time_kernel_only(inputs: dict, num_warmup: int, num_run: int) -> float:
    """Kernel-only timing (avg us)."""
    for _ in range(num_warmup):
        _call(inputs)
    torch.xpu.synchronize()

    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    for _ in range(num_run):
        _call(inputs)
    end.record()
    end.synchronize()
    torch.xpu.synchronize()
    return (start.elapsed_time(end) / num_run) * 1000


def _time_prep_plus_kernel(
    a_list,
    b_list,
    sa_list,
    sb_list,
    device: str,
    prep_fn,
    num_warmup: int,
    num_run: int,
) -> float:
    """E2E (prep + kernel) timing (avg us)."""
    for _ in range(num_warmup):
        inputs = prep_fn(a_list, b_list, sa_list, sb_list, device)
        _call(inputs)
    torch.xpu.synchronize()

    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    for _ in range(num_run):
        inputs = prep_fn(a_list, b_list, sa_list, sb_list, device)
        _call(inputs)
    end.record()
    end.synchronize()
    torch.xpu.synchronize()
    return (start.elapsed_time(end) / num_run) * 1000


def bench_mxfp4_cutlass(
    expected_m_per_group: int,
    n: int,
    k: int,
    num_groups: int,
    num_warmup: int,
    num_run: int,
    prep_mode: str = "legacy",
) -> Tuple[float, int, int, dict]:
    """Returns (kernel_us, m, k_aligned, extra)."""
    device = "xpu"
    alignment = 64
    m = ceil_div(expected_m_per_group, alignment) * alignment
    k_aligned = ceil_div(k, MXFP4_BLOCK_SIZE) * MXFP4_BLOCK_SIZE

    a_list, b_list, scales_a_list, scales_b_list = construct_mxfp4_grouped_data(
        num_groups, m, k_aligned, n, device
    )

    extra = {}

    def _measure(prep_fn):
        inputs = prep_fn(a_list, b_list, scales_a_list, scales_b_list, device)
        kernel_us = _time_kernel_only(inputs, num_warmup, num_run)
        e2e_us = _time_prep_plus_kernel(
            a_list,
            b_list,
            scales_a_list,
            scales_b_list,
            device,
            prep_fn,
            num_warmup,
            num_run,
        )
        return kernel_us, e2e_us

    if prep_mode in ("legacy", "compare"):
        legacy_kernel, legacy_e2e = _measure(prepare_kernel_inputs_legacy)
        extra["legacy_kernel_us"] = legacy_kernel
        extra["legacy_e2e_us"] = legacy_e2e
        extra["legacy_prep_us"] = max(0.0, legacy_e2e - legacy_kernel)

    if prep_mode in ("ondevice", "compare"):
        ondevice_kernel, ondevice_e2e = _measure(prepare_kernel_inputs_ondevice)
        extra["ondevice_kernel_us"] = ondevice_kernel
        extra["ondevice_e2e_us"] = ondevice_e2e
        extra["ondevice_prep_us"] = max(0.0, ondevice_e2e - ondevice_kernel)

    if prep_mode == "compare":
        extra["speedup_kernel"] = (
            extra["legacy_kernel_us"] / extra["ondevice_kernel_us"]
        )
        extra["speedup_e2e"] = extra["legacy_e2e_us"] / extra["ondevice_e2e_us"]
        extra["speedup_prep"] = (
            extra["legacy_prep_us"] / extra["ondevice_prep_us"]
            if extra["ondevice_prep_us"] > 1.0
            else float("inf")
        )

    chosen_us = extra.get("ondevice_kernel_us") or extra["legacy_kernel_us"]
    return chosen_us, m, k_aligned, extra


def benchmark_one_shape(
    shape_args: List[ShapeArg], num_warmup: int, num_run: int, prep_mode: str = "legacy"
) -> List[dict]:
    """Run benchmark for a list of shapes and collect results."""
    all_results = []

    for shape in shape_args:
        print(
            f"\nBenchmark: expected_m_per_group={shape.expected_m_per_group}, "
            f"n={shape.n}, k={shape.k}, num_groups={shape.num_groups}"
        )

        try:
            avg_time_us, actual_m, actual_k, extra = bench_mxfp4_cutlass(
                shape.expected_m_per_group,
                shape.n,
                shape.k,
                shape.num_groups,
                num_warmup,
                num_run,
                prep_mode=prep_mode,
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
            result.update(extra)
            all_results.append(result)

            print(f"  MXFP4 CUTLASS kernel-only: {avg_time_us:.2f} us")
            print(f"    Effective bandwidth: {metrics['bandwidth_gbs']:.2f} GB/s")
            print(
                f"    Performance: {metrics['gflops']:.2f} GFLOPS ({metrics['tflops']:.4f} TFLOPS)"
            )
            if prep_mode == "compare":
                print(
                    f"  Prep+kernel  legacy={extra['legacy_e2e_us']:.2f} us  "
                    f"ondevice={extra['ondevice_e2e_us']:.2f} us  "
                    f"speedup={extra['speedup_e2e']:.2f}x"
                )
                print(
                    f"  Prep-only    legacy={extra['legacy_prep_us']:.2f} us  "
                    f"ondevice={extra['ondevice_prep_us']:.2f} us"
                )

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
    parser.add_argument(
        "--prep-mode",
        choices=("legacy", "ondevice", "compare"),
        default="legacy",
        help="legacy = Python prep; ondevice = SYCL prep; compare = both + speedup",
    )
    args = parser.parse_args()

    if not is_xpu_available():
        print("Error: Intel XPU not available")
        return

    if not is_cri_device():
        print("Error: MXFP4 blockwise scaled grouped GEMM requires a CRI (Xe3P) device")
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

    results = benchmark_one_shape(
        shape_args, args.num_warmup, args.num_run, prep_mode=args.prep_mode
    )
    print_summary(results, title="MXFP4 Blockwise Group GEMM Benchmark Results")

    if args.prep_mode == "compare":
        print("\n" + "=" * 100)
        print("Prep+Kernel End-to-End Comparison")
        print("=" * 100)
        ok = [r for r in results if r.get("legacy_e2e_us") is not None]
        if ok:
            import pandas as pd

            df = pd.DataFrame(
                [
                    {
                        "M": r["actual_m"],
                        "N": r["n"],
                        "K": r["k"],
                        "E": r["num_groups"],
                        "legacy_e2e_us": round(r["legacy_e2e_us"], 2),
                        "ondevice_e2e_us": round(r["ondevice_e2e_us"], 2),
                        "legacy_prep_us": round(r["legacy_prep_us"], 2),
                        "ondevice_prep_us": round(r["ondevice_prep_us"], 2),
                        "speedup_e2e": round(r["speedup_e2e"], 2),
                    }
                    for r in ok
                ]
            )
            print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
