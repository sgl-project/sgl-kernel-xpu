# SPDX-License-Identifier: Apache-2.0
"""Benchmark FP8 / MXFP8 Blockwise Grouped GEMM (MoE) on Intel XPU.

``--dtype`` picks FP8 (BS=128, fp32 scales, SW-scaled) or MXFP8 (BS=32,
UE8M0 scales, HW-scaled). ``--prep-mode`` picks legacy (Python prep),
ondevice (SYCL prep), or compare (both, with prep-included speedup).
"""

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

FP8_BLOCK_SIZE = 128  # DSV3-style FP8, fp32 scales
MXFP8_BLOCK_SIZE = 32  # OCP MXFP8, UE8M0 uint8 scales
FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0


def is_xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def is_cri_device() -> bool:
    if not is_xpu_available():
        return False
    try:
        from sgl_kernel import is_xe3_arch

        return is_xe3_arch()
    except ImportError:
        return False


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


# ---------------------------------------------------------------------------
# Quantization helpers (FP8 e4m3 with per-block fp32 scales)
# ---------------------------------------------------------------------------


def quantize_to_fp8_e4m3(
    tensor: torch.Tensor, block_size: int = FP8_BLOCK_SIZE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """DSV3-style FP8 quant. (M, K) float -> (fp8_e4m3fn, fp32 scales [M, K//block_size])."""
    assert tensor.dim() == 2
    rows, cols = tensor.shape
    assert cols % block_size == 0

    fp32 = tensor.float()
    blocks = fp32.reshape(rows, cols // block_size, block_size)
    amax = torch.clamp(blocks.abs().amax(dim=-1), min=1e-12)
    scales = (amax / FP8_E4M3_MAX).float()
    scaled = (blocks / scales.unsqueeze(-1)).clamp(min=-FP8_E4M3_MAX, max=FP8_E4M3_MAX)
    return scaled.reshape(rows, cols).to(torch.float8_e4m3fn), scales


def quantize_matrix_blockwise_2d(
    tensor: torch.Tensor,
    block_k: int = FP8_BLOCK_SIZE,
    block_n: int = FP8_BLOCK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """DSV3-style FP8 2D-block quant. (N, K) float -> (fp8_e4m3fn, fp32 scales
    [N//block_n, K//block_k])."""
    assert tensor.dim() == 2
    n, k = tensor.shape
    assert n % block_n == 0 and k % block_k == 0

    blocked = tensor.float().reshape(n // block_n, block_n, k // block_k, block_k)
    amax = torch.clamp(blocked.abs().amax(dim=(1, 3)), min=1e-12)
    scales = (amax / FP8_E4M3_MAX).float()
    clamped = (blocked / scales.unsqueeze(1).unsqueeze(3)).clamp(
        min=-FP8_E4M3_MAX, max=FP8_E4M3_MAX
    )
    return clamped.reshape(n, k).to(torch.float8_e4m3fn), scales


def quantize_to_mxfp8(
    tensor: torch.Tensor, block_size: int = MXFP8_BLOCK_SIZE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """OCP MXFP8 quant. UE8M0 scale stored as exponent + 127."""
    assert tensor.dim() == 2
    rows, cols = tensor.shape
    assert cols % block_size == 0

    blocks = tensor.float().reshape(rows, cols // block_size, block_size)
    amax = blocks.abs().amax(dim=-1)
    ratio = torch.where(amax > 0, amax / FP8_E4M3_MAX, torch.ones_like(amax))
    exp = torch.clamp(torch.floor(torch.log2(ratio)), min=-127, max=127)
    scale_fp = torch.pow(2.0, exp)
    scaled = blocks / scale_fp.unsqueeze(-1)
    q = (
        scaled.clamp(min=-FP8_E4M3_MAX, max=FP8_E4M3_MAX)
        .reshape(rows, cols)
        .to(torch.float8_e4m3fn)
    )
    scale_u8 = (exp.to(torch.int32) + 127).to(torch.uint8)
    return q, scale_u8


def ensure_contiguous(t: torch.Tensor) -> torch.Tensor:
    return t if t.is_contiguous() else t.contiguous()


def construct_grouped_data(
    num_groups: int, m: int, k: int, n: int, device: str, dtype: str
) -> Tuple[list, list, list, list]:
    """Build per-expert (A, B, sA, sB). fp8 uses fp32 scales; mxfp8 uses
    UE8M0 uint8 per-row (un-transposed; kernel transposes on device)."""
    a_list, b_list, sa_list, sb_list = [], [], [], []
    for i in range(num_groups):
        torch.manual_seed(42 + i)
        a_orig = torch.randn(m, k, dtype=torch.float32) * 2.0
        torch.manual_seed(100 + i)
        b_orig = torch.randn(n, k, dtype=torch.float32) * 2.0
        if dtype == "mxfp8":
            a_q, sa = quantize_to_mxfp8(a_orig)
            b_q, sb = quantize_to_mxfp8(b_orig)
        else:
            a_q, sa = quantize_to_fp8_e4m3(a_orig)
            b_q, sb = quantize_matrix_blockwise_2d(b_orig)
        a_list.append(ensure_contiguous(a_q).to(device))
        b_list.append(ensure_contiguous(b_q).to(device))
        sa_list.append(ensure_contiguous(sa).to(device))
        sb_list.append(ensure_contiguous(sb).to(device))
    return a_list, b_list, sa_list, sb_list


# ---------------------------------------------------------------------------
# Input prep — both paths produce the SAME 18-arg call to the public entry point
# ---------------------------------------------------------------------------


def prepare_kernel_inputs_legacy(a_list, b_list, sa_list, sb_list, device: str) -> dict:
    """Legacy path: build pointer arrays + transpose A-scales in Python."""
    num_experts = len(a_list)
    m, k = a_list[0].shape
    n, _ = b_list[0].shape

    a_stack = torch.stack([ensure_contiguous(a) for a in a_list]).contiguous()
    b_stack = torch.stack([ensure_contiguous(b) for b in b_list]).contiguous()
    # Per-expert transpose to (K//BS, M) col-major-in-MN.
    sa_stack = torch.stack(
        [ensure_contiguous(s.t().contiguous()) for s in sa_list]
    ).contiguous()
    sb_stack = torch.stack([ensure_contiguous(s) for s in sb_list]).contiguous()

    output = torch.zeros((num_experts, m, n), dtype=torch.float32, device=device)

    def _ptrs(t):
        return torch.tensor(
            [t[i].data_ptr() for i in range(num_experts)],
            dtype=torch.uint64,
            device=device,
        )

    return _build_call_dict(
        output,
        a_ptrs=_ptrs(a_stack),
        b_ptrs=_ptrs(b_stack),
        out_ptrs=_ptrs(output),
        a_scales_ptrs=_ptrs(sa_stack),
        b_scales_ptrs=_ptrs(sb_stack),
        a_stack=a_stack,
        b_stack=b_stack,
        sa_stack=sa_stack,
        sb_stack=sb_stack,
        m=m,
        n=n,
        k=k,
        num_experts=num_experts,
        device=device,
    )


def prepare_kernel_inputs_ondevice(
    a_list, b_list, sa_list, sb_list, device: str
) -> dict:
    """Flat-2D A/scales + empty-ptr sentinels; kernel does prep on device."""
    num_experts = len(a_list)
    m, k = a_list[0].shape
    n, _ = b_list[0].shape
    total_m = num_experts * m

    a_flat = torch.cat([ensure_contiguous(a) for a in a_list], dim=0).contiguous()
    sa_flat = torch.cat([ensure_contiguous(s) for s in sa_list], dim=0).contiguous()
    b_stack = torch.stack([ensure_contiguous(b) for b in b_list]).contiguous()
    sb_stack = torch.stack([ensure_contiguous(s) for s in sb_list]).contiguous()

    output = torch.zeros((total_m, n), dtype=torch.float32, device=device)
    empty_ptrs = torch.empty((0,), dtype=torch.int64, device=device)

    inputs = _build_call_dict(
        output,
        a_ptrs=empty_ptrs,
        b_ptrs=empty_ptrs,
        out_ptrs=empty_ptrs,
        a_scales_ptrs=empty_ptrs,
        b_scales_ptrs=empty_ptrs,
        a_stack=a_flat,
        b_stack=b_stack,
        sa_stack=sa_flat,
        sb_stack=sb_stack,
        m=m,
        n=n,
        k=k,
        num_experts=num_experts,
        device=device,
    )
    # Flat-2D needs cumulative starts, not [0..E-1].
    inputs["expert_offsets"] = torch.arange(
        0, total_m, m, dtype=torch.int32, device=device
    )
    return inputs


def _build_call_dict(
    output,
    *,
    a_ptrs,
    b_ptrs,
    out_ptrs,
    a_scales_ptrs,
    b_scales_ptrs,
    a_stack,
    b_stack,
    sa_stack,
    sb_stack,
    m,
    n,
    k,
    num_experts,
    device,
) -> dict:
    return {
        "output": output,
        "a_ptrs": a_ptrs,
        "b_ptrs": b_ptrs,
        "out_ptrs": out_ptrs,
        "a_scales_ptrs": a_scales_ptrs,
        "b_scales_ptrs": b_scales_ptrs,
        "a_stack": a_stack,
        "b_stack": b_stack,
        "sa_stack": sa_stack,
        "sb_stack": sb_stack,
        "stride_a": torch.full((num_experts,), k, dtype=torch.int64, device=device),
        "stride_b": torch.full((num_experts,), k, dtype=torch.int64, device=device),
        "stride_c": torch.full((num_experts,), n, dtype=torch.int64, device=device),
        "layout_sfa": torch.empty((num_experts, 5), dtype=torch.int32, device=device),
        "layout_sfb": torch.empty((num_experts, 5), dtype=torch.int32, device=device),
        "problem_sizes": torch.tensor(
            [[m, n, k]] * num_experts, dtype=torch.int32, device=device
        ),
        "expert_offsets": torch.arange(num_experts, dtype=torch.int32, device=device),
        "workspace": torch.empty((64 * 1024 * 1024,), dtype=torch.uint8, device=device),
        "m": m,
        "n": n,
        "k": k,
    }


def _call(inputs: dict) -> None:
    from sgl_kernel import fp8_blockwise_scaled_grouped_mm

    fp8_blockwise_scaled_grouped_mm(
        inputs["output"],
        inputs["a_ptrs"],
        inputs["b_ptrs"],
        inputs["out_ptrs"],
        inputs["a_scales_ptrs"],
        inputs["b_scales_ptrs"],
        inputs["a_stack"],
        inputs["b_stack"],
        inputs["sa_stack"],
        inputs["sb_stack"],
        inputs["stride_a"],
        inputs["stride_b"],
        inputs["stride_c"],
        inputs["layout_sfa"],
        inputs["layout_sfb"],
        inputs["problem_sizes"],
        inputs["expert_offsets"],
        inputs["workspace"],
    )


# ---------------------------------------------------------------------------
# Metrics (kernel-only — prep is excluded from the FLOPS calc on purpose)
# ---------------------------------------------------------------------------


def calculate_flops(m: int, n: int, k: int, num_groups: int) -> int:
    return num_groups * 2 * m * n * k


def calculate_memory_bytes(m: int, n: int, k: int, num_groups: int, dtype: str) -> dict:
    """Bytes read+written per call. Scale layout differs by dtype."""
    if dtype == "mxfp8":
        bs = MXFP8_BLOCK_SIZE
        scale_k = k // bs
        a_bytes = num_groups * m * k  # fp8: 1 byte/elem
        b_bytes = num_groups * n * k
        sa_bytes = num_groups * m * scale_k * 1  # uint8 UE8M0
        sb_bytes = num_groups * n * scale_k * 1  # uint8 UE8M0 per-row
    else:
        bs = FP8_BLOCK_SIZE
        scale_k = k // bs
        scale_n = n // bs
        a_bytes = num_groups * m * k
        b_bytes = num_groups * n * k
        sa_bytes = num_groups * m * scale_k * 4
        sb_bytes = num_groups * scale_n * scale_k * 4
    out_bytes = num_groups * m * n * 4  # fp32 output
    total_read = a_bytes + b_bytes + sa_bytes + sb_bytes
    return {"total_bytes": total_read + out_bytes}


def calculate_metrics(
    m: int, n: int, k: int, num_groups: int, time_us: float, dtype: str
) -> dict:
    time_s = time_us / 1e6
    total_flops = calculate_flops(m, n, k, num_groups)
    total_bytes = calculate_memory_bytes(m, n, k, num_groups, dtype)["total_bytes"]
    return {
        "total_flops": total_flops,
        "gflops": (total_flops / 1e9) / time_s,
        "tflops": (total_flops / 1e12) / time_s,
        "total_bytes_mb": total_bytes / 1e6,
        "bandwidth_gbs": (total_bytes / 1e9) / time_s,
    }


# ---------------------------------------------------------------------------
# Timed runners
# ---------------------------------------------------------------------------


def _time_kernel_only(inputs: dict, num_warmup: int, num_run: int) -> float:
    """Time only the kernel call (prep already done)."""
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
    return (start.elapsed_time(end) / num_run) * 1000  # us


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
    """Time end-to-end prep + kernel call. This is the metric the on-device
    SYCL prep is designed to improve."""
    # Warmup
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
    return (start.elapsed_time(end) / num_run) * 1000  # us


# ---------------------------------------------------------------------------
# Per-shape benchmark
# ---------------------------------------------------------------------------


@dataclass
class ShapeArg:
    expected_m_per_group: int
    n: int
    k: int
    num_groups: int


def bench_one(
    shape: ShapeArg, prep_mode: str, dtype: str, num_warmup: int, num_run: int
) -> dict:
    device = "xpu"
    alignment = 64
    bs = MXFP8_BLOCK_SIZE if dtype == "mxfp8" else FP8_BLOCK_SIZE
    m = ceil_div(shape.expected_m_per_group, alignment) * alignment
    k = ceil_div(shape.k, bs) * bs
    n = ceil_div(shape.n, bs) * bs

    a_list, b_list, sa_list, sb_list = construct_grouped_data(
        shape.num_groups, m, k, n, device, dtype
    )

    result = {
        "expected_m": shape.expected_m_per_group,
        "actual_m": m,
        "n": n,
        "k": k,
        "actual_k": k,
        "num_groups": shape.num_groups,
        "dtype": dtype,
    }

    # Legacy Python prep is FP8-only (doesn't handle MXFP8 scale layouts).
    if dtype != "mxfp8" and prep_mode in ("legacy", "compare"):
        inputs = prepare_kernel_inputs_legacy(a_list, b_list, sa_list, sb_list, device)
        kernel_us = _time_kernel_only(inputs, num_warmup, num_run)
        e2e_us = _time_prep_plus_kernel(
            a_list,
            b_list,
            sa_list,
            sb_list,
            device,
            prepare_kernel_inputs_legacy,
            num_warmup,
            num_run,
        )
        result["legacy_kernel_us"] = kernel_us
        result["legacy_e2e_us"] = e2e_us
        result["legacy_prep_us"] = max(0.0, e2e_us - kernel_us)

    if prep_mode in ("ondevice", "compare"):
        inputs = prepare_kernel_inputs_ondevice(
            a_list, b_list, sa_list, sb_list, device
        )
        kernel_us = _time_kernel_only(inputs, num_warmup, num_run)
        e2e_us = _time_prep_plus_kernel(
            a_list,
            b_list,
            sa_list,
            sb_list,
            device,
            prepare_kernel_inputs_ondevice,
            num_warmup,
            num_run,
        )
        result["ondevice_kernel_us"] = kernel_us
        result["ondevice_e2e_us"] = e2e_us
        result["ondevice_prep_us"] = max(0.0, e2e_us - kernel_us)

    # `compare` only produces speedups when both paths ran; MXFP8 skips legacy.
    if prep_mode == "compare" and "legacy_kernel_us" in result:
        result["speedup_kernel"] = (
            result["legacy_kernel_us"] / result["ondevice_kernel_us"]
        )
        result["speedup_e2e"] = result["legacy_e2e_us"] / result["ondevice_e2e_us"]
        # Prep-only ratio is informative but volatile when prep ~ noise.
        if result["ondevice_prep_us"] > 1.0:
            result["speedup_prep"] = (
                result["legacy_prep_us"] / result["ondevice_prep_us"]
            )
        else:
            result["speedup_prep"] = float("inf")

    chosen_kernel_us = result.get("ondevice_kernel_us") or result.get(
        "legacy_kernel_us"
    )
    metrics = calculate_metrics(m, n, k, shape.num_groups, chosen_kernel_us, dtype)
    result["time_us"] = chosen_kernel_us
    result.update(
        {
            "bandwidth_gbs": metrics["bandwidth_gbs"],
            "total_bytes_mb": metrics["total_bytes_mb"],
            "gflops": metrics["gflops"],
            "tflops": metrics["tflops"],
            "total_flops_g": metrics["total_flops"] / 1e9,
        }
    )
    return result


def benchmark_shapes(
    shapes: List[ShapeArg],
    prep_mode: str,
    dtype: str,
    num_warmup: int,
    num_run: int,
) -> List[dict]:
    all_results = []
    for shape in shapes:
        print(
            f"\nBenchmark [{dtype}]: expected_m_per_group={shape.expected_m_per_group}, "
            f"n={shape.n}, k={shape.k}, num_groups={shape.num_groups}"
        )
        try:
            r = bench_one(shape, prep_mode, dtype, num_warmup, num_run)
            all_results.append(r)

            print(f"  Kernel-only time:  {r['time_us']:.2f} us")
            print(f"    Effective bandwidth: {r['bandwidth_gbs']:.2f} GB/s")
            print(
                f"    Performance: {r['gflops']:.2f} GFLOPS ({r['tflops']:.4f} TFLOPS)"
            )
            if prep_mode == "compare" and "legacy_e2e_us" in r:
                print(
                    f"  Prep+kernel  legacy={r['legacy_e2e_us']:.2f} us  "
                    f"ondevice={r['ondevice_e2e_us']:.2f} us  "
                    f"speedup={r['speedup_e2e']:.2f}x"
                )
                print(
                    f"  Prep-only    legacy={r['legacy_prep_us']:.2f} us  "
                    f"ondevice={r['ondevice_prep_us']:.2f} us"
                )
                print(
                    f"  Kernel-only  speedup={r['speedup_kernel']:.2f}x  (should be ~1)"
                )
            elif prep_mode == "compare":
                # dtype=mxfp8: legacy path is not applicable.
                print(
                    f"  ondevice_e2e={r['ondevice_e2e_us']:.2f} us  "
                    f"ondevice_prep={r['ondevice_prep_us']:.2f} us  "
                    "(legacy path skipped: MXFP8 is on-device-prep only)"
                )
        except Exception as e:
            print(f"  FAILED - {e}")
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FP8/MXFP8 blockwise grouped GEMM kernel"
    )
    parser.add_argument("--num-warmup", type=int, default=3)
    parser.add_argument("--num-run", type=int, default=10)
    parser.add_argument(
        "--prep-mode",
        choices=("legacy", "ondevice", "compare"),
        default="compare",
        help="legacy = Python prep; ondevice = SYCL prep; compare = both + speedup. "
        "MXFP8 ignores legacy (on-device-prep only).",
    )
    parser.add_argument(
        "--dtype",
        choices=("fp8", "mxfp8"),
        default="fp8",
        help="fp8 = DSV3-style E4M3 + fp32 scales, block=128 (SW-scaled); "
        "mxfp8 = OCP MXFP8 E4M3 + UE8M0 uint8 scales, block=32 (HW-scaled).",
    )
    args = parser.parse_args()

    if not is_xpu_available():
        print("Error: Intel XPU not available")
        return
    if not is_cri_device():
        print("Error: FP8/MXFP8 blockwise grouped GEMM requires a CRI (Xe3P) device")
        return
    try:
        from sgl_kernel import fp8_blockwise_scaled_grouped_mm

        assert callable(fp8_blockwise_scaled_grouped_mm)
    except ImportError:
        print("Error: fp8_blockwise_scaled_grouped_mm kernel not available")
        return

    bs = MXFP8_BLOCK_SIZE if args.dtype == "mxfp8" else FP8_BLOCK_SIZE
    print("Running FP8/MXFP8 Blockwise Group GEMM Benchmark")
    print(f"  Device: Intel XPU")
    print(f"  Dtype: {args.dtype}")
    print(f"  Prep mode: {args.prep_mode}")
    print(f"  Warmup iterations: {args.num_warmup}")
    print(f"  Benchmark iterations: {args.num_run}")
    print(f"  Block size: {bs}")

    if IS_CI:
        shapes = [
            ShapeArg(expected_m_per_group=128, n=128, k=128, num_groups=2),
            ShapeArg(expected_m_per_group=256, n=256, k=256, num_groups=4),
        ]
    else:
        shapes = [
            # Small / validation
            ShapeArg(expected_m_per_group=128, n=128, k=128, num_groups=2),
            ShapeArg(expected_m_per_group=128, n=256, k=256, num_groups=4),
            ShapeArg(expected_m_per_group=256, n=512, k=512, num_groups=8),
            # DSV3-style FP8 MoE shapes (illustrative; tune to your model)
            ShapeArg(expected_m_per_group=128, n=4096, k=4096, num_groups=8),
            ShapeArg(expected_m_per_group=256, n=4096, k=4096, num_groups=8),
            ShapeArg(expected_m_per_group=512, n=4096, k=4096, num_groups=8),
            ShapeArg(expected_m_per_group=1024, n=4096, k=4096, num_groups=8),
            # Decode shapes (small M) — prep overhead is most visible here
            ShapeArg(expected_m_per_group=1, n=4096, k=4096, num_groups=8),
            ShapeArg(expected_m_per_group=4, n=4096, k=4096, num_groups=8),
            ShapeArg(expected_m_per_group=16, n=4096, k=4096, num_groups=8),
            # Many experts (prep cost scales with E)
            ShapeArg(expected_m_per_group=128, n=2048, k=2048, num_groups=32),
            ShapeArg(expected_m_per_group=128, n=2048, k=2048, num_groups=64),
        ]

    results = benchmark_shapes(
        shapes, args.prep_mode, args.dtype, args.num_warmup, args.num_run
    )
    print_summary(results, title="FP8/MXFP8 Blockwise Group GEMM Benchmark Results")

    if args.prep_mode == "compare":
        # Custom comparison summary, since print_summary only knows the
        # kernel-only time_us column.
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
