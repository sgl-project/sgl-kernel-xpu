# SPDX-License-Identifier: Apache-2.0
"""Benchmark script for MXFP4 (E2M1) per-token group quantization on Intel XPU."""

import itertools
import os

import pandas as pd
import torch
import triton

MXFP4_BLOCK_SIZE = 32
FLOAT4_E2M1_MAX = 6.0

IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)

kE2M1ToFloat = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


def is_xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


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


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 into two 4-bit values."""
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    return torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], -1)


def dequantize_e2m1(
    quantized: torch.Tensor, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Dequantize E2M1 values back to float."""
    sign = ((quantized >> 3) & 1).to(torch.bool)
    magnitude_idx = (quantized & 0x07).to(torch.long)
    kE2M1 = kE2M1ToFloat.to(device=quantized.device)
    magnitude = kE2M1[magnitude_idx]
    result = torch.where(sign, -magnitude, magnitude)
    return result.to(dtype)


def quantize_to_mxfp4_ref(
    tensor: torch.Tensor, block_size: int = MXFP4_BLOCK_SIZE, eps: float = 1e-10
) -> tuple:
    """Reference implementation for MXFP4 quantization."""
    assert tensor.dim() == 2
    m, k = tensor.shape
    assert k % block_size == 0
    assert k % 2 == 0

    tensor_fp32 = tensor.float()
    num_blocks = k // block_size
    tensor_blocks = tensor_fp32.reshape(m, num_blocks, block_size)

    block_max = tensor_blocks.abs().max(dim=-1, keepdim=True).values
    block_max = torch.clamp(block_max, min=eps)

    log2_max = torch.log2(block_max / FLOAT4_E2M1_MAX)
    exponent = torch.ceil(log2_max).clamp(min=-127, max=127).to(torch.int32)
    scales_ue8m0 = (exponent + 127).to(torch.uint8).squeeze(-1)

    scale_values = torch.pow(2.0, exponent.float())
    scaled_tensor = tensor_blocks / scale_values
    quantized_blocks = quantize_to_e2m1(scaled_tensor)
    quantized = quantized_blocks.reshape(m, k)
    packed = pack_fp4(quantized)

    return packed, scales_ue8m0


def dequantize_mxfp4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    block_size: int = MXFP4_BLOCK_SIZE,
) -> torch.Tensor:
    """Dequantize MXFP4 packed values back to float."""
    m, packed_k = packed.shape
    k = packed_k * 2

    unpacked = unpack_fp4(packed)
    dequantized = dequantize_e2m1(unpacked, dtype)

    num_blocks = k // block_size
    dequantized_blocks = dequantized.reshape(m, num_blocks, block_size)

    scale_exp = scales.to(torch.int32) - 127
    scale_values = torch.pow(2.0, scale_exp.float()).unsqueeze(-1)
    scaled = dequantized_blocks * scale_values

    return scaled.reshape(m, k).to(dtype)


def reference_per_token_group_quant_mxfp4(
    x: torch.Tensor, group_size: int, eps: float = 1e-10
) -> tuple:
    """Reference implementation using PyTorch operations."""
    assert x.shape[-1] % group_size == 0
    assert x.is_contiguous()

    x_cpu = x.cpu().float()
    x_q, x_s = quantize_to_mxfp4_ref(x_cpu, group_size, eps)
    return x_q.to(x.device), x_s.to(x.device)


def sglang_per_token_group_quant_mxfp4(
    x: torch.Tensor, group_size: int, eps: float = 1e-10
) -> tuple:
    """SGL kernel wrapper for MXFP4 quantization."""
    from sgl_kernel import sgl_per_token_group_quant_fp4

    assert x.shape[-1] % group_size == 0
    assert x.is_contiguous()

    x_q, x_s = sgl_per_token_group_quant_fp4(x=x, group_size=group_size, eps=eps)
    return x_q, x_s


def calculate_diff(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    group_size: int,
    src_dtype: torch.dtype,
):
    """Verify correctness by comparing reference and kernel implementations."""
    device = torch.device("xpu")

    x = torch.randn(batch_size * seq_len, hidden_dim, device=device, dtype=src_dtype)

    x_q_ref, x_s_ref = reference_per_token_group_quant_mxfp4(x.clone(), group_size)
    x_q_sgl, x_s_sgl = sglang_per_token_group_quant_mxfp4(x.clone(), group_size)

    # Compare quantized outputs directly (packed uint8 and scales)
    q_match = torch.equal(x_q_ref.cpu(), x_q_sgl.cpu())
    s_match = torch.equal(x_s_ref.cpu(), x_s_sgl.cpu())

    if q_match and s_match:
        print(
            f"  ✅ Quantized values match (batch={batch_size}, seq={seq_len}, hidden={hidden_dim}, group={group_size}, dtype={src_dtype})"
        )
    else:
        q_mismatches = (
            (x_q_ref.cpu() != x_q_sgl.cpu()).sum().item() if not q_match else 0
        )
        s_mismatches = (
            (x_s_ref.cpu() != x_s_sgl.cpu()).sum().item() if not s_match else 0
        )
        print(
            f"  ❌ Quantized values differ: "
            f"packed_q({q_mismatches} mismatches) "
            f"scales({s_mismatches} mismatches)"
        )

    # Compare dequantized outputs
    x_dq_ref = dequantize_mxfp4(x_q_ref.cpu(), x_s_ref.cpu(), torch.float32, group_size)
    x_dq_sgl = dequantize_mxfp4(x_q_sgl.cpu(), x_s_sgl.cpu(), torch.float32, group_size)

    if torch.allclose(x_dq_ref, x_dq_sgl, rtol=0.2, atol=0.5):
        print(
            f"  ✅ Dequantized values match (batch={batch_size}, seq={seq_len}, hidden={hidden_dim}, group={group_size}, dtype={src_dtype})"
        )
    else:
        max_diff = (x_dq_ref - x_dq_sgl).abs().max().item()
        print(f"  ❌ Dequantized values differ (max_diff={max_diff:.4f})")


def calculate_flops(num_elements: int, num_groups: int) -> int:
    """Calculate FLOPs for MXFP4 per-token-group quantization."""
    flops_per_element = 5
    flops_per_group = 8
    return (num_elements * flops_per_element) + (num_groups * flops_per_group)


def calculate_effective_bandwidth(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    group_size: int,
    src_dtype: torch.dtype,
    time_ms: float,
) -> dict:
    """Calculate effective bandwidth and FLOPs for MXFP4 quantization kernel."""
    num_tokens = batch_size * seq_len
    num_elements = num_tokens * hidden_dim
    num_groups = num_elements // group_size

    dtype_size = 2 if src_dtype in (torch.float16, torch.bfloat16) else 4
    input_bytes = num_elements * dtype_size
    output_bytes = num_elements // 2
    scale_bytes = num_groups
    total_bytes = input_bytes + output_bytes + scale_bytes

    time_s = time_ms / 1000.0
    bandwidth_gbs = (total_bytes / 1e9) / time_s

    total_flops = calculate_flops(num_elements, num_groups)
    gflops = (total_flops / 1e9) / time_s

    return {
        "num_tokens": num_tokens,
        "num_elements": num_elements,
        "num_groups": num_groups,
        "total_bytes": total_bytes,
        "bandwidth_gbs": bandwidth_gbs,
        "total_flops": total_flops,
        "gflops": gflops,
    }


batch_size_range = [1, 2, 4, 8, 16, 32, 64] if not IS_CI else [1, 4, 16]
seq_len_range = [64, 128, 256, 512, 1024, 2048] if not IS_CI else [64, 256]
group_size_range = [32]
src_dtype_range = [torch.bfloat16]

configs = list(
    itertools.product(
        batch_size_range, seq_len_range, group_size_range, src_dtype_range
    )
)

all_results = []


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "seq_len", "group_size", "src_dtype"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["sglang"],
        line_names=["SGL Kernel"],
        styles=[("green", "-")],
        ylabel="us",
        plot_name="per-token-group-quant-mxfp4-performance",
        args={},
    )
)
def benchmark(batch_size, seq_len, group_size, src_dtype, provider):
    device = torch.device("xpu")
    hidden_dim = 7168

    x = torch.randn(batch_size * seq_len, hidden_dim, device=device, dtype=src_dtype)

    quantiles = [0.5, 0.2, 0.8]

    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: sglang_per_token_group_quant_mxfp4(x, group_size),
        quantiles=quantiles,
    )

    bw_metrics = calculate_effective_bandwidth(
        batch_size, seq_len, hidden_dim, group_size, src_dtype, ms
    )

    all_results.append(
        {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_tokens": bw_metrics["num_tokens"],
            "hidden_dim": hidden_dim,
            "group_size": group_size,
            "src_dtype": str(src_dtype),
            "provider": provider,
            "time_us": 1000 * ms,
            "bandwidth_gbs": bw_metrics["bandwidth_gbs"],
            "total_bytes_mb": bw_metrics["total_bytes"] / 1e6,
            "total_flops_m": bw_metrics["total_flops"] / 1e6,
            "gflops": bw_metrics["gflops"],
        }
    )

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


def print_summary(results: list):
    """Print summary statistics from benchmark results."""
    print("\n" + "=" * 100)
    print("MXFP4 Per-Token Group Quantization Benchmark Results")
    print("=" * 100)

    df = pd.DataFrame(results)
    df["bandwidth_gbs"] = df["bandwidth_gbs"].round(2)
    df["total_bytes_mb"] = df["total_bytes_mb"].round(2)
    df["time_us"] = df["time_us"].round(2)
    df["total_flops_m"] = df["total_flops_m"].round(2)
    df["gflops"] = df["gflops"].round(2)

    print("\nDetailed Results:")
    print(df.to_markdown(index=False))

    print("\n" + "=" * 100)
    print("Summary Statistics by Provider")
    print("=" * 100)
    summary = df.groupby("provider").agg(
        {
            "bandwidth_gbs": ["mean", "min", "max"],
            "time_us": ["mean", "min", "max"],
            "gflops": ["mean", "min", "max"],
        }
    )
    print(summary.to_markdown())


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

    print("Running MXFP4 Per-Token Group Quantization Benchmark")
    print("  Device: Intel XPU")
    print(f"  MXFP4 block size: {MXFP4_BLOCK_SIZE}")

    print("\n" + "=" * 80)
    print("Correctness Verification")
    print("=" * 80)
    calculate_diff(
        batch_size=2,
        seq_len=64,
        hidden_dim=128,
        group_size=32,
        src_dtype=torch.bfloat16,
    )
    calculate_diff(
        batch_size=1, seq_len=32, hidden_dim=128, group_size=32, src_dtype=torch.float32
    )

    print("\n" + "=" * 80)
    print("Performance Benchmark")
    print("=" * 80)
    benchmark.run(print_data=True)

    print_summary(all_results)


if __name__ == "__main__":
    main()
