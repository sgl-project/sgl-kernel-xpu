# SPDX-License-Identifier: Apache-2.0
"""Benchmark script for MXFP4 (E2M1) per-token group quantization on Intel XPU.

Rounding: Per OCP MX spec (section 5.3.3), FP4 conversion uses
roundTiesToEven — at midpoints between representable values, the
value with even mantissa (mantissa bit = 0) is chosen.
"""

import itertools
import os

import pandas as pd
import torch
import triton

MXFP4_BLOCK_SIZE = 32
FLOAT4_E2M1_MAX = 6.0

# E2M1 format parameters (from Microsoft microxcaling formats.py)
E2M1_EBITS = 2
E2M1_MBITS = 3  # includes sign bit and implicit one
E2M1_EMAX = 2 ** (E2M1_EBITS - 1)  # = 2
E2M1_MAX_NORM = (
    2**E2M1_EMAX * float(2 ** (E2M1_MBITS - 1) - 1) / 2 ** (E2M1_MBITS - 2)
)  # = 6.0

FP32_EXPONENT_BIAS = 127
FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)  # 2^(-126)

IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)

kE2M1ToFloat = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


def is_xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def _round_mantissa_even(A: torch.Tensor) -> torch.Tensor:
    """Round mantissa using roundTiesToEven (from Microsoft microxcaling).

    At exact 0.5 midpoints (i.e., values like 0.5, 2.5, 4.5, ...),
    round to the nearest even integer (the one whose LSB is 0).

    Ref: https://github.com/microsoft/microxcaling/blob/main/mx/elemwise_ops.py
    """
    absA = torch.abs(A)
    # Identify exact midpoints: 0.5, 2.5, 4.5, ...  i.e. (absA - 0.5) % 2 == 0
    maskA = ((absA - 0.5) % 2 == torch.zeros_like(A)).type(A.dtype)
    # round half up, then subtract 1 at midpoints to get even
    return torch.sign(A) * (torch.floor(absA + 0.5) - maskA)


def _quantize_elemwise_core_e2m1(
    A: torch.Tensor, saturate_normals: bool = True
) -> torch.Tensor:
    """Element-wise quantization to E2M1 using Microsoft microxcaling's
    _quantize_elemwise_core algorithm with round='even'.

    E2M1 format: ebits=2, mbits=3, emax=2, max_norm=6.0
    min_exp = -(2^(ebits-1)) + 2 = 0

    Algorithm (from Microsoft microxcaling elemwise_ops.py):
      1. Compute per-element private exponent = floor(log2(|A|)),
         clamped to min_exp.
      2. Left-shift: out = A / 2^private_exp * 2^(mbits-2)
      3. Round mantissa with roundTiesToEven
      4. Right-shift: out = out / 2^(mbits-2) * 2^private_exp
      5. Clamp to [-max_norm, max_norm] if saturate_normals

    Ref: https://github.com/microsoft/microxcaling/blob/main/mx/elemwise_ops.py
    """
    ebits = E2M1_EBITS  # 2
    mbits = E2M1_MBITS  # 3
    max_norm = E2M1_MAX_NORM  # 6.0

    # min representable exponent: -(2^(ebits-1)) + 2 = 0
    min_exp = -(2 ** (ebits - 1)) + 2  # 0

    out = A.clone()

    # Per-element private exponent: floor(log2(|A|))
    # Add guard for zeros: log2(0) is -inf, we use (A==0) to avoid that
    private_exp = torch.floor(torch.log2(torch.abs(A) + (A == 0).type(A.dtype)))
    private_exp = private_exp.clip(min=min_exp)

    # Left-shift: scale up so mantissa bits land in integer portion
    # out = A / 2^private_exp * 2^(mbits-2)
    shift = mbits - 2  # = 1
    out = out / (2**private_exp) * (2**shift)

    # Round mantissa with roundTiesToEven
    out = _round_mantissa_even(out)

    # Right-shift: undo scaling
    # out = out / 2^(mbits-2) * 2^private_exp
    out = out / (2**shift) * (2**private_exp)

    # Saturate to [-max_norm, max_norm]
    if saturate_normals:
        out = torch.clamp(out, min=-max_norm, max=max_norm)

    return out


def _float_to_e2m1_code(val: torch.Tensor) -> torch.Tensor:
    """Convert quantized float values back to E2M1 4-bit codes.

    After _quantize_elemwise_core_e2m1, values are one of the 8 representable
    E2M1 magnitudes: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}.
    This maps them to 4-bit codes (sign in bit 3, magnitude in bits 0-2).
    """
    sign = (val < 0).to(torch.uint8)
    abs_val = val.abs()

    # Map representable magnitudes to 3-bit indices via the kE2M1ToFloat LUT.
    indices = torch.zeros_like(abs_val, dtype=torch.uint8)
    lut = kE2M1ToFloat.to(device=val.device)
    for i in range(8):
        indices = torch.where(
            torch.isclose(abs_val, lut[i].expand_as(abs_val), atol=1e-6, rtol=0),
            torch.tensor(i, dtype=torch.uint8, device=val.device),
            indices,
        )

    return (sign << 3) | indices


def quantize_to_e2m1(tensor: torch.Tensor) -> torch.Tensor:
    """Quantize tensor values to E2M1 format (4-bit indices).

    Uses the Microsoft microxcaling _quantize_elemwise_core algorithm
    with roundTiesToEven, then maps the resulting float values to 4-bit codes.

    Ref: https://github.com/microsoft/microxcaling/blob/main/mx/elemwise_ops.py
    """
    quantized_float = _quantize_elemwise_core_e2m1(
        tensor.float(), saturate_normals=True
    )
    return _float_to_e2m1_code(quantized_float)


def pack_fp4(tensor: torch.Tensor) -> torch.Tensor:
    """Pack two 4-bit values into one uint8."""
    assert tensor.shape[-1] % 2 == 0
    shape = tensor.shape[:-1] + (tensor.shape[-1] // 2, 2)
    paired = tensor.reshape(shape)
    packed = (paired[..., 0] & 0x0F) | ((paired[..., 1] & 0x0F) << 4)
    return packed.to(torch.uint8)


def _normalize_packed_fp4_signed_zero(packed: torch.Tensor) -> torch.Tensor:
    """Canonicalize signed zeros in packed FP4 bytes.

    In E2M1, code 0x0 is +0.0 and code 0x8 is -0.0.  Both represent
    the same value, but different implementations may emit either form.
    This helper rewrites every -0.0 nibble (0x8) to +0.0 (0x0) so that
    byte-level comparisons are not tripped up by this harmless difference.
    """
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    lo = torch.where(lo == 0x08, torch.zeros_like(lo), lo)
    hi = torch.where(hi == 0x08, torch.zeros_like(hi), hi)
    return (lo | (hi << 4)).to(torch.uint8)


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


def _shared_exponents(A: torch.Tensor, axis: int) -> torch.Tensor:
    """Compute shared exponents per block using Microsoft microxcaling's
    _shared_exponents algorithm with method="max".

    Algorithm:
      1. shared_exp = max(|A|) along axis (per block)
      2. shared_exp = floor(log2(shared_exp + FP32_MIN_NORMAL * (shared_exp == 0)))
         The FP32_MIN_NORMAL guard ensures log2(0) doesn't produce -inf.
      3. Offset by emax: shared_exp = shared_exp - emax

    Ref: https://github.com/microsoft/microxcaling/blob/main/mx/mx_ops.py
    """
    shared_exp = torch.max(torch.abs(A), dim=axis, keepdim=True).values

    # floor(log2(...)) with zero-guard from microxcaling
    shared_exp = torch.floor(
        torch.log2(
            shared_exp + FP32_MIN_NORMAL * (shared_exp == 0).type(shared_exp.dtype)
        )
    )

    # Offset by the largest representable exponent in E2M1
    shared_exp = shared_exp - E2M1_EMAX

    return shared_exp


def quantize_to_mxfp4_ref(
    tensor: torch.Tensor, block_size: int = MXFP4_BLOCK_SIZE, eps: float = 1e-10
) -> tuple:
    """Reference implementation for MXFP4 quantization using Microsoft
    microxcaling's _quantize_mx algorithm.

    Algorithm (from mx_ops.py _quantize_mx):
      1. Reshape into blocks
      2. Compute shared exponent per block via _shared_exponents
      3. Clamp shared_exp to scale_emax range [-127, 127]
      4. Scale elements: A = A / 2^shared_exp
      5. Quantize element-wise with _quantize_elemwise_core (saturate_normals=True)
      6. Rescale: A = A * 2^shared_exp (implicitly stored in UE8M0 scale)

    Ref: https://github.com/microsoft/microxcaling/blob/main/mx/mx_ops.py
    """
    assert tensor.dim() == 2
    m, k = tensor.shape
    assert k % block_size == 0
    assert k % 2 == 0

    tensor_fp32 = tensor.float()
    num_blocks = k // block_size
    tensor_blocks = tensor_fp32.reshape(m, num_blocks, block_size)

    # Compute shared exponents (microxcaling _shared_exponents + offset by emax)
    shared_exp = _shared_exponents(tensor_blocks, axis=-1)

    # Clamp to UE8M0 scale range: scale_bits=8, scale_emax = 2^(8-1)-1 = 127
    scale_emax = 127
    shared_exp = shared_exp.clamp(min=-scale_emax, max=scale_emax)

    # Encode as UE8M0: stored_scale = shared_exp + 127
    scales_ue8m0 = (shared_exp.to(torch.int32) + 127).to(torch.uint8).squeeze(-1)

    # Scale elements by shared exponent: A = A / 2^shared_exp
    scaled_tensor = tensor_blocks / (2.0**shared_exp)

    # Quantize element-wise with microxcaling core (roundTiesToEven, saturate)
    quantized_float = _quantize_elemwise_core_e2m1(scaled_tensor, saturate_normals=True)

    # Convert quantized float values to 4-bit E2M1 codes
    quantized_blocks = _float_to_e2m1_code(quantized_float)

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
    eps: float = 1e-10,
):
    """Verify correctness by comparing reference and kernel implementations."""
    device = torch.device("xpu")

    x = torch.randn(batch_size * seq_len, hidden_dim, device=device, dtype=src_dtype)

    x_q_ref, x_s_ref = reference_per_token_group_quant_mxfp4(x.clone(), group_size, eps)
    x_q_sgl, x_s_sgl = sglang_per_token_group_quant_mxfp4(x.clone(), group_size, eps)

    # Compare quantized outputs directly (packed uint8 and scales).
    # Normalise signed zeros first: in E2M1 code 0x0 (+0.0) and 0x8
    # (-0.0) are semantically identical.  The kernel may preserve the
    # sign of the original float while the reference always emits +0.0,
    # so we canonicalise before comparing.
    x_q_ref_norm = _normalize_packed_fp4_signed_zero(x_q_ref.cpu())
    x_q_sgl_norm = _normalize_packed_fp4_signed_zero(x_q_sgl.cpu())
    q_match = torch.equal(x_q_ref_norm, x_q_sgl_norm)
    s_match = torch.equal(x_s_ref.cpu(), x_s_sgl.cpu())

    if q_match and s_match:
        print(
            f"  \u2705 Quantized values match (batch={batch_size}, seq={seq_len}, hidden={hidden_dim}, group={group_size}, dtype={src_dtype})"
        )
    else:
        q_mismatches = (x_q_ref_norm != x_q_sgl_norm).sum().item() if not q_match else 0
        s_mismatches = (
            (x_s_ref.cpu() != x_s_sgl.cpu()).sum().item() if not s_match else 0
        )
        print(
            f"  \u274c Quantized values differ: "
            f"packed_q({q_mismatches} mismatches) "
            f"scales({s_mismatches} mismatches)"
        )

    # Compare dequantized outputs
    x_dq_ref = dequantize_mxfp4(x_q_ref.cpu(), x_s_ref.cpu(), torch.float32, group_size)
    x_dq_sgl = dequantize_mxfp4(x_q_sgl.cpu(), x_s_sgl.cpu(), torch.float32, group_size)

    if torch.allclose(x_dq_ref, x_dq_sgl, rtol=0.2, atol=0.5):
        print(
            f"  \u2705 Dequantized values match (batch={batch_size}, seq={seq_len}, hidden={hidden_dim}, group={group_size}, dtype={src_dtype})"
        )
    else:
        max_diff = (x_dq_ref - x_dq_sgl).abs().max().item()
        print(f"  \u274c Dequantized values differ (max_diff={max_diff:.4f})")


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
# Only group_size=32 is supported for MXFP4 (per OCP MX spec block size)
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
