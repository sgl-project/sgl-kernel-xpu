import itertools

import pandas as pd
import torch
import triton
import triton.testing
from sgl_kernel import bmm_fp8

from sglang.srt.utils import is_hip

_is_hip = is_hip()
fp8_e4m3_type = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn
fp8_e5m2_type = torch.float8_e5m2fnuz if _is_hip else torch.float8_e5m2


def to_float8(x, dtype=torch.float8_e4m3fn):
    """Convert tensor to float8 with scaling."""
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()


def torch_bmm_fp8(
    input_fp8: torch.Tensor,
    mat2_fp8: torch.Tensor,
    input_inv_s: torch.Tensor,
    mat2_inv_s: torch.Tensor,
    res_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Baseline implementation using torch.bmm."""
    # Convert FP8 to bfloat16 and descale
    input_bf16 = input_fp8.to(torch.bfloat16) * input_inv_s
    mat2_bf16 = mat2_fp8.to(torch.bfloat16) * mat2_inv_s
    return torch.bmm(input_bf16, mat2_bf16).to(res_dtype)


def sglang_bmm_fp8(
    input_fp8: torch.Tensor,
    mat2_fp8: torch.Tensor,
    input_inv_s: torch.Tensor,
    mat2_inv_s: torch.Tensor,
    res_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """SGL Kernel implementation of BMM FP8."""
    # Prepare output
    batch_size, m, k = input_fp8.shape
    _, _, n = mat2_fp8.shape
    res = torch.empty([batch_size, m, n], device=input_fp8.device, dtype=res_dtype)

    # Run kernel
    bmm_fp8(input_fp8, mat2_fp8, input_inv_s, mat2_inv_s, res_dtype, res)

    return res


def calculate_diff(
    batch_size: int, m: int, k: int, n: int, input_dtype, mat2_dtype, res_dtype
):
    """Calculate difference between torch.bmm and SGL Kernel implementations."""
    device = torch.device("xpu")

    input = torch.randn([batch_size, m, k], dtype=torch.bfloat16, device=device)
    # mat2 in column-major format
    mat2 = torch.randn(
        [batch_size, n, k], dtype=torch.bfloat16, device=device
    ).transpose(-2, -1)

    # Convert to FP8
    input_fp8, input_inv_s = to_float8(input, dtype=input_dtype)
    mat2_fp8, mat2_inv_s = to_float8(mat2, dtype=mat2_dtype)

    torch_out = torch_bmm_fp8(
        input_fp8, mat2_fp8, input_inv_s, mat2_inv_s, res_dtype=res_dtype
    )
    sglang_out = sglang_bmm_fp8(
        input_fp8, mat2_fp8, input_inv_s, mat2_inv_s, res_dtype=res_dtype
    )

    output_diff = torch.abs(torch_out - sglang_out).mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        torch_out.reshape(-1), sglang_out.reshape(-1), dim=0
    ).item()

    print(f"Mean absolute difference: {output_diff:.6f}")
    print(f"Cosine similarity: {cos_sim:.6f}")

    if cos_sim > 0.99:
        print(f"✅ {input_dtype}/{mat2_dtype} -> {res_dtype} implementations match")
    else:
        print(f"❌ {input_dtype}/{mat2_dtype} -> {res_dtype} implementations differ")


batch_size_range = [1, 4, 8, 16, 32]
m_range = [32, 64, 128, 256]
k_range = [64, 128, 256, 512]
n_range = [64, 128, 256, 512]

# Create configurations (batch_size, m, k, n)
configs = list(itertools.product(batch_size_range, m_range, [128], [128]))

all_results = []


def calculate_flops(
    batch_size: int,
    m: int,
    k: int,
    n: int,
) -> int:
    """
    Calculate FLOPs for batch matrix multiplication.

    For each batch element: 2 * m * k * n FLOPs (multiply-add)
    Total: batch_size * 2 * m * k * n
    """
    flops_per_batch = 2 * m * k * n
    total_flops = batch_size * flops_per_batch

    return total_flops


def calculate_effective_bandwidth(
    batch_size: int,
    m: int,
    k: int,
    n: int,
    input_dtype: torch.dtype,
    mat2_dtype: torch.dtype,
    res_dtype: torch.dtype,
    time_ms: float,
) -> dict:
    """
    Calculate effective bandwidth and FLOPs for BMM FP8 kernel.

    Memory:
    - Input: batch_size * m * k (fp8)
    - Mat2: batch_size * k * n (fp8)
    - Scales: batch_size * 2 (fp32)
    - Output: batch_size * m * n (fp16/bf16)
    """
    # Input and mat2 are fp8 (1 byte each)
    input_bytes = batch_size * m * k * 1
    mat2_bytes = batch_size * k * n * 1
    scale_bytes = batch_size * 2 * 4  # 2 scales per batch (fp32)

    # Output is fp16 or bf16 (2 bytes)
    output_bytes = batch_size * m * n * 2

    total_bytes = input_bytes + mat2_bytes + scale_bytes + output_bytes

    time_s = time_ms / 1000.0
    bandwidth_gbs = (total_bytes / 1e9) / time_s

    total_flops = calculate_flops(batch_size, m, k, n)
    gflops = (total_flops / 1e9) / time_s

    return {
        "batch_size": batch_size,
        "m": m,
        "k": k,
        "n": n,
        "input_bytes": input_bytes,
        "mat2_bytes": mat2_bytes,
        "scale_bytes": scale_bytes,
        "output_bytes": output_bytes,
        "total_bytes": total_bytes,
        "bandwidth_gbs": bandwidth_gbs,
        "total_flops": total_flops,
        "gflops": gflops,
    }


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "m", "k", "n"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["torch", "sglang"],
        line_names=["Torch BMM", "SGL Kernel"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="bmm-fp8-performance-v2",
        args={},
    )
)
def benchmark_bmm_fp8(batch_size, m, k, n, provider):
    device = torch.device("xpu")
    res_dtype = torch.float16
    input_dtype = fp8_e4m3_type
    mat2_dtype = fp8_e4m3_type

    input = torch.randn([batch_size, m, k], device=device, dtype=torch.bfloat16)
    # mat2 in column-major format
    mat2 = torch.randn(
        [batch_size, n, k], device=device, dtype=torch.bfloat16
    ).transpose(-2, -1)

    # Convert to FP8
    input_fp8, input_inv_s = to_float8(input, dtype=input_dtype)
    mat2_fp8, mat2_inv_s = to_float8(mat2, dtype=mat2_dtype)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        fn = lambda: torch_bmm_fp8(
            input_fp8, mat2_fp8, input_inv_s, mat2_inv_s, res_dtype
        )
    elif provider == "sglang":
        fn = lambda: sglang_bmm_fp8(
            input_fp8, mat2_fp8, input_inv_s, mat2_inv_s, res_dtype=res_dtype
        )

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    # Calculate effective bandwidth and FLOPs
    bw_metrics = calculate_effective_bandwidth(
        batch_size, m, k, n, input_dtype, mat2_dtype, res_dtype, ms
    )

    all_results.append(
        {
            "batch_size": batch_size,
            "m": m,
            "k": k,
            "n": n,
            "input_dtype": str(input_dtype),
            "mat2_dtype": str(mat2_dtype),
            "res_dtype": str(res_dtype),
            "provider": provider,
            "time_us": 1000 * ms,
            "bandwidth_gbs": bw_metrics["bandwidth_gbs"],
            "total_bytes_mb": bw_metrics["total_bytes"] / 1e6,
            "total_flops_m": bw_metrics["total_flops"] / 1e6,
            "gflops": bw_metrics["gflops"],
        }
    )

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    # Test correctness with different dtype combinations
    calculate_diff(
        batch_size=16,
        m=48,
        k=64,
        n=80,
        input_dtype=fp8_e4m3_type,
        mat2_dtype=fp8_e4m3_type,
        res_dtype=torch.float16,
    )

    calculate_diff(
        batch_size=8,
        m=32,
        k=128,
        n=128,
        input_dtype=fp8_e4m3_type,
        mat2_dtype=fp8_e4m3_type,
        res_dtype=torch.bfloat16,
    )

    benchmark_bmm_fp8.run(print_data=True)

    # Print bandwidth results
    print("\n" + "=" * 80)
    print("Effective Bandwidth and FLOPS Results")
    print("=" * 80)

    df = pd.DataFrame(all_results)
    df["bandwidth_gbs"] = df["bandwidth_gbs"].round(2)
    df["total_bytes_mb"] = df["total_bytes_mb"].round(2)
    df["time_us"] = df["time_us"].round(2)
    df["total_flops_m"] = df["total_flops_m"].round(2)
    df["gflops"] = df["gflops"].round(2)

    print(df.to_markdown(index=False))

    # Print summary statistics per provider
    print("\n" + "=" * 80)
    print("Summary Statistics by Provider")
    print("=" * 80)
    summary = df.groupby("provider").agg(
        {
            "bandwidth_gbs": ["mean", "min", "max"],
            "time_us": ["mean", "min", "max"],
            "gflops": ["mean", "min", "max"],
        }
    )
    print(summary.to_markdown())
