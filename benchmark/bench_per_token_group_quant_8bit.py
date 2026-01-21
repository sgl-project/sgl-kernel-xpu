import itertools
from typing import Tuple

import pandas as pd
import torch
import triton
import triton.language as tl
from sgl_kernel import sgl_per_token_group_quant_fp8, sgl_per_token_group_quant_int8

fp8_type_ = torch.float8_e4m3fn


@triton.jit
def _per_token_group_quant_8bit(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    # Stride of input
    y_stride,
    # Columns of input
    N,
    # Avoid to divide zero
    eps,
    # Information for 8bit data type (int8 or fp8_type_)
    max_8bit,
    min_8bit,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group quantization on a
    tensor.
    This function converts the tensor values into 8bit values.
    """
    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    y_ptr += g_id * y_stride
    y_q_ptr += g_id * y_stride
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < N

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / max_8bit
    y_q = tl.clamp(y / y_s, min_8bit, max_8bit).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def triton_per_token_group_quant_8bit(
    x: torch.Tensor,
    group_size: int,
    dst_dtype: torch.dtype,
    eps: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Function to perform per-token-group quantization on an input tensor `x`.
    It converts the tensor values into signed float8 values and returns the
    quantized tensor along with the scaling factor used for quantization.
    Args:
        x: The input tenosr with ndim >= 2.
        group_size: The group size used for quantization.
        eps: The minimum to avoid dividing zero.
        dtype: The dype of output tensor. Note that only `torch.float8_e4m3fn` is supported for now.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the scaling factor for quantization.
    """
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    if dst_dtype == torch.int8:
        iinfo = torch.iinfo(dst_dtype)
        max_8bit = iinfo.max
        min_8bit = iinfo.min
    else:
        finfo = torch.finfo(dst_dtype)
        max_8bit = finfo.max
        min_8bit = finfo.min

    x_q = torch.empty_like(x, device=x.device, dtype=dst_dtype)
    M = x.numel() // group_size
    N = group_size
    x_s = torch.empty(
        x.shape[:-1] + (x.shape[-1] // group_size,),
        device=x.device,
        dtype=torch.float32,
    )

    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1
    _per_token_group_quant_8bit[(M,)](
        x,
        x_q,
        x_s,
        group_size,
        N,
        eps,
        max_8bit,
        min_8bit,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return x_q, x_s


def sglang_per_token_group_quant_8bit(
    x: torch.Tensor,
    group_size: int,
    dst_dtype: torch.dtype,
    eps: float = 1e-10,
):
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    if dst_dtype == torch.int8:
        iinfo = torch.iinfo(dst_dtype)
        x_q, x_s = sgl_per_token_group_quant_int8(x, group_size, eps)
    else:
        f8_info = torch.finfo(dst_dtype)
        x_q, x_s = sgl_per_token_group_quant_fp8(x, group_size, eps)

    return x_q, x_s


def calculate_diff(batch_size, seq_len, group_size, dst_dtype):
    device = torch.device("xpu")
    hidden_dim = 7168

    x = torch.randn(
        batch_size * seq_len, hidden_dim, device=device, dtype=torch.bfloat16
    )

    x_q_triton, x_s_triton = triton_per_token_group_quant_8bit(
        x.clone(), group_size, dst_dtype
    )
    x_q_sglang, x_s_sglang = sglang_per_token_group_quant_8bit(
        x.clone(), group_size, dst_dtype
    )

    # Dequantize and compare values (convert to float for comparison)
    x_dq_triton = (
        x_q_triton.cpu().view(batch_size * seq_len, -1, group_size).to(torch.float32)
        * x_s_triton.cpu().unsqueeze(2)
    ).view(batch_size * seq_len, hidden_dim)
    x_dq_sglang = (
        x_q_sglang.cpu().view(batch_size * seq_len, -1, group_size).to(torch.float32)
        * x_s_sglang.cpu().unsqueeze(2)
    ).view(batch_size * seq_len, hidden_dim)

    if torch.allclose(
        x_dq_triton, x_dq_sglang, rtol=1e-1, atol=1e-1
    ) and torch.allclose(x_s_triton, x_s_sglang, rtol=1e-3, atol=1e-5):
        print(f"✅ {dst_dtype} implementations match")
    else:
        print("❌ Implementations differ")


batch_size_range = [1, 2, 4, 8, 16, 32, 64]
seq_len_range = [64, 128, 256, 512, 1024, 2048]
group_size_range = [128]  # For DeepSeek V3/R1
dst_dtype_range = [torch.int8, fp8_type_]

configs = list(
    itertools.product(
        batch_size_range, seq_len_range, group_size_range, dst_dtype_range
    )
)

all_results = []


def calculate_flops(
    num_elements: int,
    num_groups: int,
    group_size: int,
) -> int:
    """
    Calculate FLOPs for per-token-group quantization kernel.

    Per element: 5 FLOPs (2 absmax: fabs+fmax, 3 quant: mul+fmax+fmin)
    Per group: 6 FLOPs (4 reduction fmax, 2 scale divisions)
    """
    flops_per_element = 5  # 2 for absmax + 3 for quantization
    flops_per_group = 6  # 4 for reduction + 2 for scale calculation

    total_flops = (num_elements * flops_per_element) + (num_groups * flops_per_group)

    return total_flops


def calculate_effective_bandwidth(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    group_size: int,
    dst_dtype: torch.dtype,
    time_ms: float,
) -> dict:
    """
    Calculate effective bandwidth and FLOPs for per-token-group quantization kernel.

    Memory: read bf16 input + write int8/fp8 output + write fp32 scales (single-pass).
    """
    num_tokens = batch_size * seq_len
    num_elements = num_tokens * hidden_dim
    num_groups = num_elements // group_size

    input_bytes = num_elements * 2  # bf16
    output_bytes = num_elements * 1  # int8/fp8
    scale_bytes = num_groups * 4  # fp32
    total_bytes = input_bytes + output_bytes + scale_bytes

    time_s = time_ms / 1000.0
    bandwidth_gbs = (total_bytes / 1e9) / time_s

    total_flops = calculate_flops(num_elements, num_groups, group_size)
    gflops = (total_flops / 1e9) / time_s

    return {
        "num_tokens": num_tokens,
        "num_elements": num_elements,
        "num_groups": num_groups,
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "scale_bytes": scale_bytes,
        "total_bytes": total_bytes,
        "bandwidth_gbs": bandwidth_gbs,
        "total_flops": total_flops,
        "gflops": gflops,
    }


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "seq_len", "group_size", "dst_dtype"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["triton", "sglang"],
        line_names=["Triton", "SGL Kernel"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="per-token-group-quant-8bit-performance",
        args={},
    )
)
def benchmark(batch_size, seq_len, group_size, dst_dtype, provider):
    device = torch.device("xpu")
    hidden_dim = 7168

    x = torch.randn(
        batch_size * seq_len, hidden_dim, device=device, dtype=torch.bfloat16
    )

    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        fn = lambda: triton_per_token_group_quant_8bit(x, group_size, dst_dtype)
    elif provider == "sglang":
        fn = lambda: sglang_per_token_group_quant_8bit(x, group_size, dst_dtype)

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    # Calculate effective bandwidth
    bw_metrics = calculate_effective_bandwidth(
        batch_size, seq_len, hidden_dim, group_size, dst_dtype, ms
    )

    all_results.append(
        {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_tokens": bw_metrics["num_tokens"],
            "hidden_dim": hidden_dim,
            "group_size": group_size,
            "dst_dtype": str(dst_dtype),
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

    calculate_diff(batch_size=4, seq_len=128, group_size=64, dst_dtype=torch.int8)
    calculate_diff(batch_size=2, seq_len=32, group_size=128, dst_dtype=fp8_type_)

    benchmark.run(print_data=True)

    # Print bandwidth results
    print("\n" + "=" * 80)
    print("Effective Bandwidth Results")
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
