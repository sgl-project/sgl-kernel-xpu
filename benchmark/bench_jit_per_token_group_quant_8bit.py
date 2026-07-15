"""Benchmark JIT per-token-group 8-bit quantization vs AOT implementation."""

import itertools
from typing import Tuple

import pandas as pd
import torch
import triton

try:
    import sgl_kernel
    from sgl_kernel import sgl_per_token_group_quant_fp8, sgl_per_token_group_quant_int8

    HAS_AOT = True
except ImportError:
    HAS_AOT = False
    print("Warning: sgl_kernel not available, AOT comparison will be skipped")

fp8_type_ = torch.float8_e4m3fn

all_results = []


def calculate_flops(num_elements: int, num_groups: int, group_size: int) -> int:
    """Per element: 5 FLOPs (fabs+fmax, mul+fmax+fmin); per group: 6 FLOPs."""
    return num_elements * 5 + num_groups * 6


def calculate_effective_bandwidth(
    num_tokens: int,
    hidden_dim: int,
    group_size: int,
    time_ms: float,
) -> dict:
    """Read bf16 input + write int8/fp8 output + write fp32 scales (single pass)."""
    num_elements = num_tokens * hidden_dim
    num_groups = num_elements // group_size

    input_bytes = num_elements * 2  # bf16
    output_bytes = num_elements * 1  # int8/fp8
    scale_bytes = num_groups * 4  # fp32
    total_bytes = input_bytes + output_bytes + scale_bytes

    time_s = time_ms / 1000.0
    bandwidth_gbs = (total_bytes / 1e9) / time_s
    gflops = (calculate_flops(num_elements, num_groups, group_size) / 1e9) / time_s

    return {
        "num_groups": num_groups,
        "total_bytes": total_bytes,
        "bandwidth_gbs": bandwidth_gbs,
        "gflops": gflops,
    }


def _minmax(dst_dtype: torch.dtype) -> Tuple[float, float]:
    info = torch.iinfo(dst_dtype) if dst_dtype == torch.int8 else torch.finfo(dst_dtype)
    return float(info.min), float(info.max)


def aot_per_token_group_quant_8bit(x, group_size, dst_dtype, eps=1e-10):
    m, n = x.shape
    x_q = torch.empty_like(x, dtype=dst_dtype)
    x_s = torch.empty((m, n // group_size), device=x.device, dtype=torch.float32)
    min_8bit, max_8bit = _minmax(dst_dtype)
    if dst_dtype == torch.int8:
        sgl_per_token_group_quant_int8(x, x_q, x_s, group_size, eps, min_8bit, max_8bit)
    else:
        sgl_per_token_group_quant_fp8(x, x_q, x_s, group_size, eps, min_8bit, max_8bit)
    return x_q, x_s


def jit_per_token_group_quant_8bit(x, group_size, dst_dtype, eps=1e-10):
    from sgl_kernel.jit import per_token_group_quant_8bit as _jit_q

    m, n = x.shape
    x_q = torch.empty_like(x, dtype=dst_dtype)
    x_s = torch.empty((m, n // group_size), device=x.device, dtype=torch.float32)
    min_8bit, max_8bit = _minmax(dst_dtype)
    _jit_q(x, x_q, x_s, group_size, eps, min_8bit, max_8bit, scale_ue8m0=False)
    return x_q, x_s


def calculate_diff(num_tokens, group_size, dst_dtype):
    """Sanity-check JIT vs AOT agree before benchmarking."""
    device = torch.device("xpu")
    hidden_dim = 7168
    x = torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.bfloat16)

    q_aot, s_aot = aot_per_token_group_quant_8bit(x.clone(), group_size, dst_dtype)
    q_jit, s_jit = jit_per_token_group_quant_8bit(x.clone(), group_size, dst_dtype)

    dq = lambda q, s: (
        q.cpu().view(num_tokens, -1, group_size).to(torch.float32)
        * s.cpu().unsqueeze(2)
    ).view(num_tokens, hidden_dim)

    if torch.allclose(
        dq(q_aot, s_aot), dq(q_jit, s_jit), rtol=1e-1, atol=1e-1
    ) and torch.allclose(s_aot, s_jit, rtol=1e-3, atol=1e-5):
        print(f"✅ {dst_dtype} JIT and AOT match")
    else:
        print(f"❌ {dst_dtype} JIT and AOT differ")


batch_size_range = [1, 2, 4, 8, 16, 32, 64]
seq_len_range = [64, 128, 256, 512, 1024, 2048]
group_size_range = [128]  # DeepSeek V3/R1
dst_dtype_range = [torch.int8, fp8_type_]

configs = list(
    itertools.product(
        batch_size_range, seq_len_range, group_size_range, dst_dtype_range
    )
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "seq_len", "group_size", "dst_dtype"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["aot", "jit"],
        line_names=["AOT (sgl_kernel)", "JIT (sglang)"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="aot-vs-jit-per-token-group-quant-8bit-performance",
        args={},
    )
)
def benchmark(batch_size, seq_len, group_size, dst_dtype, provider):
    device = torch.device("xpu")
    hidden_dim = 7168
    num_tokens = batch_size * seq_len

    x = torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.bfloat16)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "aot":
        if not HAS_AOT:
            print("Warning: sgl_kernel AOT not available, skipping")
            return 0, 0, 0
        fn = lambda: aot_per_token_group_quant_8bit(x, group_size, dst_dtype)
    elif provider == "jit":
        try:
            fn = lambda: jit_per_token_group_quant_8bit(x, group_size, dst_dtype)
        except ImportError:
            print("Warning: sgl_kernel JIT module not available, skipping")
            return 0, 0, 0

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    bw = calculate_effective_bandwidth(num_tokens, hidden_dim, group_size, ms)
    all_results.append(
        {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_tokens": num_tokens,
            "hidden_dim": hidden_dim,
            "group_size": group_size,
            "dst_dtype": str(dst_dtype),
            "provider": provider,
            "time_us": 1000 * ms,
            "bandwidth_gbs": bw["bandwidth_gbs"],
            "total_bytes_mb": bw["total_bytes"] / 1e6,
            "gflops": bw["gflops"],
        }
    )

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    if not HAS_AOT:
        print("ERROR: sgl_kernel (AOT) not available. Please install sgl-kernel-xpu.")
        exit(1)

    print("Running AOT vs JIT per-token-group 8-bit quant benchmarks...")
    print("AOT: sgl_kernel.sgl_per_token_group_quant_{int8,fp8}")
    print("JIT: sgl_kernel.jit.per_token_group_quant_8bit")
    print("\n" + "=" * 80 + "\n")

    calculate_diff(num_tokens=512, group_size=64, dst_dtype=torch.int8)
    calculate_diff(num_tokens=64, group_size=128, dst_dtype=fp8_type_)

    benchmark.run(print_data=True)

    print("\n" + "=" * 80)
    print("Effective Bandwidth Results")
    print("=" * 80)
    df = pd.DataFrame(all_results)
    for col in ("bandwidth_gbs", "total_bytes_mb", "time_us", "gflops"):
        df[col] = df[col].round(2)
    print(df.to_markdown(index=False))
