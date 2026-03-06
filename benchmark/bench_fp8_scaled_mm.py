"""
Copyright (C) 2026 Intel Corporation, All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import itertools

import pandas as pd
import torch
import torch.nn.functional as F
import triton.testing as tt
from sgl_kernel import fp8_scaled_mm

# Problem ranges
M_range = [128, 256, 512]
N_range = [128, 256, 512]
K_range = [64, 128, 256]
out_dtype_range = [torch.float32, torch.bfloat16, torch.float16]
with_bias_range = [False, True]

configs = list(
    itertools.product(M_range, N_range, K_range, out_dtype_range, with_bias_range)
)
all_results = []


def calc_flops(M: int, N: int, K: int) -> int:
    # GEMM: 2 * M * N * K (multiply + add)
    return 2 * M * N * K


def calc_bandwidth(
    M: int, N: int, K: int, out_dtype: torch.dtype, with_bias: bool, time_ms: float
) -> dict:
    # Memory traffic:
    # A: M*K (fp8, 1 byte)
    # B: N*K (fp8, 1 byte)
    # scale_a: M (fp16, 2 bytes)
    # scale_b: N (fp16, 2 bytes)
    # D: M*N (out dtype)
    # bias: M (out dtype) if present
    out_elem_bytes = torch.finfo(out_dtype).bits // 8
    bytes_a = M * K * 1
    bytes_b = N * K * 1
    bytes_sa = M * 2
    bytes_sb = N * 2
    bytes_d = M * N * out_elem_bytes
    bytes_bias = M * out_elem_bytes if with_bias else 0
    total_bytes = bytes_a + bytes_b + bytes_sa + bytes_sb + bytes_d + bytes_bias

    time_s = time_ms / 1e3
    bandwidth_gbs = (total_bytes / 1e9) / time_s

    total_flops = calc_flops(M, N, K)
    gflops = (total_flops / 1e9) / time_s
    return {
        "total_bytes": total_bytes,
        "bandwidth_gbs": bandwidth_gbs,
        "total_flops": total_flops,
        "gflops": gflops,
    }


@tt.perf_report(
    tt.Benchmark(
        x_names=["M", "N", "K", "out_dtype", "with_bias"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["sglang"],
        line_names=["SGL Kernel"],
        styles=[("green", "-")],
        ylabel="us",
        plot_name="fp8-scaled-mm-performance",
        args={},
    )
)
def benchmark(M, N, K, out_dtype, with_bias, provider):
    device = torch.device("xpu")

    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    # Inputs: FP8 e4m3 as required by the kernel
    a_fp32 = (torch.rand(M, K, dtype=torch.float32, device=device) - 0.5) * fp8_max
    A = a_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    b_fp32 = (torch.rand(N, K, dtype=torch.float32, device=device) - 0.5) * fp8_max
    B = b_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    # Scales from amax (always positive, physically meaningful)
    amax_a = A.to(torch.float32).abs().amax(dim=1)
    scale_a = (amax_a / fp8_max).to(torch.float32)

    amax_b = B.to(torch.float32).abs().amax(dim=1)
    scale_b = (amax_b / fp8_max).to(torch.float32)

    # Bias in out_dtype (matches kernel ElementBias = ElementOutput)
    bias = torch.randn((M,), device=device, dtype=out_dtype) if with_bias else None

    quantiles = [0.5, 0.2, 0.8]

    if provider == "sglang":
        fn = lambda: fp8_scaled_mm(A, B, scale_a, scale_b, out_dtype, bias)
    else:
        raise ValueError(f"Unknown provider {provider}")

    ms, min_ms, max_ms = tt.do_bench(fn, quantiles=quantiles)

    bw_metrics = calc_bandwidth(M, N, K, out_dtype, with_bias, ms)

    all_results.append(
        {
            "M": M,
            "N": N,
            "K": K,
            "out_dtype": str(out_dtype),
            "with_bias": with_bias,
            "provider": provider,
            "time_us": 1e3 * ms,
            "bandwidth_gbs": bw_metrics["bandwidth_gbs"],
            "total_bytes_mb": bw_metrics["total_bytes"] / 1e6,
            "total_flops_m": bw_metrics["total_flops"] / 1e6,
            "gflops": bw_metrics["gflops"],
        }
    )

    return 1e3 * ms, 1e3 * max_ms, 1e3 * min_ms


if __name__ == "__main__":
    # Smoke correctness check on a small case before perf
    device = "xpu"
    M, N, K = 128, 256, 64

    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    a_fp32 = (torch.rand(M, K, dtype=torch.float32, device=device) - 0.5) * fp8_max
    A = a_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    b_fp32 = (torch.rand(N, K, dtype=torch.float32, device=device) - 0.5) * fp8_max
    B = b_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    amax_a = A.to(torch.float32).abs().amax(dim=1)
    scale_a = (amax_a / fp8_max).to(torch.float32)

    amax_b = B.to(torch.float32).abs().amax(dim=1)
    scale_b = (amax_b / fp8_max).to(torch.float32)

    out = fp8_scaled_mm(A, B, scale_a, scale_b, torch.float32, None)

    # Reference on CPU
    a_cpu = A.to(torch.float32).cpu()
    b_cpu = B.to(torch.float32).cpu()
    scale_a_cpu = scale_a.to(torch.float16).cpu()
    scale_b_cpu = scale_b.to(torch.float16).cpu()

    ref = F.linear(a_cpu, b_cpu)
    ref = ref * scale_a_cpu.view(-1, 1) * scale_b_cpu.view(1, -1)

    assert torch.allclose(
        out.cpu(), ref, rtol=1e-3, atol=1e-3
    ), f"Correctness check failed! Max diff: {(out.cpu() - ref).abs().max()}"
    print("Smoke correctness check passed.")

    benchmark.run(print_data=True)

    print("\n" + "=" * 80)
    print("Effective Bandwidth / GFLOPs")
    print("=" * 80)
    df = pd.DataFrame(all_results)
    df["bandwidth_gbs"] = df["bandwidth_gbs"].round(2)
    df["total_bytes_mb"] = df["total_bytes_mb"].round(2)
    df["time_us"] = df["time_us"].round(2)
    df["total_flops_m"] = df["total_flops_m"].round(2)
    df["gflops"] = df["gflops"].round(2)
    print(df.to_markdown(index=False))

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
