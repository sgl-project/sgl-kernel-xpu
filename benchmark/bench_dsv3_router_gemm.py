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
import triton
import triton.testing as tt
import torch.nn.functional as F


# Kernel wrapper (uses the XPU implementation)
def dsv3_router_gemm(hidden_states: torch.Tensor,
                     router_weights: torch.Tensor,
                     out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    output = torch.empty(
        hidden_states.shape[0],
        router_weights.shape[0],
        device=hidden_states.device,
        dtype=out_dtype,
    )
    torch.ops.sgl_kernel.dsv3_router_gemm(output, hidden_states, router_weights)
    return output


# Problem ranges
num_tokens_range = [1, 4, 16]
num_experts_range = [256, 384]
hidden_dim = 7168
out_dtype_range = [torch.float32, torch.bfloat16, torch.float16]

configs = list(itertools.product(num_tokens_range, num_experts_range, out_dtype_range))
all_results = []


def calc_flops(M: int, N: int, K: int) -> int:
    # GEMM: 2 * M * N * K (multiply + add)
    return 2 * M * N * K


def calc_bandwidth(M: int, N: int, K: int, out_dtype: torch.dtype, time_ms: float) -> dict:
    # Approximate memory traffic:
    # A: M*K (bf16, 2 bytes)
    # B: N*K (bf16, 2 bytes)
    # D: M*N (out dtype)
    bytes_a = M * K * 2
    bytes_b = N * K * 2
    bytes_d = M * N * (torch.finfo(out_dtype).bits // 8)
    total_bytes = bytes_a + bytes_b + bytes_d

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
        x_names=["num_tokens", "num_experts", "out_dtype"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["sglang"],
        line_names=["SGL Kernel"],
        styles=[("green", "-")],
        ylabel="us",
        plot_name="dsv3-router-gemm-performance",
        args={},
    )
)
def benchmark(num_tokens, num_experts, out_dtype, provider):
    device = torch.device("xpu")
    M, N, K = num_tokens, num_experts, hidden_dim

    # Inputs: bf16 as required by the kernel
    A = torch.randn((M, K), device=device, dtype=torch.bfloat16).contiguous()
    B = torch.randn((N, K), device=device, dtype=torch.bfloat16).contiguous()

    quantiles = [0.5, 0.2, 0.8]

    if provider == "sglang":
        fn = lambda: dsv3_router_gemm(A, B, out_dtype=out_dtype)
    else:
        raise ValueError(f"Unknown provider {provider}")

    ms, min_ms, max_ms = tt.do_bench(fn, quantiles=quantiles)

    bw_metrics = calc_bandwidth(M, N, K, out_dtype, ms)

    all_results.append(
        {
          "num_tokens": M,
          "num_experts": N,
          "hidden_dim": K,
          "out_dtype": str(out_dtype),
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
    M, N, K = 4, 256, hidden_dim
    A = torch.randn((M, K), device="xpu", dtype=torch.bfloat16).contiguous()
    B = torch.randn((N, K), device="xpu", dtype=torch.bfloat16).contiguous()

    out = dsv3_router_gemm(A, B, out_dtype=torch.float32)

    # Reference on CPU using linear (weight shape [N, K])
    ref = F.linear(A.float().cpu(), B.float().cpu())
    assert torch.allclose(out.cpu(), ref, rtol=1e-3, atol=1e-3), "Correctness check failed"

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
