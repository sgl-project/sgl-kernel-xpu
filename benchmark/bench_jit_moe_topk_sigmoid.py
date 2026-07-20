"""Benchmark JIT fused sigmoid top-k MoE gate vs AOT implementation."""

import itertools
from typing import Optional

import pandas as pd
import torch
import triton

try:
    import sgl_kernel  # noqa: F401

    HAS_AOT = True
except ImportError:
    HAS_AOT = False
    print("Warning: sgl_kernel not available, AOT comparison will be skipped")

all_results = []


def _alloc_outputs(num_tokens, topk, device):
    w = torch.empty(num_tokens, topk, dtype=torch.float32, device=device)
    idx = torch.empty(num_tokens, topk, dtype=torch.int32, device=device)
    return w, idx


def aot_topk_sigmoid(gating, topk, renormalize, bias, rsf, nfse):
    w, idx = _alloc_outputs(gating.shape[0], topk, gating.device)
    torch.ops.sgl_kernel.topk_sigmoid.default(
        w, idx, gating, renormalize, bias, rsf, nfse
    )
    return w, idx


def jit_topk_sigmoid(gating, topk, renormalize, bias, rsf, nfse):
    from sgl_kernel.jit import topk_sigmoid as _jit_ts

    w, idx = _alloc_outputs(gating.shape[0], topk, gating.device)
    _jit_ts(w, idx, gating, renormalize, bias, rsf, nfse)
    return w, idx


# (num_tokens, num_experts, topk). Includes power-of-2 fast-path expert counts
# (8/32/128/256) and a non-power-of-2 fallback (12, 100).
configs = list(
    itertools.product(
        [128, 512, 4096, 16384],  # num_tokens
        [8, 32, 128, 256, 100],  # num_experts
        [6],  # topk (<= min(num_experts, 8))
    )
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens", "num_experts", "topk"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["aot", "jit"],
        line_names=["AOT (sgl_kernel)", "JIT (sglang)"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="aot-vs-jit-moe-topk-sigmoid-performance",
        args={},
    )
)
def benchmark(num_tokens, num_experts, topk, provider):
    device = torch.device("xpu")
    dtype = torch.bfloat16
    renormalize = True
    routed_scaling_factor = 1.0
    num_fused_shared_experts = 0
    correction_bias: Optional[torch.Tensor] = None

    gating = torch.randn(num_tokens, num_experts, device=device, dtype=dtype)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "aot":
        if not HAS_AOT:
            print("Warning: sgl_kernel AOT not available, skipping")
            return 0, 0, 0
        fn = lambda: aot_topk_sigmoid(
            gating,
            topk,
            renormalize,
            correction_bias,
            routed_scaling_factor,
            num_fused_shared_experts,
        )
    elif provider == "jit":
        try:
            fn = lambda: jit_topk_sigmoid(
                gating,
                topk,
                renormalize,
                correction_bias,
                routed_scaling_factor,
                num_fused_shared_experts,
            )
        except ImportError:
            print("Warning: sgl_kernel JIT module not available, skipping")
            return 0, 0, 0

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    all_results.append(
        {
            "num_tokens": num_tokens,
            "num_experts": num_experts,
            "topk": topk,
            "provider": provider,
            "time_us": 1000 * ms,
        }
    )

    return 1000 * ms, 1000 * min_ms, 1000 * max_ms


if __name__ == "__main__":
    if not HAS_AOT:
        print("ERROR: sgl_kernel (AOT) not available. Please install sgl-kernel-xpu.")
        exit(1)

    print("Running AOT vs JIT fused sigmoid top-k MoE gate benchmarks...")
    print("AOT: sgl_kernel.topk_sigmoid (compiled SYCL kernels)")
    print("JIT: sgl_kernel.jit.topk_sigmoid (runtime JIT compilation)")
    print("\n" + "=" * 80 + "\n")

    benchmark.run(print_data=True)

    print("Benchmark finished!")
    df = pd.DataFrame(all_results)
    df["time_us"] = df["time_us"].round(2)
    print(df.to_markdown(index=False))
