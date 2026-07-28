"""Benchmark JIT per-token-group 8-bit quant (v2) vs AOT implementation."""

import itertools

import pandas as pd
import torch
import triton

try:
    import sgl_kernel  # noqa: F401

    HAS_AOT = True
except ImportError:
    HAS_AOT = False
    print("Warning: sgl_kernel not available, AOT comparison will be skipped")

fp8_type_ = torch.float8_e4m3fn
fp8_max = torch.finfo(fp8_type_).max
fp8_min = -fp8_max

all_results = []


def ceil_div(x, y):
    return (x + y - 1) // y


def ceil_align(x, y):
    return ceil_div(x, y) * y


def create_ue8m0_scale(out_shape, group_size, device):
    # column-major + TMA-aligned + ue8m0 (uint32-packed), matching the AOT bench.
    *x_batch, x_q_mn, x_q_k = out_shape
    x_s_mn, x_s_k = x_q_mn, x_q_k // group_size
    aligned_mn = ceil_align(x_s_mn, 4)
    aligned_k = ceil_align(x_s_k, 4)
    return torch.empty(
        (*x_batch, aligned_k // 4, aligned_mn), device=device, dtype=torch.int
    ).transpose(-1, -2)[..., :x_s_mn, :]


def _quant(fn_kind, x, group_size):
    # fused silu+mul: input (num_tokens, 2H) -> output_q (num_tokens, H)
    out_shape = (*x.shape[:-1], x.shape[-1] // 2)
    x_q = torch.empty(out_shape, device=x.device, dtype=fp8_type_)
    x_s = create_ue8m0_scale(out_shape, group_size, x.device)
    if fn_kind == "aot":
        torch.ops.sgl_kernel.sgl_per_token_group_quant_8bit_v2.default(
            x, x_q, x_s, group_size, 1e-10, fp8_min, fp8_max, True, True, None
        )
    else:
        from sgl_kernel.jit import per_token_group_quant_8bit_v2 as _jit

        _jit(
            x,
            x_q,
            x_s,
            group_size,
            1e-10,
            fp8_min,
            fp8_max,
            scale_ue8m0=True,
            fuse_silu_and_mul=True,
            masked_m=None,
        )
    return x_q, x_s


configs = list(
    itertools.product(
        [16, 32, 64],  # batch_size
        [128, 512, 1024],  # seq_len
        [128],  # group_size (DeepSeek V3/R1)
    )
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "seq_len", "group_size"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["aot", "jit"],
        line_names=["AOT (sgl_kernel)", "JIT (sglang)"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="aot-vs-jit-per-token-group-quant-8bit-v2-performance",
        args={},
    )
)
def benchmark(batch_size, seq_len, group_size, provider):
    device = torch.device("xpu")
    hidden_dim = 7168
    num_tokens = batch_size * seq_len

    x = torch.randn(num_tokens, hidden_dim * 2, device=device, dtype=torch.bfloat16)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "aot":
        if not HAS_AOT:
            print("Warning: sgl_kernel AOT not available, skipping")
            return 0, 0, 0
        fn = lambda: _quant("aot", x, group_size)
    elif provider == "jit":
        try:
            fn = lambda: _quant("jit", x, group_size)
        except ImportError:
            print("Warning: sgl_kernel JIT module not available, skipping")
            return 0, 0, 0

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    all_results.append(
        {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_tokens": num_tokens,
            "hidden_dim": hidden_dim,
            "group_size": group_size,
            "provider": provider,
            "time_us": 1000 * ms,
        }
    )
    return 1000 * ms, 1000 * min_ms, 1000 * max_ms


if __name__ == "__main__":
    if not HAS_AOT:
        print("ERROR: sgl_kernel (AOT) not available. Please install sgl-kernel-xpu.")
        exit(1)

    print("Running AOT vs JIT per-token-group 8-bit v2 (fused silu) benchmarks...")
    print("AOT: sgl_kernel.sgl_per_token_group_quant_8bit_v2 (compiled SYCL kernels)")
    print("JIT: sgl_kernel.jit.per_token_group_quant_8bit_v2 (runtime JIT compilation)")
    print("\n" + "=" * 80 + "\n")

    benchmark.run(print_data=True)

    print("Benchmark finished!")
    df = pd.DataFrame(all_results)
    df["time_us"] = df["time_us"].round(2)
    print(df.to_markdown(index=False))
