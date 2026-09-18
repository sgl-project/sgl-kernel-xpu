from itertools import product

import pandas as pd
import torch
import triton
from sgl_kernel import fused_scale_residual_norm_scale_shift

# Diffusion DiT shapes: hidden 1536 (Wan2.1-1.3B), 3072 (FLUX), 5120 (Wan2.2-A14B);
# seq_len 32760 is Wan 720p (21 latent frames x 1560 tokens).
seq_len_range = [1024, 8192, 32760]
hidden_range = [1536, 3072, 5120]
dtype_range = [torch.bfloat16, torch.float16]

configs = list(product(seq_len_range, hidden_range, dtype_range))

all_results = []


def torch_scale_residual_norm_scale_shift(
    residual, x, gate, shift, scale, weight, bias, eps
):
    residual_output = residual + x * gate
    normed = torch.nn.functional.layer_norm(
        residual_output, (x.shape[-1],), weight, bias, eps
    )
    return normed * (1 + scale) + shift, residual_output


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len", "hidden", "dtype"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["sgl_kernel", "torch"],
        line_names=["SGL Kernel", "Torch (unfused)"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="Time (ms)",
        plot_name="scale-residual-norm-scale-shift-performance",
        args={},
    )
)
def benchmark(seq_len, hidden, dtype, provider):
    print(f"benchmark {provider} with seq_len={seq_len} hidden={hidden} dtype={dtype}")
    torch.set_default_device("xpu")
    torch.xpu.manual_seed_all(42)
    eps = 1e-6

    residual = torch.randn(1, seq_len, hidden, dtype=dtype)
    x = torch.randn(1, seq_len, hidden, dtype=dtype)
    gate = torch.randn(1, 1, hidden, dtype=dtype)
    shift = torch.randn(hidden, dtype=dtype)
    scale = torch.randn(hidden, dtype=dtype)
    weight = torch.randn(hidden, dtype=dtype)
    bias = torch.randn(hidden, dtype=dtype)

    if provider == "sgl_kernel":
        bench_lambda = lambda: fused_scale_residual_norm_scale_shift(
            residual=residual,
            x=x,
            gate=gate,
            shift=shift,
            scale=scale,
            weight=weight,
            bias=bias,
            eps=eps,
        )
    else:
        bench_lambda = lambda: torch_scale_residual_norm_scale_shift(
            residual, x, gate, shift, scale, weight, bias, eps
        )

    # Warmup
    for _ in range(10):
        bench_lambda()
    torch.xpu.synchronize()

    quantiles = [0.5, 0.25, 0.75]
    ms, _, _ = triton.testing.do_bench(
        bench_lambda, quantiles=quantiles, return_mode="median"
    )

    # Activations dominate traffic: read residual and x, write out and residual_output.
    total_bytes = 4 * x.numel() * x.element_size()
    bandwidth_gb_s = total_bytes / (ms / 1e3) / 1e9

    del residual, x, gate, shift, scale, weight, bias
    torch.xpu.empty_cache()

    all_results.append(
        {
            "seq_len": seq_len,
            "hidden": hidden,
            "dtype": str(dtype).removeprefix("torch."),
            "provider": provider,
            "bandwidth_gb_s": bandwidth_gb_s,
            "ms": ms,
        }
    )
    return ms


if __name__ == "__main__":
    benchmark.run(print_data=False)
    print("Benchmark finished!")

    df = pd.DataFrame(all_results)
    print(df.to_markdown())
