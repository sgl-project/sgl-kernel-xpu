# Op-level benchmark for the dense weight-only FP8 (W8A16) linear (fp8_linear) on
# BMG (Xe2).
# Providers:
#
#   fp8_tensor      fp8_linear, FP8 weights with one per-tensor scale.
#   fp8_channel     fp8_linear, FP8 weights with per-channel scales.
#   fp8_block       fp8_linear, FP8 weights with 128x128 block scales.
#   bf16            torch.matmul on the bf16 weights.
#   w8a8_rowwise    per-token activation quant (sgl_per_token_quant_fp8) followed by
#                   rowwise torch._scaled_mm: the online FP8 dense path the XPU
#                   backend runs without a weight-only kernel.
#
# Shapes are the dense projections of Qwen3.6-35B-A3B at TP=2 and TP=4 and Llama-3.1-8B at TP=1,
# at decode (small M) and prefill (large M) token counts.
#
# Each fp8_* row carries w8a8_rowwise as its reference when that provider runs, so
# speedup > 1 means fp8_linear is faster than the W8A8 path.
import argparse
import sys
from pathlib import Path

import torch
import triton
from _bench import BenchSpec, ProblemShape, format_tables, make_row, require_arch
from sgl_kernel import fp8_linear, sgl_per_token_quant_fp8

# The FP8 weight quantization helpers live next to the tests.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from fp8_utils import quantize_fp8_block  # noqa: E402
from fp8_utils import quantize_fp8_per_channel  # noqa: E402
from fp8_utils import quantize_fp8_per_tensor  # noqa: E402

# (N, K)
WEIGHT_SHAPES = {
    "Qwen3.6-35B-A3B-TP2": [(4608, 2048), (2048, 2048), (6144, 2048), (512, 2048)],
    "Qwen3.6-35B-A3B-TP4": [(2560, 2048), (2048, 1024), (3072, 2048), (256, 2048)],
    "Llama-3.1-8B": [(6144, 4096), (4096, 4096), (28672, 4096), (4096, 14336)],
}
M_VALUES = [1, 4, 16, 64, 256, 1024, 4096]
PROVIDERS = ["fp8_tensor", "fp8_channel", "fp8_block", "bf16", "w8a8_rowwise"]
REFERENCE = "w8a8_rowwise"

SPEC = BenchSpec(
    kernel="fp8_linear",
    shape=ProblemShape(
        ("m", "n", "k"),
        description="y[m, n] = x[m, k] (bf16) @ w[n, k]^T; fp8_* = fp8_linear "
        "with per-tensor / per-channel / 128x128 block scales",
    ),
)


def make_fn(provider, m, n, k):
    x = torch.randn(m, k, dtype=torch.bfloat16, device="xpu")
    w = torch.randn(n, k, dtype=torch.bfloat16, device="xpu") * 0.02
    if provider == "fp8_tensor":
        w_fp8, scale, _ = quantize_fp8_per_tensor(w)
        return lambda: fp8_linear(x, w_fp8, scale)
    if provider == "fp8_channel":
        w_fp8, scale, _ = quantize_fp8_per_channel(w)
        return lambda: fp8_linear(x, w_fp8, scale)
    if provider == "fp8_block":
        w_fp8, scale, _ = quantize_fp8_block(w)
        return lambda: fp8_linear(x, w_fp8, scale)
    if provider == "bf16":
        return lambda: torch.matmul(x, w.t())
    if provider == "w8a8_rowwise":
        w_fp8, w_scale, _ = quantize_fp8_per_channel(w)
        w_fp8_t = w_fp8.t()
        w_scale_t = w_scale.view(1, -1)
        x_q = torch.empty(m, k, dtype=torch.float8_e4m3fn, device="xpu")
        x_s = torch.empty(m, 1, dtype=torch.float32, device="xpu")

        def fn():
            sgl_per_token_quant_fp8(x, x_q, x_s)
            return torch._scaled_mm(
                x_q, w_fp8_t, scale_a=x_s, scale_b=w_scale_t, out_dtype=torch.bfloat16
            )

        return fn
    raise ValueError(provider)


def _time_us(fn):
    for _ in range(5):
        fn()
    torch.xpu.synchronize()
    return triton.testing.do_bench(fn, return_mode="median") * 1000


def _bytes(provider, m, n, k):
    # Activations and output in bf16; the weight in FP8 except for bf16.
    weight = n * k * (2 if provider == "bf16" else 1)
    return weight + 2 * m * k + 2 * m * n


def run(models, providers):
    rows = []
    for model in models:
        for n, k in WEIGHT_SHAPES[model]:
            for m in M_VALUES:
                times = {p: _time_us(make_fn(p, m, n, k)) for p in providers}
                for provider in providers:
                    has_ref = provider.startswith("fp8_") and REFERENCE in times
                    rows.append(
                        make_row(
                            SPEC,
                            dtype=(
                                torch.bfloat16
                                if provider == "bf16"
                                else torch.float8_e4m3fn
                            ),
                            size=(m, n, k),
                            time_us=times[provider],
                            bandwidth_gbs=_bytes(provider, m, n, k)
                            / times[provider]
                            / 1e3,
                            tflops=2 * m * n * k / times[provider] / 1e6,
                            provider=provider,
                            reference_provider=REFERENCE if has_ref else None,
                            reference_time_us=times[REFERENCE] if has_ref else None,
                        )
                    )
                torch.xpu.empty_cache()
    return rows


def main():
    require_arch("xe20")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", default=list(WEIGHT_SHAPES), choices=list(WEIGHT_SHAPES)
    )
    parser.add_argument("--providers", nargs="+", default=PROVIDERS, choices=PROVIDERS)
    args = parser.parse_args()

    print()
    print(format_tables(run(args.models, args.providers)))
    print()


if __name__ == "__main__":
    main()
