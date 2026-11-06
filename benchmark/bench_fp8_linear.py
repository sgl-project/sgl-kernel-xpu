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
# Shapes are the dense projections of Qwen3.6-35B-A3B at TP=2 and Llama-3.1-8B at TP=1,
# at decode (small M) and prefill (large M) token counts.
import argparse

import torch
import triton
from _bench import require_arch
from sgl_kernel import fp8_linear, sgl_per_token_quant_fp8

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max

# (N, K)
WEIGHT_SHAPES = {
    "Qwen3.6-35B-A3B-TP2": [(4608, 2048), (2048, 2048), (6144, 2048), (512, 2048)],
    "Llama-3.1-8B": [(6144, 4096), (4096, 4096), (28672, 4096), (4096, 14336)],
}
M_VALUES = [1, 4, 16, 64, 256, 1024, 4096]
PROVIDERS = ["fp8_tensor", "fp8_channel", "fp8_block", "bf16", "w8a8_rowwise"]


def _quant_tensor(w):
    scale = w.abs().max().float() / FP8_MAX
    return (w.float() / scale).to(torch.float8_e4m3fn), scale.reshape(1)


def _quant_block(w):
    n, k = w.shape
    blocks = w.float().view(n // 128, 128, k // 128, 128)
    scale = blocks.abs().amax(dim=(1, 3)) / FP8_MAX
    w_fp8 = (blocks / scale[:, None, :, None]).to(torch.float8_e4m3fn)
    return w_fp8.view(n, k).contiguous(), scale.contiguous()


def _quant_channel(w):
    scale = w.abs().amax(dim=1, keepdim=True).float() / FP8_MAX
    return (w.float() / scale).to(torch.float8_e4m3fn), scale


def make_fn(provider, m, n, k):
    x = torch.randn(m, k, dtype=torch.bfloat16, device="xpu")
    w = torch.randn(n, k, dtype=torch.bfloat16, device="xpu") * 0.02
    if provider == "fp8_tensor":
        w_fp8, scale = _quant_tensor(w)
        return lambda: fp8_linear(x, w_fp8, scale)
    if provider == "fp8_channel":
        w_fp8, scale = _quant_channel(w)
        return lambda: fp8_linear(x, w_fp8, scale.view(-1))
    if provider == "fp8_block":
        w_fp8, scale = _quant_block(w)
        return lambda: fp8_linear(x, w_fp8, scale)
    if provider == "bf16":
        return lambda: torch.matmul(x, w.t())
    if provider == "w8a8_rowwise":
        w_fp8, w_scale = _quant_channel(w)
        w_fp8_t = w_fp8.t()
        w_scale_t = w_scale.t().contiguous()
        x_q = torch.empty(m, k, dtype=torch.float8_e4m3fn, device="xpu")
        x_s = torch.empty(m, 1, dtype=torch.float32, device="xpu")

        def fn():
            sgl_per_token_quant_fp8(x, x_q, x_s)
            return torch._scaled_mm(
                x_q, w_fp8_t, scale_a=x_s, scale_b=w_scale_t, out_dtype=torch.bfloat16
            )

        return fn
    raise ValueError(provider)


def run(models, providers):
    rows = []
    for model in models:
        for n, k in WEIGHT_SHAPES[model]:
            for m in M_VALUES:
                row = {"model": model, "N": n, "K": k, "M": m}
                for provider in providers:
                    fn = make_fn(provider, m, n, k)
                    for _ in range(5):
                        fn()
                    torch.xpu.synchronize()
                    ms = triton.testing.do_bench(fn, return_mode="median")
                    row[provider] = ms * 1000
                rows.append(row)
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

    rows = run(args.models, args.providers)
    header = ["model", "N", "K", "M"] + [f"{p} (us)" for p in args.providers]
    print(" | ".join(header))
    for row in rows:
        values = [row["model"], row["N"], row["K"], row["M"]]
        values += [f"{row[p]:.1f}" for p in args.providers]
        print(" | ".join(str(v) for v in values))


if __name__ == "__main__":
    main()
