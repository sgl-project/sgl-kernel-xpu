"""Benchmark for per-token symmetric INT8 quantization (XPU).

Reports achieved memory bandwidth (read bf16/fp16 input + write int8 output +
scale + optional token sum) against a pure-torch reference across
token/hidden-dim shapes.

Usage:
    python benchmark/bench_per_token_quant_int8.py
    python benchmark/bench_per_token_quant_int8.py --cal-sum
"""

import argparse
import itertools

import sgl_kernel  # noqa: F401 – registers XPU ops
import torch
from sgl_kernel import sgl_per_token_quant_int8

DEVICE = "xpu"


def sglang_per_token_quant_int8(input: torch.Tensor, cal_sum: bool = False):
    if cal_sum:
        # QoQ use fp16 as default scale dtype (QoQ is deprecated in SGLang latest now)
        return sgl_per_token_quant_int8(input, scale_dtype=torch.float16, cal_sum=True)
    return sgl_per_token_quant_int8(input)


def torch_per_token_quant_int8(input: torch.Tensor, cal_sum: bool = False):
    scale_dtype = torch.float16 if cal_sum else torch.float32
    x_fp32 = input.to(torch.float32)
    absmax = x_fp32.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
    scale = (absmax / 127.0).to(scale_dtype)
    scaled = x_fp32 * (127.0 / absmax)
    q = torch.copysign((scaled.abs() + 0.5).floor(), scaled).to(torch.int8)
    if cal_sum:
        token_sum = x_fp32.sum(dim=-1).to(input.dtype)
        return q, scale, token_sum
    return q, scale


def _bench(fn, warmup=10, rep=100):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn()
    end.record()
    torch.xpu.synchronize()
    return start.elapsed_time(end) / rep  # ms


def verify(num_tokens=512, hidden_dim=4096, cal_sum=False):
    x = torch.randn((num_tokens, hidden_dim), dtype=torch.float16, device=DEVICE)
    sgl_outputs = sglang_per_token_quant_int8(x, cal_sum=cal_sum)
    ref_outputs = torch_per_token_quant_int8(x, cal_sum=cal_sum)
    ok = torch.allclose(
        sgl_outputs[0].float(), ref_outputs[0].float(), rtol=0, atol=1
    ) and torch.allclose(sgl_outputs[1], ref_outputs[1], rtol=1e-3, atol=1e-3)
    if cal_sum:
        ok = ok and torch.allclose(sgl_outputs[2], ref_outputs[2], rtol=1e-3, atol=1e-3)
    print(f"correctness ({num_tokens}x{hidden_dim}): {'PASS' if ok else 'FAIL'}")


def main(
    token_range=(512, 2048, 8192, 16384, 32768),
    hidden_range=(2048, 4096, 8192),
    cal_sum=False,
):
    scale_dtype = torch.float16 if cal_sum else torch.float32
    print(f"Device : {torch.xpu.get_device_name(0)}")
    print("dtype  : torch.float16")
    print(f"scale  : {scale_dtype} (cal_sum={cal_sum})")
    verify(cal_sum=cal_sum)

    print(f"\n  {'tokens':>8} {'hidden':>7}  {'us':>9}  {'GB/s':>8}")
    print(f"  {'-'*40}")
    for num_tokens, hidden_dim in itertools.product(token_range, hidden_range):
        x = torch.randn(num_tokens, hidden_dim, dtype=torch.float16, device=DEVICE)

        ms = _bench(
            lambda: sgl_per_token_quant_int8(x, scale_dtype=scale_dtype, cal_sum=cal_sum),
        )
        # bytes: read 2B input + write 1B output + write scale/row (+ optional sum/row)
        scale_bytes = torch.tensor([], dtype=scale_dtype).element_size()
        row_meta_bytes = scale_bytes + (x.element_size() if cal_sum else 0)
        nbytes = num_tokens * hidden_dim * (x.element_size() + 1) + (
            num_tokens * row_meta_bytes
        )
        gbps = nbytes / (ms * 1e-3) / 1e9
        print(f"  {num_tokens:>8} {hidden_dim:>7}  {ms*1000:>9.1f}  {gbps:>8.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-tokens",
        nargs="+",
        type=int,
        default=[512, 2048, 8192, 16384, 32768],
        help="List of token counts",
    )
    parser.add_argument(
        "--hidden-dims",
        nargs="+",
        type=int,
        default=[2048, 4096, 8192],
        help="List of hidden dims",
    )
    parser.add_argument(
        "--cal-sum",
        action="store_true",
    )
    args = parser.parse_args()
    main(args.num_tokens, args.hidden_dims, cal_sum=args.cal_sum)
