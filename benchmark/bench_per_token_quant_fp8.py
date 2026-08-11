"""Benchmark for per-token dynamic FP8 E4M3 quantization (XPU).

Reports achieved memory bandwidth (read bf16/fp16 input + write fp8 output +
fp32 scale) against a pure-torch reference across token/hidden-dim shapes.

Usage:
    python benchmark/bench_per_token_quant_fp8.py
"""

import itertools

import sgl_kernel  # noqa: F401 – registers XPU ops
import torch
from sgl_kernel import sgl_per_token_quant_fp8

DEVICE = "xpu"
fp8_type_ = torch.float8_e4m3fn


def sglang_per_token_quant_fp8(input: torch.Tensor):
    scale = torch.zeros(input.size(0), device=input.device, dtype=torch.float32)
    output = torch.empty_like(input, device=input.device, dtype=fp8_type_)
    sgl_per_token_quant_fp8(input, output, scale)
    return output, scale


def torch_per_token_quant_fp8(input: torch.Tensor):
    finfo = torch.finfo(torch.float8_e4m3fn)
    absmax = input.abs().to(torch.float32).amax(dim=-1, keepdim=True)
    scale = absmax / finfo.max
    scale_inv = torch.where(scale > 0, scale.reciprocal(), torch.zeros_like(scale))
    q = (input.to(torch.float32) * scale_inv).clamp(finfo.min, finfo.max)
    return q.to(torch.float8_e4m3fn), scale.squeeze(-1)


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


def verify(num_tokens=512, hidden_dim=4096):
    x = torch.rand((num_tokens, hidden_dim), dtype=torch.float16, device=DEVICE)
    sgl_out, sgl_scale = sglang_per_token_quant_fp8(x)
    ref_out, ref_scale = torch_per_token_quant_fp8(x)
    ok = torch.allclose(
        sgl_out.float(), ref_out.float(), rtol=1e-3, atol=1e-3
    ) and torch.allclose(sgl_scale, ref_scale, rtol=1e-3, atol=1e-3)
    print(f"correctness ({num_tokens}x{hidden_dim}): {'PASS' if ok else 'FAIL'}")


def main():
    print(f"Device : {torch.xpu.get_device_name(0)}")
    print(f"dtype  : torch.float16")
    verify()

    token_range = [512, 2048, 8192, 16384, 32768]
    hidden_range = [2048, 4096, 8192]

    print(f"\n  {'tokens':>8} {'hidden':>7}  {'us':>9}  {'GB/s':>8}")
    print(f"  {'-'*40}")
    for num_tokens, hidden_dim in itertools.product(token_range, hidden_range):
        x = torch.randn(num_tokens, hidden_dim, dtype=torch.float16, device=DEVICE)
        out = torch.empty_like(x, dtype=fp8_type_)
        scale = torch.zeros(num_tokens, dtype=torch.float32, device=DEVICE)

        ms = _bench(lambda: sgl_per_token_quant_fp8(x, out, scale))
        # bytes: read 2B input + write 1B output + write 4B scale/row
        nbytes = num_tokens * hidden_dim * (2 + 1) + num_tokens * 4
        gbps = nbytes / (ms * 1e-3) / 1e9
        print(f"  {num_tokens:>8} {hidden_dim:>7}  {ms*1000:>9.1f}  {gbps:>8.1f}")


if __name__ == "__main__":
    main()
