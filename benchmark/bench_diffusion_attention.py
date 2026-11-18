"""CUTE diffusion attention versus XPU FlashAttention on selected D128 shapes.

Run after installing the BMG wheel::

    python benchmark/bench_diffusion_attention.py
    python benchmark/bench_diffusion_attention.py --case 16464,7,packed,bf16

Each --case is length,heads,layout,dtype. This is an operator benchmark only.
"""

import argparse
import json
import statistics

import torch
from sgl_kernel import diffusion
from sgl_kernel.flash_attn import flash_attn_varlen_func

DEFAULT_CASES = (
    "4128,30,contiguous,bf16",
    "4160,24,packed,bf16",
    "16464,7,packed,bf16",
)
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def parse_case(value):
    try:
        length, heads, layout, dtype = value.split(",")
        length, heads = int(length), int(heads)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "case must be length,heads,layout,dtype"
        ) from exc
    if length <= 0 or heads <= 0 or layout not in {"packed", "contiguous"}:
        raise argparse.ArgumentTypeError("invalid length, heads or layout")
    if dtype not in DTYPES:
        raise argparse.ArgumentTypeError("dtype must be bf16 or fp16")
    return length, heads, layout, dtype


def measured_ms(fn):
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def bench_pair(cute, flash, warmup, samples):
    for _ in range(warmup):
        cute()
        flash()
    torch.xpu.synchronize()
    cute_times, flash_times = [], []
    for i in range(samples):
        if i % 2:
            flash_times.append(measured_ms(flash))
            cute_times.append(measured_ms(cute))
        else:
            cute_times.append(measured_ms(cute))
            flash_times.append(measured_ms(flash))
    return statistics.median(cute_times), statistics.median(flash_times)


def run_case(length, heads, layout, dtype_name, warmup, samples):
    dtype = DTYPES[dtype_name]
    if layout == "packed":
        q, k, v = torch.randn(
            1, length, 3, heads, 128, device="xpu", dtype=dtype
        ).unbind(2)
    else:
        shape = (1, length, heads, 128)
        q = torch.randn(shape, device="xpu", dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
    cu_seqlens = torch.tensor((0, length), device="xpu", dtype=torch.int32)

    def cute():
        return diffusion.sdp(q, k, v)[0]

    def flash():
        out = flash_attn_varlen_func(
            q=q[0],
            k=k[0],
            v=v[0],
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=length,
            max_seqlen_k=length,
            softmax_scale=128**-0.5,
            causal=False,
        )
        return out[0] if isinstance(out, tuple) else out

    result, reference = cute(), flash()
    torch.xpu.synchronize()
    max_abs = (result.float() - reference.float()).abs().max().item()
    tolerance = 0.00390625 if dtype == torch.bfloat16 else 0.002
    if not torch.isfinite(result).all() or max_abs > tolerance:
        raise AssertionError(
            f"{length},{heads},{layout},{dtype_name}: max_abs={max_abs} > {tolerance}"
        )
    cute_ms, flash_ms = bench_pair(cute, flash, warmup, samples)
    return {
        "shape": [1, length, heads, 128],
        "layout": layout,
        "dtype": dtype_name,
        "device": torch.xpu.get_device_name(0),
        "torch": torch.__version__,
        "warmup": warmup,
        "samples": samples,
        "max_abs": max_abs,
        "cute_ms": round(cute_ms, 5),
        "flash_ms": round(flash_ms, 5),
        "speedup": round(flash_ms / cute_ms, 4),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", type=parse_case)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--seed", type=int, default=1117)
    args = parser.parse_args()
    if args.warmup < 0 or args.samples < 1:
        parser.error("warmup must be >= 0 and samples must be >= 1")
    if not torch.xpu.is_available() or not diffusion.is_available():
        parser.error("the BMG CUTE diffusion attention wheel is required")
    torch.manual_seed(args.seed)
    cases = args.case or [parse_case(case) for case in DEFAULT_CASES]
    for case in cases:
        print(json.dumps(run_case(*case, args.warmup, args.samples)), flush=True)


if __name__ == "__main__":
    main()
