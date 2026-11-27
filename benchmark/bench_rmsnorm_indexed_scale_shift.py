"""Compare fused indexed RMSNorm with two RMSNorm-plus-modulation baselines.

Example: python benchmark/bench_rmsnorm_indexed_scale_shift.py --case 16464,5376
"""

import argparse
import importlib.metadata
import json

import torch
import torch.nn.functional as F
import triton
from _bench import require_arch
from sgl_kernel import rmsnorm, rmsnorm_indexed_scale_shift

DEFAULT_CASES = [
    (1, 5376),
    (7, 5376),
    (16, 5376),
    (128, 4096),
    (16464, 4096),
    (128, 5376),
    (2048, 5376),
    (8192, 5376),
    (16464, 5376),
    (32768, 5376),
    (128, 8192),
    (16464, 8192),
]


def parse_case(text):
    try:
        rows, hidden = map(int, text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("case must be rows,hidden") from exc
    if rows <= 0 or not 0 < hidden <= 8192:
        raise argparse.ArgumentTypeError(
            "rows must be positive and hidden must be in 1..8192"
        )
    return rows, hidden


def measure_ms(fn, warmup, rep):
    median_ms, _, _ = triton.testing.do_bench(
        fn, warmup=warmup, rep=rep, quantiles=[0.5, 0.2, 0.8]
    )
    return median_ms


def run_case(rows, hidden, warmup, rep):
    x = torch.randn(rows, hidden, device="xpu", dtype=torch.bfloat16)
    weight = torch.randn(hidden, device="xpu", dtype=torch.bfloat16)
    packed = torch.randn(3, 6 * hidden, device="xpu", dtype=torch.bfloat16)
    shift, scale = packed.chunk(6, dim=-1)[:2]
    indices = torch.randint(0, 3, (rows,), device="xpu", dtype=torch.int64)

    def fused():
        return rmsnorm_indexed_scale_shift(x, weight, shift, scale, indices, 1e-5)

    def rmsnorm_plus_eager():
        y = rmsnorm(x, weight, 1e-5)
        return y * (1 + scale.index_select(0, indices)) + shift.index_select(0, indices)

    def eager():
        y = F.rms_norm(x, (hidden,), weight, 1e-5)
        return y * (1 + scale.index_select(0, indices)) + shift.index_select(0, indices)

    candidate = fused()
    reference = eager()
    torch.testing.assert_close(
        candidate, reference, atol=1e-2, rtol=1e-2, equal_nan=True
    )
    fused_ms = measure_ms(fused, warmup, rep)
    rmsnorm_plus_eager_ms = measure_ms(rmsnorm_plus_eager, warmup, rep)
    eager_ms = measure_ms(eager, warmup, rep)
    return {
        "rows": rows,
        "hidden": hidden,
        "dtype": "bf16",
        "device": torch.xpu.get_device_name(0),
        "torch": torch.__version__,
        "sglang_kernel_xpu": importlib.metadata.version("sglang-kernel-xpu"),
        "warmup_ms": warmup,
        "rep_ms": rep,
        "fused_ms": round(fused_ms, 5),
        "rmsnorm_plus_eager_ms": round(rmsnorm_plus_eager_ms, 5),
        "eager_ms": round(eager_ms, 5),
        "speedup_vs_rmsnorm_plus_eager": round(rmsnorm_plus_eager_ms / fused_ms, 4),
        "speedup_vs_eager": round(eager_ms / fused_ms, 4),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", type=parse_case)
    parser.add_argument("--warmup", type=int, default=25, help="warmup time in ms")
    parser.add_argument("--rep", type=int, default=100, help="measurement time in ms")
    parser.add_argument("--seed", type=int, default=1117)
    args = parser.parse_args()
    if args.warmup < 1 or args.rep < 1:
        parser.error("warmup and rep must be positive")
    if not torch.xpu.is_available():
        parser.error("XPU is required")
    require_arch("xe20")
    torch.manual_seed(args.seed)
    for rows, hidden in args.case or DEFAULT_CASES:
        print(json.dumps(run_case(rows, hidden, args.warmup, args.rep)), flush=True)


if __name__ == "__main__":
    main()
