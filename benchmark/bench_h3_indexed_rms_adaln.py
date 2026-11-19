"""Compare fused H3 indexed RMSNorm/AdaLN with RMSNorm plus eager modulation.

Example: python benchmark/bench_h3_indexed_rms_adaln.py --case 16464,5376
"""

import argparse
import importlib.metadata
import json
import statistics

import torch
import torch.nn.functional as F
from sgl_kernel import rmsnorm
from sgl_kernel.h3_fused import h3_rmsnorm_indexed_adaln

DEFAULT_CASES = [
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


def elapsed_ms(fn):
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def paired_medians(functions, samples):
    for _ in range(5):
        for fn in functions:
            fn()
    torch.xpu.synchronize()
    times = [[] for _ in functions]
    for i in range(samples):
        # Rotate the first arm to avoid consistently favouring one implementation.
        for j in range(len(functions)):
            arm = (i + j) % len(functions)
            times[arm].append(elapsed_ms(functions[arm]))
    return [statistics.median(arm) for arm in times]


def run_case(rows, hidden, samples):
    x = torch.randn(rows, hidden, device="xpu", dtype=torch.bfloat16)
    weight = torch.randn(hidden, device="xpu", dtype=torch.bfloat16)
    packed = torch.randn(3, 6 * hidden, device="xpu", dtype=torch.bfloat16)
    shift, scale = packed.chunk(6, dim=-1)[:2]
    indices = torch.randint(0, 3, (rows,), device="xpu", dtype=torch.int64)

    def fused():
        return h3_rmsnorm_indexed_adaln(x, weight, shift, scale, indices, 1e-5)

    def baseline():
        y = rmsnorm(x, weight, 1e-5)
        return y * (1 + scale.index_select(0, indices)) + shift.index_select(0, indices)

    def eager():
        y = F.rms_norm(x, (hidden,), weight, 1e-5)
        return y * (1 + scale.index_select(0, indices)) + shift.index_select(0, indices)

    candidate = fused()
    reference = eager()
    torch.testing.assert_close(
        candidate.view(torch.int16), reference.view(torch.int16), atol=0, rtol=0
    )
    max_abs = (candidate.float() - reference.float()).abs().max().item()
    fused_ms, baseline_ms, eager_ms = paired_medians([fused, baseline, eager], samples)
    return {
        "rows": rows,
        "hidden": hidden,
        "dtype": "bf16",
        "device": torch.xpu.get_device_name(0),
        "torch": torch.__version__,
        "sglang_kernel_xpu": importlib.metadata.version("sglang-kernel-xpu"),
        "samples": samples,
        "bit_exact_vs_eager": True,
        "max_abs_vs_eager": max_abs,
        "fused_ms": round(fused_ms, 5),
        "baseline_ms": round(baseline_ms, 5),
        "eager_ms": round(eager_ms, 5),
        "speedup": round(baseline_ms / fused_ms, 4),
        "speedup_vs_eager": round(eager_ms / fused_ms, 4),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", type=parse_case)
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1117)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("samples must be positive")
    if not torch.xpu.is_available():
        parser.error("XPU is required")
    torch.manual_seed(args.seed)
    for rows, hidden in args.case or DEFAULT_CASES:
        print(json.dumps(run_case(rows, hidden, args.samples)), flush=True)


if __name__ == "__main__":
    main()
