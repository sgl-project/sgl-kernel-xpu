"""Compare H3 indexed gated residual with its PyTorch eager chain.

Example: python benchmark/bench_h3_indexed_gate.py --case 16464,5376
"""

import argparse
import importlib.metadata
import json
import statistics

import torch
from sgl_kernel.h3_fused import h3_indexed_gate


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


def paired_medians(candidate, baseline, reset_candidate, reset_baseline, samples):
    for _ in range(3):
        reset_candidate()
        candidate()
        reset_baseline()
        baseline()
    torch.xpu.synchronize()
    candidate_ms, baseline_ms = [], []
    for i in range(samples):
        order = (
            (
                (baseline, reset_baseline, baseline_ms),
                (candidate, reset_candidate, candidate_ms),
            )
            if i % 2
            else (
                (candidate, reset_candidate, candidate_ms),
                (baseline, reset_baseline, baseline_ms),
            )
        )
        for fn, reset, times in order:
            reset()
            times.append(elapsed_ms(fn))
    return statistics.median(candidate_ms), statistics.median(baseline_ms)


def run_case(rows, hidden, samples):
    x = torch.randn(rows, hidden, device="xpu", dtype=torch.bfloat16)
    other = torch.randn_like(x)
    packed = torch.randn(3, 6 * hidden, device="xpu", dtype=torch.bfloat16)
    gate = packed.chunk(6, dim=-1)[2]
    indices = torch.randint(0, 3, (rows,), device="xpu", dtype=torch.int64)
    candidate_x, baseline_x = x.clone(), x.clone()

    def candidate():
        h3_indexed_gate(candidate_x, gate, other, indices, inplace=True)
        return candidate_x

    def baseline():
        baseline_x.add_(gate.index_select(0, indices) * other)
        return baseline_x

    candidate()
    baseline()
    torch.testing.assert_close(candidate_x, baseline_x, atol=0, rtol=0)
    candidate_ms, baseline_ms = paired_medians(
        candidate,
        baseline,
        lambda: candidate_x.copy_(x),
        lambda: baseline_x.copy_(x),
        samples,
    )
    return {
        "rows": rows,
        "hidden": hidden,
        "dtype": "bf16",
        "device": torch.xpu.get_device_name(0),
        "torch": torch.__version__,
        "sglang_kernel_xpu": importlib.metadata.version("sglang-kernel-xpu"),
        "samples": samples,
        "candidate_ms": round(candidate_ms, 5),
        "baseline_ms": round(baseline_ms, 5),
        "speedup": round(baseline_ms / candidate_ms, 4),
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
    for rows, hidden in args.case or [(128, 5376), (16464, 5376)]:
        print(json.dumps(run_case(rows, hidden, args.samples)), flush=True)


if __name__ == "__main__":
    main()
