"""Compare main Triton position kernel and the registered SYCL kernel on XPU."""

import argparse
import os

if __name__ == "__main__":
    scope_parser_xe35 = argparse.ArgumentParser(add_help=False)
    scope_parser_xe35.add_argument("--batch", "--serving-batch", type=int, default=8)
    scope_parser_xe35.add_argument("--isl", type=int, default=4096)
    scope_parser_xe35.add_argument("--osl", type=int, default=1024)
    scope_parser_xe35.add_argument("--tp", type=int, default=4)
    scope_parser_xe35.add_argument("--ep", type=int, default=1)
    scope_xe35, _unknown_xe35 = scope_parser_xe35.parse_known_args()
    if (
        scope_xe35.batch,
        scope_xe35.isl,
        scope_xe35.osl,
        scope_xe35.tp,
        scope_xe35.ep,
    ) != (8, 4096, 1024, 4, 1) or any(
        os.environ.get(name, "8") != "8"
        for name in ("BATCHES", "BATCH_SIZE", "SERVING_BATCH")
    ):
        scope_parser_xe35.error(
            "Muse-Spark serving scope requires B8/ISL4096/OSL1024/TP4/EP1"
        )

import importlib
import json
import tempfile
import time
from pathlib import Path
from statistics import median

import sgl_kernel
import torch


def measure(triton_fn, sycl_fn, samples, reverse_order):
    order = (sycl_fn, triton_fn) if reverse_order else (triton_fn, sycl_fn)
    for _ in range(5):
        for fn in order:
            fn()
    torch.xpu.synchronize()
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.XPU,
    ]
    with torch.profiler.profile(activities=activities) as profile:
        for _ in range(samples):
            for fn in order:
                fn()
        torch.xpu.synchronize()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "trace.json"
        profile.export_chrome_trace(str(path))
        events = json.loads(path.read_text())["traceEvents"]
    kernels = [
        event for event in events if event.get("cat") == "kernel" and "dur" in event
    ]
    triton_us = [
        float(event["dur"])
        for event in kernels
        if "compute_position_kernel" in event["name"]
    ]
    sycl_us = [
        float(event["dur"])
        for event in kernels
        if "ComputePositionKernel_xe35" in event["name"]
    ]
    assert len(triton_us) == len(sycl_us) == samples, [
        (event["name"], event["dur"]) for event in kernels[:10]
    ]
    return triton_us, sycl_us


def spread(samples):
    ordered = sorted(samples)
    last = len(ordered) - 1
    return ordered[round(0.2 * last)], ordered[round(0.8 * last)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lengths", default="1,16,64,128,256,512,1024,2048,4096")
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--length-dtype", choices=["int32", "int64"], default="int32")
    parser.add_argument("--isl", type=int, default=4096)
    parser.add_argument("--osl", type=int, default=1024)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--skew", action="store_true")
    parser.add_argument("--max-seq-len-hint", type=int, default=-1)
    parser.add_argument("--reverse-order", action="store_true")
    parser.add_argument(
        "--prefix-mode", choices=["zero", "staggered", "decode"], default="zero"
    )
    args = parser.parse_args()
    if args.batch != 8:
        parser.error("Muse-Spark serving workload requires B=8")
    if (args.isl, args.osl, args.tp, args.ep) != (4096, 1024, 4, 1):
        parser.error("Muse-Spark workload requires ISL4096/OSL1024/TP4/EP1")
    if not torch.xpu.is_available() or not sgl_kernel.is_xe3_arch():
        parser.error("SYCL position kernels require Xe35 compute capability 3.5")
    position_module = importlib.import_module("sglang.kernels.ops.attention.position")
    compute_position_kernel = position_module.compute_position_kernel
    length_dtype = getattr(torch, args.length_dtype)
    assert args.batch > 0 and args.samples > 0
    assert torch.xpu.is_available()
    print(
        json.dumps(
            {
                "meta": {
                    "device": torch.xpu.get_device_name(0),
                    "dtype": f"{args.length_dtype} lengths; int64 positions; int32 starts",
                    "layout": "contiguous",
                    "method": "torch.profiler XPU kernel event duration; us",
                    "batch": args.batch,
                    "isl": args.isl,
                    "osl": args.osl,
                    "tp": args.tp,
                    "ep": args.ep,
                    "samples": args.samples,
                    "skew": args.skew,
                    "max_seq_len_hint": args.max_seq_len_hint,
                    "reverse_order": args.reverse_order,
                    "prefix_mode": args.prefix_mode,
                }
            }
        ),
        flush=True,
    )
    for length in map(int, args.lengths.split(",")):
        assert length >= 0
        lengths = (
            [length] + [0] * (args.batch - 1) if args.skew else [length] * args.batch
        )
        tokens = sum(lengths)
        seq = torch.tensor(lengths, dtype=length_dtype, device="xpu")
        prefix = torch.zeros(args.batch, dtype=length_dtype, device="xpu")
        if args.prefix_mode == "staggered":
            prefix = torch.arange(args.batch, dtype=length_dtype, device="xpu") * 3
        elif args.prefix_mode == "decode":
            prefix.fill_(args.isl)
        positions_t = torch.empty(tokens, dtype=torch.int64, device="xpu")
        starts_t = torch.empty(args.batch, dtype=torch.int32, device="xpu")
        positions_s = torch.empty_like(positions_t)
        starts_s = torch.empty_like(starts_t)

        def main_triton():
            compute_position_kernel[(args.batch,)](
                positions_t, starts_t, prefix, seq, True
            )

        def branch_sycl():
            if args.max_seq_len_hint >= 0:
                torch.ops.sgl_kernel.compute_position_with_hint_xe35(
                    positions_s, starts_s, prefix, seq, args.max_seq_len_hint
                )
            else:
                torch.ops.sgl_kernel.compute_position_xe35(
                    positions_s, starts_s, prefix, seq
                )

        main_triton()
        branch_sycl()
        torch.xpu.synchronize()
        expected = torch.cat(
            [
                torch.arange(size, dtype=torch.int64)
                + (
                    i * 3
                    if args.prefix_mode == "staggered"
                    else args.isl if args.prefix_mode == "decode" else 0
                )
                for i, size in enumerate(lengths)
            ]
        )
        expected_starts = torch.tensor(
            [sum(lengths[:i]) for i in range(args.batch)],
            dtype=torch.int32,
        )
        for pos, starts in ((positions_t, starts_t), (positions_s, starts_s)):
            torch.testing.assert_close(pos.cpu(), expected, rtol=0, atol=0)
            torch.testing.assert_close(starts.cpu(), expected_starts, rtol=0, atol=0)
        t, s = measure(main_triton, branch_sycl, args.samples, args.reverse_order)
        t_p20, t_p80 = spread(t)
        s_p20, s_p80 = spread(s)
        print(
            json.dumps(
                {
                    "length": length,
                    "tokens": tokens,
                    "main_triton_us": t,
                    "branch_sycl_us": s,
                    "main_p50_us": median(t),
                    "branch_p50_us": median(s),
                    "main_p20_us": t_p20,
                    "main_p80_us": t_p80,
                    "branch_p20_us": s_p20,
                    "branch_p80_us": s_p80,
                    "main_span_us": max(t) - min(t),
                    "branch_span_us": max(s) - min(s),
                }
            ),
            flush=True,
        )
        torch.xpu.synchronize()
        time.sleep(2)


if __name__ == "__main__":
    main()
