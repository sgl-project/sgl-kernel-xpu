"""Compare SGLang main and XPU logits/vocab paths across tensor sizes."""

import argparse
import os
import time

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

import json
import math
import tempfile
from pathlib import Path
from statistics import median

import sgl_kernel
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.kernels.ops.activation.softcap import (
    softcap_inplace_logits,
    softcap_inplace_logits_kernel,
)


@triton.jit
def softcap_flat_kernel(logits, cap, numel, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(logits + offsets, mask=offsets < numel)
    x = libdevice.tanh(x / cap) * cap
    tl.store(logits + offsets, x, mask=offsets < numel)


@triton.jit
def softcap_flat_recip_kernel(
    logits, cap, reciprocal_cap, numel, BLOCK_SIZE: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(logits + offsets, mask=offsets < numel)
    x = libdevice.tanh(x * reciprocal_cap) * cap
    tl.store(logits + offsets, x, mask=offsets < numel)


def summarize(values):
    ordered = sorted(values)

    def quantile(q):
        index = (len(ordered) - 1) * q
        lo, hi = math.floor(index), math.ceil(index)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)

    return {"p20_us": quantile(0.2), "p50_us": median(values), "p80_us": quantile(0.8)}


def measure(fn, reps, warmup=5):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.XPU,
        ]
    ) as profiler:
        for _ in range(reps):
            fn()
        torch.xpu.synchronize()
    with tempfile.TemporaryDirectory() as directory:
        trace_path = Path(directory) / "trace.json"
        profiler.export_chrome_trace(str(trace_path))
        events = json.loads(trace_path.read_text())["traceEvents"]
    kernels = sorted(
        (event for event in events if event.get("cat") == "kernel"),
        key=lambda event: event["ts"],
    )
    if len(kernels) % reps:
        raise RuntimeError(
            f"kernel count {len(kernels)} is not divisible by reps {reps}"
        )
    per_call = len(kernels) // reps
    if per_call == 0:
        raise RuntimeError("profiler recorded no XPU kernels")
    durations = [
        sum(event["dur"] for event in kernels[i : i + per_call])
        for i in range(0, len(kernels), per_call)
    ]
    return durations, sorted({event["name"] for event in kernels})


def measure_paired(cases, op, reps, warmup, order):
    primary = "triton" if op == "softcap" else "compiled"
    selected = {primary: cases[primary], "sycl": cases["sycl"]}
    names = list(selected)
    if order == "sycl-first":
        names.reverse()
    for _ in range(warmup):
        for name in names:
            selected[name]()
    torch.xpu.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.XPU,
        ]
    ) as profiler:
        for _ in range(reps):
            for name in names:
                selected[name]()
        torch.xpu.synchronize()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "trace.json"
        profiler.export_chrome_trace(str(path))
        events = json.loads(path.read_text())["traceEvents"]
    patterns = {
        "triton": "softcap_inplace_logits_kernel",
        "compiled": "triton_poi_fused",
        "sycl": "SoftcapLogits" if op == "softcap" else "VocabMaskKernel",
    }
    samples = {}
    kernels = {}
    for name in names:
        observed = [
            event
            for event in events
            if event.get("cat") == "kernel" and patterns[name] in event["name"]
        ]
        if len(observed) != reps:
            raise RuntimeError(
                f"{name}: expected {reps} device samples, got {len(observed)}"
            )
        samples[name] = [event["dur"] for event in observed]
        kernels[name] = sorted({event["name"] for event in observed})
    return samples, kernels


def softcap_cases(
    batch,
    width,
    cap,
    include_sycl,
    include_flat,
    include_recip,
    include_wrapper,
    block_size,
    num_warps,
):
    source = torch.randn(batch, width, device="xpu", dtype=torch.float32)
    triton_logits = source.clone()
    sycl_logits = source.clone()
    flat_logits = source.clone()
    recip_logits = source.clone()
    wrapper_logits = source.clone()

    def run_triton():
        if triton_logits.is_contiguous():
            nrows, ncols = 1, triton_logits.numel()
            row_stride = ncols
        else:
            nrows, ncols = triton_logits.shape
            row_stride = triton_logits.stride(0)
        softcap_inplace_logits_kernel[(triton.cdiv(ncols, block_size), nrows)](
            triton_logits, cap, ncols, row_stride, block_size, num_warps=num_warps
        )

    def run_sycl():
        torch.ops.sgl_kernel.softcap_logits_xe35(sycl_logits, cap)

    def run_triton_flat():
        softcap_flat_kernel[(triton.cdiv(flat_logits.numel(), block_size),)](
            flat_logits, cap, flat_logits.numel(), block_size, num_warps=num_warps
        )

    def run_triton_recip():
        softcap_flat_recip_kernel[(triton.cdiv(recip_logits.numel(), block_size),)](
            recip_logits,
            cap,
            1.0 / cap,
            recip_logits.numel(),
            block_size,
            num_warps=num_warps,
        )

    def run_pytorch():
        return cap * torch.tanh(source / cap)

    def run_wrapper():
        return softcap_inplace_logits(wrapper_logits, cap)

    run_triton()
    pytorch_result = run_pytorch()
    expected = cap * torch.tanh(source.cpu() / cap)
    torch.testing.assert_close(triton_logits.cpu(), expected, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(pytorch_result.cpu(), expected, atol=3e-5, rtol=3e-5)
    cases = {"triton": run_triton, "pytorch": run_pytorch}
    if include_flat:
        run_triton_flat()
        torch.testing.assert_close(flat_logits.cpu(), expected, atol=3e-5, rtol=3e-5)
        cases["triton_flat"] = run_triton_flat
    if include_recip:
        run_triton_recip()
        torch.testing.assert_close(recip_logits.cpu(), expected, atol=3e-5, rtol=3e-5)
        cases["triton_recip"] = run_triton_recip
    if include_wrapper:
        run_wrapper()
        torch.testing.assert_close(wrapper_logits.cpu(), expected, atol=3e-5, rtol=3e-5)
        cases["triton_wrapper"] = run_wrapper
    if include_sycl:
        run_sycl()
        torch.testing.assert_close(sycl_logits.cpu(), expected, atol=3e-5, rtol=3e-5)
        cases["sycl"] = run_sycl
    return cases


def vocab_cases(count, id_dtype, include_sycl):
    from sglang.srt.layers.vocab_parallel_embedding import get_masked_input_and_mask

    bounds = (0, 100, 16, 200, 207)
    values = [0, 1, 99, 100, 199, 200, 206, 207, -1]
    ids = torch.tensor(
        (values * triton.cdiv(count, len(values)))[:count],
        device="xpu",
        dtype=id_dtype,
    )
    compiled = get_masked_input_and_mask
    eager = compiled.__wrapped__

    def run_compiled():
        return compiled(ids, *bounds)

    def run_sycl():
        return torch.ops.sgl_kernel.vocab_mask_xe35(ids, *bounds)

    def run_pytorch():
        return eager(ids, *bounds)

    ids_cpu = ids.cpu().long()
    original = (ids_cpu >= 0) & (ids_cpu < 100)
    added = (ids_cpu >= 200) & (ids_cpu < 207)
    expected = (
        torch.where(original, ids_cpu, torch.where(added, ids_cpu - 84, 0)),
        ~(original | added),
    )
    methods = (
        (run_compiled, run_pytorch, run_sycl)
        if include_sycl
        else (run_compiled, run_pytorch)
    )
    for method in methods:
        actual = method()
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got.cpu(), want, atol=0, rtol=0)
    if include_sycl:
        return {"compiled": run_compiled, "sycl": run_sycl, "pytorch": run_pytorch}
    return {"compiled": run_compiled, "pytorch": run_pytorch}


def embedding_cases(count, hidden, include_sycl, include_triton, block_h, num_warps):
    from sglang.kernels.ops.embeddings.vocab_parallel_embedding import (
        _vocab_parallel_embedding_kernel,
    )
    from sglang.srt.layers.vocab_parallel_embedding import get_masked_input_and_mask

    bounds = (0, 512, 0, 1010, 1026)
    values = [0, 1, 511, 512, 1009, 1010, 1025, 1026, -1]
    ids = torch.tensor(
        (values * triton.cdiv(count, len(values)))[:count],
        device="xpu",
        dtype=torch.int64,
    )
    weight = torch.randn(528, hidden, device="xpu", dtype=torch.bfloat16)
    output = torch.empty(count, hidden, device="xpu", dtype=weight.dtype)

    def run_main():
        mapped, invalid = get_masked_input_and_mask(ids, *bounds)
        result = F.embedding(mapped.long(), weight)
        result.masked_fill_(invalid.unsqueeze(-1), 0)
        return result

    def run_sycl():
        torch.ops.sgl_kernel.vocab_parallel_embedding_xe35(ids, weight, output, *bounds)
        return output

    def run_triton():
        _vocab_parallel_embedding_kernel[(count, triton.cdiv(hidden, block_h))](
            ids,
            weight,
            output,
            hidden,
            weight.stride(0),
            *bounds,
            BLOCK_H=block_h,
            num_warps=num_warps,
        )
        return output

    ids_cpu = ids.cpu()
    original = (ids_cpu >= 0) & (ids_cpu < 512)
    added = (ids_cpu >= 1010) & (ids_cpu < 1026)
    mapped = torch.where(original, ids_cpu, torch.where(added, ids_cpu - 498, 0))
    expected = weight.cpu()[mapped]
    expected[~(original | added)] = 0
    torch.testing.assert_close(run_main().cpu(), expected, atol=0, rtol=0)
    cases = {"main_pytorch": run_main}
    if include_triton:
        torch.testing.assert_close(run_triton().cpu(), expected, atol=0, rtol=0)
        cases["main_triton_xpu"] = run_triton
    if include_sycl:
        torch.testing.assert_close(run_sycl().cpu(), expected, atol=0, rtol=0)
        cases["sycl"] = run_sycl
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--op", choices=("softcap", "vocab", "embedding"), required=True
    )
    parser.add_argument("--mode", choices=("sweep", "paired"), default="sweep")
    parser.add_argument("--sizes", type=int, nargs="+")
    parser.add_argument("--vocab", type=int)
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--batch", "--serving-batch", type=int, default=8)
    parser.add_argument("--isl", type=int, default=4096)
    parser.add_argument("--osl", type=int, default=1024)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--cap", type=float, default=20.0)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--triton-block-size", type=int, default=1024)
    parser.add_argument("--triton-num-warps", type=int, default=4)
    parser.add_argument("--id-dtype", choices=("int32", "int64"), default="int64")
    parser.add_argument("--reps", "--samples", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--order", choices=("triton-first", "sycl-first"), default="triton-first"
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--main-only", action="store_true")
    parser.add_argument("--softcap-flat", action="store_true")
    parser.add_argument("--softcap-recip", action="store_true")
    parser.add_argument("--softcap-wrapper", action="store_true")
    parser.add_argument("--embedding-triton-xpu", action="store_true")
    parser.add_argument("--embedding-block-h", type=int, default=None)
    parser.add_argument("--embedding-num-warps", type=int, default=8)
    args = parser.parse_args()
    if not torch.xpu.is_available() or not sgl_kernel.is_xe3_arch():
        parser.error("SYCL logits/vocab kernels require Xe35 compute capability 3.5")
    if args.batch != 8:
        parser.error("Muse-Spark serving workload requires B=8")
    if (args.isl, args.osl, args.tp, args.ep) != (4096, 1024, 4, 1):
        parser.error("Muse-Spark workload requires ISL4096/OSL1024/TP4/EP1")
    if sum(value is not None for value in (args.sizes, args.vocab, args.tokens)) > 1:
        parser.error("choose exactly one of sizes, vocab, or tokens")
    if args.vocab is not None and args.op != "softcap":
        parser.error(
            "vocab is the softcap width option; use tokens for vocab-mask or embedding"
        )
    if args.tokens is not None and args.op == "softcap":
        parser.error("use vocab or sizes for softcap")
    args.sizes = args.sizes or [
        (
            args.vocab
            if args.vocab is not None
            else (
                args.tokens
                if args.tokens is not None
                else 202048 if args.op == "softcap" else 8
            )
        )
    ]
    if args.warmup < 1 or (
        args.mode == "paired" and (args.op == "embedding" or args.main_only)
    ):
        parser.error(
            "warmup must be positive; paired mode requires softcap/vocab and the SYCL backend"
        )
    if (
        any(size <= 0 for size in args.sizes)
        or args.reps <= 0
        or args.rounds <= 0
        or args.hidden <= 0
    ):
        parser.error("sizes, reps, rounds, and hidden must be positive")
    if args.triton_block_size <= 0 or args.triton_num_warps <= 0:
        parser.error("Triton block size and warps must be positive")
    if (
        args.embedding_block_h is not None and args.embedding_block_h <= 0
    ) or args.embedding_num_warps <= 0:
        parser.error("embedding block size and warps must be positive")
    embedding_block_h = args.embedding_block_h or min(
        1024, triton.next_power_of_2(args.hidden)
    )
    print(
        json.dumps(
            {
                "device": torch.xpu.get_device_name(0),
                "torch": torch.__version__,
                "op": args.op,
                "mode": args.mode,
                "order": args.order,
                "batch": args.batch,
                "hidden": args.hidden,
                "isl": args.isl,
                "osl": args.osl,
                "tp": args.tp,
                "ep": args.ep,
                "sizes": args.sizes,
                "id_dtype": args.id_dtype,
                "reps": args.reps,
                "rounds": args.rounds,
                "triton_block_size": args.triton_block_size,
                "triton_num_warps": args.triton_num_warps,
                "main_only": args.main_only,
                "softcap_flat": args.softcap_flat,
                "softcap_recip": args.softcap_recip,
                "softcap_wrapper": args.softcap_wrapper,
                "embedding_triton_xpu": args.embedding_triton_xpu,
                "embedding_block_h": embedding_block_h,
                "embedding_num_warps": args.embedding_num_warps,
                "timing": "torch.profiler XPU kernel duration sum per call, microseconds",
            }
        ),
        flush=True,
    )
    for size in args.sizes:
        torch.xpu.synchronize()
        time.sleep(2)
        if args.op == "softcap":
            cases = softcap_cases(
                args.batch,
                size,
                args.cap,
                not args.main_only,
                args.softcap_flat,
                args.softcap_recip,
                args.softcap_wrapper,
                args.triton_block_size,
                args.triton_num_warps,
            )
        elif args.op == "vocab":
            cases = vocab_cases(size, getattr(torch, args.id_dtype), not args.main_only)
        else:
            cases = embedding_cases(
                size,
                args.hidden,
                not args.main_only,
                args.embedding_triton_xpu,
                embedding_block_h,
                args.embedding_num_warps,
            )
        samples = {name: [] for name in cases}
        kernel_names = {name: [] for name in cases}
        names = list(cases)
        if args.order == "sycl-first" and "sycl" in names:
            names.remove("sycl")
            names.insert(0, "sycl")
        for round_index in range(args.rounds):
            if args.mode == "paired":
                paired_order = (
                    args.order
                    if round_index % 2 == 0
                    else (
                        "sycl-first" if args.order == "triton-first" else "triton-first"
                    )
                )
                observed, observed_kernels = measure_paired(
                    cases, args.op, args.reps, args.warmup, paired_order
                )
                if round_index == 0:
                    samples = {name: [] for name in observed}
                for name, values in observed.items():
                    samples[name].extend(values)
                kernel_names = observed_kernels
                continue
            order = names if round_index % 2 == 0 else list(reversed(names))
            for name in order:
                times, observed_names = measure(cases[name], args.reps, args.warmup)
                samples[name].extend(times)
                kernel_names[name] = observed_names
        print(
            json.dumps(
                {
                    "shape": [args.batch, size] if args.op == "softcap" else [size],
                    "summary": {
                        name: summarize(values) for name, values in samples.items()
                    },
                    "kernels": kernel_names,
                    "samples_us": samples,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
