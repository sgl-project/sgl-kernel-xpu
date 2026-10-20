"""B8 memory benchmarks with profiler/event, production and tuning modes."""

import argparse
import functools
import importlib
import json
import os
import tempfile
import time
from pathlib import Path
from statistics import median


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("profiler", "event", "tiles", "write-groups", "aux", "prefill"),
        default="profiler",
    )
    parser.add_argument(
        "--op", choices=("all", "decode", "extend", "write", "fused"), default="all"
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--lengths", nargs="+", type=int)
    parser.add_argument("--page-sizes", nargs="+", type=int, default=(64,))
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--prefix-lengths", nargs="+", type=int)
    parser.add_argument(
        "--decode-steps", nargs="+", type=int, default=(0, 1, 63, 64, 1023)
    )
    parser.add_argument("--prefix-len", type=int)
    parser.add_argument("--extend-len", type=int)
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--production-only", action="store_true")
    parser.add_argument("--trace-labels", action="store_true")
    parser.add_argument("--main-only", action="store_true")
    parser.add_argument("--require-no-regression", action="store_true")
    parser.add_argument(
        "--cases", nargs="+", choices=("decode", "extend", "write", "fused")
    )
    parser.add_argument("--case", choices=("extend", "fused"))
    parser.add_argument(
        "--only",
        choices=("all", "alloc_decode", "alloc_extend", "write_req_to_token_pool_xe35"),
    )
    args = parser.parse_args(argv)
    if args.require_no_regression and (args.mode != "profiler" or args.main_only):
        parser.error("--require-no-regression requires paired profiler mode")
    if args.batch != 8 or any(
        os.environ.get(name, "8") != "8"
        for name in ("BATCHES", "BATCH_SIZE", "SERVING_BATCH")
    ):
        parser.error("Muse-Spark serving benchmarks require --batch 8")
    if args.samples is None:
        args.samples = {"tiles": 51, "write-groups": 51, "aux": 30}.get(args.mode, 31)
    if args.lengths is None:
        args.lengths = (
            (4096,)
            if args.mode == "prefill"
            else (
                (
                    64,
                    128,
                    256,
                    384,
                    511,
                    512,
                    513,
                    768,
                    1024,
                    1536,
                    2047,
                    2048,
                    2049,
                    3072,
                    4096,
                )
                if args.mode == "tiles"
                else (256, 511, 512, 513, 1024, 2047, 2048, 2049, 4096)
            )
        )
    if args.prefix_lengths is None:
        args.prefix_lengths = (0,) if args.mode == "prefill" else (0, 4096)
    if (
        args.samples < 1
        or args.repeats < 1
        or args.page_size < 1
        or any(value < 1 for value in (*args.lengths, *args.page_sizes))
        or any(value < 0 for value in args.prefix_lengths)
        or any(value < 0 or value >= 1024 for value in args.decode_steps)
    ):
        parser.error(
            "samples, repeats, lengths and page sizes must be positive; prefixes must be nonnegative"
        )
    if args.case is not None:
        if args.op != "all" and args.op != args.case:
            parser.error("--case and --op must agree")
        args.op = args.case
    if args.cases is not None and args.op != "all":
        parser.error("choose --op or --cases, not both")
    default_cases = (
        ("extend", "fused")
        if args.mode == "tiles"
        else ("decode", "extend", "write", "fused")
    )
    args.cases = args.cases or (default_cases if args.op == "all" else (args.op,))
    if args.mode == "event":
        if (
            args.op not in ("extend", "fused")
            or args.prefix_len is None
            or args.prefix_len < 0
            or args.extend_len is None
            or args.extend_len < 1
        ):
            parser.error(
                "event mode requires --op extend/fused, --prefix-len >= 0 and --extend-len > 0"
            )
        args.case = args.op
    if args.mode in ("tiles", "write-groups"):
        allowed = ("extend", "fused") if args.mode == "tiles" else ("write",)
        if args.op != "all" and args.op not in allowed:
            parser.error(f"{args.mode} mode supports {allowed}")
        if args.mode == "tiles" and any(case not in allowed for case in args.cases):
            parser.error("tiles mode supports only extend/fused")
    if args.mode == "aux":
        names = {
            "all": "all",
            "decode": "alloc_decode",
            "extend": "alloc_extend",
            "write": "write_req_to_token_pool_xe35",
        }
        if args.op not in names:
            parser.error("aux mode supports only all/decode/extend/write")
        if args.only is not None and args.op != "all" and args.only != names[args.op]:
            parser.error("--only and --op must agree")
        args.only = args.only or names[args.op]
    elif args.only is not None:
        parser.error("--only is an aux-mode option")
    if args.mode == "prefill" and args.op not in ("all", "fused"):
        parser.error("prefill mode supports only fused")
    if args.main_only and args.mode != "profiler":
        parser.error("--main-only is a profiler-mode option")
    return args


def quantile(values, fraction):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


@functools.lru_cache(maxsize=1)
def main_memory_ops():
    from sglang.kernels.ops.memory.allocator import (
        alloc_decode_kernel,
        alloc_extend_kernel,
    )
    from sglang.kernels.ops.memory.common import write_req_to_token_pool_triton

    return alloc_extend_kernel, alloc_decode_kernel, write_req_to_token_pool_triton


def profile_pair(name, shape, old, new, old_kernels, new_kernel, samples):
    for _ in range(5):
        old()
        new()
    torch.xpu.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.XPU,
        ]
    ) as profiler:
        for iteration in range(samples):
            if iteration % 2:
                new()
                old()
            else:
                old()
                new()
        torch.xpu.synchronize()
    with tempfile.TemporaryDirectory() as directory:
        trace = Path(directory) / "trace.json"
        profiler.export_chrome_trace(str(trace))
        events = json.loads(trace.read_text())["traceEvents"]
    durations = {
        needle: [
            event["dur"]
            for event in events
            if event.get("cat") == "kernel" and needle in event.get("name", "")
        ]
        for needle in (*old_kernels, new_kernel)
    }
    for needle, values in durations.items():
        if len(values) != samples:
            raise RuntimeError(
                f"{name} {shape}: expected {samples} kernels for {needle}, got {len(values)}"
            )
    old_us = [
        sum(durations[needle][index] for needle in old_kernels)
        for index in range(samples)
    ]
    new_us = durations[new_kernel]
    report = {
        "case": name,
        **shape,
        "samples": samples,
        "timing": "torch.profiler XPU kernel event duration; alternating paired calls",
        "main_samples_us": old_us,
        "sycl_samples_us": new_us,
        "main_us": {
            "p20": quantile(old_us, 0.2),
            "p50": median(old_us),
            "p80": quantile(old_us, 0.8),
            "min": min(old_us),
            "max": max(old_us),
        },
        "sycl_us": {
            "p20": quantile(new_us, 0.2),
            "p50": median(new_us),
            "p80": quantile(new_us, 0.8),
            "min": min(new_us),
            "max": max(new_us),
        },
    }
    report["speedup"] = report["main_us"]["p50"] / report["sycl_us"]["p50"]
    print(json.dumps(report), flush=True)
    if memory_setup.require_no_regression and report["speedup"] < 1:
        regressions.append(report)
    torch.xpu.synchronize()
    time.sleep(2)


def profile_main(name, shape, old, old_kernels, samples):
    for _ in range(5):
        old()
    torch.xpu.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.XPU,
        ]
    ) as profiler:
        for _ in range(samples):
            old()
        torch.xpu.synchronize()
    with tempfile.TemporaryDirectory() as directory:
        trace = Path(directory) / "trace.json"
        profiler.export_chrome_trace(str(trace))
        events = json.loads(trace.read_text())["traceEvents"]
    durations = {
        needle: [
            event["dur"]
            for event in events
            if event.get("cat") == "kernel" and needle in event.get("name", "")
        ]
        for needle in old_kernels
    }
    for needle, values in durations.items():
        if len(values) != samples:
            raise RuntimeError(
                f"{name} {shape}: expected {samples} kernels for {needle}, got {len(values)}"
            )
    old_us = [
        sum(durations[needle][index] for needle in old_kernels)
        for index in range(samples)
    ]
    print(
        json.dumps(
            {
                "case": name,
                **shape,
                "samples": samples,
                "timing": "torch.profiler XPU kernel event duration",
                "main_samples_us": old_us,
                "main_us": {
                    "p20": quantile(old_us, 0.2),
                    "p50": median(old_us),
                    "p80": quantile(old_us, 0.8),
                    "min": min(old_us),
                    "max": max(old_us),
                },
            }
        ),
        flush=True,
    )
    torch.xpu.synchronize()
    time.sleep(2)


def compare_extend(
    batch, prefix_len, extend_len, page_size, samples, fused, main_only=False
):
    alloc_extend_kernel, _, write_req_to_token_pool_triton = main_memory_ops()

    prefix = torch.full((batch,), prefix_len, dtype=torch.int64, device="xpu")
    seq = torch.full((batch,), prefix_len + extend_len, dtype=torch.int64, device="xpu")
    extensions = torch.full((batch,), extend_len, dtype=torch.int64, device="xpu")
    last = torch.full((batch,), prefix_len - 1, dtype=torch.int32, device="xpu")
    free_pages = torch.arange(
        1,
        batch * ((extend_len + page_size - 1) // page_size) + 2,
        dtype=torch.int64,
        device="xpu",
    )
    old_out = torch.empty(batch * extend_len, dtype=torch.int64, device="xpu")
    new_out = torch.empty_like(old_out)
    shape = {
        "batch": batch,
        "prefix_len": prefix_len,
        "extend_len": extend_len,
        "page_size": page_size,
        "last_dtype": "int32",
    }
    if fused:
        old_pool = torch.empty(
            (batch, prefix_len + extend_len), dtype=torch.int32, device="xpu"
        )
        new_pool = torch.empty_like(old_pool)
        req_indices = torch.arange(batch, dtype=torch.int64, device="xpu")
        prefix_values = torch.arange(
            max(1, batch * prefix_len), dtype=torch.int64, device="xpu"
        )
        pointers = torch.tensor(
            [
                prefix_values.data_ptr() + index * prefix_len * 8
                for index in range(batch)
            ],
            dtype=torch.uint64,
            device="xpu",
        )

        def old():
            alloc_extend_kernel[(batch,)](
                prefix,
                seq,
                last,
                free_pages,
                old_out,
                triton.next_power_of_2(batch),
                page_size,
            )
            write_req_to_token_pool_triton[(batch,)](
                old_pool,
                req_indices,
                pointers,
                prefix,
                seq,
                extensions,
                old_out,
                old_pool.stride(0),
            )

        def new():
            torch.ops.sgl_kernel.paged_alloc_extend_and_write_xe35(
                prefix,
                seq,
                last,
                free_pages,
                new_out,
                new_pool,
                req_indices,
                pointers,
                page_size,
            )

        if main_only:
            profile_main(
                "fused",
                shape,
                old,
                ("alloc_extend_kernel", "write_req_to_token_pool_triton"),
                samples,
            )
            return
        old()
        new()
        torch.xpu.synchronize()
        torch.testing.assert_close(new_pool.cpu(), old_pool.cpu(), rtol=0, atol=0)
        old_kernels = ("alloc_extend_kernel", "write_req_to_token_pool_triton")
        new_kernel = "PagedAllocExtendAndWriteKernel"
    else:

        def old():
            alloc_extend_kernel[(batch,)](
                prefix,
                seq,
                last,
                free_pages,
                old_out,
                triton.next_power_of_2(batch),
                page_size,
            )

        def new():
            torch.ops.sgl_kernel.paged_alloc_extend_xe35(
                prefix, seq, last, free_pages, new_out, page_size
            )

        if main_only:
            profile_main("extend", shape, old, ("alloc_extend_kernel",), samples)
            return
        old()
        new()
        torch.xpu.synchronize()
        old_kernels = ("alloc_extend_kernel",)
        new_kernel = "PagedAllocExtendKernel"
    torch.testing.assert_close(new_out.cpu(), old_out.cpu(), rtol=0, atol=0)
    profile_pair(
        "fused" if fused else "extend",
        shape,
        old,
        new,
        old_kernels,
        new_kernel,
        samples,
    )


def compare_decode(samples, main_only=False, step=0):
    _, alloc_decode_kernel, _ = main_memory_ops()

    batch, page_size = 8, 64
    seq = torch.tensor(
        [4097 + step] * batch,
        dtype=torch.int64,
        device="xpu",
    )
    last = torch.arange(batch, dtype=torch.int32, device="xpu") * 8192 + 4095 + step
    free_pages = torch.arange(1, 16, dtype=torch.int64, device="xpu")
    old_out = torch.empty(batch, dtype=torch.int64, device="xpu")
    new_out = torch.empty_like(old_out)

    def old():
        alloc_decode_kernel[(batch,)](seq, last, free_pages, old_out, batch, page_size)

    def new():
        torch.ops.sgl_kernel.paged_alloc_decode_xe35(
            seq, last, free_pages, new_out, page_size
        )

    if main_only:
        profile_main(
            "decode",
            {
                "batch": batch,
                "seq_len": 4097 + step,
                "page_size": page_size,
                "last_dtype": "int32",
            },
            old,
            ("alloc_decode_kernel",),
            samples,
        )
        return
    old()
    new()
    torch.xpu.synchronize()
    torch.testing.assert_close(new_out.cpu(), old_out.cpu(), rtol=0, atol=0)
    profile_pair(
        "decode",
        {
            "batch": batch,
            "seq_len": 4097 + step,
            "page_size": page_size,
            "last_dtype": "int32",
        },
        old,
        new,
        ("alloc_decode_kernel",),
        "PagedAllocDecodeKernel",
        samples,
    )


def compare_write(samples, prefix_len, extend_len, main_only=False):
    _, _, write_req_to_token_pool_triton = main_memory_ops()

    batch, seq_len = 8, prefix_len + extend_len
    old_pool = torch.empty((batch, seq_len), dtype=torch.int32, device="xpu")
    new_pool = torch.empty_like(old_pool)
    req_indices = torch.arange(batch, dtype=torch.int64, device="xpu")
    prefix_values = torch.arange(
        max(1, batch * prefix_len), dtype=torch.int64, device="xpu"
    )
    pointers = torch.tensor(
        [prefix_values.data_ptr() + index * prefix_len * 8 for index in range(batch)],
        dtype=torch.uint64,
        device="xpu",
    )
    prefix = torch.full((batch,), prefix_len, dtype=torch.int64, device="xpu")
    seq = torch.full((batch,), seq_len, dtype=torch.int64, device="xpu")
    extensions = torch.full((batch,), extend_len, dtype=torch.int64, device="xpu")
    cache_loc = torch.arange(batch * extend_len, dtype=torch.int64, device="xpu")

    def old():
        write_req_to_token_pool_triton[(batch,)](
            old_pool,
            req_indices,
            pointers,
            prefix,
            seq,
            extensions,
            cache_loc,
            old_pool.stride(0),
        )

    def new():
        torch.ops.sgl_kernel.write_req_to_token_pool_xe35(
            new_pool, req_indices, pointers, prefix, seq, extensions, cache_loc
        )

    if main_only:
        profile_main(
            "write",
            {
                "batch": batch,
                "prefix_len": prefix_len,
                "extend_len": extend_len,
                "seq_len": seq_len,
                "pool_dtype": "int32",
                "cache_loc_dtype": "int64",
            },
            old,
            ("write_req_to_token_pool_triton",),
            samples,
        )
        return
    old()
    new()
    torch.xpu.synchronize()
    torch.testing.assert_close(new_pool.cpu(), old_pool.cpu(), rtol=0, atol=0)
    profile_pair(
        "write",
        {
            "batch": batch,
            "prefix_len": prefix_len,
            "extend_len": extend_len,
            "seq_len": seq_len,
            "pool_dtype": "int32",
            "cache_loc_dtype": "int64",
        },
        old,
        new,
        ("write_req_to_token_pool_triton",),
        "WriteReqToTokenPoolKernel",
        samples,
    )


def measure_event(args):
    alloc_extend_kernel, _, write_req_to_token_pool_triton = main_memory_ops()
    batch, prefix_len, extend_len, page_size = (
        args.batch,
        args.prefix_len,
        args.extend_len,
        args.page_size,
    )
    prefix = torch.full((batch,), prefix_len, dtype=torch.int64, device="xpu")
    seq = torch.full((batch,), prefix_len + extend_len, dtype=torch.int64, device="xpu")
    extensions = torch.full((batch,), extend_len, dtype=torch.int64, device="xpu")
    last = torch.full((batch,), prefix_len - 1, dtype=torch.int32, device="xpu")
    free_pages = torch.arange(
        1,
        batch * ((extend_len + page_size - 1) // page_size) + 2,
        dtype=torch.int64,
        device="xpu",
    )
    labels = (
        ("main", "production")
        if args.production_only
        else ("main", "production", "tile1", "tile4", "tile16")
    )
    outputs = {
        label: torch.empty(batch * extend_len, dtype=torch.int64, device="xpu")
        for label in labels
    }
    if args.case == "fused":
        pools = {
            label: torch.empty(
                (batch, prefix_len + extend_len), dtype=torch.int32, device="xpu"
            )
            for label in labels
        }
        req_indices = torch.arange(batch, dtype=torch.int64, device="xpu")
        prefix_values = torch.arange(
            max(1, batch * prefix_len), dtype=torch.int64, device="xpu"
        )
        pointers = torch.tensor(
            [
                prefix_values.data_ptr() + index * prefix_len * 8
                for index in range(batch)
            ],
            dtype=torch.uint64,
            device="xpu",
        )

        def main():
            alloc_extend_kernel[(batch,)](
                prefix,
                seq,
                last,
                free_pages,
                outputs["main"],
                triton.next_power_of_2(batch),
                page_size,
            )
            write_req_to_token_pool_triton[(batch,)](
                pools["main"],
                req_indices,
                pointers,
                prefix,
                seq,
                extensions,
                outputs["main"],
                pools["main"].stride(0),
            )

        def production():
            torch.ops.sgl_kernel.paged_alloc_extend_and_write_xe35(
                prefix,
                seq,
                last,
                free_pages,
                outputs["production"],
                pools["production"],
                req_indices,
                pointers,
                page_size,
            )

        def tiled(label, groups):
            torch.ops.sgl_kernel.paged_alloc_extend_and_write_tiled_for_benchmark(
                prefix,
                seq,
                last,
                free_pages,
                outputs[label],
                pools[label],
                req_indices,
                pointers,
                page_size,
                groups,
            )

    else:
        pools = None

        def main():
            alloc_extend_kernel[(batch,)](
                prefix,
                seq,
                last,
                free_pages,
                outputs["main"],
                triton.next_power_of_2(batch),
                page_size,
            )

        def production():
            torch.ops.sgl_kernel.paged_alloc_extend_xe35(
                prefix,
                seq,
                last,
                free_pages,
                outputs["production"],
                page_size,
            )

        def tiled(label, groups):
            torch.ops.sgl_kernel.paged_alloc_extend_tiled_for_benchmark(
                prefix,
                seq,
                last,
                free_pages,
                outputs[label],
                page_size,
                groups,
            )

    calls = {
        "main": main,
        "production": production,
        "tile1": lambda: tiled("tile1", 1),
        "tile4": lambda: tiled("tile4", 4),
        "tile16": lambda: tiled("tile16", 16),
    }
    for _ in range(5):
        for label in labels:
            calls[label]()
    torch.xpu.synchronize()
    expected = outputs["main"].cpu()
    for label in labels[1:]:
        torch.testing.assert_close(outputs[label].cpu(), expected, rtol=0, atol=0)
        if pools is not None:
            torch.testing.assert_close(
                pools[label].cpu(), pools["main"].cpu(), rtol=0, atol=0
            )

    durations = {label: [] for label in labels}
    for iteration in range(args.samples):
        rotated = labels[iteration % len(labels) :] + labels[: iteration % len(labels)]
        for label in rotated:
            if args.trace_labels:
                print(f"window_start sample={iteration} variant={label}", flush=True)
            start = torch.xpu.Event(enable_timing=True)
            end = torch.xpu.Event(enable_timing=True)
            start.record()
            for _ in range(args.repeats):
                calls[label]()
            end.record()
            end.synchronize()
            durations[label].append(start.elapsed_time(end) * 1000 / args.repeats)
            if args.trace_labels:
                print(f"window_done sample={iteration} variant={label}", flush=True)

    print(
        json.dumps(
            {
                "case": args.case,
                "batch": batch,
                "prefix_len": prefix_len,
                "extend_len": extend_len,
                "page_size": page_size,
                "dtype": {"metadata": "int64", "last": "int32", "pool": "int32"},
                "device": torch.xpu.get_device_name(0),
                "samples": args.samples,
                "repeats_per_sample": args.repeats,
                "timing": "torch.xpu.Event window / repeats, microseconds; includes gaps between enqueues",
                "variants": {
                    label: {
                        "samples_us": values,
                        "p20": quantile(values, 0.2),
                        "p50": median(values),
                        "p80": quantile(values, 0.8),
                        "min": min(values),
                        "max": max(values),
                    }
                    for label, values in durations.items()
                },
            }
        ),
        flush=True,
    )
    torch.xpu.synchronize()
    time.sleep(2)


def measure_tiles(batch, prefix_len, extend_len, page_size, samples, fused):
    alloc_extend_kernel, _, write_req_to_token_pool_triton = main_memory_ops()
    prefix = torch.full((batch,), prefix_len, dtype=torch.int64, device="xpu")
    seq = torch.full((batch,), prefix_len + extend_len, dtype=torch.int64, device="xpu")
    extensions = torch.full((batch,), extend_len, dtype=torch.int64, device="xpu")
    last = torch.full((batch,), prefix_len - 1, dtype=torch.int32, device="xpu")
    free_pages = torch.arange(
        1,
        batch * ((extend_len + page_size - 1) // page_size) + 2,
        dtype=torch.int64,
        device="xpu",
    )
    expected = torch.empty(batch * extend_len, dtype=torch.int64, device="xpu")
    alloc_extend_kernel[(batch,)](
        prefix,
        seq,
        last,
        free_pages,
        expected,
        triton.next_power_of_2(batch),
        page_size,
    )
    outputs = {tiles: torch.empty_like(expected) for tiles in (1, 4, 16)}
    if fused:
        req_indices = torch.arange(batch, dtype=torch.int64, device="xpu")
        prefix_values = torch.arange(
            max(1, batch * prefix_len), dtype=torch.int64, device="xpu"
        )
        pointers = torch.tensor(
            [
                prefix_values.data_ptr() + index * prefix_len * 8
                for index in range(batch)
            ],
            dtype=torch.uint64,
            device="xpu",
        )
        expected_pool = torch.empty(
            (batch, prefix_len + extend_len), dtype=torch.int32, device="xpu"
        )
        write_req_to_token_pool_triton[(batch,)](
            expected_pool,
            req_indices,
            pointers,
            prefix,
            seq,
            extensions,
            expected,
            expected_pool.stride(0),
        )
        pools = {tiles: torch.empty_like(expected_pool) for tiles in (1, 4, 16)}

        def run(tiles):
            torch.ops.sgl_kernel.paged_alloc_extend_and_write_tiled_for_benchmark(
                prefix,
                seq,
                last,
                free_pages,
                outputs[tiles],
                pools[tiles],
                req_indices,
                pointers,
                page_size,
                tiles,
            )

        needle = "PagedAllocExtendAndWriteKernel"
    else:

        def run(tiles):
            torch.ops.sgl_kernel.paged_alloc_extend_tiled_for_benchmark(
                prefix, seq, last, free_pages, outputs[tiles], page_size, tiles
            )

        needle = "PagedAllocExtendKernel"
    for _ in range(5):
        for tiles in (1, 4, 16):
            run(tiles)
    torch.xpu.synchronize()
    for tiles in (1, 4, 16):
        torch.testing.assert_close(outputs[tiles].cpu(), expected.cpu(), rtol=0, atol=0)
        if fused:
            torch.testing.assert_close(
                pools[tiles].cpu(), expected_pool.cpu(), rtol=0, atol=0
            )
    order = []
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.XPU,
        ]
    ) as profiler:
        for iteration in range(samples):
            choices = (1, 4, 16)
            choices = choices[iteration % 3 :] + choices[: iteration % 3]
            for tiles in choices:
                run(tiles)
                order.append(tiles)
        torch.xpu.synchronize()
    with tempfile.TemporaryDirectory() as directory:
        trace = Path(directory) / "trace.json"
        profiler.export_chrome_trace(str(trace))
        events = json.loads(trace.read_text())["traceEvents"]
    kernels = sorted(
        (
            event
            for event in events
            if event.get("cat") == "kernel" and needle in event.get("name", "")
        ),
        key=lambda event: event["ts"],
    )
    if len(kernels) != len(order):
        raise RuntimeError(
            f"{needle}: expected {len(order)} device events, got {len(kernels)}"
        )
    durations = {tiles: [] for tiles in (1, 4, 16)}
    for tiles, event in zip(order, kernels):
        durations[tiles].append(event["dur"])
    report = {
        "case": "fused" if fused else "extend",
        "batch": batch,
        "prefix_len": prefix_len,
        "extend_len": extend_len,
        "page_size": page_size,
        "last_dtype": "int32",
        "samples": samples,
        "timing": "torch.profiler XPU kernel event duration; rotated launch order",
        "tiles": {
            tiles: {
                "samples_us": values,
                "p20": quantile(values, 0.2),
                "p50": median(values),
                "p80": quantile(values, 0.8),
                "min": min(values),
                "max": max(values),
            }
            for tiles, values in durations.items()
        },
    }
    print(json.dumps(report), flush=True)
    torch.xpu.synchronize()
    time.sleep(2)


def measure_write_groups(batch, prefix_len, extend_len, samples):
    _, _, main_write = main_memory_ops()
    seq_len = prefix_len + extend_len
    prefix_lens = torch.full((batch,), prefix_len, dtype=torch.int64, device="xpu")
    seq_lens = torch.full((batch,), seq_len, dtype=torch.int64, device="xpu")
    extend_lens = torch.full((batch,), extend_len, dtype=torch.int64, device="xpu")
    req_indices = torch.arange(batch, dtype=torch.int64, device="xpu")
    prefix_values = torch.arange(
        max(1, batch * prefix_len), dtype=torch.int64, device="xpu"
    )
    pointers = torch.tensor(
        [prefix_values.data_ptr() + index * prefix_len * 8 for index in range(batch)],
        dtype=torch.uint64,
        device="xpu",
    )
    locations = torch.arange(batch * extend_len, dtype=torch.int64, device="xpu")
    expected = torch.empty((batch, seq_len), dtype=torch.int32, device="xpu")
    main_write[(batch,)](
        expected,
        req_indices,
        pointers,
        prefix_lens,
        seq_lens,
        extend_lens,
        locations,
        expected.stride(0),
    )
    outputs = {groups: torch.empty_like(expected) for groups in (1, 2, 4, 8)}

    def run(groups):
        torch.ops.sgl_kernel.write_req_to_token_pool_tiled_for_benchmark(
            outputs[groups],
            req_indices,
            pointers,
            prefix_lens,
            seq_lens,
            extend_lens,
            locations,
            groups,
        )

    for _ in range(5):
        for groups in outputs:
            run(groups)
    torch.xpu.synchronize()
    for output in outputs.values():
        torch.testing.assert_close(output.cpu(), expected.cpu(), rtol=0, atol=0)
    order = []
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.XPU,
        ]
    ) as profiler:
        for iteration in range(samples):
            choices = (1, 2, 4, 8)
            choices = choices[iteration % 4 :] + choices[: iteration % 4]
            for groups in choices:
                run(groups)
                order.append(groups)
        torch.xpu.synchronize()
    with tempfile.TemporaryDirectory() as directory:
        trace = Path(directory) / "trace.json"
        profiler.export_chrome_trace(str(trace))
        events = json.loads(trace.read_text())["traceEvents"]
    kernels = sorted(
        (
            event
            for event in events
            if event.get("cat") == "kernel"
            and "WriteReqToTokenPoolKernel" in event.get("name", "")
        ),
        key=lambda event: event["ts"],
    )
    if len(kernels) != len(order):
        raise RuntimeError(
            f"expected {len(order)} writer device events, got {len(kernels)}"
        )
    durations = {groups: [] for groups in outputs}
    for groups, event in zip(order, kernels):
        durations[groups].append(event["dur"])
    print(
        json.dumps(
            {
                "case": "write_groups",
                "batch": batch,
                "prefix_len": prefix_len,
                "extend_len": extend_len,
                "samples": samples,
                "device": torch.xpu.get_device_name(0),
                "timing": "torch.profiler XPU kernel event duration; rotated launch order",
                "groups": {
                    groups: {
                        "samples_us": values,
                        "p20": quantile(values, 0.2),
                        "p50": median(values),
                        "p80": quantile(values, 0.8),
                        "min": min(values),
                        "max": max(values),
                    }
                    for groups, values in durations.items()
                },
            }
        ),
        flush=True,
    )
    torch.xpu.synchronize()
    time.sleep(2)


def measure_aux(label, run_triton, run_sycl, old_name, new_name, check, samples, only):
    if only != "all" and only != label:
        return
    for _ in range(10):
        run_triton()
        run_sycl()
    torch.xpu.synchronize()
    check()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.XPU,
        ]
    ) as profiler:
        for _ in range(samples):
            run_triton()
        for _ in range(samples):
            run_sycl()
        torch.xpu.synchronize()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "trace.json"
        profiler.export_chrome_trace(str(path))
        events = json.loads(path.read_text())["traceEvents"]
    old = [
        event["dur"]
        for event in events
        if event.get("cat") == "kernel" and old_name in event["name"]
    ]
    new = [
        event["dur"]
        for event in events
        if event.get("cat") == "kernel" and new_name in event["name"]
    ]
    assert len(old) == len(new) == samples, (label, len(old), len(new))
    old_us, new_us = median(old), median(new)
    print(
        f"{label}: Triton={old_us:.3f} us SYCL={new_us:.3f} us speedup={old_us / new_us:.3f}x",
        flush=True,
    )
    if label in ("write_req_to_token_pool_xe35", "alloc_extend"):
        print(
            f"{label} samples: Triton_min={min(old):.3f} p25={sorted(old)[len(old)//4]:.3f} p75={sorted(old)[3*len(old)//4]:.3f} max={max(old):.3f}; SYCL_min={min(new):.3f} p25={sorted(new)[len(new)//4]:.3f} p75={sorted(new)[3*len(new)//4]:.3f} max={max(new):.3f}",
            flush=True,
        )
    torch.xpu.synchronize()
    time.sleep(2)


def run_aux(args):
    alloc_extend_kernel, alloc_decode_kernel, write_req_to_token_pool_triton = (
        main_memory_ops()
    )

    device = "xpu"
    batch = args.batch

    seq = torch.tensor(
        [4096, 4097, 4098, 4104, 4112, 4096, 4100, 4111],
        dtype=torch.int64,
        device=device,
    )
    last = torch.arange(8, dtype=torch.int32, device=device) + 4095
    free_pages = torch.arange(1, 6001, dtype=torch.int64, device=device)
    old_out = torch.empty(8, dtype=torch.int64, device=device)
    new_out = torch.empty_like(old_out)
    measure_aux(
        "alloc_decode",
        lambda: alloc_decode_kernel[(8,)](seq, last, free_pages, old_out, 8, 64),
        lambda: torch.ops.sgl_kernel.paged_alloc_decode_xe35(
            seq, last, free_pages, new_out, 64
        ),
        "alloc_decode_kernel",
        "PagedAllocDecodeKernel",
        lambda: torch.testing.assert_close(
            new_out.cpu(), old_out.cpu(), rtol=0, atol=0
        ),
        args.samples,
        args.only,
    )

    prefix = torch.zeros(batch, dtype=torch.int64, device=device)
    lengths = torch.full((batch,), 4096, dtype=torch.int64, device=device)
    extend_last = torch.full((batch,), -1, dtype=torch.int32, device=device)
    old_out = torch.empty(batch * 4096, dtype=torch.int64, device=device)
    new_out = torch.empty_like(old_out)
    measure_aux(
        "alloc_extend",
        lambda: alloc_extend_kernel[(batch,)](
            prefix, lengths, extend_last, free_pages, old_out, 8, 64
        ),
        lambda: torch.ops.sgl_kernel.paged_alloc_extend_xe35(
            prefix, lengths, extend_last, free_pages, new_out, 64
        ),
        "alloc_extend_kernel",
        "PagedAllocExtendKernel",
        lambda: torch.testing.assert_close(
            new_out.cpu(), old_out.cpu(), rtol=0, atol=0
        ),
        args.samples,
        args.only,
    )

    old_pool = torch.empty((batch, 4096), dtype=torch.int32, device=device)
    new_pool = torch.empty_like(old_pool)
    req_indices = torch.arange(batch, dtype=torch.int64, device=device)
    dummy_prefix = torch.zeros(1, dtype=torch.int64, device=device)
    prefix_pointers = torch.tensor(
        [dummy_prefix.data_ptr()] * batch, dtype=torch.uint64, device=device
    )
    out_cache_loc = torch.arange(batch * 4096, dtype=torch.int64, device=device)
    measure_aux(
        "write_req_to_token_pool_xe35",
        lambda: write_req_to_token_pool_triton[(batch,)](
            old_pool,
            req_indices,
            prefix_pointers,
            prefix,
            lengths,
            lengths,
            out_cache_loc,
            old_pool.stride(0),
        ),
        lambda: torch.ops.sgl_kernel.write_req_to_token_pool_xe35(
            new_pool,
            req_indices,
            prefix_pointers,
            prefix,
            lengths,
            lengths,
            out_cache_loc,
        ),
        "write_req_to_token_pool_triton",
        "WriteReqToTokenPoolKernel",
        lambda: torch.testing.assert_close(
            new_pool.cpu(), old_pool.cpu(), rtol=0, atol=0
        ),
        args.samples,
        args.only,
    )


def run_profiler(args):
    if "decode" in args.cases:
        for step in args.decode_steps:
            compare_decode(args.samples, args.main_only, step)
    if "write" in args.cases:
        compare_write(args.samples, 0, 4096, args.main_only)
        compare_write(args.samples, 4096, 1, args.main_only)
        compare_write(args.samples, 5119, 1, args.main_only)
    for page_size in args.page_sizes:
        for prefix_len in args.prefix_lengths:
            for extend_len in args.lengths:
                if "extend" in args.cases:
                    compare_extend(
                        args.batch,
                        prefix_len,
                        extend_len,
                        page_size,
                        args.samples,
                        False,
                        args.main_only,
                    )
                if "fused" in args.cases:
                    compare_extend(
                        args.batch,
                        prefix_len,
                        extend_len,
                        page_size,
                        args.samples,
                        True,
                        args.main_only,
                    )


def main():
    args = parse_args()
    global torch, triton, memory_setup, regressions
    import torch
    import triton

    memory_setup = args
    regressions = []
    if not args.main_only:
        kernel = importlib.import_module("sgl_kernel")
        if not torch.xpu.is_available() or not kernel.is_xe3_arch():
            raise RuntimeError(
                "SYCL memory kernels require Xe35 compute capability 3.5"
            )

    required = []
    if args.mode == "tiles" or (args.mode == "event" and not args.production_only):
        cases = args.cases if args.mode == "tiles" else (args.op,)
        required = [
            (
                "paged_alloc_extend_and_write_tiled_for_benchmark"
                if case == "fused"
                else "paged_alloc_extend_tiled_for_benchmark"
            )
            for case in cases
        ]
    elif args.mode == "write-groups":
        required = ["write_req_to_token_pool_tiled_for_benchmark"]
    missing = [name for name in required if not hasattr(torch.ops.sgl_kernel, name)]
    if missing:
        raise RuntimeError(
            f"Selected mode requires unavailable tuning capabilities: {missing}"
        )
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "device": torch.xpu.get_device_name(0),
                "main_source": {
                    operator.__name__: operator.fn.__code__.co_filename
                    for operator in main_memory_ops()
                },
                "mode": args.mode,
                "serving_scope": {
                    "batch": 8,
                    "isl": 4096,
                    "osl": 1024,
                    "tp": 4,
                    "ep": 1,
                },
            }
        ),
        flush=True,
    )
    if args.mode == "profiler":
        run_profiler(args)
    elif args.mode == "event":
        measure_event(args)
    elif args.mode == "aux":
        run_aux(args)
    elif args.mode == "prefill":
        compare_extend(8, 0, 4096, 64, args.samples, fused=True)
    else:
        for prefix in args.prefix_lengths:
            for length in args.lengths:
                if args.mode == "write-groups":
                    measure_write_groups(args.batch, prefix, length, args.samples)
                else:
                    for fused in (False, True):
                        if ("fused" if fused else "extend") in args.cases:
                            measure_tiles(
                                args.batch,
                                prefix,
                                length,
                                args.page_size,
                                args.samples,
                                fused,
                            )

    if regressions:
        raise RuntimeError(f"Memory performance regressions: {regressions}")


if __name__ == "__main__":
    main()
