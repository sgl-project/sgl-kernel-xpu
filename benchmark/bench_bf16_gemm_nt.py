"""Compare BF16 GEMM dispatch and kernels with XPU graph event timing."""

import argparse
import csv
import hashlib
import importlib.util
import math
import os
import re
import statistics
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path


def gemm_kernel_ids_xe35(library):
    image = Path(library).read_bytes()
    identifiers = {
        identifier.decode("ascii")
        for identifier in re.findall(rb"_ZTS[^\x00\r\n ]+", image)
        if b"GemmUniversal" in identifier
    }
    if not identifiers:
        raise ValueError(f"No SYCL GEMM kernel identifiers found in {library}")
    return identifiers, hashlib.sha256(image).hexdigest()


def validate_kernel_isolation_xe35(candidate_library, baseline_library):
    candidate_ids, candidate_hash = gemm_kernel_ids_xe35(candidate_library)
    baseline_ids, baseline_hash = gemm_kernel_ids_xe35(baseline_library)
    shared = candidate_ids & baseline_ids
    if shared:
        raise ValueError(
            f"Candidate and baseline share {len(shared)} SYCL GEMM kernel IDs; "
            "rebuild the baseline with an isolated CUTLASS namespace"
        )
    mapped = {
        Path(fields[5]).resolve()
        for line in Path("/proc/self/maps").read_text().splitlines()
        if len(fields := line.split(maxsplit=5)) == 6 and fields[5].startswith("/")
    }
    for library in (candidate_library, baseline_library):
        if Path(library).resolve() not in mapped:
            raise ValueError(f"Kernel library is not loaded: {library}")
    for library in mapped:
        if (
            "bf16" in library.name
            and "gemm" in library.name
            and library != Path(candidate_library).resolve()
        ):
            image = library.read_bytes()
            identifiers = {
                identifier.decode("ascii")
                for identifier in re.findall(rb"_ZTS[^\x00\r\n ]+", image)
                if b"GemmUniversal" in identifier
            }
            if candidate_ids & identifiers:
                raise ValueError(
                    f"Loaded library shares candidate kernel IDs: {library}"
                )
    print(
        f"isolated_sycl_ids candidate={len(candidate_ids)} baseline={len(baseline_ids)} "
        f"candidate_sha256={candidate_hash} baseline_sha256={baseline_hash}",
        flush=True,
    )


def capture_xe35(functions, warmup, calls):
    for index in range(max(warmup, len(functions))):
        functions[index % len(functions)]()
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    try:
        with torch.xpu.graph(graph):
            for index in range(calls):
                functions[index % len(functions)]()
        for _ in range(warmup):
            graph.replay()
        torch.xpu.synchronize()
    except BaseException:
        graph.reset()
        raise
    return graph


@contextmanager
def case_graphs_xe35(functions, names, warmup, calls):
    graphs = {}
    try:
        for name in names:
            graphs[name] = capture_xe35(
                [entry[name] for entry in functions], warmup, calls
            )
        yield graphs
    finally:
        try:
            torch.xpu.synchronize()
        finally:
            for graph in graphs.values():
                graph.reset()
            graphs.clear()
            torch.xpu.empty_cache()


def measure_xe35(graph, calls):
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / calls


def cpu_reference_xe35(activations, weight, chunk_bytes=64 * 1024**2):
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    columns, inner = weight.shape
    activation_cpu = activations.cpu().float()
    reference = torch.empty(
        (activations.shape[0], columns), dtype=torch.float32, device="cpu"
    )
    chunk_rows = max(1, chunk_bytes // max(1, inner * weight.element_size()))
    for offset in range(0, columns, chunk_rows):
        end = min(offset + chunk_rows, columns)
        weight_cpu = weight[offset:end].cpu().float()
        reference[:, offset:end] = torch.mm(activation_cpu, weight_cpu.T)
    return reference


def force_kernel_xe35(output, activations, weight, packed, operator=None):
    (operator or gemm_operator)(output, activations, weight, 2 if packed else 0)
    return output


def parse_args_xe35():
    if os.environ.get("BATCHES", "8") != "8":
        raise SystemExit("BATCHES must be 8; rejected before GPU work")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving-batch", type=int, choices=(8,), default=8)
    parser.add_argument("--mode", choices=("all", "dispatch", "kernel"), default="all")
    parser.add_argument(
        "--shape", nargs=2, action="append", type=int, metavar=("N", "K")
    )
    parser.add_argument(
        # M counts GEMM rows, not the serving batch size.
        "--m",
        type=int,
        nargs="+",
        default=[8],
    )
    parser.add_argument("--n", type=int, default=32768)
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", "--repeats", dest="repeats", type=int, default=20)
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-mode", choices=("hot", "cold"), default="hot")
    parser.add_argument("--working-set-mib", type=int, default=512)
    parser.add_argument("--output", help="Optional CSV with medians and sample spans")
    parser.add_argument("--baseline-source", type=Path, help="Baseline checkout root")
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Time the candidate and baseline together on identical inputs",
    )
    parser.add_argument(
        "--baseline-library",
        type=Path,
        help="Registration bridge that explicitly binds the baseline GEMM library",
    )
    parser.add_argument("--baseline-kernel-library", type=Path)
    parser.add_argument("--candidate-kernel-library", type=Path)
    args = parser.parse_args()
    if (args.baseline_source is None) != (args.baseline_library is None):
        parser.error(
            "--baseline-source and --baseline-library must be supplied together"
        )
    if args.compare_baseline and args.baseline_source is None:
        parser.error("--compare-baseline requires the baseline source and library")
    if args.baseline_source is not None and args.baseline_kernel_library is None:
        parser.error("baseline timing requires --baseline-kernel-library")
    if args.compare_baseline and (
        args.baseline_kernel_library is None or args.candidate_kernel_library is None
    ):
        parser.error("paired timing requires both --*-kernel-library paths")
    args.shapes = args.shape or [(args.n, args.k)]
    if any(
        columns <= 0 or inner <= 0 or columns % 16 or inner % 16
        for columns, inner in args.shapes
    ) or any(rows <= 0 for rows in args.m):
        parser.error("positive shapes with N and K divisible by 16 are required")
    if args.warmup < 0 or args.repeats <= 0 or args.calls <= 0:
        parser.error("--warmup must be nonnegative; repeats and calls must be positive")
    if args.working_set_mib < 512:
        parser.error("--working-set-mib must be at least 512")
    if args.cache_mode == "cold" and any(
        args.calls < math.ceil(args.working_set_mib * 1024**2 / (columns * inner * 2))
        for columns, inner in args.shapes
    ):
        parser.error("--calls must cover the entire cold-weight ring for each shape")
    return args


def case_functions_xe35(
    activations, weight, packed_weight, baseline_packed_weight=None
):
    native_output = torch.empty(
        (activations.shape[0], weight.shape[0]),
        dtype=activations.dtype,
        device=activations.device,
    )
    packed_output = torch.empty(
        (activations.shape[0], packed_weight.shape[1]),
        dtype=activations.dtype,
        device=activations.device,
    )
    functions = {
        "native_mm": lambda: torch.mm(activations, weight.T),
        "native_cute": lambda: force_kernel_xe35(
            native_output, activations, weight, False
        ),
        "packed_mm": lambda: torch.mm(activations, packed_weight),
        "packed_cute": lambda: force_kernel_xe35(
            packed_output, activations, packed_weight, True
        ),
        "native_dispatch": lambda: bf16_gemm_nt_xe35(activations, weight),
        "packed_dispatch": lambda: bf16_gemm_nt_xe35(
            activations, packed_weight, packed_weight=True
        ),
        "native_mm_after": lambda: torch.mm(activations, weight.T),
    }
    if baseline_packed_weight is not None:
        baseline_native_output = torch.empty(
            (activations.shape[0], weight.shape[0]),
            dtype=activations.dtype,
            device=activations.device,
        )
        baseline_output = torch.empty(
            (activations.shape[0], baseline_packed_weight.shape[1]),
            dtype=activations.dtype,
            device=activations.device,
        )
        functions.update(
            {
                "baseline_native_dispatch": lambda: baseline_dispatch(
                    activations, weight
                ),
                "baseline_packed_dispatch": lambda: baseline_dispatch(
                    activations, baseline_packed_weight, packed_weight=True
                ),
                "baseline_native_cute": lambda: force_kernel_xe35(
                    baseline_native_output,
                    activations,
                    weight,
                    False,
                    baseline_operator,
                ),
                "baseline_packed_cute": lambda: force_kernel_xe35(
                    baseline_output,
                    activations,
                    baseline_packed_weight,
                    True,
                    baseline_operator,
                ),
                "baseline_packed_mm": lambda: torch.mm(
                    activations, baseline_packed_weight
                ),
            }
        )
    return functions


def benchmark_case_xe35(
    args, weights, packed_weights, rows, names, baseline_packed_weights=None
):
    weight = weights[0]
    columns, inner = weight.shape
    activations = torch.randn((rows, inner), dtype=torch.bfloat16, device=weight.device)
    functions = [
        case_functions_xe35(
            activations,
            native,
            packed,
            (
                baseline_packed_weights[index]
                if baseline_packed_weights is not None
                else None
            ),
        )
        for index, (native, packed) in enumerate(zip(weights, packed_weights))
    ]
    if baseline_packed_weights is not None and not getattr(
        args, "kernel_isolation_verified", False
    ):
        functions[0]["packed_cute"]()
        functions[0]["baseline_packed_cute"]()
        torch.xpu.synchronize()
        validate_kernel_isolation_xe35(
            args.candidate_kernel_library, args.baseline_kernel_library
        )
        args.kernel_isolation_verified = True
    reference = cpu_reference_xe35(activations, weight)
    for name in names:
        output = functions[0][name]()
        torch.testing.assert_close(
            output[:, :columns].cpu().float(), reference, atol=5e-3, rtol=5e-3
        )
    samples = {name: [] for name in names}
    with case_graphs_xe35(functions, names, args.warmup, args.calls) as graphs:
        for index in range(args.repeats):
            comparison_names = tuple(
                name for name in names if name != "native_mm_after"
            )
            order = (
                comparison_names
                if index % 2 == 0
                else tuple(reversed(comparison_names))
            )
            if "native_mm_after" in names:
                order = (*order, "native_mm_after")
            for name in order:
                samples[name].append(measure_xe35(graphs[name], args.calls))
    result = {
        "m": rows,
        "n": columns,
        "k": inner,
        "weight_count": len(weights),
        "weight_bytes": sum(entry.numel() * entry.element_size() for entry in weights),
        "cache_mode": args.cache_mode,
        "xpu_allocated_bytes": torch.xpu.memory_allocated(),
        "xpu_reserved_bytes": torch.xpu.memory_reserved(),
    }
    for name in names:
        result[name + "_ms"] = statistics.median(samples[name])
        result[name + "_span_ms"] = max(samples[name]) - min(samples[name])
    return result


def share_matching_packed_weights_xe35(packed_weights, baseline_packed_weights):
    for index, (packed, baseline_packed) in enumerate(
        zip(packed_weights, baseline_packed_weights)
    ):
        if (
            packed.shape == baseline_packed.shape
            and packed.stride() == baseline_packed.stride()
        ):
            if not torch.equal(packed, baseline_packed):
                raise AssertionError("candidate and baseline packing disagree")
            baseline_packed_weights[index] = packed
    return baseline_packed_weights


def main_xe35():
    global torch, bf16_gemm_nt_xe35, pack_bf16_gemm_weight_xe35
    global gemm_operator
    global baseline_dispatch, baseline_operator, baseline_pack
    args = parse_args_xe35()
    import sgl_kernel
    import torch

    if args.baseline_source is not None:
        os.environ["SGL_BF16_BASELINE_KERNEL_LIBRARY"] = str(
            args.baseline_kernel_library.resolve()
        )
        torch.ops.load_library(str(args.baseline_library.resolve()))
        gemm_operator = torch.ops.sgl_kernel_baseline.bf16_gemm_nt.default
        torch.ops.sgl_kernel.bf16_gemm_nt = torch.ops.sgl_kernel_baseline.bf16_gemm_nt
        source = args.baseline_source / "python/sgl_kernel/gemm.py"
        specification = importlib.util.spec_from_file_location(
            "bf16_gemm_baseline", source
        )
        baseline = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(baseline)
        baseline_dispatch = baseline.bf16_gemm_nt
        baseline_pack = baseline.pack_bf16_gemm_weight
        baseline_operator = gemm_operator
        bf16_gemm_nt_xe35 = baseline_dispatch
        pack_bf16_gemm_weight_xe35 = baseline_pack
    if args.baseline_source is None or args.compare_baseline:
        from sgl_kernel import bf16_gemm_nt_xe35, pack_bf16_gemm_weight_xe35

        gemm_operator = torch.ops.sgl_kernel.bf16_gemm_nt_xe35.default

    if not torch.xpu.is_available() or not sgl_kernel.is_xe3_arch():
        raise RuntimeError("BF16 NT GEMM benchmark requires Xe35 (capability 3.5)")

    names = {
        "kernel": ("native_mm", "native_cute", "packed_mm", "packed_cute"),
        "dispatch": (
            "native_mm",
            "native_dispatch",
            "packed_mm",
            "packed_dispatch",
        ),
        "all": (
            "native_mm",
            "native_cute",
            "packed_mm",
            "packed_cute",
            "native_dispatch",
            "packed_dispatch",
            "native_mm_after",
        ),
    }[args.mode]
    if args.compare_baseline:
        baseline_names = []
        if args.mode != "kernel":
            baseline_names += ["baseline_native_dispatch", "baseline_packed_dispatch"]
        if args.mode != "dispatch":
            baseline_names += ["baseline_native_cute", "baseline_packed_cute"]
        names = (*names, *baseline_names, "baseline_packed_mm")
    torch.manual_seed(args.seed)
    torch.xpu.synchronize()
    time.sleep(2)
    print(
        f"device={torch.xpu.get_device_name(0)} torch={torch.__version__} "
        f"backend={'paired' if args.compare_baseline else 'baseline' if args.baseline_source is not None else 'candidate'} "
        f"cache_mode={args.cache_mode} working_set_mib={args.working_set_mib} "
        f"timing=xpu_graph_events mode={args.mode} "
        f"calls={args.calls} repeats={args.repeats}",
        flush=True,
    )
    fields = ["m", "n", "k", "weight_count", "weight_bytes", "cache_mode"]
    fields += ["xpu_allocated_bytes", "xpu_reserved_bytes"]
    fields += [name + "_ms" for name in names]
    fields += [name + "_span_ms" for name in names]
    output_context = (
        open(args.output, "w", newline="") if args.output else nullcontext(None)
    )
    with output_context as handle:
        writer = csv.DictWriter(handle, fieldnames=fields) if handle else None
        if writer:
            writer.writeheader()
        completed_cases = 0
        for columns, inner in args.shapes:
            weight_count = (
                math.ceil(args.working_set_mib * 1024**2 / (columns * inner * 2))
                if args.cache_mode == "cold"
                else 1
            )
            weights = [
                torch.randn((columns, inner), dtype=torch.bfloat16, device="xpu")
                for _ in range(weight_count)
            ]
            packed_weights = [pack_bf16_gemm_weight_xe35(weight) for weight in weights]
            baseline_packed_weights = (
                [baseline_pack(weight) for weight in weights]
                if args.compare_baseline
                else None
            )
            if args.compare_baseline:
                baseline_packed_weights = share_matching_packed_weights_xe35(
                    packed_weights, baseline_packed_weights
                )
            for rows in args.m:
                if completed_cases:
                    torch.xpu.synchronize()
                    time.sleep(2)
                result = benchmark_case_xe35(
                    args, weights, packed_weights, rows, names, baseline_packed_weights
                )
                if writer:
                    writer.writerow(result)
                    handle.flush()
                timings = " ".join(
                    f"{name}={result[name + '_ms']:.5f}ms" for name in names
                )
                print(
                    f"M={rows} N={columns} K={inner} cache={args.cache_mode} "
                    f"weights={weight_count} {timings}",
                    flush=True,
                )
                completed_cases += 1


if __name__ == "__main__":
    main_xe35()
