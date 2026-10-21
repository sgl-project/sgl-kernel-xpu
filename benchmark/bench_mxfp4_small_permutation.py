"""Device time for the registered Muse Spark B8 small permutation kernel."""

import argparse
import ast
import hashlib
import json
import os
import statistics
import tempfile
import time
from pathlib import Path


def load_baseline_kernel(source_path):
    source = source_path.read_text()
    tree = ast.parse(source, filename=str(source_path))
    factory = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_get_xpu_w4a4_small_permutation_kernel"
    )
    namespace = {"_xpu_w4a4_small_permutation_kernel": None}
    exec(
        compile(ast.Module(body=[factory], type_ignores=[]), str(source_path), "exec"),
        namespace,
    )
    print(
        f"baseline_source={source_path} "
        f"sha256={hashlib.sha256(source.encode()).hexdigest()}"
    )
    return namespace[factory.name]()


def profile_installed_operator(operation, topk_ids, outputs, args, torch, sgl_kernel):
    for warmup_index in range(200):
        operation()
    torch.xpu.synchronize()
    time.sleep(2)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.XPU,
        ]
    ) as profiler:
        for sample_index in range(args.profile_samples):
            operation()
        torch.xpu.synchronize()
    with tempfile.TemporaryDirectory() as directory:
        trace = Path(directory) / "trace.json"
        profiler.export_chrome_trace(str(trace))
        events = json.loads(trace.read_text())["traceEvents"]
    kernel_samples = [event["dur"] for event in events if event.get("cat") == "kernel"]
    if len(kernel_samples) != args.profile_samples:
        raise RuntimeError(
            f"Expected {args.profile_samples} kernels, got {len(kernel_samples)}"
        )
    package_directory = Path(sgl_kernel.__file__).resolve().parent
    libraries = [
        library
        for library in package_directory.glob("*.so")
        if "smallpermutation" in library.name.lower()
        or "small_permutation" in library.name.lower()
        or library.name.startswith("common_ops")
    ]
    result = {
        "implementation": args.installed_operator,
        "package": str(package_directory),
        "torch": torch.__version__,
        "device": torch.xpu.get_device_name(),
        "shape": [8, 8, args.experts, args.alignment],
        "index_dtype": args.index_dtype,
        "routing": args.routing,
        "input_ids": topk_ids.cpu().tolist(),
        "samples_us": kernel_samples,
        "kernel_names": sorted(
            {event["name"] for event in events if event.get("cat") == "kernel"}
        ),
        "median_us": statistics.median(kernel_samples),
        "timing": "XPU profiler kernel durations; excludes host enqueue gaps",
        "outputs": [output.cpu().tolist() for output in outputs],
        "libraries": {
            str(library): hashlib.sha256(library.read_bytes()).hexdigest()
            for library in libraries
        },
    }
    args.result_file.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"installed_operator={args.installed_operator} "
        f"median_us={result['median_us']:.6f}"
    )


def main_xe35() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving-batch", type=int, default=8)
    parser.add_argument("--isl", type=int, default=4096)
    parser.add_argument("--osl", type=int, default=1024)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--experts", type=int, choices=[128, 256, 512], default=256)
    parser.add_argument("--alignment", type=int, choices=[4, 16], default=4)
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--rep-ms", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--installed-operator", choices=["candidate", "legacy"])
    parser.add_argument("--profile-samples", type=int, default=201)
    parser.add_argument("--result-file", type=Path)
    parser.add_argument("--max-regression", type=float, default=0.0)
    parser.add_argument("--index-dtype", choices=["int32", "int64"], default="int32")
    parser.add_argument(
        "--routing", choices=["balanced", "random", "colliding"], default="random"
    )
    args = parser.parse_args()
    if (args.serving_batch, args.isl, args.osl, args.tp, args.ep) != (
        8,
        4096,
        1024,
        4,
        1,
    ) or os.environ.get("BATCHES", "8") != "8":
        parser.error("Muse-Spark serving scope requires B=8, ISL=4096, OSL=1024, TP4/EP1")
    if (
        args.warmup_ms <= 0
        or args.rep_ms <= 0
        or args.rounds <= 0
        or args.max_regression < 0
    ):
        parser.error("warmup and repetition times must be positive")
    if args.installed_operator:
        if args.profile_samples <= 0 or args.result_file is None or args.baseline_source:
            parser.error(
                "Installed-package profiling requires positive samples, "
                "--result-file and no Triton baseline"
            )
    elif args.baseline_source is None or not args.baseline_source.is_file():
        parser.error("--baseline-source must name an existing historical mxfp4.py")

    import sgl_kernel
    import torch

    if args.installed_operator == "legacy":
        permutation_operation = torch.ops.sgl_kernel.mxfp4_small_permutation.default
    else:
        from sgl_kernel import mxfp4_small_permutation_xe35

        permutation_operation = mxfp4_small_permutation_xe35

    if not torch.xpu.is_available() or not sgl_kernel.is_xe3_arch():
        raise RuntimeError("Small permutation benchmark requires Xe35 (capability 3.5)")

    tokens, topk = 8, 8
    pairs = tokens * topk
    rows = pairs * args.alignment
    index_dtype = getattr(torch, args.index_dtype)
    generator = torch.Generator().manual_seed(42)
    if args.routing == "random":
        ids_cpu = torch.stack(
            [
                torch.randperm(args.experts, generator=generator)[:topk]
                for _ in range(tokens)
            ]
        )
    elif args.routing == "colliding":
        ids_cpu = torch.arange(topk).expand(tokens, topk).contiguous()
    else:
        ids_cpu = torch.arange(pairs).reshape(tokens, topk) % args.experts
    topk_ids = ids_cpu.to(device="xpu", dtype=index_dtype)
    padded_counts = torch.empty(args.experts, device="xpu", dtype=torch.int32)
    perm_src = torch.empty(rows, device="xpu", dtype=torch.int32)
    expanded_to_permuted = torch.empty(pairs, device="xpu", dtype=torch.int32)
    scale_map = torch.empty(rows, 2, device="xpu", dtype=torch.int32)

    def run_xe35() -> None:
        permutation_operation(
            topk_ids,
            padded_counts,
            perm_src,
            expanded_to_permuted,
            scale_map,
            args.alignment,
        )

    run_xe35()
    torch.xpu.synchronize()
    print(f"device={torch.xpu.get_device_name()} dtype={index_dtype} routing={args.routing}")
    if args.installed_operator:
        profile_installed_operator(
            run_xe35,
            topk_ids,
            (padded_counts, perm_src, expanded_to_permuted, scale_map),
            args,
            torch,
            sgl_kernel,
        )
        return

    import triton.testing

    baseline_kernel = (
        load_baseline_kernel(args.baseline_source.resolve())
        if args.baseline_source
        else None
    )
    baseline_outputs = [
        torch.empty_like(output)
        for output in (padded_counts, perm_src, expanded_to_permuted, scale_map)
    ]

    def run_baseline():
        baseline_kernel[(1,)](
            topk_ids,
            *baseline_outputs,
            pairs,
            rows,
            topk,
            NUM_EXPERTS=args.experts,
            ALIGNMENT=args.alignment,
            BLOCK_PAIRS=max(16, triton.next_power_of_2(pairs)),
            BLOCK_ROWS=64,
            num_warps=8,
        )

    if baseline_kernel:
        run_baseline()
        for candidate, baseline in zip(
            (padded_counts, perm_src, expanded_to_permuted, scale_map), baseline_outputs
        ):
            torch.testing.assert_close(candidate, baseline, rtol=0, atol=0)
        print("all four outputs match baseline bit-for-bit")
    samples = {"candidate": [], "baseline": []}
    for round_index in range(args.rounds):
        paths = [("candidate", run_xe35)]
        if baseline_kernel:
            paths.append(("baseline", run_baseline))
        if round_index % 2:
            paths.reverse()
        for name, operation in paths:
            torch.xpu.synchronize()
            time.sleep(2)
            quantiles = triton.testing.do_bench(
                operation,
                warmup=args.warmup_ms,
                rep=args.rep_ms,
                quantiles=[0.5, 0.2, 0.8],
            )
            samples[name].append(quantiles[0])
            print(f"round={round_index} path={name} p50={quantiles[0]:.6f} ms")
    p50 = statistics.median(samples["candidate"])
    print(
        f"B8 topk={topk} experts={args.experts} alignment={args.alignment}: "
        f"candidate_p50={p50:.6f} ms"
    )
    if baseline_kernel:
        baseline_p50 = statistics.median(samples["baseline"])
        ratio = p50 / baseline_p50
        print(
            f"baseline_p50={baseline_p50:.6f} ms "
            f"ratio={ratio:.4f} speedup={1 / ratio:.3f}x"
        )
        if ratio > 1 + args.max_regression:
            raise RuntimeError(f"performance regression: candidate/baseline={ratio:.4f}")


if __name__ == "__main__":
    main_xe35()
