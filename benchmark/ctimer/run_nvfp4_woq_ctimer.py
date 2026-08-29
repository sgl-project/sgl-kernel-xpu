#!/usr/bin/env python3
"""Compare NVFP4 WoQ with MXFP8 using CUTLASS GPU_Clock on the CRI simulator."""

import argparse
import importlib.util
import pathlib
import sysconfig

import torch

HERE = pathlib.Path(__file__).resolve().parent
EXTENSION = HERE / f"nvfp4_ctimer_ext{sysconfig.get_config_var('EXT_SUFFIX')}"
WORKSPACE_BYTES = 64 * 1024 * 1024


def load_extension():
    if not EXTENSION.exists():
        raise RuntimeError(f"missing {EXTENSION}; run build_nvfp4_woq_ctimer.sh first")
    spec = importlib.util.spec_from_file_location("nvfp4_ctimer_ext", EXTENSION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_inputs(groups: int, m: int, n: int, k: int):
    dev = "xpu"
    total_m = groups * m
    a_mxfp8 = torch.zeros((total_m, k), dtype=torch.float8_e4m3fn, device=dev)
    a_bf16 = torch.zeros((total_m, k), dtype=torch.bfloat16, device=dev)
    scales_a = torch.full((total_m, k // 32), 127, dtype=torch.uint8, device=dev)
    b_nvfp4 = torch.zeros((groups, n, k // 2), dtype=torch.uint8, device=dev)
    sfmx_b = torch.full((groups, k // 32, n), 127, dtype=torch.uint8, device=dev)
    c_b = torch.ones((groups, n, k // 16), dtype=torch.bfloat16, device=dev)
    b_mxfp8 = torch.zeros((groups, n, k), dtype=torch.float8_e4m3fn, device=dev)
    scales_b_mxfp8 = torch.full(
        (groups, n, k // 32), 127, dtype=torch.uint8, device=dev
    )
    problem_sizes = torch.tensor([[m, n, k]] * groups, dtype=torch.int32, device=dev)
    expert_offsets = torch.arange(groups, dtype=torch.int32, device=dev) * m
    output = torch.empty((total_m, n), dtype=torch.float32, device=dev)
    workspace = torch.empty((WORKSPACE_BYTES,), dtype=torch.uint8, device=dev)
    empty_ptrs = torch.empty((0,), dtype=torch.int64, device=dev)
    zero_stride = torch.zeros((groups,), dtype=torch.int64, device=dev)
    zero_layout = torch.zeros((groups, 5), dtype=torch.int32, device=dev)
    torch.xpu.synchronize()
    return {
        "output": output,
        "a_mxfp8": a_mxfp8,
        "a_bf16": a_bf16,
        "scales_a": scales_a,
        "b_nvfp4": b_nvfp4,
        "sfmx_b": sfmx_b,
        "c_b": c_b,
        "b_mxfp8": b_mxfp8,
        "scales_b_mxfp8": scales_b_mxfp8,
        "problem_sizes": problem_sizes,
        "expert_offsets": expert_offsets,
        "workspace": workspace,
        "empty_ptrs": empty_ptrs,
        "zero_stride": zero_stride,
        "zero_layout": zero_layout,
    }


def run_case(ext, args, case: str, m: int, n: int, k: int):
    x = make_inputs(args.groups, m, n, k)
    if args.only_bf16:
        woq_bf16 = ext.benchmark_nvfp4_bf16(
            x["output"],
            x["a_bf16"],
            x["b_nvfp4"],
            x["sfmx_b"],
            x["c_b"],
            x["problem_sizes"],
            x["expert_offsets"],
            x["workspace"],
            args.warmup,
            args.iterations,
        )
        flops = 2.0 * args.groups * m * n * k
        tflops = flops / (woq_bf16["cute_ms"] * 1e-3) / 1e12
        print(f"case={case} shape=E{args.groups} M{m} N{n} K{k}")
        print("path             cute_ms  full_device_ms  cute_TFLOP/s")
        print(
            f"{'nvfp4_bf16':13s}  {woq_bf16['cute_ms']:8.4f}  "
            f"{woq_bf16['device_span_ms']:14.4f}  {tflops:12.3f}"
        )
        return

    mx = ext.benchmark_mxfp8(
        x["output"],
        x["a_mxfp8"],
        x["b_mxfp8"],
        x["scales_a"],
        x["scales_b_mxfp8"],
        x["empty_ptrs"],
        x["zero_stride"],
        x["zero_layout"],
        x["problem_sizes"],
        x["expert_offsets"],
        x["workspace"],
        args.warmup,
        args.iterations,
    )
    woq_mxfp8 = ext.benchmark_nvfp4(
        x["output"],
        x["a_mxfp8"],
        x["b_nvfp4"],
        x["scales_a"],
        x["sfmx_b"],
        x["c_b"],
        x["problem_sizes"],
        x["expert_offsets"],
        x["workspace"],
        args.warmup,
        args.iterations,
    )
    woq_bf16 = ext.benchmark_nvfp4_bf16(
        x["output"],
        x["a_bf16"],
        x["b_nvfp4"],
        x["sfmx_b"],
        x["c_b"],
        x["problem_sizes"],
        x["expert_offsets"],
        x["workspace"],
        args.warmup,
        args.iterations,
    )

    flops = 2.0 * args.groups * m * n * k
    print(f"case={case} shape=E{args.groups} M{m} N{n} K{k}")
    print("path             cute_ms  full_device_ms  cute_TFLOP/s")
    results = (
        ("mxfp8", mx),
        ("nvfp4_mxfp8", woq_mxfp8),
        ("nvfp4_bf16", woq_bf16),
    )
    for name, result in results:
        tflops = flops / (result["cute_ms"] * 1e-3) / 1e12
        print(
            f"{name:13s}  {result['cute_ms']:8.4f}  "
            f"{result['device_span_ms']:14.4f}  {tflops:12.3f}"
        )
    print(
        "bf16/mxfp8 WoQ GEMM latency: "
        f"{woq_bf16['cute_ms'] / woq_mxfp8['cute_ms']:.3f}x"
    )
    print(
        "bf16/mxfp8 WoQ full device span: "
        f"{woq_bf16['device_span_ms'] / woq_mxfp8['device_span_ms']:.3f}x"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("decode", "prefill", "both"), default="both")
    parser.add_argument("--groups", type=int, default=2)
    parser.add_argument("--decode-m", type=int, default=1, help="decode M per group")
    parser.add_argument("--prefill-m", type=int, default=64, help="prefill M per group")
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--shapes",
        help="comma-separated M:N:K tuples; overrides the individual shape options",
    )
    parser.add_argument(
        "--only-bf16",
        action="store_true",
        help="skip the unchanged MXFP8 reference paths",
    )
    args = parser.parse_args()

    if not torch.xpu.is_available():
        raise RuntimeError("XPU is unavailable")
    ext = load_extension()
    print(f"device={torch.xpu.get_device_name()}")
    cases = []
    if args.shapes:
        for value in args.shapes.split(","):
            try:
                m, n, k = (int(dim) for dim in value.split(":"))
            except ValueError as exc:
                raise ValueError(f"invalid shape {value!r}; expected M:N:K") from exc
            cases.append(("sweep", m, n, k))
    else:
        if args.case in ("decode", "both"):
            cases.append(("decode", args.decode_m, args.n, args.k))
        if args.case in ("prefill", "both"):
            cases.append(("prefill", args.prefill_m, args.n, args.k))
    for index, (case, m, n, k) in enumerate(cases):
        if index:
            print()
        run_case(ext, args, case, m, n, k)


if __name__ == "__main__":
    main()
