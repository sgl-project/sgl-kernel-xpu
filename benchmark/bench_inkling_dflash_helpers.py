import argparse
import statistics
import sys
import types
from pathlib import Path
from typing import Optional

import torch


def _install_local_package(package_root: Optional[str], build_root: Optional[str]) -> None:
    if package_root is None:
        return
    root = Path(package_root).resolve()
    pkg_dir = root if root.name == "sgl_kernel" else root / "sgl_kernel"
    if not pkg_dir.is_dir():
        raise FileNotFoundError(f"could not find sgl_kernel package under {pkg_dir}")

    extension_dirs = []
    candidates = []
    if build_root is not None:
        candidates.append(Path(build_root).resolve())
    if pkg_dir.parent.name == "python":
        candidates.append(pkg_dir.parent.parent / "build/src")
    candidates.append(Path.cwd() / "build/src")
    for candidate in candidates:
        if (candidate / "inkling_dflash_helpers_ops.abi3.so").is_file():
            extension_dirs.append(str(candidate))

    pkg = types.ModuleType("sgl_kernel")
    pkg.__path__ = [str(pkg_dir), *extension_dirs]
    sys.modules["sgl_kernel"] = pkg


def _load_libraries(paths: list[str]) -> None:
    for path in paths:
        torch.ops.load_library(str(Path(path).resolve()))


def _default_local_paths(args) -> None:
    root = Path.cwd()
    local_pkg = root / "python" / "sgl_kernel"
    local_ext = root / "build" / "src" / "inkling_dflash_helpers_ops.abi3.so"
    if args.package_root is None and local_pkg.is_dir():
        args.package_root = str(root / "python")
    if args.build_root is None and local_ext.is_file():
        args.build_root = str(root / "build" / "src")
    if not args.load_library and local_ext.is_file():
        args.load_library.append(str(local_ext))


def _dtype(name: str) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def _bench(fn, *, warmup: int, iters: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()

    times = []
    for _ in range(iters):
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.xpu.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times), sum(times) / len(times)


def _gbps(bytes_moved: int, ms: float) -> float:
    if ms <= 0.0:
        return 0.0
    return bytes_moved / (ms * 1.0e-3) / 1.0e9


def _make_cache_inputs(args):
    batch = args.batch
    draft_tokens = args.draft_tokens
    req_rows = args.req_rows
    table_width = args.table_width
    total = batch * draft_tokens

    req_to_token = torch.arange(req_rows * table_width, dtype=torch.int64, device="xpu").reshape(
        req_rows, table_width
    )
    req_pool_indices = (torch.arange(batch, dtype=torch.int64, device="xpu") * 7 + 2) % req_rows
    pos2d = (
        torch.arange(total, dtype=torch.int64, device="xpu").reshape(batch, draft_tokens) * 17
        + 1
    ) % table_width
    mask = torch.ones((batch, draft_tokens), dtype=torch.uint8, device="xpu")
    out_offsets = torch.arange(total, dtype=torch.int32, device="xpu").reshape(batch, draft_tokens)
    logits = torch.randn((args.tokens, args.vocab), dtype=torch.float32, device="xpu")
    return req_to_token, req_pool_indices, pos2d, mask, out_offsets, total, logits


def _run_cache_path(args) -> tuple[float, float, int]:
    inputs = _make_cache_inputs(args)

    def run():
        return torch.ops.sgl_kernel.inkling_dflash_cache_path(*inputs)

    median_ms, avg_ms = _bench(run, warmup=args.warmup, iters=args.iters)
    total = args.batch * args.draft_tokens
    gather_bytes = total * (8 + 8 + 8 + 4 + 1)
    argmax_bytes = args.tokens * args.vocab * 4 + args.tokens * 8
    return median_ms, avg_ms, gather_bytes + argmax_bytes


def _make_scatter_inputs(args, dtype: torch.dtype):
    slots = args.slots
    t_max = args.t_max
    d_ssm = args.d_ssm
    conv_a_shape = (slots, args.width_a, args.d_conv)
    conv_b_shape = (slots, args.width_b, args.d_conv)
    ssm = torch.empty((slots, d_ssm), dtype=dtype, device="xpu")
    ssm_inter = torch.empty((slots, t_max, d_ssm), dtype=dtype, device="xpu")
    conv_a = torch.empty(conv_a_shape, dtype=dtype, device="xpu")
    conv_a_inter = torch.empty((slots, t_max, args.width_a, args.d_conv), dtype=dtype, device="xpu")
    conv_b = torch.empty(conv_b_shape, dtype=dtype, device="xpu")
    conv_b_inter = torch.empty((slots, t_max, args.width_b, args.d_conv), dtype=dtype, device="xpu")
    main_slots = torch.arange(args.main_count, dtype=torch.int64, device="xpu") % slots
    main_steps = torch.arange(args.main_count, dtype=torch.int64, device="xpu") % t_max
    track_slots = (torch.arange(args.track_count, dtype=torch.int64, device="xpu") * 3 + 1) % slots
    track_steps = (torch.arange(args.track_count, dtype=torch.int64, device="xpu") * 5) % t_max
    return (
        ssm,
        ssm_inter,
        conv_a,
        conv_a_inter,
        conv_b,
        conv_b_inter,
        main_slots,
        main_steps,
        track_slots,
        track_steps,
        t_max,
    )


def _run_scatter(args) -> tuple[float, float, int]:
    dtype = _dtype(args.dtype)
    inputs = _make_scatter_inputs(args, dtype)

    def run():
        torch.ops.sgl_kernel.inkling_scatter_mamba_states_after_mtp_verify(*inputs)

    median_ms, avg_ms = _bench(run, warmup=args.warmup, iters=args.iters)
    active = args.main_count + args.track_count
    row_elems = args.d_ssm + args.width_a * args.d_conv + args.width_b * args.d_conv
    bytes_moved = active * row_elems * torch.empty((), dtype=dtype).element_size() * 2
    return median_ms, avg_ms, bytes_moved


def _make_device_names(count: int, stride: int) -> torch.Tensor:
    labels = [b"cuda:0", b"xpu", b"level_zero:gpu", b"cpu"]
    names = torch.zeros((count, stride), dtype=torch.uint8)
    for i in range(count):
        label = labels[i & 3]
        names[i, : len(label)] = torch.tensor(list(label), dtype=torch.uint8)
    return names.to("xpu")


def _run_device_guard(args) -> tuple[float, float, int]:
    names = _make_device_names(args.guard_count, args.name_stride)

    def run():
        return torch.ops.sgl_kernel.inkling_dflash_device_guard(names)

    median_ms, avg_ms = _bench(run, warmup=args.warmup, iters=args.iters)
    bytes_moved = args.guard_count * (10 + 2)
    return median_ms, avg_ms, bytes_moved


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Inkling DFLASH helper XPU ops with device timing."
    )
    parser.add_argument("--mode", choices=["scatter", "cache-path", "device-guard"], default="scatter")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--target-gbps", type=float, default=350.0)

    parser.add_argument("--batch", type=int, default=16384)
    parser.add_argument("--draft-tokens", type=int, default=9)
    parser.add_argument("--req-rows", type=int, default=8192)
    parser.add_argument("--table-width", type=int, default=2048)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--vocab", type=int, default=8192)

    parser.add_argument("--slots", type=int, default=4096)
    parser.add_argument("--t-max", type=int, default=9)
    parser.add_argument("--d-ssm", type=int, default=1)
    parser.add_argument("--width-a", type=int, default=3)
    parser.add_argument("--width-b", type=int, default=3)
    parser.add_argument("--d-conv", type=int, default=1536)
    parser.add_argument("--main-count", type=int, default=2048)
    parser.add_argument("--track-count", type=int, default=512)
    parser.add_argument("--guard-count", type=int, default=1048576)
    parser.add_argument("--name-stride", type=int, default=32)

    parser.add_argument(
        "--package-root",
        help="Optional local python package root. Defaults to ./python when present.",
    )
    parser.add_argument(
        "--build-root",
        help="Optional local build library directory. Defaults to ./build/src when present.",
    )
    parser.add_argument(
        "--load-library",
        action="append",
        default=[],
        help="Shared library to load before benchmarking; may be repeated.",
    )
    args = parser.parse_args()

    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise RuntimeError("XPU device is required")
    if args.warmup < 0 or args.iters < 1:
        raise ValueError("warmup must be non-negative and iters must be positive")

    _default_local_paths(args)
    _install_local_package(args.package_root, args.build_root)
    _load_libraries(args.load_library)

    if args.mode == "cache-path":
        median_ms, avg_ms, bytes_moved = _run_cache_path(args)
        label = (
            f"cache-path batch={args.batch} draft_tokens={args.draft_tokens} "
            f"tokens={args.tokens} vocab={args.vocab}"
        )
    elif args.mode == "device-guard":
        median_ms, avg_ms, bytes_moved = _run_device_guard(args)
        label = f"device-guard count={args.guard_count} stride={args.name_stride}"
    else:
        median_ms, avg_ms, bytes_moved = _run_scatter(args)
        label = (
            f"scatter dtype={args.dtype} slots={args.slots} t_max={args.t_max} "
            f"rows=({args.d_ssm},{args.width_a * args.d_conv},{args.width_b * args.d_conv}) "
            f"active={args.main_count + args.track_count}"
        )

    gbps = _gbps(bytes_moved, median_ms)
    print(f"Inkling DFLASH {label} warmup={args.warmup} iters={args.iters}")
    print(
        f"median={median_ms:.4f} ms  avg={avg_ms:.4f} ms  "
        f"estimated={bytes_moved / 1.0e9:.3f} GB  effective={gbps:.2f} GB/s"
    )
    if args.target_gbps > 0 and gbps < args.target_gbps:
        raise RuntimeError(f"effective bandwidth {gbps:.2f} GB/s below target {args.target_gbps:.2f} GB/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
