import argparse
import statistics
import sys
import types
from pathlib import Path
from typing import Optional

import torch


_ALL_REDUCE_VARIANTS = ("direct", "two_shot", "full_oneshot", "push_oneshot")


def _install_local_package(
    package_root: Optional[str], build_root: Optional[str]
) -> None:
    if package_root is None:
        return
    root = Path(package_root).resolve()
    pkg_dir = root if root.name == "sgl_kernel" else root / "sgl_kernel"
    if not (pkg_dir / "inkling_comm_ar_sconv.py").is_file():
        raise FileNotFoundError(
            f"could not find inkling_comm_ar_sconv.py under {pkg_dir}"
        )

    extension_dirs = []
    candidates = []
    if build_root is not None:
        candidates.append(Path(build_root).resolve())
    if pkg_dir.parent.name == "python":
        candidates.append(pkg_dir.parent.parent / "build/src")
    candidates.append(Path.cwd() / "build/src")
    for candidate in candidates:
        if (candidate / "inkling_sconv_ops.abi3.so").is_file():
            extension_dirs.append(str(candidate))

    pkg = types.ModuleType("sgl_kernel")
    pkg.__path__ = [str(pkg_dir), *extension_dirs]
    sys.modules["sgl_kernel"] = pkg
    sys.modules["sgl_kernel.common_ops"] = types.ModuleType("sgl_kernel.common_ops")


def _load_libraries(paths: list[str]) -> None:
    for path in paths:
        torch.ops.load_library(str(Path(path).resolve()))


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


def _gbps(bytes_moved: float, ms: float) -> float:
    if ms <= 0.0:
        return 0.0
    return bytes_moved / (ms * 1.0e-3) / 1.0e9


def _print_result(
    name: str, median_ms: float, avg_ms: float, bytes_moved: float
) -> None:
    print(
        f"{name:30s} median={median_ms:8.4f} ms  avg={avg_ms:8.4f} ms  "
        f"effective={_gbps(bytes_moved, median_ms):8.2f} GB/s"
    )


def _all_reduce_bytes(
    variant: str, *, world: int, n: int, elem_size: int, use_shared: bool
) -> float:
    w = float(world)
    shared = 1.0 if use_shared else 0.0
    elems = float(n)
    if variant == "two_shot":
        return elem_size * (elems * w * (1.0 + shared) + elems + w * elems + w * elems)
    if variant == "push_oneshot":
        return elem_size * (
            w * elems * (1.0 + shared) + w * elems + w * w * elems + w * elems
        )
    return elem_size * (w * elems * (w * (1.0 + shared) + 1.0))


def _fused_decode_bytes(
    *,
    world: int,
    T: int,
    D: int,
    W: int,
    elem_size: int,
    use_shared: bool,
) -> float:
    td = float(T * D)
    shared = 1.0 if use_shared else 0.0
    partial_reads = td * world * (1.0 + shared) * elem_size
    cache_tap_reads = td * max(W - 1, 0) * elem_size
    weight_reads = td * W * elem_size
    residual_reads = td * elem_size
    norm_reads = td * 2.0 * elem_size
    output_writes = td * 2.0 * elem_size
    cache_update = td * (2.0 * max(W - 2, 0) + (1.0 if W > 1 else 0.0)) * elem_size
    return (
        partial_reads
        + cache_tap_reads
        + weight_reads
        + residual_reads
        + norm_reads
        + output_writes
        + cache_update
    )


def _scattered_sconv_bytes(
    *,
    world: int,
    T: int,
    B: int,
    D: int,
    W: int,
    elem_size: int,
) -> float:
    td = float(T * D)
    partial_reads = td * world * elem_size
    scratch = td * (W + 1.0) * elem_size
    weight_reads = td * W * elem_size
    cache_prefix_reads = float(B * max(W - 1, 0) * D) * elem_size
    gather_writes = td * world * elem_size
    cache_update = float(B * max(W - 1, 0) * D) * 2.0 * elem_size
    return (
        partial_reads
        + scratch
        + weight_reads
        + cache_prefix_reads
        + gather_writes
        + cache_update
    )


def _rank_inputs(
    *, world: int, T: int, D: int, dtype: torch.dtype, use_shared: bool
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    partials = torch.randn((world, T, D), dtype=dtype, device="xpu").contiguous()
    shared = None
    if use_shared:
        shared = torch.randn((world, T, D), dtype=dtype, device="xpu").contiguous()
    return partials, shared


def _fused_decode_case(args, dtype: torch.dtype):
    partials, shared = _rank_inputs(
        world=args.world, T=args.T, D=args.D, dtype=dtype, use_shared=args.shared
    )
    slots = max(args.T + 16, 32)
    residual = torch.randn((args.T, args.D), dtype=dtype, device="xpu").contiguous()
    cache = torch.randn((slots, args.W - 1, args.D), dtype=dtype, device="xpu")
    cache_indices = torch.arange(args.T, dtype=torch.int32, device="xpu") % slots
    cache_mask = torch.ones((args.T,), dtype=torch.bool, device="xpu")
    weight = torch.randn((args.D, args.W), dtype=dtype, device="xpu")
    norm_weight = torch.randn((args.D,), dtype=dtype, device="xpu") + 1
    return (
        partials,
        shared,
        residual,
        cache,
        cache_indices,
        cache_mask,
        weight,
        norm_weight,
    )


def _scattered_sconv_case(args, dtype: torch.dtype):
    T = args.B * args.tokens_per_seq
    partials, shared = _rank_inputs(
        world=args.world, T=T, D=args.D, dtype=dtype, use_shared=args.shared
    )
    slots = args.B + 16
    cache = torch.randn((slots, args.W - 1, args.D), dtype=dtype, device="xpu")
    cache_indices = torch.arange(args.B, dtype=torch.int32, device="xpu")
    cache_mask = torch.ones((args.B,), dtype=torch.bool, device="xpu")
    cu = torch.arange(
        0, T + 1, args.tokens_per_seq, dtype=torch.int64, device="xpu"
    )
    si = torch.arange(args.B, dtype=torch.int32, device="xpu").repeat_interleave(
        args.tokens_per_seq
    )
    weight = torch.randn((args.D, args.W), dtype=dtype, device="xpu")
    has_initial_state = torch.ones((args.B,), dtype=torch.bool, device="xpu")
    return (
        partials,
        shared,
        cache,
        cache_indices,
        cache_mask,
        cu,
        si,
        weight,
        has_initial_state,
    )


def _default_local_paths(args) -> None:
    root = Path.cwd()
    local_pkg = root / "python" / "sgl_kernel" / "inkling_comm_ar_sconv.py"
    local_ext = root / "build" / "src" / "inkling_sconv_ops.abi3.so"
    if args.package_root is None and local_pkg.is_file():
        args.package_root = str(root / "python")
    if args.build_root is None and local_ext.is_file():
        args.build_root = str(root / "build" / "src")
    if not args.load_library and local_ext.is_file():
        args.load_library.append(str(local_ext))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Inkling comm/AR + SConv XPU ops with device timing."
    )
    parser.add_argument(
        "--op",
        choices=["all", "all-reduce", "fused-decode", "scattered-sconv"],
        default="all",
    )
    parser.add_argument(
        "--variant",
        choices=["all", *_ALL_REDUCE_VARIANTS],
        default="all",
        help="All-reduce variant to benchmark.",
    )
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--world", type=int, default=4)
    parser.add_argument("--T", type=int, default=144)
    parser.add_argument("--B", type=int, default=16)
    parser.add_argument("--tokens-per-seq", type=int, default=9)
    parser.add_argument("--D", type=int, default=1536)
    parser.add_argument("--W", type=int, default=4)
    parser.add_argument("--shared", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--package-root",
        help="Optional local python package root. Defaults to ./python when present.",
    )
    parser.add_argument(
        "--build-root",
        help=(
            "Optional local build library directory. Defaults to ./build/src when "
            "present."
        ),
    )
    parser.add_argument(
        "--load-library",
        action="append",
        default=[],
        help="Shared library to load before importing wrappers; may be repeated.",
    )
    args = parser.parse_args()

    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise RuntimeError("XPU device is required")
    if args.world < 1:
        raise ValueError("--world must be >= 1")
    if args.T < 1 or args.B < 1 or args.tokens_per_seq < 1:
        raise ValueError("--T, --B, and --tokens-per-seq must be >= 1")
    if args.D < 1 or args.W < 2:
        raise ValueError("--D must be >= 1 and --W must be >= 2")
    if args.warmup < 0 or args.iters < 1:
        raise ValueError("--warmup must be >= 0 and --iters must be >= 1")

    _default_local_paths(args)
    _install_local_package(args.package_root, args.build_root)
    _load_libraries(args.load_library)

    from sgl_kernel.inkling_comm_ar_sconv import (
        ar_fused_decode,
        ar_scattered_sconv,
        comm_all_reduce,
    )

    torch.manual_seed(20260723)
    dtype = _dtype(args.dtype)
    elem_size = torch.empty((), dtype=dtype).element_size()
    scattered_T = args.B * args.tokens_per_seq
    print(
        "Inkling comm/AR + SConv benchmark: "
        f"dtype={args.dtype} world={args.world} T={args.T} "
        f"B={args.B} tokens_per_seq={args.tokens_per_seq} scattered_T={scattered_T} "
        f"D={args.D} W={args.W} shared={args.shared} "
        f"warmup={args.warmup} iters={args.iters}"
    )

    if args.op in ("all", "all-reduce"):
        partials, shared = _rank_inputs(
            world=args.world, T=args.T, D=args.D, dtype=dtype, use_shared=args.shared
        )
        variants = _ALL_REDUCE_VARIANTS if args.variant == "all" else (args.variant,)
        for variant in variants:
            median_ms, avg_ms = _bench(
                lambda v=variant: comm_all_reduce(partials, shared=shared, variant=v),
                warmup=args.warmup,
                iters=args.iters,
            )
            bytes_moved = _all_reduce_bytes(
                variant,
                world=args.world,
                n=args.T * args.D,
                elem_size=elem_size,
                use_shared=args.shared,
            )
            _print_result(f"all_reduce:{variant}", median_ms, avg_ms, bytes_moved)

    if args.op in ("all", "fused-decode"):
        (
            partials,
            shared,
            residual,
            cache,
            cache_indices,
            cache_mask,
            weight,
            norm_weight,
        ) = _fused_decode_case(args, dtype)
        median_ms, avg_ms = _bench(
            lambda: ar_fused_decode(
                partials,
                residual,
                cache,
                cache_indices,
                cache_mask,
                weight,
                norm_weight,
                activation="silu",
                use_residual=True,
                shared=shared,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
        bytes_moved = _fused_decode_bytes(
            world=args.world,
            T=args.T,
            D=args.D,
            W=args.W,
            elem_size=elem_size,
            use_shared=args.shared,
        )
        _print_result("ar_fused_decode", median_ms, avg_ms, bytes_moved)

    if args.op in ("all", "scattered-sconv"):
        (
            partials,
            shared,
            cache,
            cache_indices,
            cache_mask,
            cu,
            si,
            weight,
            has_initial_state,
        ) = _scattered_sconv_case(args, dtype)
        median_ms, avg_ms = _bench(
            lambda: ar_scattered_sconv(
                partials,
                cache,
                cache_indices,
                cache_mask,
                cu,
                si,
                weight,
                has_initial_state,
                activation="silu",
                use_residual=True,
                shared=shared,
                update_cache=True,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
        bytes_moved = _scattered_sconv_bytes(
            world=args.world,
            T=scattered_T,
            B=args.B,
            D=args.D,
            W=args.W,
            elem_size=elem_size,
        )
        _print_result("ar_scattered_sconv", median_ms, avg_ms, bytes_moved)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
