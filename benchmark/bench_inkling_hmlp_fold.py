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
        if (candidate / "inkling_hmlp_fold_ops.abi3.so").is_file():
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
    local_ext = root / "build" / "src" / "inkling_hmlp_fold_ops.abi3.so"
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Inkling hMLP fold_timespace_to_depth XPU op with device timing."
    )
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--B", type=int, default=16384)
    parser.add_argument("--T", type=int, default=1)
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--W", type=int, default=8)
    parser.add_argument("--C", type=int, default=64)
    parser.add_argument("--t-fold", type=int, default=1)
    parser.add_argument("--hw-fold", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--target-gbps", type=float, default=350.0)
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
    if min(args.B, args.T, args.H, args.W, args.C, args.t_fold, args.hw_fold) < 1:
        raise ValueError("shape dimensions and fold factors must be positive")
    if args.T % args.t_fold != 0 or args.H % args.hw_fold != 0 or args.W % args.hw_fold != 0:
        raise ValueError("fold factors must divide T, H, and W")
    if args.warmup < 0 or args.iters < 1:
        raise ValueError("warmup must be non-negative and iters must be positive")

    _default_local_paths(args)
    _install_local_package(args.package_root, args.build_root)
    _load_libraries(args.load_library)

    dtype = _dtype(args.dtype)
    x = torch.randn((args.B, args.T, args.H, args.W, args.C), dtype=dtype, device="xpu")
    bytes_moved = 2 * x.numel() * x.element_size()

    def run():
        return torch.ops.sgl_kernel.inkling_hmlp_fold_timespace_to_depth(
            x,
            args.t_fold,
            args.hw_fold,
        )

    median_ms, avg_ms = _bench(run, warmup=args.warmup, iters=args.iters)
    gbps = _gbps(bytes_moved, median_ms)
    print(
        "Inkling hMLP fold benchmark: "
        f"dtype={args.dtype} shape=({args.B},{args.T},{args.H},{args.W},{args.C}) "
        f"t_fold={args.t_fold} hw_fold={args.hw_fold} warmup={args.warmup} iters={args.iters}"
    )
    print(
        f"median={median_ms:.4f} ms  avg={avg_ms:.4f} ms  "
        f"estimated={bytes_moved / 1.0e9:.3f} GB  effective={gbps:.2f} GB/s"
    )
    if args.target_gbps > 0 and gbps < args.target_gbps:
        raise RuntimeError(f"effective bandwidth {gbps:.2f} GB/s below target {args.target_gbps:.2f} GB/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
