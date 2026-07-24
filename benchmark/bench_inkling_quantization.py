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
        if (candidate / "inkling_quantization_ops.abi3.so").is_file():
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
    local_ext = root / "build" / "src" / "inkling_quantization_ops.abi3.so"
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
        description="Benchmark Inkling FP4 quantization helper XPU ops with device timing."
    )
    parser.add_argument("--mode", choices=["mxfp4", "nvfp4"], default="mxfp4")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--cols", type=int, default=6144)
    parser.add_argument("--column-major-scales", action="store_true")
    parser.add_argument("--global-scale", type=float, default=0.0)
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
    if args.rows < 1 or args.cols < 1 or args.warmup < 0 or args.iters < 1:
        raise ValueError("rows/cols/iters must be positive and warmup non-negative")

    _default_local_paths(args)
    _install_local_package(args.package_root, args.build_root)
    _load_libraries(args.load_library)

    dtype = _dtype(args.dtype)
    x = torch.randn((args.rows, args.cols), dtype=dtype, device="xpu")
    elem_size = x.element_size()

    if args.mode == "mxfp4":
        if args.cols % 32 != 0:
            raise ValueError("MXFP4 cols must be divisible by 32")

        def run():
            return torch.ops.sgl_kernel.inkling_mxfp4_mapping(
                x,
                args.column_major_scales,
                1.0e-10,
            )

        bytes_moved = args.rows * args.cols * elem_size + args.rows * (args.cols // 2) + args.rows * (args.cols // 32)
        label = (
            f"mxfp4 dtype={args.dtype} rows={args.rows} cols={args.cols} "
            f"column_major_scales={args.column_major_scales}"
        )
    else:
        if args.cols % 16 != 0:
            raise ValueError("NVFP4 cols must be divisible by 16")
        if args.global_scale > 0.0:
            global_scale = args.global_scale
        else:
            global_scale = 448.0 * 6.0 / float(x.float().abs().max().item())

        def run():
            return torch.ops.sgl_kernel.inkling_nvfp4_layout(x, global_scale)

        bytes_moved = args.rows * args.cols * elem_size + args.rows * (args.cols // 2) + args.rows * (args.cols // 16)
        label = f"nvfp4 dtype={args.dtype} rows={args.rows} cols={args.cols} global_scale={global_scale:.3f}"

    median_ms, avg_ms = _bench(run, warmup=args.warmup, iters=args.iters)
    gbps = _gbps(bytes_moved, median_ms)
    print(f"Inkling quantization {label} warmup={args.warmup} iters={args.iters}")
    print(
        f"median={median_ms:.4f} ms  avg={avg_ms:.4f} ms  "
        f"estimated={bytes_moved / 1.0e9:.3f} GB  effective={gbps:.2f} GB/s"
    )
    if args.target_gbps > 0 and gbps < args.target_gbps:
        raise RuntimeError(f"effective bandwidth {gbps:.2f} GB/s below target {args.target_gbps:.2f} GB/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
