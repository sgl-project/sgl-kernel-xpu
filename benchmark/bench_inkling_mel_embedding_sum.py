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
        if (candidate / "inkling_mel_embedding_ops.abi3.so").is_file():
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
    local_ext = root / "build" / "src" / "inkling_mel_embedding_ops.abi3.so"
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


def _choose_channels_per_item(
    tokens: int,
    n_mel_bins: int,
    mel_vocab_size: int,
    hidden: int,
    channels_per_item: int,
) -> int:
    if channels_per_item != 0:
        return channels_per_item
    if hidden >= 1536 and n_mel_bins >= 64:
        return 8
    if hidden >= 4096 and (tokens >= 8192 or mel_vocab_size >= 128):
        return 4
    return 2 if hidden >= 2048 else 1


def _estimated_bytes(
    *,
    tokens: int,
    n_mel_bins: int,
    hidden: int,
    channels_per_item: int,
    elem_size: int,
) -> int:
    channel_tiles = (hidden + 256 * channels_per_item - 1) // (256 * channels_per_item)
    weight_bytes = tokens * hidden * n_mel_bins * elem_size
    output_bytes = tokens * hidden * elem_size
    feature_bytes = tokens * n_mel_bins * 4 * channel_tiles
    return weight_bytes + output_bytes + feature_bytes


def _make_inputs(args, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(20260724)
    features = torch.randint(
        0,
        args.mel_vocab_size,
        (args.tokens, args.n_mel_bins),
        dtype=torch.int32,
        device="xpu",
    )
    if args.tokens > 0:
        features[0] = torch.arange(args.n_mel_bins, dtype=torch.int32, device="xpu") % args.mel_vocab_size
        features[-1] = args.mel_vocab_size - 1 - (
            torch.arange(args.n_mel_bins, dtype=torch.int32, device="xpu") % args.mel_vocab_size
        )
    weight = torch.randn(
        (args.n_mel_bins * args.mel_vocab_size, args.hidden),
        dtype=dtype,
        device="xpu",
    )
    return features.contiguous(), weight.contiguous()


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
        description="Benchmark Inkling mel_embedding_sum XPU op with device timing."
    )
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--n-mel-bins", type=int, default=80)
    parser.add_argument("--mel-vocab-size", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=6144)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--channels-per-item", type=int, default=0)
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
    if args.tokens < 1 or args.n_mel_bins < 1 or args.mel_vocab_size < 1 or args.hidden < 1:
        raise ValueError("tokens, n_mel_bins, mel_vocab_size, and hidden must be positive")
    if args.chunk_size < 1 or args.warmup < 0 or args.iters < 1:
        raise ValueError("chunk_size and iters must be positive; warmup must be non-negative")
    if args.channels_per_item not in (0, 1, 2, 4, 8):
        raise ValueError("channels_per_item must be 0, 1, 2, 4, or 8")

    _default_local_paths(args)
    _install_local_package(args.package_root, args.build_root)
    _load_libraries(args.load_library)

    dtype = _dtype(args.dtype)
    features, weight = _make_inputs(args, dtype)
    cpi = _choose_channels_per_item(
        args.tokens,
        args.n_mel_bins,
        args.mel_vocab_size,
        args.hidden,
        args.channels_per_item,
    )
    bytes_moved = _estimated_bytes(
        tokens=args.tokens,
        n_mel_bins=args.n_mel_bins,
        hidden=args.hidden,
        channels_per_item=cpi,
        elem_size=weight.element_size(),
    )

    def run():
        return torch.ops.sgl_kernel.inkling_mel_embedding_sum(
            features,
            weight,
            args.chunk_size,
            args.channels_per_item,
        )

    median_ms, avg_ms = _bench(run, warmup=args.warmup, iters=args.iters)
    gbps = _gbps(bytes_moved, median_ms)
    print(
        "Inkling mel_embedding_sum benchmark: "
        f"dtype={args.dtype} tokens={args.tokens} n_mel_bins={args.n_mel_bins} "
        f"mel_vocab_size={args.mel_vocab_size} hidden={args.hidden} "
        f"chunk_size={args.chunk_size} channels_per_item={cpi} "
        f"warmup={args.warmup} iters={args.iters}"
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
