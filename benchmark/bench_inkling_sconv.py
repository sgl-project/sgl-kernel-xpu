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
    if not (pkg_dir / "inkling_sconv.py").is_file():
        raise FileNotFoundError(f"could not find inkling_sconv.py under {pkg_dir}")
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


def _gbps(bytes_moved: int, ms: float) -> float:
    if ms <= 0.0:
        return 0.0
    return bytes_moved / (ms * 1.0e-3) / 1.0e9


def _make_forward_case(T: int, B: int, D: int, W: int, dtype: torch.dtype):
    torch.manual_seed(123)
    tokens_per_seq = max(1, T // B)
    T = tokens_per_seq * B
    x = torch.randn((T, D), dtype=dtype, device="xpu")
    weight = torch.randn((D, W), dtype=dtype, device="xpu") * 0.1
    cache = torch.randn((B, W - 1, D), dtype=dtype, device="xpu") * 0.1
    cache_mask = torch.ones((B, 1, 1), dtype=torch.bool, device="xpu")
    safe_idx = torch.arange(B, dtype=torch.int64, device="xpu")
    cu = torch.arange(0, T + 1, tokens_per_seq, dtype=torch.int64, device="xpu")
    si = torch.arange(B, dtype=torch.int32, device="xpu").repeat_interleave(tokens_per_seq)
    return x, weight, cache, cache_mask, safe_idx, cu, si


def _make_decode_case(T: int, D: int, W: int, dtype: torch.dtype):
    torch.manual_seed(124)
    x = torch.randn((T, D), dtype=dtype, device="xpu")
    weight = torch.randn((D, W), dtype=dtype, device="xpu") * 0.1
    cache = torch.randn((T, W - 1, D), dtype=dtype, device="xpu") * 0.1
    cache_indices = torch.arange(T, dtype=torch.int32, device="xpu")
    cache_mask = torch.ones((T,), dtype=torch.bool, device="xpu")
    return x, weight, cache, cache_indices, cache_mask


def _make_update_case(T: int, B: int, D: int, W: int, dtype: torch.dtype):
    torch.manual_seed(125)
    tokens_per_seq = max(1, T // B)
    T = tokens_per_seq * B
    x = torch.randn((T, D), dtype=dtype, device="xpu")
    cache = torch.randn((B, W - 1, D), dtype=dtype, device="xpu") * 0.1
    cache_indices = torch.arange(B, dtype=torch.int32, device="xpu")
    has_initial_state = torch.ones((B,), dtype=torch.bool, device="xpu")
    query_start_loc = torch.arange(0, T + 1, tokens_per_seq, dtype=torch.int32, device="xpu")
    return x, cache, cache_indices, has_initial_state, query_start_loc


def _make_gather_scatter_case(B: int, D: int, W: int, dtype: torch.dtype):
    torch.manual_seed(126)
    W1 = W - 1
    extra_slots = max(16, B // 16)
    hidden = torch.randn((B * W1, D), dtype=dtype, device="xpu")
    cache = torch.randn((B + extra_slots, W1, D), dtype=dtype, device="xpu") * 0.1
    track_idx = torch.arange(B * W1, dtype=torch.int32, device="xpu").reshape(B, W1)
    mask = torch.ones((B,), dtype=torch.bool, device="xpu")
    dst = torch.arange(extra_slots, extra_slots + B, dtype=torch.int64, device="xpu")
    return hidden, cache, track_idx, mask, dst


def _make_draft_extend_case(B: int, D: int, W: int, draft_token_num: int, dtype: torch.dtype):
    torch.manual_seed(127)
    W1 = W - 1
    hidden = torch.randn((B, draft_token_num, D), dtype=dtype, device="xpu")
    cache = torch.randn((B, W1, D), dtype=dtype, device="xpu") * 0.1
    cache_indices = torch.arange(B, dtype=torch.int32, device="xpu")
    num_accepted = torch.full((B,), draft_token_num, dtype=torch.int32, device="xpu")
    return hidden, cache, cache_indices, num_accepted


def _make_windows_case(B: int, D: int, W: int, draft_token_num: int, dtype: torch.dtype):
    torch.manual_seed(128)
    W1 = W - 1
    hidden = torch.randn((B, draft_token_num, D), dtype=dtype, device="xpu")
    cache = torch.randn((B, W1, D), dtype=dtype, device="xpu") * 0.1
    cache_indices = torch.arange(B, dtype=torch.int32, device="xpu")
    out = torch.empty((B, draft_token_num, W1, D), dtype=dtype, device="xpu")
    return cache, hidden, cache_indices, out


def _make_decode_metadata_case(B: int):
    torch.manual_seed(129)
    cache_indices = torch.randint(0, 65536, (B,), dtype=torch.int32, device="xpu")
    cache_indices[torch.rand(B, device="xpu") < 0.05] = -1
    return cache_indices


def _make_extend_metadata_case(B: int, tokens_per_seq: int):
    torch.manual_seed(130)
    lens = torch.full((B,), tokens_per_seq, dtype=torch.int32, device="xpu")
    lens[torch.rand(B, device="xpu") < 0.05] = 0
    cache_indices = torch.randint(0, 65536, (B,), dtype=torch.int32, device="xpu")
    cache_indices[torch.rand(B, device="xpu") < 0.05] = -1
    return cache_indices, lens


def _print_result(name: str, median_ms: float, avg_ms: float, bytes_moved: int) -> None:
    print(
        f"{name:24s} median={median_ms:8.4f} ms  avg={avg_ms:8.4f} ms  "
        f"effective={_gbps(bytes_moved, median_ms):8.2f} GB/s"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Inkling SConv XPU ops with device timing.")
    parser.add_argument(
        "--op",
        choices=[
            "all",
            "forward",
            "fused-decode",
            "update-cache",
            "gather-scatter",
            "draft-extend",
            "windows",
            "decode-metadata",
            "extend-metadata",
        ],
        default="all",
    )
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--T", type=int, default=8192)
    parser.add_argument("--B", type=int, default=64)
    parser.add_argument("--D", type=int, default=256)
    parser.add_argument("--W", type=int, default=4)
    parser.add_argument("--draft-token-num", type=int, default=9)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--package-root",
        help="Optional local python package root, for targeted runs before common_ops is fully built.",
    )
    parser.add_argument(
        "--build-root",
        help="Optional local build library directory containing inkling_sconv_ops.abi3.so.",
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
    if args.W < 2:
        raise ValueError("--W must be >= 2")
    if args.draft_token_num < 0:
        raise ValueError("--draft-token-num must be non-negative")

    _install_local_package(args.package_root, args.build_root)
    _load_libraries(args.load_library)

    from sgl_kernel.inkling_sconv import (
        causal_conv1d,
        fused_causal_conv1d_update_decode,
        fused_draft_extend_sconv_cache,
        fused_decode_sconv_metadata,
        fused_extend_sconv_metadata,
        fused_gather_scatter_to_sconv_cache,
        save_intermediate_conv_windows,
        update_sconv_cache,
    )

    dtype = _dtype(args.dtype)
    elem_size = torch.empty((), dtype=dtype).element_size()
    print(
        f"Inkling SConv benchmark: dtype={args.dtype} T={args.T} B={args.B} D={args.D} W={args.W} "
        f"warmup={args.warmup} iters={args.iters}"
    )

    if args.op in ("all", "forward"):
        x, weight, cache, cache_mask, safe_idx, cu, si = _make_forward_case(
            args.T, args.B, args.D, args.W, dtype
        )

        def run_forward():
            causal_conv1d(
                x,
                weight,
                cache,
                cache_mask,
                safe_idx,
                cu,
                si,
                activation="silu",
                use_residual=True,
            )

        median_ms, avg_ms = _bench(run_forward, warmup=args.warmup, iters=args.iters)
        bytes_moved = x.numel() * elem_size * (2 * args.W + 2)
        _print_result("causal_conv1d", median_ms, avg_ms, bytes_moved)

    if args.op in ("all", "fused-decode"):
        x, weight, cache, cache_indices, cache_mask = _make_decode_case(
            args.T, args.D, args.W, dtype
        )

        def run_fused_decode():
            fused_causal_conv1d_update_decode(
                x,
                weight,
                cache,
                cache_indices,
                cache_mask,
                activation="silu",
                use_residual=True,
            )

        median_ms, avg_ms = _bench(run_fused_decode, warmup=args.warmup, iters=args.iters)
        bytes_moved = x.numel() * elem_size * (2 * args.W + 2)
        _print_result("fused_decode_update", median_ms, avg_ms, bytes_moved)

    if args.op in ("all", "update-cache"):
        x, cache, cache_indices, has_initial_state, query_start_loc = _make_update_case(
            args.T, args.B, args.D, args.W, dtype
        )

        def run_update():
            update_sconv_cache(x, cache, cache_indices, has_initial_state, query_start_loc)

        median_ms, avg_ms = _bench(run_update, warmup=args.warmup, iters=args.iters)
        bytes_moved = cache.numel() * elem_size * 2
        _print_result("update_sconv_cache", median_ms, avg_ms, bytes_moved)

    if args.op in ("all", "gather-scatter"):
        hidden, cache, track_idx, mask, dst = _make_gather_scatter_case(
            args.B, args.D, args.W, dtype
        )

        def run_gather_scatter():
            fused_gather_scatter_to_sconv_cache(hidden, cache, track_idx, mask, dst)

        median_ms, avg_ms = _bench(run_gather_scatter, warmup=args.warmup, iters=args.iters)
        bytes_moved = hidden.numel() * elem_size * 2
        _print_result("gather_scatter", median_ms, avg_ms, bytes_moved)

    if args.op in ("all", "draft-extend"):
        hidden, cache, cache_indices, num_accepted = _make_draft_extend_case(
            args.B, args.D, args.W, args.draft_token_num, dtype
        )

        def run_draft_extend():
            fused_draft_extend_sconv_cache(
                hidden,
                cache,
                cache_indices,
                num_accepted_tokens=num_accepted,
                draft_token_num=args.draft_token_num,
                do_tracking=False,
            )

        median_ms, avg_ms = _bench(run_draft_extend, warmup=args.warmup, iters=args.iters)
        bytes_moved = args.B * (args.W - 1) * args.D * elem_size * 2
        _print_result("draft_extend", median_ms, avg_ms, bytes_moved)

    if args.op in ("all", "windows"):
        cache, hidden, cache_indices, out = _make_windows_case(
            args.B, args.D, args.W, args.draft_token_num, dtype
        )

        def run_windows():
            save_intermediate_conv_windows(
                cache,
                hidden,
                cache_indices,
                out,
                batch_size=args.B,
                draft_token_num=args.draft_token_num,
            )

        median_ms, avg_ms = _bench(run_windows, warmup=args.warmup, iters=args.iters)
        bytes_moved = out.numel() * elem_size * 2
        _print_result("windows", median_ms, avg_ms, bytes_moved)

    if args.op in ("all", "decode-metadata"):
        cache_indices = _make_decode_metadata_case(args.B)

        def run_decode_metadata():
            fused_decode_sconv_metadata(B=args.B, cache_indices=cache_indices)

        median_ms, avg_ms = _bench(run_decode_metadata, warmup=args.warmup, iters=args.iters)
        bytes_moved = args.B * (
            cache_indices.element_size()
            + torch.empty((), dtype=torch.bool).element_size() * 2
            + torch.empty((), dtype=torch.int64).element_size() * 2
            + torch.empty((), dtype=torch.int32).element_size() * 2
        )
        _print_result("decode_metadata", median_ms, avg_ms, bytes_moved)

    if args.op in ("all", "extend-metadata"):
        tokens_per_seq = max(1, args.T // args.B)
        T = tokens_per_seq * args.B
        cache_indices, lens = _make_extend_metadata_case(args.B, tokens_per_seq)

        def run_extend_metadata():
            fused_extend_sconv_metadata(
                B=args.B,
                T=T,
                cache_indices=cache_indices,
                his_mode=0,
                extend_seq_lens=lens,
            )

        median_ms, avg_ms = _bench(run_extend_metadata, warmup=args.warmup, iters=args.iters)
        bytes_moved = args.B * (
            cache_indices.element_size()
            + lens.element_size()
            + torch.empty((), dtype=torch.bool).element_size() * 2
            + torch.empty((), dtype=torch.int64).element_size() * 2
            + torch.empty((), dtype=torch.int32).element_size() * 2
        ) + T * torch.empty((), dtype=torch.int32).element_size()
        _print_result("extend_metadata", median_ms, avg_ms, bytes_moved)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
