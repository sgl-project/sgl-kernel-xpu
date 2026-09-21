import argparse
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
import triton
import triton.language as tl
from sgl_kernel import chunked_sgmv_lora_shrink_forward

all_results = []
MIN_CHUNK_SIZE = 16
TRITON_MAX_CHUNK_SIZE = 16

DEFAULT_CASES: List[Dict[str, int]] = [
    {
        "num_tokens": 256,
        "num_loras": 8,
        "max_rank": 16,
        "input_dim": 4096,
        "stack_num": 3,
    },
    {
        "num_tokens": 512,
        "num_loras": 8,
        "max_rank": 64,
        "input_dim": 4096,
        "stack_num": 1,
    },
    {
        "num_tokens": 1024,
        "num_loras": 8,
        "max_rank": 32,
        "input_dim": 4096,
        "stack_num": 2,
    },
    {
        "num_tokens": 2048,
        "num_loras": 16,
        "max_rank": 64,
        "input_dim": 4096,
        "stack_num": 1,
    },
    {
        "num_tokens": 4096,
        "num_loras": 8,
        "max_rank": 32,
        "input_dim": 4096,
        "stack_num": 1,
    },
    {
        "num_tokens": 8192,
        "num_loras": 16,
        "max_rank": 64,
        "input_dim": 4096,
        "stack_num": 1,
    },
]


def _derive_triton_chunk_size(num_tokens: int) -> int:
    """Triton segment length (logical space) for the given decode batch size.

    Mirrors ChunkedSgmvLoRABackend._determine_chunk_size_for_tokens: the chunk
    grows with the batch size, capped by TRITON_MAX_CHUNK_SIZE. This is the
    chunking Triton needs (one program per <= BLOCK_M-row segment)
    """
    if TRITON_MAX_CHUNK_SIZE <= MIN_CHUNK_SIZE:
        return MIN_CHUNK_SIZE
    if num_tokens >= 256:
        chunk_size = 128
    elif num_tokens >= 64:
        chunk_size = 32
    else:
        chunk_size = 16
    return min(TRITON_MAX_CHUNK_SIZE, chunk_size)


# ---------------------------------------------------------------------------
# Triton reference kernel (inlined so the benchmark is self-contained).
# Copied verbatim from python/sglang/kernels/ops/gemm/chunked_sgmv_shrink.py
# ---------------------------------------------------------------------------
@triton.jit
def _chunked_lora_shrink_kernel(
    x,
    weights,
    output,
    seg_indptr,
    weight_indices,
    lora_ranks,
    permutation,
    num_segs,
    N: tl.constexpr,  # num_slices * r
    K: tl.constexpr,  # input_dim
    NUM_SLICES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    x_stride_1: tl.constexpr = 1
    x_stride_0: tl.constexpr = K

    w_stride_0: tl.constexpr = N * K
    w_stride_1: tl.constexpr = K
    w_stride_2: tl.constexpr = 1

    output_stride_0: tl.constexpr = N
    output_stride_1: tl.constexpr = 1

    pid_s = tl.program_id(1)
    if pid_s >= num_segs:
        return

    pid_n = tl.program_id(0)

    seg_start = tl.load(seg_indptr + pid_s)
    seg_end = tl.load(seg_indptr + pid_s + 1)
    if seg_start == seg_end:
        return

    w_index = tl.load(weight_indices + pid_s)
    rank = tl.load(lora_ranks + w_index)

    if rank == 0:
        return

    cur_n = tl.minimum(N, rank * NUM_SLICES)

    s_offset_logical = tl.arange(0, BLOCK_M) + seg_start
    s_offset_physical = tl.load(
        permutation + s_offset_logical, mask=s_offset_logical < seg_end, other=0
    )

    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.arange(0, BLOCK_K)
    x_ptrs = x + (
        s_offset_physical[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1
    )
    w_ptrs = (weights + w_index * w_stride_0) + (
        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1
    )

    partial_sum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x_tile = tl.load(
            x_ptrs,
            mask=(s_offset_logical[:, None] < seg_end)
            & (k_offset[None, :] < K - k * BLOCK_K),
            other=0.0,
        )
        w_tile = tl.load(
            w_ptrs,
            mask=(k_offset[:, None] < K - k * BLOCK_K) & (n_offset[None, :] < cur_n),
            other=0.0,
        )
        partial_sum += tl.dot(x_tile, w_tile)

        x_ptrs += BLOCK_K * x_stride_1
        w_ptrs += BLOCK_K * w_stride_2

    partial_sum = partial_sum.to(x.dtype.element_ty)
    output_ptr = output + (
        s_offset_physical[:, None] * output_stride_0
        + n_offset[None, :] * output_stride_1
    )
    output_mask = (s_offset_logical[:, None] < seg_end) & (n_offset[None, :] < cur_n)
    tl.store(output_ptr, partial_sum, mask=output_mask)


# ---------------------------------------------------------------------------
# Decode-scenario input construction
# ---------------------------------------------------------------------------


def _chunked_segments(
    reordered: torch.Tensor, chunk_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Chunk the adapter-sorted (logical) rows into fixed-size segments (Triton).

    Mirrors ChunkedSgmvLoRABackend._get_segments_info: each run of a single
    adapter is split into ceil(count / chunk_size) segments of at most chunk_size
    rows. Returns (seg_indptr, weight_indices) in logical space.
    """
    uniq, counts = torch.unique_consecutive(reordered, return_counts=True)
    seg_lens: List[int] = []
    weight_indices: List[int] = []
    for adapter, count in zip(uniq.tolist(), counts.tolist()):
        remaining = count
        while remaining > 0:
            take = min(chunk_size, remaining)
            seg_lens.append(take)
            weight_indices.append(adapter)
            remaining -= take

    seg_indptr = torch.zeros(len(seg_lens) + 1, dtype=torch.int32)
    seg_indptr[1:] = torch.cumsum(torch.tensor(seg_lens, dtype=torch.int32), dim=0)
    return seg_indptr, torch.tensor(weight_indices, dtype=torch.int32)


def _coalesced_segments(
    reordered: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One segment per active adapter — the CUTLASS layout (no chunking).

    seg_indptr size == num_active_adapters + 1. The CUTLASS grouped GEMM tiles
    over M internally, so each adapter's whole (sorted) token run is a single
    segment.
    """
    uniq, counts = torch.unique_consecutive(reordered, return_counts=True)
    seg_indptr = torch.zeros(uniq.numel() + 1, dtype=torch.int32)
    seg_indptr[1:] = torch.cumsum(counts.to(torch.int32), dim=0)
    return seg_indptr, uniq.to(torch.int32)


def _seg_bundle(
    seg_indptr: torch.Tensor,
    weight_indices: torch.Tensor,
    device: torch.device,
    block_m: int,
) -> Dict[str, Any]:
    seg_lens = (seg_indptr[1:] - seg_indptr[:-1]).to(torch.int32)
    return {
        "seg_indptr": seg_indptr.to(device),
        "seg_lens": seg_lens.to(device),
        "weight_indices": weight_indices.to(device),
        "num_segments": int(weight_indices.numel()),
        "block_m": block_m,
    }


def _make_inputs(
    case: Dict[str, int], dtype: torch.dtype, device: torch.device
) -> Dict[str, Any]:
    num_tokens = case["num_tokens"]
    num_loras = case["num_loras"]
    max_rank = case["max_rank"]
    input_dim = case["input_dim"]
    stack_num = case["stack_num"]
    total_n = stack_num * max_rank

    input_x = torch.randn(num_tokens, input_dim, dtype=dtype, device=device)
    weights = torch.randn(num_loras, total_n, input_dim, dtype=dtype, device=device)

    # Decode: one token per sequence, each assigned a random adapter (zigzag).
    row_adapters = torch.randint(0, num_loras, (num_tokens,), dtype=torch.int32)
    # permutation[logical] -> physical: stable argsort groups rows by adapter.
    permutation = torch.argsort(row_adapters, stable=True).to(torch.int64)
    reordered = row_adapters[permutation]

    # Same permutation, two segmentations: Triton uses the production chunk-size
    # heuristic; CUTLASS uses one coalesced segment per active adapter.
    triton_chunk = _derive_triton_chunk_size(num_tokens)
    triton_indptr, triton_wi = _chunked_segments(reordered, triton_chunk)
    cutlass_indptr, cutlass_wi = _coalesced_segments(reordered)
    # CUTLASS BLOCK_M is unused by the GEMM (it tiles M itself); record the
    # largest coalesced segment for the metrics/label only.
    cutlass_block_m = int((cutlass_indptr[1:] - cutlass_indptr[:-1]).max().item())

    # Full rank for every adapter -> both backends compute all N columns.
    lora_ranks = torch.tensor([max_rank] * num_loras, dtype=torch.int32)

    return {
        "input_x": input_x,
        "weights": weights,
        "stack_num": stack_num,
        "lora_ranks": lora_ranks.to(device),
        "permutation": permutation.to(device),
        "triton": _seg_bundle(triton_indptr, triton_wi, device, triton_chunk),
        "cutlass": _seg_bundle(cutlass_indptr, cutlass_wi, device, cutlass_block_m),
    }


# ---------------------------------------------------------------------------
# Metrics (GEMM-only flops/bytes; the CUTLASS path pays extra copy traffic that
# is intentionally NOT counted, so tflops reflects useful GEMM throughput).
# ---------------------------------------------------------------------------


def _compute_flops_by_segment(
    seg_lens: torch.Tensor,
    weight_indices: torch.Tensor,
    lora_ranks: torch.Tensor,
    stack_num: int,
    K: int,
) -> float:
    seg_lens_cpu = seg_lens.to("cpu")
    weight_indices_cpu = weight_indices.to("cpu")
    lora_ranks_cpu = lora_ranks.to("cpu")

    flops = 0.0
    for seg_idx in range(weight_indices_cpu.numel()):
        seg_len = int(seg_lens_cpu[seg_idx].item())
        lora = int(weight_indices_cpu[seg_idx].item())
        rank = int(lora_ranks_cpu[lora].item())
        flops += 2.0 * seg_len * (rank * stack_num) * K
    return flops


def _estimate_bytes(
    seg_lens: torch.Tensor,
    weight_indices: torch.Tensor,
    lora_ranks: torch.Tensor,
    stack_num: int,
    K: int,
    elem_size: int,
) -> float:
    seg_lens_cpu = seg_lens.to("cpu")
    weight_indices_cpu = weight_indices.to("cpu")
    lora_ranks_cpu = lora_ranks.to("cpu")
    total = 0.0
    prev_lora = -1
    for seg_idx in range(weight_indices_cpu.numel()):
        seg_len = int(seg_lens_cpu[seg_idx].item())
        lora = int(weight_indices_cpu[seg_idx].item())
        n = int(lora_ranks_cpu[lora].item()) * stack_num
        total += seg_len * K * elem_size  # bytes_x: input rows (once each)
        total += seg_len * n * elem_size  # bytes_out: output rows (once each)
        # lora == prev_lora happens only in Triton kernel
        if lora != prev_lora:
            total += (
                n * K * elem_size
            )  # bytes_w: adapter weight (re-read on adapter change)
        prev_lora = lora
    return total


def calc_metrics(
    total_flops: float, total_bytes: float, time_ms: float
) -> Dict[str, float]:
    time_s = time_ms / 1e3
    if time_s <= 0:
        raise RuntimeError("Measured time must be > 0")
    return {
        "tflops": (total_flops / 1e12) / time_s,
        "bandwidth_gbs": (total_bytes / 1e9) / time_s,
        "total_bytes_mb": total_bytes / 1e6,
    }


# ---------------------------------------------------------------------------
# Backend runners
# ---------------------------------------------------------------------------


def _run_cutlass_once(args: Dict[str, Any]):
    seg = args["cutlass"]
    return chunked_sgmv_lora_shrink_forward(
        input_x=args["input_x"],
        weights=args["weights"],
        stack_num=int(args["stack_num"]),
        seg_indptr=seg["seg_indptr"],
        weight_indices=seg["weight_indices"],
        lora_ranks=args["lora_ranks"],
        permutation=args["permutation"],
        seg_lens=seg["seg_lens"],
    )


def _run_triton_once(args: Dict[str, Any]):
    x = args["input_x"]
    weights = args["weights"]
    stack_num = int(args["stack_num"])
    seg = args["triton"]
    num_segs = int(seg["weight_indices"].numel())

    S = x.shape[0]
    N = weights.shape[1]  # stack_num * max_rank
    K = weights.shape[2]

    output = torch.empty((S, N), device=x.device, dtype=x.dtype)
    if num_segs == 0:
        return output

    BLOCK_M = seg["block_m"]  # every segment fits in one M-block
    BLOCK_N = triton.next_power_of_2(N)
    BLOCK_K = 256

    grid = (triton.cdiv(N, BLOCK_N), num_segs)
    _chunked_lora_shrink_kernel[grid](
        x,
        weights,
        output,
        seg["seg_indptr"],
        seg["weight_indices"],
        args["lora_ranks"],
        args["permutation"],
        num_segs,
        N=N,
        K=K,
        NUM_SLICES=stack_num,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return output


def _dtype_from_provider(provider: str) -> torch.dtype:
    if provider == "fp16":
        return torch.float16
    return torch.bfloat16


def _case_label(case: Dict[str, int]) -> str:
    return (
        f"tok={case['num_tokens']},lora={case['num_loras']},"
        f"r={case['max_rank']},K={case['input_dim']},stack={case['stack_num']}"
    )


CASES = DEFAULT_CASES


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["case_id"],
        x_vals=list(range(len(CASES))),
        x_log=False,
        line_arg="provider",
        line_vals=[
            "cutlass_fp16",
            "triton_fp16",
            "cutlass_bf16",
            "triton_bf16",
        ],
        line_names=[
            "CUTLASS fp16",
            "Triton fp16",
            "CUTLASS bf16",
            "Triton bf16",
        ],
        styles=[
            ("green", "-"),
            ("green", "--"),
            ("blue", "-"),
            ("blue", "--"),
        ],
        ylabel="TFLOP/s",
        plot_name="chunked-sgmv-lora-shrink-fwd-cutlass-vs-triton",
        args={},
    )
)
def benchmark(case_id, provider):
    device = torch.device("xpu")
    backend, dtype_name = provider.split("_", 1)
    dtype = _dtype_from_provider(dtype_name)

    case = CASES[case_id]
    inputs = _make_inputs(case, dtype, device)

    K = case["input_dim"]
    stack_num = case["stack_num"]
    elem_size = torch.tensor([], dtype=dtype).element_size()

    seg = inputs[backend]
    total_flops = _compute_flops_by_segment(
        seg["seg_lens"], seg["weight_indices"], inputs["lora_ranks"], stack_num, K
    )
    total_bytes = _estimate_bytes(
        seg["seg_lens"],
        seg["weight_indices"],
        inputs["lora_ranks"],
        stack_num,
        K,
        elem_size,
    )

    quantiles = [0.5, 0.2, 0.8]
    bench_res = triton.testing.do_bench(
        (
            (lambda: _run_cutlass_once(inputs))
            if backend == "cutlass"
            else (lambda: _run_triton_once(inputs))
        ),
        quantiles=quantiles,
    )
    if bench_res is None:
        raise RuntimeError("triton.testing.do_bench returned no result")
    ms, min_ms, max_ms = bench_res

    metrics = calc_metrics(total_flops, total_bytes, ms)

    all_results.append(
        {
            "case_id": case_id,
            "case_label": _case_label(case),
            "provider": provider,
            "backend": backend,
            "dtype": str(dtype),
            "time_ms": ms,
            "time_min_ms": min_ms,
            "time_max_ms": max_ms,
            "tflops": metrics["tflops"],
            "bandwidth_gbs": metrics["bandwidth_gbs"],
            "total_bytes_mb": metrics["total_bytes_mb"],
            "num_tokens": case["num_tokens"],
            "num_loras": case["num_loras"],
            "chunk_size": seg["block_m"],
            "num_segments": seg["num_segments"],
            "max_rank": case["max_rank"],
            "input_dim": case["input_dim"],
            "stack_num": case["stack_num"],
        }
    )

    tflops = lambda t_ms: total_flops * 1e-12 / (t_ms * 1e-3)
    return tflops(ms), tflops(max_ms), tflops(min_ms)


def _sanity_check() -> None:
    torch.manual_seed(123)
    device = torch.device("xpu")
    # Pick a mid-size case so both segments and the permutation exercise the path.
    case = {
        "num_tokens": 1024,
        "num_loras": 8,
        "input_dim": 4096,
        "max_rank": 64,
        "stack_num": 1,
    }
    args = _make_inputs(case, torch.float16, device)
    out = _run_cutlass_once(args)
    out_triton = _run_triton_once(args)

    total_n = case["stack_num"] * case["max_rank"]
    expected = (case["num_tokens"], total_n)
    if tuple(out.shape) != expected:
        raise RuntimeError(
            f"Unexpected CUTLASS output shape: got {tuple(out.shape)}, expected {expected}"
        )
    if tuple(out_triton.shape) != expected:
        raise RuntimeError(
            f"Unexpected Triton output shape: got {tuple(out_triton.shape)}, expected {expected}"
        )

    diff = (out.float() - out_triton.float()).abs()
    max_abs = diff.max().item()
    print(
        f"Sanity check passed: shapes OK, max |CUTLASS - Triton| = {max_abs:.4e} "
        f"(fp16, K={case['input_dim']}, r={case['max_rank']}, "
        f"triton_segments={args['triton']['num_segments']} @ chunk="
        f"{args['triton']['block_m']}, cutlass_segments={args['cutlass']['num_segments']})."
    )


def print_summary(title: str = "Chunked SGMV LoRA Shrink Forward Benchmark Results"):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)

    if not all_results:
        print("No results collected.")
        return

    df = pd.DataFrame(all_results)

    for col in ["time_ms", "tflops", "bandwidth_gbs", "total_bytes_mb"]:
        if col in df.columns:
            df[col] = df[col].round(2)

    display_cols = [
        col
        for col in [
            "case_id",
            "case_label",
            "chunk_size",
            "num_segments",
            "provider",
            "time_ms",
            "tflops",
            "bandwidth_gbs",
        ]
        if col in df.columns
    ]

    print("\nDetailed Results:")
    print(df[display_cols].to_string(index=False))

    if "provider" in df.columns and "tflops" in df.columns:
        print("\n" + "=" * 120)
        print("Summary Statistics by Provider")
        print("=" * 120)
        summary = df.groupby("provider")[["tflops", "bandwidth_gbs", "time_ms"]].agg(
            ["mean", "min", "max", "std"]
        )
        print(summary.to_string())

        # CUTLASS-vs-Triton per-case speedup = triton_ms / cutlass_ms (>1 =>
        # CUTLASS faster). Report the GEOMETRIC mean (the honest central value
        # for ratios; the arithmetic mean is inflated by a few large-chunk wins)
        # and the median, alongside the raw min/max range.
        print("\n" + "=" * 120)
        print(
            "CUTLASS vs Triton per-case speedup = triton_ms / cutlass_ms (>1 => CUTLASS faster)"
        )
        print("=" * 120)
        pivot = df.pivot_table(
            index="case_id", columns="provider", values="time_ms", aggfunc="mean"
        )
        for dtype_name in ["fp16", "bf16"]:
            ccol = f"cutlass_{dtype_name}"
            tcol = f"triton_{dtype_name}"
            if ccol in pivot.columns and tcol in pivot.columns:
                speedup = (pivot[tcol] / pivot[ccol]).dropna()
                if not speedup.empty:
                    import numpy as np

                    geomean = float(np.exp(np.log(speedup).mean()))
                    print(
                        f"  {dtype_name}: geomean={geomean:.2f}x  "
                        f"median={speedup.median():.2f}x  "
                        f"min={speedup.min():.2f}x  max={speedup.max():.2f}x  "
                        f"(CUTLASS wins {int((speedup > 1).sum())}/{len(speedup)} cases)"
                    )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark chunked_sgmv_lora_shrink_forward (decode path) on XPU"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible input generation.",
    )
    parser.add_argument(
        "--print-cases",
        action="store_true",
        help="Print selected benchmark cases before running.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    torch.manual_seed(args.seed)
    CASES = DEFAULT_CASES
    if args.print_cases:
        for i, c in enumerate(CASES):
            print(f"case {i}: {_case_label(c)}")

    _sanity_check()
    benchmark.run(print_data=True)
    print_summary()
    print("Benchmark finished!")
