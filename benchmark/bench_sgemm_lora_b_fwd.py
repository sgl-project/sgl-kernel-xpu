import argparse
from typing import Any, Dict, List

import pandas as pd
import torch
import triton
import triton.language as tl
from sgl_kernel import sgemm_lora_b_fwd

all_results = []

# Each case is a segmented batched LoRA-B GEMM:
#   input_x: (num_tokens, max_rank)
#   weights: (num_loras, output_dim, max_rank)
#   output:  (num_tokens, output_dim)
# Cases with "use_base_output": True fuse a residual base_output:
#   D = scalings[l] * (input_x @ weights[l]^T) + base_output   (beta=1);
# otherwise only the scaled LoRA-B projection is computed (beta=0).
DEFAULT_CASES: List[Dict[str, int]] = [
    {
        "num_tokens": 65536,
        "num_segments": 4,
        "num_loras": 2,
        "max_rank": 16,
        "output_dim": 2048,
    },
    {
        "num_tokens": 65536,
        "num_segments": 8,
        "num_loras": 4,
        "max_rank": 32,
        "output_dim": 4096,
    },
    {
        "num_tokens": 65536,
        "num_segments": 8,
        "num_loras": 4,
        "max_rank": 64,
        "output_dim": 4096,
    },
    {
        "num_tokens": 65536,
        "num_segments": 16,
        "num_loras": 2,
        "max_rank": 16,
        "output_dim": 4096,
    },
    {
        "num_tokens": 65536,
        "num_segments": 16,
        "num_loras": 4,
        "max_rank": 32,
        "output_dim": 5120,
    },
    {
        "num_tokens": 81920,
        "num_segments": 32,
        "num_loras": 8,
        "max_rank": 64,
        "output_dim": 4096,
    },
    {
        "num_tokens": 81920,
        "num_segments": 32,
        "num_loras": 4,
        "max_rank": 16,
        "output_dim": 8192,
    },
    {
        "num_tokens": 98304,
        "num_segments": 64,
        "num_loras": 8,
        "max_rank": 32,
        "output_dim": 4096,
    },
    {
        "num_tokens": 98304,
        "num_segments": 64,
        "num_loras": 4,
        "max_rank": 64,
        "output_dim": 5120,
    },
    {
        "num_tokens": 122880,
        "num_segments": 128,
        "num_loras": 8,
        "max_rank": 16,
        "output_dim": 8192,
    },
    {
        "num_tokens": 122880,
        "num_segments": 128,
        "num_loras": 4,
        "max_rank": 32,
        "output_dim": 4096,
    },
    {
        "num_tokens": 122880,
        "num_segments": 256,
        "num_loras": 8,
        "max_rank": 64,
        "output_dim": 8192,
    },
    # ----- Fused residual (base_output) cases: D = scalings*(x@W^T) + base_output
    # (beta=1). Same shape family as above, exercising the fused-add epilogue path.
    {
        "num_tokens": 122880,
        "num_segments": 32,
        "num_loras": 4,
        "max_rank": 16,
        "output_dim": 8192,
        "use_base_output": True,
    },
    {
        "num_tokens": 122880,
        "num_segments": 64,
        "num_loras": 4,
        "max_rank": 64,
        "output_dim": 5120,
        "use_base_output": True,
    },
    {
        "num_tokens": 122880,
        "num_segments": 256,
        "num_loras": 8,
        "max_rank": 64,
        "output_dim": 8192,
        "use_base_output": True,
    },
]


# ---------------------------------------------------------------------------
# Triton reference kernel (inlined so the benchmark is self-contained).
# Copied from sglang/srt/lora/triton_ops/sgemm_lora_b.py; the permutation /
# sorted-by-adapter path is dropped since the benchmark runs unsorted tokens
# (s_physical == seg_start + s_offset).
# ---------------------------------------------------------------------------
@triton.jit
def _sgemm_lora_b_kernel(
    x,
    weights,
    output,
    N,  # output_dim
    K,  # max_rank
    x_stride_0,
    x_stride_1,
    w_stride_0,
    w_stride_1,
    w_stride_2,
    output_stride_0,
    output_stride_1,
    seg_lens,
    seg_indptr,
    weight_indices,
    lora_ranks,
    scalings,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch_id = tl.program_id(axis=1)
    w_index = tl.load(weight_indices + batch_id)
    rank = tl.load(lora_ranks + w_index)

    # If rank is 0, this kernel is a no-op.
    if rank == 0:
        return

    pid = tl.program_id(axis=0)
    seg_len = tl.load(seg_lens + batch_id)
    if seg_len == 0:
        return
    seg_start = tl.load(seg_indptr + batch_id)
    scaling = tl.load(scalings + w_index)
    # Adjust K (rank) according to the specific LoRA adapter.
    K = tl.minimum(K, rank)

    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_s = pid // num_pid_n
    pid_n = pid % num_pid_n
    if pid_s * BLOCK_S >= seg_len:
        return

    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.arange(0, BLOCK_K)
    s_physical = (seg_start + s_offset).to(tl.int64)
    x_ptrs = x + (s_physical[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1)
    w_ptrs = (weights + w_index * w_stride_0) + (
        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1
    )

    n_mask = n_offset[None, :] < N
    partial_sum = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x_tile = tl.load(
            x_ptrs,
            mask=(s_offset[:, None] < seg_len) & (k_offset[None, :] < K - k * BLOCK_K),
            other=0.0,
        )
        w_tile = tl.load(
            w_ptrs,
            mask=(k_offset[:, None] < K - k * BLOCK_K) & n_mask,
            other=0.0,
        )
        partial_sum += tl.dot(x_tile, w_tile)

        x_ptrs += BLOCK_K * x_stride_1
        w_ptrs += BLOCK_K * w_stride_2

    partial_sum *= scaling
    partial_sum = partial_sum.to(x.dtype.element_ty)
    output_ptr = output + (
        s_physical[:, None] * output_stride_0 + n_offset[None, :] * output_stride_1
    )
    output_mask = (s_offset[:, None] < seg_len) & n_mask
    partial_sum += tl.load(output_ptr, mask=output_mask, other=0.0)
    tl.store(output_ptr, partial_sum, mask=output_mask)


def _build_seg_indptr(
    num_tokens: int, num_segments: int, device: torch.device
) -> torch.Tensor:
    seg = min(num_segments, num_tokens)
    lengths = torch.full((seg,), num_tokens // seg, dtype=torch.int32)
    lengths[: num_tokens % seg] += 1
    seg_indptr = torch.zeros(seg + 1, dtype=torch.int32)
    seg_indptr[1:] = torch.cumsum(lengths, dim=0)
    return seg_indptr.to(device)


def _build_seg_lens(seg_indptr: torch.Tensor) -> torch.Tensor:
    return (seg_indptr[1:] - seg_indptr[:-1]).to(torch.int32)


def _compute_flops_by_segment(
    seg_lens: torch.Tensor,
    weight_indices: torch.Tensor,
    lora_ranks: torch.Tensor,
    N: int,
) -> float:
    """Effective GEMM flops: sum over segments of 2 * M_s * N * rank_s."""
    seg_lens_cpu = seg_lens.to("cpu")
    weight_indices_cpu = weight_indices.to("cpu")
    lora_ranks_cpu = lora_ranks.to("cpu")

    flops = 0.0
    for seg_idx in range(weight_indices_cpu.numel()):
        seg_len = int(seg_lens_cpu[seg_idx].item())
        lora = int(weight_indices_cpu[seg_idx].item())
        rank = int(lora_ranks_cpu[lora].item())
        flops += 2.0 * seg_len * N * rank
    return flops


def _estimate_bytes(
    seg_lens: torch.Tensor,
    weight_indices: torch.Tensor,
    lora_ranks: torch.Tensor,
    N: int,
    elem_size: int,
) -> float:
    """Memory traffic estimate: read x + read the referenced weight, write output."""
    seg_lens_cpu = seg_lens.to("cpu")
    weight_indices_cpu = weight_indices.to("cpu")
    lora_ranks_cpu = lora_ranks.to("cpu")

    total = 0.0
    for seg_idx in range(weight_indices_cpu.numel()):
        seg_len = int(seg_lens_cpu[seg_idx].item())
        lora = int(weight_indices_cpu[seg_idx].item())
        rank = int(lora_ranks_cpu[lora].item())
        bytes_x = seg_len * rank * elem_size
        bytes_w = N * rank * elem_size
        bytes_out = seg_len * N * elem_size
        total += bytes_x + bytes_w + bytes_out
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


def _make_inputs(
    case: Dict[str, int], dtype: torch.dtype, device: torch.device
) -> Dict[str, Any]:
    num_tokens = case["num_tokens"]
    num_segments = min(case["num_segments"], num_tokens)
    num_loras = case["num_loras"]
    max_rank = case["max_rank"]
    output_dim = case["output_dim"]

    # input_x is the LoRA-A projection: (num_tokens, max_rank), K == max_rank.
    input_x = torch.randn(num_tokens, max_rank, dtype=dtype, device=device)
    weights = torch.randn(num_loras, output_dim, max_rank, dtype=dtype, device=device)

    seg_indptr = _build_seg_indptr(num_tokens, num_segments, device)
    seg_lens = _build_seg_lens(seg_indptr)
    weight_indices = torch.randint(
        0, num_loras, (seg_lens.numel(),), dtype=torch.int32, device=device
    )
    # Full rank for every adapter -> both backends compute the full K reduction.
    lora_ranks = torch.tensor([max_rank] * num_loras, dtype=torch.int32, device=device)
    # Per-adapter scaling (lora_alpha / rank), exercising the per-segment alpha.
    scalings = torch.rand(num_loras, dtype=torch.float32, device=device) * 2.0 + 0.5

    # Optional residual: when the case sets use_base_output, both backends fuse
    # D = scalings[l] * (x @ W^T) + base_output (beta=1). Otherwise it stays None
    # (beta=0, pure scaled projection). See _run_*_once for how each path uses it.
    base_output = None
    if case.get("use_base_output", False):
        base_output = torch.randn(num_tokens, output_dim, dtype=dtype, device=device)

    return {
        "input_x": input_x,
        "weights": weights,
        "seg_indptr": seg_indptr,
        "seg_lens": seg_lens,
        "weight_indices": weight_indices,
        "lora_ranks": lora_ranks,
        "scalings": scalings,
        "base_output": base_output,
    }


def _run_cutlass_once(args: Dict[str, Any]):
    # base_output is a read-only residual source C here (the kernel writes a
    # fresh output D), so it is never mutated and needs no per-call reset.
    return sgemm_lora_b_fwd(
        input_x=args["input_x"],
        weights=args["weights"],
        seg_indptr=args["seg_indptr"],
        weight_indices=args["weight_indices"],
        lora_ranks=args["lora_ranks"],
        scalings=args["scalings"],
        seg_lens=args["seg_lens"],
        base_output=args["base_output"],
    )


def _run_triton_once(args: Dict[str, Any]):
    x = args["input_x"]
    weights = args["weights"]
    seg_lens = args["seg_lens"]
    seg_indptr = args["seg_indptr"]
    weight_indices = args["weight_indices"]
    lora_ranks = args["lora_ranks"]
    scalings = args["scalings"]

    S = x.shape[0]
    N = weights.shape[-2]  # output_dim
    K = weights.shape[-1]  # max_rank

    BLOCK_S = 16
    BLOCK_N = 256
    BLOCK_K = 16

    max_len = int(seg_lens.max().item()) if seg_lens.numel() > 0 else 0
    bs = int(seg_lens.numel())

    # The Triton kernel accumulates in place (partial_sum += output). With a
    # residual we start from a fresh copy of base_output so repeated do_bench
    # invocations don't drift; without one we start from zeros -> pure scaled
    # projection (matches the CUTLASS call above).
    base_output = args["base_output"]
    if base_output is not None:
        output = base_output.clone()
    else:
        output = torch.zeros((S, N), device=x.device, dtype=x.dtype)
    if max_len == 0 or bs == 0:
        return output

    grid = (triton.cdiv(max_len, BLOCK_S) * triton.cdiv(N, BLOCK_N), bs)
    _sgemm_lora_b_kernel[grid](
        x,
        weights,
        output,
        N,
        K,
        x.stride(0),
        x.stride(1),
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        output.stride(0),
        output.stride(1),
        seg_lens,
        seg_indptr,
        weight_indices,
        lora_ranks,
        scalings,
        BLOCK_S,
        BLOCK_N,
        BLOCK_K,
    )
    return output


def _dtype_from_provider(provider: str) -> torch.dtype:
    if provider == "fp16":
        return torch.float16
    return torch.bfloat16


def _case_label(case: Dict[str, int]) -> str:
    residual = "+base" if case.get("use_base_output", False) else ""
    return (
        f"tok={case['num_tokens']},seg={case['num_segments']},lora={case['num_loras']},"
        f"r={case['max_rank']},N={case['output_dim']}{residual}"
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
        plot_name="sgemm-lora-b-fwd-cutlass-vs-triton",
        args={},
    )
)
def benchmark(case_id, provider):
    device = torch.device("xpu")
    backend, dtype_name = provider.split("_", 1)
    dtype = _dtype_from_provider(dtype_name)

    case = CASES[case_id]
    inputs = _make_inputs(case, dtype, device)

    N = case["output_dim"]
    elem_size = torch.tensor([], dtype=dtype).element_size()

    total_flops = _compute_flops_by_segment(
        inputs["seg_lens"], inputs["weight_indices"], inputs["lora_ranks"], N
    )
    total_bytes = _estimate_bytes(
        inputs["seg_lens"],
        inputs["weight_indices"],
        inputs["lora_ranks"],
        N,
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
            "num_segments": case["num_segments"],
            "num_loras": case["num_loras"],
            "max_rank": case["max_rank"],
            "output_dim": case["output_dim"],
            "use_base_output": case.get("use_base_output", False),
        }
    )

    tflops = lambda t_ms: total_flops * 1e-12 / (t_ms * 1e-3)
    return tflops(ms), tflops(max_ms), tflops(min_ms)


def _sanity_check() -> None:
    torch.manual_seed(123)
    device = torch.device("xpu")

    # Check both the pure projection (beta=0) and the fused residual (beta=1)
    # paths so the base_output add is validated too, not just the shapes.
    for label, case in (
        ("no-residual", dict(DEFAULT_CASES[0], use_base_output=False)),
        ("residual", dict(DEFAULT_CASES[0], use_base_output=True)),
    ):
        args = _make_inputs(case, torch.float16, device)
        out = _run_cutlass_once(args)
        out_triton = _run_triton_once(args)

        expected = (case["num_tokens"], case["output_dim"])
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
            f"Sanity check passed ({label}): shapes OK, "
            f"max |CUTLASS - Triton| = {max_abs:.4e} (fp16, N={case['output_dim']})."
        )


def print_summary(title: str = "SGEMM LoRA-B Forward Benchmark Results"):
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


def _parse_args():
    parser = argparse.ArgumentParser(description="Benchmark sgemm_lora_b_fwd on XPU")
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
