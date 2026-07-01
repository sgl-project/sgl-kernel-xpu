import argparse
from typing import Any, Dict, List

import pandas as pd
import torch
import triton
import triton.language as tl
from sgl_kernel import sgemm_lora_a_fwd

all_results = []

# Each case is a segmented batched LoRA-A GEMM:
#   x:       (num_tokens, input_dim)
#   weights: (num_loras, stack_num * max_rank, input_dim)
#   output:  (num_tokens, stack_num * max_rank)
# input_dim (K) is the model hidden size; max_rank (the LoRA rank) is small.
DEFAULT_CASES: List[Dict[str, int]] = [
    {"num_tokens": 4096, "num_segments": 4, "num_loras": 2, "max_rank": 16, "input_dim": 2048, "stack_num": 1},
    {"num_tokens": 6144, "num_segments": 8, "num_loras": 4, "max_rank": 32, "input_dim": 4096, "stack_num": 1},
    {"num_tokens": 8192, "num_segments": 8, "num_loras": 4, "max_rank": 64, "input_dim": 4096, "stack_num": 1},
    {"num_tokens": 12288, "num_segments": 16, "num_loras": 2, "max_rank": 16, "input_dim": 4096, "stack_num": 3},
    {"num_tokens": 16384, "num_segments": 16, "num_loras": 4, "max_rank": 32, "input_dim": 5120, "stack_num": 1},
    {"num_tokens": 24576, "num_segments": 32, "num_loras": 8, "max_rank": 64, "input_dim": 4096, "stack_num": 2},
    {"num_tokens": 32768, "num_segments": 32, "num_loras": 4, "max_rank": 16, "input_dim": 8192, "stack_num": 1},
    {"num_tokens": 49152, "num_segments": 64, "num_loras": 8, "max_rank": 32, "input_dim": 4096, "stack_num": 3},
    {"num_tokens": 65536, "num_segments": 64, "num_loras": 4, "max_rank": 64, "input_dim": 5120, "stack_num": 1},
    {"num_tokens": 81920, "num_segments": 128, "num_loras": 8, "max_rank": 16, "input_dim": 8192, "stack_num": 2},
    {"num_tokens": 98304, "num_segments": 128, "num_loras": 4, "max_rank": 32, "input_dim": 4096, "stack_num": 1},
    {"num_tokens": 122880, "num_segments": 256, "num_loras": 8, "max_rank": 64, "input_dim": 8192, "stack_num": 3},
]


# ---------------------------------------------------------------------------
# Triton reference kernel (inlined so the benchmark is self-contained).
# Copied from sglang/srt/lora/triton_ops/sgemm_lora_a.py; the permutation /
# sorted-by-adapter path is dropped since the benchmark runs unsorted tokens.
# ---------------------------------------------------------------------------
@triton.jit
def _sgemm_lora_a_kernel(
    x,
    weights,
    output,
    N,  # stack_num * r
    K,  # input_dim
    stack_num,
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
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch_id = tl.program_id(axis=1)
    w_index = tl.load(weight_indices + batch_id)
    rank = tl.load(lora_ranks + w_index)

    if rank == 0:
        return

    pid = tl.program_id(axis=0)
    seg_start = tl.load(seg_indptr + batch_id)
    seg_len = tl.load(seg_lens + batch_id)
    if seg_len == 0:
        return

    N = tl.minimum(N, rank * stack_num)

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

    partial_sum = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x_tile = tl.load(
            x_ptrs,
            mask=(s_offset[:, None] < seg_len) & (k_offset[None, :] < K - k * BLOCK_K),
            other=0.0,
        )
        w_tile = tl.load(
            w_ptrs,
            mask=(k_offset[:, None] < K - k * BLOCK_K) & (n_offset[None, :] < N),
            other=0.0,
        )
        partial_sum += tl.dot(x_tile, w_tile)

        x_ptrs += BLOCK_K * x_stride_1
        w_ptrs += BLOCK_K * w_stride_2

    partial_sum = partial_sum.to(x.dtype.element_ty)
    output_mask = (s_offset[:, None] < seg_len) & (n_offset[None, :] < N)
    output_ptr = output + (
        s_physical[:, None] * output_stride_0 + n_offset[None, :] * output_stride_1
    )
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
    stack_num: int,
    K: int,
) -> float:
    """Effective GEMM flops: sum over segments of 2 * M_s * (rank_s * stack_num) * K."""
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
    """Memory traffic estimate: read x + read the referenced weight, write output."""
    seg_lens_cpu = seg_lens.to("cpu")
    weight_indices_cpu = weight_indices.to("cpu")
    lora_ranks_cpu = lora_ranks.to("cpu")

    total = 0.0
    for seg_idx in range(weight_indices_cpu.numel()):
        seg_len = int(seg_lens_cpu[seg_idx].item())
        lora = int(weight_indices_cpu[seg_idx].item())
        n = int(lora_ranks_cpu[lora].item()) * stack_num
        bytes_x = seg_len * K * elem_size
        bytes_w = n * K * elem_size
        bytes_out = seg_len * n * elem_size
        total += bytes_x + bytes_w + bytes_out
    return total


def calc_metrics(total_flops: float, total_bytes: float, time_ms: float) -> Dict[str, float]:
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
    input_dim = case["input_dim"]
    stack_num = case["stack_num"]
    total_n = stack_num * max_rank

    input_x = torch.randn(num_tokens, input_dim, dtype=dtype, device=device)
    weights = torch.randn(num_loras, total_n, input_dim, dtype=dtype, device=device)

    seg_indptr = _build_seg_indptr(num_tokens, num_segments, device)
    seg_lens = _build_seg_lens(seg_indptr)
    weight_indices = torch.randint(
        0, num_loras, (seg_lens.numel(),), dtype=torch.int32, device=device
    )
    # Full rank for every adapter -> both backends compute the full N columns.
    lora_ranks = torch.tensor([max_rank] * num_loras, dtype=torch.int32, device=device)

    return {
        "input_x": input_x,
        "weights": weights,
        "stack_num": stack_num,
        "seg_indptr": seg_indptr,
        "seg_lens": seg_lens,
        "weight_indices": weight_indices,
        "lora_ranks": lora_ranks,
    }


def _run_sgl_once(args: Dict[str, Any]):
    return sgemm_lora_a_fwd(
        input_x=args["input_x"],
        weights=args["weights"],
        stack_num=int(args["stack_num"]),
        seg_indptr=args["seg_indptr"],
        weight_indices=args["weight_indices"],
        lora_ranks=args["lora_ranks"],
        seg_lens=args["seg_lens"],
    )


def _run_triton_once(args: Dict[str, Any]):
    x = args["input_x"]
    weights = args["weights"]
    seg_lens = args["seg_lens"]
    seg_indptr = args["seg_indptr"]
    weight_indices = args["weight_indices"]
    lora_ranks = args["lora_ranks"]
    stack_num = int(args["stack_num"])

    S = x.shape[0]
    R = weights.shape[-2]  # stack_num * max_rank
    K = weights.shape[-1]

    BLOCK_S = 16
    BLOCK_K = 256
    BLOCK_R = 16

    max_len = int(seg_lens.max().item()) if seg_lens.numel() > 0 else 0
    bs = int(seg_lens.numel())

    output = torch.empty((S, R), device=x.device, dtype=x.dtype)
    if max_len == 0 or bs == 0:
        return output

    grid = (triton.cdiv(max_len, BLOCK_S) * triton.cdiv(R, BLOCK_R), bs)
    _sgemm_lora_a_kernel[grid](
        x,
        weights,
        output,
        R,
        K,
        stack_num,
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
        BLOCK_S,
        BLOCK_R,
        BLOCK_K,
    )
    return output


def _dtype_from_provider(provider: str) -> torch.dtype:
    if provider == "fp16":
        return torch.float16
    if provider == "bf16":
        return torch.bfloat16
    return torch.float32


def _case_label(case: Dict[str, int]) -> str:
    return (
        f"tok={case['num_tokens']},seg={case['num_segments']},lora={case['num_loras']},"
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
            "sgl_fp16",
            "triton_fp16",
            "sgl_bf16",
            "triton_bf16",
        ],
        line_names=[
            "SGL fp16",
            "Triton fp16",
            "SGL bf16",
            "Triton bf16",
        ],
        styles=[
            ("green", "-"),
            ("green", "--"),
            ("blue", "-"),
            ("blue", "--"),
        ],
        ylabel="TFLOP/s",
        plot_name="sgemm-lora-a-fwd-sgl-vs-triton",
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

    total_flops = _compute_flops_by_segment(
        inputs["seg_lens"], inputs["weight_indices"], inputs["lora_ranks"], stack_num, K
    )
    total_bytes = _estimate_bytes(
        inputs["seg_lens"], inputs["weight_indices"], inputs["lora_ranks"],
        stack_num, K, elem_size,
    )

    quantiles = [0.5, 0.2, 0.8]
    bench_res = triton.testing.do_bench(
        (
            (lambda: _run_sgl_once(inputs))
            if backend == "sgl"
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
            "time_us": 1e3 * ms,
            "time_min_us": 1e3 * min_ms,
            "time_max_us": 1e3 * max_ms,
            "tflops": metrics["tflops"],
            "bandwidth_gbs": metrics["bandwidth_gbs"],
            "total_bytes_mb": metrics["total_bytes_mb"],
            "num_tokens": case["num_tokens"],
            "num_segments": case["num_segments"],
            "num_loras": case["num_loras"],
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
    case = DEFAULT_CASES[0]
    args = _make_inputs(case, torch.float16, device)
    out = _run_sgl_once(args)
    out_triton = _run_triton_once(args)

    total_n = case["stack_num"] * case["max_rank"]
    expected = (case["num_tokens"], total_n)
    if tuple(out.shape) != expected:
        raise RuntimeError(
            f"Unexpected SGL output shape: got {tuple(out.shape)}, expected {expected}"
        )
    if tuple(out_triton.shape) != expected:
        raise RuntimeError(
            f"Unexpected Triton output shape: got {tuple(out_triton.shape)}, expected {expected}"
        )

    diff = (out.float() - out_triton.float()).abs()
    max_abs = diff.max().item()
    print(
        f"Sanity check passed: shapes OK, max |SGL - Triton| = {max_abs:.4e} "
        f"(fp16, K={case['input_dim']})."
    )


def print_summary(title: str = "SGEMM LoRA-A Forward Benchmark Results"):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)

    if not all_results:
        print("No results collected.")
        return

    df = pd.DataFrame(all_results)

    for col in ["time_us", "tflops", "bandwidth_gbs", "total_bytes_mb"]:
        if col in df.columns:
            df[col] = df[col].round(2)

    display_cols = [
        col
        for col in [
            "case_id",
            "case_label",
            "provider",
            "time_us",
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
        summary = df.groupby("provider")[["tflops", "bandwidth_gbs", "time_us"]].agg(
            ["mean", "min", "max", "std"]
        )
        print(summary.to_string())


def _parse_args():
    parser = argparse.ArgumentParser(description="Benchmark sgemm_lora_a_fwd on XPU")
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
