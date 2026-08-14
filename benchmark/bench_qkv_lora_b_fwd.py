import argparse
from typing import Any, Dict, List

import pandas as pd
import torch
import triton
import triton.language as tl
from sgl_kernel import qkv_lora_b_fwd

all_results = []

# Each case is a fused, segmented batched QKV LoRA-B GEMM:
#   input_x:    (num_tokens, 3 * max_rank)          -- packed q/k/v LoRA-A output
#   qkv_lora_b: (num_loras, N_Q + 2 * N_KV, max_rank)
#   output:     (num_tokens, N_Q + 2 * N_KV)
# The output columns are partitioned into q/k/v bands by output_offset =
# [0, N_Q, N_Q + N_KV, N_Q + 2 * N_KV]. Cases with "use_base_output": True fuse a
# residual base_output:
#   D[:, band_p] = scalings[l] * (x[:, band_p] @ W[l, band_p]^T) + base_output[:, band_p]
# (beta=1); otherwise only the scaled LoRA-B projection is computed (beta=0).
DEFAULT_CASES: List[Dict[str, int]] = [
    {
        "num_tokens": 65536,
        "num_segments": 8,
        "num_loras": 8,
        "max_rank": 32,
        "n_q": 4096,
        "n_kv": 1024,
    },
    {
        "num_tokens": 65536,
        "num_segments": 8,
        "num_loras": 8,
        "max_rank": 64,
        "n_q": 4096,
        "n_kv": 512,
    },
    {
        "num_tokens": 65536,
        "num_segments": 16,
        "num_loras": 16,
        "max_rank": 16,
        "n_q": 4096,
        "n_kv": 4096,
    },
    {
        "num_tokens": 81920,
        "num_segments": 32,
        "num_loras": 32,
        "max_rank": 16,
        "n_q": 8192,
        "n_kv": 1024,
    },
    {
        "num_tokens": 98304,
        "num_segments": 64,
        "num_loras": 64,
        "max_rank": 32,
        "n_q": 4096,
        "n_kv": 512,
    },
    {
        "num_tokens": 122880,
        "num_segments": 128,
        "num_loras": 128,
        "max_rank": 16,
        "n_q": 8192,
        "n_kv": 1024,
    },
    {
        "num_tokens": 122880,
        "num_segments": 256,
        "num_loras": 256,
        "max_rank": 64,
        "n_q": 8192,
        "n_kv": 1024,
    },
    # ----- Fused residual (base_output) cases: D = scalings*(x@W^T) + base_output
    # (beta=1). Same shape family as above, exercising the fused-add epilogue path.
    {
        "num_tokens": 122880,
        "num_segments": 32,
        "num_loras": 32,
        "max_rank": 16,
        "n_q": 8192,
        "n_kv": 1024,
        "use_base_output": True,
    },
    {
        "num_tokens": 122880,
        "num_segments": 64,
        "num_loras": 64,
        "max_rank": 64,
        "n_q": 4096,
        "n_kv": 512,
        "use_base_output": True,
    },
    {
        "num_tokens": 122880,
        "num_segments": 256,
        "num_loras": 256,
        "max_rank": 64,
        "n_q": 8192,
        "n_kv": 1024,
        "use_base_output": True,
    },
]


# ---------------------------------------------------------------------------
# Triton reference kernel (inlined so the benchmark is self-contained).
# Copied from sglang/srt/lora/triton_ops/qkv_lora_b.py, taking the logic for the
# case when permutations are None (SORTED_BY_ADAPTER=False), so the physical
# token position is simply seg_start + s_offset.
# ---------------------------------------------------------------------------
@triton.jit
def _qkv_lora_b_kernel(
    x,
    weights,
    output,
    K,  # max_rank
    max_qkv_out_dim,  # max(output_q_dim, output_kv_dim)
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
    n_offs,  # output_offset: q/k/v slice boundaries on the output dim
    scalings,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # qkv_id decides which of q,k,v to compute (0: q, 1: k, 2: v).
    batch_id = tl.program_id(axis=2)
    w_index = tl.load(weight_indices + batch_id)
    rank = tl.load(lora_ranks + w_index)

    # If rank is 0, this kernel is a no-op.
    if rank == 0:
        return

    qkv_id = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)
    seg_len = tl.load(seg_lens + batch_id)
    if seg_len == 0:
        return
    seg_start = tl.load(seg_indptr + batch_id)
    n_start = tl.load(n_offs + qkv_id)
    n_size = tl.load(n_offs + qkv_id + 1) - n_start
    scaling = tl.load(scalings + w_index)
    # Adjust K (rank) according to the specific LoRA adapter.
    K = tl.minimum(K, rank)

    num_pid_n = tl.cdiv(max_qkv_out_dim, BLOCK_N)
    pid_s = pid // num_pid_n
    pid_n = pid % num_pid_n
    if pid_s * BLOCK_S >= seg_len:
        return

    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.arange(0, BLOCK_K)

    s_physical = (seg_start + s_offset).to(tl.int64)
    x_ptrs = (
        x
        + (qkv_id * K) * x_stride_1
        + (s_physical[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1)
    )
    w_ptrs = (weights + w_index * w_stride_0 + n_start * w_stride_1) + (
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
            mask=(k_offset[:, None] < K - k * BLOCK_K) & (n_offset[None, :] < n_size),
            other=0.0,
        )
        partial_sum += tl.dot(x_tile, w_tile)

        x_ptrs += BLOCK_K * x_stride_1
        w_ptrs += BLOCK_K * w_stride_2

    partial_sum *= scaling
    partial_sum = partial_sum.to(x.dtype.element_ty)
    output_ptr = (
        output
        + n_start * output_stride_1
        + (s_physical[:, None] * output_stride_0 + n_offset[None, :] * output_stride_1)
    )
    output_mask = (s_offset[:, None] < seg_len) & (n_offset[None, :] < n_size)
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


def _output_offset(n_q: int, n_kv: int, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [0, n_q, n_q + n_kv, n_q + 2 * n_kv], dtype=torch.int32, device=device
    )


def _band_dims(n_q: int, n_kv: int) -> List[int]:
    """Per-projection output dims [N_Q, N_KV, N_KV]."""
    return [n_q, n_kv, n_kv]


def _compute_flops_by_segment(
    seg_lens: torch.Tensor,
    weight_indices: torch.Tensor,
    lora_ranks: torch.Tensor,
    band_dims: List[int],
) -> float:
    """Effective GEMM flops: sum over segments and q/k/v of 2 * M_s * N_p * rank_s."""
    seg_lens_cpu = seg_lens.to("cpu")
    weight_indices_cpu = weight_indices.to("cpu")
    lora_ranks_cpu = lora_ranks.to("cpu")

    flops = 0.0
    for seg_idx in range(weight_indices_cpu.numel()):
        seg_len = int(seg_lens_cpu[seg_idx].item())
        lora = int(weight_indices_cpu[seg_idx].item())
        rank = int(lora_ranks_cpu[lora].item())
        for n_p in band_dims:
            flops += 2.0 * seg_len * n_p * rank
    return flops


def _estimate_bytes(
    seg_lens: torch.Tensor,
    weight_indices: torch.Tensor,
    lora_ranks: torch.Tensor,
    band_dims: List[int],
    elem_size: int,
) -> float:
    """Memory traffic estimate: read x band + weight band, write output band, per q/k/v."""
    seg_lens_cpu = seg_lens.to("cpu")
    weight_indices_cpu = weight_indices.to("cpu")
    lora_ranks_cpu = lora_ranks.to("cpu")

    total = 0.0
    for seg_idx in range(weight_indices_cpu.numel()):
        seg_len = int(seg_lens_cpu[seg_idx].item())
        lora = int(weight_indices_cpu[seg_idx].item())
        rank = int(lora_ranks_cpu[lora].item())
        for n_p in band_dims:
            bytes_x = seg_len * rank * elem_size
            bytes_w = n_p * rank * elem_size
            bytes_out = seg_len * n_p * elem_size
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
    n_q = case["n_q"]
    n_kv = case["n_kv"]
    n_total = n_q + 2 * n_kv

    # input_x is the packed q/k/v LoRA-A projection: (num_tokens, 3 * max_rank).
    input_x = torch.randn(num_tokens, 3 * max_rank, dtype=dtype, device=device)
    qkv_lora_b = torch.randn(num_loras, n_total, max_rank, dtype=dtype, device=device)

    seg_indptr = _build_seg_indptr(num_tokens, num_segments, device)
    seg_lens = _build_seg_lens(seg_indptr)
    weight_indices = torch.randint(
        0, num_loras, (seg_lens.numel(),), dtype=torch.int32, device=device
    )
    lora_ranks = torch.tensor([max_rank] * num_loras, dtype=torch.int32, device=device)
    # Per-adapter scaling (lora_alpha / rank), exercising the per-group alpha.
    scalings = torch.rand(num_loras, dtype=torch.float32, device=device) * 2.0 + 0.5

    output_offset = _output_offset(n_q, n_kv, device)
    max_qkv_out_dim = max(n_q, n_kv)

    # Optional residual: when the case sets use_base_output, both backends fuse
    # D[:, band] = scalings[l] * (x @ W^T) + base_output[:, band] (beta=1).
    base_output = None
    if case.get("use_base_output", False):
        base_output = torch.randn(num_tokens, n_total, dtype=dtype, device=device)

    return {
        "input_x": input_x,
        "qkv_lora_b": qkv_lora_b,
        "output_offset": output_offset,
        "max_qkv_out_dim": max_qkv_out_dim,
        "seg_indptr": seg_indptr,
        "seg_lens": seg_lens,
        "weight_indices": weight_indices,
        "lora_ranks": lora_ranks,
        "scalings": scalings,
        "base_output": base_output,
        "band_dims": _band_dims(n_q, n_kv),
        "n_total": n_total,
    }


def _run_cutlass_once(args: Dict[str, Any]):
    # base_output is a read-only residual source C here (the kernel writes a fresh
    # output D), so it is never mutated and needs no per-call reset.
    return qkv_lora_b_fwd(
        input_x=args["input_x"],
        qkv_lora_b=args["qkv_lora_b"],
        output_offset=args["output_offset"],
        max_qkv_out_dim=args["max_qkv_out_dim"],
        seg_indptr=args["seg_indptr"],
        weight_indices=args["weight_indices"],
        lora_ranks=args["lora_ranks"],
        scalings=args["scalings"],
        seg_lens=args["seg_lens"],
        base_output=args["base_output"],
    )


def _run_triton_once(args: Dict[str, Any]):
    x = args["input_x"]
    weights = args["qkv_lora_b"]
    seg_lens = args["seg_lens"]
    seg_indptr = args["seg_indptr"]
    weight_indices = args["weight_indices"]
    lora_ranks = args["lora_ranks"]
    scalings = args["scalings"]
    output_offset = args["output_offset"]
    max_qkv_out_dim = args["max_qkv_out_dim"]

    S = x.shape[0]
    K = weights.shape[-1]  # max_rank
    n_total = args["n_total"]

    BLOCK_S = 16
    BLOCK_N = 64
    BLOCK_K = 16

    max_len = int(seg_lens.max().item()) if seg_lens.numel() > 0 else 0
    bs = int(seg_lens.numel())

    base_output = args["base_output"]
    if base_output is not None:
        output = base_output.clone()
    else:
        output = torch.zeros((S, n_total), device=x.device, dtype=x.dtype)
    if max_len == 0 or bs == 0:
        return output

    grid = (
        triton.cdiv(max_len, BLOCK_S) * triton.cdiv(max_qkv_out_dim, BLOCK_N),
        3,
        bs,
    )
    _qkv_lora_b_kernel[grid](
        x,
        weights,
        output,
        K,
        max_qkv_out_dim,
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
        output_offset,
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
        f"r={case['max_rank']},Nq={case['n_q']},Nkv={case['n_kv']}{residual}"
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
        plot_name="qkv-lora-b-fwd-cutlass-vs-triton",
        args={},
    )
)
def benchmark(case_id, provider):
    device = torch.device("xpu")
    backend, dtype_name = provider.split("_", 1)
    dtype = _dtype_from_provider(dtype_name)

    case = CASES[case_id]
    inputs = _make_inputs(case, dtype, device)

    band_dims = inputs["band_dims"]
    elem_size = torch.tensor([], dtype=dtype).element_size()

    total_flops = _compute_flops_by_segment(
        inputs["seg_lens"], inputs["weight_indices"], inputs["lora_ranks"], band_dims
    )
    total_bytes = _estimate_bytes(
        inputs["seg_lens"],
        inputs["weight_indices"],
        inputs["lora_ranks"],
        band_dims,
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
            "n_q": case["n_q"],
            "n_kv": case["n_kv"],
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
        out_triton = _run_triton_once(args)
        expected = (case["num_tokens"], args["n_total"])
        if tuple(out_triton.shape) != expected:
            raise RuntimeError(
                f"Unexpected Triton output shape: got {tuple(out_triton.shape)}, expected {expected}"
            )

        out = _run_cutlass_once(args)
        if tuple(out.shape) != expected:
            raise RuntimeError(
                f"Unexpected CUTLASS output shape: got {tuple(out.shape)}, expected {expected}"
            )
        max_abs = (out.float() - out_triton.float()).abs().max().item()
        print(
            f"Sanity check passed ({label}): shapes OK, "
            f"max |CUTLASS - Triton| = {max_abs:.4e} "
            f"(fp16, Nq={case['n_q']}, Nkv={case['n_kv']})."
        )


def print_summary(title: str = "QKV LoRA-B Forward Benchmark Results"):
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
    parser = argparse.ArgumentParser(description="Benchmark qkv_lora_b_fwd on XPU")
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
