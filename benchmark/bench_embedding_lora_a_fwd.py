import argparse
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
import triton
import triton.language as tl
from sgl_kernel.lora import embedding_lora_a_fwd

all_results = []

DEFAULT_CASES: List[Dict[str, int]] = [
    {
        "num_tokens": 384,
        "num_segments": 1,
        "num_loras": 2,
        "max_rank": 48,
        "vocab_size": 36864,
        "num_extra_tokens": 0,
        "extra_token_ratio_percent": 0,
        "out_of_range_ratio_percent": 0,
    },
    {
        "num_tokens": 512,
        "num_segments": 1,
        "num_loras": 4,
        "max_rank": 64,
        "vocab_size": 40960,
        "num_extra_tokens": 0,
        "extra_token_ratio_percent": 0,
        "out_of_range_ratio_percent": 0,
    },
    {
        "num_tokens": 3072,
        "num_segments": 2,
        "num_loras": 4,
        "max_rank": 77,
        "vocab_size": 49152,
        "num_extra_tokens": 0,
        "extra_token_ratio_percent": 0,
        "out_of_range_ratio_percent": 0,
    },
    {
        "num_tokens": 4608,
        "num_segments": 8,
        "num_loras": 2,
        "max_rank": 96,
        "vocab_size": 57344,
        "num_extra_tokens": 0,
        "extra_token_ratio_percent": 0,
        "out_of_range_ratio_percent": 0,
    },
    {
        "num_tokens": 6144,
        "num_segments": 4,
        "num_loras": 4,
        "max_rank": 123,
        "vocab_size": 65536,
        "num_extra_tokens": 0,
        "extra_token_ratio_percent": 0,
        "out_of_range_ratio_percent": 0,
    },
    {
        "num_tokens": 8192,
        "num_segments": 16,
        "num_loras": 8,
        "max_rank": 128,
        "vocab_size": 73728,
        "num_extra_tokens": 0,
        "extra_token_ratio_percent": 0,
        "out_of_range_ratio_percent": 0,
    },
    {
        "num_tokens": 9216,
        "num_segments": 32,
        "num_loras": 4,
        "max_rank": 175,
        "vocab_size": 81920,
        "num_extra_tokens": 0,
        "extra_token_ratio_percent": 0,
        "out_of_range_ratio_percent": 0,
    },
    {
        "num_tokens": 10240,
        "num_segments": 8,
        "num_loras": 4,
        "max_rank": 192,
        "vocab_size": 98304,
        "num_extra_tokens": 0,
        "extra_token_ratio_percent": 0,
        "out_of_range_ratio_percent": 0,
    },
    {
        "num_tokens": 11264,
        "num_segments": 32,
        "num_loras": 2,
        "max_rank": 221,
        "vocab_size": 65536,
        "num_extra_tokens": 0,
        "extra_token_ratio_percent": 0,
        "out_of_range_ratio_percent": 0,
    },
    {
        "num_tokens": 12288,
        "num_segments": 64,
        "num_loras": 8,
        "max_rank": 256,
        "vocab_size": 86016,
        "num_extra_tokens": 0,
        "extra_token_ratio_percent": 0,
        "out_of_range_ratio_percent": 0,
    },
    {
        "num_tokens": 14336,
        "num_segments": 16,
        "num_loras": 4,
        "max_rank": 320,
        "vocab_size": 98304,
        "num_extra_tokens": 0,
        "extra_token_ratio_percent": 0,
        "out_of_range_ratio_percent": 0,
    },
    {
        "num_tokens": 15872,
        "num_segments": 64,
        "num_loras": 2,
        "max_rank": 365,
        "vocab_size": 110592,
        "num_extra_tokens": 0,
        "extra_token_ratio_percent": 0,
        "out_of_range_ratio_percent": 0,
    },
    {
        "num_tokens": 16384,
        "num_segments": 32,
        "num_loras": 4,
        "max_rank": 448,
        "vocab_size": 122880,
        "num_extra_tokens": 0,
        "extra_token_ratio_percent": 0,
        "out_of_range_ratio_percent": 0,
    },
    {
        "num_tokens": 18432,
        "num_segments": 64,
        "num_loras": 2,
        "max_rank": 512,
        "vocab_size": 114688,
        "num_extra_tokens": 0,
        "extra_token_ratio_percent": 0,
        "out_of_range_ratio_percent": 0,
    },
    {
        "num_tokens": 6144,
        "num_segments": 2,
        "num_loras": 4,
        "max_rank": 69,
        "vocab_size": 53248,
        "num_extra_tokens": 0,
        "extra_token_ratio_percent": 0,
        "out_of_range_ratio_percent": 0,
    },
    {
        "num_tokens": 10240,
        "num_segments": 4,
        "num_loras": 2,
        "max_rank": 144,
        "vocab_size": 69632,
        "num_extra_tokens": 0,
        "extra_token_ratio_percent": 0,
        "out_of_range_ratio_percent": 0,
    },
    {
        "num_tokens": 14336,
        "num_segments": 32,
        "num_loras": 2,
        "max_rank": 192,
        "vocab_size": 86016,
        "num_extra_tokens": 0,
        "extra_token_ratio_percent": 0,
        "out_of_range_ratio_percent": 0,
    },
    {
        "num_tokens": 16384,
        "num_segments": 16,
        "num_loras": 4,
        "max_rank": 576,
        "vocab_size": 98304,
        "num_extra_tokens": 0,
        "extra_token_ratio_percent": 0,
        "out_of_range_ratio_percent": 0,
    },
    {
        "num_tokens": 19456,
        "num_segments": 64,
        "num_loras": 4,
        "max_rank": 640,
        "vocab_size": 122880,
        "num_extra_tokens": 0,
        "extra_token_ratio_percent": 0,
        "out_of_range_ratio_percent": 0,
    },
    {
        "num_tokens": 5050,
        "num_segments": 1,
        "num_loras": 2,
        "max_rank": 288,
        "vocab_size": 120832,
        "num_extra_tokens": 0,
        "extra_token_ratio_percent": 0,
        "out_of_range_ratio_percent": 0,
    },
    {
        "num_tokens": 7000,
        "num_segments": 1,
        "num_loras": 1,
        "max_rank": 95,
        "vocab_size": 118784,
        "num_extra_tokens": 0,
        "extra_token_ratio_percent": 0,
        "out_of_range_ratio_percent": 0,
    },
    {
        "num_tokens": 9500,
        "num_segments": 2,
        "num_loras": 4,
        "max_rank": 230,
        "vocab_size": 116736,
        "num_extra_tokens": 0,
        "extra_token_ratio_percent": 0,
        "out_of_range_ratio_percent": 0,
    },
    {
        "num_tokens": 12000,
        "num_segments": 4,
        "num_loras": 2,
        "max_rank": 704,
        "vocab_size": 122880,
        "num_extra_tokens": 0,
        "extra_token_ratio_percent": 0,
        "out_of_range_ratio_percent": 0,
    },
    {
        "num_tokens": 13824,
        "num_segments": 8,
        "num_loras": 4,
        "max_rank": 460,
        "vocab_size": 110592,
        "num_extra_tokens": 0,
        "extra_token_ratio_percent": 0,
        "out_of_range_ratio_percent": 0,
    },
    {
        "num_tokens": 16384,
        "num_segments": 16,
        "num_loras": 8,
        "max_rank": 384,
        "vocab_size": 98304,
        "num_extra_tokens": 0,
        "extra_token_ratio_percent": 0,
        "out_of_range_ratio_percent": 0,
    },
    {
        "num_tokens": 21504,
        "num_segments": 32,
        "num_loras": 2,
        "max_rank": 731,
        "vocab_size": 121856,
        "num_extra_tokens": 0,
        "extra_token_ratio_percent": 0,
        "out_of_range_ratio_percent": 0,
    },
]


@triton.jit
def _embedding_lora_a_kernel(
    # Pointers to tensors
    input_ids,
    weights,
    output,
    extra_embeddings,
    # Dimensions
    vocab_size,
    rank,
    num_loras,
    # Strides
    w_stride_0,  # stride for lora index
    w_stride_1,  # stride for rank
    w_stride_2,  # stride for vocab
    output_stride_0,
    output_stride_1,
    extra_emb_stride_0,  # stride for lora index
    extra_emb_stride_1,  # stride for token
    extra_emb_stride_2,  # stride for hidden dim (= rank for extra embeddings)
    # Batch info
    seg_lens,
    seg_indptr,
    weight_indices,
    lora_ranks,
    # Meta-parameters
    BLOCK_RANK: tl.constexpr,
    HAS_EXTRA_EMBEDDINGS: tl.constexpr,
):
    """
    Embedding lookup for LoRA A weights with support for extra tokens.

    Each program handles one token across a block of rank dimensions.
    Grid: (cdiv(max_len, 1), bs) - one program per token in each batch
    """
    batch_id = tl.program_id(axis=1)
    token_idx = tl.program_id(axis=0)

    w_index = tl.load(weight_indices + batch_id)
    rank_val = tl.load(lora_ranks + w_index)

    # If rank is 0, skip
    if rank_val == 0:
        return

    seg_start = tl.load(seg_indptr + batch_id)
    seg_len = tl.load(seg_lens + batch_id)

    # Check if this token is within the segment
    if token_idx >= seg_len:
        return

    # Load the token ID
    token_id = tl.load(input_ids + seg_start + token_idx)

    # Process in chunks of BLOCK_RANK dimensions
    num_blocks = tl.cdiv(rank_val, BLOCK_RANK)

    for block_id in range(num_blocks):
        rank_offset = tl.arange(0, BLOCK_RANK) + block_id * BLOCK_RANK
        rank_mask = rank_offset < rank_val

        # Check if this is an extra token
        is_extra_token = token_id >= vocab_size

        if HAS_EXTRA_EMBEDDINGS and is_extra_token:
            # Use extra embeddings
            extra_token_id = token_id - vocab_size
            extra_emb_ptr = (
                extra_embeddings
                + w_index * extra_emb_stride_0
                + extra_token_id * extra_emb_stride_1
                + rank_offset * extra_emb_stride_2
            )
            emb_values = tl.load(extra_emb_ptr, mask=rank_mask, other=0.0)
        else:
            # Use regular LoRA A weights
            # weights shape: (num_loras, rank, vocab_size)
            # We need to load weights[w_index, rank_offset, token_id]
            token_id_clamped = tl.minimum(token_id, vocab_size - 1)
            weight_ptr = (
                weights
                + w_index * w_stride_0
                + rank_offset * w_stride_1
                + token_id_clamped * w_stride_2
            )
            emb_values = tl.load(weight_ptr, mask=rank_mask, other=0.0)

        # Write to output
        output_ptr = (
            output
            + (seg_start + token_idx) * output_stride_0
            + rank_offset * output_stride_1
        )
        tl.store(output_ptr, emb_values, mask=rank_mask)


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


def _compute_rank_sum_by_segment(
    seg_lens: torch.Tensor,
    weight_indices: torch.Tensor,
    lora_ranks: torch.Tensor,
) -> int:
    seg_lens_cpu = seg_lens.to("cpu")
    weight_indices_cpu = weight_indices.to("cpu")
    lora_ranks_cpu = lora_ranks.to("cpu")

    total_rank = 0
    for seg_idx in range(weight_indices_cpu.numel()):
        seg_len = int(seg_lens_cpu[seg_idx].item())
        lora = int(weight_indices_cpu[seg_idx].item())
        rank = int(lora_ranks_cpu[lora].item())
        total_rank += seg_len * rank
    return total_rank


def _count_token_classes(
    input_ids: torch.Tensor,
    vocab_size: int,
    num_extra_tokens: int,
) -> Tuple[int, int, int]:
    ids = input_ids.to("cpu")
    base = int((ids < vocab_size).sum().item())
    if num_extra_tokens > 0:
        extra_hi = vocab_size + num_extra_tokens
        extra = int(((ids >= vocab_size) & (ids < extra_hi)).sum().item())
    else:
        extra = 0
    oor = ids.numel() - base - extra
    return base, extra, oor


def _estimate_bytes(
    num_tokens: int,
    num_segments: int,
    max_rank: int,
    max_len: int,
    elem_size: int,
    rank_sum: int,
) -> float:
    """Memory traffic estimate for Triton launch semantics."""
    launched_programs = max_len * num_segments
    bytes_per_program_meta = 4 + 4 + 4 + 4
    bytes_metadata = launched_programs * bytes_per_program_meta

    bytes_input_ids = num_tokens * 8
    bytes_embedding_reads = rank_sum * elem_size
    bytes_out = num_tokens * max_rank * elem_size

    return bytes_metadata + bytes_input_ids + bytes_embedding_reads + bytes_out


def calc_metrics(total_bytes: float, time_ms: float) -> Dict[str, float]:
    time_s = time_ms / 1e3
    if time_s <= 0:
        raise RuntimeError("Measured time must be > 0")

    bandwidth_gbs = (total_bytes / 1e9) / time_s
    total_bytes_mb = total_bytes / 1e6

    return {
        "total_bytes": total_bytes,
        "total_bytes_mb": total_bytes_mb,
        "bandwidth_gbs": bandwidth_gbs,
    }


def _make_inputs(
    case: Dict[str, int], dtype: torch.dtype, device: torch.device
) -> Dict[str, Any]:
    num_tokens = case["num_tokens"]
    num_segments = min(case["num_segments"], num_tokens)
    num_loras = case["num_loras"]
    max_rank = case["max_rank"]
    vocab_size = case["vocab_size"]
    num_extra_tokens = case["num_extra_tokens"]
    extra_ratio = case["extra_token_ratio_percent"]
    out_of_range_ratio = case["out_of_range_ratio_percent"]

    input_ids = torch.randint(
        0, vocab_size, (num_tokens,), dtype=torch.int64, device=device
    )
    if num_extra_tokens > 0 and extra_ratio > 0:
        num_extra = num_tokens * extra_ratio // 100
        num_extra = min(num_extra, num_tokens)
        if num_extra > 0:
            idx = torch.randperm(num_tokens, device=device)[:num_extra]
            input_ids[idx] = vocab_size + torch.randint(
                0, num_extra_tokens, (num_extra,), dtype=torch.int64, device=device
            )

        num_oor = num_tokens * out_of_range_ratio // 100
        num_oor = min(num_oor, num_tokens)
        if num_oor > 0:
            idx_oor = torch.randperm(num_tokens, device=device)[:num_oor]
            input_ids[idx_oor] = (
                vocab_size
                + num_extra_tokens
                + torch.randint(0, 16, (num_oor,), dtype=torch.int64, device=device)
            )

    weights = torch.randn(num_loras, max_rank, vocab_size, dtype=dtype, device=device)
    extra_embeddings = None
    if num_extra_tokens > 0:
        extra_embeddings = torch.randn(
            num_loras, num_extra_tokens, max_rank, dtype=dtype, device=device
        )

    seg_indptr = _build_seg_indptr(num_tokens, num_segments, device)
    seg_lens = _build_seg_lens(seg_indptr)
    weight_indices = torch.randint(
        0, num_loras, (num_segments,), dtype=torch.int32, device=device
    )
    lora_ranks = torch.tensor([max_rank] * num_loras, dtype=torch.int32, device=device)

    return {
        "input_ids": input_ids,
        "weights": weights,
        "vocab_size": vocab_size,
        "seg_indptr": seg_indptr,
        "seg_lens": seg_lens,
        "weight_indices": weight_indices,
        "lora_ranks": lora_ranks,
        "extra_embeddings": extra_embeddings,
    }


def _run_sgl_once(args: Dict[str, Any]):
    return embedding_lora_a_fwd(
        input_ids=args["input_ids"],
        weights=args["weights"],
        vocab_size=int(args["vocab_size"]),
        seg_indptr=args["seg_indptr"],
        weight_indices=args["weight_indices"],
        lora_ranks=args["lora_ranks"],
        extra_embeddings=args["extra_embeddings"],
        seg_lens=args["seg_lens"],
    )


def _run_triton_once(args: Dict[str, Any]):
    input_ids = args["input_ids"]
    weights = args["weights"]
    seg_lens = args["seg_lens"]
    seg_indptr = args["seg_indptr"]
    weight_indices = args["weight_indices"]
    lora_ranks = args["lora_ranks"]
    extra_embeddings = args["extra_embeddings"]
    vocab_size = int(args["vocab_size"])

    rank = weights.shape[1]
    max_len = int(seg_lens.max().item()) if seg_lens.numel() > 0 else 0
    bs = int(seg_lens.numel())

    has_extra_embeddings = extra_embeddings is not None
    if has_extra_embeddings:
        extra_embeddings = extra_embeddings.contiguous()
        extra_emb_stride = (
            extra_embeddings.stride(0),
            extra_embeddings.stride(1),
            extra_embeddings.stride(2),
        )
    else:
        extra_embeddings = torch.empty(
            (1, 1, 1), device=input_ids.device, dtype=weights.dtype
        )
        extra_emb_stride = (1, 1, 1)

    output = torch.zeros(
        (input_ids.shape[0], rank), device=input_ids.device, dtype=weights.dtype
    )
    if max_len == 0 or bs == 0:
        return output

    grid = (max_len, bs)
    if has_extra_embeddings:
        _embedding_lora_a_kernel[grid](
            input_ids,
            weights,
            output,
            extra_embeddings,
            vocab_size,
            rank,
            weights.shape[0],
            weights.stride(0),
            weights.stride(1),
            weights.stride(2),
            output.stride(0),
            output.stride(1),
            extra_emb_stride[0],
            extra_emb_stride[1],
            extra_emb_stride[2],
            seg_lens,
            seg_indptr,
            weight_indices,
            lora_ranks,
            BLOCK_RANK=128,  # pyright: ignore[reportArgumentType]
            HAS_EXTRA_EMBEDDINGS=True,  # pyright: ignore[reportArgumentType]
        )
    else:
        _embedding_lora_a_kernel[grid](
            input_ids,
            weights,
            output,
            extra_embeddings,
            vocab_size,
            rank,
            weights.shape[0],
            weights.stride(0),
            weights.stride(1),
            weights.stride(2),
            output.stride(0),
            output.stride(1),
            extra_emb_stride[0],
            extra_emb_stride[1],
            extra_emb_stride[2],
            seg_lens,
            seg_indptr,
            weight_indices,
            lora_ranks,
            BLOCK_RANK=128,  # pyright: ignore[reportArgumentType]
            HAS_EXTRA_EMBEDDINGS=False,  # pyright: ignore[reportArgumentType]
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
        f"r={case['max_rank']},vocab={case['vocab_size']},extra={case['num_extra_tokens']}"
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
            "sgl_fp32",
            "triton_fp32",
        ],
        line_names=[
            "SGL fp16",
            "Triton fp16",
            "SGL bf16",
            "Triton bf16",
            "SGL fp32",
            "Triton fp32",
        ],
        styles=[
            ("green", "-"),
            ("green", "--"),
            ("blue", "-"),
            ("blue", "--"),
            ("orange", "-"),
            ("orange", "--"),
        ],
        ylabel="GB/s",
        plot_name="embedding-lora-a-fwd-sgl-vs-triton",
        args={},
    )
)
def benchmark(case_id, provider):
    device = torch.device("xpu")
    backend, dtype_name = provider.split("_", 1)
    dtype = _dtype_from_provider(dtype_name)

    case = CASES[case_id]
    inputs = _make_inputs(case, dtype, device)
    rank_sum = _compute_rank_sum_by_segment(
        inputs["seg_lens"],
        inputs["weight_indices"],
        inputs["lora_ranks"],
    )
    _, num_extra_in_range, _ = _count_token_classes(
        inputs["input_ids"],
        inputs["vocab_size"],
        case["num_extra_tokens"],
    )
    max_len = (
        int(inputs["seg_lens"].max().item()) if inputs["seg_lens"].numel() > 0 else 0
    )
    num_segments = min(case["num_segments"], case["num_tokens"])
    elem_size = torch.tensor([], dtype=dtype).element_size()

    total_bytes = _estimate_bytes(
        num_tokens=case["num_tokens"],
        num_segments=num_segments,
        max_rank=case["max_rank"],
        max_len=max_len,
        elem_size=elem_size,
        rank_sum=rank_sum,
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

    # metrics = calc_metrics(total_bytes_sgl if backend == "sgl" else total_bytes_triton, ms)
    metrics = calc_metrics(total_bytes, ms)

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
            "bandwidth_gbs": metrics["bandwidth_gbs"],
            "total_bytes_mb": metrics["total_bytes_mb"],
            "max_len": max_len,
            "rank_sum": rank_sum,
            "extra_tokens_in_range": num_extra_in_range,
            "num_tokens": case["num_tokens"],
            "num_segments": case["num_segments"],
            "num_loras": case["num_loras"],
            "max_rank": case["max_rank"],
            "vocab_size": case["vocab_size"],
        }
    )

    gbps = lambda t_ms: metrics["total_bytes"] * 1e-9 / (t_ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


def _sanity_check() -> None:
    torch.manual_seed(123)
    device = torch.device("xpu")
    case = DEFAULT_CASES[0]
    args = _make_inputs(case, torch.float32, device)
    out = _run_sgl_once(args)
    out_triton = _run_triton_once(args)
    if out.shape != (case["num_tokens"], case["max_rank"]):
        raise RuntimeError(
            f"Unexpected output shape: got {tuple(out.shape)}, expected {(case['num_tokens'], case['max_rank'])}"
        )
    if out_triton.shape != out.shape:
        raise RuntimeError(
            f"Unexpected Triton output shape: got {tuple(out_triton.shape)}, expected {tuple(out.shape)}"
        )
    print("Sanity check passed: output shape looks correct.")


def print_summary(title: str = "Embedding LoRA Benchmark Results"):
    """Print detailed benchmark results in tabular format."""
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)

    if not all_results:
        print("No results collected.")
        return

    df = pd.DataFrame(all_results)

    # Round numeric columns for display
    for col in ["time_us", "bandwidth_gbs", "total_bytes_mb"]:
        if col in df.columns:
            df[col] = df[col].round(2)

    # Select key columns for display
    display_cols = [
        col
        for col in [
            "case_id",
            "case_label",
            "provider",
            "time_us",
            "bandwidth_gbs",
            "total_bytes_mb",
        ]
        if col in df.columns
    ]

    print("\nDetailed Results:")
    print(df[display_cols].to_string(index=False))

    # Print summary statistics by provider
    if "provider" in df.columns and "bandwidth_gbs" in df.columns:
        print("\n" + "=" * 120)
        print("Summary Statistics by Provider")
        print("=" * 120)
        summary = df.groupby("provider")[["bandwidth_gbs", "time_us"]].agg(
            ["mean", "min", "max", "std"]
        )
        print(summary.to_string())


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark embedding_lora_a_fwd on XPU"
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
