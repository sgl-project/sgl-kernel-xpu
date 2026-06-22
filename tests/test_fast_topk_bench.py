"""Accuracy + performance comparison: torch.topk reference vs SYCL fast_topk family.

Run:
    pytest tests/test_fast_topk_bench.py -v -s
or, for just the benchmark table:
    python tests/test_fast_topk_bench.py
"""

from typing import Optional, Tuple

import pytest
import torch
import utils
from sgl_kernel import (
    fast_topk_transform_fused,
    fast_topk_transform_ragged_fused,
    fast_topk_v2,
)

device = utils.get_device()
MAX_SEQ_LEN = 131072
TOPK = 2048


# -----------------------------------------------------------------------------
# Reference (torch) implementations -- the "original kernel" baseline
# -----------------------------------------------------------------------------
def _ref_topk(
    score: torch.Tensor,
    seq_len: int,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if row_starts is None:
        return torch.topk(score[:, :seq_len], topk, dim=-1, sorted=False).indices
    out = []
    starts = row_starts.tolist()
    for i, s in enumerate(starts):
        out.append(score[i, s : s + seq_len].unsqueeze(0))
    sliced = torch.cat(out, dim=0)
    return torch.topk(sliced, topk, dim=-1, sorted=False).indices


def _ref_topk_transform_decode(
    score: torch.Tensor,
    seq_len: int,
    src_page_table: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    bs = score.shape[0]
    indices = _ref_topk(score, seq_len, topk, row_starts=row_starts)
    out = torch.empty((bs, topk), dtype=torch.int32, device=score.device)
    for i in range(bs):
        out[i] = src_page_table[i, indices[i]]
    return out


def _ref_topk_transform_ragged(
    score: torch.Tensor,
    seq_len: int,
    topk_indices_offset: torch.Tensor,
    topk: int,
    row_starts: torch.Tensor,
) -> torch.Tensor:
    indices = _ref_topk(score, seq_len, topk, row_starts=row_starts)
    mask = indices != -1
    return torch.where(mask, indices + topk_indices_offset.unsqueeze(1), indices)


# -----------------------------------------------------------------------------
# Accuracy comparison helper (set-equality with tolerance for ties)
# -----------------------------------------------------------------------------
def _assert_topk_equal(
    score: torch.Tensor,
    ref: torch.Tensor,
    our: torch.Tensor,
    *,
    offset: Optional[torch.Tensor] = None,
    max_permit_error: int = 5,
) -> None:
    """Match CUDA's test_topk.py semantics: total mismatches across all rows
    must be <= max_permit_error (after accepting true score-equal ties).

    The radix top-k kernel is *exact* for unique-valued inputs, so 5 is a
    generous bound covering the few same-bucket ties that randn can produce.
    """
    bs = ref.shape[0]
    ref_cpu = ref.cpu().tolist()
    our_cpu = our.cpu().tolist()
    wrong = 0
    for i in range(bs):
        rset = set(ref_cpu[i])
        oset = set(our_cpu[i])
        more = oset - rset
        less = rset - oset
        if not more and not less:
            continue
        off = offset[i].item() if offset is not None else 0
        more_vals = sorted(score[i, idx - off].item() for idx in more)
        less_vals = sorted(score[i, idx - off].item() for idx in less)
        if more_vals != less_vals:
            wrong += len(more)
    assert wrong <= max_permit_error, (
        f"too many mismatches: {wrong} (limit {max_permit_error}); "
        f"likely a real SYCL kernel correctness bug, not tie ambiguity"
    )


# -----------------------------------------------------------------------------
# Generic XPU/CUDA event-based timer
# -----------------------------------------------------------------------------
def _sync():
    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _event():
    if device.type == "xpu":
        return torch.xpu.Event(enable_timing=True)
    return torch.cuda.Event(enable_timing=True)


def _bench(fn, *, warmup: int = 5, iters: int = 20) -> float:
    """Return median latency in milliseconds."""
    for _ in range(warmup):
        fn()
    _sync()
    times = []
    for _ in range(iters):
        start = _event()
        end = _event()
        start.record()
        fn()
        end.record()
        _sync()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


# =============================================================================
# Accuracy tests (lighter sweep than test_topk.py — meant to be a smoke check)
# =============================================================================
@pytest.mark.parametrize("bs", [1, 32, 256])
@pytest.mark.parametrize("seq_len", [2048, 4096, 16384])
@pytest.mark.parametrize("has_row_starts", [False, True])
@torch.inference_mode()
def test_acc_fast_topk_v2(bs: int, seq_len: int, has_row_starts: bool) -> None:
    torch.manual_seed(0)
    score = torch.randn(bs, MAX_SEQ_LEN, dtype=torch.float32, device=device)
    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device=device)
    row_starts = (
        torch.randint(0, 2048, (bs,), dtype=torch.int32, device=device)
        if has_row_starts
        else None
    )

    ref = _ref_topk(score, seq_len, TOPK, row_starts=row_starts)
    our = fast_topk_v2(score, lengths, TOPK, row_starts=row_starts)
    ref = torch.sort(ref, dim=-1).values
    our = torch.sort(our, dim=-1).values
    _assert_topk_equal(score, ref, our)


@pytest.mark.parametrize("bs", [1, 32, 256])
@pytest.mark.parametrize("seq_len", [2048, 4096, 16384])
@pytest.mark.parametrize("mode", ["extend", "decode", "target_verify"])
@torch.inference_mode()
def test_acc_fast_topk_transform_fused(bs: int, seq_len: int, mode: str) -> None:
    torch.manual_seed(0)
    if mode == "decode":
        step = 1
    else:
        step = 4 if bs % 4 == 0 else 1
    num_tokens = bs
    bs = bs // step

    row_starts = (
        torch.randint(0, 2048, (bs,), dtype=torch.int32, device=device)
        if mode == "extend"
        else None
    )
    score = torch.randn(bs, MAX_SEQ_LEN, dtype=torch.float32, device=device)
    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.arange(
        0, num_tokens + 1, step=step, dtype=torch.int32, device=device
    )
    src_page_table = (
        torch.arange(0, seq_len, dtype=torch.int32, device=device)
        .unsqueeze(0)
        .expand(bs, -1)
    )

    ref = _ref_topk_transform_decode(
        score, seq_len, src_page_table, TOPK, row_starts=row_starts
    )
    our = fast_topk_transform_fused(
        score=score,
        lengths=lengths,
        page_table_size_1=src_page_table,
        cu_seqlens_q=cu_seqlens_q,
        topk=TOPK,
        row_starts=row_starts,
    )
    ref = torch.sort(ref, dim=-1).values
    our = torch.sort(our, dim=-1).values
    _assert_topk_equal(score, ref, our)


@pytest.mark.parametrize("bs", [1, 32, 256])
@pytest.mark.parametrize("seq_len", [2048, 4096, 16384])
@pytest.mark.parametrize("has_row_starts", [False, True])
@torch.inference_mode()
def test_acc_fast_topk_transform_ragged(
    bs: int, seq_len: int, has_row_starts: bool
) -> None:
    torch.manual_seed(0)
    score = torch.randn(bs, MAX_SEQ_LEN, dtype=torch.float32, device=device)
    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device=device)
    row_starts = (
        torch.randint(0, 2048, (bs,), dtype=torch.int32, device=device)
        if has_row_starts
        else None
    )
    topk_indices_offset = torch.randint(
        0, 1024, (bs,), dtype=torch.int32, device=device
    )

    ref = _ref_topk_transform_ragged(
        score, seq_len, topk_indices_offset, TOPK, row_starts=row_starts
    )
    our = fast_topk_transform_ragged_fused(
        score=score,
        lengths=lengths,
        topk_indices_offset=topk_indices_offset,
        topk=TOPK,
        row_starts=row_starts,
    )
    ref = torch.sort(ref, dim=-1).values
    our = torch.sort(our, dim=-1).values
    _assert_topk_equal(score, ref, our, offset=topk_indices_offset)


# =============================================================================
# Performance benchmarks (use -s to see the table)
# =============================================================================
_PERF_SHAPES = [
    # (bs, seq_len)
    (1, 2048),
    (1, 16384),
    (1, 65536),
    (32, 4096),
    (32, 16384),
    (132, 4096),
    (132, 16384),
    (256, 4096),
    (256, 16384),
    (4096, 4096),
]


def _print_row(name: str, bs: int, sl: int, t_ref: float, t_our: float) -> None:
    speedup = t_ref / t_our if t_our > 0 else float("inf")
    print(
        f"  {name:<32s} bs={bs:<5d} seq_len={sl:<6d}  "
        f"ref={t_ref:8.3f} ms   sycl={t_our:8.3f} ms   speedup={speedup:6.2f}x"
    )


@pytest.mark.parametrize("bs,seq_len", _PERF_SHAPES)
@torch.inference_mode()
def test_perf_fast_topk_v2(bs: int, seq_len: int) -> None:
    torch.manual_seed(0)
    score = torch.randn(bs, MAX_SEQ_LEN, dtype=torch.float32, device=device)
    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device=device)

    t_ref = _bench(lambda: _ref_topk(score, seq_len, TOPK))
    t_our = _bench(lambda: fast_topk_v2(score, lengths, TOPK))
    _print_row("fast_topk_v2", bs, seq_len, t_ref, t_our)


@pytest.mark.parametrize("bs,seq_len", _PERF_SHAPES)
@torch.inference_mode()
def test_perf_fast_topk_transform_fused(bs: int, seq_len: int) -> None:
    torch.manual_seed(0)
    score = torch.randn(bs, MAX_SEQ_LEN, dtype=torch.float32, device=device)
    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.arange(0, bs + 1, dtype=torch.int32, device=device)
    src_page_table = (
        torch.arange(0, seq_len, dtype=torch.int32, device=device)
        .unsqueeze(0)
        .expand(bs, -1)
        .contiguous()
    )

    t_ref = _bench(
        lambda: _ref_topk_transform_decode(score, seq_len, src_page_table, TOPK)
    )
    t_our = _bench(
        lambda: fast_topk_transform_fused(
            score=score,
            lengths=lengths,
            page_table_size_1=src_page_table,
            cu_seqlens_q=cu_seqlens_q,
            topk=TOPK,
        )
    )
    _print_row("fast_topk_transform_fused", bs, seq_len, t_ref, t_our)


@pytest.mark.parametrize("bs,seq_len", _PERF_SHAPES)
@torch.inference_mode()
def test_perf_fast_topk_transform_ragged(bs: int, seq_len: int) -> None:
    torch.manual_seed(0)
    score = torch.randn(bs, MAX_SEQ_LEN, dtype=torch.float32, device=device)
    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device=device)
    row_starts = torch.zeros((bs,), dtype=torch.int32, device=device)
    topk_indices_offset = torch.randint(
        0, 1024, (bs,), dtype=torch.int32, device=device
    )

    t_ref = _bench(
        lambda: _ref_topk_transform_ragged(
            score, seq_len, topk_indices_offset, TOPK, row_starts=row_starts
        )
    )
    t_our = _bench(
        lambda: fast_topk_transform_ragged_fused(
            score=score,
            lengths=lengths,
            topk_indices_offset=topk_indices_offset,
            topk=TOPK,
            row_starts=row_starts,
        )
    )
    _print_row("fast_topk_transform_ragged", bs, seq_len, t_ref, t_our)


# -----------------------------------------------------------------------------
# Standalone runner: prints a benchmark table when executed directly.
# -----------------------------------------------------------------------------
def _main() -> None:
    print(f"\nDevice: {device}")
    print(f"\n[fast_topk_v2]")
    for bs, sl in _PERF_SHAPES:
        test_perf_fast_topk_v2(bs, sl)
    print(f"\n[fast_topk_transform_fused]")
    for bs, sl in _PERF_SHAPES:
        test_perf_fast_topk_transform_fused(bs, sl)
    print(f"\n[fast_topk_transform_ragged_fused]")
    for bs, sl in _PERF_SHAPES:
        test_perf_fast_topk_transform_ragged(bs, sl)


if __name__ == "__main__":
    _main()
