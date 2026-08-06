import sys
from typing import Any, Callable, Dict, Optional, Tuple

import pytest
import torch
import torch.nn.functional as F
from sgl_kernel import (
    fast_topk_transform_fused,
    fast_topk_transform_ragged_fused,
    fast_topk_v2,
    topk_transform_512,
    topk_transform_512_v2,
)


def _ref_torch_impl(
    score: torch.Tensor,
    seq_len: int,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert score.dim() == 2
    if row_starts is None:
        return torch.topk(score[:, :seq_len], topk, dim=-1, sorted=False).indices
    else:
        ks = row_starts.cpu().tolist()
        ke = (row_starts + seq_len).tolist()
        scores = []
        for i, (start, end) in enumerate(zip(ks, ke)):
            scores.append(score[i, start:end].unsqueeze(0))
        score = torch.cat(scores, dim=0)
        return torch.topk(score, topk, dim=-1, sorted=False).indices


def _ref_torch_transform_decode_impl(
    score: torch.Tensor,
    seq_len: int,
    src_page_table: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    batch_size, _ = score.shape
    assert score.shape[0] == src_page_table.shape[0]
    assert seq_len >= topk
    indices = _ref_torch_impl(score, seq_len, topk, row_starts=row_starts)
    topk_indices = torch.empty(
        (batch_size, topk), dtype=torch.int32, device=score.device
    )
    for i in range(batch_size):
        topk_indices[i] = src_page_table[i, indices[i]]
    return topk_indices


def _ref_torch_transform_ragged_impl(
    score: torch.Tensor,
    seq_len: int,
    topk_indices_offset: torch.Tensor,
    topk: int,
    row_starts: torch.Tensor,
) -> torch.Tensor:
    assert score.shape[0] == topk_indices_offset.shape[0]
    assert seq_len >= topk
    indices = _ref_torch_impl(score, seq_len, topk, row_starts=row_starts)

    mask = indices != -1
    topk_indices_offset = topk_indices_offset.unsqueeze(1)
    return torch.where(mask, indices + topk_indices_offset, indices)


def assert_equal(
    score: torch.Tensor,
    indices_ref: torch.Tensor,
    indices_our: torch.Tensor,
    bs: int,
    k: int,
    seq_len: int,
    topk_indices_offset: Optional[torch.Tensor] = None,
    row_starts: Optional[torch.Tensor] = None,
    max_permit_error: int = 0,
):
    indices_our_cpu = indices_our.cpu().tolist()
    indices_ref_cpu = indices_ref.cpu().tolist()

    wrong_values = 0
    for i in range(bs):
        indices_ref_set_i = set(indices_ref_cpu[i])
        indices_our_set_i = set(indices_our_cpu[i])
        more = indices_our_set_i - indices_ref_set_i
        less = indices_ref_set_i - indices_our_set_i
        offset = topk_indices_offset[i].item() if topk_indices_offset is not None else 0
        row_start = row_starts[i].item() if row_starts is not None else 0
        if len(more) > 0 or len(less) > 0:
            more_values = sorted(
                score[i, idx - offset + row_start].item() for idx in more
            )
            less_values = sorted(
                score[i, idx - offset + row_start].item() for idx in less
            )
            if more_values != less_values:
                wrong_values += len(more)
                print(
                    f"{bs=}, {k=}, {seq_len=}, {i=}, {more=}, {less=} failed, with {more_values=}, {less_values=}"
                )
        assert wrong_values <= max_permit_error, f"{wrong_values=}, {max_permit_error=}"


def _bench(fn, *, warmup: int = 5, iters: int = 20) -> float:
    """Return median latency in milliseconds."""
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
    times.sort()
    return times[len(times) // 2]


def _setup_fast_topk_v2(bs: int, seq_len: int, has_row_starts: bool):
    torch.manual_seed(42)

    stream = torch.xpu.Stream()
    torch.xpu.set_stream(stream)
    score = torch.randn(
        bs,
        seq_len + (2048 if has_row_starts else 0),
        dtype=torch.float32,
        device="xpu",
    )
    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device="xpu")

    if has_row_starts:
        row_starts = torch.randint(0, 2048, (bs,), dtype=torch.int32, device="xpu")
    else:
        row_starts = None

    return score, lengths, row_starts


def _setup_fast_topk_transform_fused(bs: int, seq_len: int, mode: str):
    torch.manual_seed(42)

    stream = torch.xpu.Stream()
    torch.xpu.set_stream(stream)

    # NOTE: for decode, cumulative seqlens_q is just 0..=bs
    # NOTE: since page table is arange, they equal topk indices
    if mode == "decode":
        step = 1
    else:
        step = 4 if bs % 4 == 0 else 1
    num_tokens = bs
    bs = bs // step

    if mode == "extend":
        row_starts = torch.randint(0, 2048, (bs,), dtype=torch.int32, device="xpu")
    else:
        row_starts = None

    score = torch.randn(
        bs,
        seq_len + (2048 if row_starts is not None else 0),
        dtype=torch.float32,
        device="xpu",
    )
    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device="xpu")
    cu_seqlens_q = torch.arange(
        0, num_tokens + 1, step=step, dtype=torch.int32, device="xpu"
    )
    src_page_table = torch.arange(0, seq_len, dtype=torch.int32, device="xpu")
    src_page_table = src_page_table.unsqueeze(0).expand(bs, -1)

    return bs, score, lengths, row_starts, cu_seqlens_q, src_page_table


def _setup_fast_topk_transform_ragged(bs: int, seq_len: int, has_row_starts: bool):
    # Used in prefill only
    torch.manual_seed(42)

    stream = torch.xpu.Stream()
    torch.xpu.set_stream(stream)
    # bs: # of q tokens
    score = torch.randn(
        bs,
        seq_len + (2048 if has_row_starts else 0),
        dtype=torch.float32,
        device="xpu",
    )
    # kv_len
    if has_row_starts:
        row_starts = torch.randint(0, 2048, (bs,), dtype=torch.int32, device="xpu")
    else:
        row_starts = None
    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device="xpu")
    topk_indices_offset = torch.randint(0, 1024, (bs,), dtype=torch.int32, device="xpu")

    return score, lengths, row_starts, topk_indices_offset


@pytest.mark.parametrize("bs", [1, 132, 256, 4096])
@pytest.mark.parametrize("k", [2048])  # we only support 2048 now
@pytest.mark.parametrize("seq_len", [2048, 4096, 16384, 65536])
@pytest.mark.parametrize("has_row_starts", [True, False])
@torch.inference_mode()
def test_fast_topk_v2(bs: int, k: int, seq_len: int, has_row_starts: bool) -> None:
    score, lengths, row_starts = _setup_fast_topk_v2(bs, seq_len, has_row_starts)

    indices_ref = _ref_torch_impl(score, seq_len, k, row_starts=row_starts)
    indices_our = fast_topk_v2(score, lengths, k, row_starts=row_starts)

    # sort and compare
    indices_ref = torch.sort(indices_ref, dim=-1).values
    indices_our = torch.sort(indices_our, dim=-1).values

    # Tests can pass with max_permit_error=3, set to 5 for safety
    assert_equal(
        score,
        indices_ref,
        indices_our,
        bs,
        k,
        seq_len,
        row_starts=row_starts,
        max_permit_error=5,
    )


@pytest.mark.parametrize("bs", [1, 132, 256, 4096])
@pytest.mark.parametrize("k", [2048])  # we only support 2048 now
@pytest.mark.parametrize("seq_len", [2048, 4096, 16384, 65536])
@pytest.mark.parametrize("mode", ["extend", "decode", "target_verify"])
@torch.inference_mode()
def test_fast_topk_transform_fused(bs: int, k: int, seq_len: int, mode: str) -> None:
    bs, score, lengths, row_starts, cu_seqlens_q, src_page_table = (
        _setup_fast_topk_transform_fused(bs, seq_len, mode)
    )

    dst_page_table_ref = _ref_torch_transform_decode_impl(
        score=score,
        seq_len=seq_len,
        src_page_table=src_page_table,
        topk=k,
        row_starts=row_starts,
    )
    dst_page_table_our = fast_topk_transform_fused(
        score=score,
        lengths=lengths,
        page_table_size_1=src_page_table,
        cu_seqlens_q=cu_seqlens_q,
        topk=k,
        row_starts=row_starts,
    )

    # sort and compare
    dst_page_table_our = torch.sort(dst_page_table_our, dim=-1).values
    dst_page_table_ref = torch.sort(dst_page_table_ref, dim=-1).values

    assert_equal(
        score,
        dst_page_table_ref,
        dst_page_table_our,
        bs,
        k,
        seq_len,
        row_starts=row_starts,
        max_permit_error=5,
    )


@pytest.mark.parametrize("bs", [1, 132, 256, 4096])
@pytest.mark.parametrize("k", [2048])  # we only support 2048 now
@pytest.mark.parametrize("seq_len", [2048, 4096, 16384, 65536])
@pytest.mark.parametrize("has_row_starts", [True, False])
@torch.inference_mode()
def test_fast_topk_transform_ragged(
    bs: int, k: int, seq_len: int, has_row_starts: bool
) -> None:
    score, lengths, row_starts, topk_indices_offset = _setup_fast_topk_transform_ragged(
        bs, seq_len, has_row_starts
    )

    dst_page_table_ref = _ref_torch_transform_ragged_impl(
        score=score,
        seq_len=seq_len,
        topk_indices_offset=topk_indices_offset,
        topk=k,
        row_starts=row_starts,
    )
    dst_page_table_our = fast_topk_transform_ragged_fused(
        score=score,
        lengths=lengths,
        topk_indices_offset=topk_indices_offset,
        topk=k,
        row_starts=row_starts,
    )

    # sort and compare
    dst_page_table_our = torch.sort(dst_page_table_our, dim=-1).values
    dst_page_table_ref = torch.sort(dst_page_table_ref, dim=-1).values

    assert_equal(
        score,
        dst_page_table_ref,
        dst_page_table_our,
        bs,
        k,
        seq_len,
        topk_indices_offset,
        row_starts=row_starts,
        max_permit_error=5,
    )


@pytest.mark.parametrize("bs", [132, 256, 4096])
@pytest.mark.parametrize("k", [2048])  # we only support 2048 now
@pytest.mark.parametrize("seq_len", [2048, 4096, 16384, 65536])
@pytest.mark.parametrize("has_row_starts", [True, False])
@torch.inference_mode()
def test_fast_topk_v2_perf(bs: int, k: int, seq_len: int, has_row_starts: bool) -> None:
    score, lengths, row_starts = _setup_fast_topk_v2(bs, seq_len, has_row_starts)

    t_ref = _bench(lambda: _ref_torch_impl(score, seq_len, k, row_starts=row_starts))
    t_our = _bench(lambda: fast_topk_v2(score, lengths, k, row_starts=row_starts))
    assert (
        t_our < t_ref
    ), f"sycl ({t_our:.3f} ms) not faster than torch ({t_ref:.3f} ms)"


@pytest.mark.parametrize("bs", [132, 256, 4096])
@pytest.mark.parametrize("k", [2048])  # we only support 2048 now
@pytest.mark.parametrize("seq_len", [2048, 4096, 16384, 65536])
@pytest.mark.parametrize("mode", ["extend", "decode", "target_verify"])
@torch.inference_mode()
def test_fast_topk_transform_fused_perf(
    bs: int, k: int, seq_len: int, mode: str
) -> None:
    bs, score, lengths, row_starts, cu_seqlens_q, src_page_table = (
        _setup_fast_topk_transform_fused(bs, seq_len, mode)
    )

    t_ref = _bench(
        lambda: _ref_torch_transform_decode_impl(
            score=score,
            seq_len=seq_len,
            src_page_table=src_page_table,
            topk=k,
            row_starts=row_starts,
        )
    )
    t_our = _bench(
        lambda: fast_topk_transform_fused(
            score=score,
            lengths=lengths,
            page_table_size_1=src_page_table,
            cu_seqlens_q=cu_seqlens_q,
            topk=k,
            row_starts=row_starts,
        )
    )
    assert (
        t_our < t_ref
    ), f"sycl ({t_our:.3f} ms) not faster than torch ({t_ref:.3f} ms)"


@pytest.mark.parametrize("bs", [132, 256, 4096])
@pytest.mark.parametrize("k", [2048])  # we only support 2048 now
@pytest.mark.parametrize("seq_len", [2048, 4096, 16384, 65536])
@pytest.mark.parametrize("has_row_starts", [True, False])
@torch.inference_mode()
def test_fast_topk_transform_ragged_perf(
    bs: int, k: int, seq_len: int, has_row_starts: bool
) -> None:
    score, lengths, row_starts, topk_indices_offset = _setup_fast_topk_transform_ragged(
        bs, seq_len, has_row_starts
    )

    t_ref = _bench(
        lambda: _ref_torch_transform_ragged_impl(
            score=score,
            seq_len=seq_len,
            topk_indices_offset=topk_indices_offset,
            topk=k,
            row_starts=row_starts,
        )
    )
    t_our = _bench(
        lambda: fast_topk_transform_ragged_fused(
            score=score,
            lengths=lengths,
            topk_indices_offset=topk_indices_offset,
            topk=k,
            row_starts=row_starts,
        )
    )
    assert (
        t_our < t_ref
    ), f"sycl ({t_our:.3f} ms) not faster than torch ({t_ref:.3f} ms)"


_arange_cache: Dict[str, torch.Tensor] = {}


def _topk_transform_512_vectorized(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: Optional[torch.Tensor] = None,
    topk_op: Callable[..., Tuple[torch.Tensor, torch.Tensor]] = torch.topk,
    topk_op_kwargs: Optional[Dict[str, object]] = None,
    contiguous_topk_input: bool = False,
) -> None:
    TOPK = out_page_indices.shape[1]
    batch_size = scores.shape[0]
    max_seq_len = scores.shape[1]
    device = scores.device

    page_bits = (page_size - 1).bit_length() if page_size > 1 else 0
    page_mask = page_size - 1

    cache = _arange_cache
    key_seq = f"arange_{max_seq_len}_{device}"
    key_topk = f"arange_{TOPK}_{device}"
    key_bs = f"arange_{batch_size}_{device}"
    if key_seq not in cache:
        cache[key_seq] = torch.arange(max_seq_len, device=device)
    if key_topk not in cache:
        cache[key_topk] = torch.arange(TOPK, device=device, dtype=torch.int32)
    if key_bs not in cache:
        cache[key_bs] = torch.arange(batch_size, device=device)

    positions = cache[key_seq].unsqueeze(0).expand(batch_size, -1)

    if (seq_lens == max_seq_len).all():
        masked_scores = scores
    else:
        valid_mask = positions < seq_lens.unsqueeze(1)
        masked_scores = scores.clone()
        masked_scores.masked_fill_(~valid_mask, float("-inf"))

    actual_k = min(TOPK, max_seq_len)
    topk_kwargs = (
        {"dim": 1, "largest": True, "sorted": False}
        if topk_op_kwargs is None
        else topk_op_kwargs
    )
    topk_input = masked_scores.contiguous() if contiguous_topk_input else masked_scores
    _, raw_indices = topk_op(topk_input, actual_k, **topk_kwargs)
    raw_indices = raw_indices.to(torch.int32)

    if actual_k < TOPK:
        raw_indices = F.pad(raw_indices, (0, TOPK - actual_k), value=0)

    batch_indices = cache[key_bs].unsqueeze(1).expand(-1, TOPK)
    gathered_scores = scores[
        batch_indices.flatten(), raw_indices.clamp(min=0).flatten()
    ].view(batch_size, TOPK)

    valid_topk = gathered_scores != float("-inf")
    if actual_k < TOPK:
        pad_mask = cache[key_topk].unsqueeze(0) >= actual_k
        valid_topk = valid_topk & ~pad_mask

    needs_sequential = seq_lens <= TOPK
    sequential_indices = cache[key_topk].unsqueeze(0).expand(batch_size, -1)
    sequential_valid = sequential_indices < seq_lens.unsqueeze(1)

    seq_indices_or_neg1 = sequential_indices.clone()
    seq_indices_or_neg1.masked_fill_(~sequential_valid, -1)

    needs_seq_mask = needs_sequential.unsqueeze(1).expand(-1, TOPK)
    raw_indices = torch.where(needs_seq_mask, seq_indices_or_neg1, raw_indices)
    valid_topk = torch.where(needs_seq_mask, sequential_valid, valid_topk)

    page_idx = raw_indices >> page_bits
    offset_in_page = raw_indices & page_mask

    page_idx_clamped = torch.clamp(page_idx, min=0)
    physical_pages = torch.gather(page_tables, dim=1, index=page_idx_clamped.long())

    page_indices = (physical_pages << page_bits) | offset_in_page
    page_indices = page_indices.to(torch.int32)
    page_indices.masked_fill_(~valid_topk, -1)

    out_page_indices.copy_(page_indices)

    if out_raw_indices is not None:
        raw_indices = raw_indices.clone()
        raw_indices.masked_fill_(~valid_topk, -1)
        out_raw_indices.copy_(raw_indices)


def _ref_torch_topk_transform_512_impl(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:
    """Verbatim copy of sglang's vectorised PyTorch fallback."""
    _topk_transform_512_vectorized(
        scores,
        seq_lens,
        page_tables,
        out_page_indices,
        page_size,
        out_raw_indices,
        topk_op=torch.topk,
        topk_op_kwargs={"dim": 1, "largest": True, "sorted": False},
    )


def _setup_topk_transform_512(
    bs: int,
    seq_len: int,
    page_size: int,
    length_override: Optional[int] = None,
):
    torch.manual_seed(42)
    stream = torch.xpu.Stream()
    torch.xpu.set_stream(stream)

    score = torch.randn(bs, seq_len, dtype=torch.float32, device="xpu")

    length_val = seq_len if length_override is None else length_override
    lengths = torch.full((bs,), length_val, dtype=torch.int32, device="xpu")

    num_pages = (seq_len + page_size - 1) // page_size
    page_table = (
        torch.arange(0, num_pages, dtype=torch.int32, device="xpu")
        .unsqueeze(0)
        .expand(bs, -1)
        .contiguous()
    )
    return score, lengths, page_table


def _ref_topk_transform_512(
    score: torch.Tensor,
    lengths: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
    topk: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bs = score.shape[0]
    device = score.device
    out_pi = torch.full((bs, topk), -1, dtype=torch.int32, device=device)
    out_ri = torch.full((bs, topk), -1, dtype=torch.int32, device=device)
    # The upstream ref reads ``seq_lens`` as an int32 tensor sized like the
    # batch; our callers already pass such a tensor for ``lengths`` so no
    # reshape/dtype coercion is needed.
    _ref_torch_topk_transform_512_impl(
        score, lengths, page_table, out_pi, page_size, out_ri
    )
    return out_pi, out_ri


def _empty_v2_metadata(bs: int) -> torch.Tensor:
    return torch.empty((bs + 1, 2), dtype=torch.int32, device="xpu")


@pytest.mark.parametrize("bs", [1, 132, 256, 4096])
@pytest.mark.parametrize("topk", [512, 1024])
@pytest.mark.parametrize("seq_len", [4096, 16384, 65536])
@pytest.mark.parametrize("page_size", [1, 64, 256])
@pytest.mark.parametrize("with_raw", [False, True])
@torch.inference_mode()
def test_topk_transform_512(
    bs: int, topk: int, seq_len: int, page_size: int, with_raw: bool
) -> None:
    score, lengths, page_table = _setup_topk_transform_512(bs, seq_len, page_size)

    out_page = torch.full((bs, topk), -1, dtype=torch.int32, device="xpu")
    out_raw = (
        torch.full((bs, topk), -1, dtype=torch.int32, device="xpu")
        if with_raw
        else None
    )
    topk_transform_512(score, lengths, page_table, out_page, page_size, out_raw)

    ref_page, ref_raw = _ref_topk_transform_512(
        score, lengths, page_table, page_size, topk
    )

    assert_equal(
        score,
        torch.sort(ref_page, dim=-1).values,
        torch.sort(out_page, dim=-1).values,
        bs,
        topk,
        seq_len,
        max_permit_error=5,
    )

    if with_raw:
        assert_equal(
            score,
            torch.sort(ref_raw, dim=-1).values,
            torch.sort(out_raw, dim=-1).values,
            bs,
            topk,
            seq_len,
            max_permit_error=5,
        )


@pytest.mark.parametrize("bs", [1, 132, 256, 4096])
@pytest.mark.parametrize("topk", [512, 1024, 2048])
@pytest.mark.parametrize("seq_len", [4096, 16384, 65536])
@pytest.mark.parametrize("page_size", [1, 64, 256])
@pytest.mark.parametrize("with_raw", [False, True])
@torch.inference_mode()
def test_topk_transform_512_v2(
    bs: int, topk: int, seq_len: int, page_size: int, with_raw: bool
) -> None:
    score, lengths, page_table = _setup_topk_transform_512(bs, seq_len, page_size)
    metadata = _empty_v2_metadata(bs)

    out_page = torch.full((bs, topk), -1, dtype=torch.int32, device="xpu")
    out_raw = (
        torch.full((bs, topk), -1, dtype=torch.int32, device="xpu")
        if with_raw
        else None
    )
    topk_transform_512_v2(
        score, lengths, page_table, out_page, page_size, metadata, out_raw
    )

    ref_page, ref_raw = _ref_topk_transform_512(
        score, lengths, page_table, page_size, topk
    )

    assert_equal(
        score,
        torch.sort(ref_page, dim=-1).values,
        torch.sort(out_page, dim=-1).values,
        bs,
        topk,
        seq_len,
        max_permit_error=5,
    )

    if with_raw:
        assert_equal(
            score,
            torch.sort(ref_raw, dim=-1).values,
            torch.sort(out_raw, dim=-1).values,
            bs,
            topk,
            seq_len,
            max_permit_error=5,
        )


@pytest.mark.parametrize("bs", [132, 256, 4096])
@pytest.mark.parametrize("topk", [512, 1024])
@pytest.mark.parametrize("seq_len", [4096, 16384, 65536])
@pytest.mark.parametrize("page_size", [64, 256])
@torch.inference_mode()
def test_topk_transform_512_perf(
    bs: int, topk: int, seq_len: int, page_size: int
) -> None:
    score, lengths, page_table = _setup_topk_transform_512(bs, seq_len, page_size)
    out_page = torch.full((bs, topk), -1, dtype=torch.int32, device="xpu")
    out_page_ref = torch.full((bs, topk), -1, dtype=torch.int32, device="xpu")

    t_ref = _bench(
        lambda: _ref_torch_topk_transform_512_impl(
            score, lengths, page_table, out_page_ref, page_size, None
        )
    )
    t_our = _bench(
        lambda: topk_transform_512(
            score, lengths, page_table, out_page, page_size, None
        )
    )
    assert (
        t_our < t_ref
    ), f"sycl ({t_our:.3f} ms) not faster than torch ({t_ref:.3f} ms)"


@pytest.mark.parametrize("bs", [132, 256, 4096])
@pytest.mark.parametrize("topk", [512, 1024, 2048])
@pytest.mark.parametrize("seq_len", [4096, 16384, 65536])
@pytest.mark.parametrize("page_size", [64, 256])
@torch.inference_mode()
def test_topk_transform_512_v2_perf(
    bs: int, topk: int, seq_len: int, page_size: int
) -> None:
    score, lengths, page_table = _setup_topk_transform_512(bs, seq_len, page_size)
    metadata = _empty_v2_metadata(bs)
    out_page = torch.full((bs, topk), -1, dtype=torch.int32, device="xpu")
    out_page_ref = torch.full((bs, topk), -1, dtype=torch.int32, device="xpu")

    t_ref = _bench(
        lambda: _ref_torch_topk_transform_512_impl(
            score, lengths, page_table, out_page_ref, page_size, None
        )
    )
    t_our = _bench(
        lambda: topk_transform_512_v2(
            score, lengths, page_table, out_page, page_size, metadata, None
        )
    )
    assert (
        t_our < t_ref
    ), f"sycl ({t_our:.3f} ms) not faster than torch ({t_ref:.3f} ms)"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
