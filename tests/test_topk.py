import sys
from typing import Any, Optional

import pytest
import torch
from sgl_kernel import (
    fast_topk_transform_fused,
    fast_topk_transform_ragged_fused,
    fast_topk_v2,
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


MAX_SEQ_LEN = 131072


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
@pytest.mark.parametrize("has_row_starts", [True, False])
@torch.inference_mode()
def test_fast_topk_v2_perf(bs: int, k: int, seq_len: int, has_row_starts: bool) -> None:
    score, lengths, row_starts = _setup_fast_topk_v2(bs, seq_len, has_row_starts)

    t_ref = _bench(lambda: _ref_torch_impl(score, seq_len, k, row_starts=row_starts))
    t_our = _bench(lambda: fast_topk_v2(score, lengths, k, row_starts=row_starts))
    assert (
        t_our < t_ref
    ), f"sycl ({t_our:.3f} ms) not faster than torch ({t_ref:.3f} ms)"


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


@pytest.mark.parametrize("bs", [1, 132, 256, 4096])
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
