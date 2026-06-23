from typing import Optional

import pytest
import torch
import utils
from sgl_kernel import (
    fast_topk_transform_fused,
    fast_topk_transform_ragged_fused,
    fast_topk_v2,
)


def _ref_topk(
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


def _ref_topk_transform_decode(
    score: torch.Tensor,
    seq_len: int,
    src_page_table: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    batch_size, _ = score.shape
    assert score.shape[0] == src_page_table.shape[0]
    assert seq_len >= topk
    indices = _ref_topk(score, seq_len, topk, row_starts=row_starts)
    topk_indices = torch.empty(
        (batch_size, topk), dtype=torch.int32, device=score.device
    )
    for i in range(batch_size):
        topk_indices[i] = src_page_table[i, indices[i]]
    return topk_indices


def _ref_topk_transform_ragged(
    score: torch.Tensor,
    seq_len: int,
    topk_indices_offset: torch.Tensor,
    topk: int,
    row_starts: torch.Tensor,
) -> torch.Tensor:
    assert score.shape[0] == topk_indices_offset.shape[0]
    assert seq_len >= topk
    indices = _ref_topk(score, seq_len, topk, row_starts=row_starts)

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
        if len(more) > 0 or len(less) > 0:
            # check whether more values are the same with less values
            # if so, either one is acceptable, since their values are the same
            more_values = sorted(score[i, idx - offset].item() for idx in more)
            less_values = sorted(score[i, idx - offset].item() for idx in less)
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


@pytest.mark.parametrize("bs", [1, 32, 256])
@pytest.mark.parametrize("k", [2048])  # we only support 2048 now
@pytest.mark.parametrize("seq_len", [2048, 4096, 16384])
@pytest.mark.parametrize("has_row_starts", [True, False])
@torch.inference_mode()
def test_fast_topk_v2(bs: int, k: int, seq_len: int, has_row_starts: bool) -> None:
    torch.manual_seed(42)
    score = torch.randn(
        bs,
        seq_len + (2048 if has_row_starts else 0),
        dtype=torch.float32,
        device="xpu",
    )
    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device="xpu")
    row_starts = (
        torch.randint(0, 2048, (bs,), dtype=torch.int32, device="xpu")
        if has_row_starts
        else None
    )

    ref = _ref_topk(score, seq_len, k, row_starts=row_starts)
    our = fast_topk_v2(score, lengths, k, row_starts=row_starts)

    ref = torch.sort(ref, dim=-1).values
    our = torch.sort(our, dim=-1).values

    assert_equal(score, ref, our, bs, k, seq_len, max_permit_error=5)

    t_ref = _bench(lambda: _ref_topk(score, seq_len, k, row_starts=row_starts))
    t_our = _bench(lambda: fast_topk_v2(score, lengths, k, row_starts=row_starts))
    assert t_our < t_ref, f"sycl ({t_our:.3f} ms) not faster than torch ({t_ref:.3f} ms)"


@pytest.mark.parametrize("bs", [1, 32, 256])
@pytest.mark.parametrize("k", [2048])  # we only support 2048 now
@pytest.mark.parametrize("seq_len", [2048, 4096, 16384])
@pytest.mark.parametrize("mode", ["extend", "decode", "target_verify"])
@torch.inference_mode()
def test_fast_topk_transform_fused(bs: int, k: int, seq_len: int, mode: str) -> None:
    torch.manual_seed(42)
    if mode == "decode":
        step = 1
    else:
        step = 4 if bs % 4 == 0 else 1
    num_tokens = bs
    bs = bs // step

    row_starts = (
        torch.randint(0, 2048, (bs,), dtype=torch.int32, device="xpu")
        if mode == "extend"
        else None
    )
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
    src_page_table = (
        torch.arange(0, seq_len, dtype=torch.int32, device="xpu")
        .unsqueeze(0)
        .expand(bs, -1)
    )

    ref = _ref_topk_transform_decode(
        score, seq_len, src_page_table, k, row_starts=row_starts
    )
    our = fast_topk_transform_fused(
        score=score,
        lengths=lengths,
        page_table_size_1=src_page_table,
        cu_seqlens_q=cu_seqlens_q,
        topk=k,
        row_starts=row_starts,
    )
    ref = torch.sort(ref, dim=-1).values
    our = torch.sort(our, dim=-1).values
    assert_equal(score, ref, our, bs, k, seq_len, max_permit_error=5)

    t_ref = _bench(
        lambda: _ref_topk_transform_decode(
            score, seq_len, src_page_table, k, row_starts=row_starts
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
    assert t_our < t_ref, f"sycl ({t_our:.3f} ms) not faster than torch ({t_ref:.3f} ms)"


@pytest.mark.parametrize("bs", [1, 32, 256])
@pytest.mark.parametrize("k", [2048])  # we only support 2048 now
@pytest.mark.parametrize("seq_len", [2048, 4096, 16384])
@pytest.mark.parametrize("has_row_starts", [True, False])
@torch.inference_mode()
def test_fast_topk_transform_ragged(
    bs: int, k: int, seq_len: int, has_row_starts: bool
) -> None:
    torch.manual_seed(42)
    score = torch.randn(
        bs,
        seq_len + (2048 if has_row_starts else 0),
        dtype=torch.float32,
        device="xpu",
    )
    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device="xpu")
    row_starts = (
        torch.randint(0, 2048, (bs,), dtype=torch.int32, device="xpu")
        if has_row_starts
        else None
    )
    topk_indices_offset = torch.randint(
        0, 1024, (bs,), dtype=torch.int32, device="xpu"
    )

    ref = _ref_topk_transform_ragged(
        score, seq_len, topk_indices_offset, k, row_starts=row_starts
    )
    our = fast_topk_transform_ragged_fused(
        score=score,
        lengths=lengths,
        topk_indices_offset=topk_indices_offset,
        topk=k,
        row_starts=row_starts,
    )
    ref = torch.sort(ref, dim=-1).values
    our = torch.sort(our, dim=-1).values
    assert_equal(score, ref, our, bs, k, seq_len, max_permit_error=5)

    t_ref = _bench(
        lambda: _ref_topk_transform_ragged(
            score, seq_len, topk_indices_offset, k, row_starts=row_starts
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
    assert t_our < t_ref, f"sycl ({t_our:.3f} ms) not faster than torch ({t_ref:.3f} ms)"


if __name__ == "__main__":
    pytest.main([__file__])
