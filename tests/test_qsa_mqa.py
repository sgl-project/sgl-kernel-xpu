import math

import pytest
import sgl_kernel  # noqa: F401
import torch

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available()
    or not hasattr(torch.ops.sgl_kernel, "qsa_mqa_prefill")
    or not hasattr(torch.ops.sgl_kernel, "qsa_mqa_decode"),
    reason="Requires an XPU build with QSA MQA operators enabled",
)


@pytest.mark.parametrize(
    "rows,columns",
    [(0, 0), (1, 1), (3, 17), (4, 128), (5, 131), (8, 513), (5, 4097)],
)
@pytest.mark.parametrize("scale", [None, 4.0])
def test_prefill_matches_reference(rows, columns, scale):
    torch.manual_seed(61)
    q = torch.randn(rows, 4, 128, device="xpu", dtype=torch.bfloat16)
    k = torch.randn(columns, 1, 128, device="xpu", dtype=torch.bfloat16)
    starts = torch.tensor([i % 3 for i in range(rows)], device="xpu", dtype=torch.int32)
    ends = torch.tensor(
        [max(i % 3, columns - i * 3) for i in range(rows)],
        device="xpu",
        dtype=torch.int32,
    )
    actual = torch.ops.sgl_kernel.qsa_mqa_prefill(q, k, starts, ends, scale)
    assert actual.shape == (rows, columns)
    if not rows or not columns:
        return
    scores = torch.einsum("rhd,cd->rch", q.float(), k[:, 0].float())
    expected = torch.relu(scores).sum(dim=-1) / (scale or math.sqrt(128))
    positions = torch.arange(columns, device="xpu")[None, :]
    expected.masked_fill_(
        (positions < starts[:, None]) | (positions >= ends[:, None]), -float("inf")
    )
    torch.testing.assert_close(actual, expected, rtol=0.004, atol=0.004)


@pytest.mark.parametrize(
    "rows,page_size,width", [(1, 8, 3), (2, 16, 5), (3, 32, 4), (5, 64, 2)]
)
@pytest.mark.parametrize("extra", [0, 19])
def test_decode_matches_reference(rows, page_size, width, extra):
    torch.manual_seed(62)
    q = torch.randn(rows, 4, 128, device="xpu", dtype=torch.bfloat16)
    cache = torch.randn(11, page_size, 1, 128, device="xpu", dtype=torch.bfloat16)
    pages = torch.randint(0, 11, (rows, width), device="xpu", dtype=torch.int32)
    pages[-1, -1] = -1
    lengths = torch.tensor(
        [0 if row == 0 else page_size * width - row for row in range(rows)],
        device="xpu",
        dtype=torch.int32,
    )
    columns = page_size * width + extra
    actual = torch.ops.sgl_kernel.qsa_mqa_decode(
        q, cache, pages, lengths, columns, None
    )
    expected = torch.full((rows, columns), -float("inf"), device="xpu")
    for row in range(rows):
        for col in range(min(int(lengths[row]), columns)):
            page = int(pages[row, col // page_size])
            if page < 0:
                continue
            values = torch.einsum(
                "hd,d->h", q[row].float(), cache[page, col % page_size, 0].float()
            )
            expected[row, col] = values.relu().sum() / math.sqrt(128)
    torch.testing.assert_close(actual, expected, rtol=0.004, atol=0.004)


def test_rejects_invalid_inputs():
    q = torch.zeros(1, 4, 128, device="xpu", dtype=torch.bfloat16)
    k = torch.zeros(2, 1, 128, device="xpu", dtype=torch.bfloat16)
    starts = torch.zeros(1, device="xpu", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="BF16 or float8_e4m3fn q"):
        torch.ops.sgl_kernel.qsa_mqa_prefill(q.float(), k, starts, starts, None)
    with pytest.raises(RuntimeError, match="int32 row ranges"):
        torch.ops.sgl_kernel.qsa_mqa_prefill(q, k, starts.long(), starts, None)
    with pytest.raises(RuntimeError, match="must be on XPU"):
        torch.ops.sgl_kernel.qsa_mqa_prefill(q, k, starts, starts.cpu(), None)
    with pytest.raises(RuntimeError, match="score scale must be finite"):
        torch.ops.sgl_kernel.qsa_mqa_prefill(q, k, starts, starts, float("nan"))
    cache = k.reshape(1, 2, 1, 128)
    pages = starts.reshape(1, 1)
    with pytest.raises(RuntimeError, match="output width"):
        torch.ops.sgl_kernel.qsa_mqa_decode(q, cache, pages, starts, -1, None)


def test_prefill_dpas_maps_rows_heads_and_dimensions():
    q = torch.zeros(5, 4, 128, device="xpu", dtype=torch.bfloat16)
    k = torch.zeros(17, 1, 128, device="xpu", dtype=torch.bfloat16)
    starts = torch.zeros(5, device="xpu", dtype=torch.int32)
    ends = torch.full_like(starts, 17)
    columns = torch.arange(17, device="xpu", dtype=torch.bfloat16)
    for head in range(4):
        for dim in (0, 1, 15, 16, 63, 127):
            q.zero_()
            k.zero_()
            q[:, head, dim] = 1
            k[:, 0, dim] = columns
            actual = torch.ops.sgl_kernel.qsa_mqa_prefill(q, k, starts, ends, 1.0)
            torch.testing.assert_close(actual, columns.float()[None, :].expand(5, -1))


@pytest.mark.parametrize("scale", [None, 4.0])
def test_fp8_prefill_scores_and_ranges(scale):
    torch.manual_seed(771)
    rows, columns = 5, 135
    q = torch.randn(rows, 4, 128, device="xpu", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    k = torch.randn(columns, 1, 128, device="xpu", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    starts = torch.tensor([0, 7, 131, 4, 140], device="xpu", dtype=torch.int32)
    ends = torch.tensor([135, 100, 135, 4, 150], device="xpu", dtype=torch.int32)
    actual = torch.ops.sgl_kernel.qsa_mqa_prefill(q, k, starts, ends, scale)
    scores = torch.einsum("rhd,cd->rch", q.float(), k[:, 0].float())
    expected = scores.relu().sum(-1) / (scale or math.sqrt(128))
    col = torch.arange(columns, device="xpu")
    expected.masked_fill_(
        (col[None, :] < starts[:, None]) | (col[None, :] >= ends[:, None]),
        -float("inf"),
    )
    torch.testing.assert_close(actual, expected, rtol=0.004, atol=0.004)


@pytest.mark.parametrize("scale", [None, 4.0])
def test_fp8_decode_scores_and_masks(scale):
    torch.manual_seed(772)
    rows, page_size, columns = 3, 16, 69
    q = torch.randn(rows, 4, 128, device="xpu", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    cache = torch.randn(7, page_size, 1, 128, device="xpu", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    pages = torch.tensor(
        [[0, 1, 2], [5, -1, 4], [2, 7, 1]], device="xpu", dtype=torch.int32
    )
    lengths = torch.tensor([0, 47, 47], device="xpu", dtype=torch.int32)
    actual = torch.ops.sgl_kernel.qsa_mqa_decode(
        q, cache, pages, lengths, columns, scale
    )
    expected = torch.full((rows, columns), -float("inf"), device="xpu")
    for row in range(rows):
        for col in range(min(int(lengths[row]), columns)):
            page = int(pages[row, col // page_size])
            if page < 0 or page >= cache.size(0):
                continue
            scores = torch.einsum(
                "hd,d->h", q[row].float(), cache[page, col % page_size, 0].float()
            )
            expected[row, col] = scores.relu().sum() / (scale or math.sqrt(128))
    torch.testing.assert_close(actual, expected, rtol=0.004, atol=0.004)


def test_rejects_mixed_qk_dtypes():
    q = torch.zeros(1, 4, 128, device="xpu", dtype=torch.bfloat16)
    k = torch.zeros(1, 1, 128, device="xpu", dtype=torch.bfloat16)
    starts = torch.zeros(1, device="xpu", dtype=torch.int32)
    pages = starts.reshape(1, 1)
    for q_dtype, k_dtype in (
        (torch.bfloat16, torch.float8_e4m3fn),
        (torch.float8_e4m3fn, torch.bfloat16),
    ):
        query, key = q.to(q_dtype), k.to(k_dtype)
        with pytest.raises(RuntimeError, match="Q/K dtype mismatch"):
            torch.ops.sgl_kernel.qsa_mqa_prefill(query, key, starts, starts, None)
        with pytest.raises(RuntimeError, match="Q/K dtype mismatch"):
            torch.ops.sgl_kernel.qsa_mqa_decode(
                query, key.reshape(1, 1, 1, 128), pages, starts, 1, None
            )
