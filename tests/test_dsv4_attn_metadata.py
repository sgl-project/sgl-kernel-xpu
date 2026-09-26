"""Unit tests for the DeepSeek V4 expand-prefill-causally SYCL operator.

Covers ``dsv4_expand_prefill_causally_out`` (``src/sycl/DSV4ExpandPrefill.cpp``)
against a Python oracle transcribed directly from the SYCL kernel rather than
from a framework implementation.

Importing ``sgl_kernel`` is what registers the operator under
``torch.ops.sgl_kernel``.
"""

from __future__ import annotations

import sys

import pytest
import torch
from utils import get_device

sgl_kernel = pytest.importorskip("sgl_kernel")

DEVICE = get_device()

pytestmark = pytest.mark.skipif(
    DEVICE.type != "xpu",
    reason="DSV4 attention metadata operators are implemented for Intel XPU only",
)

POISON_VALUE = 424242


EXPAND_PREFILL_CASES = {
    "single_request": {
        "seq_lens": [7],
        "extend_seq_lens": [7],
        "padded_num_tokens": 8,
    },
    "ragged_four": {
        "seq_lens": [17, 33, 65, 129],
        "extend_seq_lens": [3, 12, 1, 7],
        "padded_num_tokens": 32,
    },
    "ragged_five_with_empty_extend": {
        "seq_lens": [1, 40, 40, 300, 1024],
        "extend_seq_lens": [1, 0, 9, 130, 2],
        "padded_num_tokens": 160,
    },
    "wide_batch": {
        "seq_lens": [513, 514, 515, 516, 517, 518, 519, 520, 521],
        "extend_seq_lens": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "padded_num_tokens": 64,
    },
}


def _poisoned_pair(padded_num_tokens: int, req_dtype: torch.dtype):
    """Return the CPU poison pattern the expand-prefill outputs start from."""
    return (
        torch.full((padded_num_tokens,), POISON_VALUE, dtype=torch.int32),
        torch.full((padded_num_tokens,), POISON_VALUE, dtype=req_dtype),
    )


def _expand_prefill_reference(
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    extend_start_loc,
    num_tokens: int,
    padded_num_tokens: int,
    initial_causal: torch.Tensor,
    initial_repeated: torch.Tensor,
    causal_base_offset: int = 1,
):
    """Python oracle for ``dsv4_expand_prefill_causally_out``.

    One workgroup per request. ``start`` is either read from
    ``extend_start_loc`` or recovered as the exclusive prefix sum of
    ``extend_seq_lens``. The request owns the token span
    ``[start, start + extend_len)`` clamped into ``[0, num_tokens]``, and writes
    ``seq_lens - extend_seq_lens + 1 + (token - start)`` there, i.e. the causal
    sequence length seen by each extend token. Tokens inside ``[0, num_tokens)``
    that no request claims are never written and keep their prior contents. The
    last request additionally fills ``[num_tokens, padded_num_tokens)`` with a
    causal length of one and its own pool index.

    ``causal_base_offset`` exists only so a negative control can build a
    deliberately off-by-one oracle; the kernel uses one.
    """
    causal = initial_causal.clone()
    repeated = initial_repeated.clone()
    batch_size = req_pool_indices.numel()
    if batch_size == 0:
        return causal, repeated

    extend_list = extend_seq_lens.tolist()
    for row in range(batch_size):
        if extend_start_loc is None:
            start = sum(extend_list[:row])
        else:
            start = int(extend_start_loc[row])
        extend_len = max(extend_list[row], 0)
        begin = min(max(start, 0), num_tokens)
        end = min(max(start + extend_len, 0), num_tokens)
        causal_begin = int(seq_lens[row]) - extend_list[row] + causal_base_offset
        for token in range(begin, end):
            causal[token] = causal_begin + token - start
            repeated[token] = int(req_pool_indices[row])

    last_index = int(req_pool_indices[batch_size - 1])
    for token in range(num_tokens, padded_num_tokens):
        causal[token] = 1
        repeated[token] = last_index
    return causal, repeated


def _run_expand_prefill(
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    extend_start_loc,
    num_tokens: int,
    padded_num_tokens: int,
):
    """Run the operator on freshly poisoned outputs and return the device pair."""
    causal_cpu, repeated_cpu = _poisoned_pair(padded_num_tokens, req_pool_indices.dtype)
    seq_lens_causal = causal_cpu.to(DEVICE)
    req_pool_indices_repeated = repeated_cpu.to(DEVICE)
    pointers = (seq_lens_causal.data_ptr(), req_pool_indices_repeated.data_ptr())
    torch.ops.sgl_kernel.dsv4_expand_prefill_causally_out(
        req_pool_indices.to(DEVICE),
        seq_lens.to(DEVICE),
        extend_seq_lens.to(DEVICE),
        None if extend_start_loc is None else extend_start_loc.to(DEVICE),
        seq_lens_causal,
        req_pool_indices_repeated,
        num_tokens,
        padded_num_tokens,
    )
    torch.xpu.synchronize()
    assert (
        seq_lens_causal.data_ptr(),
        req_pool_indices_repeated.data_ptr(),
    ) == pointers
    return seq_lens_causal, req_pool_indices_repeated


def _assert_expand_prefill_layout(
    seq_lens_causal: torch.Tensor,
    req_pool_indices_repeated: torch.Tensor,
    padded_num_tokens: int,
    req_dtype: torch.dtype,
) -> None:
    """Assert the mutated expand-prefill tensors kept dtype, shape and layout."""
    assert seq_lens_causal.dtype == torch.int32
    assert tuple(seq_lens_causal.shape) == (padded_num_tokens,)
    assert seq_lens_causal.device.type == "xpu"
    assert seq_lens_causal.is_contiguous()
    assert req_pool_indices_repeated.dtype == req_dtype
    assert tuple(req_pool_indices_repeated.shape) == (padded_num_tokens,)
    assert req_pool_indices_repeated.device.type == "xpu"
    assert req_pool_indices_repeated.is_contiguous()


def _expand_prefill_case(name: str, req_dtype: torch.dtype, seq_len_dtype: torch.dtype):
    """Materialise one expand-prefill case as CPU tensors."""
    case = EXPAND_PREFILL_CASES[name]
    batch_size = len(case["seq_lens"])
    req_pool_indices = torch.arange(batch_size, dtype=req_dtype) * 3 + 1
    seq_lens = torch.tensor(case["seq_lens"], dtype=seq_len_dtype)
    extend_seq_lens = torch.tensor(case["extend_seq_lens"], dtype=torch.int32)
    return req_pool_indices, seq_lens, extend_seq_lens, case["padded_num_tokens"]


def _exclusive_prefix(extend_seq_lens: torch.Tensor, dtype: torch.dtype):
    """Build the ``extend_start_loc`` the kernel would otherwise recompute."""
    return torch.nn.functional.pad(
        extend_seq_lens.to(torch.int64).cumsum(0)[:-1], (1, 0)
    ).to(dtype)


@pytest.mark.parametrize("case_name", sorted(EXPAND_PREFILL_CASES))
@pytest.mark.parametrize("req_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("seq_len_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("start_loc_dtype", [None, torch.int32, torch.int64])
def test_expand_prefill_matches_reference(
    case_name: str,
    req_dtype: torch.dtype,
    seq_len_dtype: torch.dtype,
    start_loc_dtype,
) -> None:
    """Ragged extend batches agree with the oracle for every accepted dtype mix."""
    req_pool_indices, seq_lens, extend_seq_lens, padded = _expand_prefill_case(
        case_name, req_dtype, seq_len_dtype
    )
    num_tokens = int(extend_seq_lens.sum())
    extend_start_loc = (
        None
        if start_loc_dtype is None
        else _exclusive_prefix(extend_seq_lens, start_loc_dtype)
    )
    causal, repeated = _run_expand_prefill(
        req_pool_indices,
        seq_lens,
        extend_seq_lens,
        extend_start_loc,
        num_tokens,
        padded,
    )
    _assert_expand_prefill_layout(causal, repeated, padded, req_dtype)
    initial_causal, initial_repeated = _poisoned_pair(padded, req_dtype)
    expected_causal, expected_repeated = _expand_prefill_reference(
        req_pool_indices,
        seq_lens,
        extend_seq_lens,
        extend_start_loc,
        num_tokens,
        padded,
        initial_causal,
        initial_repeated,
    )
    torch.testing.assert_close(causal.cpu(), expected_causal, rtol=0, atol=0)
    torch.testing.assert_close(repeated.cpu(), expected_repeated, rtol=0, atol=0)


@pytest.mark.parametrize("token_delta", [-5, 0, 4])
@pytest.mark.parametrize("use_start_loc", [False, True])
def test_expand_prefill_clamped_span_and_padding_tail(
    token_delta: int, use_start_loc: bool
) -> None:
    """Spans clamp to ``num_tokens`` and unclaimed slots keep their prior contents."""
    req_pool_indices = torch.tensor([9, 2, 41, 7], dtype=torch.int64)
    seq_lens = torch.tensor([5, 21, 100, 257], dtype=torch.int32)
    extend_seq_lens = torch.tensor([5, 1, 7, 9], dtype=torch.int32)
    num_tokens = int(extend_seq_lens.sum()) + token_delta
    padded = 32
    extend_start_loc = (
        _exclusive_prefix(extend_seq_lens, torch.int32) if use_start_loc else None
    )
    causal, repeated = _run_expand_prefill(
        req_pool_indices,
        seq_lens,
        extend_seq_lens,
        extend_start_loc,
        num_tokens,
        padded,
    )
    _assert_expand_prefill_layout(causal, repeated, padded, torch.int64)
    initial_causal, initial_repeated = _poisoned_pair(padded, torch.int64)
    expected_causal, expected_repeated = _expand_prefill_reference(
        req_pool_indices,
        seq_lens,
        extend_seq_lens,
        extend_start_loc,
        num_tokens,
        padded,
        initial_causal,
        initial_repeated,
    )
    torch.testing.assert_close(causal.cpu(), expected_causal, rtol=0, atol=0)
    torch.testing.assert_close(repeated.cpu(), expected_repeated, rtol=0, atol=0)
    assert causal.cpu()[num_tokens:].tolist() == [1] * (padded - num_tokens)
    assert repeated.cpu()[num_tokens:].tolist() == [7] * (padded - num_tokens)


def test_expand_prefill_empty_batch_is_a_noop() -> None:
    """A zero-request batch with a zero-length padded output is accepted."""
    seq_lens_causal = torch.empty(0, dtype=torch.int32, device=DEVICE)
    req_pool_indices_repeated = torch.empty(0, dtype=torch.int64, device=DEVICE)
    torch.ops.sgl_kernel.dsv4_expand_prefill_causally_out(
        torch.empty(0, dtype=torch.int64, device=DEVICE),
        torch.empty(0, dtype=torch.int32, device=DEVICE),
        torch.empty(0, dtype=torch.int32, device=DEVICE),
        None,
        seq_lens_causal,
        req_pool_indices_repeated,
        0,
        0,
    )
    torch.xpu.synchronize()
    _assert_expand_prefill_layout(
        seq_lens_causal, req_pool_indices_repeated, 0, torch.int64
    )


def test_expand_prefill_negative_control_causal_base() -> None:
    """An oracle without the ``+1`` causal base must disagree with the kernel."""
    req_pool_indices = torch.tensor([4, 11], dtype=torch.int32)
    seq_lens = torch.tensor([64, 96], dtype=torch.int32)
    extend_seq_lens = torch.tensor([6, 10], dtype=torch.int32)
    num_tokens = 16
    padded = 16
    causal, _ = _run_expand_prefill(
        req_pool_indices, seq_lens, extend_seq_lens, None, num_tokens, padded
    )
    initial_causal, initial_repeated = _poisoned_pair(padded, torch.int32)
    correct, _ = _expand_prefill_reference(
        req_pool_indices,
        seq_lens,
        extend_seq_lens,
        None,
        num_tokens,
        padded,
        initial_causal,
        initial_repeated,
    )
    wrong, _ = _expand_prefill_reference(
        req_pool_indices,
        seq_lens,
        extend_seq_lens,
        None,
        num_tokens,
        padded,
        initial_causal,
        initial_repeated,
        causal_base_offset=0,
    )
    assert not torch.equal(correct, wrong)
    torch.testing.assert_close(causal.cpu(), correct, rtol=0, atol=0)
    assert not torch.equal(causal.cpu(), wrong)


def test_expand_prefill_negative_control_shifted_start_loc() -> None:
    """A shifted ``extend_start_loc`` must move the written span."""
    req_pool_indices = torch.tensor([1, 5, 9], dtype=torch.int32)
    seq_lens = torch.tensor([30, 44, 70], dtype=torch.int32)
    extend_seq_lens = torch.tensor([4, 4, 4], dtype=torch.int32)
    num_tokens = 12
    padded = 12
    honest = _exclusive_prefix(extend_seq_lens, torch.int32)
    shifted = honest + 4
    from_honest, _ = _run_expand_prefill(
        req_pool_indices, seq_lens, extend_seq_lens, honest, num_tokens, padded
    )
    from_shifted, _ = _run_expand_prefill(
        req_pool_indices, seq_lens, extend_seq_lens, shifted, num_tokens, padded
    )
    assert not torch.equal(from_honest.cpu(), from_shifted.cpu())
    assert from_shifted.cpu()[:4].tolist() == [POISON_VALUE] * 4


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("repeated_dtype", "matching req_pool_indices dtype"),
        ("causal_dtype", "seq_lens_causal must be a contiguous 1D int32"),
        ("extend_dtype", "extend_seq_lens must be a contiguous 1D int32"),
        ("short_padding", "padded_num_tokens must be at least num_tokens"),
        ("short_output", "seq_lens_causal length must equal padded_num_tokens"),
        ("empty_batch_padded", "a non-empty padded output requires at least one"),
    ],
)
def test_expand_prefill_rejects_invalid_arguments(mutation: str, message: str) -> None:
    """Every documented contract violation raises instead of writing garbage."""
    req_pool_indices = torch.tensor([3, 4], dtype=torch.int32, device=DEVICE)
    seq_lens = torch.tensor([8, 12], dtype=torch.int32, device=DEVICE)
    extend_seq_lens = torch.tensor([2, 2], dtype=torch.int32, device=DEVICE)
    seq_lens_causal = torch.empty(4, dtype=torch.int32, device=DEVICE)
    req_pool_indices_repeated = torch.empty(4, dtype=torch.int32, device=DEVICE)
    num_tokens = 4
    padded = 4

    if mutation == "repeated_dtype":
        req_pool_indices_repeated = torch.empty(4, dtype=torch.int64, device=DEVICE)
    elif mutation == "causal_dtype":
        seq_lens_causal = torch.empty(4, dtype=torch.int64, device=DEVICE)
    elif mutation == "extend_dtype":
        extend_seq_lens = torch.tensor([2, 2], dtype=torch.int64, device=DEVICE)
    elif mutation == "short_padding":
        num_tokens = 8
    elif mutation == "short_output":
        padded = 6
    elif mutation == "empty_batch_padded":
        req_pool_indices = torch.empty(0, dtype=torch.int32, device=DEVICE)
        seq_lens = torch.empty(0, dtype=torch.int32, device=DEVICE)
        extend_seq_lens = torch.empty(0, dtype=torch.int32, device=DEVICE)
        num_tokens = 0

    with pytest.raises(RuntimeError, match=message):
        torch.ops.sgl_kernel.dsv4_expand_prefill_causally_out(
            req_pool_indices,
            seq_lens,
            extend_seq_lens,
            None,
            seq_lens_causal,
            req_pool_indices_repeated,
            num_tokens,
            padded,
        )


def test_expand_prefill_graph_replay_rereads_inputs() -> None:
    """A captured expand-prefill replays against updated device inputs."""
    if not hasattr(torch.xpu, "XPUGraph") or not hasattr(torch.xpu, "graph"):
        pytest.skip("XPU graph API is unavailable")

    req_pool_indices = torch.tensor([3, 17, 5, 29], dtype=torch.int32, device=DEVICE)
    seq_lens = torch.tensor([23, 40, 90, 130], dtype=torch.int32, device=DEVICE)
    extend_seq_lens = torch.tensor([2, 5, 1, 4], dtype=torch.int32, device=DEVICE)
    num_tokens = 12
    padded = 16
    causal_cpu, repeated_cpu = _poisoned_pair(padded, torch.int32)
    seq_lens_causal = causal_cpu.to(DEVICE)
    req_pool_indices_repeated = repeated_cpu.to(DEVICE)
    args = (
        req_pool_indices,
        seq_lens,
        extend_seq_lens,
        None,
        seq_lens_causal,
        req_pool_indices_repeated,
        num_tokens,
        padded,
    )
    torch.ops.sgl_kernel.dsv4_expand_prefill_causally_out(*args)
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        torch.ops.sgl_kernel.dsv4_expand_prefill_causally_out(*args)

    next_req = torch.tensor([31, 8, 44, 6], dtype=torch.int32)
    next_seq = torch.tensor([30, 70, 110, 150], dtype=torch.int32)
    next_extend = torch.tensor([6, 1, 3, 2], dtype=torch.int32)
    req_pool_indices.copy_(next_req)
    seq_lens.copy_(next_seq)
    extend_seq_lens.copy_(next_extend)
    torch.xpu.synchronize()
    initial_causal = seq_lens_causal.cpu()
    initial_repeated = req_pool_indices_repeated.cpu()
    pointers = (seq_lens_causal.data_ptr(), req_pool_indices_repeated.data_ptr())
    graph.replay()
    torch.xpu.synchronize()
    assert (
        seq_lens_causal.data_ptr(),
        req_pool_indices_repeated.data_ptr(),
    ) == pointers
    expected_causal, expected_repeated = _expand_prefill_reference(
        next_req,
        next_seq,
        next_extend,
        None,
        num_tokens,
        padded,
        initial_causal,
        initial_repeated,
    )
    torch.testing.assert_close(seq_lens_causal.cpu(), expected_causal, rtol=0, atol=0)
    torch.testing.assert_close(
        req_pool_indices_repeated.cpu(), expected_repeated, rtol=0, atol=0
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
