import sys
import math
from typing import Optional

import pytest
import torch
from sgl_kernel import sgemm_lora_a_fwd


def _tolerances(dtype: torch.dtype, k: int = 0):
    if dtype == torch.float16:
        return 1e-2, 1e-2
    if dtype == torch.bfloat16:
        return 2e-2, 2e-2
    if dtype == torch.float32:
        # fp32 flows through the XMX DPAS, which has no true-fp32 path: each
        # operand is truncated to TF32 (10-bit mantissa) before the multiply.
        # Unlike the fp16/bf16 tests -- where the reference starts from the same
        # low-precision operands -- the fp32 reference multiplies full-precision
        # values, so the kernel's TF32 input-truncation error (~2^-11 per term)
        # is entirely uncanceled and accumulates as ~sqrt(K) over the reduction.
        # Scale the tolerance with sqrt(K) to track that growth.
        scale = math.sqrt(max(1.0, k / 512.0))
        return 4e-2 * scale, 4e-2 * scale
    return 1e-5, 1e-5


def _zero_weight_rank_tail(
    weights: torch.Tensor, lora_ranks: torch.Tensor, stack_num: int
) -> torch.Tensor:
    """Zero-pad the weight rows beyond ``lora_ranks[lora]`` per stack.

    The kernel trusts that weight rows beyond the active rank are zero (it
    computes the full ``M_s x N`` output unconditionally). This helper enforces
    that contract for test inputs so the reference and kernel outputs agree.
    """
    num_loras, total_n, _ = weights.shape
    max_rank = total_n // stack_num
    out = weights.clone()
    ranks_cpu = lora_ranks.cpu()
    for l in range(num_loras):
        r = int(ranks_cpu[l].item())
        for k in range(stack_num):
            base = k * max_rank
            out[l, base + r : base + max_rank, :] = 0
    return out


def _reference_sgemm_lora_a_fwd(
    input_x: torch.Tensor,
    weights: torch.Tensor,
    stack_num: int,
    seg_indptr: torch.Tensor,
    weight_indices: torch.Tensor,
) -> torch.Tensor:
    """Reference implementation computed on CPU in fp32, narrowed to weight dtype."""
    input_x_cpu = input_x.cpu()
    weights_cpu = weights.cpu()
    seg_cpu = seg_indptr.cpu()
    wi_cpu = weight_indices.cpu()

    num_tokens, _ = input_x_cpu.shape
    total_n = weights_cpu.size(1)
    out = torch.zeros((num_tokens, total_n), dtype=weights_cpu.dtype)

    num_segments = seg_cpu.numel() - 1
    for s in range(num_segments):
        start = int(seg_cpu[s].item())
        end = int(seg_cpu[s + 1].item())
        if end == start:
            continue
        lora = int(wi_cpu[s].item())
        x = input_x_cpu[start:end].float()
        w = weights_cpu[lora].float()  # [total_n, input_dim]
        out[start:end] = (x @ w.T).to(weights_cpu.dtype)
    return out


def _run_and_compare(
    *,
    dtype: torch.dtype,
    num_tokens: int,
    input_dim: int,
    max_rank: int,
    stack_num: int,
    num_loras: int,
    seg_indptr: torch.Tensor,
    weight_indices: torch.Tensor,
    lora_ranks: torch.Tensor,
    seg_lens: Optional[torch.Tensor] = None,
) -> None:
    torch.manual_seed(0)
    total_n = stack_num * max_rank
    input_x = torch.randn(num_tokens, input_dim, dtype=dtype, device="xpu")
    weights = torch.randn(num_loras, total_n, input_dim, dtype=dtype, device="xpu")
    weights = _zero_weight_rank_tail(weights, lora_ranks, stack_num)

    out = sgemm_lora_a_fwd(
        input_x=input_x,
        weights=weights,
        stack_num=stack_num,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        seg_lens=seg_lens,
    )

    ref = _reference_sgemm_lora_a_fwd(
        input_x, weights, stack_num, seg_indptr, weight_indices
    )

    out_cpu = out.cpu()

    assert out_cpu.shape == (num_tokens, total_n)
    assert out_cpu.dtype == dtype
    rtol, atol = _tolerances(dtype, k=input_dim)
    torch.testing.assert_close(out_cpu, ref, rtol=rtol, atol=atol)


# ----------------------------------------------------------------------------
# Correctness
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("input_dim", [64, 4096])
@pytest.mark.parametrize("max_rank", [8, 64])
def test_sgemm_lora_a_fwd_basic_shapes(dtype, input_dim, max_rank):
    num_tokens = 64
    num_loras = 3
    stack_num = 1

    seg_indptr = torch.tensor([0, 16, 48, 64], dtype=torch.int32, device="xpu")
    weight_indices = torch.tensor([0, 2, 1], dtype=torch.int32, device="xpu")
    lora_ranks = torch.tensor(
        [max_rank, max(1, max_rank // 2), max(1, max_rank // 4)],
        dtype=torch.int32,
        device="xpu",
    )

    _run_and_compare(
        dtype=dtype,
        num_tokens=num_tokens,
        input_dim=input_dim,
        max_rank=max_rank,
        stack_num=stack_num,
        num_loras=num_loras,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("stack_num", [1, 2, 3])
def test_sgemm_lora_a_fwd_stack_num(dtype, stack_num):
    num_tokens = 32
    input_dim = 256
    max_rank = 8
    num_loras = 2

    seg_indptr = torch.tensor([0, 16, 32], dtype=torch.int32, device="xpu")
    weight_indices = torch.tensor([0, 1], dtype=torch.int32, device="xpu")
    lora_ranks = torch.tensor([max_rank, max_rank // 2], dtype=torch.int32, device="xpu")

    _run_and_compare(
        dtype=dtype,
        num_tokens=num_tokens,
        input_dim=input_dim,
        max_rank=max_rank,
        stack_num=stack_num,
        num_loras=num_loras,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_sgemm_lora_a_fwd_single_segment_single_lora(dtype):
    num_tokens = 128
    input_dim = 512
    max_rank = 16
    num_loras = 1
    stack_num = 1

    seg_indptr = torch.tensor([0, num_tokens], dtype=torch.int32, device="xpu")
    weight_indices = torch.tensor([0], dtype=torch.int32, device="xpu")
    lora_ranks = torch.tensor([max_rank], dtype=torch.int32, device="xpu")

    _run_and_compare(
        dtype=dtype,
        num_tokens=num_tokens,
        input_dim=input_dim,
        max_rank=max_rank,
        stack_num=stack_num,
        num_loras=num_loras,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_sgemm_lora_a_fwd_single_token_segments(dtype):
    """One-token-per-segment stress: maximum segment count for the token count."""
    num_tokens = 64
    input_dim = 128
    max_rank = 8
    num_loras = 4
    stack_num = 1

    seg_indptr = torch.arange(0, num_tokens + 1, dtype=torch.int32, device="xpu")
    weight_indices = torch.arange(num_tokens, dtype=torch.int32, device="xpu") % num_loras
    weight_indices = weight_indices.to(torch.int32)
    lora_ranks = torch.tensor([1, 3, 5, 8], dtype=torch.int32, device="xpu")

    _run_and_compare(
        dtype=dtype,
        num_tokens=num_tokens,
        input_dim=input_dim,
        max_rank=max_rank,
        stack_num=stack_num,
        num_loras=num_loras,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
    )


# Stress test for many segments + large token counts. 1024 is a near-duplicate
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("num_tokens", [4096])
def test_sgemm_lora_a_fwd_large_num_tokens_many_segments(dtype, num_tokens):
    input_dim = 512
    max_rank = 16
    num_loras = 4
    stack_num = 1

    # Many variable-sized segments.
    torch.manual_seed(7)
    num_segments = 17
    raw = torch.randint(1, 32, (num_segments,), dtype=torch.int32)
    # Rescale to sum to num_tokens.
    total = int(raw.sum().item())
    raw = (raw.float() * (num_tokens / total)).round().to(torch.int32)
    # Patch up the rounding so it sums to exactly num_tokens.
    diff = num_tokens - int(raw.sum().item())
    raw[0] = max(0, int(raw[0].item()) + diff)
    assert int(raw.sum().item()) == num_tokens

    seg_indptr = torch.zeros(num_segments + 1, dtype=torch.int32)
    seg_indptr[1:] = torch.cumsum(raw, dim=0).to(torch.int32)
    seg_indptr = seg_indptr.to("xpu")
    weight_indices = torch.randint(
        0, num_loras, (num_segments,), dtype=torch.int32, device="xpu"
    )
    lora_ranks = torch.tensor([1, 4, 8, 16], dtype=torch.int32, device="xpu")

    _run_and_compare(
        dtype=dtype,
        num_tokens=num_tokens,
        input_dim=input_dim,
        max_rank=max_rank,
        stack_num=stack_num,
        num_loras=num_loras,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
    )


# ----------------------------------------------------------------------------
# Rank padding / zero behaviour
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_sgemm_lora_a_fwd_rank_padding_zero_tail(dtype):
    """Output lanes beyond per-lora rank must be zero (relies on weight pre-zeroing)."""
    torch.manual_seed(11)
    num_tokens = 32
    input_dim = 128
    max_rank = 8
    num_loras = 2
    stack_num = 1
    total_n = stack_num * max_rank

    seg_indptr = torch.tensor([0, 16, 32], dtype=torch.int32, device="xpu")
    weight_indices = torch.tensor([0, 1], dtype=torch.int32, device="xpu")
    lora_ranks = torch.tensor([2, 5], dtype=torch.int32, device="xpu")

    input_x = torch.randn(num_tokens, input_dim, dtype=dtype, device="xpu")
    weights = torch.randn(num_loras, total_n, input_dim, dtype=dtype, device="xpu")
    weights = _zero_weight_rank_tail(weights, lora_ranks, stack_num)

    out = sgemm_lora_a_fwd(
        input_x=input_x,
        weights=weights,
        stack_num=stack_num,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        seg_lens=None,
    )

    # Segment 0: lora 0 rank=2 -> columns [2,8) must be zero.
    assert torch.count_nonzero(out[:16, 2:]).item() == 0
    # Segment 1: lora 1 rank=5 -> columns [5,8) must be zero.
    assert torch.count_nonzero(out[16:, 5:]).item() == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_sgemm_lora_a_fwd_zero_rank_all_zero_output(dtype):
    torch.manual_seed(14)
    num_tokens = 16
    input_dim = 64
    max_rank = 8
    num_loras = 2
    stack_num = 1
    total_n = stack_num * max_rank

    seg_indptr = torch.tensor([0, 8, 16], dtype=torch.int32, device="xpu")
    weight_indices = torch.tensor([0, 1], dtype=torch.int32, device="xpu")
    lora_ranks = torch.tensor([0, 0], dtype=torch.int32, device="xpu")

    input_x = torch.randn(num_tokens, input_dim, dtype=dtype, device="xpu")
    weights = torch.randn(num_loras, total_n, input_dim, dtype=dtype, device="xpu")
    weights = _zero_weight_rank_tail(weights, lora_ranks, stack_num)

    out = sgemm_lora_a_fwd(
        input_x=input_x,
        weights=weights,
        stack_num=stack_num,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        seg_lens=None,
    )

    assert torch.count_nonzero(out).item() == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_sgemm_lora_a_fwd_segment_boundaries_precise_routing(dtype):
    """Variable segment sizes + non-identity weight_indices verifies routing."""
    torch.manual_seed(15)
    input_dim = 128
    max_rank = 8
    num_loras = 3
    stack_num = 1

    # Segment sizes: 3, 1, 5  ->  9 tokens total.
    seg_indptr = torch.tensor([0, 3, 4, 9], dtype=torch.int32, device="xpu")
    weight_indices = torch.tensor([2, 1, 0], dtype=torch.int32, device="xpu")
    lora_ranks = torch.tensor([8, 2, 4], dtype=torch.int32, device="xpu")

    _run_and_compare(
        dtype=dtype,
        num_tokens=9,
        input_dim=input_dim,
        max_rank=max_rank,
        stack_num=stack_num,
        num_loras=num_loras,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_sgemm_lora_a_fwd_empty_segments_mixed_in(dtype):
    """Empty segments (start == end) must be skipped without affecting neighbours."""
    torch.manual_seed(8)
    input_dim = 128
    max_rank = 8
    num_loras = 2
    stack_num = 1

    # Segment 1 is empty (indices 4 -> 4), segment 3 is empty (8 -> 8).
    seg_indptr = torch.tensor([0, 4, 4, 8, 8, 16], dtype=torch.int32, device="xpu")
    weight_indices = torch.tensor([0, 1, 0, 1, 0], dtype=torch.int32, device="xpu")
    lora_ranks = torch.tensor([8, 4], dtype=torch.int32, device="xpu")

    _run_and_compare(
        dtype=dtype,
        num_tokens=16,
        input_dim=input_dim,
        max_rank=max_rank,
        stack_num=stack_num,
        num_loras=num_loras,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
    )


# ----------------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_sgemm_lora_a_fwd_empty_input(dtype):
    """num_tokens == 0 must short-circuit cleanly and return a 0-row tensor."""
    input_dim = 64
    max_rank = 8
    num_loras = 1
    stack_num = 1
    total_n = stack_num * max_rank

    input_x = torch.empty((0, input_dim), dtype=dtype, device="xpu")
    weights = torch.randn(num_loras, total_n, input_dim, dtype=dtype, device="xpu")
    seg_indptr = torch.tensor([0], dtype=torch.int32, device="xpu")
    weight_indices = torch.empty((0,), dtype=torch.int32, device="xpu")
    lora_ranks = torch.tensor([max_rank], dtype=torch.int32, device="xpu")

    out = sgemm_lora_a_fwd(
        input_x=input_x,
        weights=weights,
        stack_num=stack_num,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        seg_lens=None,
    )

    assert out.shape == (0, total_n)
    assert out.numel() == 0


def test_sgemm_lora_a_fwd_int64_index_tensors_accepted():
    """seg_indptr / weight_indices in int64 should be cast internally and work.

    The dtype of input/weights is orthogonal to the index-tensor cast path, so
    a single dtype is sufficient here.
    """
    dtype = torch.bfloat16
    num_tokens = 32
    input_dim = 128
    max_rank = 8
    num_loras = 2
    stack_num = 1

    seg_indptr = torch.tensor([0, 16, 32], dtype=torch.int64, device="xpu")
    weight_indices = torch.tensor([1, 0], dtype=torch.int64, device="xpu")
    lora_ranks = torch.tensor([max_rank, max_rank // 2], dtype=torch.int32, device="xpu")

    _run_and_compare(
        dtype=dtype,
        num_tokens=num_tokens,
        input_dim=input_dim,
        max_rank=max_rank,
        stack_num=stack_num,
        num_loras=num_loras,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
    )


# ----------------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------------


def _make_valid_kwargs():
    num_tokens = 4
    input_dim = 64
    max_rank = 4
    num_loras = 1
    stack_num = 1
    total_n = stack_num * max_rank

    return dict(
        input_x=torch.randn(num_tokens, input_dim, dtype=torch.float16, device="xpu"),
        weights=torch.randn(
            num_loras, total_n, input_dim, dtype=torch.float16, device="xpu"
        ),
        stack_num=stack_num,
        seg_indptr=torch.tensor([0, num_tokens], dtype=torch.int32, device="xpu"),
        weight_indices=torch.tensor([0], dtype=torch.int32, device="xpu"),
        lora_ranks=torch.tensor([max_rank], dtype=torch.int32, device="xpu"),
        seg_lens=None,
    )


@pytest.mark.parametrize(
    "bad_case, expected_msg",
    [
        ("input_x_dim", "input_x must be a 2D tensor"),
        ("weights_dim", "weights must be a 3D tensor"),
        ("seg_indptr_dim", "seg_indptr must be a 1D tensor"),
        ("weight_indices_dim", "weight_indices must be a 1D tensor"),
        ("lora_ranks_dim", "lora_ranks must be a 1D tensor"),
        (
            "lora_ranks_size",
            "lora_ranks.numel\\(\\) must equal weights.size\\(0\\)",
        ),
        (
            "weight_indices_size",
            "weight_indices.numel\\(\\) must equal seg_indptr.numel\\(\\) - 1",
        ),
        (
            "weight_indices_out_of_range",
            "weight_indices values must be in",
        ),
        ("seg_indptr_start_nonzero", "seg_indptr\\[0\\] must be 0"),
        (
            "seg_indptr_end_mismatch",
            "seg_indptr\\[-1\\] must equal num_tokens",
        ),
        ("seg_indptr_decreasing", "seg_indptr must be non-decreasing"),
        (
            "lora_ranks_out_of_range",
            "lora_ranks must be within the range",
        ),
        ("dtype_mismatch", "Input tensor dtype must match weights dtype"),
    ],
)
def test_sgemm_lora_a_fwd_input_validation(bad_case, expected_msg):
    kwargs = _make_valid_kwargs()

    if bad_case == "input_x_dim":
        kwargs["input_x"] = kwargs["input_x"].view(-1)
    elif bad_case == "weights_dim":
        kwargs["weights"] = kwargs["weights"].view(1, -1)
    elif bad_case == "seg_indptr_dim":
        kwargs["seg_indptr"] = kwargs["seg_indptr"].view(1, -1)
    elif bad_case == "weight_indices_dim":
        kwargs["weight_indices"] = kwargs["weight_indices"].view(1, 1)
    elif bad_case == "lora_ranks_dim":
        kwargs["lora_ranks"] = kwargs["lora_ranks"].view(1, 1)
    elif bad_case == "lora_ranks_size":
        kwargs["lora_ranks"] = torch.tensor(
            [4, 4], dtype=torch.int32, device="xpu"
        )
    elif bad_case == "weight_indices_size":
        kwargs["weight_indices"] = torch.tensor(
            [0, 0], dtype=torch.int32, device="xpu"
        )
    elif bad_case == "weight_indices_out_of_range":
        kwargs["weight_indices"] = torch.tensor(
            [5], dtype=torch.int32, device="xpu"
        )
    elif bad_case == "seg_indptr_start_nonzero":
        kwargs["seg_indptr"] = torch.tensor(
            [1, 4], dtype=torch.int32, device="xpu"
        )
    elif bad_case == "seg_indptr_end_mismatch":
        kwargs["seg_indptr"] = torch.tensor(
            [0, 3], dtype=torch.int32, device="xpu"
        )
    elif bad_case == "seg_indptr_decreasing":
        # 2-segment indptr, decreasing in the middle (4 -> 2).
        kwargs["seg_indptr"] = torch.tensor(
            [0, 4, 2, 4], dtype=torch.int32, device="xpu"
        )
        kwargs["weight_indices"] = torch.tensor(
            [0, 0, 0], dtype=torch.int32, device="xpu"
        )
    elif bad_case == "lora_ranks_out_of_range":
        # max_rank = weights.size(1) / stack_num = 4
        kwargs["lora_ranks"] = torch.tensor([5], dtype=torch.int32, device="xpu")
    elif bad_case == "dtype_mismatch":
        kwargs["input_x"] = kwargs["input_x"].to(torch.bfloat16)

    with pytest.raises(RuntimeError, match=expected_msg):
        sgemm_lora_a_fwd(**kwargs)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
