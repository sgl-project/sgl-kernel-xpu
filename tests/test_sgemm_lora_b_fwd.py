import sys
from typing import Optional

import pytest
import torch
from sgl_kernel import sgemm_lora_b_fwd

if not torch.xpu.is_available():
    pytest.skip(reason="sgemm_lora_b_fwd requires XPU device.", allow_module_level=True)


def _tolerances(dtype: torch.dtype):
    if dtype == torch.float16:
        return 1e-2, 1e-2
    if dtype == torch.bfloat16:
        return 2e-2, 2e-2
    return 1e-5, 1e-5


def _zero_weight_rank_tail(
    weights: torch.Tensor, lora_ranks: torch.Tensor
) -> torch.Tensor:
    """Zero the weight columns beyond ``lora_ranks[lora]`` in the reduction dim.

    For LoRA-B the rank axis is the reduction (K) dimension -- ``weights`` is
    ``[num_loras, output_dim, max_rank]`` -- so an adapter of rank ``R_l`` only
    activates the first ``R_l`` columns. The kernel computes the full ``max_rank``
    reduction unconditionally and trusts that the tail columns are zero; this
    helper enforces that contract for test inputs so the reference and kernel
    outputs agree (and makes the rank == 0 case a genuine all-zero LoRA term).
    """
    out = weights.clone()
    ranks_cpu = lora_ranks.cpu()
    for l in range(weights.size(0)):
        r = int(ranks_cpu[l].item())
        out[l, :, r:] = 0
    return out


def _reference_sgemm_lora_b_fwd(
    input_x: torch.Tensor,
    weights: torch.Tensor,
    seg_indptr: torch.Tensor,
    weight_indices: torch.Tensor,
    scalings: torch.Tensor,
    base_output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reference computed on CPU in fp32, narrowed to weight dtype.

    Implements ``D = scalings[l] * (input_x @ weights[l]^T) + base_output`` where
    the residual term is only added when ``base_output`` is supplied (beta == 1);
    otherwise the pure scaled LoRA projection is returned (beta == 0).
    """
    input_x_cpu = input_x.cpu()
    weights_cpu = weights.cpu()
    seg_cpu = seg_indptr.cpu()
    wi_cpu = weight_indices.cpu()
    scal_cpu = scalings.cpu().float()

    num_tokens = input_x_cpu.size(0)
    output_dim = weights_cpu.size(1)

    if base_output is not None:
        out = base_output.cpu().float().clone()
    else:
        out = torch.zeros((num_tokens, output_dim), dtype=torch.float32)

    num_segments = seg_cpu.numel() - 1
    for s in range(num_segments):
        start = int(seg_cpu[s].item())
        end = int(seg_cpu[s + 1].item())
        if end == start:
            continue
        lora = int(wi_cpu[s].item())
        x = input_x_cpu[start:end].float()  # [seg, max_rank]
        w = weights_cpu[lora].float()  # [output_dim, max_rank]
        out[start:end] += float(scal_cpu[lora].item()) * (x @ w.T)
    return out.to(weights_cpu.dtype)


def _run_and_compare(
    *,
    dtype: torch.dtype,
    num_tokens: int,
    max_rank: int,
    output_dim: int,
    num_loras: int,
    seg_indptr: torch.Tensor,
    weight_indices: torch.Tensor,
    lora_ranks: torch.Tensor,
    scalings: Optional[torch.Tensor] = None,
    use_base_output: bool = False,
    seg_lens: Optional[torch.Tensor] = None,
) -> None:
    torch.manual_seed(0)
    if scalings is None:
        scalings = torch.ones(num_loras, dtype=torch.float32, device="xpu")

    # input_x is the LoRA-A projection: [num_tokens, max_rank] (K == max_rank).
    input_x = torch.randn(num_tokens, max_rank, dtype=dtype, device="xpu")
    weights = torch.randn(num_loras, output_dim, max_rank, dtype=dtype, device="xpu")
    weights = _zero_weight_rank_tail(weights, lora_ranks)

    base_output = None
    if use_base_output:
        base_output = torch.randn(num_tokens, output_dim, dtype=dtype, device="xpu")

    out = sgemm_lora_b_fwd(
        input_x=input_x,
        weights=weights,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        scalings=scalings,
        seg_lens=seg_lens,
        base_output=base_output,
    )

    ref = _reference_sgemm_lora_b_fwd(
        input_x, weights, seg_indptr, weight_indices, scalings, base_output
    )

    out_cpu = out.cpu()

    assert out_cpu.shape == (num_tokens, output_dim)
    assert out_cpu.dtype == dtype
    rtol, atol = _tolerances(dtype)
    torch.testing.assert_close(out_cpu, ref, rtol=rtol, atol=atol)


# ----------------------------------------------------------------------------
# Correctness
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("output_dim", [64, 4096])
@pytest.mark.parametrize("max_rank", [8, 64])
def test_sgemm_lora_b_fwd_basic_shapes(dtype, output_dim, max_rank):
    num_tokens = 64
    num_loras = 3

    seg_indptr = torch.tensor([0, 16, 48, 64], dtype=torch.int32, device="xpu")
    weight_indices = torch.tensor([0, 2, 1], dtype=torch.int32, device="xpu")
    lora_ranks = torch.tensor(
        [max_rank, max(1, max_rank // 2), max(1, max_rank // 4)],
        dtype=torch.int32,
        device="xpu",
    )
    scalings = torch.tensor([0.5, 2.0, 1.25], dtype=torch.float32, device="xpu")

    _run_and_compare(
        dtype=dtype,
        num_tokens=num_tokens,
        max_rank=max_rank,
        output_dim=output_dim,
        num_loras=num_loras,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        scalings=scalings,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_base_output", [False, True])
def test_sgemm_lora_b_fwd_base_output_residual(dtype, use_base_output):
    """With base_output the kernel fuses D = scalings*(A@B^T) + base_output (beta=1)."""
    num_tokens = 48
    max_rank = 16
    output_dim = 256
    num_loras = 2

    seg_indptr = torch.tensor([0, 20, 48], dtype=torch.int32, device="xpu")
    weight_indices = torch.tensor([1, 0], dtype=torch.int32, device="xpu")
    lora_ranks = torch.tensor([max_rank, max_rank // 2], dtype=torch.int32, device="xpu")
    scalings = torch.tensor([1.5, 0.75], dtype=torch.float32, device="xpu")

    _run_and_compare(
        dtype=dtype,
        num_tokens=num_tokens,
        max_rank=max_rank,
        output_dim=output_dim,
        num_loras=num_loras,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        scalings=scalings,
        use_base_output=use_base_output,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sgemm_lora_b_fwd_per_segment_scalings(dtype):
    """Distinct per-adapter scalings on the same weights verify per-segment alpha."""
    num_tokens = 40
    max_rank = 8
    output_dim = 128
    num_loras = 4

    seg_indptr = torch.tensor([0, 10, 20, 30, 40], dtype=torch.int32, device="xpu")
    # Two segments share adapter 0 with the same weights: only scalings differ,
    # so the outputs must scale accordingly -- isolating the per-segment alpha.
    weight_indices = torch.tensor([0, 3, 0, 2], dtype=torch.int32, device="xpu")
    lora_ranks = torch.tensor([max_rank] * num_loras, dtype=torch.int32, device="xpu")
    scalings = torch.tensor([0.25, 4.0, 1.0, 2.5], dtype=torch.float32, device="xpu")

    _run_and_compare(
        dtype=dtype,
        num_tokens=num_tokens,
        max_rank=max_rank,
        output_dim=output_dim,
        num_loras=num_loras,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        scalings=scalings,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sgemm_lora_b_fwd_single_segment_single_lora(dtype):
    num_tokens = 128
    max_rank = 16
    output_dim = 512
    num_loras = 1

    seg_indptr = torch.tensor([0, num_tokens], dtype=torch.int32, device="xpu")
    weight_indices = torch.tensor([0], dtype=torch.int32, device="xpu")
    lora_ranks = torch.tensor([max_rank], dtype=torch.int32, device="xpu")
    scalings = torch.tensor([1.75], dtype=torch.float32, device="xpu")

    _run_and_compare(
        dtype=dtype,
        num_tokens=num_tokens,
        max_rank=max_rank,
        output_dim=output_dim,
        num_loras=num_loras,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        scalings=scalings,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sgemm_lora_b_fwd_single_token_segments(dtype):
    """One-token-per-segment stress: maximum segment count for the token count."""
    num_tokens = 64
    max_rank = 8
    output_dim = 128
    num_loras = 4

    seg_indptr = torch.arange(0, num_tokens + 1, dtype=torch.int32, device="xpu")
    weight_indices = (
        torch.arange(num_tokens, dtype=torch.int32, device="xpu") % num_loras
    ).to(torch.int32)
    lora_ranks = torch.tensor([1, 3, 5, 8], dtype=torch.int32, device="xpu")
    scalings = torch.tensor([2.0, 0.5, 1.0, 3.0], dtype=torch.float32, device="xpu")

    _run_and_compare(
        dtype=dtype,
        num_tokens=num_tokens,
        max_rank=max_rank,
        output_dim=output_dim,
        num_loras=num_loras,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        scalings=scalings,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [4096])
def test_sgemm_lora_b_fwd_large_num_tokens_many_segments(dtype, num_tokens):
    max_rank = 16
    output_dim = 512
    num_loras = 4

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
    scalings = torch.tensor([0.5, 1.0, 2.0, 1.5], dtype=torch.float32, device="xpu")

    _run_and_compare(
        dtype=dtype,
        num_tokens=num_tokens,
        max_rank=max_rank,
        output_dim=output_dim,
        num_loras=num_loras,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        scalings=scalings,
    )


# ----------------------------------------------------------------------------
# Rank padding / zero behaviour
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sgemm_lora_b_fwd_rank_padding_matches_reference(dtype):
    """Partial ranks (zeroed weight K-tail) must still match the fp32 reference."""
    num_tokens = 32
    max_rank = 8
    output_dim = 128
    num_loras = 2

    seg_indptr = torch.tensor([0, 16, 32], dtype=torch.int32, device="xpu")
    weight_indices = torch.tensor([0, 1], dtype=torch.int32, device="xpu")
    lora_ranks = torch.tensor([2, 5], dtype=torch.int32, device="xpu")
    scalings = torch.tensor([1.0, 2.0], dtype=torch.float32, device="xpu")

    _run_and_compare(
        dtype=dtype,
        num_tokens=num_tokens,
        max_rank=max_rank,
        output_dim=output_dim,
        num_loras=num_loras,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        scalings=scalings,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_base_output", [False, True])
def test_sgemm_lora_b_fwd_zero_rank(dtype, use_base_output):
    """rank == 0 zeroes the LoRA term -> output is base_output (or zero without it)."""
    torch.manual_seed(14)
    num_tokens = 16
    max_rank = 8
    output_dim = 64
    num_loras = 2

    seg_indptr = torch.tensor([0, 8, 16], dtype=torch.int32, device="xpu")
    weight_indices = torch.tensor([0, 1], dtype=torch.int32, device="xpu")
    lora_ranks = torch.tensor([0, 0], dtype=torch.int32, device="xpu")
    scalings = torch.tensor([2.0, 0.5], dtype=torch.float32, device="xpu")

    input_x = torch.randn(num_tokens, max_rank, dtype=dtype, device="xpu")
    weights = torch.randn(num_loras, output_dim, max_rank, dtype=dtype, device="xpu")
    weights = _zero_weight_rank_tail(weights, lora_ranks)

    base_output = None
    if use_base_output:
        base_output = torch.randn(num_tokens, output_dim, dtype=dtype, device="xpu")

    out = sgemm_lora_b_fwd(
        input_x=input_x,
        weights=weights,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        scalings=scalings,
        seg_lens=None,
        base_output=base_output,
    )

    out_cpu = out.cpu()
    if use_base_output:
        rtol, atol = _tolerances(dtype)
        torch.testing.assert_close(out_cpu, base_output.cpu(), rtol=rtol, atol=atol)
    else:
        assert torch.count_nonzero(out_cpu).item() == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sgemm_lora_b_fwd_segment_boundaries_precise_routing(dtype):
    """Variable segment sizes + non-identity weight_indices verifies routing."""
    max_rank = 8
    output_dim = 128
    num_loras = 3

    # Segment sizes: 3, 1, 5  ->  9 tokens total.
    seg_indptr = torch.tensor([0, 3, 4, 9], dtype=torch.int32, device="xpu")
    weight_indices = torch.tensor([2, 1, 0], dtype=torch.int32, device="xpu")
    lora_ranks = torch.tensor([8, 2, 4], dtype=torch.int32, device="xpu")
    scalings = torch.tensor([1.0, 3.0, 0.5], dtype=torch.float32, device="xpu")

    _run_and_compare(
        dtype=dtype,
        num_tokens=9,
        max_rank=max_rank,
        output_dim=output_dim,
        num_loras=num_loras,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        scalings=scalings,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sgemm_lora_b_fwd_empty_segments_mixed_in(dtype):
    """Empty segments (start == end) must be skipped without affecting neighbours."""
    max_rank = 8
    output_dim = 128
    num_loras = 2

    # Segment 1 is empty (indices 4 -> 4), segment 3 is empty (8 -> 8).
    seg_indptr = torch.tensor([0, 4, 4, 8, 8, 16], dtype=torch.int32, device="xpu")
    weight_indices = torch.tensor([0, 1, 0, 1, 0], dtype=torch.int32, device="xpu")
    lora_ranks = torch.tensor([8, 4], dtype=torch.int32, device="xpu")
    scalings = torch.tensor([1.5, 0.5], dtype=torch.float32, device="xpu")

    _run_and_compare(
        dtype=dtype,
        num_tokens=16,
        max_rank=max_rank,
        output_dim=output_dim,
        num_loras=num_loras,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        scalings=scalings,
    )


# ----------------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sgemm_lora_b_fwd_empty_input(dtype):
    """num_tokens == 0 must short-circuit cleanly and return a 0-row tensor."""
    max_rank = 8
    output_dim = 64
    num_loras = 1

    input_x = torch.empty((0, max_rank), dtype=dtype, device="xpu")
    weights = torch.randn(num_loras, output_dim, max_rank, dtype=dtype, device="xpu")
    seg_indptr = torch.tensor([0], dtype=torch.int32, device="xpu")
    weight_indices = torch.empty((0,), dtype=torch.int32, device="xpu")
    lora_ranks = torch.tensor([max_rank], dtype=torch.int32, device="xpu")
    scalings = torch.tensor([1.0], dtype=torch.float32, device="xpu")

    out = sgemm_lora_b_fwd(
        input_x=input_x,
        weights=weights,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        scalings=scalings,
        seg_lens=None,
        base_output=None,
    )

    assert out.shape == (0, output_dim)
    assert out.numel() == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_base_output", [False, True])
def test_sgemm_lora_b_fwd_zero_max_rank(dtype, use_base_output):
    """max_rank == 0 (K == 0) is an empty reduction -> output is base_output or zero."""
    torch.manual_seed(3)
    num_tokens = 32
    max_rank = 0
    output_dim = 64
    num_loras = 2

    input_x = torch.empty((num_tokens, max_rank), dtype=dtype, device="xpu")
    weights = torch.empty((num_loras, output_dim, max_rank), dtype=dtype, device="xpu")
    seg_indptr = torch.tensor([0, 16, 32], dtype=torch.int32, device="xpu")
    weight_indices = torch.tensor([0, 1], dtype=torch.int32, device="xpu")
    lora_ranks = torch.tensor([0, 0], dtype=torch.int32, device="xpu")
    scalings = torch.tensor([1.0, 2.0], dtype=torch.float32, device="xpu")

    base_output = None
    if use_base_output:
        base_output = torch.randn(num_tokens, output_dim, dtype=dtype, device="xpu")

    out = sgemm_lora_b_fwd(
        input_x=input_x,
        weights=weights,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        scalings=scalings,
        seg_lens=None,
        base_output=base_output,
    )

    assert out.shape == (num_tokens, output_dim)
    assert out.dtype == dtype
    if use_base_output:
        rtol, atol = _tolerances(dtype)
        torch.testing.assert_close(out.cpu(), base_output.cpu(), rtol=rtol, atol=atol)
    else:
        assert torch.count_nonzero(out).item() == 0


def test_sgemm_lora_b_fwd_int64_index_tensors_accepted():
    """seg_indptr / weight_indices in int64 should be cast internally and work."""
    dtype = torch.bfloat16
    num_tokens = 32
    max_rank = 8
    output_dim = 128
    num_loras = 2

    seg_indptr = torch.tensor([0, 16, 32], dtype=torch.int64, device="xpu")
    weight_indices = torch.tensor([1, 0], dtype=torch.int64, device="xpu")
    lora_ranks = torch.tensor([max_rank, max_rank // 2], dtype=torch.int32, device="xpu")
    scalings = torch.tensor([1.25, 0.75], dtype=torch.float32, device="xpu")

    _run_and_compare(
        dtype=dtype,
        num_tokens=num_tokens,
        max_rank=max_rank,
        output_dim=output_dim,
        num_loras=num_loras,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        scalings=scalings,
    )


# ----------------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------------


def _make_valid_kwargs():
    num_tokens = 4
    max_rank = 4  # K == input_x.size(1) == weights.size(2)
    output_dim = 8
    num_loras = 1

    return dict(
        input_x=torch.randn(num_tokens, max_rank, dtype=torch.float16, device="xpu"),
        weights=torch.randn(
            num_loras, output_dim, max_rank, dtype=torch.float16, device="xpu"
        ),
        seg_indptr=torch.tensor([0, num_tokens], dtype=torch.int32, device="xpu"),
        weight_indices=torch.tensor([0], dtype=torch.int32, device="xpu"),
        lora_ranks=torch.tensor([max_rank], dtype=torch.int32, device="xpu"),
        scalings=torch.tensor([1.0], dtype=torch.float32, device="xpu"),
        seg_lens=None,
        base_output=None,
    )


@pytest.mark.parametrize(
    "bad_case, expected_msg",
    [
        ("input_x_dim", "input_x must be a 2D tensor"),
        ("weights_dim", "weights must be a 3D tensor"),
        ("seg_indptr_dim", "seg_indptr must be a 1D tensor"),
        ("weight_indices_dim", "weight_indices must be a 1D tensor"),
        ("lora_ranks_dim", "lora_ranks must be a 1D tensor"),
        ("scalings_dim", "scalings must be a 1D tensor"),
        ("input_x_rank_mismatch", "input_x.size\\(1\\) must equal max_rank"),
        ("lora_ranks_size", "lora_ranks.numel\\(\\) must equal weights.size\\(0\\)"),
        ("scalings_size", "scalings.numel\\(\\) must equal weights.size\\(0\\)"),
        (
            "weight_indices_size",
            "weight_indices.numel\\(\\) must equal seg_indptr.numel\\(\\) - 1",
        ),
        ("weight_indices_out_of_range", "weight_indices values must be in"),
        ("seg_indptr_start_nonzero", "seg_indptr\\[0\\] must be 0"),
        ("seg_indptr_end_mismatch", "seg_indptr\\[-1\\] must equal num_tokens"),
        ("seg_indptr_decreasing", "seg_indptr must be non-decreasing"),
        ("lora_ranks_out_of_range", "lora_ranks must be within the range"),
        ("dtype_mismatch", "Input tensor dtype must match weights dtype"),
        ("base_output_dim", "base_output must be a 2D tensor"),
        ("base_output_shape", "base_output must have shape"),
        ("base_output_dtype", "base_output dtype must match weights dtype"),
    ],
)
def test_sgemm_lora_b_fwd_input_validation(bad_case, expected_msg):
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
    elif bad_case == "scalings_dim":
        kwargs["scalings"] = kwargs["scalings"].view(1, 1)
    elif bad_case == "input_x_rank_mismatch":
        # input_x.size(1) must equal weights.size(2) == max_rank (4).
        kwargs["input_x"] = torch.randn(4, 5, dtype=torch.float16, device="xpu")
    elif bad_case == "lora_ranks_size":
        kwargs["lora_ranks"] = torch.tensor([4, 4], dtype=torch.int32, device="xpu")
    elif bad_case == "scalings_size":
        kwargs["scalings"] = torch.tensor([1.0, 1.0], dtype=torch.float32, device="xpu")
    elif bad_case == "weight_indices_size":
        kwargs["weight_indices"] = torch.tensor([0, 0], dtype=torch.int32, device="xpu")
    elif bad_case == "weight_indices_out_of_range":
        kwargs["weight_indices"] = torch.tensor([5], dtype=torch.int32, device="xpu")
    elif bad_case == "seg_indptr_start_nonzero":
        kwargs["seg_indptr"] = torch.tensor([1, 4], dtype=torch.int32, device="xpu")
    elif bad_case == "seg_indptr_end_mismatch":
        kwargs["seg_indptr"] = torch.tensor([0, 3], dtype=torch.int32, device="xpu")
    elif bad_case == "seg_indptr_decreasing":
        # 3-segment indptr, decreasing in the middle (4 -> 2).
        kwargs["seg_indptr"] = torch.tensor([0, 4, 2, 4], dtype=torch.int32, device="xpu")
        kwargs["weight_indices"] = torch.tensor(
            [0, 0, 0], dtype=torch.int32, device="xpu"
        )
    elif bad_case == "lora_ranks_out_of_range":
        # max_rank = weights.size(2) = 4
        kwargs["lora_ranks"] = torch.tensor([5], dtype=torch.int32, device="xpu")
    elif bad_case == "dtype_mismatch":
        kwargs["input_x"] = kwargs["input_x"].to(torch.bfloat16)
    elif bad_case == "base_output_dim":
        kwargs["base_output"] = torch.randn(4 * 8, dtype=torch.float16, device="xpu")
    elif bad_case == "base_output_shape":
        kwargs["base_output"] = torch.randn(4, 16, dtype=torch.float16, device="xpu")
    elif bad_case == "base_output_dtype":
        kwargs["base_output"] = torch.randn(4, 8, dtype=torch.bfloat16, device="xpu")

    with pytest.raises(RuntimeError, match=expected_msg):
        sgemm_lora_b_fwd(**kwargs)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
