import sys
from typing import Optional

import pytest
import torch
from sgl_kernel import qkv_lora_b_fwd

if not torch.xpu.is_available():
    pytest.skip(reason="qkv_lora_b_fwd requires XPU device.", allow_module_level=True)


def _tolerances(dtype: torch.dtype):
    if dtype == torch.float16:
        return 1e-2, 1e-2
    if dtype == torch.bfloat16:
        return 2e-2, 2e-2
    return 1e-5, 1e-5


def _make_output_offset(n_q: int, n_kv: int) -> torch.Tensor:
    """Build the q/k/v output-column boundaries [0, N_Q, N_Q+N_KV, N_Q+2*N_KV]."""
    return torch.tensor(
        [0, n_q, n_q + n_kv, n_q + 2 * n_kv], dtype=torch.int32, device="xpu"
    )


def _zero_weight_rank_tail(
    weights: torch.Tensor, lora_ranks: torch.Tensor
) -> torch.Tensor:
    """Zero the weight columns beyond ``lora_ranks[lora]`` in the reduction dim.

    For LoRA-B the rank axis is the reduction (K) dimension -- ``qkv_lora_b`` is
    ``[num_loras, N_total, max_rank]`` -- so an adapter of rank ``R_l`` only
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


def _reference_qkv_lora_b_fwd(
    input_x: torch.Tensor,
    qkv_lora_b: torch.Tensor,
    output_offset: torch.Tensor,
    seg_indptr: torch.Tensor,
    weight_indices: torch.Tensor,
    scalings: torch.Tensor,
    base_output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reference computed on-device (XPU) in fp32, narrowed to weight dtype.

    Implements, per segment ``s`` (adapter ``l``) and projection ``p in {q,k,v}``::

        out[i, band_p] = scalings[l] * (input_x[i, band_p] @ qkv_lora_b[l, band_p]^T)
                         + base_output[i, band_p]

    where ``band_p`` is the input K-band ``[p*K, (p+1)*K)`` and the output/weight
    band ``[output_offset[p], output_offset[p+1])``; the residual term is only
    added when ``base_output`` is supplied.
    """
    device = input_x.device
    scal = scalings.float()

    num_tokens = input_x.size(0)
    n_total = qkv_lora_b.size(1)
    max_rank = qkv_lora_b.size(2)

    if base_output is not None:
        out = base_output.float().clone()
    else:
        out = torch.zeros((num_tokens, n_total), dtype=torch.float32, device=device)

    seg_cpu = seg_indptr.cpu()
    wi_cpu = weight_indices.cpu()
    oo = output_offset.cpu().tolist()

    num_segments = seg_cpu.numel() - 1
    for s in range(num_segments):
        start = int(seg_cpu[s].item())
        end = int(seg_cpu[s + 1].item())
        if end == start:
            continue
        lora = int(wi_cpu[s].item())
        for p in range(3):
            in_band = input_x[start:end, p * max_rank : (p + 1) * max_rank].float()
            w = qkv_lora_b[lora, oo[p] : oo[p + 1], :].float()  # [N_p, max_rank]
            out[start:end, oo[p] : oo[p + 1]] += scal[lora] * (in_band @ w.T)
    return out.to(qkv_lora_b.dtype)


def _run_and_compare(
    *,
    dtype: torch.dtype,
    num_tokens: int,
    max_rank: int,
    n_q: int,
    n_kv: int,
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

    n_total = n_q + 2 * n_kv
    output_offset = _make_output_offset(n_q, n_kv)
    max_qkv_out_dim = max(n_q, n_kv)

    # input_x is the LoRA-A projection packed for q/k/v: [num_tokens, 3 * max_rank].
    input_x = torch.randn(num_tokens, 3 * max_rank, dtype=dtype, device="xpu")
    qkv_lora_b = torch.randn(num_loras, n_total, max_rank, dtype=dtype, device="xpu")
    qkv_lora_b = _zero_weight_rank_tail(qkv_lora_b, lora_ranks)

    base_output = None
    if use_base_output:
        base_output = torch.randn(num_tokens, n_total, dtype=dtype, device="xpu")

    out = qkv_lora_b_fwd(
        input_x=input_x,
        qkv_lora_b=qkv_lora_b,
        output_offset=output_offset,
        max_qkv_out_dim=max_qkv_out_dim,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        scalings=scalings,
        seg_lens=seg_lens,
        base_output=base_output,
    )

    ref = _reference_qkv_lora_b_fwd(
        input_x,
        qkv_lora_b,
        output_offset,
        seg_indptr,
        weight_indices,
        scalings,
        base_output,
    )

    assert out.shape == (num_tokens, n_total)
    assert out.dtype == dtype
    rtol, atol = _tolerances(dtype)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


# ----------------------------------------------------------------------------
# Correctness
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("n_q,n_kv", [(64, 16), (4096, 512), (256, 256)])
@pytest.mark.parametrize("max_rank", [8, 64])
def test_qkv_lora_b_fwd_basic_shapes(dtype, n_q, n_kv, max_rank):
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
        n_q=n_q,
        n_kv=n_kv,
        num_loras=num_loras,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        scalings=scalings,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_base_output", [False, True])
def test_qkv_lora_b_fwd_base_output_residual(dtype, use_base_output):
    """With base_output the kernel fuses D = scalings*(A@B^T) + base_output (beta=1)."""
    num_tokens = 48
    max_rank = 16
    n_q, n_kv = 256, 64
    num_loras = 2

    seg_indptr = torch.tensor([0, 20, 48], dtype=torch.int32, device="xpu")
    weight_indices = torch.tensor([1, 0], dtype=torch.int32, device="xpu")
    lora_ranks = torch.tensor(
        [max_rank, max_rank // 2], dtype=torch.int32, device="xpu"
    )
    scalings = torch.tensor([1.5, 0.75], dtype=torch.float32, device="xpu")

    _run_and_compare(
        dtype=dtype,
        num_tokens=num_tokens,
        max_rank=max_rank,
        n_q=n_q,
        n_kv=n_kv,
        num_loras=num_loras,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        scalings=scalings,
        use_base_output=use_base_output,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_qkv_lora_b_fwd_per_segment_scalings(dtype):
    """Distinct per-adapter scalings on the same weights verify per-group alpha."""
    num_tokens = 40
    max_rank = 8
    n_q, n_kv = 128, 32
    num_loras = 4

    seg_indptr = torch.tensor([0, 10, 20, 30, 40], dtype=torch.int32, device="xpu")
    # Two segments share adapter 0 with the same weights: only scalings differ.
    weight_indices = torch.tensor([0, 3, 0, 2], dtype=torch.int32, device="xpu")
    lora_ranks = torch.tensor([max_rank] * num_loras, dtype=torch.int32, device="xpu")
    scalings = torch.tensor([0.25, 4.0, 1.0, 2.5], dtype=torch.float32, device="xpu")

    _run_and_compare(
        dtype=dtype,
        num_tokens=num_tokens,
        max_rank=max_rank,
        n_q=n_q,
        n_kv=n_kv,
        num_loras=num_loras,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        scalings=scalings,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_qkv_lora_b_fwd_single_segment_single_lora(dtype):
    num_tokens = 128
    max_rank = 16
    n_q, n_kv = 512, 128
    num_loras = 1

    seg_indptr = torch.tensor([0, num_tokens], dtype=torch.int32, device="xpu")
    weight_indices = torch.tensor([0], dtype=torch.int32, device="xpu")
    lora_ranks = torch.tensor([max_rank], dtype=torch.int32, device="xpu")
    scalings = torch.tensor([1.75], dtype=torch.float32, device="xpu")

    _run_and_compare(
        dtype=dtype,
        num_tokens=num_tokens,
        max_rank=max_rank,
        n_q=n_q,
        n_kv=n_kv,
        num_loras=num_loras,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        scalings=scalings,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_qkv_lora_b_fwd_single_token_segments(dtype):
    """One-token-per-segment stress: maximum segment count for the token count."""
    num_tokens = 64
    max_rank = 8
    n_q, n_kv = 128, 32
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
        n_q=n_q,
        n_kv=n_kv,
        num_loras=num_loras,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        scalings=scalings,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [4096])
def test_qkv_lora_b_fwd_large_num_tokens_many_segments(dtype, num_tokens):
    max_rank = 16
    n_q, n_kv = 512, 128
    num_loras = 4

    torch.manual_seed(7)
    num_segments = 17
    raw = torch.randint(1, 32, (num_segments,), dtype=torch.int32)
    total = int(raw.sum().item())
    raw = (raw.float() * (num_tokens / total)).round().to(torch.int32)
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
        n_q=n_q,
        n_kv=n_kv,
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
def test_qkv_lora_b_fwd_rank_padding_matches_reference(dtype):
    """Partial ranks (zeroed weight K-tail) must still match the fp32 reference."""
    num_tokens = 32
    max_rank = 8
    n_q, n_kv = 128, 32
    num_loras = 2

    seg_indptr = torch.tensor([0, 16, 32], dtype=torch.int32, device="xpu")
    weight_indices = torch.tensor([0, 1], dtype=torch.int32, device="xpu")
    lora_ranks = torch.tensor([2, 5], dtype=torch.int32, device="xpu")
    scalings = torch.tensor([1.0, 2.0], dtype=torch.float32, device="xpu")

    _run_and_compare(
        dtype=dtype,
        num_tokens=num_tokens,
        max_rank=max_rank,
        n_q=n_q,
        n_kv=n_kv,
        num_loras=num_loras,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        scalings=scalings,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_base_output", [False, True])
def test_qkv_lora_b_fwd_zero_rank(dtype, use_base_output):
    """rank == 0 zeroes the LoRA term -> output is base_output (or zero without it)."""
    torch.manual_seed(14)
    num_tokens = 16
    max_rank = 8
    n_q, n_kv = 64, 16
    n_total = n_q + 2 * n_kv
    num_loras = 2

    output_offset = _make_output_offset(n_q, n_kv)
    seg_indptr = torch.tensor([0, 8, 16], dtype=torch.int32, device="xpu")
    weight_indices = torch.tensor([0, 1], dtype=torch.int32, device="xpu")
    lora_ranks = torch.tensor([0, 0], dtype=torch.int32, device="xpu")
    scalings = torch.tensor([2.0, 0.5], dtype=torch.float32, device="xpu")

    input_x = torch.randn(num_tokens, 3 * max_rank, dtype=dtype, device="xpu")
    qkv_lora_b = torch.randn(num_loras, n_total, max_rank, dtype=dtype, device="xpu")
    qkv_lora_b = _zero_weight_rank_tail(qkv_lora_b, lora_ranks)

    base_output = None
    if use_base_output:
        base_output = torch.randn(num_tokens, n_total, dtype=dtype, device="xpu")

    out = qkv_lora_b_fwd(
        input_x=input_x,
        qkv_lora_b=qkv_lora_b,
        output_offset=output_offset,
        max_qkv_out_dim=max(n_q, n_kv),
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
def test_qkv_lora_b_fwd_segment_boundaries_precise_routing(dtype):
    """Variable segment sizes + non-identity weight_indices verifies routing."""
    max_rank = 8
    n_q, n_kv = 128, 32
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
        n_q=n_q,
        n_kv=n_kv,
        num_loras=num_loras,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        scalings=scalings,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_qkv_lora_b_fwd_empty_segments_mixed_in(dtype):
    """Empty segments (start == end) must be skipped without affecting neighbours."""
    max_rank = 8
    n_q, n_kv = 128, 32
    num_loras = 2

    seg_indptr = torch.tensor([0, 4, 4, 8, 8, 16], dtype=torch.int32, device="xpu")
    weight_indices = torch.tensor([0, 1, 0, 1, 0], dtype=torch.int32, device="xpu")
    lora_ranks = torch.tensor([8, 4], dtype=torch.int32, device="xpu")
    scalings = torch.tensor([1.5, 0.5], dtype=torch.float32, device="xpu")

    _run_and_compare(
        dtype=dtype,
        num_tokens=16,
        max_rank=max_rank,
        n_q=n_q,
        n_kv=n_kv,
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
def test_qkv_lora_b_fwd_empty_input(dtype):
    """num_tokens == 0 must short-circuit cleanly and return a 0-row tensor."""
    max_rank = 8
    n_q, n_kv = 64, 16
    n_total = n_q + 2 * n_kv
    num_loras = 1

    input_x = torch.empty((0, 3 * max_rank), dtype=dtype, device="xpu")
    qkv_lora_b = torch.randn(num_loras, n_total, max_rank, dtype=dtype, device="xpu")
    output_offset = _make_output_offset(n_q, n_kv)
    seg_indptr = torch.tensor([0], dtype=torch.int32, device="xpu")
    weight_indices = torch.empty((0,), dtype=torch.int32, device="xpu")
    lora_ranks = torch.tensor([max_rank], dtype=torch.int32, device="xpu")
    scalings = torch.tensor([1.0], dtype=torch.float32, device="xpu")

    out = qkv_lora_b_fwd(
        input_x=input_x,
        qkv_lora_b=qkv_lora_b,
        output_offset=output_offset,
        max_qkv_out_dim=max(n_q, n_kv),
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        scalings=scalings,
        seg_lens=None,
        base_output=None,
    )

    assert out.shape == (0, n_total)
    assert out.numel() == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_base_output", [False, True])
def test_qkv_lora_b_fwd_zero_max_rank(dtype, use_base_output):
    """max_rank == 0 (K == 0) is an empty reduction -> output is base_output or zero."""
    torch.manual_seed(3)
    num_tokens = 32
    max_rank = 0
    n_q, n_kv = 64, 16
    n_total = n_q + 2 * n_kv
    num_loras = 2

    input_x = torch.empty((num_tokens, 3 * max_rank), dtype=dtype, device="xpu")
    qkv_lora_b = torch.empty((num_loras, n_total, max_rank), dtype=dtype, device="xpu")
    output_offset = _make_output_offset(n_q, n_kv)
    seg_indptr = torch.tensor([0, 16, 32], dtype=torch.int32, device="xpu")
    weight_indices = torch.tensor([0, 1], dtype=torch.int32, device="xpu")
    lora_ranks = torch.tensor([0, 0], dtype=torch.int32, device="xpu")
    scalings = torch.tensor([1.0, 2.0], dtype=torch.float32, device="xpu")

    base_output = None
    if use_base_output:
        base_output = torch.randn(num_tokens, n_total, dtype=dtype, device="xpu")

    out = qkv_lora_b_fwd(
        input_x=input_x,
        qkv_lora_b=qkv_lora_b,
        output_offset=output_offset,
        max_qkv_out_dim=max(n_q, n_kv),
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        scalings=scalings,
        seg_lens=None,
        base_output=base_output,
    )

    assert out.shape == (num_tokens, n_total)
    assert out.dtype == dtype
    if use_base_output:
        rtol, atol = _tolerances(dtype)
        torch.testing.assert_close(out.cpu(), base_output.cpu(), rtol=rtol, atol=atol)
    else:
        assert torch.count_nonzero(out).item() == 0


def test_qkv_lora_b_fwd_int64_index_tensors_accepted():
    """seg_indptr / weight_indices / output_offset in int64 should be cast internally."""
    dtype = torch.bfloat16
    num_tokens = 32
    max_rank = 8
    n_q, n_kv = 128, 32
    n_total = n_q + 2 * n_kv
    num_loras = 2

    output_offset = torch.tensor(
        [0, n_q, n_q + n_kv, n_total], dtype=torch.int64, device="xpu"
    )
    seg_indptr = torch.tensor([0, 16, 32], dtype=torch.int64, device="xpu")
    weight_indices = torch.tensor([1, 0], dtype=torch.int64, device="xpu")
    lora_ranks = torch.tensor(
        [max_rank, max_rank // 2], dtype=torch.int32, device="xpu"
    )
    scalings = torch.tensor([1.25, 0.75], dtype=torch.float32, device="xpu")

    torch.manual_seed(0)
    input_x = torch.randn(num_tokens, 3 * max_rank, dtype=dtype, device="xpu")
    qkv_lora_b = torch.randn(num_loras, n_total, max_rank, dtype=dtype, device="xpu")
    qkv_lora_b = _zero_weight_rank_tail(qkv_lora_b, lora_ranks)

    out = qkv_lora_b_fwd(
        input_x=input_x,
        qkv_lora_b=qkv_lora_b,
        output_offset=output_offset,
        max_qkv_out_dim=max(n_q, n_kv),
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        scalings=scalings,
    )
    ref = _reference_qkv_lora_b_fwd(
        input_x, qkv_lora_b, output_offset, seg_indptr, weight_indices, scalings
    )
    rtol, atol = _tolerances(dtype)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


# ----------------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------------


def _make_valid_kwargs():
    num_tokens = 4
    max_rank = 4  # K; input_x.size(1) == 3 * max_rank, qkv_lora_b.size(2) == max_rank
    n_q, n_kv = 8, 4
    n_total = n_q + 2 * n_kv
    num_loras = 1

    return dict(
        input_x=torch.randn(num_tokens, 3 * max_rank, dtype=torch.float16, device="xpu"),
        qkv_lora_b=torch.randn(
            num_loras, n_total, max_rank, dtype=torch.float16, device="xpu"
        ),
        output_offset=torch.tensor(
            [0, n_q, n_q + n_kv, n_total], dtype=torch.int32, device="xpu"
        ),
        max_qkv_out_dim=max(n_q, n_kv),
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
        ("qkv_lora_b_dim", "qkv_lora_b must be a 3D tensor"),
        ("output_offset_dim", "output_offset must be a 1D tensor"),
        ("output_offset_numel", "output_offset must have 4 elements"),
        ("seg_indptr_dim", "seg_indptr must be a 1D tensor"),
        ("weight_indices_dim", "weight_indices must be a 1D tensor"),
        ("lora_ranks_dim", "lora_ranks must be a 1D tensor"),
        ("scalings_dim", "scalings must be a 1D tensor"),
        ("input_x_rank_mismatch", "input_x.size\\(1\\) must equal 3 \\* max_rank"),
        ("lora_ranks_size", "lora_ranks.numel\\(\\) must equal qkv_lora_b.size\\(0\\)"),
        ("scalings_size", "scalings.numel\\(\\) must equal qkv_lora_b.size\\(0\\)"),
        (
            "weight_indices_size",
            "weight_indices.numel\\(\\) must equal seg_indptr.numel\\(\\) - 1",
        ),
        ("weight_indices_out_of_range", "weight_indices values must be in"),
        ("output_offset_start", "output_offset\\[0\\] must be 0"),
        ("output_offset_end", "output_offset\\[-1\\] must equal"),
        ("max_qkv_out_dim_mismatch", "max_qkv_out_dim must equal"),
        ("seg_indptr_start_nonzero", "seg_indptr\\[0\\] must be 0"),
        ("seg_indptr_end_mismatch", "seg_indptr\\[-1\\] must equal num_tokens"),
        ("lora_ranks_out_of_range", "lora_ranks must be within the range"),
        ("dtype_mismatch", "Input tensor dtype must match qkv_lora_b dtype"),
        ("base_output_dim", "base_output must be a 2D tensor"),
        ("base_output_shape", "base_output must have shape"),
        ("base_output_dtype", "base_output dtype must match qkv_lora_b dtype"),
    ],
)
def test_qkv_lora_b_fwd_input_validation(bad_case, expected_msg):
    kwargs = _make_valid_kwargs()

    if bad_case == "input_x_dim":
        kwargs["input_x"] = kwargs["input_x"].view(-1)
    elif bad_case == "qkv_lora_b_dim":
        kwargs["qkv_lora_b"] = kwargs["qkv_lora_b"].view(1, -1)
    elif bad_case == "output_offset_dim":
        kwargs["output_offset"] = kwargs["output_offset"].view(1, -1)
    elif bad_case == "output_offset_numel":
        kwargs["output_offset"] = torch.tensor(
            [0, 8, 16], dtype=torch.int32, device="xpu"
        )
    elif bad_case == "seg_indptr_dim":
        kwargs["seg_indptr"] = kwargs["seg_indptr"].view(1, -1)
    elif bad_case == "weight_indices_dim":
        kwargs["weight_indices"] = kwargs["weight_indices"].view(1, 1)
    elif bad_case == "lora_ranks_dim":
        kwargs["lora_ranks"] = kwargs["lora_ranks"].view(1, 1)
    elif bad_case == "scalings_dim":
        kwargs["scalings"] = kwargs["scalings"].view(1, 1)
    elif bad_case == "input_x_rank_mismatch":
        # input_x.size(1) must equal 3 * max_rank (12).
        kwargs["input_x"] = torch.randn(4, 10, dtype=torch.float16, device="xpu")
    elif bad_case == "lora_ranks_size":
        kwargs["lora_ranks"] = torch.tensor([4, 4], dtype=torch.int32, device="xpu")
    elif bad_case == "scalings_size":
        kwargs["scalings"] = torch.tensor([1.0, 1.0], dtype=torch.float32, device="xpu")
    elif bad_case == "weight_indices_size":
        kwargs["weight_indices"] = torch.tensor([0, 0], dtype=torch.int32, device="xpu")
    elif bad_case == "weight_indices_out_of_range":
        kwargs["weight_indices"] = torch.tensor([5], dtype=torch.int32, device="xpu")
    elif bad_case == "output_offset_start":
        kwargs["output_offset"] = torch.tensor(
            [1, 8, 12, 16], dtype=torch.int32, device="xpu"
        )
    elif bad_case == "output_offset_end":
        kwargs["output_offset"] = torch.tensor(
            [0, 8, 12, 99], dtype=torch.int32, device="xpu"
        )
    elif bad_case == "max_qkv_out_dim_mismatch":
        kwargs["max_qkv_out_dim"] = 3
    elif bad_case == "seg_indptr_start_nonzero":
        kwargs["seg_indptr"] = torch.tensor([1, 4], dtype=torch.int32, device="xpu")
    elif bad_case == "seg_indptr_end_mismatch":
        kwargs["seg_indptr"] = torch.tensor([0, 3], dtype=torch.int32, device="xpu")
    elif bad_case == "lora_ranks_out_of_range":
        # max_rank = qkv_lora_b.size(2) = 4
        kwargs["lora_ranks"] = torch.tensor([5], dtype=torch.int32, device="xpu")
    elif bad_case == "dtype_mismatch":
        kwargs["input_x"] = kwargs["input_x"].to(torch.bfloat16)
    elif bad_case == "base_output_dim":
        kwargs["base_output"] = torch.randn(4 * 16, dtype=torch.float16, device="xpu")
    elif bad_case == "base_output_shape":
        kwargs["base_output"] = torch.randn(4, 99, dtype=torch.float16, device="xpu")
    elif bad_case == "base_output_dtype":
        kwargs["base_output"] = torch.randn(4, 16, dtype=torch.bfloat16, device="xpu")

    with pytest.raises(RuntimeError, match=expected_msg):
        qkv_lora_b_fwd(**kwargs)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
