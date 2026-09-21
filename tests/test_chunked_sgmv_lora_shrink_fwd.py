import sys
from typing import List

import pytest
import torch
from sgl_kernel import (
    chunked_sgmv_lora_shrink_forward,
    lora_gather_rows,
    lora_scatter_rows,
    sgemm_lora_a_fwd,
)

if not torch.xpu.is_available():
    pytest.skip(
        reason="chunked_sgmv_lora_shrink_forward requires XPU device.",
        allow_module_level=True,
    )


def _tolerances(dtype: torch.dtype):
    if dtype == torch.float16:
        return 1e-2, 1e-2
    if dtype == torch.bfloat16:
        return 2e-2, 2e-2
    return 1e-5, 1e-5


# ----------------------------------------------------------------------------
# Helpers (mirror the three-kernel decomposition on CPU in fp32)
# ----------------------------------------------------------------------------


def _zero_weight_rank_tail(
    weights: torch.Tensor, lora_ranks: torch.Tensor, stack_num: int
) -> torch.Tensor:
    """Zero-pad weight rows beyond ``lora_ranks[lora]`` per stack (kernel contract)."""
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


def _reference_sgemm(
    input_x: torch.Tensor,
    weights: torch.Tensor,
    seg_indptr: torch.Tensor,
    weight_indices: torch.Tensor,
) -> torch.Tensor:
    """Segmented grouped GEMM computed on-device (XPU) in fp32"""
    device = input_x.device
    num_tokens = input_x.size(0)
    total_n = weights.size(1)
    out = torch.zeros((num_tokens, total_n), dtype=torch.float32, device=device)

    # seg_indptr / weight_indices are tiny; read them on the host to bound the
    # loop (the heavy matmuls stay on device).
    seg_cpu = seg_indptr.cpu()
    wi_cpu = weight_indices.cpu()

    for s in range(seg_cpu.numel() - 1):
        start = int(seg_cpu[s].item())
        end = int(seg_cpu[s + 1].item())
        if end == start:
            continue
        lora = int(wi_cpu[s].item())
        x = input_x[start:end].float()
        w = weights[lora].float()
        out[start:end] = x @ w.T
    return out.to(weights.dtype)


def _reference_chunked_shrink(
    input_x: torch.Tensor,
    weights: torch.Tensor,
    seg_indptr: torch.Tensor,
    weight_indices: torch.Tensor,
    permutation: torch.Tensor,
) -> torch.Tensor:
    """gather (physical->logical) -> sgemm -> scatter (logical->physical), on-device (XPU)."""
    perm = permutation.to(torch.int64)
    x_sorted = input_x[perm]  # x_sorted[i] = x[perm[i]]
    out_sorted = _reference_sgemm(x_sorted, weights, seg_indptr, weight_indices)
    out = torch.empty_like(out_sorted)
    out[perm] = out_sorted  # out[perm[i]] = out_sorted[i]
    return out


def _build_logical_segments(row_adapters: List[int]):
    """Build (permutation, seg_indptr, weight_indices) from a physical-order adapter list.

    Mirrors ChunkedLoraBackend._get_permutation / _get_segments_info: sort rows by
    adapter (stable argsort => logical order), then run-length encode into segments.
    """
    ra = torch.tensor(row_adapters, dtype=torch.int32)
    permutation = torch.argsort(ra, stable=True).to(torch.int32)  # logical -> physical
    reordered = ra[permutation.to(torch.int64)]
    uniq, counts = torch.unique_consecutive(reordered, return_counts=True)
    seg_indptr = torch.zeros(uniq.numel() + 1, dtype=torch.int32)
    seg_indptr[1:] = torch.cumsum(counts, dim=0).to(torch.int32)
    weight_indices = uniq.to(torch.int32)
    return permutation, seg_indptr, weight_indices


# ----------------------------------------------------------------------------
# gather / scatter primitives
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("width", [1, 8, 4096])
def test_gather_matches_indexing(dtype, width):
    torch.manual_seed(0)
    num_rows = 37
    x = torch.randn(num_rows, width, dtype=dtype, device="xpu")
    perm = torch.randperm(num_rows, device="xpu").to(torch.int32)

    out = lora_gather_rows(x, perm)

    ref = x.cpu()[perm.cpu().to(torch.int64)]
    torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("width", [1, 8, 4096])
def test_scatter_matches_indexing(dtype, width):
    torch.manual_seed(1)
    num_rows = 37
    x = torch.randn(num_rows, width, dtype=dtype, device="xpu")
    perm = torch.randperm(num_rows, device="xpu").to(torch.int32)

    out = lora_scatter_rows(x, perm)

    ref = torch.empty_like(x.cpu())
    ref[perm.cpu().to(torch.int64)] = x.cpu()
    torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_gather_scatter_round_trip(dtype):
    """scatter(gather(x, p), p) == x for any permutation p."""
    torch.manual_seed(2)
    num_rows, width = 64, 128
    x = torch.randn(num_rows, width, dtype=dtype, device="xpu")
    perm = torch.randperm(num_rows, device="xpu").to(torch.int64)

    round_trip = lora_scatter_rows(lora_gather_rows(x, perm), perm)
    torch.testing.assert_close(round_trip.cpu(), x.cpu(), rtol=0, atol=0)


def test_gather_int64_permutation_accepted():
    x = torch.randn(8, 16, dtype=torch.float16, device="xpu")
    perm = torch.randperm(8, device="xpu").to(torch.int64)
    out = lora_gather_rows(x, perm)
    torch.testing.assert_close(out.cpu(), x.cpu()[perm.cpu()], rtol=0, atol=0)


@pytest.mark.parametrize(
    "bad_case, expected_msg",
    [
        ("input_dim", "input must be a 2D tensor"),
        ("perm_dim", "permutation must be a 1D tensor"),
        ("perm_size", "permutation.numel\\(\\) must equal input.size\\(0\\)"),
        ("perm_out_of_range", "permutation values must be in"),
    ],
)
def test_gather_input_validation(bad_case, expected_msg):
    x = torch.randn(8, 16, dtype=torch.float16, device="xpu")
    perm = torch.randperm(8, device="xpu").to(torch.int32)

    if bad_case == "input_dim":
        x = x.view(-1)
    elif bad_case == "perm_dim":
        perm = perm.view(1, -1)
    elif bad_case == "perm_size":
        perm = perm[:4]
    elif bad_case == "perm_out_of_range":
        perm = perm.clone()
        perm[0] = 8

    with pytest.raises(RuntimeError, match=expected_msg):
        lora_gather_rows(x, perm)


# ----------------------------------------------------------------------------
# chunked_sgmv_lora_shrink_forward (three-kernel orchestration)
# ----------------------------------------------------------------------------


def _run_and_compare_chunked(
    *,
    dtype: torch.dtype,
    row_adapters: List[int],
    input_dim: int,
    max_rank: int,
    stack_num: int,
    num_loras: int,
    lora_ranks: torch.Tensor,
) -> None:
    torch.manual_seed(0)
    num_tokens = len(row_adapters)
    total_n = stack_num * max_rank

    permutation, seg_indptr, weight_indices = _build_logical_segments(row_adapters)
    permutation = permutation.to("xpu")
    seg_indptr = seg_indptr.to("xpu")
    weight_indices = weight_indices.to("xpu")

    input_x = torch.randn(num_tokens, input_dim, dtype=dtype, device="xpu")
    weights = torch.randn(num_loras, total_n, input_dim, dtype=dtype, device="xpu")
    weights = _zero_weight_rank_tail(weights, lora_ranks, stack_num)

    out = chunked_sgmv_lora_shrink_forward(
        input_x=input_x,
        weights=weights,
        stack_num=stack_num,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        permutation=permutation,
    )

    ref = _reference_chunked_shrink(
        input_x, weights, seg_indptr, weight_indices, permutation
    )

    assert out.shape == (num_tokens, total_n)
    assert out.dtype == dtype
    # Compare on device: both `out` and `ref` live on XPU, so torch.testing
    # validates them without a host round-trip.
    rtol, atol = _tolerances(dtype)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("input_dim", [64, 4096])
@pytest.mark.parametrize("max_rank", [8, 64])
def test_chunked_shrink_zigzag_decode(dtype, input_dim, max_rank):
    """Interleaved (zigzag) adapters across single-token decode rows."""
    num_loras = 3
    row_adapters = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]  # zigzag
    lora_ranks = torch.tensor(
        [max_rank, max(1, max_rank // 2), max(1, max_rank // 4)],
        dtype=torch.int32,
        device="xpu",
    )
    _run_and_compare_chunked(
        dtype=dtype,
        row_adapters=row_adapters,
        input_dim=input_dim,
        max_rank=max_rank,
        stack_num=1,
        num_loras=num_loras,
        lora_ranks=lora_ranks,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("stack_num", [1, 2, 3])
def test_chunked_shrink_stack_num(dtype, stack_num):
    num_loras = 2
    max_rank = 8
    row_adapters = [0, 1, 0, 1, 0, 1, 0, 1]
    lora_ranks = torch.tensor(
        [max_rank, max_rank // 2], dtype=torch.int32, device="xpu"
    )
    _run_and_compare_chunked(
        dtype=dtype,
        row_adapters=row_adapters,
        input_dim=256,
        max_rank=max_rank,
        stack_num=stack_num,
        num_loras=num_loras,
        lora_ranks=lora_ranks,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_chunked_shrink_zero_rank_adapter(dtype):
    """An adapter with rank 0 must contribute an all-zero output for its rows."""
    num_loras = 2
    max_rank = 8
    row_adapters = [0, 1, 0, 1, 0, 1]
    lora_ranks = torch.tensor([0, max_rank], dtype=torch.int32, device="xpu")
    _run_and_compare_chunked(
        dtype=dtype,
        row_adapters=row_adapters,
        input_dim=128,
        max_rank=max_rank,
        stack_num=1,
        num_loras=num_loras,
        lora_ranks=lora_ranks,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_chunked_shrink_large_many_adapters(dtype):
    torch.manual_seed(7)
    num_loras = 4
    max_rank = 16
    num_tokens = 512
    row_adapters = torch.randint(0, num_loras, (num_tokens,)).tolist()
    lora_ranks = torch.tensor([1, 4, 8, 16], dtype=torch.int32, device="xpu")
    _run_and_compare_chunked(
        dtype=dtype,
        row_adapters=row_adapters,
        input_dim=512,
        max_rank=max_rank,
        stack_num=1,
        num_loras=num_loras,
        lora_ranks=lora_ranks,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_chunked_shrink_permutation_none_matches_sgemm(dtype):
    """permutation=None must reduce exactly to sgemm_lora_a_fwd (prefill fast path)."""
    torch.manual_seed(3)
    num_tokens = 64
    input_dim = 256
    max_rank = 8
    num_loras = 3
    stack_num = 1
    total_n = stack_num * max_rank

    seg_indptr = torch.tensor([0, 16, 48, 64], dtype=torch.int32, device="xpu")
    weight_indices = torch.tensor([0, 2, 1], dtype=torch.int32, device="xpu")
    lora_ranks = torch.tensor([max_rank, 4, 2], dtype=torch.int32, device="xpu")

    input_x = torch.randn(num_tokens, input_dim, dtype=dtype, device="xpu")
    weights = torch.randn(num_loras, total_n, input_dim, dtype=dtype, device="xpu")
    weights = _zero_weight_rank_tail(weights, lora_ranks, stack_num)

    out = chunked_sgmv_lora_shrink_forward(
        input_x=input_x,
        weights=weights,
        stack_num=stack_num,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        permutation=None,
    )
    ref = sgemm_lora_a_fwd(
        input_x=input_x,
        weights=weights,
        stack_num=stack_num,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
    )
    # Mathematically equivalent to a plain A-fwd, but the chunked shrink uses a
    # different (small-N) GEMM tile, so compare within dtype tolerance.
    rtol, atol = _tolerances(dtype)
    torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_chunked_shrink_identity_permutation_matches_sgemm(dtype):
    """An identity permutation must match a direct sgemm on the same rows."""
    torch.manual_seed(5)
    num_tokens = 32
    input_dim = 128
    max_rank = 8
    num_loras = 2
    stack_num = 1
    total_n = stack_num * max_rank

    seg_indptr = torch.tensor([0, 16, 32], dtype=torch.int32, device="xpu")
    weight_indices = torch.tensor([0, 1], dtype=torch.int32, device="xpu")
    lora_ranks = torch.tensor([max_rank, 4], dtype=torch.int32, device="xpu")
    permutation = torch.arange(num_tokens, dtype=torch.int32, device="xpu")

    input_x = torch.randn(num_tokens, input_dim, dtype=dtype, device="xpu")
    weights = torch.randn(num_loras, total_n, input_dim, dtype=dtype, device="xpu")
    weights = _zero_weight_rank_tail(weights, lora_ranks, stack_num)

    out = chunked_sgmv_lora_shrink_forward(
        input_x=input_x,
        weights=weights,
        stack_num=stack_num,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        permutation=permutation,
    )
    ref = sgemm_lora_a_fwd(
        input_x=input_x,
        weights=weights,
        stack_num=stack_num,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
    )
    # Mathematically equivalent to a plain A-fwd, but the chunked shrink uses a
    # different (small-N) GEMM tile, so compare within dtype tolerance.
    rtol, atol = _tolerances(dtype)
    torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=rtol, atol=atol)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
