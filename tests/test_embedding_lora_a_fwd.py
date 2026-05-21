import sys
from typing import Optional

import pytest
import torch

from sgl_kernel.lora import embedding_lora_a_fwd


def _tolerances(dtype: torch.dtype):
    if dtype == torch.float32:
        return 1e-5, 1e-5
    if dtype in (torch.float16, torch.bfloat16):
        return 1e-3, 1e-3
    return 1e-5, 1e-5


def _find_segment(token_pos: int, seg_indptr_cpu: torch.Tensor) -> int:
    lo = 0
    hi = seg_indptr_cpu.numel() - 2
    while lo <= hi:
        mid = (lo + hi) >> 1
        left = int(seg_indptr_cpu[mid].item())
        right = int(seg_indptr_cpu[mid + 1].item())
        if token_pos < left:
            hi = mid - 1
        elif token_pos >= right:
            lo = mid + 1
        else:
            return mid
    return seg_indptr_cpu.numel() - 2


def _reference_embedding_lora_a_fwd(
    input_ids: torch.Tensor,
    weights: torch.Tensor,
    vocab_size: int,
    seg_indptr: torch.Tensor,
    weight_indices: torch.Tensor,
    lora_ranks: torch.Tensor,
    extra_embeddings: Optional[torch.Tensor],
) -> torch.Tensor:
    input_ids_cpu = input_ids.cpu()
    weights_cpu = weights.cpu()
    seg_indptr_cpu = seg_indptr.cpu()
    weight_indices_cpu = weight_indices.cpu()
    lora_ranks_cpu = lora_ranks.cpu()
    extra_cpu = None if extra_embeddings is None else extra_embeddings.cpu()

    num_tokens = input_ids_cpu.numel()
    max_rank = weights_cpu.size(1)

    out = torch.zeros((num_tokens, max_rank), dtype=weights_cpu.dtype)

    num_extra_tokens = 0 if extra_cpu is None else extra_cpu.size(1)

    for gid in range(num_tokens):
        seg = _find_segment(gid, seg_indptr_cpu)
        lora = int(weight_indices_cpu[seg].item())
        rank = int(lora_ranks_cpu[lora].item())
        tok = int(input_ids_cpu[gid].item())

        if tok < vocab_size:
            out[gid, :rank] = weights_cpu[lora, :rank, tok]
        else:
            e = tok - vocab_size
            if extra_cpu is not None and e < num_extra_tokens:
                out[gid, :rank] = extra_cpu[lora, e, :rank]
            # else remains zeros

        # Tail [rank:max_rank] remains zeros
    return out


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("vocab_size", [128, 32000, 128256])
def test_embedding_lora_a_fwd_large_vocab_boundaries(dtype, vocab_size):
    torch.manual_seed(11)

    max_rank = 8
    num_loras = 3
    num_extra_tokens = 4

    # Boundary-heavy token ids:
    # base low/high, extra low/high, and out-of-range extra.
    input_ids = torch.tensor(
        [
            0,
            vocab_size - 1,
            vocab_size,
            vocab_size + num_extra_tokens - 1,
            vocab_size + num_extra_tokens,
        ],
        dtype=torch.int64,
        device="xpu",
    )

    weights = torch.randn(num_loras, max_rank, vocab_size, dtype=dtype, device="xpu")
    extra_embeddings = torch.randn(
        num_loras, num_extra_tokens, max_rank, dtype=dtype, device="xpu"
    )

    seg_indptr = torch.tensor([0, 2, 5], dtype=torch.int32, device="xpu")
    weight_indices = torch.tensor([0, 2], dtype=torch.int32, device="xpu")
    lora_ranks = torch.tensor([8, 4, 3], dtype=torch.int32, device="xpu")

    out = embedding_lora_a_fwd(
        input_ids=input_ids,
        weights=weights,
        vocab_size=vocab_size,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        extra_embeddings=extra_embeddings,
        seg_lens=None,
    )

    ref = _reference_embedding_lora_a_fwd(
        input_ids,
        weights,
        vocab_size,
        seg_indptr,
        weight_indices,
        lora_ranks,
        extra_embeddings,
    ).to("xpu")

    rtol, atol = _tolerances(dtype)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("num_tokens", [1024, 8192, 16384])
def test_embedding_lora_a_fwd_large_num_tokens_many_small_segments(num_tokens):
    torch.manual_seed(12)

    dtype = torch.float32
    vocab_size = 4096
    max_rank = 8
    num_loras = 4

    input_ids = torch.randint(
        0, vocab_size, (num_tokens,), dtype=torch.int64, device="xpu"
    )
    weights = torch.randn(num_loras, max_rank, vocab_size, dtype=dtype, device="xpu")

    # One-token-per-segment stress: maximum segment count for the token count.
    seg_indptr = torch.arange(0, num_tokens + 1, dtype=torch.int32, device="xpu")
    weight_indices = torch.randint(
        0, num_loras, (num_tokens,), dtype=torch.int32, device="xpu"
    )
    lora_ranks = torch.tensor([1, 3, 5, 8], dtype=torch.int32, device="xpu")

    out = embedding_lora_a_fwd(
        input_ids=input_ids,
        weights=weights,
        vocab_size=vocab_size,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        extra_embeddings=None,
        seg_lens=None,
    )

    ref = _reference_embedding_lora_a_fwd(
        input_ids, weights, vocab_size, seg_indptr, weight_indices, lora_ranks, None
    ).to("xpu")

    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_embedding_lora_a_fwd_empty_input():
    torch.manual_seed(13)

    vocab_size = 32
    max_rank = 4

    input_ids = torch.empty((0,), dtype=torch.int64, device="xpu")
    weights = torch.randn(1, max_rank, vocab_size, dtype=torch.float32, device="xpu")
    seg_indptr = torch.tensor([0], dtype=torch.int32, device="xpu")
    weight_indices = torch.empty((0,), dtype=torch.int32, device="xpu")
    lora_ranks = torch.tensor([max_rank], dtype=torch.int32, device="xpu")

    out = embedding_lora_a_fwd(
        input_ids=input_ids,
        weights=weights,
        vocab_size=vocab_size,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        extra_embeddings=None,
        seg_lens=None,
    )

    assert out.shape == (0, max_rank)
    assert out.numel() == 0


def test_embedding_lora_a_fwd_zero_rank_all_zero_output():
    torch.manual_seed(14)

    vocab_size = 64
    max_rank = 8

    input_ids = torch.tensor([1, 5, 10, 63], dtype=torch.int64, device="xpu")
    weights = torch.randn(2, max_rank, vocab_size, dtype=torch.float32, device="xpu")
    seg_indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device="xpu")
    weight_indices = torch.tensor([0, 1], dtype=torch.int32, device="xpu")
    lora_ranks = torch.tensor([0, 0], dtype=torch.int32, device="xpu")

    out = embedding_lora_a_fwd(
        input_ids=input_ids,
        weights=weights,
        vocab_size=vocab_size,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        extra_embeddings=None,
        seg_lens=None,
    )

    assert torch.count_nonzero(out).item() == 0


def test_embedding_lora_a_fwd_segment_boundaries_precise_routing():
    torch.manual_seed(15)

    dtype = torch.float32
    vocab_size = 128
    max_rank = 4

    # Segment sizes: 3,1,5
    input_ids = torch.tensor(
        [7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=torch.int64, device="xpu"
    )
    seg_indptr = torch.tensor([0, 3, 4, 9], dtype=torch.int32, device="xpu")
    weight_indices = torch.tensor([2, 1, 0], dtype=torch.int32, device="xpu")
    lora_ranks = torch.tensor([4, 2, 3], dtype=torch.int32, device="xpu")
    weights = torch.randn(3, max_rank, vocab_size, dtype=dtype, device="xpu")

    out = embedding_lora_a_fwd(
        input_ids=input_ids,
        weights=weights,
        vocab_size=vocab_size,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        extra_embeddings=None,
        seg_lens=None,
    )

    ref = _reference_embedding_lora_a_fwd(
        input_ids, weights, vocab_size, seg_indptr, weight_indices, lora_ranks, None
    ).to("xpu")

    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_embedding_lora_a_fwd_extra_tokens_without_extra_embeddings_are_zero():
    torch.manual_seed(16)

    vocab_size = 32
    max_rank = 4

    input_ids = torch.tensor(
        [vocab_size, vocab_size + 3, vocab_size + 99], dtype=torch.int64, device="xpu"
    )
    weights = torch.randn(1, max_rank, vocab_size, dtype=torch.float32, device="xpu")
    seg_indptr = torch.tensor([0, 3], dtype=torch.int32, device="xpu")
    weight_indices = torch.tensor([0], dtype=torch.int32, device="xpu")
    lora_ranks = torch.tensor([4], dtype=torch.int32, device="xpu")

    out = embedding_lora_a_fwd(
        input_ids=input_ids,
        weights=weights,
        vocab_size=vocab_size,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        extra_embeddings=None,
        seg_lens=None,
    )

    assert torch.count_nonzero(out).item() == 0


def test_embedding_lora_a_fwd_rank_padding_zero_tail():
    torch.manual_seed(1)

    input_ids = torch.tensor([2, 5, 1, 3], dtype=torch.int64, device="xpu")
    vocab_size = 8
    max_rank = 4

    # Two loras with different effective ranks.
    weights = torch.randn(2, max_rank, vocab_size, dtype=torch.float32, device="xpu")
    seg_indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device="xpu")
    weight_indices = torch.tensor([0, 1], dtype=torch.int32, device="xpu")
    lora_ranks = torch.tensor([1, 3], dtype=torch.int32, device="xpu")

    out = embedding_lora_a_fwd(
        input_ids=input_ids,
        weights=weights,
        vocab_size=vocab_size,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        extra_embeddings=None,
        seg_lens=None,
    )

    # token 0,1 -> lora 0 rank=1 ; token 2,3 -> lora 1 rank=3
    expected_ranks = [1, 1, 3, 3]
    for i, rank in enumerate(expected_ranks):
        tail = out[i, rank:]
        assert torch.count_nonzero(tail).item() == 0, f"Tail not zero for token {i}"


def test_embedding_lora_a_fwd_extra_embeddings_in_range():
    torch.manual_seed(2)

    vocab_size = 6
    max_rank = 4
    num_extra_tokens = 2

    # Includes base tokens and extra tokens: vocab+0 and vocab+1.
    input_ids = torch.tensor([1, 6, 3, 7], dtype=torch.int64, device="xpu")
    weights = torch.randn(2, max_rank, vocab_size, dtype=torch.float32, device="xpu")
    extra_embeddings = torch.randn(
        2, num_extra_tokens, max_rank, dtype=torch.float32, device="xpu"
    )

    seg_indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device="xpu")
    weight_indices = torch.tensor([0, 1], dtype=torch.int32, device="xpu")
    lora_ranks = torch.tensor([4, 2], dtype=torch.int32, device="xpu")

    out = embedding_lora_a_fwd(
        input_ids=input_ids,
        weights=weights,
        vocab_size=vocab_size,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        extra_embeddings=extra_embeddings,
        seg_lens=None,
    )

    ref = _reference_embedding_lora_a_fwd(
        input_ids,
        weights,
        vocab_size,
        seg_indptr,
        weight_indices,
        lora_ranks,
        extra_embeddings,
    ).to("xpu")

    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_embedding_lora_a_fwd_extra_token_out_of_range_zeroed():
    torch.manual_seed(3)

    vocab_size = 6
    max_rank = 4

    # vocab+5 is out of range when num_extra_tokens=2.
    input_ids = torch.tensor([11], dtype=torch.int64, device="xpu")
    weights = torch.randn(1, max_rank, vocab_size, dtype=torch.float32, device="xpu")
    extra_embeddings = torch.randn(1, 2, max_rank, dtype=torch.float32, device="xpu")

    seg_indptr = torch.tensor([0, 1], dtype=torch.int32, device="xpu")
    weight_indices = torch.tensor([0], dtype=torch.int32, device="xpu")
    lora_ranks = torch.tensor([4], dtype=torch.int32, device="xpu")

    out = embedding_lora_a_fwd(
        input_ids=input_ids,
        weights=weights,
        vocab_size=vocab_size,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        extra_embeddings=extra_embeddings,
        seg_lens=None,
    )

    assert (
        torch.count_nonzero(out).item() == 0
    ), "Out-of-range extra token must produce zeros"


def test_embedding_lora_a_fwd_negative_token_id_zero_row():
    torch.manual_seed(17)

    vocab_size = 16
    max_rank = 4

    # Includes one negative token to exercise the dedicated kernel branch.
    input_ids = torch.tensor([3, -1, vocab_size + 1], dtype=torch.int64, device="xpu")
    weights = torch.randn(1, max_rank, vocab_size, dtype=torch.float32, device="xpu")
    extra_embeddings = torch.randn(1, 2, max_rank, dtype=torch.float32, device="xpu")

    seg_indptr = torch.tensor([0, 3], dtype=torch.int32, device="xpu")
    weight_indices = torch.tensor([0], dtype=torch.int32, device="xpu")
    lora_ranks = torch.tensor([4], dtype=torch.int32, device="xpu")

    out = embedding_lora_a_fwd(
        input_ids=input_ids,
        weights=weights,
        vocab_size=vocab_size,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        extra_embeddings=extra_embeddings,
        seg_lens=None,
    )

    # Negative token row must be fully zero.
    assert torch.count_nonzero(out[1]).item() == 0

    # Neighboring rows should still follow normal base/extra paths.
    torch.testing.assert_close(out[0], weights[0, :, 3], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out[2], extra_embeddings[0, 1, :], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "bad_case, expected_msg",
    [
        ("input_ids_dim", "input_ids must be a 1D tensor"),
        ("weights_dim", "weights must be a 3D tensor"),
        ("seg_indptr_dim", "seg_indptr must be a 1D tensor"),
        ("weight_indices_dim", "weight_indices must be a 1D tensor"),
        ("lora_ranks_dim", "lora_ranks must be a 1D tensor"),
        (
            "vocab_mismatch",
            "weights' vocab_size dimension must match the provided vocab_size",
        ),
    ],
)
def test_embedding_lora_a_fwd_input_validation(bad_case, expected_msg):
    input_ids = torch.tensor([1, 2], dtype=torch.int64, device="xpu")
    weights = torch.randn(1, 4, 8, dtype=torch.float32, device="xpu")
    seg_indptr = torch.tensor([0, 2], dtype=torch.int32, device="xpu")
    weight_indices = torch.tensor([0], dtype=torch.int32, device="xpu")
    lora_ranks = torch.tensor([4], dtype=torch.int32, device="xpu")
    vocab_size = 8

    if bad_case == "input_ids_dim":
        input_ids = input_ids.view(1, 2)
    elif bad_case == "weights_dim":
        weights = weights.view(1, -1)
    elif bad_case == "seg_indptr_dim":
        seg_indptr = seg_indptr.view(1, 2)
    elif bad_case == "weight_indices_dim":
        weight_indices = weight_indices.view(1, 1)
    elif bad_case == "lora_ranks_dim":
        lora_ranks = lora_ranks.view(1, 1)
    elif bad_case == "vocab_mismatch":
        vocab_size = 7

    with pytest.raises(RuntimeError, match=expected_msg):
        embedding_lora_a_fwd(
            input_ids=input_ids,
            weights=weights,
            vocab_size=vocab_size,
            seg_indptr=seg_indptr,
            weight_indices=weight_indices,
            lora_ranks=lora_ranks,
            extra_embeddings=None,
            seg_lens=None,
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
