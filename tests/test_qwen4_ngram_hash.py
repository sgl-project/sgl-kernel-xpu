import sys

import pytest
import sgl_kernel
import torch
import utils

device = utils.get_device()

NGRAM_SIZE = 3
HEADS_PER_NGRAM = 8
NGRAM_HEADS = 16
EOS = 151645
UNIGRAM_VOCAB = 160000
MASK64 = (1 << 64) - 1


def splitmix64(x):
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return (x ^ (x >> 31)) & MASK64


def make_multipliers(seed=1234):
    # Same construction as Qwen4ExpNGramEmbedding._build_layer_multipliers: odd values that
    # keep token * multiplier below 2**63 for tokens within the unigram vocab.
    half_bound = max(1, ((1 << 63) - 1) // UNIGRAM_VOCAB // 2)
    values = []
    for idx in range(NGRAM_SIZE):
        x0 = (seed + 0x9E3779B97F4A7C15 * (idx + 1)) & MASK64
        values.append(2 * (splitmix64(x0) % half_bound) + 1)
    return torch.tensor(values, dtype=torch.long)


def make_tables(seed=0):
    g = torch.Generator().manual_seed(seed)
    sizes = torch.randint(100_000, 5_000_000, (NGRAM_HEADS,), generator=g)
    offsets = torch.cumsum(sizes, 0) - sizes
    return sizes.long(), offsets.long()


def to_device(*tensors):
    return [t.to(device) for t in tensors]


def shift_right_ignore_eos(tensor, n, eos):
    # Copied from Qwen4ExpNGramEmbedding._shift_right_ignore_eos.
    if n == 0:
        return tensor
    batch_size, seq_len = tensor.shape
    idx = torch.arange(seq_len, device=tensor.device, dtype=torch.long)
    eos_pos = torch.where(tensor == eos, idx, -1)
    prev_eos_inclusive = torch.cummax(eos_pos, dim=1).values
    prev_eos = torch.cat(
        [eos_pos.new_full((batch_size, 1), -1), prev_eos_inclusive[:, :-1]], dim=1
    )
    pos_in_segment = idx.unsqueeze(0) - (prev_eos + 1)
    src_idx = idx - n
    gather_idx = torch.clamp(src_idx, min=0).unsqueeze(0).expand(batch_size, -1)
    shifted = tensor.gather(dim=1, index=gather_idx)
    valid_mask = (pos_in_segment >= n) & (src_idx.unsqueeze(0) >= 0)
    return torch.where(valid_mask, shifted, tensor.new_full((), eos))


def ngram_hash_eager(contexts, multipliers, vocab_sizes, offsets, eos):
    # The eager path of Qwen4ExpNGramEmbedding for a window of NGRAM_SIZE tokens.
    shifted = [contexts] + [
        shift_right_ignore_eos(contexts, s, eos) for s in range(1, NGRAM_SIZE)
    ]
    blocks = []
    for ngram in range(2, NGRAM_SIZE + 1):
        start = (ngram - 2) * HEADS_PER_NGRAM
        end = start + HEADS_PER_NGRAM
        mix = shifted[0] * multipliers[0]
        for pos in range(1, ngram):
            mix = torch.bitwise_xor(mix, shifted[pos] * multipliers[pos])
        ids = torch.remainder(
            mix[:, -1:].unsqueeze(-1), vocab_sizes[start:end].view(1, 1, -1)
        )
        blocks.append((ids + offsets[start:end].view(1, 1, -1))[:, 0])
    return torch.cat(blocks, dim=-1)


def random_contexts(n, eos=EOS, eos_prob=0.2, seed=0):
    g = torch.Generator().manual_seed(seed)
    ctx = torch.randint(0, UNIGRAM_VOCAB, (n, NGRAM_SIZE), generator=g)
    ctx = torch.where(torch.rand(n, NGRAM_SIZE, generator=g) < eos_prob, eos, ctx)
    return ctx.long()


@pytest.mark.parametrize("specify_out", [True, False])
# 1281 and 40961 make the kernel batch 2 and 16 blocks per work-group with a partial tail.
@pytest.mark.parametrize("tokens", [1, 3, 17, 255, 256, 257, 1000, 1281, 40961, 100003])
def test_qwen4_ngram_hash_matches_eager(tokens, specify_out):
    ctx = random_contexts(tokens)
    mult = make_multipliers()
    sizes, offsets = make_tables()
    ctx, mult, sizes, offsets = to_device(ctx, mult, sizes, offsets)
    expected = ngram_hash_eager(ctx, mult, sizes, offsets, EOS)
    if specify_out:
        out = torch.empty(tokens, NGRAM_HEADS, dtype=torch.long, device=device)
        sgl_kernel.qwen4_ngram_hash(ctx, mult, sizes, offsets, EOS, out=out)
    else:
        out = sgl_kernel.qwen4_ngram_hash(ctx, mult, sizes, offsets, EOS)
    assert out.dtype == torch.long and out.shape == (tokens, NGRAM_HEADS)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_qwen4_ngram_hash_all_eos_patterns():
    # Every EOS/non-EOS combination of the three positions.
    rows = []
    for mask in range(8):
        rows.append(
            [EOS if (mask >> i) & 1 else 1000 + 7 * i for i in range(NGRAM_SIZE)]
        )
    ctx = torch.tensor(rows, dtype=torch.long)
    mult = make_multipliers()
    sizes, offsets = make_tables()
    ctx, mult, sizes, offsets = to_device(ctx, mult, sizes, offsets)
    out = sgl_kernel.qwen4_ngram_hash(ctx, mult, sizes, offsets, EOS)
    torch.testing.assert_close(
        out, ngram_hash_eager(ctx, mult, sizes, offsets, EOS), rtol=0, atol=0
    )


def test_qwen4_ngram_hash_ids_stay_in_head_range():
    ctx = random_contexts(5000, seed=1)
    mult = make_multipliers()
    sizes, offsets = make_tables()
    out = sgl_kernel.qwen4_ngram_hash(*to_device(ctx, mult, sizes, offsets), EOS).cpu()
    assert (out >= offsets).all() and (out < offsets + sizes).all()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_qwen4_ngram_hash_wraps_and_floors_like_torch(seed):
    # Arbitrary 63-bit multipliers and negative tokens overflow int64 and make the mix negative;
    # the result must still equal torch's wrapping multiply and floored remainder.
    g = torch.Generator().manual_seed(seed)
    ctx = torch.randint(-(2**40), 2**40, (4096, NGRAM_SIZE), generator=g)
    mult = torch.randint(1, 2**62, (NGRAM_SIZE,), generator=g) | 1
    sizes, offsets = make_tables(seed)
    ctx, mult, sizes, offsets = to_device(ctx, mult, sizes, offsets)
    out = sgl_kernel.qwen4_ngram_hash(ctx, mult, sizes, offsets, EOS)
    torch.testing.assert_close(
        out, ngram_hash_eager(ctx, mult, sizes, offsets, EOS), rtol=0, atol=0
    )
    assert (out >= offsets).all()


@pytest.mark.parametrize("seed", [0, 1])
def test_qwen4_ngram_hash_extreme_vocab_sizes(seed):
    # The kernel replaces the 64-bit division by a reciprocal; check divisor edge cases with negative mixes.
    g = torch.Generator().manual_seed(seed)
    ctx = torch.randint(-(2**62), 2**62, (2048, NGRAM_SIZE), generator=g)
    mult = torch.randint(1, 2**62, (NGRAM_SIZE,), generator=g) | 1
    sizes = torch.tensor(
        [1, 2, 3, 7, 2**31 - 1, 2**31, 2**32 - 1, 2**32, 2**32 + 1]
        + [2**40 + 3, 2**52, 2**62 - 1, 2**62, 2**62 + 1, 2**63 - 2, 2**63 - 1],
        dtype=torch.long,
    )
    offsets = torch.zeros_like(sizes)
    ctx, mult, sizes, offsets = to_device(ctx, mult, sizes, offsets)
    out = sgl_kernel.qwen4_ngram_hash(ctx, mult, sizes, offsets, EOS)
    torch.testing.assert_close(
        out, ngram_hash_eager(ctx, mult, sizes, offsets, EOS), rtol=0, atol=0
    )


def test_qwen4_ngram_hash_negative_eos():
    ctx = torch.tensor([[-1, -1, 5], [-1, 3, -1], [4, -1, 9]], dtype=torch.long)
    mult = make_multipliers()
    sizes, offsets = make_tables()
    ctx, mult, sizes, offsets = to_device(ctx, mult, sizes, offsets)
    out = sgl_kernel.qwen4_ngram_hash(ctx, mult, sizes, offsets, -1)
    torch.testing.assert_close(
        out, ngram_hash_eager(ctx, mult, sizes, offsets, -1), rtol=0, atol=0
    )


def test_qwen4_ngram_hash_empty():
    mult = make_multipliers()
    sizes, offsets = make_tables()
    ctx = torch.empty(0, NGRAM_SIZE, dtype=torch.long)
    out = sgl_kernel.qwen4_ngram_hash(*to_device(ctx, mult, sizes, offsets), EOS)
    assert out.shape == (0, NGRAM_HEADS)


@pytest.mark.parametrize(
    "case",
    [
        "int32_contexts",
        "int32_multipliers",
        "non_contiguous_contexts",
        "wrong_context_width",
        "wrong_multiplier_count",
        "wrong_vocab_count",
        "wrong_offset_count",
        "output_wrong_shape",
        "output_int32",
        "cpu_tensor",
    ],
)
def test_qwen4_ngram_hash_rejects_invalid(case):
    ctx = random_contexts(4)
    mult = make_multipliers()
    sizes, offsets = make_tables()
    out = None
    if case == "int32_contexts":
        ctx = ctx.int()
    elif case == "int32_multipliers":
        mult = mult.int()
    elif case == "non_contiguous_contexts":
        ctx = random_contexts(8)
    elif case == "wrong_context_width":
        ctx = torch.zeros(4, 4, dtype=torch.long)
    elif case == "wrong_multiplier_count":
        mult = mult[:2].contiguous()
    elif case == "wrong_vocab_count":
        sizes = sizes[:8].contiguous()
    elif case == "wrong_offset_count":
        offsets = torch.cat([offsets, offsets[:1]])
    elif case == "output_wrong_shape":
        out = torch.empty(4, 8, dtype=torch.long, device=device)
    elif case == "output_int32":
        out = torch.empty(4, NGRAM_HEADS, dtype=torch.int32, device=device)
    ctx, mult, sizes, offsets = to_device(ctx, mult, sizes, offsets)
    if case == "cpu_tensor":
        mult = mult.cpu()
    if case == "non_contiguous_contexts":
        ctx = ctx[::2]
    with pytest.raises(RuntimeError):
        sgl_kernel.qwen4_ngram_hash(ctx, mult, sizes, offsets, EOS, out=out)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
