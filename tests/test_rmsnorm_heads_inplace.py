"""rmsnorm_heads_inplace tests.

`rmsnorm_heads_inplace` norms the leading heads of a packed row in place by
running `fused_qk_norm_rope` with rotary off and the rest of the row declared
as untouched V heads. That relies on the op striding rows by all of its head
counts while only touching Q and K, so these cases pin that contract:

- the leading heads match `F.rms_norm`, at Inkling's packed QKVR geometries
  (`[q | k | v | r]`, TP=8/TP=4) in bf16, fp16 and fp32
- the rest of the row stays bit-for-bit untouched
- the weight survives a call (the op's schema declares it mutable)
- rotary is really off, not merely inert at position 0
- the cached position-id buffer reslices when the token count shrinks
- geometries whose remainder is not whole heads are refused
"""

import sys

import pytest
import torch
import torch.nn.functional as F
import utils
from sgl_kernel import (
    elementwise,
    fused_qk_norm_rope,
    rmsnorm_heads_inplace,
    rmsnorm_heads_inplace_supported,
)

device = utils.get_device()

HEAD_DIM = 128
D_REL = 16
EPS = 1e-6

# (num_tp_heads, num_tp_kv_heads) for Inkling's two layer groups at TP=8/TP=4.
ADMITTED = {
    "swa_tp8": (8, 2),
    "full_tp8": (8, 1),
    "swa_tp4": (16, 4),
    "full_tp4": (16, 2),
}
# TP=16/32 shrink d_rel * num_tp_heads below head_dim, so the tail is not whole heads.
REJECTED = {"swa_tp16": (4, 1), "swa_tp32": (2, 1)}

TOLERANCE = {
    torch.bfloat16: dict(rtol=8e-3, atol=1e-3),
    torch.float16: dict(rtol=1e-3, atol=1e-4),
    torch.float32: dict(rtol=1e-5, atol=1e-5),
}


def row_width(heads, kv_heads):
    return HEAD_DIM * heads + 2 * HEAD_DIM * kv_heads + D_REL * heads


def make_row(num_tokens, heads, kv_heads, dtype, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    width = row_width(heads, kv_heads)
    return torch.randn(num_tokens, width, generator=g, device=device, dtype=dtype)


def make_weight(dtype):
    g = torch.Generator(device=device).manual_seed(11)
    w = 1.0 + 0.05 * torch.randn(HEAD_DIM, generator=g, device=device)
    return w.to(dtype)


def reference(row, heads, weight):
    q = row[:, : heads * HEAD_DIM].reshape(-1, HEAD_DIM)
    return F.rms_norm(q.float(), (HEAD_DIM,), weight.float(), EPS).to(row.dtype)


@pytest.mark.parametrize("name", list(ADMITTED))
def test_supported_admits_inkling_geometries(name):
    heads, kv_heads = ADMITTED[name]
    width = row_width(heads, kv_heads)
    assert rmsnorm_heads_inplace_supported(width, heads, HEAD_DIM, torch.bfloat16)
    # The V-head padding has to cover the tail exactly.
    pad = width // HEAD_DIM - heads
    assert pad * HEAD_DIM == width - heads * HEAD_DIM


@pytest.mark.parametrize("name", list(REJECTED))
def test_tail_not_whole_heads_is_refused(name):
    heads, kv_heads = REJECTED[name]
    width = row_width(heads, kv_heads)
    assert not rmsnorm_heads_inplace_supported(width, heads, HEAD_DIM, torch.bfloat16)
    row = make_row(4, heads, kv_heads, torch.bfloat16)
    with pytest.raises(AssertionError, match="unsupported geometry"):
        rmsnorm_heads_inplace(row, heads, HEAD_DIM, make_weight(row.dtype), EPS)


def test_supported_refuses_unsupported_head_dim_and_dtype():
    assert not rmsnorm_heads_inplace_supported(96 * 4, 2, 96, torch.bfloat16)
    assert not rmsnorm_heads_inplace_supported(1024, 4, 128, torch.float8_e4m3fn)
    assert not rmsnorm_heads_inplace_supported(1024, 9, 128, torch.bfloat16)


@pytest.mark.parametrize("name", list(ADMITTED))
@pytest.mark.parametrize("dtype", list(TOLERANCE))
@pytest.mark.parametrize("num_tokens", [1, 2, 8, 512, 4096])
def test_matches_rms_norm_and_leaves_tail_untouched(name, dtype, num_tokens):
    heads, kv_heads = ADMITTED[name]
    row = make_row(num_tokens, heads, kv_heads, dtype)
    before = row.clone()
    weight = make_weight(dtype)
    rmsnorm_heads_inplace(row, heads, HEAD_DIM, weight, EPS)
    q_width = heads * HEAD_DIM
    torch.testing.assert_close(
        row[:, :q_width].reshape(-1, HEAD_DIM),
        reference(before, heads, weight),
        **TOLERANCE[dtype],
    )
    assert torch.equal(row[:, q_width:], before[:, q_width:])


def test_weight_is_not_mutated():
    # Same dtype as the row, so the op receives the caller's tensor itself.
    heads, kv_heads = ADMITTED["swa_tp8"]
    row = make_row(512, heads, kv_heads, torch.bfloat16)
    weight = make_weight(torch.bfloat16)
    expected = weight.clone()
    rmsnorm_heads_inplace(row, heads, HEAD_DIM, weight, EPS)
    assert torch.equal(weight, expected)


def test_rotary_off_even_at_nonzero_positions():
    heads, kv_heads = ADMITTED["swa_tp8"]
    num_tokens = 64
    row = make_row(num_tokens, heads, kv_heads, torch.bfloat16)
    weight = make_weight(torch.bfloat16)
    expected = reference(row, heads, weight)
    # Rotation at position 0 is the identity, so the zero-filled buffer alone
    # could not tell rotary off from rotary on. Poison it with real positions.
    pos = torch.arange(1, num_tokens + 1, dtype=torch.int32, device=device) * 97
    elementwise._RMSNORM_HEADS_POS[row.device] = pos
    try:
        out = row.clone()
        rmsnorm_heads_inplace(out, heads, HEAD_DIM, weight, EPS)
        torch.testing.assert_close(
            out[:, : heads * HEAD_DIM].reshape(-1, HEAD_DIM),
            expected,
            **TOLERANCE[torch.bfloat16],
        )
        # Control: the same positions with rotary on must move q, or the case
        # above has no power.
        control = row.clone()
        fused_qk_norm_rope(
            control,
            num_heads_q=heads,
            num_heads_k=0,
            num_heads_v=row.size(1) // HEAD_DIM - heads,
            head_dim=HEAD_DIM,
            eps=EPS,
            q_weight=weight,
            k_weight=weight,
            base=1.0e4,
            is_neox=False,
            position_ids=pos,
            rotary_dim=HEAD_DIM,
        )
        diff = control[:, : heads * HEAD_DIM].reshape(-1, HEAD_DIM) - expected
        assert diff.abs().max() > 0.1
    finally:
        elementwise._RMSNORM_HEADS_POS.pop(row.device, None)


def test_position_cache_reslices_when_batch_shrinks():
    heads, kv_heads = ADMITTED["full_tp8"]
    weight = make_weight(torch.bfloat16)
    for num_tokens in (512, 8, 1, 37):
        row = make_row(num_tokens, heads, kv_heads, torch.bfloat16, seed=num_tokens)
        before = row.clone()
        rmsnorm_heads_inplace(row, heads, HEAD_DIM, weight, EPS)
        torch.testing.assert_close(
            row[:, : heads * HEAD_DIM].reshape(-1, HEAD_DIM),
            reference(before, heads, weight),
            **TOLERANCE[torch.bfloat16],
        )


def test_non_contiguous_row_is_refused():
    heads, kv_heads = ADMITTED["swa_tp8"]
    padded = make_row(4, heads, kv_heads + 1, torch.bfloat16)
    view = padded[:, : row_width(heads, kv_heads)]
    with pytest.raises(AssertionError):
        rmsnorm_heads_inplace(view, heads, HEAD_DIM, make_weight(view.dtype), EPS)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
