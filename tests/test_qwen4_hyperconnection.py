import sys

import pytest
import sgl_kernel
import torch
import utils

device = utils.get_device()

# The kernels accumulate in fp32 and round once, so they can differ from the multi-rounding eager bf16 ops by one
# bf16 ulp.
ONE_ULP = dict(atol=4e-3, rtol=2**-7)


def make(*shape, offset=0):
    # A non-zero offset shifts the view off 16-byte alignment to exercise the scalar fallback.
    buf = torch.randn(
        int(torch.Size(shape).numel()) + offset, device=device, dtype=torch.bfloat16
    )
    return buf[offset:].view(*shape)


def silu_ref(x, hc):
    return torch.nn.functional.silu(x.float() / hc).to(x.dtype)


def gate_mix_ref(x, gate, hc):
    hidden = x.shape[-1] // hc
    return (
        torch.sigmoid(gate.float().unflatten(-1, (hc, hidden)))
        * x.float().unflatten(-1, (hc, hidden))
    ).mean(-2)


def combine_ref(residual, block, injection, hc):
    hidden = block.shape[-1]
    w = (
        torch.ones(residual.shape[0], hc, device=residual.device)
        if injection is None
        else (2 * torch.sigmoid(injection.float() / hc))
    )
    out = residual.float().unflatten(-1, (hc, hidden)) + block.float().unsqueeze(
        1
    ) * w.unsqueeze(-1)
    return out.flatten(-2)


@pytest.mark.parametrize("hc", [4, 3])
@pytest.mark.parametrize(
    "numel", [1, 17, 320, 4 * 2560, 70000, 65536 * 5, 65536 * 5 + 3]
)
@pytest.mark.parametrize("offset", [0, 1])
def test_hc_silu(hc, numel, offset):
    torch.manual_seed(numel)
    x = make(numel, offset=offset) * 4
    torch.testing.assert_close(
        sgl_kernel.hc_silu(x, hc).float(), silu_ref(x, hc).float(), **ONE_ULP
    )


@pytest.mark.parametrize(
    "tokens, hc, hidden, offset",
    [
        (1, 4, 2560, 0),  # decode
        (7, 4, 2560, 0),
        (1031, 4, 2560, 0),  # prefill
        (5, 4, 12, 0),  # scalar: hidden not a multiple of 8
        (5, 3, 64, 0),  # scalar: hc != 4
        (5, 4, 2560, 1),  # scalar: misaligned
    ],
)
def test_hc_gate_mix(tokens, hc, hidden, offset):
    torch.manual_seed(tokens)
    x = make(tokens, hc * hidden, offset=offset)
    gate = make(tokens, hc * hidden, offset=offset)
    actual = sgl_kernel.hc_gate_mix(x, gate, hc)
    assert actual.shape == (tokens, hidden) and actual.dtype == torch.bfloat16
    torch.testing.assert_close(
        actual.float(), gate_mix_ref(x, gate, hc).to(torch.bfloat16).float(), **ONE_ULP
    )


@pytest.mark.parametrize("with_injection", [True, False])
@pytest.mark.parametrize(
    "tokens, hc, hidden, offset",
    [
        (1, 4, 2560, 0),
        (4, 4, 2560, 0),
        (63, 4, 2560, 0),  # last single-chunk configuration
        (64, 4, 2560, 0),  # first two-chunk configuration
        (1031, 4, 2560, 0),
        (5, 4, 12, 0),
        (5, 3, 64, 0),
        (5, 4, 2560, 1),
    ],
)
def test_hc_combine(tokens, hc, hidden, offset, with_injection):
    torch.manual_seed(tokens)
    residual = make(tokens, hc * hidden, offset=offset)
    block = make(tokens, hidden, offset=offset)
    injection = make(tokens, hc, offset=offset) if with_injection else None
    actual = sgl_kernel.hc_combine(residual, block, injection, hc)
    expected = combine_ref(residual, block, injection, hc).to(torch.bfloat16)
    if with_injection:
        torch.testing.assert_close(actual.float(), expected.float(), **ONE_ULP)
    else:
        assert torch.equal(actual, expected)


def test_hc_combine_matches_sglang_eager():
    # The unfused sglang path: Linear -> 2 * sigmoid(. / hc) -> residual + block * weight, all in bf16.
    torch.manual_seed(0)
    tokens, hc, hidden = 33, 4, 2560
    residual, normed = make(tokens, hc * hidden), make(tokens, hc * hidden)
    block = make(tokens, hidden)
    weight = make(hc, hc * hidden) * 0.02
    injection = torch.nn.functional.linear(normed, weight)
    w = 2 * torch.sigmoid(injection / hc)
    eager = (
        residual.unflatten(-1, (hc, hidden)) + block.unsqueeze(-2) * w.unsqueeze(-1)
    ).flatten(-2)
    # Eager rounds the product before the add, so near-cancelling sums can differ by a couple of ulps of the terms.
    torch.testing.assert_close(
        sgl_kernel.hc_combine(residual, block, injection, hc).float(),
        eager.float(),
        atol=0.07,
        rtol=2**-6,
    )


def grouped_norm_ref(x, weight, hc, eps):
    g = x.float().unflatten(-1, (hc, -1))
    out = g * torch.rsqrt(g.square().mean(-1, keepdim=True) + eps)
    w = weight.float()
    w = w.view(hc, -1) if w.numel() == x.shape[-1] else w
    return (out * (1.0 + w)).flatten(-2).to(x.dtype)


@pytest.mark.parametrize("shared_weight", [False, True])
@pytest.mark.parametrize("with_injection", [True, False])
@pytest.mark.parametrize(
    "tokens, hc, hidden, offset",
    [
        (1, 4, 2560, 0),  # decode
        (63, 4, 2560, 0),  # 252 rows: last decode configuration
        (64, 4, 2560, 0),  # 256 rows: first prefill configuration
        (1031, 4, 2560, 0),
        (3, 4, 256, 0),  # smaller row: lanes are partly idle
        (5, 4, 12, 0),  # scalar: hidden not a multiple of 8
        (5, 3, 64, 0),  # hc other than 4
        (5, 4, 2560, 1),  # scalar: misaligned
        (3, 4, 5120, 0),  # row too long for the register tile: scalar
    ],
)
def test_hc_combine_norm(tokens, hc, hidden, offset, with_injection, shared_weight):
    torch.manual_seed(tokens)
    eps = 1e-6
    residual = make(tokens, hc * hidden, offset=offset)
    block = make(tokens, hidden, offset=offset)
    injection = make(tokens, hc, offset=offset) if with_injection else None
    weight = make(hidden if shared_weight else hc * hidden, offset=offset) * 0.1
    combined, normalized = sgl_kernel.hc_combine_norm(
        residual, block, injection, weight, eps, hc
    )
    expected = combine_ref(residual, block, injection, hc).to(torch.bfloat16)
    if with_injection:
        torch.testing.assert_close(combined.float(), expected.float(), **ONE_ULP)
    else:
        assert torch.equal(combined, expected)
    # The norm reads the bf16-rounded combined, whatever it rounded to.
    torch.testing.assert_close(
        normalized.float(),
        grouped_norm_ref(combined, weight, hc, eps).float(),
        atol=2e-2,
        rtol=2**-6,
    )
    # And it matches the separate combine kernel.
    assert torch.equal(combined, sgl_kernel.hc_combine(residual, block, injection, hc))


def test_empty():
    hc, hidden = 4, 2560
    assert sgl_kernel.hc_silu(make(0, 320), hc).shape == (0, 320)
    assert sgl_kernel.hc_gate_mix(
        make(0, hc * hidden), make(0, hc * hidden), hc
    ).shape == (0, hidden)
    assert sgl_kernel.hc_combine(
        make(0, hc * hidden), make(0, hidden), make(0, hc), hc
    ).shape == (0, hc * hidden)


def test_rejects_bad_inputs():
    hc, hidden = 4, 64
    residual, block, injection = make(2, hc * hidden), make(2, hidden), make(2, hc)
    with pytest.raises(RuntimeError, match="bfloat16"):
        sgl_kernel.hc_combine(residual.half(), block.half(), None, hc)
    with pytest.raises(RuntimeError, match="contiguous"):
        sgl_kernel.hc_combine(make(hc * hidden, 2).t(), block, None, hc)
    with pytest.raises(RuntimeError, match="injection"):
        sgl_kernel.hc_combine(residual, block, make(2, hc + 1), hc)
    with pytest.raises(RuntimeError, match="block_output"):
        sgl_kernel.hc_combine(residual, make(2, hidden + 8), injection, hc)
    with pytest.raises(RuntimeError, match="gate"):
        sgl_kernel.hc_gate_mix(residual, make(2, hc * hidden + 4), hc)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
