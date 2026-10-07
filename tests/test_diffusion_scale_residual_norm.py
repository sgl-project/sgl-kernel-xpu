"""Tests for the fused scale-residual + LayerNorm + scale-shift diffusion kernel."""

import sys

import pytest
import torch
import utils
from sgl_kernel import fused_scale_residual_norm_scale_shift

device = utils.get_device()


def reference_scale_residual_norm_scale_shift(
    residual, x, gate, shift, scale, weight, bias, eps
):
    # The gated add is done in fp32 and rounded once on output, as the kernel does.
    if isinstance(gate, torch.Tensor):
        if gate.dim() == 4:
            num_frames = gate.shape[1]
            frame_seqlen = x.shape[1] // num_frames
            gated = (
                x.float().unflatten(dim=1, sizes=(num_frames, frame_seqlen))
                * gate.float()
            ).flatten(1, 2)
        else:
            gated = x.float() * gate.float()
    else:
        assert gate == 1
        gated = x.float()

    residual_f32 = residual.float() + gated
    mean = residual_f32.mean(dim=-1, keepdim=True)
    centered = residual_f32 - mean
    var = (centered * centered).mean(dim=-1, keepdim=True)
    normed = centered / torch.sqrt(var + eps)
    if weight is not None:
        normed = normed * weight.float().reshape(-1)
    if bias is not None:
        normed = normed + bias.float().reshape(-1)
    out = normed * (1.0 + scale.float().reshape(-1)) + shift.float().reshape(-1)
    return out.to(x.dtype), residual_f32.to(x.dtype)


def make_operands(
    *,
    seq_len,
    hidden,
    dtype,
    param_dtype,
    gate_mode="per_token",
    num_frames=1,
    scale_numel=None,
    shift_numel=None,
    affine=True,
):
    torch.manual_seed(0)
    x = torch.randn(1, seq_len, hidden, dtype=dtype, device=device)
    residual = torch.randn(1, seq_len, hidden, dtype=dtype, device=device)

    if gate_mode == "none":
        gate = 1
    elif gate_mode == "per_token":
        gate = torch.randn(1, 1, hidden, dtype=param_dtype, device=device)
    else:
        gate = torch.randn(1, num_frames, 1, hidden, dtype=param_dtype, device=device)

    scale = torch.randn(
        hidden if scale_numel is None else scale_numel, dtype=param_dtype, device=device
    )
    shift = torch.randn(
        hidden if shift_numel is None else shift_numel, dtype=param_dtype, device=device
    )

    weight = (
        torch.randn(hidden, dtype=param_dtype, device=device)
        if affine in (True, "weight_only")
        else None
    )
    bias = (
        torch.randn(hidden, dtype=param_dtype, device=device)
        if affine in (True, "bias_only")
        else None
    )

    return residual, x, gate, shift, scale, weight, bias


def check_against_reference(residual, x, gate, shift, scale, weight, bias, eps=1e-6):
    out, residual_out = fused_scale_residual_norm_scale_shift(
        residual=residual,
        x=x,
        gate=gate,
        shift=shift,
        scale=scale,
        weight=weight,
        bias=bias,
        eps=eps,
    )
    out_ref, residual_out_ref = reference_scale_residual_norm_scale_shift(
        residual, x, gate, shift, scale, weight, bias, eps
    )

    assert out.shape == x.shape and out.dtype == x.dtype
    assert residual_out.shape == x.shape and residual_out.dtype == x.dtype
    rtol, atol = (1e-5, 1e-5) if x.dtype == torch.float32 else (1e-2, 1e-2)
    torch.testing.assert_close(residual_out, residual_out_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(out, out_ref, rtol=rtol, atol=atol)


# One compiled kernel per (dtype, vector width): vec 2 (130), vec 1 (255), vec 4 (6660)
# and vec 8 at the 8192 bound, where every thread runs the full loop.
# Rows are independent work-groups, so one multi-row seq_len covers row indexing.
@pytest.mark.parametrize("hidden", [130, 255, 6660, 8192])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_shapes(hidden, dtype):
    check_against_reference(
        *make_operands(seq_len=17, hidden=hidden, dtype=dtype, param_dtype=dtype)
    )


# Vary one operand branch at a time from the per-token-gated, vector-modulated, affine
# bf16 case that test_shapes already covers.
@pytest.mark.parametrize(
    "override",
    [
        {"gate_mode": "none"},
        {"gate_mode": "per_frame", "num_frames": 5},
        {"scale_numel": 1},
        {"shift_numel": 1},
        {"affine": "weight_only"},
        {"affine": "bias_only"},
        {"param_dtype": torch.float32},
    ],
    ids=lambda o: ",".join(f"{k}={v}" for k, v in o.items()),
)
def test_operand_variants(override):
    base = dict(
        seq_len=120, hidden=1536, dtype=torch.bfloat16, param_dtype=torch.bfloat16
    )
    check_against_reference(*make_operands(**{**base, **override}))


def test_misaligned_params():
    """A misaligned but contiguous scale slice must narrow the vector width."""
    # A 2-byte offset: B60 tolerates 4- and 8-byte misaligned vector loads, so only
    # this one gives wrong results if the narrowing is skipped.
    hidden, dtype, storage_offset = 1536, torch.bfloat16, 1
    residual, x, gate, shift, scale, weight, bias = make_operands(
        seq_len=32,
        hidden=hidden,
        dtype=dtype,
        param_dtype=dtype,
        scale_numel=hidden + storage_offset,
    )
    scale = scale[storage_offset:]
    assert scale.is_contiguous() and scale.numel() == hidden
    check_against_reference(residual, x, gate, shift, scale, weight, bias)


_HIDDEN, _SEQ_LEN, _DTYPE = 1536, 32, torch.bfloat16
_OVER_MAX = 2 * 8192


def _randn(*shape, dtype=_DTYPE):
    return torch.randn(*shape, dtype=dtype, device=device)


_UNSUPPORTED = {
    "hidden above MAX_FUSED_HIDDEN": lambda: dict(
        residual=_randn(1, 4, _OVER_MAX),
        x=_randn(1, 4, _OVER_MAX),
        gate=_randn(1, 1, _OVER_MAX),
        shift=_randn(_OVER_MAX),
        scale=_randn(_OVER_MAX),
        weight=_randn(_OVER_MAX),
        bias=_randn(_OVER_MAX),
    ),
    "residual shape mismatch": lambda: dict(residual=_randn(1, _SEQ_LEN // 2, _HIDDEN)),
    "batch size above 1": lambda: dict(
        residual=_randn(2, _SEQ_LEN, _HIDDEN), x=_randn(2, _SEQ_LEN, _HIDDEN)
    ),
    "non-contiguous x": lambda: dict(x=_randn(1, _SEQ_LEN, 2 * _HIDDEN)[..., :_HIDDEN]),
    "gate hidden mismatch": lambda: dict(gate=_randn(1, 1, _HIDDEN // 2)),
    "gate int other than 1": lambda: dict(gate=2),
    "bool gate": lambda: dict(gate=True),
    "num_frames not dividing seq_len": lambda: dict(gate=_randn(1, 7, 1, _HIDDEN)),
    "zero num_frames": lambda: dict(gate=_randn(1, 0, 1, _HIDDEN)),
    "modulation numel mismatch": lambda: dict(scale=_randn(3)),
    "affine numel mismatch": lambda: dict(weight=_randn(_HIDDEN // 2)),
    "mixed parameter dtypes": lambda: dict(scale=_randn(_HIDDEN, dtype=torch.float32)),
    "unsupported dtype": lambda: dict(
        residual=_randn(1, _SEQ_LEN, _HIDDEN, dtype=torch.float64),
        x=_randn(1, _SEQ_LEN, _HIDDEN, dtype=torch.float64),
    ),
}


@pytest.mark.parametrize("reason", list(_UNSUPPORTED))
def test_rejects_unsupported(reason):
    base = dict(
        residual=_randn(1, _SEQ_LEN, _HIDDEN),
        x=_randn(1, _SEQ_LEN, _HIDDEN),
        gate=_randn(1, 1, _HIDDEN),
        shift=_randn(_HIDDEN),
        scale=_randn(_HIDDEN),
        weight=_randn(_HIDDEN),
        bias=_randn(_HIDDEN),
    )
    with pytest.raises(ValueError):
        fused_scale_residual_norm_scale_shift(
            **{**base, **_UNSUPPORTED[reason]()}, eps=1e-6
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
