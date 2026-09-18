"""Tests for the fused scale-residual + LayerNorm + scale-shift diffusion kernel."""

import sys

import pytest
import sgl_kernel
import torch
import utils
from sgl_kernel import (
    can_use_fused_scale_residual_norm_scale_shift,
    fused_scale_residual_norm_scale_shift,
)

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


def assert_matches_reference(*, eps=1e-6, **kwargs):
    dtype = kwargs["dtype"]
    kwargs.setdefault("param_dtype", dtype)
    residual, x, gate, shift, scale, weight, bias = make_operands(**kwargs)

    assert can_use_fused_scale_residual_norm_scale_shift(
        residual=residual,
        x=x,
        gate=gate,
        shift=shift,
        scale=scale,
        weight=weight,
        bias=bias,
    )

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
    rtol, atol = (1e-5, 1e-5) if dtype == torch.float32 else (1e-2, 1e-2)
    torch.testing.assert_close(residual_out, residual_out_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(out, out_ref, rtol=rtol, atol=atol)


# Hits every vector width (%8, %4, %2, odd), the 8192 bound, and sizes that are not
# a multiple of the 256-thread work-group.
@pytest.mark.parametrize("hidden", [64, 130, 255, 1536, 3072, 3080, 5120, 6660, 8192])
@pytest.mark.parametrize("seq_len", [1, 17, 1024])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_shapes(hidden, seq_len, dtype):
    assert_matches_reference(seq_len=seq_len, hidden=hidden, dtype=dtype)


@pytest.mark.parametrize("gate_mode", ["none", "per_token", "per_frame"])
@pytest.mark.parametrize("scale_kind", ["vector", "scalar"])
@pytest.mark.parametrize("shift_kind", ["vector", "scalar"])
@pytest.mark.parametrize("affine", [True, False, "weight_only", "bias_only"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_operand_variants(gate_mode, scale_kind, shift_kind, affine, dtype):
    hidden = 1536
    assert_matches_reference(
        seq_len=64,
        hidden=hidden,
        dtype=dtype,
        gate_mode=gate_mode,
        num_frames=8,
        scale_numel=hidden if scale_kind == "vector" else 1,
        shift_numel=hidden if shift_kind == "vector" else 1,
        affine=affine,
    )


@pytest.mark.parametrize("seq_len,num_frames", [(8, 8), (120, 5), (1024, 16)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_per_frame_gate(seq_len, num_frames, dtype):
    assert_matches_reference(
        seq_len=seq_len,
        hidden=2048,
        dtype=dtype,
        gate_mode="per_frame",
        num_frames=num_frames,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fp32_params(dtype):
    assert_matches_reference(
        seq_len=128, hidden=3072, dtype=dtype, param_dtype=torch.float32
    )


@pytest.mark.parametrize("eps", [1e-6, 1e-5, 1e-3])
def test_eps(eps):
    assert_matches_reference(seq_len=32, hidden=1536, dtype=torch.bfloat16, eps=eps)


@pytest.mark.parametrize("storage_offset", [1, 3])
def test_misaligned_params(storage_offset):
    """A misaligned but contiguous scale slice must narrow the vector width."""
    hidden, dtype = 1536, torch.bfloat16
    residual, x, gate, shift, scale, weight, bias = make_operands(
        seq_len=32,
        hidden=hidden,
        dtype=dtype,
        param_dtype=dtype,
        scale_numel=hidden + storage_offset,
    )
    scale = scale[storage_offset:]
    assert scale.is_contiguous() and scale.numel() == hidden

    out, residual_out = fused_scale_residual_norm_scale_shift(
        residual=residual,
        x=x,
        gate=gate,
        shift=shift,
        scale=scale,
        weight=weight,
        bias=bias,
        eps=1e-6,
    )
    out_ref, residual_out_ref = reference_scale_residual_norm_scale_shift(
        residual, x, gate, shift, scale, weight, bias, 1e-6
    )
    torch.testing.assert_close(residual_out, residual_out_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=1e-2)


def _base_operands():
    hidden, seq_len, dtype = 1536, 32, torch.bfloat16
    return hidden, dict(
        residual=torch.randn(1, seq_len, hidden, dtype=dtype, device=device),
        x=torch.randn(1, seq_len, hidden, dtype=dtype, device=device),
        gate=torch.randn(1, 1, hidden, dtype=dtype, device=device),
        shift=torch.randn(hidden, dtype=dtype, device=device),
        scale=torch.randn(hidden, dtype=dtype, device=device),
        weight=torch.randn(hidden, dtype=dtype, device=device),
        bias=torch.randn(hidden, dtype=dtype, device=device),
    )


def test_can_use_rejects_unsupported():
    hidden, base = _base_operands()
    dtype, seq_len = torch.bfloat16, 32
    assert can_use_fused_scale_residual_norm_scale_shift(**base)

    over_max = 2 * 8192
    cases = {
        "hidden above MAX_FUSED_HIDDEN": dict(
            residual=torch.randn(1, 4, over_max, dtype=dtype, device=device),
            x=torch.randn(1, 4, over_max, dtype=dtype, device=device),
            gate=torch.randn(1, 1, over_max, dtype=dtype, device=device),
            shift=torch.randn(over_max, dtype=dtype, device=device),
            scale=torch.randn(over_max, dtype=dtype, device=device),
            weight=torch.randn(over_max, dtype=dtype, device=device),
            bias=torch.randn(over_max, dtype=dtype, device=device),
        ),
        "batch size above 1": dict(
            residual=torch.randn(2, seq_len, hidden, dtype=dtype, device=device),
            x=torch.randn(2, seq_len, hidden, dtype=dtype, device=device),
        ),
        "non-contiguous x": dict(
            x=torch.randn(1, seq_len, 2 * hidden, dtype=dtype, device=device)[
                ..., :hidden
            ],
        ),
        "gate hidden mismatch": dict(
            gate=torch.randn(1, 1, hidden // 2, dtype=dtype, device=device)
        ),
        "gate int other than 1": dict(gate=2),
        "bool gate": dict(gate=True),
        "float gate": dict(gate=1.0),
        "num_frames not dividing seq_len": dict(
            gate=torch.randn(1, 7, 1, hidden, dtype=dtype, device=device)
        ),
        "modulation numel mismatch": dict(
            scale=torch.randn(3, dtype=dtype, device=device)
        ),
        "mixed parameter dtypes": dict(
            scale=torch.randn(hidden, dtype=torch.float32, device=device)
        ),
        "unsupported dtype": dict(
            residual=torch.randn(1, seq_len, hidden, device=device).double(),
            x=torch.randn(1, seq_len, hidden, device=device).double(),
        ),
    }
    for reason, override in cases.items():
        assert not can_use_fused_scale_residual_norm_scale_shift(
            **{**base, **override}
        ), reason


def test_kernel_rejects_bad_shape():
    _, base = _base_operands()
    bad = dict(base)
    bad["gate"] = torch.randn(
        1, 7, 1, base["x"].shape[-1], dtype=torch.bfloat16, device=device
    )
    with pytest.raises(RuntimeError, match="num_frames"):
        fused_scale_residual_norm_scale_shift(**bad, eps=1e-6)


@pytest.mark.parametrize("gate", [2, True, 1.0, 0])
def test_kernel_rejects_non_unit_int_gate(gate):
    """True and 1.0 compare equal to 1 but must not be read as "no gate"."""
    _, base = _base_operands()
    with pytest.raises(ValueError, match="gate of 1"):
        fused_scale_residual_norm_scale_shift(**{**base, "gate": gate}, eps=1e-6)


@pytest.mark.parametrize("affine", ["weight_only", "bias_only"])
def test_lone_weight_or_bias(affine):
    assert_matches_reference(
        seq_len=64, hidden=1536, dtype=torch.bfloat16, affine=affine
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
