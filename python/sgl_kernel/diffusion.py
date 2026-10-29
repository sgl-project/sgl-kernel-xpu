from typing import Optional, Tuple, Union

import torch

# Mirrors kSrnssMaxHidden in src/sycl/ScaleResidualNormScaleShift.cpp.
MAX_FUSED_HIDDEN = 8192

_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _is_ungated(gate: Union[torch.Tensor, int]) -> bool:
    # bool is a subclass of int and True == 1, so exclude it explicitly; a float 1.0
    # would also compare equal and must not be read as "no gate".
    return type(gate) is int and gate == 1


def fused_scale_residual_norm_scale_shift(
    *,
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: Union[torch.Tensor, int],
    shift: torch.Tensor,
    scale: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Fused scale-residual + LayerNorm + scale-shift, as used by diffusion DiT blocks.

    ``residual_output = residual + x * gate``

    ``out = layer_norm(residual_output)[* weight + bias] * (1 + scale) + shift``

    ``gate`` is ``(1, 1, hidden)``, or ``(1, num_frames, 1, hidden)`` with
    ``num_frames`` dividing ``seq_len``, or the int ``1``. ``scale`` and ``shift``
    are ``(hidden,)`` or single elements. ``weight`` and ``bias`` are ``(hidden,)``
    and independently optional. Returns ``(out, residual_output)``.
    """
    if x.device.type != "xpu" or x.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"x must be an XPU float32, float16 or bfloat16 tensor, got {x.dtype} on {x.device}"
        )
    if x.dim() != 3 or x.shape[0] != 1:
        raise ValueError(f"x must be [1, seq_len, hidden], got {tuple(x.shape)}")
    if residual.shape != x.shape or residual.dtype != x.dtype:
        raise ValueError("residual must match x in shape and dtype")
    for name, operand in (
        ("residual", residual),
        ("x", x),
        ("gate", gate),
        ("shift", shift),
        ("scale", scale),
        ("weight", weight),
        ("bias", bias),
    ):
        if isinstance(operand, torch.Tensor) and (
            operand.device.type != "xpu" or not operand.is_contiguous()
        ):
            raise ValueError(f"{name} must be a contiguous XPU tensor")
    hidden = x.shape[-1]
    if hidden > MAX_FUSED_HIDDEN:
        raise ValueError(f"hidden must be <= {MAX_FUSED_HIDDEN}, got {hidden}")
    if isinstance(gate, torch.Tensor):
        if gate.dim() == 3:
            gate_ok = gate.shape[:2] == (1, 1) and gate.shape[2] == hidden
        elif gate.dim() == 4:
            num_frames = gate.shape[1]
            gate_ok = (
                gate.shape[0] == 1
                and gate.shape[2] == 1
                and gate.shape[3] == hidden
                and num_frames > 0
                and x.shape[1] % num_frames == 0
            )
        else:
            gate_ok = False
        if not gate_ok:
            raise ValueError(
                "gate must be [1, 1, hidden] or [1, num_frames, 1, hidden] with a "
                f"positive num_frames dividing seq_len, got {tuple(gate.shape)}"
            )
    elif not _is_ungated(gate):
        raise ValueError(f"Only an int gate of 1 is supported, got {gate!r}")
    for name, modulation in (("scale", scale), ("shift", shift)):
        if modulation.numel() not in (1, hidden):
            raise ValueError(
                f"{name} must have 1 or hidden elements, got {modulation.numel()}"
            )
    for name, affine in (("weight", weight), ("bias", bias)):
        if affine is not None and affine.numel() != hidden:
            raise ValueError(f"{name} must have hidden elements, got {affine.numel()}")
    param_dtypes = {
        p.dtype
        for p in (gate, shift, scale, weight, bias)
        if isinstance(p, torch.Tensor)
    }
    if len(param_dtypes) != 1 or shift.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            "gate/weight/bias/scale/shift must share one float32, float16 or "
            f"bfloat16 dtype, got {param_dtypes}"
        )

    return torch.ops.sgl_kernel.fused_scale_residual_norm_scale_shift(
        residual,
        x,
        gate if isinstance(gate, torch.Tensor) else None,
        weight,
        bias,
        scale,
        shift,
        eps,
    )
