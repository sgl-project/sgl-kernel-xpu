from typing import Optional, Tuple, Union

import torch

# Mirrors kSrnssMaxHidden in src/sycl/ScaleResidualNormScaleShift.cpp.
MAX_FUSED_HIDDEN = 8192

_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _is_ungated(gate: Union[torch.Tensor, int]) -> bool:
    # bool is a subclass of int and True == 1, so exclude it explicitly; a float 1.0
    # would also compare equal and must not be read as "no gate".
    return type(gate) is int and gate == 1


def can_use_fused_scale_residual_norm_scale_shift(
    *,
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: Union[torch.Tensor, int],
    shift: torch.Tensor,
    scale: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
) -> bool:
    r"""Whether :func:`fused_scale_residual_norm_scale_shift` supports these operands.

    Lets a caller fall back to eager without paying for an exception; the kernel
    re-checks every condition anyway.
    """
    if x.device.type != "xpu" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.dim() != 3 or x.shape[0] != 1 or not x.is_contiguous():
        return False
    for operand in (residual, gate, shift, scale, weight, bias):
        if isinstance(operand, torch.Tensor) and (
            operand.device != x.device or not operand.is_contiguous()
        ):
            return False
    if residual.shape != x.shape or residual.dtype != x.dtype:
        return False
    hidden = x.shape[-1]
    if hidden > MAX_FUSED_HIDDEN:
        return False
    if isinstance(gate, torch.Tensor):
        if gate.dim() not in (3, 4) or gate.shape[0] != 1 or gate.shape[-1] != hidden:
            return False
        if gate.dim() == 3:
            if gate.shape[1] != 1:
                return False
        elif gate.shape[2] != 1 or x.shape[1] % gate.shape[1] != 0:
            return False
    elif not _is_ungated(gate):
        return False
    for modulation in (scale, shift):
        if not isinstance(modulation, torch.Tensor):
            return False
        if modulation.numel() not in (1, hidden):
            return False
    if weight is not None and weight.numel() != hidden:
        return False
    if bias is not None and bias.numel() != hidden:
        return False

    param_dtypes = {shift.dtype, scale.dtype}
    if isinstance(gate, torch.Tensor):
        param_dtypes.add(gate.dtype)
    if weight is not None:
        param_dtypes.add(weight.dtype)
    if bias is not None:
        param_dtypes.add(bias.dtype)
    if len(param_dtypes) != 1:
        return False
    return param_dtypes.pop() in _SUPPORTED_DTYPES


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
    if not isinstance(gate, torch.Tensor) and not _is_ungated(gate):
        raise ValueError(f"Only an int gate of 1 is supported, got {gate!r}")

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
