from typing import Optional, Tuple

import torch


def hc_silu(input: torch.Tensor, hc_count: int) -> torch.Tensor:
    r"""HyperConnection mix activation: ``silu(input / hc_count)``.

    ``input`` is a contiguous BF16 tensor of any shape.
    """
    out = torch.empty_like(input)
    torch.ops.sgl_kernel.qwen4_hc_silu(out, input, hc_count)
    return out


def hc_gate_mix(input: torch.Tensor, gate: torch.Tensor, hc_count: int) -> torch.Tensor:
    r"""HyperConnection input mix: ``mean_s(sigmoid(gate_s) * input_s)``.

    ``input`` and ``gate`` are contiguous BF16 ``(tokens, hc_count * hidden)``;
    the result is ``(tokens, hidden)``.
    """
    if input.dim() != 2 or hc_count <= 0:
        raise ValueError("input must be 2D and hc_count positive")
    out = input.new_empty(input.shape[0], input.shape[1] // hc_count)
    torch.ops.sgl_kernel.qwen4_hc_gate_mix(out, input, gate, hc_count)
    return out


def hc_combine(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection: Optional[torch.Tensor],
    hc_count: int,
) -> torch.Tensor:
    r"""HyperConnection combine: ``residual_s + block_output * 2 * sigmoid(injection_s / hc_count)``.

    ``residual`` is contiguous BF16 ``(tokens, hc_count * hidden)``,
    ``block_output`` is ``(tokens, hidden)`` and ``injection`` is the
    ``(tokens, hc_count)`` output of the inject projection. With
    ``injection=None`` the weight is 1.
    """
    out = torch.empty_like(residual)
    torch.ops.sgl_kernel.qwen4_hc_combine(
        out, residual, block_output, injection, hc_count
    )
    return out


def hc_combine_norm(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection: Optional[torch.Tensor],
    weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""``hc_combine`` fused with the next grouped Gemma RMSNorm.

    Returns ``(combined, normalized)``, where ``normalized`` is the per-stream
    ``combined * rsqrt(mean(combined^2) + eps) * (1 + weight)`` of the
    BF16-rounded ``combined``. ``weight`` is BF16 with ``hidden`` or
    ``hc_count * hidden`` elements.
    """
    combined = torch.empty_like(residual)
    normalized = torch.empty_like(residual)
    torch.ops.sgl_kernel.qwen4_hc_combine_norm(
        combined, normalized, residual, block_output, injection, weight, eps, hc_count
    )
    return combined, normalized
