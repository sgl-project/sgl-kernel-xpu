"""Inkling MoE gate projection and top-k/renormalization XPU ops."""

from __future__ import annotations

import torch

# Device timings favor torch.mm for both dtypes through 512 tokens.
_PROJECTION_MM_MAX_TOKENS = 512


def inkling_moe_gate_projection(
    input: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Compute ``input @ weight.T`` into FP32 padded gate logits.

    ``input`` is ``[tokens, 6144]`` and ``weight`` is the model's padded
    ``[264, 6144]`` parameter. The returned ``[tokens, 258]`` view keeps a
    physical row stride of 264 for the top-k kernel.
    """
    if input.dim() == 2 and input.size(0) <= _PROJECTION_MM_MAX_TOKENS:
        if input.size(1) != 6144:
            raise RuntimeError("input must have shape [tokens, 6144]")
        if weight.shape != (264, 6144):
            raise RuntimeError("weight must have shape [264, 6144]")
        if (
            input.dtype not in (torch.bfloat16, torch.float16)
            or weight.dtype != input.dtype
        ):
            raise RuntimeError(
                "input and weight must have matching bfloat16 or float16 dtype"
            )
        if input.device.type != "xpu" or weight.device != input.device:
            raise RuntimeError("input and weight must be on the same XPU device")
        if not input.is_contiguous() or not weight.is_contiguous():
            raise RuntimeError("input and weight must be contiguous")
        return torch.mm(input, weight.T, out_dtype=torch.float32)[:, :258]
    return torch.ops.sgl_kernel.inkling_moe_gate_projection.default(input, weight)


def inkling_moe_gate_topk_renorm(
    logits: torch.Tensor,
    bias: torch.Tensor,
    global_scale: torch.Tensor,
    route_scale: float,
    *,
    return_packed: bool = False,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor | None,
]:
    """Select six routed experts and renormalize with two shared experts."""
    output = torch.ops.sgl_kernel.inkling_moe_gate_topk_renorm.default(
        logits,
        bias,
        global_scale,
        float(route_scale),
        bool(return_packed),
    )
    if return_packed:
        packed, shared_weights = output
        return None, None, shared_weights, packed
    routed_weights, indices, shared_weights = output
    return routed_weights, indices, shared_weights, None


__all__ = [
    "inkling_moe_gate_projection",
    "inkling_moe_gate_topk_renorm",
]
