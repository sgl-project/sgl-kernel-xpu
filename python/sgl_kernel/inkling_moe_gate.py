"""Inkling MoE gate projection and top-k/renormalization XPU ops."""

from __future__ import annotations

import torch


def inkling_moe_gate_projection(
    input: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Compute ``input @ weight.T`` into FP32 padded gate logits.

    ``input`` is ``[tokens, 6144]`` and ``weight`` is the model's padded
    ``[264, 6144]`` parameter. The returned ``[tokens, 258]`` view keeps a
    physical row stride of 264 for the top-k kernel.
    """
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
