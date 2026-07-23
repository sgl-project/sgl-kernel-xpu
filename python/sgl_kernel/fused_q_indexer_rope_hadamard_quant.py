from __future__ import annotations

import torch


def fused_q_indexer_rope_hadamard_quant(
    q_input: torch.Tensor,
    q_fp8: torch.Tensor,
    weight: torch.Tensor,
    weights_out: torch.Tensor,
    weight_scale: float,
    rope_cache: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    op = torch.ops.sgl_kernel.fused_q_indexer_rope_hadamard_quant
    op.default(
        q_input,
        q_fp8,
        weight,
        weights_out,
        float(weight_scale),
        rope_cache,
        positions,
    )
