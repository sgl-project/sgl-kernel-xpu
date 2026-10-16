"""Python entry point for QSA sparse attention on XPU."""

from typing import Optional

import torch


def qsa_sparse_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    token_slots: torch.Tensor,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Apply sparse attention over physical K/V cache slots."""
    return torch.ops.sgl_kernel.qsa_sparse_attention(
        q, k_cache, v_cache, token_slots, softmax_scale
    )
