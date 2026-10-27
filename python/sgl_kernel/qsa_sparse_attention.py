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


def qsa_mqa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Compute the attention scores for prefill with MQA."""
    return torch.ops.sgl_kernel.qsa_mqa_prefill(q, k, starts, ends, scale)


def qsa_mqa_decode(
    q: torch.Tensor,
    cache: torch.Tensor,
    pages: torch.Tensor,
    lengths: torch.Tensor,
    columns: int,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Compute the attention scores for decode with MQA."""
    return torch.ops.sgl_kernel.qsa_mqa_decode(q, cache, pages, lengths, columns, scale)
