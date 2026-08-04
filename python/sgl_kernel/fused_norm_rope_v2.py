"""XPU fused norm+rope+store wrapper.

This module dispatches directly to the native XPU op and keeps the same public
Python signature as before.
"""

from __future__ import annotations

import torch


def compress_norm_rope_store(
    input: torch.Tensor,  # (N, head_dim)
    plan: torch.Tensor,  # (N, 16) uint8
    norm_weight: torch.Tensor,  # (head_dim,)
    norm_eps: float,
    freq_cis: torch.Tensor,  # (max_pos, rope_dim) fp32 interleaved
    out_loc: torch.Tensor,  # (M,)
    kvcache: torch.Tensor,  # (npages, page_bytes) uint8
    is_decode: bool,
    compress_ratio: int,
    page_size: int,
    use_fp4: bool,
) -> None:
    """FusedNormRopeKernel::forward for XPU.

    This path is intentionally op-only (no Python fallback).
    """
    torch.ops.sgl_kernel.fused_norm_rope_store(
        input,
        plan,
        norm_weight,
        float(norm_eps),
        freq_cis,
        out_loc,
        kvcache,
        is_decode,
        int(compress_ratio),
        int(page_size),
        use_fp4,
    )
