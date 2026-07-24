from __future__ import annotations

import torch


def row_scale_bf16(
    x: torch.Tensor, tau: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Scale each bf16 row by its fp32 tau value and return a contiguous output."""
    if out is None:
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    return torch.ops.sgl_kernel.inkling_row_scale_bf16.default(x, tau, out)


def row_compact_bf16(
    x: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Copy a row-strided bf16 tensor into a contiguous output."""
    if out is None:
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    return torch.ops.sgl_kernel.inkling_row_compact_bf16.default(x, out)
