"""MiniMax-H3 indexed gated residual on XPU."""

import torch


def can_use_h3_indexed_gate(x, gate, other, indices) -> bool:
    return (
        x.device.type == "xpu"
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and x.is_contiguous()
        and x.shape[0] > 0
        and 0 < x.shape[1] <= 8192
        and other.device == x.device
        and other.dtype == x.dtype
        and other.shape == x.shape
        and other.is_contiguous()
        and gate.device == x.device
        and gate.dtype == x.dtype
        and gate.ndim == 2
        and gate.shape[1] == x.shape[1]
        and gate.stride(1) == 1
        and indices.device == x.device
        and indices.ndim == 1
        and indices.numel() == x.shape[0]
        and indices.dtype in (torch.int32, torch.int64)
    )


def h3_indexed_gate(x, gate, other, indices, *, inplace: bool = False):
    if not can_use_h3_indexed_gate(x, gate, other, indices):
        raise ValueError("unsupported H3 indexed gate shape, dtype, layout or device")
    out = x if inplace else torch.empty_like(x)
    torch.ops.sgl_kernel.h3_indexed_gate(out, x, gate, other, indices)
    return out
