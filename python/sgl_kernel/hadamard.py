from __future__ import annotations

import torch


def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return torch.ops.sgl_kernel.hadamard_transform(x, scale)


__all__ = ["hadamard_transform"]
