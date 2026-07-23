from __future__ import annotations

import math

import torch
import torch.nn.functional as F

_SUPPORTED_XPU_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_MIN_LOG_N = 3
_MAX_LOG_N = 15
_MAX_N = 1 << _MAX_LOG_N


def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    if not x.is_xpu:
        raise RuntimeError("hadamard_transform only supports XPU tensors")
    if x.dtype not in _SUPPORTED_XPU_DTYPES:
        raise RuntimeError(
            f"hadamard_transform only supports dtypes {_SUPPORTED_XPU_DTYPES}, got {x.dtype}"
        )

    dim_og = x.size(-1)
    if not (0 < dim_og <= _MAX_N):
        raise RuntimeError(
            f"hadamard_transform only supports last dim in [1, {_MAX_N}], got {dim_og}"
        )

    shapes_og = x.size()

    x_flat = x.reshape(-1, dim_og)
    if x_flat.stride(-1) != 1:
        x_flat = x_flat.contiguous()

    log_n = max(_MIN_LOG_N, math.ceil(math.log2(max(dim_og, 1))))
    padded_dim = 1 << log_n

    if padded_dim != dim_og:
        x_flat = F.pad(x_flat, (0, padded_dim - dim_og))
    else:
        x_flat = x_flat.contiguous()

    out = torch.empty_like(x_flat)
    torch.ops.sgl_kernel.hadamard_transform(out, x_flat, float(scale))

    if padded_dim != dim_og:
        out = out[:, :dim_og]
    return out.reshape(shapes_og)


__all__ = ["hadamard_transform"]
