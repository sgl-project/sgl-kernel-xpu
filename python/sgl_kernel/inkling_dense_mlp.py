from __future__ import annotations

import torch

_ROW_PADDING = 64
_INTERLEAVE = 16


def _padded_k_by_n(matrix: torch.Tensor) -> torch.Tensor:
    k, n = matrix.shape
    storage = torch.empty(
        (k, n + _ROW_PADDING), dtype=matrix.dtype, device=matrix.device
    )
    storage[:, :n].copy_(matrix)
    return storage[:, :n]


def repack_inkling_gate_up_weight(weight: torch.Tensor) -> torch.Tensor:
    """Repack a gate-major [2*I,K] BF16 weight for fused gate/up + SwiGLU."""
    if weight.ndim != 2 or weight.shape[0] % (2 * _INTERLEAVE):
        raise ValueError("gate/up weight must be [2*I,K] with 2*I divisible by 32")
    if weight.dtype is not torch.bfloat16:
        raise TypeError("gate/up weight must be bfloat16")
    n, k = weight.shape
    intermediate = n // 2
    interleaved = (
        weight.reshape(2, intermediate // _INTERLEAVE, _INTERLEAVE, k)
        .permute(3, 1, 0, 2)
        .reshape(k, n)
        .contiguous()
    )
    return _padded_k_by_n(interleaved)


def repack_inkling_down_weight(weight: torch.Tensor) -> torch.Tensor:
    """Repack a conventional [N,K] BF16 down weight as padded logical [K,N]."""
    if weight.ndim != 2:
        raise ValueError("down weight must have shape [N,K]")
    if weight.dtype is not torch.bfloat16:
        raise TypeError("down weight must be bfloat16")
    return _padded_k_by_n(weight.t().contiguous())


def inkling_gate_up(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run gate/up GEMM with fused SwiGLU, returning padded logical [M,I]."""
    if out is None:
        return torch.ops.sgl_kernel.inkling_gate_up.default(input, packed_weight)
    torch.ops.sgl_kernel.inkling_gate_up_out.default(input, packed_weight, out)
    return out


def inkling_down(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the down projection, returning contiguous logical [M,H]."""
    if out is None:
        return torch.ops.sgl_kernel.inkling_down.default(input, packed_weight)
    torch.ops.sgl_kernel.inkling_down_out.default(input, packed_weight, out)
    return out
