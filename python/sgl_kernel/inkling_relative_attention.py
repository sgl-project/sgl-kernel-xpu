from __future__ import annotations

import importlib
from typing import Optional, Tuple

import torch


def _ops_registered() -> bool:
    return hasattr(torch.ops.sgl_kernel, "inkling_relative_attention")


def _ensure_ops_registered() -> None:
    if _ops_registered():
        return
    try:
        importlib.import_module("sgl_kernel.inkling_relative_attention_ops")
    except ImportError as exc:
        raise ImportError(
            "Inkling relative attention ops are not registered. Build/install the "
            "inkling_relative_attention_ops extension before calling "
            "sgl_kernel.inkling_relative_attention."
        ) from exc
    if not _ops_registered():
        raise RuntimeError(
            "sgl_kernel.inkling_relative_attention_ops loaded without registering "
            "Inkling relative attention ops"
        )


def inkling_relative_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_to_seq: torch.Tensor,
    q_pos: torch.Tensor,
    cu_k: torch.Tensor,
    *,
    rel_bias: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    local_size: int = 0,
    out: Optional[torch.Tensor] = None,
    return_softmax_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Packed ragged Inkling relative-bias attention for XPU.

    ``q`` is ``[total_q, q_heads, d]`` and ``k``/``v`` are
    ``[total_k, kv_heads, d/dv]``. ``q_to_seq`` maps each query row to a sequence,
    ``q_pos`` is the query's absolute position within that sequence, and ``cu_k``
    is the cumulative KV length array. ``rel_bias`` is optional
    ``[total_q, q_heads, rel_extent]`` and contributes
    ``rel_bias[q_idx, head, q_pos - kv_pos]`` when the relative index is valid.
    """
    _ensure_ops_registered()
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    result = torch.ops.sgl_kernel.inkling_relative_attention(
        q,
        k,
        v,
        q_to_seq,
        q_pos,
        cu_k,
        rel_bias,
        float(softmax_scale),
        bool(causal),
        int(window_size[0]),
        int(window_size[1]),
        float(softcap),
        int(local_size),
        out,
    )
    return result if return_softmax_lse else result[0]
