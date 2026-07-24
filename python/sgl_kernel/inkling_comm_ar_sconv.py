from __future__ import annotations

from typing import Optional

import torch

from sgl_kernel.inkling_sconv import (
    _activation_is_silu,
    _as_int32,
    _as_int64,
    _ensure_ops_registered,
)

ALL_REDUCE_DIRECT = 0
ALL_REDUCE_TWO_SHOT = 1
ALL_REDUCE_FULL_ONESHOT = 2
ALL_REDUCE_PUSH_ONESHOT = 3

_VARIANTS = {
    "direct": ALL_REDUCE_DIRECT,
    "fallback": ALL_REDUCE_DIRECT,
    "two_shot": ALL_REDUCE_TWO_SHOT,
    "full_oneshot": ALL_REDUCE_FULL_ONESHOT,
    "v4": ALL_REDUCE_FULL_ONESHOT,
    "push_oneshot": ALL_REDUCE_PUSH_ONESHOT,
    "v5": ALL_REDUCE_PUSH_ONESHOT,
}


def _variant_id(variant: int | str) -> int:
    if isinstance(variant, int):
        if variant not in (
            ALL_REDUCE_DIRECT,
            ALL_REDUCE_TWO_SHOT,
            ALL_REDUCE_FULL_ONESHOT,
            ALL_REDUCE_PUSH_ONESHOT,
        ):
            raise ValueError(f"unsupported Inkling all-reduce variant: {variant}")
        return variant
    try:
        return _VARIANTS[variant]
    except KeyError as exc:
        raise ValueError(f"unsupported Inkling all-reduce variant: {variant}") from exc


def comm_all_reduce(
    partials: torch.Tensor,
    shared: Optional[torch.Tensor] = None,
    *,
    variant: int | str = "direct",
) -> torch.Tensor:
    _ensure_ops_registered()
    return torch.ops.sgl_kernel.inkling_comm_all_reduce(
        partials, shared, _variant_id(variant)
    )


def ar_fused_decode(
    partials: torch.Tensor,
    residual: torch.Tensor,
    sconv_cache: torch.Tensor,
    cache_indices: torch.Tensor,
    cache_mask: torch.Tensor,
    weight: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float = 1.0e-5,
    activation: Optional[str] = None,
    use_residual: bool = True,
    shared: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _ensure_ops_registered()
    return torch.ops.sgl_kernel.inkling_ar_fused_decode(
        partials,
        residual,
        sconv_cache,
        _as_int32(cache_indices),
        cache_mask.reshape(-1),
        weight,
        norm_weight,
        float(eps),
        _activation_is_silu(activation),
        bool(use_residual),
        shared,
    )


def ar_scattered_sconv(
    partials: torch.Tensor,
    sconv_cache: torch.Tensor,
    cache_indices: torch.Tensor,
    cache_mask: torch.Tensor,
    cu: torch.Tensor,
    si: torch.Tensor,
    weight: torch.Tensor,
    has_initial_state: torch.Tensor,
    activation: Optional[str] = None,
    use_residual: bool = True,
    shared: Optional[torch.Tensor] = None,
    update_cache: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    _ensure_ops_registered()
    return torch.ops.sgl_kernel.inkling_ar_scattered_sconv(
        partials,
        sconv_cache,
        _as_int32(cache_indices),
        cache_mask.reshape(-1),
        _as_int64(cu),
        _as_int32(si),
        weight,
        has_initial_state,
        _activation_is_silu(activation),
        bool(use_residual),
        shared,
        bool(update_cache),
    )
