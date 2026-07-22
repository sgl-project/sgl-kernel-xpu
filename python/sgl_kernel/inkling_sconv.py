from __future__ import annotations

import importlib
from typing import Optional, TypedDict

import torch

PAD_SLOT_ID = -1
CHUNK_SIZE = 64

HIS_ZEROS = 0
HIS_PREFIX = 1
HIS_SEQ_MINUS_EXT = 2
HIS_ONES = 3
_FUSED_EXTEND_MAX_B = 1023


class SconvDecodeMetadata(TypedDict):
    cache_mask: torch.Tensor
    safe_idx: torch.Tensor
    cu: torch.Tensor
    si: torch.Tensor


class SconvExtendMetadata(TypedDict):
    cache_mask: torch.Tensor
    safe_idx: torch.Tensor
    cu: torch.Tensor
    si: torch.Tensor


def _as_int32(x: torch.Tensor) -> torch.Tensor:
    return x if x.dtype == torch.int32 else x.to(torch.int32)


def _as_int64(x: torch.Tensor) -> torch.Tensor:
    return x if x.dtype == torch.int64 else x.to(torch.int64)


def _activation_is_silu(activation: Optional[str]) -> bool:
    if activation == "swish":
        activation = "silu"
    if activation not in (None, "silu"):
        raise NotImplementedError("activation must be None, silu, or swish")
    return activation == "silu"


def _ops_registered() -> bool:
    return hasattr(torch.ops.sgl_kernel, "inkling_sconv_forward")


def _ensure_ops_registered() -> None:
    if _ops_registered():
        return
    try:
        importlib.import_module("sgl_kernel.inkling_sconv_ops")
    except ImportError as exc:
        raise ImportError(
            "Inkling sconv ops are not registered. Build/install the "
            "inkling_sconv_ops extension before calling sgl_kernel.inkling_sconv."
        ) from exc
    if not _ops_registered():
        raise RuntimeError(
            "sgl_kernel.inkling_sconv_ops loaded without registering Inkling sconv ops"
        )


def fused_decode_sconv_metadata(
    B: int, cache_indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, SconvDecodeMetadata]:
    _ensure_ops_registered()
    (
        query_start_loc,
        has_initial_state,
        cache_mask,
        safe_idx,
        cu,
        si,
    ) = torch.ops.sgl_kernel.inkling_fused_decode_sconv_metadata(
        B, _as_int32(cache_indices)
    )
    return (
        query_start_loc,
        has_initial_state,
        SconvDecodeMetadata(cache_mask=cache_mask, safe_idx=safe_idx, cu=cu, si=si),
    )


def fused_extend_sconv_metadata(
    *,
    B: int,
    T: int,
    cache_indices: torch.Tensor,
    his_mode: int,
    extend_seq_lens: Optional[torch.Tensor] = None,
    his_src: Optional[torch.Tensor] = None,
    draft_token_num: Optional[int] = None,
) -> Optional[tuple[torch.Tensor, torch.Tensor, SconvExtendMetadata]]:
    if B > _FUSED_EXTEND_MAX_B or not getattr(cache_indices, "is_xpu", False):
        return None
    _ensure_ops_registered()
    (
        query_start_loc,
        has_initial_state,
        cache_mask,
        safe_idx,
        cu,
        si,
    ) = torch.ops.sgl_kernel.inkling_fused_extend_sconv_metadata(
        B,
        T,
        _as_int32(cache_indices),
        his_mode,
        _as_int32(extend_seq_lens) if extend_seq_lens is not None else None,
        _as_int32(his_src) if his_src is not None else None,
        1 if draft_token_num is None else int(draft_token_num),
    )
    return (
        query_start_loc,
        has_initial_state,
        SconvExtendMetadata(cache_mask=cache_mask, safe_idx=safe_idx, cu=cu, si=si),
    )


def precompute_helion_decode_metadata(
    B: int,
    W: int,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> SconvDecodeMetadata:
    del W
    valid = cache_indices != PAD_SLOT_ID
    cache_mask = (has_initial_state & valid)[:, None, None]
    safe_idx = cache_indices.clamp(min=0).to(torch.int64)
    cu = torch.arange(B + 1, dtype=torch.int64, device=cache_indices.device)
    si = torch.arange(B, dtype=torch.int32, device=cache_indices.device)
    return SconvDecodeMetadata(
        cache_mask=cache_mask,
        safe_idx=safe_idx,
        cu=cu,
        si=si,
    )


def precompute_helion_extend_metadata(
    B: int,
    T: int,
    W: int,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> SconvExtendMetadata:
    del W
    valid = cache_indices != PAD_SLOT_ID
    cache_mask = (has_initial_state & valid)[:, None, None]
    safe_idx = cache_indices.clamp(min=0).to(torch.int64)
    cu = query_start_loc.to(torch.int64)
    t = torch.arange(T, dtype=torch.int64, device=cache_indices.device)
    si = (torch.searchsorted(cu, t, right=True) - 1).clamp(max=B - 1).to(torch.int32)
    return SconvExtendMetadata(
        cache_mask=cache_mask,
        safe_idx=safe_idx,
        cu=cu,
        si=si,
    )


def track_conv_indices(
    query_start_loc: torch.Tensor,
    mamba_track_seqlens: torch.Tensor,
    extend_prefix_lens: torch.Tensor,
    width_minus_one: int,
    chunk_size: int = CHUNK_SIZE,
    total_tokens: Optional[int] = None,
) -> torch.Tensor:
    _ensure_ops_registered()
    if total_tokens is None:
        total_tokens = int(query_start_loc[-1].item())
    return torch.ops.sgl_kernel.inkling_track_conv_indices(
        _as_int32(query_start_loc),
        _as_int32(mamba_track_seqlens),
        _as_int32(extend_prefix_lens),
        int(width_minus_one),
        int(chunk_size),
        int(total_tokens),
    )


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    sconv_cache: torch.Tensor,
    cache_mask: torch.Tensor,
    safe_idx: torch.Tensor,
    cu: torch.Tensor,
    si: torch.Tensor,
    activation: Optional[str] = None,
    use_residual: bool = True,
    is_decode: bool = False,
) -> torch.Tensor:
    _ensure_ops_registered()
    if x.shape[0] == 0:
        return torch.empty_like(x)
    return torch.ops.sgl_kernel.inkling_sconv_forward(
        x,
        weight,
        sconv_cache,
        cache_mask,
        _as_int64(safe_idx),
        _as_int64(cu),
        _as_int32(si),
        _activation_is_silu(activation),
        bool(use_residual),
        bool(is_decode),
    )


def update_sconv_cache(
    x: torch.Tensor,
    sconv_cache: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> None:
    _ensure_ops_registered()
    torch.ops.sgl_kernel.inkling_update_sconv_cache(
        x,
        sconv_cache,
        _as_int32(cache_indices),
        has_initial_state,
        _as_int32(query_start_loc),
    )


def fused_causal_conv1d_update_decode(
    x: torch.Tensor,
    weight: torch.Tensor,
    sconv_cache: torch.Tensor,
    cache_indices: torch.Tensor,
    cache_mask: torch.Tensor,
    activation: Optional[str] = None,
    use_residual: bool = True,
    track_mask: Optional[torch.Tensor] = None,
    track_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    _ensure_ops_registered()
    return torch.ops.sgl_kernel.inkling_fused_decode_update_sconv(
        x,
        weight,
        sconv_cache,
        _as_int32(cache_indices),
        cache_mask.reshape(-1),
        _activation_is_silu(activation),
        bool(use_residual),
        track_mask.reshape(-1) if track_mask is not None else None,
        _as_int64(track_indices) if track_indices is not None else None,
    )


def fused_gather_scatter_to_sconv_cache(
    hidden_states: torch.Tensor,
    sconv_cache: torch.Tensor,
    track_conv_indices: torch.Tensor,
    mask: torch.Tensor,
    dst_indices: torch.Tensor,
) -> None:
    _ensure_ops_registered()
    torch.ops.sgl_kernel.inkling_gather_scatter_sconv_cache(
        hidden_states,
        sconv_cache,
        _as_int32(track_conv_indices),
        mask,
        _as_int64(dst_indices),
    )


def fused_draft_extend_sconv_cache(
    hidden_states: torch.Tensor,
    sconv_cache: torch.Tensor,
    cache_indices: torch.Tensor,
    num_accept_tokens: Optional[torch.Tensor] = None,
    draft_token_num: int = 0,
    do_tracking: bool = False,
    crossed: Optional[torch.Tensor] = None,
    track_step: Optional[torch.Tensor] = None,
    mamba_track_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
) -> None:
    _ensure_ops_registered()
    if num_accepted_tokens is None:
        if num_accept_tokens is None:
            raise TypeError("num_accept_tokens or num_accepted_tokens is required")
        num_accepted_tokens = num_accept_tokens
    torch.ops.sgl_kernel.inkling_draft_extend_sconv_cache(
        hidden_states,
        sconv_cache,
        _as_int32(cache_indices),
        _as_int32(num_accepted_tokens),
        int(draft_token_num),
        bool(do_tracking),
        crossed if do_tracking else None,
        _as_int32(track_step) if do_tracking and track_step is not None else None,
        (
            _as_int64(mamba_track_indices)
            if do_tracking and mamba_track_indices is not None
            else None
        ),
    )


def save_intermediate_conv_windows(
    sconv_cache: torch.Tensor,
    hidden_states: torch.Tensor,
    cache_indices: torch.Tensor,
    intermediate_out: torch.Tensor,
    batch_size: int,
    draft_token_num: int,
) -> None:
    _ensure_ops_registered()
    torch.ops.sgl_kernel.inkling_save_intermediate_conv_windows(
        sconv_cache,
        hidden_states,
        _as_int32(cache_indices),
        intermediate_out,
        int(batch_size),
        int(draft_token_num),
    )
