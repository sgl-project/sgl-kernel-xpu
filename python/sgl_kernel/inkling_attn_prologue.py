from __future__ import annotations

import importlib
from typing import Optional

import torch


def _activation_is_silu(activation: Optional[str]) -> bool:
    if activation == "swish":
        activation = "silu"
    if activation not in (None, "silu"):
        raise NotImplementedError("activation must be None, silu, or swish")
    return activation == "silu"


def _as_int32(x: torch.Tensor) -> torch.Tensor:
    return x if x.dtype == torch.int32 else x.to(torch.int32)


def _as_int64(x: torch.Tensor) -> torch.Tensor:
    return x if x.dtype == torch.int64 else x.to(torch.int64)


def _ops_registered() -> bool:
    return hasattr(torch.ops.sgl_kernel, "inkling_attn_prologue_verify")


def _ensure_ops_registered() -> None:
    if _ops_registered():
        return
    try:
        importlib.import_module("sgl_kernel.inkling_attn_prologue_ops")
    except ImportError as exc:
        raise ImportError(
            "Inkling attention prologue ops are not registered. Build/install "
            "the inkling_attn_prologue_ops extension before calling "
            "sgl_kernel.inkling_attn_prologue."
        ) from exc
    if not _ops_registered():
        raise RuntimeError(
            "sgl_kernel.inkling_attn_prologue_ops loaded without registering "
            "Inkling attention prologue ops"
        )


def _reject_mxfp8_or_tau(
    *,
    mxfp8_quant: bool,
    log_tau: Optional[torch.Tensor],
) -> None:
    if mxfp8_quant:
        raise NotImplementedError(
            "Inkling XPU attention prologue MXFP8 store is not wired yet"
        )
    if log_tau is not None and log_tau.numel() > 0:
        raise NotImplementedError(
            "Inkling XPU attention prologue log_tau path is not wired yet"
        )


def _kv_view(buf: torch.Tensor, dkv: int) -> torch.Tensor:
    if buf.dim() == 2 and buf.shape[1] == dkv:
        return buf
    return buf.view(-1, dkv)


def _infer_weight_width(weight: torch.Tensor, dkv: int) -> int:
    if weight.dim() != 2:
        raise ValueError("weight must be 2D")
    if weight.shape[0] == dkv:
        return int(weight.shape[1])
    if weight.shape[1] == dkv:
        return int(weight.shape[0])
    raise ValueError(f"weight must have one dimension equal to dkv={dkv}")


def inkling_attn_prologue_verify(
    qkvr: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_indices: torch.Tensor,
    cache_mask: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    k_inter: torch.Tensor,
    v_inter: torch.Tensor,
    q_gamma: torch.Tensor,
    k_gamma: torch.Tensor,
    eps: float,
    loc: torch.Tensor,
    k_buf: torch.Tensor,
    v_buf: torch.Tensor,
    q_off: int,
    k_off: int,
    v_off: int,
    dq: int,
    dkv: int,
    draft_token_num: int,
    activation: Optional[str] = None,
    use_residual: bool = True,
    do_store: bool = True,
    mxfp8_quant: bool = False,
    sfk: Optional[torch.Tensor] = None,
    sfv: Optional[torch.Tensor] = None,
    page_size: int = 128,
    log_tau: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    del sfk, sfv, page_size
    _reject_mxfp8_or_tau(mxfp8_quant=mxfp8_quant, log_tau=log_tau)
    _ensure_ops_registered()
    q_out, k_out, v_out = torch.ops.sgl_kernel.inkling_attn_prologue_verify(
        qkvr,
        k_cache,
        v_cache,
        _as_int32(cache_indices),
        cache_mask.reshape(-1),
        k_weight,
        v_weight,
        k_inter,
        v_inter,
        q_gamma,
        k_gamma,
        float(eps),
        _as_int64(loc),
        _kv_view(k_buf, int(dkv)),
        _kv_view(v_buf, int(dkv)),
        int(q_off),
        int(k_off),
        int(v_off),
        int(dq),
        int(dkv),
        int(draft_token_num),
        _activation_is_silu(activation),
        bool(use_residual),
        bool(do_store),
    )
    return q_out, k_out, v_out, None


def inkling_attn_prologue_decode(
    qkvr: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_indices: torch.Tensor,
    cache_mask: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    q_gamma: torch.Tensor,
    k_gamma: torch.Tensor,
    eps: float,
    loc: torch.Tensor,
    k_buf: torch.Tensor,
    v_buf: torch.Tensor,
    q_off: int,
    k_off: int,
    v_off: int,
    dq: int,
    dkv: int,
    activation: Optional[str] = None,
    use_residual: bool = True,
    track_mask: Optional[torch.Tensor] = None,
    track_indices: Optional[torch.Tensor] = None,
    do_store: bool = True,
    mxfp8_quant: bool = False,
    sfk: Optional[torch.Tensor] = None,
    sfv: Optional[torch.Tensor] = None,
    page_size: int = 128,
    log_tau: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    del sfk, sfv, page_size
    _reject_mxfp8_or_tau(mxfp8_quant=mxfp8_quant, log_tau=log_tau)
    _ensure_ops_registered()
    q_out, k_out, v_out = torch.ops.sgl_kernel.inkling_attn_prologue_decode(
        qkvr,
        k_cache,
        v_cache,
        _as_int32(cache_indices),
        cache_mask.reshape(-1),
        k_weight,
        v_weight,
        track_mask.reshape(-1) if track_mask is not None else None,
        _as_int64(track_indices) if track_indices is not None else None,
        q_gamma,
        k_gamma,
        float(eps),
        _as_int64(loc),
        _kv_view(k_buf, int(dkv)),
        _kv_view(v_buf, int(dkv)),
        int(q_off),
        int(k_off),
        int(v_off),
        int(dq),
        int(dkv),
        _activation_is_silu(activation),
        bool(use_residual),
        bool(do_store),
    )
    return q_out, k_out, v_out, None


def inkling_attn_prologue_extend(
    qkvr: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_indices: torch.Tensor,
    cache_mask: torch.Tensor,
    has_initial_state: torch.Tensor,
    cu: torch.Tensor,
    si: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    track_rows: Optional[torch.Tensor],
    track_mask: Optional[torch.Tensor],
    track_dst: Optional[torch.Tensor],
    q_gamma: torch.Tensor,
    k_gamma: torch.Tensor,
    eps: float,
    loc: torch.Tensor,
    k_buf: torch.Tensor,
    v_buf: torch.Tensor,
    q_off: int,
    k_off: int,
    v_off: int,
    dq: int,
    dkv: int,
    activation: Optional[str] = None,
    use_residual: bool = True,
    do_store: bool = True,
    mxfp8_quant: bool = False,
    sfk: Optional[torch.Tensor] = None,
    sfv: Optional[torch.Tensor] = None,
    page_size: int = 128,
    log_tau: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    del sfk, sfv, page_size
    _reject_mxfp8_or_tau(mxfp8_quant=mxfp8_quant, log_tau=log_tau)
    _ensure_ops_registered()
    q_out, k_out, v_out = torch.ops.sgl_kernel.inkling_attn_prologue_extend(
        qkvr,
        k_cache,
        v_cache,
        _as_int32(cache_indices),
        cache_mask.reshape(-1),
        has_initial_state.reshape(-1),
        _as_int64(cu),
        _as_int32(si),
        k_weight,
        v_weight,
        _as_int64(track_rows) if track_rows is not None else None,
        track_mask.reshape(-1) if track_mask is not None else None,
        _as_int64(track_dst) if track_dst is not None else None,
        q_gamma,
        k_gamma,
        float(eps),
        _as_int64(loc),
        _kv_view(k_buf, int(dkv)),
        _kv_view(v_buf, int(dkv)),
        int(q_off),
        int(k_off),
        int(v_off),
        int(dq),
        int(dkv),
        _activation_is_silu(activation),
        bool(use_residual),
        bool(do_store),
    )
    return q_out, k_out, v_out, None


def compile_inkling_attn_prologue(
    dtype: torch.dtype,
    w: int,
    use_silu: bool,
    use_residual: bool,
    use_mxfp8: bool = False,
) -> None:
    del dtype, w, use_silu, use_residual
    if use_mxfp8:
        raise NotImplementedError(
            "Inkling XPU attention prologue MXFP8 store is not wired yet"
        )
    _ensure_ops_registered()
