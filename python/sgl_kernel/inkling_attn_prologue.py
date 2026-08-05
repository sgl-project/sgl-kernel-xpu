from __future__ import annotations

from typing import Optional

import torch


def _activation_is_silu(activation: Optional[str]) -> bool:
    if activation == "swish":
        activation = "silu"
    if activation not in (None, "silu"):
        raise NotImplementedError("activation must be None, silu, or swish")
    return activation == "silu"


def _require_dtype(x: torch.Tensor, dtype: torch.dtype, name: str) -> torch.Tensor:
    if x.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {x.dtype}")
    return x


def _resolve_log_scaling_tau(
    log_scaling_tau: Optional[torch.Tensor],
    log_tau: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if log_scaling_tau is not None and log_tau is not None:
        raise ValueError("pass only one of log_scaling_tau or log_tau")
    return log_scaling_tau if log_scaling_tau is not None else log_tau


def _log_tau_arg(
    log_scaling_tau: Optional[torch.Tensor], device: torch.device
) -> torch.Tensor:
    if log_scaling_tau is None:
        return torch.empty(0, dtype=torch.float32, device=device)
    return log_scaling_tau.reshape(-1).float()


def _kv_view(buf: torch.Tensor, dkv: int) -> torch.Tensor:
    if buf.dim() == 2 and buf.shape[1] == dkv:
        return buf
    return buf.view(-1, dkv)


def _prepare_mxfp8_args(
    k_buf: torch.Tensor,
    v_buf: torch.Tensor,
    sfk: Optional[torch.Tensor],
    sfv: Optional[torch.Tensor],
    *,
    dkv: int,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not hasattr(torch, "float8_e4m3fn") or not hasattr(torch, "float8_e8m0fnu"):
        raise RuntimeError("MXFP8 attention prologue requires PyTorch float8 dtypes")
    if dkv % 128 != 0:
        raise ValueError("MXFP8 fused prologue requires head_dim-aligned K/V")
    if page_size <= 0 or page_size % 32 != 0:
        raise ValueError("MXFP8 page_size must be a positive multiple of 32")
    if sfk is None or sfv is None:
        raise ValueError("MXFP8 fused prologue requires K/V scale buffers")

    k_view = _kv_view(k_buf, int(dkv))
    v_view = _kv_view(v_buf, int(dkv))
    if k_view.dtype != torch.float8_e4m3fn or v_view.dtype != torch.float8_e4m3fn:
        raise ValueError("MXFP8 k_buf/v_buf must be torch.float8_e4m3fn")
    sf_shape = (k_view.shape[0] // page_size, dkv // 128, 32, page_size // 32, 4)
    if k_view.shape[0] % page_size != 0:
        raise ValueError("MXFP8 k_buf slots must be divisible by page_size")
    if tuple(sfk.shape) != sf_shape or tuple(sfv.shape) != sf_shape:
        raise ValueError(
            "MXFP8 fused prologue requires interleaved K/V scale buffers "
            f"with shape {sf_shape}, got {tuple(sfk.shape)} and {tuple(sfv.shape)}."
        )
    if not sfk.is_contiguous() or not sfv.is_contiguous():
        raise ValueError("MXFP8 fused prologue requires contiguous interleaved SFK/SFV")
    return k_view, v_view, sfk.view(torch.uint8), sfv.view(torch.uint8)


def _view_mxfp8_outputs(
    q_out_u8: torch.Tensor, q_scale_u8: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return q_out_u8.view(torch.float8_e4m3fn), q_scale_u8.view(torch.float8_e8m0fnu)


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
    log_scaling_tau: Optional[torch.Tensor] = None,
    log_tau: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    log_scaling_tau = _resolve_log_scaling_tau(log_scaling_tau, log_tau)
    log_tau_tensor = _log_tau_arg(log_scaling_tau, qkvr.device)
    if mxfp8_quant:
        k_view, v_view, sfk_u8, sfv_u8 = _prepare_mxfp8_args(
            k_buf, v_buf, sfk, sfv, dkv=int(dkv), page_size=int(page_size)
        )
        q_out_u8, k_out, v_out, q_scale_u8 = (
            torch.ops.sgl_kernel.inkling_attn_prologue_verify_mxfp8(
                qkvr,
                k_cache,
                v_cache,
                _require_dtype(cache_indices, torch.int32, "cache_indices"),
                cache_mask.reshape(-1),
                k_weight,
                v_weight,
                k_inter,
                v_inter,
                q_gamma,
                k_gamma,
                float(eps),
                _require_dtype(loc, torch.int64, "loc"),
                k_view,
                v_view,
                sfk_u8,
                sfv_u8,
                int(q_off),
                int(k_off),
                int(v_off),
                int(dq),
                int(dkv),
                int(draft_token_num),
                _activation_is_silu(activation),
                bool(use_residual),
                bool(do_store),
                int(page_size),
                log_tau_tensor,
            )
        )
        q_out, q_scale = _view_mxfp8_outputs(q_out_u8, q_scale_u8)
        return q_out, k_out, v_out, q_scale
    q_out, k_out, v_out = torch.ops.sgl_kernel.inkling_attn_prologue_verify(
        qkvr,
        k_cache,
        v_cache,
        _require_dtype(cache_indices, torch.int32, "cache_indices"),
        cache_mask.reshape(-1),
        k_weight,
        v_weight,
        k_inter,
        v_inter,
        q_gamma,
        k_gamma,
        float(eps),
        _require_dtype(loc, torch.int64, "loc"),
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
        log_tau_tensor,
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
    log_scaling_tau: Optional[torch.Tensor] = None,
    log_tau: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    log_scaling_tau = _resolve_log_scaling_tau(log_scaling_tau, log_tau)
    log_tau_tensor = _log_tau_arg(log_scaling_tau, qkvr.device)
    if mxfp8_quant:
        k_view, v_view, sfk_u8, sfv_u8 = _prepare_mxfp8_args(
            k_buf, v_buf, sfk, sfv, dkv=int(dkv), page_size=int(page_size)
        )
        q_out_u8, k_out, v_out, q_scale_u8 = (
            torch.ops.sgl_kernel.inkling_attn_prologue_decode_mxfp8(
                qkvr,
                k_cache,
                v_cache,
                _require_dtype(cache_indices, torch.int32, "cache_indices"),
                cache_mask.reshape(-1),
                k_weight,
                v_weight,
                track_mask.reshape(-1) if track_mask is not None else None,
                _require_dtype(track_indices, torch.int64, "track_indices")
                if track_indices is not None
                else None,
                q_gamma,
                k_gamma,
                float(eps),
                _require_dtype(loc, torch.int64, "loc"),
                k_view,
                v_view,
                sfk_u8,
                sfv_u8,
                int(q_off),
                int(k_off),
                int(v_off),
                int(dq),
                int(dkv),
                _activation_is_silu(activation),
                bool(use_residual),
                bool(do_store),
                int(page_size),
                log_tau_tensor,
            )
        )
        q_out, q_scale = _view_mxfp8_outputs(q_out_u8, q_scale_u8)
        return q_out, k_out, v_out, q_scale
    q_out, k_out, v_out = torch.ops.sgl_kernel.inkling_attn_prologue_decode(
        qkvr,
        k_cache,
        v_cache,
        _require_dtype(cache_indices, torch.int32, "cache_indices"),
        cache_mask.reshape(-1),
        k_weight,
        v_weight,
        track_mask.reshape(-1) if track_mask is not None else None,
        _require_dtype(track_indices, torch.int64, "track_indices")
        if track_indices is not None
        else None,
        q_gamma,
        k_gamma,
        float(eps),
        _require_dtype(loc, torch.int64, "loc"),
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
        log_tau_tensor,
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
    do_cache_update: bool = True,
    log_scaling_tau: Optional[torch.Tensor] = None,
    log_tau: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    log_scaling_tau = _resolve_log_scaling_tau(log_scaling_tau, log_tau)
    log_tau_tensor = _log_tau_arg(log_scaling_tau, qkvr.device)
    if mxfp8_quant:
        k_view, v_view, sfk_u8, sfv_u8 = _prepare_mxfp8_args(
            k_buf, v_buf, sfk, sfv, dkv=int(dkv), page_size=int(page_size)
        )
        q_out_u8, k_out, v_out, q_scale_u8 = (
            torch.ops.sgl_kernel.inkling_attn_prologue_extend_mxfp8(
                qkvr,
                k_cache,
                v_cache,
                _require_dtype(cache_indices, torch.int32, "cache_indices"),
                cache_mask.reshape(-1),
                has_initial_state.reshape(-1),
                _require_dtype(cu, torch.int64, "cu"),
                _require_dtype(si, torch.int32, "si"),
                k_weight,
                v_weight,
                _require_dtype(track_rows, torch.int64, "track_rows")
                if track_rows is not None
                else None,
                track_mask.reshape(-1) if track_mask is not None else None,
                _require_dtype(track_dst, torch.int64, "track_dst")
                if track_dst is not None
                else None,
                q_gamma,
                k_gamma,
                float(eps),
                _require_dtype(loc, torch.int64, "loc"),
                k_view,
                v_view,
                sfk_u8,
                sfv_u8,
                int(q_off),
                int(k_off),
                int(v_off),
                int(dq),
                int(dkv),
                _activation_is_silu(activation),
                bool(use_residual),
                bool(do_store),
                bool(do_cache_update),
                int(page_size),
                log_tau_tensor,
            )
        )
        q_out, q_scale = _view_mxfp8_outputs(q_out_u8, q_scale_u8)
        return q_out, k_out, v_out, q_scale
    q_out, k_out, v_out = torch.ops.sgl_kernel.inkling_attn_prologue_extend(
        qkvr,
        k_cache,
        v_cache,
        _require_dtype(cache_indices, torch.int32, "cache_indices"),
        cache_mask.reshape(-1),
        has_initial_state.reshape(-1),
        _require_dtype(cu, torch.int64, "cu"),
        _require_dtype(si, torch.int32, "si"),
        k_weight,
        v_weight,
        _require_dtype(track_rows, torch.int64, "track_rows")
        if track_rows is not None
        else None,
        track_mask.reshape(-1) if track_mask is not None else None,
        _require_dtype(track_dst, torch.int64, "track_dst")
        if track_dst is not None
        else None,
        q_gamma,
        k_gamma,
        float(eps),
        _require_dtype(loc, torch.int64, "loc"),
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
        bool(do_cache_update),
        log_tau_tensor,
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
    del use_mxfp8
