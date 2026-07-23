from __future__ import annotations

import importlib
from typing import Optional

import torch

LOGITS_PAD = 264
HIDDEN = 6144
TOPK = 6
N_ROUTED = 256
N_SHARED = 2
N_TOTAL = N_ROUTED + N_SHARED
FUSED_MAX_TOKENS = 64

_fused_scratch: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def _ops_registered() -> bool:
    return hasattr(torch.ops.sgl_kernel, "inkling_moe_gate_gemv")


def _ensure_ops_registered() -> None:
    if _ops_registered():
        return
    try:
        importlib.import_module("sgl_kernel.inkling_moe_gate_ops")
    except ImportError as exc:
        raise ImportError(
            "Inkling MoE gate ops are not registered. Build/install the "
            "inkling_moe_gate_ops extension before calling sgl_kernel.inkling_moe_gate."
        ) from exc
    if not _ops_registered():
        raise RuntimeError(
            "sgl_kernel.inkling_moe_gate_ops loaded without registering Inkling MoE gate ops"
        )


def _device_key(device: torch.device) -> int:
    if device.index is not None:
        return int(device.index)
    return int(torch.xpu.current_device())


def _get_fused_scratch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = _device_key(device)
    scratch = _fused_scratch.get(key)
    if scratch is None:
        workspace = torch.empty(
            (FUSED_MAX_TOKENS, LOGITS_PAD), dtype=torch.float32, device=device
        )
        ticket = torch.zeros((1,), dtype=torch.int32, device=device)
        scratch = (workspace, ticket)
        _fused_scratch[key] = scratch
    return scratch


def ensure_inkling_moe_gate_fused_scratch(device: Optional[torch.device] = None) -> None:
    device = torch.device("xpu", torch.xpu.current_device()) if device is None else device
    _get_fused_scratch(device)


def inkling_moe_gate_topk_renorm(
    logits: torch.Tensor,
    bias: torch.Tensor,
    global_scale: torch.Tensor,
    route_scale: float,
    *,
    return_packed: bool = False,
    rows_per_workgroup: int = 0,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
    _ensure_ops_registered()
    out = torch.ops.sgl_kernel.inkling_moe_gate_topk_renorm(
        logits,
        bias,
        global_scale,
        float(route_scale),
        bool(return_packed),
        int(rows_per_workgroup),
    )
    if return_packed:
        packed, shared_w = out
        return None, None, shared_w, packed
    routed_w, indices, shared_w = out
    return routed_w, indices, shared_w, None


def inkling_moe_gate_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    experts_per_workgroup: int = 0,
    subgroup_size: int = 0,
) -> torch.Tensor:
    _ensure_ops_registered()
    return torch.ops.sgl_kernel.inkling_moe_gate_gemv(
        x,
        weight,
        int(experts_per_workgroup),
        int(subgroup_size),
    )


def inkling_moe_gate_gemv_fused(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    global_scale: torch.Tensor,
    route_scale: float,
    *,
    return_packed: bool = False,
    experts_per_workgroup: int = 0,
    subgroup_size: int = 0,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
    _ensure_ops_registered()
    if x.shape[0] > FUSED_MAX_TOKENS:
        raise ValueError(f"fused Inkling MoE gate supports at most 64 tokens: {x.shape[0]}")
    workspace, ticket = _get_fused_scratch(x.device)
    out = torch.ops.sgl_kernel.inkling_moe_gate_gemv_fused(
        x,
        weight,
        bias,
        global_scale,
        workspace,
        ticket,
        float(route_scale),
        bool(return_packed),
        int(experts_per_workgroup),
        int(subgroup_size),
    )
    if return_packed:
        packed, shared_w = out
        return None, None, shared_w, packed
    routed_w, indices, shared_w = out
    return routed_w, indices, shared_w, None
