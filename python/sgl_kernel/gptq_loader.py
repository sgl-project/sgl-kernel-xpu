"""Load the GPTQ side library without replacing the image's sgl_kernel."""

from __future__ import annotations

import os

import torch

_LIB = os.environ.get("SGL_GPTQ_OPS", "/opt/sgl-kernel-xpu/sgl_gptq_ops.so")
_LOADED = False


def _ensure() -> None:
    global _LOADED
    if _LOADED:
        return
    if not hasattr(torch.ops.sgl_kernel, "gptq_gemm"):
        torch.ops.load_library(_LIB)
    _LOADED = True


def gptq_entry_probe() -> None:
    _ensure()
    device_index = torch.empty((), device="xpu")
    torch.ops.sgl_kernel.gptq_entry_probe.default(device_index)


def gptq_gemm(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_gptq_qzeros: torch.Tensor,
    b_gptq_scales: torch.Tensor,
    b_g_idx: torch.Tensor,
    use_shuffle: bool,
    bit: int,
    zero_format: int = 1,
) -> torch.Tensor:
    _ensure()
    return torch.ops.sgl_kernel.gptq_gemm.default(
        a,
        b_q_weight,
        b_gptq_qzeros,
        b_gptq_scales,
        b_g_idx,
        use_shuffle,
        bit,
        zero_format,
    )
