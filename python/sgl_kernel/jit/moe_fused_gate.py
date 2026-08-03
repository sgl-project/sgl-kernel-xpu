"""
XPU/SYCL MoE fused-gate kernel wrapper.

Provides a JIT-compiled hierarchical grouped-topk expert selection kernel
(DeepSeek-V3 style) for Intel XPU devices. Mirrors the AOT *dynamic* kernel
(moe_fused_gate_kernel_dynamic in src/sycl/MoE_fused_gate.cpp): num_experts and
num_expert_group are runtime values, so one compiled .so per dtype serves any
config with num_experts / num_expert_group <= 32.
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch

from .compiler import load_jit_sycl
from .utils import cache_once

logger = logging.getLogger(__name__)

_SUPPORTED_MOE_GATE_DTYPES = {
    torch.float32: "fp32",
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
}

# Max values-per-thread supported by the kernel (== kMoeGateMaxVpt in the
# SYCL header). num_experts / num_expert_group must not exceed this.
_MOE_GATE_MAX_VPT = 32


def _validate_moe_gate_config(num_experts: int, num_expert_group: int) -> None:
    """Enforce the same constraints as the AOT dynamic kernel."""
    if num_experts & (num_experts - 1) != 0:
        raise ValueError(f"num_experts must be a power of 2, got {num_experts}")
    if num_expert_group <= 0 or num_experts % num_expert_group != 0:
        raise ValueError(
            f"num_experts ({num_experts}) must be divisible by "
            f"num_expert_group ({num_expert_group})"
        )
    vpt = num_experts // num_expert_group
    if vpt > _MOE_GATE_MAX_VPT:
        raise ValueError(
            f"num_experts / num_expert_group = {vpt} exceeds the maximum "
            f"supported ({_MOE_GATE_MAX_VPT})"
        )


@cache_once
def _jit_moe_fused_gate_module_xpu(dtype: torch.dtype):
    """Compile/load the XPU/SYCL moe_fused_gate module for the given dtype.

    The kernel is dynamic: a single .so per dtype handles every supported
    (num_experts, num_expert_group) combination, so only the dtype is part of
    the module identity / cache key.
    """
    if dtype not in _SUPPORTED_MOE_GATE_DTYPES:
        raise ValueError(
            f"Unsupported dtype for XPU moe_fused_gate: {dtype}. "
            f"Supported: {list(_SUPPORTED_MOE_GATE_DTYPES)}"
        )

    dtype_str = _SUPPORTED_MOE_GATE_DTYPES[dtype]

    # sycl_files are resolved relative to include/sgl_kernel/jit_kernel/
    module = load_jit_sycl(
        "moe_fused_gate",
        dtype_str,
        sycl_files=["moe/moe_fused_gate.hpp"],
        extra_sycl_cflags=[f"-DSGL_MOE_GATE_DTYPE_{dtype_str}"],
    )
    return _XPUMoeFusedGateWrapper(module, dtype_str)


class _XPUMoeFusedGateWrapper:
    def __init__(self, module, dtype_str: str):
        import ctypes

        self._module = module
        self._func_name = f"moe_fused_gate_forward_{dtype_str}"
        self._argtypes = [
            ctypes.c_void_p,  # queue
            ctypes.c_void_p,  # input
            ctypes.c_void_p,  # bias
            ctypes.c_void_p,  # output (float32)
            ctypes.c_void_p,  # indices (int32)
            ctypes.c_int64,  # num_rows
            ctypes.c_int64,  # num_experts
            ctypes.c_int64,  # num_expert_group
            ctypes.c_int64,  # topk_group
            ctypes.c_int64,  # topk
            ctypes.c_int64,  # num_fused_shared_experts
            ctypes.c_float,  # routed_scaling_factor
            ctypes.c_int32,  # apply_routed_scaling_factor_on_output (bool)
        ]

    def moe_fused_gate(
        self,
        input: torch.Tensor,
        bias: torch.Tensor,
        output: torch.Tensor,
        indices: torch.Tensor,
        num_expert_group: int,
        topk_group: int,
        topk: int,
        num_fused_shared_experts: int,
        routed_scaling_factor: float,
        apply_routed_scaling_factor_on_output: bool,
    ) -> None:
        # Validate the layout assumptions the SYCL kernel relies on.
        if not input.is_contiguous() or not bias.is_contiguous():
            raise ValueError("XPU moe_fused_gate requires contiguous input/bias")
        if not output.is_contiguous() or not indices.is_contiguous():
            raise ValueError("XPU moe_fused_gate requires contiguous output/indices")

        queue = torch.xpu.current_stream().sycl_queue

        func = self._module.get_function(self._func_name, self._argtypes)
        func(
            queue,
            input.data_ptr(),
            bias.data_ptr(),
            output.data_ptr(),
            indices.data_ptr(),
            input.size(0),
            input.size(1),
            int(num_expert_group),
            int(topk_group),
            int(topk),
            int(num_fused_shared_experts),
            float(routed_scaling_factor),
            1 if apply_routed_scaling_factor_on_output else 0,
        )


@cache_once
def can_use_moe_fused_gate(
    num_experts: int, num_expert_group: int, dtype: torch.dtype
) -> bool:
    """Whether the JIT moe_fused_gate kernel supports the given config."""
    try:
        _validate_moe_gate_config(num_experts, num_expert_group)
        _jit_moe_fused_gate_module_xpu(dtype)
        return True
    except Exception as e:  # pragma: no cover - depends on toolchain
        logger.warning(f"Failed to load JIT MoE fused gate kernel: {e}")
        return False


def moe_fused_gate(
    input: torch.Tensor,
    bias: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: float = 1.0,
    apply_routed_scaling_factor_on_output: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Hierarchical grouped-topk MoE gate (DeepSeek-V3 style) on Intel XPU.

    Splits ``num_experts`` into ``num_expert_group`` groups, selects the top
    ``topk_group`` groups by their top-2 sigmoid-score sum, then selects
    ``topk`` experts within the surviving groups.

    Supports any config with ``num_experts`` a power of 2, divisible by
    ``num_expert_group``, and ``num_experts / num_expert_group <= 32``.

    Args:
        input: [num_rows, num_experts] gate logits.
        bias: [num_experts] per-expert bias.
        num_expert_group: number of expert groups (== threads-per-row).
        topk_group: number of groups to keep.
        topk: total experts to select per row (incl. fused shared experts).
        num_fused_shared_experts: trailing experts replaced by shared experts.
        routed_scaling_factor: scale factor for shared-expert weights / output.
        apply_routed_scaling_factor_on_output: scale final output by the factor.

    Returns:
        (output, indices): output weights [num_rows, topk] (float32) and
        expert indices [num_rows, topk] (int32).
    """
    assert input.ndim == 2, "input must be 2D [num_rows, num_experts]"
    assert bias.ndim == 1, "bias must be 1D [num_experts]"
    assert input.size(1) == bias.size(0), "input and bias num_experts mismatch"
    assert input.dtype == bias.dtype, "input and bias must share a dtype"
    assert topk > num_fused_shared_experts, "topk must be > num_fused_shared_experts"

    num_rows, num_experts = input.shape
    _validate_moe_gate_config(num_experts, num_expert_group)
    device = input.device

    output = torch.empty(num_rows, topk, dtype=torch.float32, device=device)
    indices = torch.empty(num_rows, topk, dtype=torch.int32, device=device)

    if hasattr(torch, "xpu") and input.device.type == "xpu":
        module = _jit_moe_fused_gate_module_xpu(input.dtype)
        module.moe_fused_gate(
            input,
            bias,
            output,
            indices,
            num_expert_group,
            topk_group,
            topk,
            num_fused_shared_experts,
            routed_scaling_factor,
            apply_routed_scaling_factor_on_output,
        )
        return output, indices

    raise RuntimeError("moe_fused_gate JIT kernel requires an XPU device")


__all__ = [
    "moe_fused_gate",
    "can_use_moe_fused_gate",
]
