"""
XPU/SYCL fused sigmoid top-k MoE gate kernel wrapper.

JIT-compiled port of the AOT sgl_kernel.topk_sigmoid op
(src/sycl/TopKSigMoid.cpp). Fuses sigmoid + top-k selection + optional
bias-aware ranking, renormalization, routed scaling, and a fused shared expert.
One compiled .so per dtype serves every expert count (num_experts is a runtime
arg), mirroring the CUDA JIT moe_topk_sigmoid design.
"""

from __future__ import annotations

from typing import Optional

import torch

from .compiler import load_jit_sycl
from .utils import cache_once

_SUPPORTED_TOPK_SIGMOID_DTYPES = {
    torch.float32: "fp32",
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
}

# AOT caps at 256 experts (n_experts <= 256) and topk <= min(num_experts, 8).
_MAX_NUM_EXPERTS = 256
_MAX_TOPK = 8


@cache_once
def _jit_topk_sigmoid_module_xpu(dtype: torch.dtype):
    """Compile/load the XPU/SYCL topk_sigmoid module for the given dtype."""
    if dtype not in _SUPPORTED_TOPK_SIGMOID_DTYPES:
        raise ValueError(
            f"Unsupported dtype for XPU topk_sigmoid: {dtype}. "
            f"Supported: {list(_SUPPORTED_TOPK_SIGMOID_DTYPES)}"
        )

    dtype_str = _SUPPORTED_TOPK_SIGMOID_DTYPES[dtype]

    module = load_jit_sycl(
        "moe_topk_sigmoid",
        dtype_str,
        sycl_files=["moe/moe_topk_sigmoid.hpp"],
        extra_sycl_cflags=[f"-DSGL_TOPK_SIGMOID_DTYPE_{dtype_str}"],
    )
    return _XPUTopkSigmoidWrapper(module, dtype_str)


class _XPUTopkSigmoidWrapper:
    def __init__(self, module, dtype_str: str):
        import ctypes

        self._module = module
        self._func_name = f"topk_sigmoid_forward_{dtype_str}"
        self._argtypes = [
            ctypes.c_void_p,  # queue
            ctypes.c_void_p,  # gating_output
            ctypes.c_void_p,  # topk_weights (float32)
            ctypes.c_void_p,  # topk_ids (int32)
            ctypes.c_void_p,  # correction_bias (float32 or null)
            ctypes.c_int64,  # num_tokens
            ctypes.c_int64,  # num_experts
            ctypes.c_int64,  # topk
            ctypes.c_int32,  # renormalize
            ctypes.c_float,  # routed_scaling_factor
            ctypes.c_int64,  # num_fused_shared_experts
        ]

    def run(
        self,
        gating_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        correction_bias: Optional[torch.Tensor],
        renormalize: bool,
        routed_scaling_factor: float,
        num_fused_shared_experts: int,
    ) -> None:
        if not gating_output.is_contiguous():
            raise ValueError("XPU topk_sigmoid requires contiguous gating_output")
        if not topk_weights.is_contiguous() or not topk_ids.is_contiguous():
            raise ValueError("XPU topk_sigmoid requires contiguous outputs")

        num_tokens, num_experts = gating_output.shape
        topk = topk_weights.shape[1]

        bias_ptr = 0
        if correction_bias is not None:
            if not correction_bias.is_contiguous():
                raise ValueError("correction_bias must be contiguous")
            bias_ptr = correction_bias.data_ptr()

        queue = torch.xpu.current_stream().sycl_queue
        func = self._module.get_function(self._func_name, self._argtypes)
        func(
            queue,
            gating_output.data_ptr(),
            topk_weights.data_ptr(),
            topk_ids.data_ptr(),
            bias_ptr,
            num_tokens,
            num_experts,
            topk,
            1 if renormalize else 0,
            float(routed_scaling_factor),
            int(num_fused_shared_experts),
        )


def topk_sigmoid(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    correction_bias: Optional[torch.Tensor] = None,
    routed_scaling_factor: float = 1.0,
    num_fused_shared_experts: int = 0,
) -> None:
    """
    Fused sigmoid top-k MoE gate (destination-passing, in-place), matching the
    AOT sgl_kernel.topk_sigmoid signature.

    Args:
        topk_weights: [num_tokens, topk] float32, written in place.
        topk_ids: [num_tokens, topk] int32, written in place.
        gating_output: [num_tokens, num_experts] fp32/fp16/bf16 router logits.
        renormalize: renormalize the selected weights to sum to 1 per row
            (scaled by routed_scaling_factor).
        correction_bias: [num_experts] float32 per-expert bias used for ranking
            only (subtracted back out of the output weights), or None.
        routed_scaling_factor: scaling applied during renormalization and to the
            fused shared expert weight.
        num_fused_shared_experts: 0 or 1; if 1, the last topk slot is a shared
            expert with index num_experts.
    """
    assert gating_output.dim() == 2, "gating_output must be 2D"
    assert topk_weights.dim() == 2 and topk_ids.dim() == 2, "outputs must be 2D"
    assert topk_weights.dtype == torch.float32, "topk_weights must be float32"
    assert topk_ids.dtype == torch.int32, "topk_ids must be int32"

    num_tokens, num_experts = gating_output.shape
    topk = topk_weights.shape[1]
    assert num_experts <= _MAX_NUM_EXPERTS, f"num_experts must be <= {_MAX_NUM_EXPERTS}"
    max_topk = min(num_experts, _MAX_TOPK)
    assert 0 < topk <= max_topk, f"topk must be in (0, {max_topk}]"
    assert 0 <= num_fused_shared_experts <= 1, "num_fused_shared_experts in [0, 1]"
    assert topk > num_fused_shared_experts, "topk must be > num_fused_shared_experts"
    assert topk_ids.shape == (num_tokens, topk), "topk_ids shape mismatch"
    if correction_bias is not None:
        assert correction_bias.shape == (num_experts,), "correction_bias shape"
        assert correction_bias.dtype == torch.float32, "correction_bias float32"

    if not (hasattr(torch, "xpu") and gating_output.device.type == "xpu"):
        raise RuntimeError("topk_sigmoid JIT kernel requires an XPU device")

    module = _jit_topk_sigmoid_module_xpu(gating_output.dtype)
    module.run(
        gating_output,
        topk_weights,
        topk_ids,
        correction_bias,
        renormalize,
        routed_scaling_factor,
        num_fused_shared_experts,
    )


__all__ = [
    "topk_sigmoid",
]
