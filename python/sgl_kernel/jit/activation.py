"""
XPU/SYCL fused activation-and-mul kernel wrappers.

JIT-compiled SwiGLU/GeGLU activations for Intel XPU: silu_and_mul,
gelu_and_mul (erf), and gelu_tanh_and_mul. Ports the AOT TripleOps.cpp kernels;
one compiled .so per dtype serves all three activations (the activation is
selected at runtime), mirroring the CUDA JIT run_activation design.
"""

from __future__ import annotations

from typing import Optional

import torch

from .compiler import load_jit_sycl
from .utils import cache_once

_SUPPORTED_ACT_DTYPES = {
    torch.float32: "fp32",
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
}

# Must match ActivationKind in elementwise/activation.hpp.
_ACT_KIND_MAP = {
    "silu": 0,
    "gelu": 1,  # erf-based (exact) GELU
    "gelu_tanh": 2,  # tanh approximation
}


@cache_once
def _jit_activation_module_xpu(dtype: torch.dtype):
    """Compile/load the XPU/SYCL activation module for the given dtype."""
    if dtype not in _SUPPORTED_ACT_DTYPES:
        raise ValueError(
            f"Unsupported dtype for XPU activation: {dtype}. "
            f"Supported: {list(_SUPPORTED_ACT_DTYPES)}"
        )

    dtype_str = _SUPPORTED_ACT_DTYPES[dtype]

    module = load_jit_sycl(
        "activation",
        dtype_str,
        sycl_files=["elementwise/activation.hpp"],
        extra_sycl_cflags=[f"-DSGL_ACT_DTYPE_{dtype_str}"],
    )
    return _XPUActivationWrapper(module, dtype_str)


class _XPUActivationWrapper:
    def __init__(self, module, dtype_str: str):
        import ctypes

        self._module = module
        self._func_name = f"act_and_mul_forward_{dtype_str}"
        self._argtypes = [
            ctypes.c_void_p,  # queue
            ctypes.c_void_p,  # input
            ctypes.c_void_p,  # output
            ctypes.c_int64,  # num_tokens
            ctypes.c_int64,  # dim (== output hidden dim)
            ctypes.c_int32,  # act_kind
        ]

    def run(self, input: torch.Tensor, output: torch.Tensor, act_kind: int) -> None:
        if not input.is_contiguous() or not output.is_contiguous():
            raise ValueError("XPU activation requires contiguous input/output tensors")

        queue = torch.xpu.current_stream().sycl_queue

        # Flatten to 2D so multi-dim tensors are handled correctly.
        input_2d = input.view(-1, input.size(-1))
        output_2d = output.view(-1, output.size(-1))
        num_tokens = output_2d.shape[0]
        dim = output_2d.shape[1]

        func = self._module.get_function(self._func_name, self._argtypes)
        func(queue, input.data_ptr(), output.data_ptr(), num_tokens, dim, int(act_kind))


def _run_activation(
    op_name: str, input: torch.Tensor, out: Optional[torch.Tensor]
) -> torch.Tensor:
    assert op_name in _ACT_KIND_MAP, f"Unsupported activation: {op_name}"
    assert (
        input.size(-1) % 2 == 0
    ), "activation input last dim must be even (gate|up split)"

    if out is None:
        out = input.new_empty(*input.shape[:-1], input.shape[-1] // 2)

    if not (hasattr(torch, "xpu") and input.device.type == "xpu"):
        raise RuntimeError("activation JIT kernel requires an XPU device")

    module = _jit_activation_module_xpu(input.dtype)
    module.run(input, out, _ACT_KIND_MAP[op_name])
    return out


def silu_and_mul(
    input: torch.Tensor, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """out = silu(input[..., :d]) * input[..., d:], d = input.size(-1) // 2."""
    return _run_activation("silu", input, out)


def gelu_and_mul(
    input: torch.Tensor, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """out = gelu_erf(input[..., :d]) * input[..., d:], d = input.size(-1) // 2."""
    return _run_activation("gelu", input, out)


def gelu_tanh_and_mul(
    input: torch.Tensor, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """out = gelu_tanh(input[..., :d]) * input[..., d:], d = input.size(-1) // 2."""
    return _run_activation("gelu_tanh", input, out)


__all__ = [
    "silu_and_mul",
    "gelu_and_mul",
    "gelu_tanh_and_mul",
]
