"""
XPU/SYCL normalization kernel wrappers.

Provides JIT-compiled RMSNorm and QKNorm kernels for Intel XPU devices.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from .compiler import load_jit_sycl
from .utils import cache_once

logger = logging.getLogger(__name__)


@cache_once
def _jit_rmsnorm_module_xpu(hidden_size: int, dtype: torch.dtype):
    """XPU/SYCL version of RMSNorm JIT compilation"""
    dtype_map = {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
    }

    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype for XPU RMSNorm: {dtype}")

    dtype_str = dtype_map[dtype]

    # Supported hidden sizes (must match C API instantiations in rmsnorm.hpp)
    supported_sizes = [
        64,
        128,
        256,
        512,
        1024,
        1536,
        2048,
        2304,
        2560,
        3072,
        4096,
        5120,
        6144,
        7168,
        8192,
        12288,
        16384,
    ]
    if hidden_size not in supported_sizes:
        raise ValueError(
            f"Unsupported hidden_size for XPU RMSNorm: {hidden_size}. "
            f"Supported: {supported_sizes}"
        )

    # Load the SYCL module — compile only the requested hidden_size + dtype
    module = load_jit_sycl(
        "rmsnorm",
        str(hidden_size),
        dtype_str,
        sycl_files=["elementwise/rmsnorm.hpp"],
        extra_sycl_cflags=[
            f"-DSGL_RMSNORM_HIDDEN_SIZE={hidden_size}",
            f"-DSGL_RMSNORM_DTYPE_{dtype_str}",
        ],
    )

    class XPURMSNormWrapper:
        def __init__(self, module, hidden_size, dtype_str):
            import ctypes

            self._module = module
            self._func_name = f"rmsnorm_forward_{dtype_str}_{hidden_size}"
            self._argtypes = [
                ctypes.c_void_p,  # queue
                ctypes.c_void_p,  # input
                ctypes.c_void_p,  # weight
                ctypes.c_void_p,  # output
                ctypes.c_int64,  # num_tokens
                ctypes.c_int64,  # input_stride
                ctypes.c_int64,  # output_stride
                ctypes.c_float,  # eps
            ]

        def rmsnorm(self, input, weight, output, eps):
            if not input.is_contiguous():
                raise ValueError(
                    f"XPU RMSNorm requires contiguous input, "
                    f"got stride={input.stride()}"
                )
            if not output.is_contiguous():
                raise ValueError(
                    f"XPU RMSNorm requires contiguous output, "
                    f"got stride={output.stride()}"
                )
            if not weight.is_contiguous():
                raise ValueError("XPU RMSNorm requires contiguous weight tensor")

            queue = torch.xpu.current_stream().sycl_queue

            # Flatten to 2D to correctly handle multi-dimensional tensors
            input_2d = input.view(-1, input.size(-1))
            output_2d = output.view(-1, output.size(-1))
            num_tokens = input_2d.shape[0]
            input_stride = input_2d.stride(0)
            output_stride = output_2d.stride(0)

            func = self._module.get_function(self._func_name, self._argtypes)

            func(
                queue,
                input_2d.data_ptr(),
                weight.data_ptr(),
                output_2d.data_ptr(),
                num_tokens,
                input_stride,
                output_stride,
                eps,
            )

    return XPURMSNormWrapper(module, hidden_size, dtype_str)


@cache_once
def _jit_qknorm_module_xpu(head_dim: int, dtype: torch.dtype):
    """XPU/SYCL version of QKNorm JIT compilation"""
    dtype_map = {
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
    }

    if dtype not in dtype_map:
        raise ValueError(
            f"Unsupported dtype for XPU QKNorm: {dtype}. " "Only fp16/bf16 supported."
        )

    dtype_str = dtype_map[dtype]

    supported_head_dims = [64, 128, 256, 512, 1024]
    if head_dim not in supported_head_dims:
        raise ValueError(
            f"Unsupported head_dim for XPU QKNorm: {head_dim}. "
            f"Supported: {supported_head_dims}"
        )

    module = load_jit_sycl(
        "qknorm",
        str(head_dim),
        dtype_str,
        sycl_files=["elementwise/qknorm.hpp"],
        extra_sycl_cflags=[
            f"-DSGL_QKNORM_HEAD_DIM={head_dim}",
            f"-DSGL_QKNORM_DTYPE_{dtype_str}",
        ],
    )

    return _XPUQKNormWrapper(module, head_dim, dtype_str)


class _XPUQKNormWrapper:
    def __init__(self, module, head_dim, dtype_str):
        import ctypes

        self._module = module
        self._func_name = f"qknorm_forward_{dtype_str}_{head_dim}"
        self._argtypes = [
            ctypes.c_void_p,  # queue
            ctypes.c_void_p,  # q
            ctypes.c_void_p,  # k
            ctypes.c_void_p,  # q_weight
            ctypes.c_void_p,  # k_weight
            ctypes.c_int64,  # q_stride
            ctypes.c_int64,  # k_stride
            ctypes.c_uint32,  # num_qo_heads
            ctypes.c_uint32,  # num_kv_heads
            ctypes.c_uint32,  # num_tokens
            ctypes.c_float,  # eps
        ]

    def qknorm(self, q, k, q_weight, k_weight, eps):
        if not q.is_contiguous():
            raise ValueError(
                f"XPU QKNorm requires contiguous q, got stride={q.stride()}"
            )
        if not k.is_contiguous():
            raise ValueError(
                f"XPU QKNorm requires contiguous k, got stride={k.stride()}"
            )
        if not q_weight.is_contiguous():
            raise ValueError("XPU QKNorm requires contiguous q_weight")
        if not k_weight.is_contiguous():
            raise ValueError("XPU QKNorm requires contiguous k_weight")

        queue = torch.xpu.current_stream().sycl_queue

        num_tokens = q.shape[0]
        num_qo_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        q_stride = q.stride(0)
        k_stride = k.stride(0)

        func = self._module.get_function(self._func_name, self._argtypes)

        func(
            queue,
            q.data_ptr(),
            k.data_ptr(),
            q_weight.data_ptr(),
            k_weight.data_ptr(),
            q_stride,
            k_stride,
            num_qo_heads,
            num_kv_heads,
            num_tokens,
            eps,
        )


@torch.compiler.assume_constant_result
@cache_once
def can_use_fused_inplace_qknorm(head_dim: int, dtype: torch.dtype) -> bool:
    """Check if fused inplace QKNorm can be used for given parameters."""
    if head_dim not in [64, 128, 256, 512, 1024]:
        logger.warning(f"Unsupported head_dim={head_dim} for JIT QK-Norm kernel")
        return False
    try:
        _jit_qknorm_module_xpu(head_dim, dtype)
        return True
    except Exception as e:
        logger.warning(f"Failed to load JIT QK-Norm kernel: {e}")
        return False


def fused_inplace_qknorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    head_dim: int = 0,
) -> None:
    """
    Fused in-place QK normalization.

    Args:
        q: Query tensor of shape [num_tokens, num_qo_heads, head_dim]
        k: Key tensor of shape [num_tokens, num_kv_heads, head_dim]
        q_weight: Query weight tensor of shape [head_dim]
        k_weight: Key weight tensor of shape [head_dim]
        eps: Epsilon for numerical stability
        head_dim: Head dimension (auto-detected if 0)
    """
    head_dim = head_dim or q.size(-1)
    module = _jit_qknorm_module_xpu(head_dim, q.dtype)
    module.qknorm(q, k, q_weight, k_weight, eps)


def rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> None:
    """
    RMSNorm normalization.

    Args:
        input: Input tensor
        weight: Weight tensor of shape [hidden_size]
        out: Output tensor (defaults to input for in-place operation)
        eps: Epsilon for numerical stability
    """
    out = out if out is not None else input
    hidden_size = input.size(-1)
    module = _jit_rmsnorm_module_xpu(hidden_size, input.dtype)
    module.rmsnorm(input, weight, out, eps)


__all__ = [
    "fused_inplace_qknorm",
    "rmsnorm",
    "can_use_fused_inplace_qknorm",
]
