"""
XPU/SYCL RoPE (Rotary Position Embedding) kernel wrappers.

Provides JIT-compiled RoPE kernels for Intel XPU devices.
"""

from __future__ import annotations

import torch

from .compiler import load_jit_sycl
from .utils import cache_once

_SUPPORTED_XPU_ROPE_DTYPES = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
}
_SUPPORTED_XPU_ROPE_DIMS = [64, 80, 96, 128, 256, 512]


@cache_once
def _jit_fused_rope_base_module_xpu(rope_dim: int):
    """Compile/load the shared XPU/SYCL fused RoPE module for rope_dim."""
    if rope_dim not in _SUPPORTED_XPU_ROPE_DIMS:
        raise ValueError(
            f"Unsupported rope_dim for XPU RoPE: {rope_dim}. "
            f"Supported: {_SUPPORTED_XPU_ROPE_DIMS}"
        )

    return load_jit_sycl(
        "fused_rope",
        str(rope_dim),
        sycl_files=["elementwise/rope.hpp"],
        extra_sycl_cflags=[f"-DSGL_ROPE_DIM={rope_dim}"],
    )


@cache_once
def _jit_fused_rope_module_xpu(is_neox: bool, rope_dim: int, dtype: torch.dtype):
    """Return a cached XPUFusedRopeWrapper for the given configuration."""
    if dtype not in _SUPPORTED_XPU_ROPE_DTYPES:
        raise ValueError(
            f"Unsupported dtype for XPU RoPE: {dtype}. Only fp16/bf16 supported."
        )

    dtype_str = _SUPPORTED_XPU_ROPE_DTYPES[dtype]
    module = _jit_fused_rope_base_module_xpu(rope_dim)
    return _XPUFusedRopeWrapper(module, is_neox, rope_dim, dtype_str)


class _XPUFusedRopeWrapper:
    """Wrapper for XPU fused RoPE kernel matching CUDA API."""

    def __init__(self, module, is_neox: bool, rope_dim: int, dtype_str: str):
        import ctypes

        self._module = module
        self._rope_dim = rope_dim
        self._dtype_str = dtype_str
        self._is_neox_str = "true" if is_neox else "false"

        # Define argtypes for run_rope
        self._rope_argtypes = [
            ctypes.c_void_p,  # queue
            ctypes.c_void_p,  # q_ptr
            ctypes.c_void_p,  # k_ptr
            ctypes.c_void_p,  # cos_sin_cache_ptr
            ctypes.c_void_p,  # positions
            ctypes.c_int64,  # q_stride
            ctypes.c_int64,  # k_stride
            ctypes.c_int64,  # head_stride
            ctypes.c_uint32,  # num_qo_heads
            ctypes.c_uint32,  # num_kv_heads
            ctypes.c_uint32,  # num_tokens
        ]

        # Define argtypes for run_rope_store
        self._rope_store_argtypes = [
            ctypes.c_void_p,  # queue
            ctypes.c_void_p,  # q_ptr
            ctypes.c_void_p,  # k_ptr
            ctypes.c_void_p,  # v_ptr
            ctypes.c_void_p,  # k_cache
            ctypes.c_void_p,  # v_cache
            ctypes.c_void_p,  # cos_sin_cache_ptr
            ctypes.c_void_p,  # positions
            ctypes.c_void_p,  # out_loc
            ctypes.c_int64,  # q_stride
            ctypes.c_int64,  # k_stride
            ctypes.c_int64,  # v_stride
            ctypes.c_int64,  # head_stride
            ctypes.c_int64,  # cache_stride
            ctypes.c_uint32,  # num_qo_heads
            ctypes.c_uint32,  # num_kv_heads
            ctypes.c_uint32,  # num_tokens
        ]

    def run_rope(self, q, k, cos_sin_cache, positions):
        """Apply RoPE inplace to q and k."""
        queue = torch.xpu.current_stream().sycl_queue

        num_tokens = q.shape[0]
        num_qo_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        q_stride = q.stride(0)
        k_stride = k.stride(0)
        head_stride = q.stride(1)

        idtype_str = "i32" if positions.dtype == torch.int32 else "i64"
        func_name = (
            f"fused_rope_{self._is_neox_str}_{self._rope_dim}_"
            f"{self._dtype_str}_{idtype_str}"
        )

        func = self._module.get_function(func_name, self._rope_argtypes)

        func(
            queue,
            q.data_ptr(),
            k.data_ptr(),
            cos_sin_cache.data_ptr(),
            positions.data_ptr(),
            q_stride,
            k_stride,
            head_stride,
            num_qo_heads,
            num_kv_heads,
            num_tokens,
        )

    def run_rope_store(
        self, q, k, v, k_cache, v_cache, cos_sin_cache, positions, out_loc
    ):
        """Apply RoPE inplace to q/k and store rotated k and v to cache."""
        queue = torch.xpu.current_stream().sycl_queue

        num_tokens = q.shape[0]
        num_qo_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        q_stride = q.stride(0)
        k_stride = k.stride(0)
        v_stride = v.stride(0)
        head_stride = q.stride(1)
        cache_stride = k_cache.stride(0)

        idtype_str = "i32" if positions.dtype == torch.int32 else "i64"
        func_name = (
            f"fused_rope_store_{self._is_neox_str}_{self._rope_dim}_"
            f"{self._dtype_str}_{idtype_str}"
        )

        func = self._module.get_function(func_name, self._rope_store_argtypes)

        func(
            queue,
            q.data_ptr(),
            k.data_ptr(),
            v.data_ptr(),
            k_cache.data_ptr(),
            v_cache.data_ptr(),
            cos_sin_cache.data_ptr(),
            positions.data_ptr(),
            out_loc.data_ptr(),
            q_stride,
            k_stride,
            v_stride,
            head_stride,
            cache_stride,
            num_qo_heads,
            num_kv_heads,
            num_tokens,
        )


def apply_rope_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    is_neox: bool,
    rope_dim: int = 0,
) -> None:
    """
    Apply rotary position embedding inplace to q and k.

    Args:
        q: Query tensor of shape [num_tokens, num_qo_heads, head_dim]
        k: Key tensor of shape [num_tokens, num_kv_heads, head_dim]
        cos_sin_cache: Precomputed cos/sin cache
        positions: Position indices
        is_neox: Whether to use NeoX-style RoPE
        rope_dim: RoPE dimension (auto-detected if 0)
    """
    rope_dim = rope_dim or cos_sin_cache.size(-1)
    module = _jit_fused_rope_module_xpu(is_neox, rope_dim, q.dtype)
    module.run_rope(q, k, cos_sin_cache, positions)


def apply_rope_inplace_with_kvcache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    out_loc: torch.Tensor,
    *,
    is_neox: bool,
    rope_dim: int = 0,
) -> None:
    """
    Apply rotary position embedding inplace to q and k, and store k/v to cache.

    This is a fused operation that applies RoPE to q and k inplace, then stores
    the rotated k and original v to the KV cache.

    Args:
        q: Query tensor of shape [num_tokens, num_qo_heads, head_dim]
        k: Key tensor of shape [num_tokens, num_kv_heads, head_dim]
        v: Value tensor of shape [num_tokens, num_kv_heads, head_dim]
        k_cache: Key cache of shape [cache_size, num_kv_heads * head_dim]
        v_cache: Value cache of shape [cache_size, num_kv_heads * head_dim]
        cos_sin_cache: Precomputed cos/sin cache
        positions: Position indices
        out_loc: Cache write locations
        is_neox: Whether to use NeoX-style RoPE
        rope_dim: RoPE dimension (auto-detected if 0)
    """
    rope_dim = rope_dim or cos_sin_cache.size(-1)
    v = v.view_as(k)
    module = _jit_fused_rope_module_xpu(is_neox, rope_dim, q.dtype)
    module.run_rope_store(q, k, v, k_cache, v_cache, cos_sin_cache, positions, out_loc)


__all__ = [
    "apply_rope_inplace",
    "apply_rope_inplace_with_kvcache",
]
