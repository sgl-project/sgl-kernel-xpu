"""
XPU/SYCL per-tensor FP8 quantization kernel wrapper.

JIT-compiled port of the AOT sgl_per_tensor_quant_fp8 op
(src/sycl/per_tensor_quant_fp8.cpp). Quantizes a whole tensor to fp8 (e4m3fn)
with a single global scale. One compiled .so per (dtype, is_static); the static
path is specialized so no absmax kernel is compiled in, mirroring the CUDA JIT.
"""

from __future__ import annotations

import torch

from .compiler import load_jit_sycl
from .utils import cache_once

_SUPPORTED_PTQ_DTYPES = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
}


@cache_once
def _jit_per_tensor_quant_fp8_module_xpu(dtype: torch.dtype, is_static: bool):
    """Compile/load the XPU/SYCL module for one (dtype, is_static) config."""
    if dtype not in _SUPPORTED_PTQ_DTYPES:
        raise ValueError(
            f"Unsupported input dtype for XPU per_tensor_quant_fp8: {dtype}. "
            f"Supported: {list(_SUPPORTED_PTQ_DTYPES)}"
        )

    dtype_str = _SUPPORTED_PTQ_DTYPES[dtype]
    static_str = "static" if is_static else "dynamic"

    module = load_jit_sycl(
        "per_tensor_quant_fp8",
        dtype_str,
        static_str,
        sycl_files=["gemm/per_tensor_quant_fp8.hpp"],
        extra_sycl_cflags=[
            f"-DSGL_PTQ_DTYPE_{dtype_str}",
            f"-DSGL_PTQ_STATIC_{'true' if is_static else 'false'}",
        ],
    )
    return _XPUPerTensorQuantFP8Wrapper(module, dtype_str, static_str)


class _XPUPerTensorQuantFP8Wrapper:
    def __init__(self, module, dtype_str: str, static_str: str):
        import ctypes

        self._module = module
        self._func_name = f"per_tensor_quant_fp8_forward_{dtype_str}_{static_str}"
        self._argtypes = [
            ctypes.c_void_p,  # queue
            ctypes.c_void_p,  # input
            ctypes.c_void_p,  # output_q
            ctypes.c_void_p,  # output_s
            ctypes.c_int64,  # num_elements
        ]

    def run(
        self,
        input: torch.Tensor,
        output_q: torch.Tensor,
        output_s: torch.Tensor,
    ) -> None:
        if not input.is_contiguous() or not output_q.is_contiguous():
            raise ValueError(
                "XPU per_tensor_quant_fp8 requires contiguous input/output_q"
            )
        if not output_s.is_contiguous():
            raise ValueError("XPU per_tensor_quant_fp8 requires contiguous output_s")

        queue = torch.xpu.current_stream().sycl_queue
        func = self._module.get_function(self._func_name, self._argtypes)
        func(
            queue,
            input.data_ptr(),
            output_q.data_ptr(),
            output_s.data_ptr(),
            input.numel(),
        )


def per_tensor_quant_fp8(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    is_static: bool = False,
) -> None:
    """
    Per-tensor FP8 (e4m3fn) quantization on Intel XPU, matching the AOT
    sgl_per_tensor_quant_fp8 signature.

    Args:
        input: Input tensor (float16 or bfloat16), any shape (flattened).
        output_q: Preallocated fp8 (float8_e4m3fn) output, same shape as input.
        output_s: Preallocated float32 scale (scalar / 1 element). For the
            dynamic path it is computed (absmax / 448); for is_static=True it
            must already hold the scale and the absmax pass is skipped.
        is_static: If True, use the precomputed scale and skip absmax.
    """
    assert (
        input.dtype in _SUPPORTED_PTQ_DTYPES
    ), f"input must be float16/bfloat16, got {input.dtype}"
    assert output_q.dtype == torch.float8_e4m3fn, "output_q must be float8_e4m3fn"
    assert output_s.dtype == torch.float32, "output_s must be float32"

    if not (hasattr(torch, "xpu") and input.device.type == "xpu"):
        raise RuntimeError("per_tensor_quant_fp8 JIT kernel requires an XPU device")

    module = _jit_per_tensor_quant_fp8_module_xpu(input.dtype, is_static)
    module.run(input.view(-1), output_q.view(-1), output_s.view(-1))


__all__ = [
    "per_tensor_quant_fp8",
]
