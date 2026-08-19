"""
XPU/SYCL per-token-group 8-bit quantization kernel wrapper.

JIT-compiled port of the AOT sgl_per_token_group_quant_8bit op
(src/sycl/per_token_group_quant_8bit.cpp). Quantizes an input tensor in
fixed-size groups along the last dim to int8 or fp8 (e4m3fn), writing one fp32
(or UE8M0-packed) scale per group.
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch

from .compiler import load_jit_sycl
from .utils import cache_once

logger = logging.getLogger(__name__)

_SUPPORTED_INPUT_DTYPES = {
    torch.float32: "fp32",
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
}

_SUPPORTED_OUTPUT_DTYPES = {
    torch.int8: "int8",
    torch.float8_e4m3fn: "fp8",
}

# Group sizes compiled by the SYCL header (VEC_SIZE = 16 / sizeof(T) must divide
# GROUP_SIZE; all supported dtypes satisfy this for these sizes).
_SUPPORTED_GROUP_SIZES = (32, 64, 128, 256, 512)


@cache_once
def _jit_per_token_group_quant_8bit_module_xpu(
    input_dtype: torch.dtype, output_dtype: torch.dtype, group_size: int
):
    """Compile/load the XPU/SYCL module for one (in, out, group_size) config."""
    if input_dtype not in _SUPPORTED_INPUT_DTYPES:
        raise ValueError(
            f"Unsupported input dtype for XPU per_token_group_quant_8bit: "
            f"{input_dtype}. Supported: {list(_SUPPORTED_INPUT_DTYPES)}"
        )
    if output_dtype not in _SUPPORTED_OUTPUT_DTYPES:
        raise ValueError(
            f"Unsupported output dtype for XPU per_token_group_quant_8bit: "
            f"{output_dtype}. Supported: {list(_SUPPORTED_OUTPUT_DTYPES)}"
        )
    if group_size not in _SUPPORTED_GROUP_SIZES:
        raise ValueError(
            f"Unsupported group_size for XPU per_token_group_quant_8bit: "
            f"{group_size}. Supported: {_SUPPORTED_GROUP_SIZES}"
        )

    in_str = _SUPPORTED_INPUT_DTYPES[input_dtype]
    out_str = _SUPPORTED_OUTPUT_DTYPES[output_dtype]

    module = load_jit_sycl(
        "per_token_group_quant_8bit",
        in_str,
        out_str,
        str(group_size),
        sycl_files=["gemm/per_token_group_quant_8bit.hpp"],
        extra_sycl_cflags=[
            f"-DSGL_PTGQ_GROUP_SIZE={group_size}",
            f"-DSGL_PTGQ_IN_{in_str}",
            f"-DSGL_PTGQ_OUT_{out_str}",
        ],
    )
    return _XPUPerTokenGroupQuant8bitWrapper(module, in_str, out_str, group_size)


class _XPUPerTokenGroupQuant8bitWrapper:
    def __init__(self, module, in_str: str, out_str: str, group_size: int):
        import ctypes

        self._module = module
        self._func_name = (
            f"per_token_group_quant_8bit_forward_{in_str}_{out_str}_{group_size}"
        )
        self._argtypes = [
            ctypes.c_void_p,  # queue
            ctypes.c_void_p,  # input
            ctypes.c_void_p,  # output_q
            ctypes.c_void_p,  # output_s
            ctypes.c_int64,  # num_groups
            ctypes.c_int64,  # groups_per_block
            ctypes.c_float,  # eps
            ctypes.c_float,  # min_8bit
            ctypes.c_float,  # max_8bit
            ctypes.c_int32,  # scale_ue8m0
            ctypes.c_int32,  # is_column_major
            ctypes.c_int64,  # num_groups_per_row
            ctypes.c_int64,  # scale_stride
        ]

    def run(
        self,
        input: torch.Tensor,
        output_q: torch.Tensor,
        output_s: torch.Tensor,
        group_size: int,
        eps: float,
        min_8bit: float,
        max_8bit: float,
        scale_ue8m0: bool,
    ) -> None:
        if not input.is_contiguous() or not output_q.is_contiguous():
            raise ValueError(
                "XPU per_token_group_quant_8bit requires contiguous input/output_q"
            )
        assert output_s.dim() == 2, "output_s must be 2D"

        num_groups = input.numel() // group_size
        assert input.numel() % group_size == 0, "numel must be divisible by group_size"

        # groups_per_block: largest of {16,8,4,2,1} dividing num_groups.
        groups_per_block = 1
        for candidate in (16, 8, 4, 2):
            if num_groups % candidate == 0:
                groups_per_block = candidate
                break

        # Scale layout: column-major when stride(0) < stride(1) (matches AOT).
        is_column_major = output_s.stride(0) < output_s.stride(1)
        hidden_dim = input.size(-1)
        num_groups_per_row = hidden_dim // group_size
        scale_stride = output_s.stride(1)

        queue = torch.xpu.current_stream().sycl_queue
        func = self._module.get_function(self._func_name, self._argtypes)
        func(
            queue,
            input.data_ptr(),
            output_q.data_ptr(),
            output_s.data_ptr(),
            num_groups,
            groups_per_block,
            float(eps),
            float(min_8bit),
            float(max_8bit),
            1 if scale_ue8m0 else 0,
            1 if is_column_major else 0,
            num_groups_per_row,
            scale_stride,
        )


def per_token_group_quant_8bit(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    min_8bit: float,
    max_8bit: float,
    scale_ue8m0: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-token-group quantization to 8-bit (int8 or fp8 e4m3fn) on Intel XPU.

    Args:
        input: Contiguous input tensor (float32, float16, or bfloat16).
        output_q: Preallocated quantized output (int8 or float8_e4m3fn),
            same shape as input.
        output_s: Preallocated 2D scale tensor. Row-major → [num_tokens,
            groups_per_row]; column-major (stride(0) < stride(1)) is detected
            automatically. UE8M0 scales require a column-major uint32 layout.
        group_size: Quantization group size (one of 32/64/128/256/512).
        eps: Lower bound on the group absmax.
        min_8bit, max_8bit: Clamp range of the output dtype (e.g. -448/448 for
            e4m3fn, -127/127 for int8).
        scale_ue8m0: Emit UE8M0 (power-of-two exponent) scales.

    Returns:
        (output_q, output_s), the same tensors passed in (filled in place).
    """
    if not (hasattr(torch, "xpu") and input.device.type == "xpu"):
        raise RuntimeError(
            "per_token_group_quant_8bit JIT kernel requires an XPU device"
        )

    module = _jit_per_token_group_quant_8bit_module_xpu(
        input.dtype, output_q.dtype, group_size
    )
    module.run(
        input,
        output_q,
        output_s,
        group_size,
        eps,
        min_8bit,
        max_8bit,
        scale_ue8m0,
    )
    return output_q, output_s


__all__ = [
    "per_token_group_quant_8bit",
]
