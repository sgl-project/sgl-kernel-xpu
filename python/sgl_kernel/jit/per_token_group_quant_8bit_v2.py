"""
XPU/SYCL per-token-group 8-bit quantization (v2) kernel wrapper.

JIT-compiled port of the AOT sgl_per_token_group_quant_8bit_v2 op
(src/sycl/per_token_group_quant_8bit_v2.cpp). DeepGEMM/DeepSeek-style grouped
quantization to int8 / fp8 (e4m3fn) with optional fused SiLU-and-mul, masked MoE
layout, and UE8M0 column-major scales. One compiled .so per (in, out) dtype;
group_size and all runtime flags are arguments (mirrors the CUDA JIT design,
with shape scalars derived here in Python).
"""

from __future__ import annotations

from typing import Optional

import torch

from .compiler import load_jit_sycl
from .utils import cache_once

_SUPPORTED_INPUT_DTYPES = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
}

_SUPPORTED_OUTPUT_DTYPES = {
    torch.int8: "int8",
    torch.float8_e4m3fn: "fp8",
}

_SUPPORTED_GROUP_SIZES = (16, 32, 64, 128)


@cache_once
def _jit_ptgq_v2_module_xpu(in_dtype: torch.dtype, out_dtype: torch.dtype):
    """Compile/load the XPU/SYCL v2 module for one (in, out) dtype pair."""
    if in_dtype not in _SUPPORTED_INPUT_DTYPES:
        raise ValueError(
            f"Unsupported input dtype for XPU per_token_group_quant_8bit_v2: "
            f"{in_dtype}. Supported: {list(_SUPPORTED_INPUT_DTYPES)}"
        )
    if out_dtype not in _SUPPORTED_OUTPUT_DTYPES:
        raise ValueError(
            f"Unsupported output dtype for XPU per_token_group_quant_8bit_v2: "
            f"{out_dtype}. Supported: {list(_SUPPORTED_OUTPUT_DTYPES)}"
        )

    in_str = _SUPPORTED_INPUT_DTYPES[in_dtype]
    out_str = _SUPPORTED_OUTPUT_DTYPES[out_dtype]

    module = load_jit_sycl(
        "per_token_group_quant_8bit_v2",
        in_str,
        out_str,
        sycl_files=["gemm/per_token_group_quant_8bit_v2.hpp"],
        extra_sycl_cflags=[
            f"-DSGL_PTGQV2_IN_{in_str}",
            f"-DSGL_PTGQV2_OUT_{out_str}",
        ],
    )
    return _XPUPtgqV2Wrapper(module, in_str, out_str)


class _XPUPtgqV2Wrapper:
    def __init__(self, module, in_str: str, out_str: str):
        import ctypes

        self._module = module
        self._func_name = f"per_token_group_quant_8bit_v2_forward_{in_str}_{out_str}"
        self._argtypes = [
            ctypes.c_void_p,  # queue
            ctypes.c_void_p,  # input
            ctypes.c_void_p,  # output_q
            ctypes.c_void_p,  # output_s
            ctypes.c_void_p,  # masked_m (or null)
            ctypes.c_int64,  # group_size
            ctypes.c_float,  # eps
            ctypes.c_float,  # min_8bit
            ctypes.c_float,  # max_8bit
            ctypes.c_int32,  # scale_ue8m0
            ctypes.c_int32,  # fuse_silu_and_mul
            ctypes.c_int32,  # masked_layout
            ctypes.c_int32,  # is_column_major
            ctypes.c_int64,  # num_local_experts
            ctypes.c_int64,  # hidden_dim_num_groups
            ctypes.c_int64,  # num_groups
            ctypes.c_int64,  # scale_expert_stride
            ctypes.c_int64,  # scale_hidden_stride
            ctypes.c_int64,  # num_tokens_per_expert
        ]

    def run(self, args: dict) -> None:
        queue = torch.xpu.current_stream().sycl_queue
        func = self._module.get_function(self._func_name, self._argtypes)
        func(
            queue,
            args["input"],
            args["output_q"],
            args["output_s"],
            args["masked_m"],
            args["group_size"],
            args["eps"],
            args["min_8bit"],
            args["max_8bit"],
            args["scale_ue8m0"],
            args["fuse_silu_and_mul"],
            args["masked_layout"],
            args["is_column_major"],
            args["num_local_experts"],
            args["hidden_dim_num_groups"],
            args["num_groups"],
            args["scale_expert_stride"],
            args["scale_hidden_stride"],
            args["num_tokens_per_expert"],
        )


def per_token_group_quant_8bit_v2(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    min_8bit: float,
    max_8bit: float,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    masked_m: Optional[torch.Tensor] = None,
) -> None:
    """
    Per-token-group 8-bit quantization (v2) on Intel XPU, matching the AOT
    sgl_per_token_group_quant_8bit_v2 signature.

    Layouts (matching the AOT v2):
      vanilla:                   input (num_tokens, hidden),   output_q (num_tokens, hidden)
      fuse_silu_and_mul:         input (num_tokens, hidden*2), output_q (num_tokens, hidden)
      fuse_silu_and_mul+masked:  input (num_experts, tokens_pad, hidden*2),
                                 output_q (num_experts, tokens_pad, hidden), masked_m (num_experts,)

    output_s is 2D (row/col-major, detected from strides) or 3D for the masked
    layout; UE8M0 scales require a column-major uint32-packed layout.
    """
    assert (
        group_size in _SUPPORTED_GROUP_SIZES
    ), f"group_size must be one of {_SUPPORTED_GROUP_SIZES}, got {group_size}"
    assert (
        input.is_contiguous() and output_q.is_contiguous()
    ), "input/output_q must be contiguous"
    assert abs(eps - 1e-10) < 1e-13, "eps must be 1e-10 (matches AOT LOCAL_ABSMAX_ABS)"

    if not (hasattr(torch, "xpu") and input.device.type == "xpu"):
        raise RuntimeError(
            "per_token_group_quant_8bit_v2 JIT kernel requires an XPU device"
        )

    masked_layout = masked_m is not None
    numel = input.numel()
    num_groups = numel // group_size // (2 if fuse_silu_and_mul else 1)
    if num_groups == 0:  # empty input -> 0 grid; nothing to do
        return

    num_local_experts = input.shape[0] if masked_layout else 1
    last = output_q.dim() - 1
    is_column_major = output_s.stride(last - 1) < output_s.stride(last)
    hidden_dim_num_groups = output_q.shape[last] // group_size
    num_tokens_per_expert = output_q.shape[last - 1]
    scale_expert_stride = output_s.stride(0) if masked_layout else 0
    scale_hidden_stride = output_s.stride(last)

    module = _jit_ptgq_v2_module_xpu(input.dtype, output_q.dtype)
    module.run(
        {
            "input": input.data_ptr(),
            "output_q": output_q.data_ptr(),
            "output_s": output_s.data_ptr(),
            "masked_m": masked_m.data_ptr() if masked_layout else 0,
            "group_size": int(group_size),
            "eps": float(eps),
            "min_8bit": float(min_8bit),
            "max_8bit": float(max_8bit),
            "scale_ue8m0": 1 if scale_ue8m0 else 0,
            "fuse_silu_and_mul": 1 if fuse_silu_and_mul else 0,
            "masked_layout": 1 if masked_layout else 0,
            "is_column_major": 1 if is_column_major else 0,
            "num_local_experts": int(num_local_experts),
            "hidden_dim_num_groups": int(hidden_dim_num_groups),
            "num_groups": int(num_groups),
            "scale_expert_stride": int(scale_expert_stride),
            "scale_hidden_stride": int(scale_hidden_stride),
            "num_tokens_per_expert": int(num_tokens_per_expert),
        }
    )


__all__ = [
    "per_token_group_quant_8bit_v2",
]
