"""
XPU/SYCL MoE align-block-size kernel wrapper.

JIT-compiled port of the AOT sgl_kernel.moe_align_block_size op
(src/sycl/MoEAlign.cpp). Prepares MoE routing for block-wise grouped GEMM:
pads each expert's token count to a multiple of block_size and produces the
sorted token ids, per-block expert ids, padded token total, and cumsum. One
compiled .so per topk_ids integer dtype serves every shape.
"""

from __future__ import annotations

import torch

from .compiler import load_jit_sycl
from .utils import cache_once

_SUPPORTED_MOE_ALIGN_DTYPES = {
    torch.int32: "i32",
    torch.int64: "i64",
}


@cache_once
def _jit_moe_align_module_xpu(dtype: torch.dtype):
    """Compile/load the XPU/SYCL moe_align_block_size module for topk_ids dtype."""
    if dtype not in _SUPPORTED_MOE_ALIGN_DTYPES:
        raise ValueError(
            f"Unsupported topk_ids dtype for XPU moe_align_block_size: {dtype}. "
            f"Supported: {list(_SUPPORTED_MOE_ALIGN_DTYPES)}"
        )

    dtype_str = _SUPPORTED_MOE_ALIGN_DTYPES[dtype]

    module = load_jit_sycl(
        "moe_align_block_size",
        dtype_str,
        sycl_files=["moe/moe_align_block_size.hpp"],
        extra_sycl_cflags=[f"-DSGL_MOE_ALIGN_DTYPE_{dtype_str}"],
    )
    return _XPUMoeAlignWrapper(module, dtype_str)


class _XPUMoeAlignWrapper:
    def __init__(self, module, dtype_str: str):
        import ctypes

        self._module = module
        self._func_name = f"moe_align_block_size_forward_{dtype_str}"
        self._argtypes = [
            ctypes.c_void_p,  # queue
            ctypes.c_void_p,  # topk_ids
            ctypes.c_void_p,  # sorted_token_ids
            ctypes.c_void_p,  # expert_ids
            ctypes.c_void_p,  # num_tokens_post_pad
            ctypes.c_void_p,  # cumsum_buffer
            ctypes.c_int64,  # num_experts
            ctypes.c_int64,  # block_size
            ctypes.c_int64,  # numel
            ctypes.c_int32,  # pad_sorted_token_ids
        ]

    def run(
        self,
        topk_ids: torch.Tensor,
        num_experts: int,
        block_size: int,
        sorted_token_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        num_tokens_post_pad: torch.Tensor,
        cumsum_buffer: torch.Tensor,
        pad_sorted_token_ids: bool,
    ) -> None:
        for name, t in (
            ("topk_ids", topk_ids),
            ("sorted_token_ids", sorted_token_ids),
            ("expert_ids", expert_ids),
            ("num_tokens_post_pad", num_tokens_post_pad),
            ("cumsum_buffer", cumsum_buffer),
        ):
            if not t.is_contiguous():
                raise ValueError(f"XPU moe_align_block_size requires contiguous {name}")

        queue = torch.xpu.current_stream().sycl_queue
        func = self._module.get_function(self._func_name, self._argtypes)
        func(
            queue,
            topk_ids.data_ptr(),
            sorted_token_ids.data_ptr(),
            expert_ids.data_ptr(),
            num_tokens_post_pad.data_ptr(),
            cumsum_buffer.data_ptr(),
            int(num_experts),
            int(block_size),
            topk_ids.numel(),
            1 if pad_sorted_token_ids else 0,
        )


def moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    cumsum_buffer: torch.Tensor,
    pad_sorted_token_ids: bool = False,
) -> None:
    """
    MoE align-block-size (destination-passing, in-place), matching the AOT
    sgl_kernel.moe_align_block_size signature.

    Args:
        topk_ids: [num_tokens, topk] int32/int64 selected expert per token-slot.
        num_experts: total experts (already includes the +1 offset bucket, i.e.
            actual_experts + 1) as passed by callers of the AOT op.
        block_size: grouping block size to pad each expert's token count to.
        sorted_token_ids: [max_num_tokens_padded] int32 output, token ids grouped
            by expert (padded entries set to topk_ids.numel()).
        expert_ids: [max_num_blocks] int32 output, expert index per output block.
        num_tokens_post_pad: [1] int32 output, total padded token count.
        cumsum_buffer: [num_experts + 1] int32 scratch/output, per-expert padded
            prefix sums (also the atomic offsets for the sort pass).
        pad_sorted_token_ids: if True, prefill sorted_token_ids with numel.
    """
    assert (
        topk_ids.dtype in _SUPPORTED_MOE_ALIGN_DTYPES
    ), f"topk_ids must be int32/int64, got {topk_ids.dtype}"

    if not (hasattr(torch, "xpu") and topk_ids.device.type == "xpu"):
        raise RuntimeError("moe_align_block_size JIT kernel requires an XPU device")

    module = _jit_moe_align_module_xpu(topk_ids.dtype)
    module.run(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        pad_sorted_token_ids,
    )


__all__ = [
    "moe_align_block_size",
]
