"""
JIT (Just-In-Time) compilation support for XPU/SYCL kernels.

This module provides runtime compilation of SYCL kernels using the Intel icpx compiler.
It includes both the compilation infrastructure and kernel wrappers.
"""

from __future__ import annotations

import torch

# Import utilities first
from .utils import cache_once


# Check if we're on XPU
def is_xpu() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


if is_xpu():
    from .compiler import (
        SYCLModule,
        clear_module_cache,
        is_icpx_available,
        load_jit_sycl,
    )
    from .moe_align_block_size import moe_align_block_size
    from .norm import can_use_fused_inplace_qknorm, fused_inplace_qknorm, rmsnorm
    from .rope import apply_rope_inplace, apply_rope_inplace_with_kvcache
    from .timestep_embedding import timestep_embedding

    __all__ = [
        # Utilities
        "cache_once",
        # Compiler
        "SYCLModule",
        "clear_module_cache",
        "is_icpx_available",
        "load_jit_sycl",
        # Kernels
        "can_use_fused_inplace_qknorm",
        "fused_inplace_qknorm",
        "moe_align_block_size",
        "rmsnorm",
        "apply_rope_inplace",
        "apply_rope_inplace_with_kvcache",
        "timestep_embedding",
    ]
else:
    # Non-XPU environment - provide stubs
    __all__ = [
        "cache_once",
    ]
