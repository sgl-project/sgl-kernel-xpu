"""
XPU/SYCL HiSparse KV-offload swap-in kernel wrappers.

Provides JIT-compiled SYCL ports of the CUDA HiSparse kernels used for
hierarchical sparse attention (DeepSeek DSA / V4). Two kernels are exposed:

- ``transfer_cache_dsv4_mla``: bulk-copy DSv4 C4 tokens between page-padded C4
  buffers, one set of buffers per model layer (evict / backup path).
- ``load_cache_to_device_buffer_mla`` / ``..._dsv4_mla``: per-request swap-in of
  the current top-k tokens into a small hot device buffer, maintaining an LRU
  ordering and streaming misses in from the host cache.

These mirror the API of ``sglang.jit_kernel.hisparse`` (the CUDA path).
"""

from __future__ import annotations

import ctypes

import torch

from .compiler import load_jit_sycl
from .utils import cache_once

# Block sizes for which the transfer kernel is pre-instantiated in the header's
# default (non-macro) path. Other sizes are compiled on demand via -D.
_SUPPORTED_TRANSFER_BLOCK_SIZES = (256, 512, 1024)


# ---------------------------------------------------------------------------
# transfer_cache_dsv4_mla
# ---------------------------------------------------------------------------


@cache_once
def _jit_transfer_cache_dsv4_mla_module(block_size: int):
    """Compile/load the DSv4 C4 transfer module for a given block size."""
    if block_size % 32 != 0:
        raise ValueError(f"block_size must be a multiple of 32, got {block_size}")
    return load_jit_sycl(
        "hisparse_transfer_cache_dsv4_mla",
        str(block_size),
        sycl_files=["hisparse/transfer_cache_dsv4_mla.hpp"],
        extra_sycl_cflags=[f"-DSGL_HISPARSE_BLOCK_SIZE={block_size}"],
    )


_TRANSFER_ARGTYPES = [
    ctypes.c_void_p,  # queue
    ctypes.c_void_p,  # src_caches (void**)
    ctypes.c_void_p,  # dst_caches (void**)
    ctypes.c_void_p,  # src_indices (const int64_t*)
    ctypes.c_void_p,  # dst_indices (const int64_t*)
    ctypes.c_uint32,  # num_items
    ctypes.c_uint32,  # num_layers
]


def transfer_cache_dsv4_mla(
    src_ptrs: torch.Tensor,
    dst_ptrs: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    block_size: int = 1024,
) -> None:
    """Transfer DSv4 C4 tokens between page-padded C4 buffers.

    Args:
        src_ptrs: 1-D uint64 tensor of per-layer source cache base pointers.
        dst_ptrs: 1-D uint64 tensor of per-layer destination cache base pointers.
        src_indices: 1-D int64 tensor of source token slot indices.
        dst_indices: 1-D int64 tensor of destination token slot indices.
        block_size: SYCL work-group size (multiple of 32).
    """
    assert src_ptrs.dtype == torch.uint64 and dst_ptrs.dtype == torch.uint64
    assert src_indices.dtype == torch.int64 and dst_indices.dtype == torch.int64
    assert src_ptrs.numel() == dst_ptrs.numel()
    assert src_indices.numel() == dst_indices.numel()

    num_items = src_indices.numel()
    if num_items == 0:
        return
    num_layers = src_ptrs.numel()

    module = _jit_transfer_cache_dsv4_mla_module(block_size)
    func = module.get_function(
        f"transfer_cache_dsv4_mla_{block_size}", _TRANSFER_ARGTYPES
    )
    queue = torch.xpu.current_stream().sycl_queue
    func(
        queue,
        src_ptrs.data_ptr(),
        dst_ptrs.data_ptr(),
        src_indices.data_ptr(),
        dst_indices.data_ptr(),
        num_items,
        num_layers,
    )


# ---------------------------------------------------------------------------
# load_cache_to_device_buffer
# ---------------------------------------------------------------------------

_LOAD_CACHE_ARGTYPES = [
    ctypes.c_void_p,  # queue
    ctypes.c_void_p,  # top_k_tokens (const int32_t*)
    ctypes.c_void_p,  # device_buffer_tokens (int32_t*)
    ctypes.c_void_p,  # host_cache_locs (const int64_t*)
    ctypes.c_void_p,  # device_buffer_locs (const int32_t*)
    ctypes.c_void_p,  # host_cache_k
    ctypes.c_void_p,  # host_cache_v
    ctypes.c_void_p,  # device_buffer_k
    ctypes.c_void_p,  # device_buffer_v
    ctypes.c_void_p,  # top_k_device_locs (int32_t*)
    ctypes.c_void_p,  # req_pool_indices
    ctypes.c_void_p,  # seq_lens
    ctypes.c_void_p,  # lru_slots (int16_t*)
    ctypes.c_void_p,  # num_real_reqs (const int32_t*)
    ctypes.c_int64,  # batch_size
    ctypes.c_int64,  # buffer_stride_0
    ctypes.c_int64,  # host_stride
    ctypes.c_int64,  # lru_slot_stride_0
    ctypes.c_int64,  # top_k_tokens_stride
    ctypes.c_int64,  # top_k_device_locs_stride
    ctypes.c_int64,  # page_size
    ctypes.c_int64,  # item_size_bytes
]


@cache_once
def _jit_load_cache_module(
    block_size: int,
    num_top_k: int,
    hot_buffer_size: int,
    is_mla: bool,
    is_dsv4_layout: bool,
):
    """Compile/load the swap-in module for a fixed template configuration."""
    if block_size % 32 != 0:
        raise ValueError(f"block_size must be a multiple of 32, got {block_size}")
    if hot_buffer_size < num_top_k:
        raise ValueError(
            f"hot_buffer_size ({hot_buffer_size}) must be >= num_top_k ({num_top_k})"
        )
    return load_jit_sycl(
        "hisparse_load_cache_to_device_buffer",
        str(block_size),
        str(num_top_k),
        str(hot_buffer_size),
        "mla" if is_mla else "gqa",
        "dsv4" if is_dsv4_layout else "linear",
        sycl_files=["hisparse/load_cache_to_device_buffer.hpp"],
        extra_sycl_cflags=[
            f"-DSGL_HISPARSE_BLOCK_SIZE={block_size}",
            f"-DSGL_HISPARSE_NUM_TOP_K={num_top_k}",
            f"-DSGL_HISPARSE_HOT_BUFFER_SIZE={hot_buffer_size}",
            f"-DSGL_HISPARSE_IS_MLA={1 if is_mla else 0}",
            f"-DSGL_HISPARSE_IS_DSV4={1 if is_dsv4_layout else 0}",
        ],
    )


def _dtype_suffix(t: torch.Tensor) -> str:
    return "i64" if t.dtype == torch.int64 else "i32"


def _load_cache_to_device_buffer_mla(
    *,
    is_dsv4_layout: bool,
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    lru_slots: torch.Tensor,
    item_size_bytes: int,
    num_top_k: int,
    hot_buffer_size: int,
    page_size: int,
    block_size: int,
    num_real_reqs: torch.Tensor | None,
) -> None:
    assert (
        hot_buffer_size >= num_top_k
    ), f"hot_buffer_size ({hot_buffer_size}) must be >= num_top_k ({num_top_k})"

    module = _jit_load_cache_module(
        block_size,
        num_top_k,
        hot_buffer_size,
        True,  # is_mla
        is_dsv4_layout,
    )

    if num_real_reqs is None:
        num_real_reqs = torch.tensor(
            [top_k_tokens.size(0)], dtype=torch.int32, device=top_k_tokens.device
        )

    batch_size = top_k_tokens.size(0)
    host_stride = host_cache_locs.size(1)
    buffer_stride_0 = device_buffer_tokens.stride(0)
    lru_slot_stride_0 = lru_slots.stride(0)
    top_k_tokens_stride = top_k_tokens.stride(0)
    top_k_device_locs_stride = top_k_device_locs.stride(0)

    func_name = (
        f"load_cache_to_device_buffer_"
        f"{_dtype_suffix(seq_lens)}_{_dtype_suffix(req_pool_indices)}"
    )
    func = module.get_function(func_name, _LOAD_CACHE_ARGTYPES)
    queue = torch.xpu.current_stream().sycl_queue

    func(
        queue,
        top_k_tokens.data_ptr(),
        device_buffer_tokens.data_ptr(),
        host_cache_locs.data_ptr(),
        device_buffer_locs.data_ptr(),
        host_cache.data_ptr(),
        0,  # host_cache_v (MLA: unused)
        device_buffer.data_ptr(),
        0,  # device_buffer_v (MLA: unused)
        top_k_device_locs.data_ptr(),
        req_pool_indices.data_ptr(),
        seq_lens.data_ptr(),
        lru_slots.data_ptr(),
        num_real_reqs.data_ptr(),
        batch_size,
        buffer_stride_0,
        host_stride,
        lru_slot_stride_0,
        top_k_tokens_stride,
        top_k_device_locs_stride,
        page_size,
        item_size_bytes,
    )


def load_cache_to_device_buffer_mla(
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    lru_slots: torch.Tensor,
    item_size_bytes: int,
    num_top_k: int,
    hot_buffer_size: int,
    page_size: int = 1,
    block_size: int = 256,
    num_real_reqs: torch.Tensor | None = None,
) -> None:
    """Generic MLA hisparse swap-in: device + host both linear (stride=item_size_bytes)."""
    _load_cache_to_device_buffer_mla(
        is_dsv4_layout=False,
        top_k_tokens=top_k_tokens,
        device_buffer_tokens=device_buffer_tokens,
        host_cache_locs=host_cache_locs,
        device_buffer_locs=device_buffer_locs,
        host_cache=host_cache,
        device_buffer=device_buffer,
        top_k_device_locs=top_k_device_locs,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        lru_slots=lru_slots,
        item_size_bytes=item_size_bytes,
        num_top_k=num_top_k,
        hot_buffer_size=hot_buffer_size,
        page_size=page_size,
        block_size=block_size,
        num_real_reqs=num_real_reqs,
    )


def load_cache_to_device_buffer_dsv4_mla(
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    lru_slots: torch.Tensor,
    item_size_bytes: int,
    num_top_k: int,
    hot_buffer_size: int,
    page_size: int = 1,
    block_size: int = 256,
    num_real_reqs: torch.Tensor | None = None,
) -> None:
    """DSv4 hisparse swap-in: page-padded device + page-padded host C4 layout."""
    _load_cache_to_device_buffer_mla(
        is_dsv4_layout=True,
        top_k_tokens=top_k_tokens,
        device_buffer_tokens=device_buffer_tokens,
        host_cache_locs=host_cache_locs,
        device_buffer_locs=device_buffer_locs,
        host_cache=host_cache,
        device_buffer=device_buffer,
        top_k_device_locs=top_k_device_locs,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        lru_slots=lru_slots,
        item_size_bytes=item_size_bytes,
        num_top_k=num_top_k,
        hot_buffer_size=hot_buffer_size,
        page_size=page_size,
        block_size=block_size,
        num_real_reqs=num_real_reqs,
    )


__all__ = [
    "transfer_cache_dsv4_mla",
    "load_cache_to_device_buffer_mla",
    "load_cache_to_device_buffer_dsv4_mla",
]
