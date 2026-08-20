# HiSparse hierarchical sparse KV-cache ops for Intel XPU, used by DeepSeek
# DSA / V4 hierarchical sparse attention.
# Paged C4 layout (per page of 64 tokens): 64 value slots of 576 B, then 64 scale
# slots of 8 B, then padding so each page starts on a 576-byte boundary.

from typing import Optional

import torch

# Work-group size for transfer_cache_dsv4_mla; only 256/512/1024 are compiled.
# All three measure within noise on BMG (Xe2), so this is an escape hatch rather
# than a tunable.
_DEFAULT_TRANSFER_BLOCK_SIZE = 1024

# Work-group size for the swap-in kernel: a plain runtime value, so any multiple
# of 32 works.
_DEFAULT_SWAP_IN_BLOCK_SIZE = 256


def transfer_cache_dsv4_mla(
    src_ptrs: torch.Tensor,
    dst_ptrs: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    block_size: int = _DEFAULT_TRANSFER_BLOCK_SIZE,
) -> None:
    """Transfer DSv4 C4 tokens between page-padded C4 buffers, all layers.

    Args:
        src_ptrs: 1-D uint64 tensor of per-layer source cache base pointers.
        dst_ptrs: 1-D uint64 tensor of per-layer destination cache base pointers.
        src_indices: 1-D int64 tensor of source token slot indices.
        dst_indices: 1-D int64 tensor of destination token slot indices.
        block_size: work-group size; one of 256, 512, 1024.
    """
    torch.ops.sgl_kernel.transfer_cache_dsv4_mla.default(
        src_ptrs,
        dst_ptrs,
        src_indices,
        dst_indices,
        block_size,
    )


def _load_cache_to_device_buffer_mla(
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
    num_real_reqs: Optional[torch.Tensor],
) -> None:
    torch.ops.sgl_kernel.load_cache_to_device_buffer_mla.default(
        top_k_tokens,
        device_buffer_tokens,
        host_cache_locs,
        device_buffer_locs,
        host_cache,
        device_buffer,
        top_k_device_locs,
        req_pool_indices,
        seq_lens,
        lru_slots,
        num_real_reqs,
        item_size_bytes,
        num_top_k,
        hot_buffer_size,
        page_size,
        block_size,
        is_dsv4_layout,
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
    block_size: int = _DEFAULT_SWAP_IN_BLOCK_SIZE,
    num_real_reqs: Optional[torch.Tensor] = None,
) -> None:
    """Generic MLA hisparse swap-in: device + host both linear (stride=item_size_bytes).

    Swaps each request's top-k tokens into a small hot device buffer, maintaining
    the per-request LRU order, streaming misses in from the host cache and writing
    every top-k token's device slot to ``top_k_device_locs``.

    Args:
        top_k_tokens: (batch, num_top_k) int32 top-k token positions per request.
        device_buffer_tokens: (num_reqs, hot_buffer_size + 1) int32 token position
            resident in each device-buffer slot; updated in place.
        host_cache_locs: (num_reqs, max_seq_len) int64 host cache slot per token.
        device_buffer_locs: (num_reqs, hot_buffer_size + 1) int32 device cache slot
            per buffer slot. Must share ``stride(0)`` with device_buffer_tokens.
        host_cache: host-side KV cache tensor.
        device_buffer: device-side hot KV buffer tensor.
        top_k_device_locs: (batch, num_top_k) int32 output device slots.
        req_pool_indices: (batch,) int32/int64 request-pool row per batch entry.
        seq_lens: (batch,) int32/int64 sequence lengths.
        lru_slots: (num_reqs, hot_buffer_size) int16 LRU order; updated in place.
        item_size_bytes: bytes per KV item (one token, all heads).
        num_top_k: top-k count; must be <= hot_buffer_size.
        hot_buffer_size: device buffer capacity in tokens, excluding the extra
            slot reserved for the newest token.
        page_size: accepted for API parity; unused.
        block_size: work-group size; a multiple of 32.
        num_real_reqs: (1,) int32 count of non-padded requests, for graph-captured
            padded batches. Defaults to the full batch.
    """
    _load_cache_to_device_buffer_mla(
        False,
        top_k_tokens,
        device_buffer_tokens,
        host_cache_locs,
        device_buffer_locs,
        host_cache,
        device_buffer,
        top_k_device_locs,
        req_pool_indices,
        seq_lens,
        lru_slots,
        item_size_bytes,
        num_top_k,
        hot_buffer_size,
        page_size,
        block_size,
        num_real_reqs,
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
    block_size: int = _DEFAULT_SWAP_IN_BLOCK_SIZE,
    num_real_reqs: Optional[torch.Tensor] = None,
) -> None:
    """DSv4 hisparse swap-in: page-padded device + page-padded host C4 layout.

    Same as :func:`load_cache_to_device_buffer_mla`, except the miss copy walks
    the paged C4 layout described at the top of this module, so ``host_cache`` and
    ``device_buffer`` must both be page-padded C4 buffers.
    """
    _load_cache_to_device_buffer_mla(
        True,
        top_k_tokens,
        device_buffer_tokens,
        host_cache_locs,
        device_buffer_locs,
        host_cache,
        device_buffer,
        top_k_device_locs,
        req_pool_indices,
        seq_lens,
        lru_slots,
        item_size_bytes,
        num_top_k,
        hot_buffer_size,
        page_size,
        block_size,
        num_real_reqs,
    )
