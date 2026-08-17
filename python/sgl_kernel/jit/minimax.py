"""XPU/SYCL MiniMax decode block top-k kernel wrapper.

Provides JIT-compiled SYCL ports of ``minimax_decode_topk`` and
``minimax_decode_topk_page_table``, matching the CUDA path at
``sglang/python/sglang/kernels/ops/attention/minimax_decode_topk.py`` (backed
by the ``minimax_decode_topk.cuh`` nvcc JIT).

Two exposed functions, both accepting XPU tensors and dispatching to the
matching per-``SeqLenT`` exported symbol at call time:

- ``minimax_decode_topk``: block-id output ``[num_heads, batch, topk]``
  (front-packed, ``-1`` padded). Drop-in for the Triton 2-stage top-k
  fallback when short-context / small-topk performance matters.

- ``minimax_decode_topk_page_table``: fused top-k + page-table transform for
  the dense backend (trtllm_mha / fa3). Returns the per-``(batch, kv_head)``
  page table plus the effective KV length; page indices head-encoded for DP
  attention (``base_page * num_heads + h``).

- ``@cache_once`` compile/load, one shared module (both kernels compiled
  together since they share ``TopKTrait::forward``).
- ``ctypes`` argtypes fixed at module scope.
- SYCL ``queue`` obtained from ``torch.xpu.current_stream().sycl_queue``.
- All host-side layout checks live in Python; the kernel assumes the
  invariants hold.
"""

from __future__ import annotations

import ctypes
from typing import Optional, Tuple

import torch

from .compiler import load_jit_sycl
from .utils import cache_once

# The header exports both kernels together (they share device helpers). One
# JIT module covers both; the wrappers pick the exported symbol by name.
_TOPK_ARGTYPES = [
    ctypes.c_void_p,  # queue
    ctypes.c_void_p,  # score (const float*)
    ctypes.c_void_p,  # seq_lens (const SeqLenT*)
    ctypes.c_void_p,  # topk_idx (int32_t*, out)
    ctypes.c_int32,  # batch
    ctypes.c_int32,  # num_heads
    ctypes.c_int32,  # max_seqblock
    ctypes.c_int32,  # block_size
    ctypes.c_int32,  # topk
]

_PAGE_TABLE_ARGTYPES = [
    ctypes.c_void_p,  # queue
    ctypes.c_void_p,  # score (const float*)
    ctypes.c_void_p,  # seq_lens (const SeqLenT*)
    ctypes.c_void_p,  # req_to_token (const int32_t*)
    ctypes.c_void_p,  # slot_ids (const int64_t*)
    ctypes.c_void_p,  # page_table (int32_t*, out)
    ctypes.c_void_p,  # seq_lens_out (int32_t*, out)
    ctypes.c_int32,  # batch
    ctypes.c_int32,  # num_heads
    ctypes.c_int32,  # max_seqblock
    ctypes.c_int32,  # block_size
    ctypes.c_int32,  # topk
    ctypes.c_int32,  # page_size
    ctypes.c_int32,  # r2t_stride
    ctypes.c_int32,  # max_kv_len
    ctypes.c_int32,  # max_reqs
    ctypes.c_int32,  # max_sparse_pages
]

# Bounds inherited from the CUDA header's TopKTrait.
_MAX_TOPK = 32
_MAX_NUM_BLOCKS = 4096


@cache_once
def _jit_minimax_module():
    """Compile/load the shared module. No compile-time template parameters --
    block_size / topk / page_size are runtime args, so a single ``.so`` covers
    all shapes."""
    return load_jit_sycl(
        "minimax_decode_topk",
        sycl_files=["minimax/minimax_decode_topk.hpp"],
    )


def _seq_lens_suffix(seq_lens: torch.Tensor) -> str:
    """Pick the ``_i32`` / ``_i64`` exported symbol suffix from ``seq_lens.dtype``."""
    if seq_lens.dtype == torch.int32:
        return "i32"
    if seq_lens.dtype == torch.int64:
        return "i64"
    raise ValueError(f"seq_lens must be int32 or int64, got {seq_lens.dtype}")


def minimax_decode_topk(
    score: torch.Tensor,  # [num_heads, batch, max_seqblock] fp32
    seq_lens: torch.Tensor,  # [batch] int32/int64
    block_size: int,
    topk: int,
    out: Optional[torch.Tensor] = None,  # [num_heads, batch, topk] int32
) -> torch.Tensor:
    """Select the top-k highest-scoring block ids per (head, batch) row.

    Output contract:
      - ``out[h, b, 0:k_eff)`` = selected block ids (front-packed, unordered).
      - ``out[h, b, k_eff:topk)`` = ``-1``.
      - ``k_eff = min(topk, num_blocks_for_batch_b)``.
      - Tie-breaking among exactly-equal scores is unspecified: which of several
        equal-scoring blocks lands in the last slot may differ from the CUDA
        kernel and between runs. Compare selections as sets, not element-wise.
    """
    if score.dtype != torch.float32:
        raise ValueError(f"score must be float32, got {score.dtype}")
    if score.dim() != 3:
        raise ValueError(f"score must be 3-D, got shape {tuple(score.shape)}")
    if seq_lens.dim() != 1:
        raise ValueError(f"seq_lens must be 1-D, got shape {tuple(seq_lens.shape)}")
    if score.device != seq_lens.device:
        raise ValueError(
            f"score and seq_lens must be on the same device "
            f"({score.device} vs {seq_lens.device})"
        )

    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    # topk < 1 would enter the radix path with topk_remain == 0, leaving
    # threshold_bin uninitialized in find_threshold.
    if topk < 1:
        raise ValueError(f"topk must be >= 1, got {topk}")
    if topk > _MAX_TOPK:
        raise ValueError(f"topk ({topk}) exceeds kMaxTopK ({_MAX_TOPK})")

    num_heads, batch, max_seqblock = score.shape
    if seq_lens.shape[0] != batch:
        raise ValueError(
            f"seq_lens length ({seq_lens.shape[0]}) must match batch ({batch})"
        )
    if max_seqblock > _MAX_NUM_BLOCKS:
        raise ValueError(
            f"max_seqblock ({max_seqblock}) exceeds kMaxNumBlocks "
            f"({_MAX_NUM_BLOCKS}); increase kMaxNumBlocks in the header if needed"
        )

    if not score.is_contiguous():
        score = score.contiguous()
    if not seq_lens.is_contiguous():
        seq_lens = seq_lens.contiguous()

    if out is None:
        out = torch.empty(
            (num_heads, batch, topk), dtype=torch.int32, device=score.device
        )
    else:
        if out.shape != (num_heads, batch, topk):
            raise ValueError(
                f"out shape {tuple(out.shape)} must equal "
                f"({num_heads}, {batch}, {topk})"
            )
        if out.dtype != torch.int32:
            raise ValueError(f"out dtype must be int32, got {out.dtype}")
        if out.device != score.device:
            raise ValueError(
                f"out device ({out.device}) must match score device ({score.device})"
            )
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")

    module = _jit_minimax_module()
    func_name = f"minimax_decode_topk_{_seq_lens_suffix(seq_lens)}"
    func = module.get_function(func_name, _TOPK_ARGTYPES)
    queue = torch.xpu.current_stream().sycl_queue

    func(
        queue,
        score.data_ptr(),
        seq_lens.data_ptr(),
        out.data_ptr(),
        int(batch),
        int(num_heads),
        int(max_seqblock),
        int(block_size),
        int(topk),
    )
    return out


def minimax_decode_topk_page_table(
    score: torch.Tensor,  # [num_kv_heads, batch, max_seqblock] fp32
    seq_lens: torch.Tensor,  # [batch] int32/int64
    req_to_token: torch.Tensor,  # [max_reqs, max_kv_len] int32
    slot_ids: torch.Tensor,  # [batch] int64 (req_pool_indices)
    block_size: int,
    topk: int,
    page_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused top-k selection + paged page-table emission for the dense backend.

    Returns:
        page_table: ``[batch * num_kv_heads, topk * block_size / page_size]``
            int32. Row ``b * num_heads + h`` holds the paged addresses for
            pseudo-request ``(b, h)``, selected blocks sorted ascending and
            each expanded to ``ppb = block_size / page_size`` pages via
            ``req_to_token``; page indices are head-encoded as
            ``base_page * num_heads + h``.
        real_seq_lens: ``[batch * num_kv_heads]`` int32. Effective KV length
            per pseudo-request (only the final selected block can be partial).

    ``slot_ids`` is wrapped modulo ``req_to_token.shape[0]`` in the kernel, as
    the Triton reference does, so out-of-range or negative values cannot read
    out of bounds.
    """
    if score.dtype != torch.float32:
        raise ValueError(f"score must be float32, got {score.dtype}")
    if score.dim() != 3:
        raise ValueError(f"score must be 3-D, got shape {tuple(score.shape)}")
    if req_to_token.dtype != torch.int32:
        raise ValueError(f"req_to_token must be int32, got {req_to_token.dtype}")
    if slot_ids.dtype != torch.int64:
        raise ValueError(f"slot_ids must be int64, got {slot_ids.dtype}")
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    if page_size < 1:
        raise ValueError(f"page_size must be >= 1, got {page_size}")
    if topk < 1:
        raise ValueError(f"topk must be >= 1, got {topk}")
    if block_size % page_size != 0:
        raise ValueError(
            f"block_size ({block_size}) must be a multiple of page_size "
            f"({page_size})"
        )
    if topk > _MAX_TOPK:
        raise ValueError(
            f"topk ({topk}) exceeds kMaxTopK ({_MAX_TOPK}) for the page-table "
            f"kernel"
        )
    if score.device != seq_lens.device:
        raise ValueError("score and seq_lens must be on the same device")
    if score.device != req_to_token.device:
        raise ValueError("score and req_to_token must be on the same device")
    if score.device != slot_ids.device:
        raise ValueError("score and slot_ids must be on the same device")

    if seq_lens.dim() != 1:
        raise ValueError(f"seq_lens must be 1-D, got shape {tuple(seq_lens.shape)}")
    if slot_ids.dim() != 1:
        raise ValueError(f"slot_ids must be 1-D, got shape {tuple(slot_ids.shape)}")
    if req_to_token.dim() != 2:
        raise ValueError(
            f"req_to_token must be 2-D, got shape {tuple(req_to_token.shape)}"
        )

    num_heads, batch, max_seqblock = score.shape
    # The kernel reads seq_lens_[b] / slot_ids_[b] for every b < batch, and
    # slot_ids_[b] feeds r2t pointer arithmetic.
    if seq_lens.shape[0] != batch:
        raise ValueError(
            f"seq_lens length ({seq_lens.shape[0]}) must match batch ({batch})"
        )
    if slot_ids.shape[0] != batch:
        raise ValueError(
            f"slot_ids length ({slot_ids.shape[0]}) must match batch ({batch})"
        )
    if max_seqblock > _MAX_NUM_BLOCKS:
        raise ValueError(
            f"max_seqblock ({max_seqblock}) exceeds kMaxNumBlocks "
            f"({_MAX_NUM_BLOCKS})"
        )

    if not score.is_contiguous():
        score = score.contiguous()
    if not seq_lens.is_contiguous():
        seq_lens = seq_lens.contiguous()
    if not slot_ids.is_contiguous():
        slot_ids = slot_ids.contiguous()

    ppb = block_size // page_size
    max_sparse_pages = topk * ppb
    max_kv_len = req_to_token.shape[1]
    max_reqs = req_to_token.shape[0]
    r2t_stride = req_to_token.stride(0)
    # The kernel addresses req_to_token as flat row-major (r2t_base + tok), so
    # the inner stride must be 1. A row-pitched slice of a larger pool tensor
    # (stride(0) > shape[1]) is fine and common, hence no full contiguity check.
    if req_to_token.stride(1) != 1:
        raise ValueError(
            f"req_to_token must have unit inner stride, got strides "
            f"{tuple(req_to_token.stride())}"
        )

    page_table = torch.empty(
        (batch * num_heads, max_sparse_pages),
        dtype=torch.int32,
        device=score.device,
    )
    real_seq_lens = torch.empty(
        (batch * num_heads,),
        dtype=torch.int32,
        device=score.device,
    )

    module = _jit_minimax_module()
    func_name = f"minimax_decode_topk_page_table_{_seq_lens_suffix(seq_lens)}"
    func = module.get_function(func_name, _PAGE_TABLE_ARGTYPES)
    queue = torch.xpu.current_stream().sycl_queue

    func(
        queue,
        score.data_ptr(),
        seq_lens.data_ptr(),
        req_to_token.data_ptr(),
        slot_ids.data_ptr(),
        page_table.data_ptr(),
        real_seq_lens.data_ptr(),
        int(batch),
        int(num_heads),
        int(max_seqblock),
        int(block_size),
        int(topk),
        int(page_size),
        int(r2t_stride),
        int(max_kv_len),
        int(max_reqs),
        int(max_sparse_pages),
    )
    return page_table, real_seq_lens


__all__ = [
    "minimax_decode_topk",
    "minimax_decode_topk_page_table",
]
