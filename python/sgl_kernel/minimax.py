# Shape/dtype validation lives in src/sycl/MinimaxDecodeTopK.cpp so the ops stay safe
# to call directly through torch.ops; these wrappers only allocate outputs.

from typing import Optional, Tuple

import torch


def minimax_decode_topk(
    score: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    topk: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Select the top-k highest-scoring block ids per (head, batch) row.

    Args:
        score: [num_heads, batch, max_seqblock] float32 block scores.
        seq_lens: [batch] int32/int64 sequence lengths, in tokens.
        block_size: tokens per score block.
        topk: blocks to select per row; at most 32.
        out: optional [num_heads, batch, topk] int32 destination.

    Returns:
        out, where out[h, b, 0:k_eff) holds the selected block ids
        (front-packed, unordered) and out[h, b, k_eff:topk) is -1, with
        k_eff = min(topk, num_blocks for batch b).

    Tie-breaking among exactly-equal scores is unspecified and can vary between
    runs, so compare selections as sets rather than element-wise.
    """
    if out is None:
        num_heads, batch, _ = score.shape
        out = torch.empty(
            (num_heads, batch, topk), dtype=torch.int32, device=score.device
        )
    torch.ops.sgl_kernel.minimax_decode_topk.default(
        score,
        seq_lens,
        out,
        block_size,
        topk,
    )
    return out


def minimax_decode_topk_page_table(
    score: torch.Tensor,
    seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    block_size: int,
    topk: int,
    page_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused top-k selection + paged page-table emission for the dense backend.

    Args:
        score: [num_kv_heads, batch, max_seqblock] float32 block scores.
        seq_lens: [batch] int32/int64 sequence lengths, in tokens.
        req_to_token: [max_reqs, max_kv_len] int32 token pool; inner stride must
            be 1, but a row-pitched slice of a larger pool is fine.
        slot_ids: [batch] int64 request pool indices.
        block_size: tokens per score block; must be a multiple of page_size.
        topk: blocks to select per row; at most 32.
        page_size: tokens per page.

    Returns:
        page_table: [batch * num_kv_heads, topk * block_size / page_size] int32.
            Row b * num_heads + h holds the paged addresses for pseudo-request
            (b, h), selected blocks sorted ascending and each expanded to
            ppb = block_size / page_size pages via req_to_token; page indices are
            head-encoded as base_page * num_heads + h.
        real_seq_lens: [batch * num_kv_heads] int32 effective KV length per
            pseudo-request (only the final selected block can be partial).

    slot_ids is wrapped modulo req_to_token.shape[0] in the kernel, so out-of-range
    or negative values cannot read out of bounds.
    """
    return torch.ops.sgl_kernel.minimax_decode_topk_page_table.default(
        score,
        seq_lens,
        req_to_token,
        slot_ids,
        block_size,
        topk,
        page_size,
    )
