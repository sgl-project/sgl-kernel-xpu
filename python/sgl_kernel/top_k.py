from typing import Optional

import torch


def fast_topk_v2(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Top-k indices of `score`. See sglang fast_topk_v2 docstring."""
    assert topk == 2048, "fast_topk_v2 only supports topk=2048 (DeepSeek V3.2 indexer)"
    assert score.dim() == 2
    topk_indices = score.new_empty((score.size(0), topk), dtype=torch.int32)
    torch.ops.sgl_kernel.fast_topk.default(score, topk_indices, lengths, row_starts)
    return topk_indices


def fast_topk_transform_fused(
    score: torch.Tensor,
    lengths: torch.Tensor,
    page_table_size_1: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Top-k + paged gather. See sglang fast_topk_transform_fused docstring."""
    assert topk == 2048, "fast_topk_transform_fused only supports topk=2048"
    assert score.dim() == 2
    dst_page_table = score.new_empty((score.shape[0], topk), dtype=torch.int32)
    torch.ops.sgl_kernel.fast_topk_transform_fused.default(
        score, lengths, dst_page_table, page_table_size_1, cu_seqlens_q, row_starts
    )
    return dst_page_table


def fast_topk_transform_ragged_fused(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk_indices_offset: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Top-k + ragged offset. See sglang fast_topk_transform_ragged_fused docstring."""
    assert topk == 2048, "fast_topk_transform_ragged_fused only supports topk=2048"
    assert score.dim() == 2
    topk_indices_ragged = score.new_empty((score.shape[0], topk), dtype=torch.int32)
    torch.ops.sgl_kernel.fast_topk_transform_ragged_fused.default(
        score, lengths, topk_indices_ragged, topk_indices_offset, row_starts
    )
    return topk_indices_ragged
