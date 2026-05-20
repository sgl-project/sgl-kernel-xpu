from typing import Optional, Tuple

import torch


def fp8_mqa_logits(
    q_fp8: torch.Tensor,
    kv_fp8: Tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    clean_logits: bool = False,
) -> torch.Tensor:
    """FP8 MQA logits for ragged (prefill/extend) path.

    Computes indexer scores: for each query, dot-product with K tokens
    in range [ks, ke), apply ReLU, weight by head gates, reduce across
    heads, and scale by per-token K scale.

    Args:
        q_fp8: (Nq, H, D) fp8 e4m3 queries (stored as uint8)
        kv_fp8: tuple of (k_fp8, k_scale) where
            k_fp8: (Nk, D) fp8 e4m3 keys (stored as uint8)
            k_scale: (Nk,) float32 per-token dequant scales
        weights: (Nq, H) float32 combined gate weights
        ks: (Nq,) int32 start indices
        ke: (Nq,) int32 end indices
        clean_logits: if True, zero out invalid positions (default: False)

    Returns:
        logits: (Nq, Nk) float32
    """
    k_fp8, k_scale = kv_fp8
    q_input = q_fp8.view(torch.uint8) if q_fp8.dtype != torch.uint8 else q_fp8
    k_input = k_fp8.view(torch.uint8) if k_fp8.dtype != torch.uint8 else k_fp8
    return torch.ops.sgl_kernel.fp8_mqa_logits.default(
        q_input, k_input, k_scale, weights, ks, ke
    )


def fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: Optional[torch.Tensor],
    max_seq_len: int,
    clean_logits: bool = False,
) -> torch.Tensor:
    """FP8 MQA logits for paged (decode) path.

    Args:
        q_fp8: (B, 1, H, D) fp8 e4m3 queries (stored as uint8)
        kv_cache: (num_pages, page_size, 1, D+4) uint8 paged KV cache
        weights: (B, H) float32 combined gate weights
        seq_lens: (B,) or (B,1) int32 actual sequence lengths
        block_tables: (B, max_num_blocks) int32 page table
        schedule_metadata: optional scheduling metadata (ignored on XPU)
        max_seq_len: maximum sequence length
        clean_logits: if True, zero out invalid positions

    Returns:
        logits: (B, max_seq_len) float32
    """
    q_input = q_fp8.view(torch.uint8) if q_fp8.dtype != torch.uint8 else q_fp8
    return torch.ops.sgl_kernel.fp8_paged_mqa_logits.default(
        q_input,
        kv_cache,
        weights,
        seq_lens,
        block_tables,
        schedule_metadata,
        max_seq_len,
        clean_logits,
    )
