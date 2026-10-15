"""Xe2 BF16 QSA preparation operators."""

from typing import Optional

import torch


def qsa_index_q_norm_rope_store(
    qk: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    axis_map: torch.Tensor,
    weight: torch.Tensor,
    cache_loc: torch.Tensor,
    key_state_buffer: torch.Tensor,
    rope_position_buffer: torch.Tensor,
    num_q_heads: int,
    rotary_dim: int,
    eps: float,
    is_neox_style: bool,
    q_heads_padded: Optional[int] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Normalize and rotate Q; store raw K and three position axes at valid slots.

    BF16 input, weight and output only. Negative or out-of-range cache slots
    skip the state/position store without suppressing query output.
    """
    if out_dtype is not None and out_dtype != torch.bfloat16:
        raise NotImplementedError("Xe2 QSA preparation supports BF16 output only")
    return torch.ops.sgl_kernel.qsa_index_q_norm_rope_store(
        qk,
        positions,
        cos_sin_cache,
        axis_map,
        weight,
        cache_loc,
        key_state_buffer,
        rope_position_buffer,
        num_q_heads,
        rotary_dim,
        eps,
        is_neox_style,
        q_heads_padded,
        out_dtype,
    )


def qsa_index_k_compress_store(
    key_state_buffer: torch.Tensor,
    group_locs: torch.Tensor,
    rope_position_buffer: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    axis_map: torch.Tensor,
    weight: torch.Tensor,
    write_locs: torch.Tensor,
    compressed_k_buffer: torch.Tensor,
    compress_ratio: int,
    rotary_dim: int,
    eps: float,
    is_neox_style: bool,
) -> None:
    """Pool each completed group, normalize and rotate at its first slot.

    Invalid group members or output slots skip that group; other slots remain
    unchanged. BF16 storage only; FP8 cache output is not supported.
    """
    torch.ops.sgl_kernel.qsa_index_k_compress_store(
        key_state_buffer,
        group_locs,
        rope_position_buffer,
        cos_sin_cache,
        axis_map,
        weight,
        write_locs,
        compressed_k_buffer,
        compress_ratio,
        rotary_dim,
        eps,
        is_neox_style,
    )
