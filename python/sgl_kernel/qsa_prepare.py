"""Xe2 QSA preparation operators."""

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

    BF16 input, weight, and pending key state; BF16 or unscaled E4M3FN output.
    Invalid cache slots skip the state/position store but still produce Q.
    """
    if out_dtype not in (None, torch.bfloat16, torch.float8_e4m3fn):
        raise NotImplementedError(
            "Xe2 QSA preparation supports BF16 or float8_e4m3fn output only"
        )
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
    unchanged. Pending key state is BF16; cache storage can be BF16 or
    unscaled float8_e4m3fn.
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
