import torch
import utils

"""Common utilities for calculating rope reference"""
DEVICE = utils.get_device()
DTYPE = torch.bfloat16
MAX_SEQ_LEN = 131072  # common seq length
ROPE_BASE = 10000.0
CACHE_SIZE = 1024 * 128


def create_cos_sin_cache(
    rotary_dim: int,
    max_position: int = MAX_SEQ_LEN,
    base: float = ROPE_BASE,
) -> torch.Tensor:
    """Create cos/sin cache compatible with SGLang layout: [max_pos, rotary_dim]."""
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=DEVICE)
            / rotary_dim
        )
    )
    t = torch.arange(max_position, dtype=torch.float32, device=DEVICE)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    cache = torch.cat((cos, sin), dim=-1)  # [max_pos, rotary_dim]
    return cache


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    """
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    """
    # Match kernel behavior: use fp32 cos/sin and fp32 arithmetic,
    # then cast back to the input dtype on store.
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    x_fp32 = x.float()
    if is_neox_style:
        x1, x2 = torch.chunk(x_fp32, 2, dim=-1)
    else:
        x1 = x_fp32[..., ::2]
        x2 = x_fp32[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1).to(x.dtype)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2).to(x.dtype)
