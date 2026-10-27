import torch


def index_q_rope_pack_weights(
    input: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    head_weights: torch.Tensor,
    weight_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RoPE and single-stage FP4 packing for DSV4.1 index queries.

    Returns packed int8 payload ``[tokens * heads, 64]``, packed int32 UE8M0
    exponents ``[tokens * heads]``, and scaled float32 weights
    ``[tokens, heads]``.
    """
    return torch.ops.sgl_kernel.index_q_rope_pack_weights.default(
        input,
        freqs_cis,
        positions,
        head_weights,
        weight_scale,
    )


def index_k_norm_rope_pack_store(
    input: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    loc: torch.Tensor,
    cache: torch.Tensor,
    *,
    ratio: int,
) -> None:
    """Normalize, RoPE, single-stage quantize, and store DSV4.1 FP4 index keys.

    Slot zero is reserved and is not written. ``cache`` uses page-separated
    payload and scale regions with 68 bytes per slot.
    """
    torch.ops.sgl_kernel.index_k_norm_rope_pack_store.default(
        input, norm_weight, eps, freqs_cis, positions, loc, cache, ratio
    )
