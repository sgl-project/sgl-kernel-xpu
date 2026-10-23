import torch


def index_q_rope_pack_weights(
    input: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    head_weights: torch.Tensor,
    weight_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RoPE and two-stage FP4 packing for DSV4.1 index queries.

    Returns packed int8 payload ``[tokens * heads, 64]``, packed int32 UE8M0
    exponents ``[tokens * heads]``, and scaled BF16-rounded weights
    ``[tokens, heads]`` widened to float32.
    """
    if input.ndim != 3 or input.shape[-1] != 128:
        raise ValueError(
            f"input must have shape [tokens, heads, 128], got {input.shape}"
        )
    num_tokens, num_heads, _ = input.shape
    payload = torch.empty(
        (num_tokens * num_heads, 64), dtype=torch.int8, device=input.device
    )
    scale = torch.empty(
        (num_tokens * num_heads,), dtype=torch.int32, device=input.device
    )
    weights_out = torch.empty(
        (num_tokens, num_heads), dtype=torch.float32, device=input.device
    )
    torch.ops.sgl_kernel.index_q_rope_pack_weights.default(
        input,
        freqs_cis,
        positions,
        head_weights,
        payload,
        scale,
        weights_out,
        weight_scale,
    )
    return payload, scale, weights_out


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
    """Normalize, RoPE, fake-quantize, and store DSV4.1 FP4 index keys.

    Slot zero is reserved and is not written. ``cache`` uses page-separated
    payload and scale regions with 68 bytes per slot.
    """
    if input.ndim != 2 or input.shape[-1] != 128:
        raise ValueError(f"input must have shape [tokens, 128], got {input.shape}")
    torch.ops.sgl_kernel.index_k_norm_rope_pack_store.default(
        input, norm_weight, eps, freqs_cis, positions, loc, cache, ratio
    )
