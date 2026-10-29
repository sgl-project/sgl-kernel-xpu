from typing import Optional

import torch


def causal_conv1d_fn_xpu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    conv_states: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = -1,
    **kwargs,
):
    _ = kwargs
    torch.ops.sgl_kernel.causal_conv1d_fwd(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        activation in ["silu", "swish"],
        pad_slot_id,
    )
    return x


def causal_conv1d_update_xpu(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    pad_slot_id: int = -1,
):

    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)

    torch.ops.sgl_kernel.causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        activation in ["silu", "swish"],
        cache_seqlens,
        conv_state_indices,
        pad_slot_id,
    )
    if unsqueeze:
        x = x.squeeze(-1)
    return x


def causal_conv1d_update_verify_xpu(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    intermediate_conv_window: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
    retrieve_next_token: Optional[torch.Tensor] = None,
    retrieve_next_sibling: Optional[torch.Tensor] = None,
    retrieve_parent_token: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    cache_indices: Optional[torch.Tensor] = None,
    pad_slot_id: int = -1,
    **kwargs,
):
    """GDN target-verify conv1d update (speculative decoding).

    x: (batch, dim, cache_steps) -- not modified in-place.
    conv_state: (num_cache_lines, dim, width-1), read-only (seeds ancestor
        history for each sequence's tree root).
    intermediate_conv_window: (scratch_rows, cache_steps, dim, width-1),
        written with every draft token's post-update window.
    retrieve_next_token / retrieve_next_sibling: optional first-child/next-sibling
        tree links (batch, cache_steps), int32 or int64 with matching dtype.
    retrieve_parent_token: matching tree index dtype, written with parent
        indices. Omit all three for an implicit linear draft chain.
    """
    _ = kwargs
    return torch.ops.sgl_kernel.causal_conv1d_update_verify(
        x,
        conv_state,
        weight,
        bias,
        activation in ["silu", "swish"],
        cache_indices,
        intermediate_conv_window,
        intermediate_state_indices,
        retrieve_next_token,
        retrieve_next_sibling,
        retrieve_parent_token,
        pad_slot_id,
    )


def qwen4_short_conv_state(
    state: torch.Tensor,
    state_indices: torch.Tensor,
    x: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Qwen3.8-Flash-Next PLE decode short-conv state step.

    Returns ``conv_input = cat([state[state_indices], x.unsqueeze(-1)], dim=-1)`` and advances
    ``state[state_indices]`` in place to ``conv_input[:, :, 1:]``. Slot 0 is the CUDA-graph
    padding slot: it may repeat and is never written. Other slots must be unique.

    state: [slots, C, L] bfloat16 or float16 with 1 <= L <= 16, contiguous.
    state_indices: [N] int64.
    x: [N, C] with the same dtype as state.
    out: optional [N, C, L + 1] output; must not overlap state or x.
    """
    if out is None:
        out = torch.empty(
            (x.shape[0], x.shape[1], state.shape[2] + 1),
            dtype=x.dtype,
            device=state.device,
        )
    torch.ops.sgl_kernel.qwen4_short_conv_state(out, state, state_indices, x)
    return out


def qwen4_verify_conv(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    valid: torch.Tensor,
    width: int,
    dilation: int,
    intermediate: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Qwen3.8-Flash-Next PLE short convolution for a target-verify step.

    Returns ``residual + silu(conv1d(cat([state[state_indices], x_r.T]), weight))`` per request,
    zeroed where ``valid`` is false. state is not modified.

    x, residual: [R * W, C] bfloat16, contiguous, with 1 <= W <= 32.
    weight: [C, 1, K] bfloat16 depthwise kernel with (K - 1) * dilation == L.
    state: [slots, C, L] bfloat16 or float32 (any strides).
    state_indices: [R] int32 or int64.
    valid: [R * W] bool.
    intermediate: optional [>=R, >=W, C, L] bfloat16 or float32 buffer; row [r, s] receives the
        conv window after step s (zero for invalid steps).
    """
    output = torch.empty_like(x)
    torch.ops.sgl_kernel.qwen4_verify_conv(
        output,
        intermediate,
        x,
        residual,
        weight,
        state,
        state_indices,
        valid,
        width,
        dilation,
    )
    return output
