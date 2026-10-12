from typing import Optional, Union

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
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    retrieve_parent_token: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    cache_indices: Optional[torch.Tensor] = None,
    pad_slot_id: int = -1,
    **kwargs,
):
    """GDN target-verify conv1d update (speculative decoding).

    x: (batch, dim, cache_steps) -- left unchanged; a separate output is returned.
    conv_state: (num_cache_lines, dim, width-1), read-only (seeds ancestor
        history for each sequence's tree root).
    intermediate_conv_window: (scratch_rows, cache_steps, dim, width-1),
        written with every draft token's post-update window.
    retrieve_next_token / retrieve_next_sibling: (batch, cache_steps) int32,
        first-child/next-sibling encoding of the speculative tree.
    retrieve_parent_token: (batch, cache_steps) int32, written in place with
        the decoded parent index of each draft token.
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
