from typing import Optional

import torch


def gdn_attention(
    core_attn_out: torch.Tensor,
    z: torch.Tensor,
    projected_states_qkvz: torch.Tensor,
    projected_states_ba: torch.Tensor,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_bias: Optional[torch.Tensor],
    activation: str,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_prefills: int,
    num_decodes: int,
    num_spec_decodes: int,
    has_initial_state: Optional[torch.Tensor],
    non_spec_query_start_loc: Optional[torch.Tensor],
    non_spec_token_indx: Optional[torch.Tensor],
    non_spec_state_indices_tensor: Optional[torch.Tensor],
    spec_query_start_loc: Optional[torch.Tensor],
    spec_token_indx: Optional[torch.Tensor],
    spec_state_indices_tensor: Optional[torch.Tensor],
    num_accepted_tokens: Optional[torch.Tensor],
    num_actual_tokens: int,
    tp_size: int,
    reorder_input: bool,
) -> None:
    """Fused Gated-DeltaNet (GDN) attention for Intel Xe2 (BMG).

    Writes results in place into ``core_attn_out`` (and ``z``), and updates
    ``conv_state`` / ``ssm_state`` in place. Mirrors the vLLM-XPU
    ``gdn_attention`` op semantics. See ``sgl_kernel`` C++ binding for the
    argument layout / shapes.
    """
    torch.ops.sgl_kernel.gdn_attention.default(
        core_attn_out,
        z,
        projected_states_qkvz,
        projected_states_ba,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        conv_state,
        ssm_state,
        conv_weights,
        conv_bias,
        activation,
        A_log,
        dt_bias,
        num_prefills,
        num_decodes,
        num_spec_decodes,
        has_initial_state,
        non_spec_query_start_loc,
        non_spec_token_indx,
        non_spec_state_indices_tensor,
        spec_query_start_loc,
        spec_token_indx,
        spec_state_indices_tensor,
        num_accepted_tokens,
        num_actual_tokens,
        tp_size,
        reorder_input,
    )
