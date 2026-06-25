# Copyright 2025 SGLang Team. All Rights Reserved.
# Thin wrapper around torch.ops.sgl_kernel.gdn_attention, the native SYCL
# Gated-Delta-Network attention kernel vendored from vllm-xpu-kernels.
#
# Shapes (see src/sycl/gdn_attn/gdn_attn_interface.cpp for exact constraints):
#   projected_states_qkvz : [total_seqlen, H_k * (2*head_k_dim + 2*head_v_dim*(H_v/H_k))]
#   projected_states_ba   : [total_seqlen, 2 * H_v]
#   conv_state            : [cache_batch, width-1, H_k*(2*head_k_dim + head_v_dim*(H_v/H_k))]
#   ssm_state             : [cache_batch, H_v, head_v_dim, head_k_dim]
#
# num_prefills > 0 activates the Xe2 chunked-GDR fast path (chunk_size=64).

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
    has_initial_state: Optional[torch.Tensor],
    non_spec_query_start_loc: torch.Tensor,
    non_spec_state_indices_tensor: torch.Tensor,
    num_actual_tokens: int,
    tp_size: int,
    # RADIX TRACK-BUFFER FIX: optional per-chunk intermediate ssm snapshot.
    inter_ssm: Optional[torch.Tensor] = None,
    inter_ssm_indices: Optional[torch.Tensor] = None,
    # RADIX TRACK-BUFFER FIX: optional aligned-boundary conv snapshot.
    inter_conv: Optional[torch.Tensor] = None,
    inter_conv_indices: Optional[torch.Tensor] = None,
) -> None:
    torch.ops.sgl_kernel.gdn_attention(
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
        has_initial_state,
        non_spec_query_start_loc,
        non_spec_state_indices_tensor,
        num_actual_tokens,
        tp_size,
        inter_ssm,
        inter_ssm_indices,
        inter_conv,
        inter_conv_indices,
    )
