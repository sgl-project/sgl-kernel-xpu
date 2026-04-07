from typing import Optional

import torch


def _prepare_chunk_indices_offsets(cu_seqlens: torch.Tensor, chunk_size: int):
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    chunk_counts = torch.div(lens + chunk_size - 1, chunk_size, rounding_mode="floor")
    chunk_offsets = torch.cat((cu_seqlens.new_zeros(1), chunk_counts)).cumsum(dim=0)
    seq_ids = torch.arange(
        chunk_counts.numel(), device=cu_seqlens.device, dtype=torch.int32
    )
    chunk_seq = torch.repeat_interleave(seq_ids, chunk_counts)
    starts = torch.repeat_interleave(chunk_offsets[:-1], chunk_counts)
    local = (
        torch.arange(chunk_seq.numel(), device=cu_seqlens.device, dtype=torch.int64)
        - starts
    )
    chunk_indices = torch.stack((chunk_seq, local.to(torch.int32)), dim=1)
    return chunk_indices, chunk_offsets.to(torch.int32)


def causal_conv1d_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias_: Optional[torch.Tensor],
    conv_states: Optional[torch.Tensor],
    query_start_loc: Optional[torch.Tensor],
    cache_indices: Optional[torch.Tensor],
    has_initial_state: Optional[torch.Tensor],
    silu_activation: bool,
    pad_slot_id: int,
) -> None:
    torch.ops.sgl_kernel.causal_conv1d_fwd(
        x,
        weight,
        bias_,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        silu_activation,
        pad_slot_id,
    )


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias_: Optional[torch.Tensor],
    silu_activation: bool,
    cache_seqlens: Optional[torch.Tensor],
    conv_state_indices: Optional[torch.Tensor],
    pad_slot_id: int,
) -> None:
    torch.ops.sgl_kernel.causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias_,
        silu_activation,
        cache_seqlens,
        conv_state_indices,
        pad_slot_id,
    )


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    head_first: bool,
    use_qk_l2norm_in_kernel: bool,
    chunk_size: int = 64,
):
    chunk_indices, chunk_offsets = _prepare_chunk_indices_offsets(
        cu_seqlens, chunk_size
    )
    core_attn_out, last_recurrent_state = torch.ops.sgl_kernel.chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        True,
        cu_seqlens,
        chunk_indices,
        chunk_offsets,
        head_first,
        use_qk_l2norm_in_kernel,
    )
    h = None  # Todo: add return h support
    return core_attn_out, last_recurrent_state, h


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
):
    return torch.ops.sgl_kernel.fused_gdn_gating(A_log, a, b, dt_bias)


def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    use_qk_l2norm_in_kernel: bool,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
):
    return torch.ops.sgl_kernel.fused_sigmoid_gating_delta_rule_update(
        A_log,
        dt_bias,
        q,
        k,
        v,
        a,
        b,
        initial_state_source,
        initial_state_indices,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
        softplus_beta,
        softplus_threshold,
    )
