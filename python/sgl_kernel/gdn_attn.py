from typing import Dict, Optional

import torch

# Headroom applied when (re)growing the cached workspace buffer, so a call
# that needs slightly more than the previous largest call doesn't force a
# reallocation on every subsequent call of a similar size.
_GDN_WS_HEADROOM = 1.25

# Persistent, grow-only workspace buffers for the fused GDN op, keyed by
# device and shared process-wide across every caller/layer that goes through
# this wrapper.
_gdn_ws_cache: Dict[torch.device, torch.Tensor] = {}


def _stream_key(device: torch.device) -> tuple[torch.device, int]:
    stream = torch.xpu.current_stream(device)
    return device, stream.sycl_queue


def _gdn_workspace_bytes_needed(
    num_prefills: int,
    num_decodes: int,
    non_spec_token: int,
    batch_size: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    tp_size: int,
    dtype: torch.dtype,
) -> int:
    """Compute the exact number of bytes `gdn_attention`'s C++ implementation
    will carve out of the workspace buffer for this call's shapes."""
    return torch.ops.sgl_kernel.gdn_attention_workspace_bytes_needed.default(
        num_prefills,
        num_decodes,
        non_spec_token,
        batch_size,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        tp_size,
        dtype,
    )


def _get_gdn_workspace(nbytes: int, device: torch.device) -> torch.Tensor:
    """Return a flat 1-D `torch.uint8` buffer with >= `nbytes` capacity on
    `device`, cached/grown (grow-only, with headroom) across calls."""
    key = _stream_key(device)
    cur = _gdn_ws_cache.get(key)
    if cur is None or cur.numel() < nbytes:
        new_numel = max(nbytes, int(nbytes * _GDN_WS_HEADROOM))
        cur = torch.empty(new_numel, dtype=torch.uint8, device=device)
        _gdn_ws_cache[key] = cur
    return cur


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
    ``conv_state`` / ``ssm_state`` in place.
    """
    non_spec_token = (
        non_spec_token_indx.numel()
        if non_spec_token_indx is not None
        else num_actual_tokens
    )
    batch_size = (
        (non_spec_query_start_loc.numel() - 1)
        if non_spec_query_start_loc is not None
        else 0
    )
    nbytes = _gdn_workspace_bytes_needed(
        num_prefills,
        num_decodes,
        non_spec_token,
        batch_size,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        tp_size,
        projected_states_qkvz.dtype,
    )
    workspace: Optional[torch.Tensor] = None
    if nbytes > 0:
        workspace = _get_gdn_workspace(nbytes, projected_states_qkvz.device)
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
        workspace,
    )


def gated_delta_rule(
    core_attn_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    num_prefills: int,
    num_decodes: int,
    token_indx: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
) -> None:
    """Standalone GDN (Gated DeltaNet) recurrent op for plain (non-spec)
    prefill/decode -- the SSM-only half of what the fused ``gdn_attention``
    op does internally after conv1d. Exposed separately so callers can drive
    prefill/decode via ``causal_conv1d_fwd``/``causal_conv1d_update`` +
    this op, matching the same two-call convention CUDA/Triton/NPU/CPU
    already use for GDN, instead of the fused monolithic op.

    Writes the result in place into ``core_attn_out``, and updates
    ``ssm_state`` (the committed recurrent-state pool) in place -- unlike
    ``gated_delta_rule_verify``, there is no separate scratch pool here.

    Args:
        core_attn_out: ``[total_seqlen, num_v_heads, head_v_dim]``.
        q/k: ``[total_seqlen, num_k_heads, head_k_dim]``.
        v: ``[total_seqlen, num_v_heads, head_v_dim]``.
        b/a: ``[total_seqlen, num_v_heads]``.
        ssm_state: committed pool, ``[cache_batch_size, num_v_heads,
            head_v_dim, head_k_dim]``, updated in place at the rows named by
            ``cache_indices``.
        query_start_loc: ``[num_prefills + num_decodes + 1]`` int32, cu_seqlen
            boundaries (same convention as the standalone conv1d ops).
        cache_indices: ``[num_prefills + num_decodes]`` int32, row into
            ``ssm_state`` per sequence.
        num_prefills/num_decodes: sequence-count split of ``query_start_loc``;
            every prefill sequence may have length > 1, every decode sequence
            has length exactly 1.
        token_indx: optional ``[total_seqlen]`` remap from local to global
            token position for ``core_attn_out``.
        has_initial_state: optional ``[num_prefills + num_decodes]`` bool; if
            provided, sequences with ``False`` start from a zero state
            instead of seeding from ``ssm_state[cache_indices[i]]``.
    """
    torch.ops.sgl_kernel.gated_delta_rule.default(
        core_attn_out,
        q,
        k,
        v,
        b,
        a,
        A_log,
        dt_bias,
        ssm_state,
        query_start_loc,
        token_indx,
        cache_indices,
        has_initial_state,
        num_prefills,
        num_decodes,
    )


def gated_delta_rule_verify(
    core_attn_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state: torch.Tensor,
    cache_indices: torch.Tensor,
    intermediate_states_buffer: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
    cache_steps: int,
    token_indx: Optional[torch.Tensor] = None,
    retrieve_parent_token: Optional[torch.Tensor] = None,
) -> None:
    """Standalone GDN (Gated DeltaNet) recurrent ``target_verify`` op for
    speculative decoding, using SGLang's native verify convention (matches
    the Triton/KDA recurrent verify kernels): seeds this round's initial
    state from the COMMITTED ``ssm_state`` pool via a plain 1D
    ``cache_indices`` (same convention as decode/extend), and snapshots every
    draft position's post-update state into ``intermediate_states_buffer``
    (indexed by ``intermediate_state_indices`` + ``cache_steps``) instead of
    writing into the committed pool. ``ssm_state`` is never mutated by this
    op -- the caller commits the accepted-length snapshot into it afterwards,
    once acceptance is known (mirrors SGLang's
    ``update_mamba_state_after_mtp_verify``).

    Writes results in place into ``core_attn_out`` and
    ``intermediate_states_buffer``.

    Args:
        core_attn_out: ``[num_spec_decodes * num_spec_tokens, num_v_heads,
            head_v_dim]``.
        q/k: ``[num_spec_decodes * num_spec_tokens, num_k_heads, head_k_dim]``.
        v: ``[num_spec_decodes * num_spec_tokens, num_v_heads, head_v_dim]``.
        b/a: ``[num_spec_decodes * num_spec_tokens, num_v_heads]``.
        ssm_state: committed pool, ``[cache_batch_size, num_v_heads,
            head_v_dim, head_k_dim]``.
        cache_indices: ``[num_spec_decodes]`` int32, row into ``ssm_state``
            used to seed this round's initial state.
        intermediate_states_buffer: ``[scratch_rows, cache_steps,
            num_v_heads, head_v_dim, head_k_dim]``, contiguous.
        intermediate_state_indices: ``[num_spec_decodes]`` int32, row into
            ``intermediate_states_buffer``.
        cache_steps: capacity (in steps) of each scratch row; must be
            ``>= num_spec_tokens``.
        token_indx: optional ``[num_spec_decodes * num_spec_tokens]`` remap
            from local to global token position for ``core_attn_out``.
        retrieve_parent_token: optional ``[num_spec_decodes,
            num_spec_tokens]`` int32 or int64 EAGLE-tree parent-step index per draft
            position; ``None`` selects the (faster) linear-chain path.
    """
    torch.ops.sgl_kernel.gated_delta_rule_verify.default(
        core_attn_out,
        q,
        k,
        v,
        b,
        a,
        A_log,
        dt_bias,
        ssm_state,
        cache_indices,
        intermediate_states_buffer,
        intermediate_state_indices,
        cache_steps,
        token_indx,
        retrieve_parent_token,
    )
