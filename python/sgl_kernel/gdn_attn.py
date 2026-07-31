from typing import Dict, Optional

import torch

# Must match `gdn::chunk_size_xe2` in sgl-kernel-xpu's
# src/sycl/gdn_attn/gdn_attn_utils.h -- the prefill/chunk-scan path pads
# non-spec tokens up to a multiple of this chunk size.
_GDN_CHUNK_SIZE_XE2 = 64
# Byte alignment the C++ side (`gdn_attention`'s `make_ws_tensor`) uses when
# carving scratch tensors out of the workspace buffer. Kept in sync so the
# byte-size computed here exactly matches what the op will actually consume.
_GDN_WS_ALIGN = 256
# Headroom applied when (re)growing the cached workspace buffer, so a call
# that needs slightly more than the previous largest call doesn't force a
# reallocation on every subsequent call of a similar size.
_GDN_WS_HEADROOM = 1.25

# Persistent, grow-only workspace buffers for the fused GDN op, keyed by
# device and shared process-wide across every caller/layer that goes through
# this wrapper.
_gdn_ws_cache: Dict[torch.device, torch.Tensor] = {}


def _align_up(nbytes: int) -> int:
    return (nbytes + _GDN_WS_ALIGN - 1) // _GDN_WS_ALIGN * _GDN_WS_ALIGN


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
    will carve out of the workspace buffer for this call's shapes, mirroring
    its internal (aligned, sequential-cursor) layout math exactly -- so the
    Python-side buffer is grown to precisely the size actually needed (no
    guessing/duplication of unrelated logic, no over/under allocation)."""
    if non_spec_token <= 0 or (num_prefills + num_decodes) <= 0:
        return 0
    nk = num_k_heads // tp_size
    nv = num_v_heads // tp_size
    itemsize = torch.tensor([], dtype=dtype).element_size()
    if num_prefills > 0:
        # Prefill/chunk path: q, k, v (activation dtype) + b_prefill,
        # a_prefill (always float32), all padded to a chunk-size multiple.
        padding_size = batch_size * (_GDN_CHUNK_SIZE_XE2 - 1)
        n = non_spec_token + padding_size
        qk_bytes = _align_up(n * nk * head_k_dim * itemsize)
        v_bytes = _align_up(n * nv * head_v_dim * itemsize)
        ba_bytes = _align_up(n * nv * 4)  # float32
    else:
        # Decode path: q, k, v, b, a, all activation dtype, unpadded.
        n = non_spec_token
        qk_bytes = _align_up(n * nk * head_k_dim * itemsize)
        v_bytes = _align_up(n * nv * head_v_dim * itemsize)
        ba_bytes = _align_up(n * nv * itemsize)
    return 2 * qk_bytes + v_bytes + 2 * ba_bytes


def _get_gdn_workspace(nbytes: int, device: torch.device) -> torch.Tensor:
    """Return a flat 1-D `torch.uint8` buffer with >= `nbytes` capacity on
    `device`, cached/grown (grow-only, with headroom) across calls."""
    cur = _gdn_ws_cache.get(device)
    if cur is None or cur.numel() < nbytes:
        new_numel = max(nbytes, int(nbytes * _GDN_WS_HEADROOM))
        cur = torch.empty(new_numel, dtype=torch.uint8, device=device)
        _gdn_ws_cache[device] = cur
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
