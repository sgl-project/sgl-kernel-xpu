"""Pure-PyTorch reference implementation of the fused GDN (Gated DeltaNet)
attention op, used to validate ``torch.ops.sgl_kernel.gdn_attention`` for
numerical correctness.

This reimplements, from first principles (i.e. by re-deriving the math, not
by transcribing the SYCL kernels), the *sequential*, per-token, per-head
gated delta-rule recurrence:

    g_t       = -exp(A_log_h) * softplus(a_t + dt_bias_h)   # per-step log-decay
    decay_t   = exp(g_t)
    beta_t    = sigmoid(b_t)
    readout_t = S_{t-1} @ k_t
    delta_t   = beta_t * (v_t - decay_t * readout_t)
    S_t       = decay_t * S_{t-1} + outer(delta_t, k_t)
    o_t       = S_t @ q_t

This closed-form recurrence was derived directly from, and cross-checked
against, two independent implementations in the C++/SYCL sources:

  - The chunked/parallel prefill formulation (``chunk_prepare_kernel``,
    ``chunk_compute_A_kernel``, ``chunk_inverse_opt_kernel``,
    ``chunk_compute_wu_kernel``, ``chunk_fwd_o_kernel`` in
    ``src/sycl/kernels/gdn_attn/chunk_gated_delta_rule_kernels_xe20.hpp``),
    reduced algebraically to the ``chunk_size == 1`` case.
  - The sequential decode formulation (``gated_delta_rule_kernel`` in
    ``src/sycl/gdn_attn/gated_delta_rule.hpp``), which implements exactly
    this recurrence directly, token by token.

Both reduce to the same recurrence above, which gives strong confidence the
derivation is correct independent of either kernel implementation.

Only the ``reorder_input=False`` input layout (as used by
``tests/test_gdn_attention.py``) is supported; conv1d channel grouping and
``qkvz``/``ba`` splitting follow ``chunk_causal_conv1d_kernel`` /
``chunk_reorder_zba_kernel`` in ``src/sycl/gdn_attn/chunk_causal_conv1d.hpp``.
"""

from typing import Optional

import torch
import torch.nn.functional as F


def _causal_depthwise_conv1d(
    x: torch.Tensor,  # [seq_len, channels], float32
    history: torch.Tensor,  # [width - 1, channels], float32
    weight: torch.Tensor,  # [channels, width], float32
    bias: Optional[torch.Tensor],  # [channels], float32 or None
) -> torch.Tensor:
    """Causal depthwise conv1d matching ``chunk_causal_conv1d_kernel``'s tap
    convention: ``output[t] = sum_i weight[:, i] * x_padded[t + i]``, where
    ``x_padded = cat([history, x])`` and ``weight[:, -1]`` pairs with the
    current token (i.e. weights are in chronological, not flipped, order).
    """
    width = weight.shape[1]
    seq_len = x.shape[0]
    x_padded = torch.cat([history, x], dim=0)  # [seq_len + width - 1, channels]
    out = torch.zeros_like(x)
    for i in range(width):
        out = out + x_padded[i : i + seq_len, :] * weight[:, i]
    if bias is not None:
        out = out + bias
    return out


def _silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def reference_gdn_attention(
    qkvz: torch.Tensor,
    ba: torch.Tensor,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_bias: Optional[torch.Tensor],
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    query_start_loc: torch.Tensor,
    has_initial_state: torch.Tensor,
    state_indices: torch.Tensor,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    l2norm_eps: float = 1e-6,
):
    """Independent reference for ``torch.ops.sgl_kernel.gdn_attention``.

    All math is done in float32 (matching the kernels' internal float
    accumulators). Returns ``(core_attn_out, z, ssm_state_new)`` as float32
    tensors with shapes matching the op's ``core_attn_out``/``z``/``ssm_state``.
    """
    device = qkvz.device
    dtype = torch.float32
    hk, hv = head_k_dim, head_v_dim
    nk, nv = num_k_heads, num_v_heads
    kv_ratio = nv // nk
    width = conv_weights.shape[1]
    num_tokens = qkvz.shape[0]
    qkvz_dim = 2 * hk + 2 * hv * kv_ratio

    qkvz_f = qkvz.to(dtype)
    ba_f = ba.to(dtype)

    # --- Split mixed qkvz (per-k-head layout: [q | k | v | z]) ---
    qkvz_v = qkvz_f.view(num_tokens, nk, qkvz_dim)
    q_all = qkvz_v[:, :, 0:hk].reshape(num_tokens, nk * hk)
    k_all = qkvz_v[:, :, hk : 2 * hk].reshape(num_tokens, nk * hk)
    v_all = qkvz_v[:, :, 2 * hk : 2 * hk + hv * kv_ratio].reshape(
        num_tokens, nk * hv * kv_ratio
    )
    z_all = qkvz_v[:, :, 2 * hk + hv * kv_ratio : qkvz_dim].reshape(num_tokens, nv, hv)

    # Canonical channel order used by conv_weights/conv_bias/conv_state:
    # all q channels (grouped by k-head), then all k channels, then all v
    # channels -- which is exactly what concatenating q_all/k_all/v_all gives.
    qkv_all = torch.cat([q_all, k_all, v_all], dim=-1)  # [tokens, qkv_size]
    qkv_size = qkv_all.shape[1]

    # --- Split mixed ba (per-k-head layout: [b_0..b_{r-1} | a_0..a_{r-1}]) ---
    ba_v = ba_f.view(num_tokens, nk, 2, kv_ratio)
    b_all = ba_v[:, :, 0, :].reshape(num_tokens, nv)  # raw beta logits
    a_all = ba_v[:, :, 1, :].reshape(num_tokens, nv)  # raw decay logits

    conv_w = conv_weights.to(dtype)  # [qkv_size, width]
    conv_b = conv_bias.to(dtype) if conv_bias is not None else None
    conv_state_f = conv_state.to(dtype)  # [cache_bs, width - 1, qkv_size]

    batch_size = query_start_loc.numel() - 1

    # --- Causal depthwise conv1d + SiLU, per sequence ---
    conv_out = torch.empty(num_tokens, qkv_size, dtype=dtype, device=device)
    for b in range(batch_size):
        s, e = int(query_start_loc[b]), int(query_start_loc[b + 1])
        cache_idx = int(state_indices[b])
        if bool(has_initial_state[b]):
            history = conv_state_f[cache_idx]  # [width - 1, qkv_size]
        else:
            history = torch.zeros(width - 1, qkv_size, dtype=dtype, device=device)
        conv_out[s:e] = _causal_depthwise_conv1d(qkv_all[s:e], history, conv_w, conv_b)
    conv_out = _silu(conv_out)

    q_conv, k_conv, v_conv = conv_out.split(
        [nk * hk, nk * hk, nk * hv * kv_ratio], dim=-1
    )
    q_conv = q_conv.view(num_tokens, nk, hk)
    k_conv = k_conv.view(num_tokens, nk, hk)
    v_conv = v_conv.view(num_tokens, nk, kv_ratio, hv).reshape(num_tokens, nv, hv)

    # --- L2 norm (q additionally scaled by 1/sqrt(head_k_dim)) ---
    q_norm = (
        q_conv
        * torch.rsqrt(q_conv.pow(2).sum(-1, keepdim=True) + l2norm_eps)
        * (hk**-0.5)
    )
    k_norm = k_conv * torch.rsqrt(k_conv.pow(2).sum(-1, keepdim=True) + l2norm_eps)

    q_exp = q_norm.repeat_interleave(kv_ratio, dim=1)  # [tokens, nv, hk]
    k_exp = k_norm.repeat_interleave(kv_ratio, dim=1)  # [tokens, nv, hk]

    beta = torch.sigmoid(b_all)  # [tokens, nv]
    decay = torch.exp(
        -torch.exp(A_log.to(dtype)) * F.softplus(a_all + dt_bias.to(dtype))
    )  # [tokens, nv]

    core_attn_out = torch.zeros(num_tokens, nv, hv, dtype=dtype, device=device)
    ssm_state_new = ssm_state.to(dtype).clone()

    # --- Sequential gated delta-rule recurrence, per sequence ---
    for b in range(batch_size):
        s, e = int(query_start_loc[b]), int(query_start_loc[b + 1])
        cache_idx = int(state_indices[b])
        if bool(has_initial_state[b]):
            S = ssm_state_new[cache_idx].clone()  # [nv, hv, hk]
        else:
            S = torch.zeros(nv, hv, hk, dtype=dtype, device=device)
        for t in range(s, e):
            k_t = k_exp[t]  # [nv, hk]
            q_t = q_exp[t]  # [nv, hk]
            v_t = v_conv[t]  # [nv, hv]
            beta_t = beta[t]  # [nv]
            decay_t = decay[t]  # [nv]

            readout = torch.matmul(S, k_t.unsqueeze(-1)).squeeze(-1)  # [nv, hv]
            delta = beta_t.unsqueeze(-1) * (
                v_t - decay_t.unsqueeze(-1) * readout
            )  # [nv, hv]
            S = decay_t.unsqueeze(-1).unsqueeze(-1) * S + delta.unsqueeze(
                -1
            ) * k_t.unsqueeze(-2)
            core_attn_out[t] = torch.matmul(S, q_t.unsqueeze(-1)).squeeze(-1)
        ssm_state_new[cache_idx] = S

    return core_attn_out, z_all.to(dtype), ssm_state_new
