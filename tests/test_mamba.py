import unittest

import sgl_kernel
import torch
import torch.nn.functional as F
from torch.nn.functional import softplus
from utils import precision

torch.manual_seed(1234)


def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def prepare_chunk_indices_offsets(cu_seqlens: torch.Tensor, chunk_size: int = 64):
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


def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, sequence_length, num_heads, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - num_heads % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    tot_heads = num_heads + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, sequence_length, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )

    # for each chunk
    for i in range(0, tot_heads // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(
                -1, -2
            )
            @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :num_heads]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def chunk_gated_delta_rule_update(
    query,  # [B, T, HK, K]
    key,  # [B, T, HK, K]
    value,  # [B, T, HV, V]
    g,  # [B, T, HV]
    beta,  # [B, T, HV]
    cu_seqlens,  # [N+1]
    initial_state,  # [N, HV, K, V]
    use_qk_l2norm_in_kernel,  # True
):
    num_heads = query.shape[2]
    num_value_heads = value.shape[2]
    batch_size = initial_state.shape[0]
    if num_value_heads // num_heads > 1:
        query = query.repeat_interleave(num_value_heads // num_heads, dim=2)
        key = key.repeat_interleave(num_value_heads // num_heads, dim=2)
    output = torch.empty_like(value)
    final_state = torch.empty_like(initial_state)
    start_q = 0
    for i in range(batch_size):
        end_q = cu_seqlens[i + 1]
        core_attn_outi, last_recurrent_state = torch_chunk_gated_delta_rule(
            query=query[:, start_q:end_q, :, :],
            key=key[:, start_q:end_q, :, :],
            value=value[:, start_q:end_q, :, :],
            g=g[:, start_q:end_q, :],
            beta=beta[:, start_q:end_q, :],
            initial_state=initial_state[i],
            output_final_state=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        output[:, start_q:end_q, :, :] = core_attn_outi
        final_state[i] = last_recurrent_state
        start_q = end_q
    return output, final_state


def torch_recurrent_gated_delta_rule(
    query,  # [B, HK, K]
    key,  # [B, HK, K]
    value,  # [B, HV, V]
    g,  # [B, HV]
    beta,  # [B, HV]
    initial_state,  # [N, HV, K, V]
    output_final_state,
    use_qk_l2norm_in_kernel,  # True
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]
    B, H, T, K = key.shape
    V = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    out = torch.zeros(B, H, T, V, device=value.device)
    state = (
        torch.zeros(B, H, K, V, device=value.device)
        if initial_state is None
        else initial_state.float()
    )

    for i in range(T):
        q_t, k_t, v_t = query[:, :, i], key[:, :, i], value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)
        state = state * g_t
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, i] = (state * q_t.unsqueeze(-1)).sum(dim=-2)

    out = out.transpose(1, 2).contiguous().to(initial_dtype)
    return out, (state if output_final_state else None)


def sigmoid_gating_delta_rule_update(
    query,
    key,
    value,
    A_log,
    a,
    dt_bias,
    b,
    cu_seqlens,
    initial_state_indices,
    initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel=False,
    softplus_beta=1.0,
    softplus_threshold=20.0,
):
    # Match fused decode kernel semantics: iterate sequences in batch order,
    # consume per-sequence local timesteps using cu_seqlens, and update state
    # in-place at initial_state_indices.
    assert query.size(0) == 1, "reference path expects decode mode (seq_len == 1)"
    batch_size = query.size(1)
    assert cu_seqlens.numel() == batch_size + 1

    beta = b.float().sigmoid()
    g = -A_log.float().exp() * softplus(
        a.float() + dt_bias, beta=softplus_beta, threshold=softplus_threshold
    )

    ratio = value.size(2) // key.size(2)
    q_expand = query.repeat_interleave(ratio, dim=2)
    k_expand = key.repeat_interleave(ratio, dim=2)

    out = torch.empty_like(value)
    final_state = initial_state.clone().float()

    for b_idx in range(batch_size):
        seq_start = cu_seqlens[b_idx].item()
        seq_end = cu_seqlens[b_idx + 1].item()
        seq_len = seq_end - seq_start
        state_idx = initial_state_indices[b_idx].item()

        q_i = q_expand[:seq_len, b_idx : b_idx + 1]
        k_i = k_expand[:seq_len, b_idx : b_idx + 1]
        v_i = value[:seq_len, b_idx : b_idx + 1]
        g_i = g[b_idx : b_idx + 1, :].unsqueeze(1).expand(1, seq_len, -1)
        beta_i = beta[b_idx : b_idx + 1, :].unsqueeze(1).expand(1, seq_len, -1)

        out_i, state_i = torch_recurrent_gated_delta_rule(
            q_i.transpose(0, 1),
            k_i.transpose(0, 1),
            v_i.transpose(0, 1),
            g_i,
            beta_i,
            final_state[state_idx : state_idx + 1],
            output_final_state=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        out[:seq_len, b_idx : b_idx + 1] = out_i.transpose(0, 1)
        final_state[state_idx] = state_i[0]

    return out, (final_state[initial_state_indices] if output_final_state else None)


def torch_gdn_gating(A_log, a, b, dt_bias):
    return -A_log.float().exp() * softplus(a.float() + dt_bias).unsqueeze(
        0
    ), b.sigmoid().unsqueeze(0)


torch.set_default_device("xpu")


class TestMambaAttention(unittest.TestCase):
    def test_chunk_gated_delta_rule_multi_chunk_seq(self):
        B, HK, HV, EK, EV, N = 1, 4, 8, 64, 64, 2
        # Force one long sequence (> 2 * 64) so the chunked kernel path is exercised.
        cu_seqlens_ = torch.tensor([0, 129, 161], dtype=torch.int32)
        T = cu_seqlens_[-1].item()
        query_ = torch.rand((B, T, HK, EK), dtype=torch.bfloat16) * 0.05
        key_ = torch.rand((B, T, HK, EK), dtype=torch.bfloat16) * 0.05
        value_ = torch.rand((B, T, HV, EV), dtype=torch.bfloat16) * 0.05
        g_ = torch.rand((B, T, HV), dtype=torch.float32) * 0.05
        beta_ = torch.rand((B, T, HV), dtype=torch.bfloat16) * 0.05
        initial_state_ = torch.rand((N, HV, EK, EV), dtype=torch.float32) * 0.05

        for use_qk_l2norm_in_kernel in [True, False]:
            chunk_indices_, chunk_offsets_ = prepare_chunk_indices_offsets(cu_seqlens_)
            core_attn_out_ref, last_recurrent_state_ref = chunk_gated_delta_rule_update(
                query=query_,
                key=key_,
                value=value_,
                g=g_,
                beta=beta_,
                cu_seqlens=cu_seqlens_,
                initial_state=initial_state_,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )

            core_attn_out, last_recurrent_state = (
                torch.ops.sgl_kernel.chunk_gated_delta_rule(
                    query=query_.clone(),
                    key=key_.clone(),
                    value=value_.clone(),
                    g=g_.clone(),
                    beta=beta_.clone(),
                    initial_state=initial_state_.clone(),
                    output_final_state=True,
                    cu_seqlens=cu_seqlens_.clone(),
                    chunk_indices=chunk_indices_.clone(),
                    chunk_offsets=chunk_offsets_.clone(),
                    head_first=False,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                )
            )

            atol = rtol = precision[core_attn_out.dtype]
            torch.testing.assert_close(
                core_attn_out, core_attn_out_ref, atol=atol, rtol=rtol
            )
            torch.testing.assert_close(
                last_recurrent_state, last_recurrent_state_ref, atol=atol, rtol=rtol
            )

    def test_chunk_gated_delta_rule(self):
        B, L, HK, HV, EK, EV, N = 1, 100, 3, 6, 64, 64, 4
        seqlens = torch.randint(1, L, (N + 1,))
        seqlens[0] = 0
        cu_seqlens_ = torch.cumsum(seqlens, dim=0).to(torch.int32)
        T = cu_seqlens_[-1].item()
        query_ = torch.rand((B, T, HK, EK), dtype=torch.bfloat16) * 0.05
        key_ = torch.rand((B, T, HK, EK), dtype=torch.bfloat16) * 0.05
        value_ = torch.rand((B, T, HV, EV), dtype=torch.bfloat16) * 0.05
        g_ = torch.rand((B, T, HV), dtype=torch.float32) * 0.05
        beta_ = torch.rand((B, T, HV), dtype=torch.bfloat16) * 0.05
        initial_state_ = torch.rand((N, HV, EK, EV), dtype=torch.float32) * 0.05

        for use_qk_l2norm_in_kernel in [True, False]:
            chunk_indices_, chunk_offsets_ = prepare_chunk_indices_offsets(cu_seqlens_)
            core_attn_out_ref, last_recurrent_state_ref = chunk_gated_delta_rule_update(
                query=query_,
                key=key_,
                value=value_,
                g=g_,
                beta=beta_,
                cu_seqlens=cu_seqlens_,
                initial_state=initial_state_,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )

            query = query_.clone()
            key = key_.clone()
            value = value_.clone()
            g = g_.clone()
            beta = beta_.clone()
            cu_seqlens = cu_seqlens_.clone()
            initial_state = initial_state_.clone()

            core_attn_out, last_recurrent_state = (
                torch.ops.sgl_kernel.chunk_gated_delta_rule(
                    query=query,
                    key=key,
                    value=value,
                    g=g,
                    beta=beta,
                    initial_state=initial_state,
                    output_final_state=True,
                    cu_seqlens=cu_seqlens,
                    chunk_indices=chunk_indices_.clone(),
                    chunk_offsets=chunk_offsets_.clone(),
                    head_first=False,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                )
            )
            atol = rtol = precision[core_attn_out.dtype]
            torch.testing.assert_close(
                core_attn_out, core_attn_out_ref, atol=atol, rtol=rtol
            )
            torch.testing.assert_close(
                last_recurrent_state, last_recurrent_state_ref, atol=atol, rtol=rtol
            )

    def test_fused_gdn_gating(self):
        dims = [6, 32]
        for dim in dims:
            A_log = torch.rand(dim)
            a = torch.rand(1024, dim, dtype=torch.bfloat16)
            b = torch.rand(1024, dim, dtype=torch.bfloat16)
            dt_bias = torch.rand(dim, dtype=torch.bfloat16)

            g, beta = torch_gdn_gating(A_log, a, b, dt_bias)
            g_sgl, beta_sgl = torch.ops.sgl_kernel.fused_gdn_gating(
                A_log, a, b, dt_bias
            )
            atol = rtol = precision[g.dtype]
            atol2 = rtol2 = precision[beta.dtype]
            torch.testing.assert_close(g, g_sgl, atol=atol, rtol=rtol)
            torch.testing.assert_close(beta, beta_sgl, atol=atol2, rtol=rtol2)

    def test_fused_sigmoid_gating_delta_rule_update(self):
        batch_size = 5
        num_value_heads = 32
        head_k_dim = 128
        head_v_dim = 128
        num_heads = 16
        seq_len = 1
        attn_tp_size = 1
        key_dim = head_k_dim * num_heads
        value_dim = head_v_dim * num_value_heads
        mixed_qkv_dim = (key_dim * 2 + value_dim) // attn_tp_size
        mixed_qkv = torch.rand(
            seq_len * batch_size, mixed_qkv_dim, dtype=torch.bfloat16
        )
        query, key, value = torch.split(
            mixed_qkv,
            [
                key_dim // attn_tp_size,
                key_dim // attn_tp_size,
                value_dim // attn_tp_size,
            ],
            dim=-1,
        )
        query = query.view(1, batch_size, num_heads, head_k_dim)
        key = key.view(1, batch_size, num_heads, head_k_dim)
        value = value.view(1, batch_size, num_value_heads, head_v_dim)
        A_log = torch.rand(num_value_heads, dtype=torch.float32)
        a = torch.rand(batch_size, num_value_heads, dtype=torch.bfloat16)
        b = torch.rand(batch_size, num_value_heads, dtype=torch.bfloat16)
        dt_bias = torch.rand(num_value_heads, dtype=torch.bfloat16)
        ssm_states = torch.rand(
            513, num_value_heads, head_k_dim, head_v_dim, dtype=torch.float32
        )
        cache_indices = torch.arange(batch_size, dtype=torch.int32)
        cu_seqlens = torch.arange(batch_size + 1, dtype=torch.int32)
        use_qk_l2norm_in_kernel = True
        core_attn_out_ref, last_recurrent_state_ref = sigmoid_gating_delta_rule_update(
            query,
            key,
            value,
            A_log,
            a,
            dt_bias,
            b,
            cu_seqlens=cu_seqlens,
            initial_state_indices=cache_indices,
            initial_state=ssm_states[cache_indices],
            output_final_state=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        core_attn_out = torch.ops.sgl_kernel.fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=query,
            k=key,
            v=value,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            softplus_beta=1.0,
            softplus_threshold=20.0,
        )
        last_recurrent_state = ssm_states[cache_indices]
        atol = rtol = precision[core_attn_out.dtype]
        torch.xpu.synchronize()
        torch.testing.assert_close(
            core_attn_out,
            core_attn_out_ref.transpose(0, 1).to(core_attn_out.dtype),
            atol=atol,
            rtol=rtol,
        )
        torch.testing.assert_close(
            last_recurrent_state, last_recurrent_state_ref, atol=atol, rtol=rtol
        )


if __name__ == "__main__":
    unittest.main()
