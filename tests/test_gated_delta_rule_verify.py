"""Tests for ``torch.ops.sgl_kernel.gated_delta_rule_verify``: the standalone
GDN recurrent ``target_verify`` (speculative-decode) op that uses SGLang's
native verify convention (matches the Triton/KDA recurrent verify kernels in
``fused_sigmoid_gating_delta_rule_update``), as opposed to the
vLLM-style 2D ``spec_state_indices_tensor`` + ``num_accepted_tokens``
convention still used internally by the fused ``gdn_attention`` op's own
(unused/untested) spec branch.

Contract under test:
  - ``cache_indices`` (1D) seeds this round's initial state from the
    COMMITTED ``ssm_state`` pool; that pool must be left byte-identical
    afterwards (the op never writes back to it).
  - Every draft position's post-update state is snapshotted into
    ``intermediate_states_buffer`` at row ``intermediate_state_indices``,
    step ``t_local``.
  - ``retrieve_parent_token`` (optional) makes each step re-seed from its
    parent's own snapshot (EAGLE tree) instead of the previous step
    (linear chain).

The reference implementation below is a direct transcription of the
sequential gated delta-rule recurrence (independently re-derived, matching
``gdn_reference.py``'s decode-path reference and cross-checked against the
kernel's algebraic form: the kernel decays state before the readout/output
matmuls, which is equivalent to decaying the readout itself, as used here).
"""

import pytest
import sgl_kernel  # noqa: F401  registers torch.ops.sgl_kernel.gated_delta_rule_verify
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available()
    or not hasattr(torch.ops.sgl_kernel, "gated_delta_rule_verify"),
    reason="Requires Intel XPU build with gated_delta_rule_verify op",
)

NUM_K_HEADS = 16
NUM_V_HEADS = 32
HEAD_K_DIM = 128
HEAD_V_DIM = 128
L2NORM_EPS = 1e-6


def _reference_verify(
    q,
    k,
    v,
    b,
    a,
    A_log,
    dt_bias,
    ssm_state,
    cache_indices,
    num_spec_tokens,
    retrieve_parent_token=None,
):
    """Sequential reference matching the kernel's per-step recurrence.

    Returns ``(core_attn_out, snapshots)`` where ``snapshots`` has shape
    ``[num_spec_decodes, num_spec_tokens, num_v_heads, head_v_dim, head_k_dim]``
    -- one entry per draft position, indexed by (request, local step).
    """
    device = q.device
    dtype = torch.float32
    num_spec_decodes = cache_indices.numel()
    total, nk, hk = q.shape
    nv, hv = v.shape[1], v.shape[2]
    kv_ratio = nv // nk

    q32, k32, v32 = q.to(dtype), k.to(dtype), v.to(dtype)
    b32, a32 = b.to(dtype), a.to(dtype)

    q_norm = (
        q32 * torch.rsqrt(q32.pow(2).sum(-1, keepdim=True) + L2NORM_EPS) * (hk**-0.5)
    )
    k_norm = k32 * torch.rsqrt(k32.pow(2).sum(-1, keepdim=True) + L2NORM_EPS)
    q_exp = q_norm.repeat_interleave(kv_ratio, dim=1)  # [total, nv, hk]
    k_exp = k_norm.repeat_interleave(kv_ratio, dim=1)  # [total, nv, hk]

    beta = torch.sigmoid(b32)  # [total, nv]
    decay = torch.exp(
        -torch.exp(A_log.to(dtype)) * F.softplus(a32 + dt_bias.to(dtype))
    )  # [total, nv]

    core_attn_out = torch.zeros(total, nv, hv, dtype=dtype, device=device)
    snapshots = torch.zeros(
        num_spec_decodes, num_spec_tokens, nv, hv, hk, dtype=dtype, device=device
    )

    for n in range(num_spec_decodes):
        seed_idx = int(cache_indices[n])
        S = ssm_state[seed_idx].to(dtype).clone()  # [nv, hv, hk]
        for t_local in range(num_spec_tokens):
            if retrieve_parent_token is not None and t_local != 0:
                parent = int(retrieve_parent_token[n, t_local])
                if parent >= 0:
                    S = snapshots[n, parent].clone()
            t = n * num_spec_tokens + t_local
            k_t, q_t, v_t = k_exp[t], q_exp[t], v32[t]
            beta_t, decay_t = beta[t], decay[t]

            readout = torch.matmul(S, k_t.unsqueeze(-1)).squeeze(-1)  # [nv, hv]
            delta = beta_t.unsqueeze(-1) * (v_t - decay_t.unsqueeze(-1) * readout)
            S = decay_t.unsqueeze(-1).unsqueeze(-1) * S + delta.unsqueeze(
                -1
            ) * k_t.unsqueeze(-2)
            core_attn_out[t] = torch.matmul(S, q_t.unsqueeze(-1)).squeeze(-1)
            snapshots[n, t_local] = S

    return core_attn_out, snapshots


def _make_inputs(
    num_spec_decodes, num_spec_tokens, dtype, device, nk=NUM_K_HEADS, nv=NUM_V_HEADS
):
    hk, hv = HEAD_K_DIM, HEAD_V_DIM
    total = num_spec_decodes * num_spec_tokens
    cache_bs = max(64, num_spec_decodes * 2)
    scratch_rows = max(64, num_spec_decodes * 2)

    g = torch.Generator(device=device).manual_seed(0)
    rnd = lambda *s: torch.randn(*s, dtype=dtype, device=device, generator=g)

    q = rnd(total, nk, hk)
    k = rnd(total, nk, hk)
    v = rnd(total, nv, hv)
    b = rnd(total, nv)
    a = rnd(total, nv)
    A_log = torch.randn(nv, dtype=torch.float32, device=device, generator=g)
    dt_bias = rnd(nv)
    ssm_state = rnd(cache_bs, nv, hv, hk)

    # Distinct rows so committed-pool/scratch aliasing bugs surface as
    # mismatches rather than silently canceling out. randperm requires a CPU
    # generator; move the result to `device` afterwards.
    cpu_g = torch.Generator(device="cpu").manual_seed(0)
    cache_indices = (
        torch.randperm(cache_bs, generator=cpu_g)[:num_spec_decodes]
        .to(torch.int32)
        .to(device)
    )
    intermediate_state_indices = (
        torch.randperm(scratch_rows, generator=cpu_g)[:num_spec_decodes]
        .to(torch.int32)
        .to(device)
    )

    core_attn_out = torch.zeros(total, nv, hv, dtype=dtype, device=device)
    intermediate_states_buffer = torch.zeros(
        scratch_rows, num_spec_tokens, nv, hv, hk, dtype=dtype, device=device
    )

    return dict(
        q=q,
        k=k,
        v=v,
        b=b,
        a=a,
        A_log=A_log,
        dt_bias=dt_bias,
        ssm_state=ssm_state,
        cache_indices=cache_indices,
        intermediate_states_buffer=intermediate_states_buffer,
        intermediate_state_indices=intermediate_state_indices,
        core_attn_out=core_attn_out,
    )


@pytest.mark.parametrize(
    "num_spec_decodes,num_spec_tokens", [(1, 1), (1, 4), (4, 3), (8, 5)]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gated_delta_rule_verify_chain(num_spec_decodes, num_spec_tokens, dtype):
    device = torch.device("xpu")
    i = _make_inputs(num_spec_decodes, num_spec_tokens, dtype, device)
    ssm_state_before = i["ssm_state"].clone()

    torch.ops.sgl_kernel.gated_delta_rule_verify(
        i["core_attn_out"],
        i["q"],
        i["k"],
        i["v"],
        i["b"],
        i["a"],
        i["A_log"],
        i["dt_bias"],
        i["ssm_state"],
        i["cache_indices"],
        i["intermediate_states_buffer"],
        i["intermediate_state_indices"],
        num_spec_tokens,
        None,
        None,
    )
    torch.xpu.synchronize()

    # The committed pool must be untouched: acceptance isn't known yet.
    torch.testing.assert_close(i["ssm_state"], ssm_state_before, rtol=0, atol=0)

    ref_out, ref_snapshots = _reference_verify(
        i["q"],
        i["k"],
        i["v"],
        i["b"],
        i["a"],
        i["A_log"],
        i["dt_bias"],
        ssm_state_before,
        i["cache_indices"],
        num_spec_tokens,
    )

    rtol, atol = (3e-2, 5e-2) if dtype == torch.bfloat16 else (1e-2, 1e-2)
    torch.testing.assert_close(
        i["core_attn_out"].float(), ref_out, rtol=rtol, atol=atol
    )

    got_snapshots = torch.stack(
        [
            i["intermediate_states_buffer"][int(row)]
            for row in i["intermediate_state_indices"]
        ]
    )
    torch.testing.assert_close(
        got_snapshots.float(), ref_snapshots, rtol=rtol, atol=atol
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gated_delta_rule_verify_tree(dtype):
    """EAGLE-style tree draft: request 0 is a depth-3 binary tree (7 nodes),
    request 1 is a plain linear chain (parent == t_local - 1), exercised
    together to make sure per-request ``retrieve_parent_token`` rows don't
    interfere with each other."""
    device = torch.device("xpu")
    num_spec_decodes, num_spec_tokens = 2, 7
    i = _make_inputs(num_spec_decodes, num_spec_tokens, dtype, device)
    ssm_state_before = i["ssm_state"].clone()

    # Binary tree parent indices for a 7-node complete binary tree (root=0):
    # node t's parent is (t - 1) // 2, root's parent is itself (t_local == 0
    # is never read regardless of its value).
    tree_parents = [0, 0, 0, 1, 1, 2, 2]
    chain_parents = list(range(-1, num_spec_tokens - 1))  # [-1, 0, 1, ..., 5]
    retrieve_parent_token = torch.tensor(
        [tree_parents, chain_parents], dtype=torch.int32, device=device
    )

    torch.ops.sgl_kernel.gated_delta_rule_verify(
        i["core_attn_out"],
        i["q"],
        i["k"],
        i["v"],
        i["b"],
        i["a"],
        i["A_log"],
        i["dt_bias"],
        i["ssm_state"],
        i["cache_indices"],
        i["intermediate_states_buffer"],
        i["intermediate_state_indices"],
        num_spec_tokens,
        None,
        retrieve_parent_token,
    )
    torch.xpu.synchronize()

    torch.testing.assert_close(i["ssm_state"], ssm_state_before, rtol=0, atol=0)

    ref_out, ref_snapshots = _reference_verify(
        i["q"],
        i["k"],
        i["v"],
        i["b"],
        i["a"],
        i["A_log"],
        i["dt_bias"],
        ssm_state_before,
        i["cache_indices"],
        num_spec_tokens,
        retrieve_parent_token=retrieve_parent_token,
    )

    rtol, atol = (3e-2, 5e-2) if dtype == torch.bfloat16 else (1e-2, 1e-2)
    torch.testing.assert_close(
        i["core_attn_out"].float(), ref_out, rtol=rtol, atol=atol
    )

    got_snapshots = torch.stack(
        [
            i["intermediate_states_buffer"][int(row)]
            for row in i["intermediate_state_indices"]
        ]
    )
    torch.testing.assert_close(
        got_snapshots.float(), ref_snapshots, rtol=rtol, atol=atol
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_gated_delta_rule_verify_token_indx_remap(dtype):
    """``core_attn_out`` may be laid out by a global token order different
    from the local (request, step) order; ``token_indx`` remaps writes."""
    device = torch.device("xpu")
    num_spec_decodes, num_spec_tokens = 3, 2
    i = _make_inputs(num_spec_decodes, num_spec_tokens, dtype, device)
    total = num_spec_decodes * num_spec_tokens

    # Reverse global order: local token t writes to global position
    # (total - 1 - t).
    token_indx = torch.arange(total - 1, -1, -1, dtype=torch.int32, device=device)
    core_attn_out = torch.zeros_like(i["core_attn_out"])

    torch.ops.sgl_kernel.gated_delta_rule_verify(
        core_attn_out,
        i["q"],
        i["k"],
        i["v"],
        i["b"],
        i["a"],
        i["A_log"],
        i["dt_bias"],
        i["ssm_state"],
        i["cache_indices"],
        i["intermediate_states_buffer"],
        i["intermediate_state_indices"],
        num_spec_tokens,
        token_indx,
        None,
    )
    torch.xpu.synchronize()

    ref_out, _ = _reference_verify(
        i["q"],
        i["k"],
        i["v"],
        i["b"],
        i["a"],
        i["A_log"],
        i["dt_bias"],
        i["ssm_state"],
        i["cache_indices"],
        num_spec_tokens,
    )
    rtol, atol = 3e-2, 5e-2
    torch.testing.assert_close(
        core_attn_out.float(), ref_out.flip(0), rtol=rtol, atol=atol
    )


@pytest.mark.parametrize("accepted_len", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_gated_delta_rule_verify_commit_matches_decode(accepted_len, dtype):
    """The snapshot at step ``accepted_len - 1`` (what a caller would commit
    into the committed pool after sampling decides ``accepted_len`` tokens
    were accepted) must equal the state produced by running the plain
    (non-verify) recurrence for exactly ``accepted_len`` tokens from the same
    seed -- i.e. the snapshot IS the correct post-accept committed state,
    with no additional bookkeeping required."""
    device = torch.device("xpu")
    num_spec_decodes, num_spec_tokens = 2, 4
    i = _make_inputs(num_spec_decodes, num_spec_tokens, dtype, device)
    ssm_state_before = i["ssm_state"].clone()

    torch.ops.sgl_kernel.gated_delta_rule_verify(
        i["core_attn_out"],
        i["q"],
        i["k"],
        i["v"],
        i["b"],
        i["a"],
        i["A_log"],
        i["dt_bias"],
        i["ssm_state"],
        i["cache_indices"],
        i["intermediate_states_buffer"],
        i["intermediate_state_indices"],
        num_spec_tokens,
        None,
        None,
    )
    torch.xpu.synchronize()

    _, ref_snapshots = _reference_verify(
        i["q"],
        i["k"],
        i["v"],
        i["b"],
        i["a"],
        i["A_log"],
        i["dt_bias"],
        ssm_state_before,
        i["cache_indices"],
        num_spec_tokens,
    )

    for n in range(num_spec_decodes):
        row = int(i["intermediate_state_indices"][n])
        got = i["intermediate_states_buffer"][row, accepted_len - 1].float()
        expected = ref_snapshots[n, accepted_len - 1]
        torch.testing.assert_close(got, expected, rtol=3e-2, atol=5e-2)
