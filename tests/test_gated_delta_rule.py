"""Tests for ``torch.ops.sgl_kernel.gated_delta_rule``: the standalone GDN
recurrent op for plain (non-spec) prefill/decode -- the SSM-only half of
what the fused ``gdn_attention`` op does internally after conv1d.

Unlike ``gated_delta_rule_verify`` (spec-decode/verify, never mutates the
committed ``ssm_state`` pool), this op is the "commit directly" convention
used by ordinary prefill/decode: ``ssm_state`` is read from and written back
to in place at the rows named by ``cache_indices``, exactly mirroring
CUDA/Triton/NPU/CPU's plain (non-spec) GDN kernels.

The reference implementation below is the same sequential gated delta-rule
recurrence used by ``tests/test_gated_delta_rule_verify.py`` and
``tests/gdn_reference.py`` (independently re-derived, cross-checked against
both the chunked/parallel and sequential/decode SYCL kernel formulations),
generalized here to variable-per-sequence lengths via ``query_start_loc``
(matching plain prefill/decode, as opposed to verify's fixed
``num_spec_tokens``-per-request layout).
"""

import pytest
import sgl_kernel  # noqa: F401  registers torch.ops.sgl_kernel.gated_delta_rule
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available()
    or not hasattr(torch.ops.sgl_kernel, "gated_delta_rule"),
    reason="Requires Intel XPU build with gated_delta_rule op",
)

NUM_K_HEADS = 16
NUM_V_HEADS = 32
HEAD_K_DIM = 128
HEAD_V_DIM = 128
L2NORM_EPS = 1e-6


def _reference(
    q,
    k,
    v,
    b,
    a,
    A_log,
    dt_bias,
    ssm_state,
    query_start_loc,
    cache_indices,
    has_initial_state=None,
):
    """Sequential reference matching the kernel's per-step recurrence,
    generalized to variable-length sequences (arbitrary mix of
    prefill/decode) via ``query_start_loc``.

    Returns ``(core_attn_out, ssm_state_new)`` in local-token order (i.e.
    NOT accounting for any ``token_indx`` remap -- callers that use
    ``token_indx`` should compare against a ``token_indx``-scattered view of
    this function's output).
    """
    device = q.device
    dtype = torch.float32
    batch_size = query_start_loc.numel() - 1
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
    ssm_state_new = ssm_state.to(dtype).clone()

    for n in range(batch_size):
        s, e = int(query_start_loc[n]), int(query_start_loc[n + 1])
        cache_idx = int(cache_indices[n])
        seeded = has_initial_state is None or bool(has_initial_state[n])
        if seeded:
            S = ssm_state_new[cache_idx].clone()  # [nv, hv, hk]
        else:
            S = torch.zeros(nv, hv, hk, dtype=dtype, device=device)
        for t in range(s, e):
            k_t, q_t, v_t = k_exp[t], q_exp[t], v32[t]
            beta_t, decay_t = beta[t], decay[t]

            readout = torch.matmul(S, k_t.unsqueeze(-1)).squeeze(-1)  # [nv, hv]
            delta = beta_t.unsqueeze(-1) * (v_t - decay_t.unsqueeze(-1) * readout)
            S = decay_t.unsqueeze(-1).unsqueeze(-1) * S + delta.unsqueeze(
                -1
            ) * k_t.unsqueeze(-2)
            core_attn_out[t] = torch.matmul(S, q_t.unsqueeze(-1)).squeeze(-1)
        ssm_state_new[cache_idx] = S

    return core_attn_out, ssm_state_new


def _make_inputs(
    seq_lens,
    dtype,
    device,
    nk=NUM_K_HEADS,
    nv=NUM_V_HEADS,
    has_initial_state=None,
):
    hk, hv = HEAD_K_DIM, HEAD_V_DIM
    batch_size = len(seq_lens)
    total = sum(seq_lens)
    cache_bs = max(64, batch_size * 2)

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

    cpu_g = torch.Generator(device="cpu").manual_seed(0)
    cache_indices = (
        torch.randperm(cache_bs, generator=cpu_g)[:batch_size]
        .to(torch.int32)
        .to(device)
    )

    query_start_loc = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    query_start_loc[1:] = torch.cumsum(
        torch.tensor(seq_lens, dtype=torch.int32, device=device), dim=0
    )

    has_initial_state_t = None
    if has_initial_state is not None:
        has_initial_state_t = torch.tensor(
            has_initial_state, dtype=torch.bool, device=device
        )

    core_attn_out = torch.zeros(total, nv, hv, dtype=dtype, device=device)

    return dict(
        q=q,
        k=k,
        v=v,
        b=b,
        a=a,
        A_log=A_log,
        dt_bias=dt_bias,
        ssm_state=ssm_state,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state_t,
        core_attn_out=core_attn_out,
    )


def _run_op(i, token_indx=None):
    torch.ops.sgl_kernel.gated_delta_rule(
        i["core_attn_out"],
        i["q"],
        i["k"],
        i["v"],
        i["b"],
        i["a"],
        i["A_log"],
        i["dt_bias"],
        i["ssm_state"],
        i["query_start_loc"],
        token_indx,
        i["cache_indices"],
        i["has_initial_state"],
        # num_prefills/num_decodes only affect an internal fast-path/kernel
        # selection heuristic, not correctness, per gdn::gated_delta_rule;
        # pass conservative values covering "there might be either".
        i["query_start_loc"].numel() - 1,
        0,
    )
    torch.xpu.synchronize()


@pytest.mark.parametrize(
    "seq_lens", [[1024], [37], [128, 256, 5], [1, 1, 1, 1], [513, 1, 200, 1]]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gated_delta_rule_variable_length(seq_lens, dtype):
    """Covers prefill-only, decode-only, and arbitrary mixed prefill/decode
    batches (a mix of lengths, including length-1 decode-style sequences,
    within a single call), all seeded from the committed ``ssm_state``."""
    device = torch.device("xpu")
    i = _make_inputs(seq_lens, dtype, device)
    ssm_state_before = i["ssm_state"].clone()

    _run_op(i)

    ref_out, ref_state = _reference(
        i["q"],
        i["k"],
        i["v"],
        i["b"],
        i["a"],
        i["A_log"],
        i["dt_bias"],
        ssm_state_before,
        i["query_start_loc"],
        i["cache_indices"],
    )

    rtol, atol = (3e-2, 5e-2) if dtype == torch.bfloat16 else (1e-2, 1e-2)
    torch.testing.assert_close(
        i["core_attn_out"].float(), ref_out, rtol=rtol, atol=atol
    )

    # ssm_state is the COMMITTED pool here (unlike gated_delta_rule_verify):
    # it must be updated in place at cache_indices rows to the new state.
    got_state = torch.stack(
        [i["ssm_state"][int(row)].float() for row in i["cache_indices"]]
    )
    exp_state = torch.stack([ref_state[int(row)] for row in i["cache_indices"]])
    torch.testing.assert_close(got_state, exp_state, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gated_delta_rule_has_initial_state_false(dtype):
    """Sequences with ``has_initial_state=False`` must start from a zero
    state, ignoring whatever garbage is currently at their ``ssm_state``
    row (as opposed to seeding from it)."""
    device = torch.device("xpu")
    seq_lens = [64, 1, 33, 1]
    i = _make_inputs(
        seq_lens, dtype, device, has_initial_state=[True, False, False, True]
    )
    ssm_state_before = i["ssm_state"].clone()

    _run_op(i)

    ref_out, ref_state = _reference(
        i["q"],
        i["k"],
        i["v"],
        i["b"],
        i["a"],
        i["A_log"],
        i["dt_bias"],
        ssm_state_before,
        i["query_start_loc"],
        i["cache_indices"],
        has_initial_state=i["has_initial_state"],
    )

    rtol, atol = (3e-2, 5e-2) if dtype == torch.bfloat16 else (1e-2, 1e-2)
    torch.testing.assert_close(
        i["core_attn_out"].float(), ref_out, rtol=rtol, atol=atol
    )

    got_state = torch.stack(
        [i["ssm_state"][int(row)].float() for row in i["cache_indices"]]
    )
    exp_state = torch.stack([ref_state[int(row)] for row in i["cache_indices"]])
    torch.testing.assert_close(got_state, exp_state, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_gated_delta_rule_token_indx_remap(dtype):
    """``core_attn_out`` may be laid out by a global token order different
    from the local (batch-packed) order; ``token_indx`` remaps writes."""
    device = torch.device("xpu")
    seq_lens = [3, 5, 2]
    i = _make_inputs(seq_lens, dtype, device)
    total = sum(seq_lens)

    token_indx = torch.arange(total - 1, -1, -1, dtype=torch.int32, device=device)
    core_attn_out = torch.zeros_like(i["core_attn_out"])
    i["core_attn_out"] = core_attn_out
    ssm_state_before = i["ssm_state"].clone()

    _run_op(i, token_indx=token_indx)

    ref_out, _ = _reference(
        i["q"],
        i["k"],
        i["v"],
        i["b"],
        i["a"],
        i["A_log"],
        i["dt_bias"],
        ssm_state_before,
        i["query_start_loc"],
        i["cache_indices"],
    )
    rtol, atol = 3e-2, 5e-2
    torch.testing.assert_close(
        core_attn_out.float(), ref_out.flip(0), rtol=rtol, atol=atol
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_gated_delta_rule_decode_matches_verify(dtype):
    """Plain single-token decode (via ``gated_delta_rule``) and a
    single-step ``gated_delta_rule_verify`` call, seeded from the same
    committed state, must produce identical output and next-state -- both
    implement exactly the same one-token recurrence step, just via
    different commit conventions."""
    device = torch.device("xpu")
    batch_size = 4
    i = _make_inputs([1] * batch_size, dtype, device)
    ssm_state_before = i["ssm_state"].clone()

    _run_op(i)

    # gated_delta_rule_verify with num_spec_tokens=1: functionally the same
    # single-step recurrence, seeded from the same ssm_state rows.
    scratch_rows = max(64, batch_size * 2)
    intermediate_states_buffer = torch.zeros(
        scratch_rows,
        1,
        NUM_V_HEADS,
        HEAD_V_DIM,
        HEAD_K_DIM,
        dtype=dtype,
        device=device,
    )
    intermediate_state_indices = torch.arange(
        batch_size, dtype=torch.int32, device=device
    )
    verify_out = torch.zeros_like(i["core_attn_out"])
    torch.ops.sgl_kernel.gated_delta_rule_verify(
        verify_out,
        i["q"],
        i["k"],
        i["v"],
        i["b"],
        i["a"],
        i["A_log"],
        i["dt_bias"],
        ssm_state_before,
        i["cache_indices"],
        intermediate_states_buffer,
        intermediate_state_indices,
        1,
        None,
        None,
    )
    torch.xpu.synchronize()

    rtol, atol = 3e-2, 5e-2
    torch.testing.assert_close(
        i["core_attn_out"].float(), verify_out.float(), rtol=rtol, atol=atol
    )
    got_state = torch.stack(
        [i["ssm_state"][int(row)].float() for row in i["cache_indices"]]
    )
    got_snapshot = torch.stack(
        [
            intermediate_states_buffer[int(row), 0].float()
            for row in intermediate_state_indices
        ]
    )
    torch.testing.assert_close(got_state, got_snapshot, rtol=rtol, atol=atol)
