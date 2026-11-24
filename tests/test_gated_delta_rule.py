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

import json

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
        # num_prefills/num_decodes are CALLER scalars that must be consistent
        # with query_start_loc: the chunk fast path (gated by gdn::kChunkMinTokens)
        # is taken only when the scalar counts AND the tensor shapes both agree
        # on a single prefill sequence. Here we pass num_prefills = (number of
        # sequences) and num_decodes = 0, which matches query_start_loc: one
        # sequence correctly enables the chunk path (when long enough), while
        # multi-sequence batches correctly stay on the sequential kernel and
        # are validated for correctness against the reference below.
        i["query_start_loc"].numel() - 1,
        0,
    )
    torch.xpu.synchronize()


@pytest.mark.parametrize("length,seeded", [(2048, None), (2053, True), (8192, False)])
@pytest.mark.parametrize("strided_gates", [False, True])
def test_long_prefill_matches_sequential_without_mutating_inputs(
    length, seeded, strided_gates
):
    inputs = _make_inputs(
        [length],
        torch.bfloat16,
        torch.device("xpu"),
        nk=4,
        nv=12,
        has_initial_state=None if seeded is None else [seeded],
    )
    inputs["ssm_state"] = inputs["ssm_state"].float()
    if strided_gates:
        packed_gates = torch.cat((inputs["b"], inputs["a"]), dim=1)
        inputs["b"], inputs["a"] = packed_gates.chunk(2, dim=1)
    expected = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    original = {key: inputs[key].clone() for key in ("q", "k", "v", "a", "b")}
    identity = torch.arange(length, device="xpu", dtype=torch.int32)
    # expected -> sequential kernel (token_indx forces it); inputs -> chunk
    # kernel (no remap, single long sequence). Both are bf16, so this is a
    # chunk-vs-sequential comparison. Tolerances are kept well above the
    # observed per-element error (~1e-3) -- an earlier (rtol=3e-2, atol=5e-2)
    # pair was ~50x that and would let a common-mode scale/offset error of a
    # few percent pass undetected -- while still comfortably below it.
    _run_op(expected, token_indx=identity)
    _run_op(inputs)
    torch.testing.assert_close(
        inputs["core_attn_out"], expected["core_attn_out"], rtol=2e-2, atol=5e-3
    )
    torch.testing.assert_close(
        inputs["ssm_state"], expected["ssm_state"], rtol=2e-2, atol=1e-2
    )
    for key, value in original.items():
        torch.testing.assert_close(inputs[key], value, rtol=0, atol=0)


@pytest.mark.arch("xe20")
def test_long_prefill_selects_chunk_kernel(tmp_path):
    inputs = _make_inputs([2048], torch.bfloat16, torch.device("xpu"), nk=4, nv=12)
    inputs["ssm_state"] = inputs["ssm_state"].float()
    _run_op(inputs)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.XPU,
        ]
    ) as profiler:
        _run_op(inputs)
    path = tmp_path / "chunk_dispatch.json"
    profiler.export_chrome_trace(str(path))
    kernels = {
        event["name"]
        for event in json.loads(path.read_text())["traceEvents"]
        if event.get("cat") == "kernel"
    }
    assert any("ChunkFwdOKernel" in name for name in kernels), kernels
    assert not any("gated_delta_rule_kernel<" in name for name in kernels), kernels


@pytest.mark.arch("xe20")
@pytest.mark.parametrize(
    "length,expect_chunk",
    [(511, False), (512, True)],
    ids=["just-below-threshold", "at-threshold"],
)
def test_dispatch_boundary_selects_expected_kernel(length, expect_chunk, tmp_path):
    """Pin the chunk/sequential dispatch BOUNDARY. A single long sequence at
    length == gdn::kChunkMinTokens - 1 must NOT run the chunk kernel; at length
    == gdn::kChunkMinTokens it must. Keep the two lengths in sync with
    gdn::kChunkMinTokens (src/sycl/gdn_attn/gdn_attn_utils.h). Catches someone
    silently widening or narrowing the fast-path guard's token gate."""
    inputs = _make_inputs([length], torch.bfloat16, torch.device("xpu"), nk=4, nv=12)
    inputs["ssm_state"] = inputs["ssm_state"].float()
    _run_op(inputs)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.XPU,
        ]
    ) as profiler:
        _run_op(inputs)
    path = tmp_path / "boundary.json"
    profiler.export_chrome_trace(str(path))
    kernels = {
        event["name"]
        for event in json.loads(path.read_text())["traceEvents"]
        if event.get("cat") == "kernel"
    }
    has_chunk = any("ChunkFwdOKernel" in name for name in kernels)
    has_seq = any("gated_delta_rule_kernel<" in name for name in kernels)
    if expect_chunk:
        assert has_chunk, f"expected chunk kernel at length {length}: {kernels}"
        assert (
            not has_seq
        ), f"unexpected sequential kernel at length {length}: {kernels}"
    else:
        assert not has_chunk, f"unexpected chunk kernel at length {length}: {kernels}"
        assert has_seq, f"expected sequential kernel at length {length}: {kernels}"


def test_long_prefill_matches_fp32_reference():
    """The chunk-selected bf16 output/state must also agree with an independent
    fp32 sequential reference, not just with the bf16 sequential kernel. A
    chunk-vs-sequential comparison (same op, same dtype) can share a
    common-mode scale/offset error between the two paths and thus miss it; an
    fp32 re-derivation cannot. Tolerances are looser than the tight
    chunk-vs-sequential gate (bf16 op vs fp32 ref) but still catch a ~2%
    common-mode error."""
    device = torch.device("xpu")
    i = _make_inputs([2048], torch.bfloat16, device, nk=4, nv=12)
    i["ssm_state"] = i["ssm_state"].float()
    ssm_state_before = i["ssm_state"].clone()
    _run_op(i)  # single long sequence -> chunk path

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
    torch.testing.assert_close(
        i["core_attn_out"].float(), ref_out, rtol=2e-2, atol=1e-2
    )
    got_state = torch.stack(
        [i["ssm_state"][int(row)].float() for row in i["cache_indices"]]
    )
    exp_state = torch.stack([ref_state[int(row)] for row in i["cache_indices"]])
    torch.testing.assert_close(got_state, exp_state, rtol=2e-2, atol=1e-2)


@pytest.mark.parametrize(
    "seq_lens",
    [[300, 300], [1024, 1024], [2048, 5]],
    ids=["total-600-below-old-above-new", "total-2048-balanced", "total-2053-ragged"],
)
def test_multiseq_long_total_stays_correct(seq_lens):
    """A MULTI-sequence batch whose TOTAL token count is at/above the chunk
    threshold must still be correct. The chunk path is single-sequence-only in
    this PR, so these calls must route to the sequential kernel and must NOT
    (a) take the single-sequence chunk path (which mis-handles >1 sequence) or
    (b) produce NaN. The totals span the newly-changed [512, 2048) band as well
    as above the old 2048 cutoff, so this guards against the widened threshold
    ever leaking the multi-sequence case into the single-sequence chunk path.
    (previously nothing exercised a >= threshold multi-sequence batch)."""
    device = torch.device("xpu")
    i = _make_inputs(seq_lens, torch.bfloat16, device, nk=4, nv=12)
    i["ssm_state"] = i["ssm_state"].float()
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
    torch.testing.assert_close(
        i["core_attn_out"].float(), ref_out, rtol=2e-2, atol=1e-2
    )
    got_state = torch.stack(
        [i["ssm_state"][int(row)].float() for row in i["cache_indices"]]
    )
    exp_state = torch.stack([ref_state[int(row)] for row in i["cache_indices"]])
    torch.testing.assert_close(got_state, exp_state, rtol=2e-2, atol=1e-2)


def test_prefill_count_mismatch_is_not_corrupted():
    """Regression guard for the review finding that num_prefills=1 with a
    query_start_loc describing MORE than one sequence used to silently select
    the single-sequence chunk path: the chunk kernel re-derives batch_size from
    query_start_loc, reads has_initial_state out of bounds, and emitted NaN in
    ~8% of the outputs. The fast path is now only taken when the scalar counts
    AND the tensor shapes agree on a single sequence, so this mismatch must
    fall back to the sequential kernel and stay finite (no NaN). We assert
    finiteness / no crash only, because the mismatch is a caller-contract
    violation and the sequential kernel defines its (tolerable) numeric
    behavior for it, not the op."""
    device = torch.device("xpu")
    # Two sequences, but the caller claims num_prefills=1.
    i = _make_inputs([2048, 2048], torch.bfloat16, device, nk=4, nv=12)
    i["ssm_state"] = i["ssm_state"].float()
    # token_indx=None, has_initial_state=None so the ONLY inconsistency is the
    # scalar-vs-tensor sequence count (the thing under test).
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
        None,
        i["cache_indices"],
        None,
        1,
        0,
    )
    torch.xpu.synchronize()
    assert torch.isfinite(
        i["core_attn_out"].float()
    ).all(), "mismatched num_prefills/query_start_loc produced non-finite outputs"
    assert torch.isfinite(
        i["ssm_state"].float()
    ).all(), "mismatched num_prefills/query_start_loc produced non-finite state"


@pytest.mark.parametrize("length,remap", [(37, False), (2048, True)])
def test_sequential_strided_gates_match_contiguous(length, remap):
    inputs = _make_inputs([length], torch.bfloat16, torch.device("xpu"), nk=4, nv=12)
    inputs["ssm_state"] = inputs["ssm_state"].float()
    packed_gates = torch.cat((inputs["b"], inputs["a"]), dim=1)
    inputs["b"], inputs["a"] = packed_gates.chunk(2, dim=1)
    assert not inputs["b"].is_contiguous() and not inputs["a"].is_contiguous()
    expected = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    assert expected["b"].is_contiguous() and expected["a"].is_contiguous()
    original_gates = packed_gates.clone()
    token_indx = (
        torch.arange(length, device="xpu", dtype=torch.int32) if remap else None
    )
    _run_op(expected, token_indx=token_indx)
    _run_op(inputs, token_indx=token_indx)
    torch.testing.assert_close(inputs["core_attn_out"], expected["core_attn_out"])
    torch.testing.assert_close(inputs["ssm_state"], expected["ssm_state"])
    torch.testing.assert_close(packed_gates, original_gates, rtol=0, atol=0)


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
