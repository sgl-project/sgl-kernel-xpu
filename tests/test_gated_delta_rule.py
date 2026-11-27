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


@pytest.mark.parametrize(
    "length,seeded",
    [(2048, None), (2048, True), (2053, True), (8192, False)],
    ids=["seed-off", "seed-on", "ragged-seed-on", "long-seed-off"],
)
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
    # In addition, the (chunk-path) result is validated against the
    # INDEPENDENT fp32 reference on both output and final state: a
    # chunk-vs-sequential comparison alone can share a common-mode error
    # between the two bf16 kernels, and historical regressions (a
    # write-after-read on the committed state, visible only with
    # has_initial_state=True and a large max-diff of O(0.1-0.2) vs the
    # sequential kernel's O(5e-4)) are caught directly here.
    _run_op(expected, token_indx=identity)
    ssm_state_before = inputs["ssm_state"].clone()
    _run_op(inputs)
    torch.testing.assert_close(
        inputs["core_attn_out"], expected["core_attn_out"], rtol=2e-2, atol=5e-3
    )
    torch.testing.assert_close(
        inputs["ssm_state"], expected["ssm_state"], rtol=2e-2, atol=1e-2
    )
    ref_out, ref_state = _reference(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["b"],
        inputs["a"],
        inputs["A_log"],
        inputs["dt_bias"],
        ssm_state_before,
        inputs["query_start_loc"],
        inputs["cache_indices"],
        has_initial_state=inputs["has_initial_state"],
    )
    torch.testing.assert_close(
        inputs["core_attn_out"].float(), ref_out, rtol=2e-2, atol=1e-2
    )
    got_state = torch.stack(
        [inputs["ssm_state"][int(row)].float() for row in inputs["cache_indices"]]
    )
    exp_state = torch.stack([ref_state[int(row)] for row in inputs["cache_indices"]])
    torch.testing.assert_close(got_state, exp_state, rtol=2e-2, atol=1e-2)
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


@pytest.mark.arch("xe20")
@pytest.mark.parametrize(
    "seq_lens,expect_chunk",
    [
        ([256, 255], False),
        ([256, 256], True),
    ],
    ids=["multi-total-511-sequential", "multi-total-512-chunk"],
)
def test_multiseq_dispatch_boundary_selects_expected_kernel(
    seq_lens, expect_chunk, tmp_path
):
    """Multi-sequence counterpart of test_dispatch_boundary_selects_expected_kernel:
    the token gate is on TOTAL tokens across sequences, so two sequences whose
    totals straddle gdn::kChunkMinTokens must dispatch to (and only to) the
    expected kernel. Guards that the multi-sequence rollout kept the same
    boundary it had as a single sequence, and that the per-sequence
    multi-length predicate (num_prefills == numel(query_start_loc) - 1)
    agrees with the total-length gate."""
    inputs = _make_inputs(seq_lens, torch.bfloat16, torch.device("xpu"), nk=4, nv=12)
    inputs["ssm_state"] = inputs["ssm_state"].float()
    _run_op(inputs)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.XPU,
        ]
    ) as profiler:
        _run_op(inputs)
    path = tmp_path / "multiseq_boundary.json"
    profiler.export_chrome_trace(str(path))
    kernels = {
        event["name"]
        for event in json.loads(path.read_text())["traceEvents"]
        if event.get("cat") == "kernel"
    }
    has_chunk = any("ChunkFwdOKernel" in name for name in kernels)
    has_seq = any("gated_delta_rule_kernel<" in name for name in kernels)
    if expect_chunk:
        assert has_chunk, f"expected chunk kernel for {seq_lens}: {kernels}"
        assert not has_seq, f"unexpected sequential kernel for {seq_lens}: {kernels}"
    else:
        assert not has_chunk, f"unexpected chunk kernel for {seq_lens}: {kernels}"
        assert has_seq, f"expected sequential kernel for {seq_lens}: {kernels}"


@pytest.mark.parametrize(
    "dtype, state_dtype, strided_qkv, length",
    [
        (
            torch.float16,
            torch.float32,
            False,
            1024,
        ),  # fp16 activations, chunk-eligible length
        (
            torch.bfloat16,
            torch.bfloat16,
            False,
            2048,
        ),  # bf16 state (fp32 state required by the chunk path)
        (torch.bfloat16, torch.float32, True, 700),  # strided q/k/v, chunk-routed
        (torch.bfloat16, torch.float32, True, 255),  # strided q/k/v, sequential-routed
    ],
    ids=[
        "fp16_activation",
        "bf16_state",
        "strided_qkv_chunk_len",
        "strided_qkv_seq_len",
    ],
)
def test_non_selected_dispatch_stays_correct(dtype, state_dtype, strided_qkv, length):
    """The chunk fast path is a PERFORMANCE choice restricted to
    (bf16 activation, fp32 state, pure-prefill, no remap). Inputs that match
    the SHAPE gate but violate exactly one other dimension of the predicate --
    fp16 activations, a bf16 state (both at chunk-eligible token counts), and
    strided q/k/v (which the fast path's stride-aware scatter supports, while
    the sequential kernel needs the re-fixup from review item #2) -- must keep
    producing correct results on the path they are actually routed to. These
    are the negative selections that previously had no coverage: if someone
    widens or drops a gate without understanding the fallback, one of these
    will break first."""
    device = torch.device("xpu")
    i = _make_inputs([length], dtype, device, nk=4, nv=12)
    i["ssm_state"] = i["ssm_state"].to(state_dtype)
    if strided_qkv:
        # SGLang's real layout: q/k/v are head-slices of one packed projection.
        packed = torch.cat((i["q"], i["k"], i["v"]), dim=1)
        nk, nv, hk, hv = i["q"].size(1), i["v"].size(1), i["q"].size(2), i["v"].size(2)
        i["q"] = packed[:, 0:nk]
        i["k"] = packed[:, nk : 2 * nk]
        i["v"] = packed[:, 2 * nk : 2 * nk + nv]
        assert not (
            i["q"].is_contiguous() and i["k"].is_contiguous() and i["v"].is_contiguous()
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
    )
    # fp32 activations: ~bf16-equivalent precision; fp16/bf16-state: the
    # pre-existing loose pair for reduced-precision head products.
    if dtype == torch.bfloat16 and state_dtype == torch.float32:
        rtol, atol = 2e-2, 1e-2
    elif dtype == torch.float16:
        rtol, atol = 1e-2, 1e-2
    else:
        rtol, atol = 3e-2, 5e-2
    torch.testing.assert_close(
        i["core_attn_out"].float(), ref_out, rtol=rtol, atol=atol
    )
    got_state = torch.stack(
        [i["ssm_state"][int(row)].float() for row in i["cache_indices"]]
    )
    exp_state = torch.stack([ref_state[int(row)] for row in i["cache_indices"]])
    torch.testing.assert_close(got_state, exp_state, rtol=rtol, atol=atol)


def test_sequential_strided_qkv_bit_exact():
    """On the SEQUENTIAL path, a strided q/k/v (head-slices of a packed
    projection) must match the contiguous copy BIT-EXACTLY:
    ``.contiguous()`` copies the identical bf16 values, so the (pure) kernel
    products are unchanged. This is the exact scenario the pre-PR sequential
    kernel got ~60% off (the reviewer's finding), and any layout-dependent
    non-determinism in the copy would violate this."""
    device = torch.device("xpu")
    i = _make_inputs([255], torch.bfloat16, device, nk=4, nv=12)
    i["ssm_state"] = i["ssm_state"].float()
    packed = torch.cat((i["q"], i["k"], i["v"]), dim=1)
    nk, nv, _, _ = i["q"].size(1), i["v"].size(1), i["q"].size(2), i["v"].size(2)
    i["q"] = packed[:, 0:nk]
    i["k"] = packed[:, nk : 2 * nk]
    i["v"] = packed[:, 2 * nk : 2 * nk + nv]
    expected = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in i.items()
    }

    _run_op(i)
    _run_op(expected)
    torch.testing.assert_close(
        i["core_attn_out"], expected["core_attn_out"], rtol=0, atol=0
    )
    torch.testing.assert_close(i["ssm_state"], expected["ssm_state"], rtol=0, atol=0)


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
    [[64, 447], [64, 448], [300, 300], [2048, 5]],
    ids=[
        "total-511-sequential",
        "total-512-chunk",
        "total-600-chunk",
        "total-2053-ragged-chunk",
    ],
)
def test_multiseq_long_total_stays_correct(seq_lens):
    """A MULTI-sequence prefill batch -- on BOTH sides of the chunk threshold,
    ragged or balanced -- must produce correct output and final state against
    the independent fp32 reference. The chunk fast path handles any number of
    sequences now, so these dispatch on TOTAL token count: totals >=
    gdn::kChunkMinTokens run the (padded, per-sequence) chunk path with
    per-sequence state reads/writes, totals below it run the sequential
    kernel. The ragged [2048, 5] case exercises a sequence shorter than ONE
    chunk; [2048, 5] also spans the old 2048 single-seq cutoff to guard the
    migration. (Which kernel runs is pinned by
    test_dispatch_boundary_selects_expected_kernel and its multi-sequence
    sibling, which also use the profiler.)"""
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


@pytest.mark.parametrize(
    "seq_lens,has_initial_state",
    [
        ([300, 300], [True, False]),
        ([667, 510], [False, True]),
        ([1024, 5], [True, True]),
    ],
    ids=["seq2-unseeded", "seq1-unseeded-ragged", "extreme-ragged"],
)
def test_multiseq_chunk_seeded_state_matches_fp32_reference(
    seq_lens, has_initial_state
):
    """Seeded-state correctness for the MULTI-SEQUENCE chunk path, which
    previously carried a write-after-read race on ``ssm_state`` (the output
    state-READ of chunk c raced the state-WRITE of chunk c-1+ ... concretely:
    the W@S/Q@S phase of a chunk read from ``ssm_state`` while the state-update
    of the same chunk already stored the new state, so with
    ``has_initial_state=True`` the read occasionally observed the just-written
    state instead of the seed, amplifying forward). All totals here are >=
    gdn::kChunkMinTokens so the calls take the chunk path (verified by the
    boundary tests), lengths are ragged so sequences contain internal chunk
    boundaries, and per-sequence masks exercise every (seeded, unseeded)
    combination per sequence. Both OUTPUT and FINAL STATE are validated
    against the independent fp32 reference (not only against the bf16
    sequential kernel, which could share a common-mode error)."""
    device = torch.device("xpu")
    i = _make_inputs(
        seq_lens,
        torch.bfloat16,
        device,
        nk=4,
        nv=12,
        has_initial_state=has_initial_state,
    )
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
    "seq_lens,caller_num_prefills",
    [
        ([2048, 2048], 1),  # two sequences, caller claims one (original finding)
        ([2048, 2048], 3),  # two sequences, caller claims three
    ],
    ids=["scalar_low", "scalar_high"],
)
def test_prefill_count_mismatch_is_not_corrupted(seq_lens, caller_num_prefills):
    """Regression guard for the review finding that an inconsistent
    (num_prefills, query_start_loc) pair could silently select the chunk path
    with the wrong batch count: the chunk kernels re-derive batch_size from
    query_start_loc while the private scratch / mask were sized for a
    different count, which emitted NaN in ~8% of the outputs and read
    has_initial_state out of bounds. The fast path is now gated on the scalar
    counts AND the tensor shapes agreeing on the pure-prefill batch, so every
    mismatch (in either direction) must fall back to the sequential kernel
    and stay finite. Finiteness / no-crash is asserted only: the mismatch is a
    caller-contract violation, and the sequential kernel -- not the op --
    defines its (tolerable) numeric behavior for such inputs."""
    device = torch.device("xpu")
    i = _make_inputs(seq_lens, torch.bfloat16, device, nk=4, nv=12)
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
        caller_num_prefills,
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
