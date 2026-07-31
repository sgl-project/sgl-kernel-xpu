"""Tests for the optional ``workspace`` argument of the fused GDN attention op
(Intel Xe2 / BMG): ``torch.ops.sgl_kernel.gdn_attention(..., workspace=...)``.
These tests only check functional correctness of the op's use of the
``workspace`` argument; they don't attempt to measure/assert on actual memory usage.
"""

import sys

import pytest
import sgl_kernel  # noqa: F401  registers torch.ops.sgl_kernel.gdn_attention
import torch
from test_gdn_attention import (
    HEAD_K_DIM,
    HEAD_V_DIM,
    NUM_K_HEADS,
    NUM_V_HEADS,
    TP_SIZE,
    _make_inputs,
)

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available() or not hasattr(torch.ops.sgl_kernel, "gdn_attention"),
    reason="Requires Intel XPU build with gdn_attention op",
)

# Must match `gdn::chunk_size_xe2` in sgl-kernel-xpu's gdn_attn_utils.h --
# the prefill chunk-scan path pads non-spec tokens up to a multiple of this.
GDN_XE2_CHUNK_SIZE = 64
# Generous safety margin over the exact padded-token count so any small
# mismatch between this test's shape bookkeeping and the kernel's own
# padding formula still results in a workspace big enough to be used (as
# opposed to silently falling back to a fresh allocation, which would make
# the test pass without actually exercising the workspace path).
_SAFETY_MARGIN = 2


def _padded_tokens(mode, batch_size, num_actual):
    padding = batch_size * (GDN_XE2_CHUNK_SIZE - 1) if mode == "prefill" else 0
    return num_actual + padding


def _make_workspace(max_padded_tokens, dtype, device, nk=NUM_K_HEADS, nv=NUM_V_HEADS):
    """Allocate a single flat 1-D ``torch.uint8`` scratch buffer, generously
    sized to hold whichever set of internal scratch tensors (decode:
    q/k/v/b/a, or prefill/chunk: q/k/v/b_prefill/a_prefill -- these two sets
    are mutually exclusive within a single call) the op needs for
    `max_padded_tokens`. The op itself computes byte offsets for each
    tensor internally; callers only need to size the buffer generously
    enough (no fixed internal layout is exposed)."""
    n = max_padded_tokens * _SAFETY_MARGIN
    itemsize = torch.empty(0, dtype=dtype).element_size()
    qk_bytes = n * nk * HEAD_K_DIM * itemsize
    v_bytes = n * nv * HEAD_V_DIM * itemsize
    ba_bytes = n * nv * itemsize
    ba_prefill_bytes = n * nv * 4  # b_prefill/a_prefill are always float32
    decode_bytes = 2 * qk_bytes + v_bytes + 2 * ba_bytes
    prefill_bytes = 2 * qk_bytes + v_bytes + 2 * ba_prefill_bytes
    # Generous alignment/rounding slack per carved-out tensor (the op aligns
    # each to 256 bytes internally) plus overall safety margin.
    total_bytes = max(decode_bytes, prefill_bytes) + 5 * 256
    return torch.empty(total_bytes, dtype=torch.uint8, device=device)


def _run_op_ws(i, conv_state, ssm_state, state_idx, reorder_input, workspace):
    torch.ops.sgl_kernel.gdn_attention(
        i["core_attn_out"],
        i["z"],
        i["qkvz"],
        i["ba"],
        NUM_K_HEADS,
        NUM_V_HEADS,
        HEAD_K_DIM,
        HEAD_V_DIM,
        conv_state,
        ssm_state,
        i["conv_w"],
        i["conv_b"],
        "silu",
        i["A_log"],
        i["dt_bias"],
        i["num_prefills"],
        i["num_decodes"],
        0,
        i["has_init"],
        i["qsl"],
        None,
        state_idx,
        None,
        None,
        None,
        None,
        i["num_actual"],
        TP_SIZE,
        reorder_input,
        workspace,
    )
    torch.xpu.synchronize()


def _assert_matches(candidate, baseline, conv_cand, conv_ref, ssm_cand, ssm_ref):
    # A small tolerance is needed here to accommodate the harmless GPU floating-point
    # non-associativity. It does not prevent the test from catching a workspace buffer
    # aliasing/corruption bug, which would show up as large, widespread mismatches.
    rtol, atol = 1e-2, 2e-3
    torch.testing.assert_close(
        candidate["core_attn_out"], baseline["core_attn_out"], rtol=rtol, atol=atol
    )
    torch.testing.assert_close(candidate["z"], baseline["z"], rtol=rtol, atol=atol)
    torch.testing.assert_close(ssm_cand, ssm_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(conv_cand, conv_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "mode,batch_size,seqlen",
    [("decode", 1, 1), ("decode", 4, 1), ("prefill", 1, 256), ("prefill", 2, 128)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gdn_attention_workspace_matches_baseline(mode, batch_size, seqlen, dtype):
    """A single call with a sufficiently large workspace must produce
    (within a small floating-point tolerance) the same results as the same
    call with no workspace at all (the `workspace=None` default)."""
    device = torch.device("xpu")

    baseline = _make_inputs(mode, batch_size, seqlen, dtype, device)
    conv_ref = baseline["conv_state"].clone()
    ssm_ref = baseline["ssm_state"].clone()
    _run_op_ws(baseline, conv_ref, ssm_ref, baseline["state_idx"], False, None)

    candidate = _make_inputs(mode, batch_size, seqlen, dtype, device)
    conv_cand = candidate["conv_state"].clone()
    ssm_cand = candidate["ssm_state"].clone()
    workspace = _make_workspace(
        _padded_tokens(mode, batch_size, candidate["num_actual"]), dtype, device
    )
    _run_op_ws(candidate, conv_cand, ssm_cand, candidate["state_idx"], False, workspace)

    _assert_matches(candidate, baseline, conv_cand, conv_ref, ssm_cand, ssm_ref)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gdn_attention_workspace_reused_across_varying_shapes(dtype):
    """One workspace list, sized for the largest case in the sequence, must
    be safe to reuse unmodified across a sequence of differently-shaped
    decode/prefill calls.
    Each call is checked against a fresh no-workspace baseline of the same
    shape, so any cross-call aliasing corruption would be caught."""
    device = torch.device("xpu")
    sequence = [
        ("decode", 1, 1),
        ("prefill", 2, 128),
        ("decode", 4, 1),
        ("prefill", 1, 256),
        ("decode", 1, 1),  # shrink back down after the largest prefill call
    ]
    max_padded = max(
        _padded_tokens(mode, bs, bs if mode == "decode" else bs * seqlen)
        for mode, bs, seqlen in sequence
    )
    workspace = _make_workspace(max_padded, dtype, device)

    for mode, batch_size, seqlen in sequence:
        baseline = _make_inputs(mode, batch_size, seqlen, dtype, device)
        conv_ref = baseline["conv_state"].clone()
        ssm_ref = baseline["ssm_state"].clone()
        _run_op_ws(baseline, conv_ref, ssm_ref, baseline["state_idx"], False, None)

        candidate = _make_inputs(mode, batch_size, seqlen, dtype, device)
        conv_cand = candidate["conv_state"].clone()
        ssm_cand = candidate["ssm_state"].clone()
        _run_op_ws(
            candidate, conv_cand, ssm_cand, candidate["state_idx"], False, workspace
        )

        _assert_matches(candidate, baseline, conv_cand, conv_ref, ssm_cand, ssm_ref)


@pytest.mark.parametrize(
    "mode,batch_size,seqlen", [("decode", 4, 1), ("prefill", 2, 128)]
)
def test_gdn_attention_workspace_undersized_falls_back(mode, batch_size, seqlen):
    """A too-small workspace buffer must be silently ignored (the op falls
    back to its original per-call allocation), matching the no-workspace
    baseline exactly -- not crash or silently corrupt results."""
    device = torch.device("xpu")
    dtype = torch.bfloat16

    baseline = _make_inputs(mode, batch_size, seqlen, dtype, device)
    conv_ref = baseline["conv_state"].clone()
    ssm_ref = baseline["ssm_state"].clone()
    _run_op_ws(baseline, conv_ref, ssm_ref, baseline["state_idx"], False, None)

    candidate = _make_inputs(mode, batch_size, seqlen, dtype, device)
    conv_cand = candidate["conv_state"].clone()
    ssm_cand = candidate["ssm_state"].clone()
    # Deliberately far too small to hold even the first scratch tensor.
    undersized_ws = torch.empty(4, dtype=torch.uint8, device=device)
    _run_op_ws(
        candidate, conv_cand, ssm_cand, candidate["state_idx"], False, undersized_ws
    )

    _assert_matches(candidate, baseline, conv_cand, conv_ref, ssm_cand, ssm_ref)


@pytest.mark.parametrize(
    "mode,batch_size,seqlen", [("decode", 4, 1), ("prefill", 2, 128)]
)
def test_gdn_attention_workspace_wrong_dtype_falls_back(mode, batch_size, seqlen):
    """A workspace buffer that isn't a 1-D ``torch.uint8`` tensor must be
    silently ignored (falls back to fresh per-call allocation), not crash or
    silently corrupt results."""
    device = torch.device("xpu")
    dtype = torch.bfloat16

    baseline = _make_inputs(mode, batch_size, seqlen, dtype, device)
    conv_ref = baseline["conv_state"].clone()
    ssm_ref = baseline["ssm_state"].clone()
    _run_op_ws(baseline, conv_ref, ssm_ref, baseline["state_idx"], False, None)

    candidate = _make_inputs(mode, batch_size, seqlen, dtype, device)
    conv_cand = candidate["conv_state"].clone()
    ssm_cand = candidate["ssm_state"].clone()
    # Plenty big, but wrong dtype (float32 instead of uint8).
    wrong_dtype_ws = torch.empty(1 << 20, dtype=torch.float32, device=device)
    _run_op_ws(
        candidate, conv_cand, ssm_cand, candidate["state_idx"], False, wrong_dtype_ws
    )

    _assert_matches(candidate, baseline, conv_cand, conv_ref, ssm_cand, ssm_ref)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
