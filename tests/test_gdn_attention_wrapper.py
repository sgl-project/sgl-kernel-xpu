"""Tests for the ``sgl_kernel.gdn_attention`` Python wrapper's automatic,
grow-only workspace management (as opposed to ``test_gdn_attention_workspace.py``,
which exercises the raw ``torch.ops.sgl_kernel.gdn_attention`` op's
``workspace`` argument directly).

The wrapper is expected to transparently manage a persistent, grow-only
``torch.uint8`` scratch buffer per device -- callers should get identical
results to the raw op with ``workspace=None``, without having to allocate or
track anything themselves.
"""

import sys

import pytest
import sgl_kernel
import sgl_kernel.gdn_attn as gdn_attn_mod
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


@pytest.fixture(autouse=True)
def _clear_gdn_ws_cache():
    """Each test starts from a clean slate so growth/reuse behavior is
    deterministic and tests don't leak state into one another."""
    gdn_attn_mod._gdn_ws_cache.clear()
    yield
    gdn_attn_mod._gdn_ws_cache.clear()


def _run_wrapper(i, conv_state, ssm_state, state_idx, reorder_input):
    gdn_attn_mod.gdn_attention(
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
    )
    torch.xpu.synchronize()


def _run_op_no_ws(i, conv_state, ssm_state, state_idx, reorder_input):
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
        None,
    )
    torch.xpu.synchronize()


def _assert_matches(candidate, baseline, conv_cand, conv_ref, ssm_cand, ssm_ref):
    # Small tolerance for harmless GPU floating-point non-associativity (see
    # test_gdn_attention_workspace.py for the same rationale).
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
def test_wrapper_matches_no_workspace_baseline(mode, batch_size, seqlen, dtype):
    """A single call through the wrapper (auto-managed workspace) must
    produce the same results as the raw op with `workspace=None`."""
    device = torch.device("xpu")

    baseline = _make_inputs(mode, batch_size, seqlen, dtype, device)
    conv_ref = baseline["conv_state"].clone()
    ssm_ref = baseline["ssm_state"].clone()
    _run_op_no_ws(baseline, conv_ref, ssm_ref, baseline["state_idx"], False)

    candidate = _make_inputs(mode, batch_size, seqlen, dtype, device)
    conv_cand = candidate["conv_state"].clone()
    ssm_cand = candidate["ssm_state"].clone()
    _run_wrapper(candidate, conv_cand, ssm_cand, candidate["state_idx"], False)

    _assert_matches(candidate, baseline, conv_cand, conv_ref, ssm_cand, ssm_ref)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_wrapper_auto_manages_across_varying_shapes(dtype):
    """Calling the wrapper repeatedly with varying decode/prefill shapes
    (growing then shrinking) must be safe and always match a fresh
    no-workspace baseline of the same shape -- with no caller-side workspace
    bookkeeping at all."""
    device = torch.device("xpu")
    sequence = [
        ("decode", 1, 1),
        ("prefill", 2, 128),
        ("decode", 4, 1),
        ("prefill", 1, 256),  # grows the cached buffer again
        ("decode", 1, 1),  # shrink back down after the largest prefill call
    ]

    for mode, batch_size, seqlen in sequence:
        baseline = _make_inputs(mode, batch_size, seqlen, dtype, device)
        conv_ref = baseline["conv_state"].clone()
        ssm_ref = baseline["ssm_state"].clone()
        _run_op_no_ws(baseline, conv_ref, ssm_ref, baseline["state_idx"], False)

        candidate = _make_inputs(mode, batch_size, seqlen, dtype, device)
        conv_cand = candidate["conv_state"].clone()
        ssm_cand = candidate["ssm_state"].clone()
        _run_wrapper(candidate, conv_cand, ssm_cand, candidate["state_idx"], False)

        _assert_matches(candidate, baseline, conv_cand, conv_ref, ssm_cand, ssm_ref)


def test_wrapper_workspace_is_grow_only_and_cached_per_device():
    """The cached buffer must only grow (never shrink) across calls, and be
    reused (same underlying storage) whenever a call doesn't need more
    capacity than what's already cached."""
    dtype = torch.bfloat16

    small = _make_inputs("decode", 1, 1, dtype, torch.device("xpu"))
    device = small["core_attn_out"].device  # concrete indexed device, e.g. xpu:0
    _run_wrapper(
        small,
        small["conv_state"].clone(),
        small["ssm_state"].clone(),
        small["state_idx"],
        False,
    )
    assert device in gdn_attn_mod._gdn_ws_cache
    buf_after_small = gdn_attn_mod._gdn_ws_cache[device]
    small_numel = buf_after_small.numel()

    large = _make_inputs("prefill", 2, 128, dtype, device)
    _run_wrapper(
        large,
        large["conv_state"].clone(),
        large["ssm_state"].clone(),
        large["state_idx"],
        False,
    )
    buf_after_large = gdn_attn_mod._gdn_ws_cache[device]
    assert buf_after_large.numel() >= small_numel
    assert buf_after_large.numel() > small_numel  # the prefill call is much larger

    # A smaller call afterwards must reuse the same (already large enough)
    # underlying storage rather than shrinking/reallocating.
    tiny = _make_inputs("decode", 1, 1, dtype, device)
    _run_wrapper(
        tiny,
        tiny["conv_state"].clone(),
        tiny["ssm_state"].clone(),
        tiny["state_idx"],
        False,
    )
    buf_after_tiny = gdn_attn_mod._gdn_ws_cache[device]
    assert buf_after_tiny.data_ptr() == buf_after_large.data_ptr()
    assert buf_after_tiny.numel() == buf_after_large.numel()


def test_wrapper_explicit_workspace_bypasses_auto_management():
    """The wrapper always self-manages its own workspace and does not accept
    a caller-provided one (unlike the raw ``torch.ops.sgl_kernel.gdn_attention``
    op, whose ``workspace`` argument the wrapper itself supplies internally).
    Passing an extra positional argument must therefore fail with a
    ``TypeError``, confirming the wrapper's signature has no such
    escape hatch."""
    device = torch.device("xpu")
    dtype = torch.bfloat16

    candidate = _make_inputs("decode", 1, 1, dtype, device)
    conv_cand = candidate["conv_state"].clone()
    ssm_cand = candidate["ssm_state"].clone()
    explicit_ws = torch.empty(1 << 20, dtype=torch.uint8, device=device)

    with pytest.raises(TypeError):
        gdn_attn_mod.gdn_attention(
            candidate["core_attn_out"],
            candidate["z"],
            candidate["qkvz"],
            candidate["ba"],
            NUM_K_HEADS,
            NUM_V_HEADS,
            HEAD_K_DIM,
            HEAD_V_DIM,
            conv_cand,
            ssm_cand,
            candidate["conv_w"],
            candidate["conv_b"],
            "silu",
            candidate["A_log"],
            candidate["dt_bias"],
            candidate["num_prefills"],
            candidate["num_decodes"],
            0,
            candidate["has_init"],
            candidate["qsl"],
            None,
            candidate["state_idx"],
            None,
            None,
            None,
            None,
            candidate["num_actual"],
            TP_SIZE,
            False,
            explicit_ws,
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
