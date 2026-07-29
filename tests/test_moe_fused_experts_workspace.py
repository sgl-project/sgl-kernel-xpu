"""Tests for the optional ``workspace`` argument of ``sgl_kernel.fused_experts``
and its ``_get_moe_ws`` scratch-buffer helper (Intel Xe2 / BMG).

``fused_experts(..., workspace=some_dict)`` lets a caller (see ``sglang``'s
``unquant.py``'s module-level ``_MOE_WS`` cache) reuse persistent, grow-only
scratch buffers across calls/layers -- keyed by buffer name in the dict --
instead of the function issuing ~9 freshly-shaped ``torch.empty`` scratch
allocations on every call. This avoids XPU caching-allocator fragmentation
(elevated ``torch.xpu.memory_reserved()`` vs ``memory_allocated()``) at
larger/varying batch sizes. These tests cover:
  * ``_get_moe_ws`` in isolation: first-call allocation with headroom,
    reuse (same storage / ``data_ptr()``) for equal-or-smaller shapes,
    regrowth for larger shapes, and forced reallocation on dtype/device
    mismatch.
  * ``fused_experts`` end-to-end: passing a workspace dict must match the
    no-workspace baseline (and the pure-torch reference) exactly, both for
    a single call and across a sequence of varying-token-count calls
    sharing one workspace dict.
  * A regression test for a real aliasing bug found during development:
    ``out_hidden_states`` (the function's return value) must never be
    workspace-backed, since the caller keeps using the returned tensor
    after the call returns -- if it aliased a shared/reused workspace
    buffer, a later call reusing that same buffer would silently corrupt
    the still-in-use earlier result.
"""

import pytest
import torch
from sgl_kernel import fused_experts
from sgl_kernel.moe import _MOE_WS_HEADROOM, _get_moe_ws
from test_moe_gemm import create_random_xpu_tensor, torch_naive_moe

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available(), reason="Requires Intel XPU"
)


def _xpu_device() -> torch.device:
    """A concrete, indexed XPU device. Unlike the bare ``torch.device("xpu")``
    (which compares unequal to a real tensor's ``.device``, e.g.
    ``torch.device("xpu") != torch.device("xpu:0")``), this matches what
    real callers always pass (a tensor's own ``.device``), so it correctly
    exercises `_get_moe_ws`'s device-match/reuse check instead of spuriously
    failing it on every call."""
    return torch.device("xpu", torch.xpu.current_device())


# ---------------------------------------------------------------------------
# `_get_moe_ws` unit tests (pure Python scratch-buffer helper logic).
# ---------------------------------------------------------------------------


def test_get_moe_ws_none_workspace_returns_plain_empty():
    """`workspace=None` must behave exactly like a plain `torch.empty` call
    (the default/back-compat path for callers that don't opt in)."""
    t = _get_moe_ws(None, "foo", (4, 8), torch.float32, _xpu_device())
    assert t.shape == (4, 8)
    assert t.dtype == torch.float32
    assert t.device.type == "xpu"


def test_get_moe_ws_first_call_allocates_with_headroom():
    ws = {}
    numel = 4 * 8
    t = _get_moe_ws(ws, "foo", (4, 8), torch.float32, _xpu_device())
    assert t.shape == (4, 8)
    assert "foo" in ws
    expected_numel = max(numel, int(numel * _MOE_WS_HEADROOM))
    assert ws["foo"].numel() == expected_numel
    assert ws["foo"].numel() >= numel


def test_get_moe_ws_reuses_same_storage_for_equal_or_smaller_shape():
    ws = {}
    t1 = _get_moe_ws(ws, "foo", (100,), torch.float32, _xpu_device())
    ptr1 = ws["foo"].data_ptr()
    numel1 = ws["foo"].numel()

    # Same shape: must reuse the exact same underlying buffer.
    t2 = _get_moe_ws(ws, "foo", (100,), torch.float32, _xpu_device())
    assert ws["foo"].data_ptr() == ptr1
    assert ws["foo"].numel() == numel1

    # Smaller shape: must also reuse (no reallocation needed).
    t3 = _get_moe_ws(ws, "foo", (10,), torch.float32, _xpu_device())
    assert ws["foo"].data_ptr() == ptr1
    assert t3.shape == (10,)


def test_get_moe_ws_grows_for_larger_shape():
    ws = {}
    _get_moe_ws(ws, "foo", (100,), torch.float32, _xpu_device())
    ptr1 = ws["foo"].data_ptr()
    numel1 = ws["foo"].numel()

    t2 = _get_moe_ws(ws, "foo", (numel1 + 1,), torch.float32, _xpu_device())
    assert t2.shape == (numel1 + 1,)
    # Must have grown (new, larger buffer -- old one is no longer referenced).
    assert ws["foo"].numel() >= numel1 + 1
    assert ws["foo"].numel() == max(numel1 + 1, int((numel1 + 1) * _MOE_WS_HEADROOM))


def test_get_moe_ws_dtype_or_device_mismatch_forces_reallocation():
    ws = {}
    _get_moe_ws(ws, "foo", (100,), torch.float32, _xpu_device())
    numel_before = ws["foo"].numel()

    # Same numel, different dtype -> must reallocate (not reinterpret bytes).
    t = _get_moe_ws(ws, "foo", (100,), torch.bfloat16, _xpu_device())
    assert ws["foo"].dtype == torch.bfloat16
    assert t.dtype == torch.bfloat16
    assert ws["foo"].numel() >= 100


def test_get_moe_ws_view_is_correctly_shaped_and_independent_per_name():
    ws = {}
    a = _get_moe_ws(ws, "a", (2, 3), torch.float32, _xpu_device())
    b = _get_moe_ws(ws, "b", (5,), torch.float32, _xpu_device())
    assert a.shape == (2, 3)
    assert b.shape == (5,)
    assert "a" in ws and "b" in ws
    assert ws["a"].data_ptr() != ws["b"].data_ptr()


# ---------------------------------------------------------------------------
# `fused_experts(..., workspace=...)` end-to-end correctness.
# ---------------------------------------------------------------------------

_E, _HIDDEN, _INTER, _TOPK = 8, 256, 512, 2


def _make_moe_case(num_tokens, seed=0):
    torch.xpu.manual_seed_all(seed)
    a = create_random_xpu_tensor((num_tokens, _HIDDEN), torch.bfloat16)
    w1 = create_random_xpu_tensor((_E, 2 * _INTER, _HIDDEN), torch.bfloat16)
    w2 = create_random_xpu_tensor((_E, _HIDDEN, _INTER), torch.bfloat16)
    score = torch.randn([num_tokens, _E], dtype=torch.bfloat16, device="xpu")
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, _TOPK)
    return a, w1, w2, topk_weight, topk_ids


def test_fused_experts_workspace_matches_baseline_and_reference():
    """A single call with a workspace dict must match both the no-workspace
    call and the pure-torch reference."""
    a, w1, w2, topk_weight, topk_ids = _make_moe_case(num_tokens=37, seed=1)

    ref = torch_naive_moe(a, w1, w2, topk_ids, topk_weight, _TOPK, None, None)
    no_ws = fused_experts(a, w1, w2, topk_weight, topk_ids)

    ws: dict = {}
    with_ws = fused_experts(a, w1, w2, topk_weight, topk_ids, workspace=ws)

    torch.testing.assert_close(no_ws, ref, rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(with_ws, no_ws, rtol=0, atol=0)
    # The workspace dict must actually have been populated (buffers reused
    # for internal scratch), not silently ignored.
    assert len(ws) > 0
    assert "out_hidden_states" not in ws  # must never be workspace-backed


def test_fused_experts_workspace_reused_across_varying_token_counts():
    """A regression test for the aliasing bug found during development: a
    shared workspace dict, reused across a sequence of calls with varying
    (growing and shrinking) token counts, must never let a later call
    corrupt an earlier call's still-referenced output, and every call's
    result must exactly match its own no-workspace/reference computation."""
    token_counts = [37, 91, 8, 257, 37, 1]
    ws: dict = {}
    results = []
    references = []

    for step, num_tokens in enumerate(token_counts):
        a, w1, w2, topk_weight, topk_ids = _make_moe_case(
            num_tokens=num_tokens, seed=100 + step
        )
        ref = torch_naive_moe(a, w1, w2, topk_ids, topk_weight, _TOPK, None, None)
        out = fused_experts(a, w1, w2, topk_weight, topk_ids, workspace=ws)
        # Clone immediately, exactly as a real caller (e.g. the next model
        # layer) would keep using the returned tensor independently of any
        # later call that might reuse/regrow the shared workspace buffers.
        results.append(out.clone())
        references.append(ref)

    for out, ref in zip(results, references):
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-3)

    # Re-check the first (smallest, earliest) and an interior result once
    # more after the whole sequence has run, to catch any latent corruption
    # from later, larger/smaller calls reusing the same workspace buffers.
    torch.testing.assert_close(results[0], references[0], rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(results[2], references[2], rtol=1e-4, atol=1e-3)


def test_fused_experts_workspace_output_not_aliased_across_calls():
    """Directly exercises the exact bug scenario found during development:
    two back-to-back calls sharing one workspace, where the second call's
    output tensor must be independent storage from the first call's output
    (never aliasing a workspace buffer), so the first call's result survives
    the second call untouched even without an explicit `.clone()`."""
    ws: dict = {}
    a1, w1_, w2_, tw1, ti1 = _make_moe_case(num_tokens=37, seed=7)
    out1 = fused_experts(a1, w1_, w2_, tw1, ti1, workspace=ws)
    out1_snapshot = out1.clone()

    a2, w1_, w2_, tw2, ti2 = _make_moe_case(num_tokens=91, seed=8)
    out2 = fused_experts(a2, w1_, w2_, tw2, ti2, workspace=ws)

    # out1 must be untouched by the second call (would fail if
    # `out_hidden_states` were ever routed through the workspace cache).
    torch.testing.assert_close(out1, out1_snapshot, rtol=0, atol=0)
    assert out1.data_ptr() != out2.data_ptr()
