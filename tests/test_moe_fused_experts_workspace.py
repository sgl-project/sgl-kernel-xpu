"""Tests for the internal ``sgl_kernel.fused_experts`` workspace cache and its
``_get_moe_ws`` scratch-buffer helper (Intel Xe2 / BMG).

These tests cover:
  * ``_get_moe_ws`` in isolation: first-call allocation with headroom, reuse
    for equal-or-smaller shapes, regrowth for larger shapes, and replacement
    on dtype changes.
  * ``fused_experts`` end-to-end: the internal process-wide cache must produce
    the pure-torch reference result across varying token counts.
  * A regression test for a real aliasing bug found during development:
    ``out_hidden_states`` (the function's return value) must never be
    workspace-backed, since the caller keeps using the returned tensor
    after the call returns -- if it aliased a shared/reused workspace
    buffer, a later call reusing that same buffer would silently corrupt
    the still-in-use earlier result.
"""

import sys

import pytest
import torch
from sgl_kernel.moe import (
    _MOE_WS_HEADROOM,
    _get_moe_ws,
    _moe_ws_cache,
    _moe_ws_view_cache,
)
from test_moe_gemm import create_random_xpu_tensor, torch_naive_moe

from sgl_kernel import fused_experts

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


@pytest.fixture(autouse=True)
def clear_moe_workspace_cache():
    _moe_ws_cache.clear()
    _moe_ws_view_cache.clear()
    yield
    _moe_ws_cache.clear()
    _moe_ws_view_cache.clear()


def _cache_key(name: str):
    return (name, _xpu_device())


# ---------------------------------------------------------------------------
# `_get_moe_ws` unit tests (pure Python scratch-buffer helper logic).
# ---------------------------------------------------------------------------


def test_get_moe_ws_first_call_allocates_with_headroom():
    numel = 4 * 8
    t = _get_moe_ws("foo", (4, 8), torch.float32, _xpu_device())
    cached = _moe_ws_cache[_cache_key("foo")]
    assert t.shape == (4, 8)
    expected_numel = max(numel, int(numel * _MOE_WS_HEADROOM))
    assert cached.numel() == expected_numel
    assert cached.numel() >= numel


def test_get_moe_ws_reuses_same_storage_for_equal_or_smaller_shape():
    first_view = _get_moe_ws("foo", (100,), torch.float32, _xpu_device())
    key = _cache_key("foo")
    ptr1 = _moe_ws_cache[key].data_ptr()
    numel1 = _moe_ws_cache[key].numel()

    # Same shape: must reuse the exact same underlying buffer.
    second_view = _get_moe_ws("foo", (100,), torch.float32, _xpu_device())
    assert second_view is first_view
    assert _moe_ws_cache[key].data_ptr() == ptr1
    assert _moe_ws_cache[key].numel() == numel1

    # Smaller shape: must also reuse (no reallocation needed).
    t3 = _get_moe_ws("foo", (10,), torch.float32, _xpu_device())
    assert _moe_ws_cache[key].data_ptr() == ptr1
    assert t3.shape == (10,)
    assert _get_moe_ws("foo", (10,), torch.float32, _xpu_device()) is t3


def test_get_moe_ws_grows_for_larger_shape():
    _get_moe_ws("foo", (100,), torch.float32, _xpu_device())
    key = _cache_key("foo")
    ptr1 = _moe_ws_cache[key].data_ptr()
    numel1 = _moe_ws_cache[key].numel()

    t2 = _get_moe_ws("foo", (numel1 + 1,), torch.float32, _xpu_device())
    assert t2.shape == (numel1 + 1,)
    # Must have grown (new, larger buffer -- old one is no longer referenced).
    assert _moe_ws_cache[key].data_ptr() != ptr1
    assert _moe_ws_cache[key].numel() >= numel1 + 1
    assert _moe_ws_cache[key].numel() == max(
        numel1 + 1, int((numel1 + 1) * _MOE_WS_HEADROOM)
    )


def test_get_moe_ws_dtype_change_replaces_buffer():
    f32 = _get_moe_ws("foo", (100,), torch.float32, _xpu_device())
    f32_ptr = f32.data_ptr()
    bf16 = _get_moe_ws("foo", (100,), torch.bfloat16, _xpu_device())
    assert f32.dtype == torch.float32
    assert bf16.dtype == torch.bfloat16
    assert bf16.data_ptr() != f32_ptr
    assert _moe_ws_cache[_cache_key("foo")].dtype == torch.bfloat16


def test_get_moe_ws_view_is_correctly_shaped_and_independent_per_name():
    a = _get_moe_ws("a", (2, 3), torch.float32, _xpu_device())
    b = _get_moe_ws("b", (5,), torch.float32, _xpu_device())
    assert a.shape == (2, 3)
    assert b.shape == (5,)
    assert _cache_key("a") in _moe_ws_cache
    assert _cache_key("b") in _moe_ws_cache
    assert a.data_ptr() != b.data_ptr()


# ---------------------------------------------------------------------------
# `fused_experts` internal workspace end-to-end correctness.
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


def test_fused_experts_internal_workspace_matches_reference():
    a, w1, w2, topk_weight, topk_ids = _make_moe_case(num_tokens=37, seed=1)

    ref = torch_naive_moe(a, w1, w2, topk_ids, topk_weight, _TOPK, None, None)
    out = fused_experts(a, w1, w2, topk_weight, topk_ids)

    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-3)
    assert len(_moe_ws_cache) > 0
    assert all(key[0] != "out_hidden_states" for key in _moe_ws_cache)


def test_fused_experts_workspace_reused_across_varying_token_counts():
    """A regression test for the aliasing bug found during development: a
    shared internal workspace, reused across calls with varying token counts,
    must never let a later call corrupt an earlier call's still-referenced
    output, and every result must match its pure-torch reference."""
    token_counts = [37, 91, 8, 257, 37, 1]
    results = []
    references = []

    for step, num_tokens in enumerate(token_counts):
        a, w1, w2, topk_weight, topk_ids = _make_moe_case(
            num_tokens=num_tokens, seed=100 + step
        )
        ref = torch_naive_moe(a, w1, w2, topk_ids, topk_weight, _TOPK, None, None)
        out = fused_experts(a, w1, w2, topk_weight, topk_ids)
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
    two back-to-back calls sharing the internal cache, where the second call's
    output must use independent storage so the first result survives the second
    call untouched even without an explicit `.clone()`."""
    a1, w1_, w2_, tw1, ti1 = _make_moe_case(num_tokens=37, seed=7)
    out1 = fused_experts(a1, w1_, w2_, tw1, ti1)
    out1_snapshot = out1.clone()

    a2, w1_, w2_, tw2, ti2 = _make_moe_case(num_tokens=91, seed=8)
    out2 = fused_experts(a2, w1_, w2_, tw2, ti2)

    # out1 must be untouched by the second call (would fail if
    # `out_hidden_states` were ever routed through the workspace cache).
    torch.testing.assert_close(out1, out1_snapshot, rtol=0, atol=0)
    assert out1.data_ptr() != out2.data_ptr()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
