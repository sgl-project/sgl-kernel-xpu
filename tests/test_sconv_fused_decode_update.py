import pytest
import torch
from sconv_reference import assert_close, fused_decode_ref, rand

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="Inkling sconv ops are XPU-only",
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("activation", [None, "silu"])
@pytest.mark.parametrize("use_residual", [False, True])
def test_forward_decode_matches_fused_decode(dtype, activation, use_residual):
    from sgl_kernel.inkling_sconv import (
        causal_conv1d,
        fused_causal_conv1d_update_decode,
    )

    torch.manual_seed(1)
    B, D, W = 5, 8, 4
    x = rand((B, D), dtype)
    weight = rand((D, W), dtype, scale=0.2)
    cache = rand((B, W - 1, D), dtype, scale=0.1)
    cache_mask = torch.ones(B, dtype=torch.bool, device="xpu")
    cache_indices = torch.arange(B, dtype=torch.int32, device="xpu")
    safe_idx = cache_indices.to(torch.int64)
    cu = torch.arange(B + 1, dtype=torch.int64, device="xpu")
    si = torch.arange(B, dtype=torch.int32, device="xpu")

    forward = causal_conv1d(
        x,
        weight,
        cache,
        cache_mask.reshape(B, 1, 1),
        safe_idx,
        cu,
        si,
        activation=activation,
        use_residual=use_residual,
        is_decode=True,
    )
    fused_cache = cache.clone()
    fused = fused_causal_conv1d_update_decode(
        x,
        weight,
        fused_cache,
        cache_indices,
        cache_mask,
        activation=activation,
        use_residual=use_residual,
    )
    assert_close(forward, fused.detach().cpu().float(), dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("W,D", [(3, 7), (4, 8)])
@pytest.mark.parametrize("activation", [None, "silu"])
@pytest.mark.parametrize("use_residual", [False, True])
def test_fused_decode_update_matches_inkling_pr_semantics(
    dtype, W, D, activation, use_residual
):
    from sgl_kernel.inkling_sconv import fused_causal_conv1d_update_decode

    torch.manual_seed(3)
    T = 5
    x = rand((T, D), dtype)
    weight = rand((D, W), dtype, scale=0.2)
    cache = rand((9, W - 1, D), dtype, scale=0.1)
    cache_indices = torch.tensor([0, 1, -1, 3, 4], dtype=torch.int32, device="xpu")
    cache_mask = torch.tensor(
        [True, False, False, True, True], dtype=torch.bool, device="xpu"
    )
    track_mask = torch.tensor(
        [True, True, False, False, True], dtype=torch.bool, device="xpu"
    )
    track_indices = torch.tensor([6, 7, 8, 8, 5], dtype=torch.int64, device="xpu")
    expected_y, expected_cache = fused_decode_ref(
        x,
        weight,
        cache,
        cache_indices,
        cache_mask,
        activation=activation,
        use_residual=use_residual,
        track_mask=track_mask,
        track_indices=track_indices,
    )

    actual = fused_causal_conv1d_update_decode(
        x,
        weight,
        cache,
        cache_indices,
        cache_mask,
        activation=activation,
        use_residual=use_residual,
        track_mask=track_mask,
        track_indices=track_indices,
    )
    assert_close(actual, expected_y, dtype)
    torch.testing.assert_close(
        cache.detach().cpu(), expected_cache, atol=0, rtol=0, check_dtype=False
    )
