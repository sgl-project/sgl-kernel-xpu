import pytest
import torch
from sconv_reference import assert_close, causal_conv1d_ref, rand

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="Inkling sconv ops are XPU-only",
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("W,D", [(3, 7), (4, 8)])
@pytest.mark.parametrize("activation", [None, "silu"])
@pytest.mark.parametrize("use_residual", [False, True])
def test_causal_conv1d_matches_inkling_pr_semantics(
    dtype, W, D, activation, use_residual
):
    from sgl_kernel.inkling_sconv import causal_conv1d

    torch.manual_seed(0)
    B = 3
    T = 7
    x = rand((T, D), dtype)
    weight = rand((D, W), dtype, scale=0.2)
    cache = rand((B, W - 1, D), dtype, scale=0.1)
    cache_mask = torch.tensor(
        [True, False, True], dtype=torch.bool, device="xpu"
    ).reshape(B, 1, 1)
    safe_idx = torch.arange(B, dtype=torch.int64, device="xpu")
    cu = torch.tensor([0, 1, 5, 7], dtype=torch.int64, device="xpu")
    si = torch.tensor([0, 1, 1, 1, 1, 2, 2], dtype=torch.int32, device="xpu")

    actual = causal_conv1d(
        x,
        weight,
        cache,
        cache_mask,
        safe_idx,
        cu,
        si,
        activation=activation,
        use_residual=use_residual,
    )
    expected = causal_conv1d_ref(
        x,
        weight,
        cache,
        cache_mask,
        safe_idx,
        cu,
        si,
        activation=activation,
        use_residual=use_residual,
    )
    assert_close(actual, expected, dtype)
