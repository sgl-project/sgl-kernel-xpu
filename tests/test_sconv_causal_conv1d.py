import pytest
import torch
from sconv_reference import assert_close, causal_conv1d_ref, rand

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="Inkling sconv ops are XPU-only",
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "lengths,D",
    [
        ([1, 128, 1, 145], 17),  # Ragged sequences and non-vector channels.
        ([64, 64, 64, 64], 64),  # Prefill dispatch threshold with four sequences.
        ([0, 1, 255, 1], 31),  # Empty sequence and a tail past the threshold.
    ],
)
def test_causal_conv1d_rolling_window_boundaries_and_tails(dtype, lengths, D):
    """Cover prefill blocks, packed-sequence resets, cache masks, and tails."""
    from sgl_kernel.inkling_sconv import causal_conv1d

    torch.manual_seed(1)
    B, T, W = len(lengths), sum(lengths), 4
    x = rand((T, D), dtype)
    weight = rand((D, W), dtype, scale=0.2)
    cache = rand((B, W - 1, D), dtype, scale=0.1)
    cache_mask = torch.tensor(
        [True, False, True, False], dtype=torch.bool, device="xpu"
    ).reshape(B, 1, 1)
    safe_idx = torch.tensor([3, 1, 0, 2], dtype=torch.int64, device="xpu")
    cu = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        dtype=torch.int64,
        device="xpu",
    )
    si = torch.repeat_interleave(
        torch.arange(B, dtype=torch.int32, device="xpu"),
        torch.tensor(lengths, dtype=torch.int64, device="xpu"),
    )

    actual = causal_conv1d(
        x,
        weight,
        cache,
        cache_mask,
        safe_idx,
        cu,
        si,
        activation="silu",
        use_residual=True,
    )
    expected = causal_conv1d_ref(
        x,
        weight,
        cache,
        cache_mask,
        safe_idx,
        cu,
        si,
        activation="silu",
        use_residual=True,
    )
    assert_close(actual, expected, dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "T,D",
    [
        (255, 32),  # Below the prefill specialization threshold.
        (256, 32),  # First shape eligible for the packed path.
        (257, 32),  # One-token tail.
        (259, 40),  # Tail with several channel groups.
        (260, 36),  # BF16 uses the block path; FP16 can use the packed path.
        (256, 34),  # Both dtypes fall back for unaligned channels.
    ],
)
def test_causal_conv1d_vector_prefill_tail(dtype, T, D):
    """Check the packed and block paths around the prefill dispatch boundary."""
    from sgl_kernel.inkling_sconv import causal_conv1d

    torch.manual_seed(3)
    W = 4
    x = rand((T, D), dtype)
    weight = rand((D, W), dtype, scale=0.2)
    cache = rand((1, W - 1, D), dtype, scale=0.1)
    cache_mask = torch.ones((1, 1, 1), dtype=torch.bool, device="xpu")
    safe_idx = torch.zeros(1, dtype=torch.int64, device="xpu")
    cu = torch.tensor([0, T], dtype=torch.int64, device="xpu")
    si = torch.zeros(T, dtype=torch.int32, device="xpu")

    actual = causal_conv1d(
        x,
        weight,
        cache,
        cache_mask,
        safe_idx,
        cu,
        si,
        activation="silu",
        use_residual=True,
    )
    expected = causal_conv1d_ref(
        x,
        weight,
        cache,
        cache_mask,
        safe_idx,
        cu,
        si,
        activation="silu",
        use_residual=True,
    )
    assert_close(actual, expected, dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_causal_conv1d_inkling_prefill_shape(dtype):
    """Exercise the production single-sequence specialization at M4096/H6144."""
    from sgl_kernel.inkling_sconv import causal_conv1d

    torch.manual_seed(2)
    T, D, W = 4096, 6144, 4
    x = rand((T, D), dtype)
    weight = rand((D, W), dtype, scale=0.2)
    cache = rand((1, W - 1, D), dtype, scale=0.1)
    cache_mask = torch.ones((1, 1, 1), dtype=torch.bool, device="xpu")
    safe_idx = torch.zeros(1, dtype=torch.int64, device="xpu")
    cu = torch.tensor([0, T], dtype=torch.int64, device="xpu")
    si = torch.zeros(T, dtype=torch.int32, device="xpu")

    actual = causal_conv1d(
        x,
        weight,
        cache,
        cache_mask,
        safe_idx,
        cu,
        si,
        activation="silu",
        use_residual=True,
    )

    # A vectorized FP32 reference makes checking every production output
    # practical, including cache-prefixed rows and every tail/channel tile.
    x_float = x.float()
    weight_float = weight.float()
    cache_float = cache[0].float()
    acc = torch.zeros_like(x_float)
    for iw in range(W):
        offset = iw - (W - 1)
        if offset < 0:
            prefix_len = -offset
            acc[:prefix_len] += cache_float[W - 1 - prefix_len :] * weight_float[:, iw]
            acc[prefix_len:] += x_float[: T - prefix_len] * weight_float[:, iw]
        else:
            acc += x_float * weight_float[:, iw]
    expected = torch.nn.functional.silu(acc) + x_float

    assert_close(actual, expected.cpu(), dtype)
