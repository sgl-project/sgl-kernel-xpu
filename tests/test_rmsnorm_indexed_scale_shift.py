"""Correctness and contract checks for indexed RMSNorm with scale and shift."""

import pytest
import torch
import torch.nn.functional as F
from sgl_kernel import rmsnorm_indexed_scale_shift

pytestmark = pytest.mark.arch("xe20")


def make_inputs(rows, hidden, index_dtype, strided_table):
    x = torch.randn(rows, hidden, device="xpu", dtype=torch.bfloat16)
    weight = torch.randn(hidden, device="xpu", dtype=torch.bfloat16)
    if strided_table:
        packed = torch.randn(3, 6 * hidden, device="xpu", dtype=torch.bfloat16)
        shift, scale = packed.chunk(6, dim=-1)[:2]
    else:
        shift = torch.randn(3, hidden, device="xpu", dtype=torch.bfloat16)
        scale = torch.randn_like(shift)
    indices = torch.randint(0, 3, (rows,), device="xpu", dtype=index_dtype)
    return x, weight, shift, scale, indices


def eager(x, weight, shift, scale, indices, eps):
    normalized = F.rms_norm(x, (x.shape[-1],), weight, eps)
    return normalized * (
        1 + scale.index_select(0, indices.long())
    ) + shift.index_select(0, indices.long())


def assert_matches_eager(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2, equal_nan=True)


def assert_within_one_bf16_ulp(actual, expected):
    """Compare adjacent BF16 encodings, including either sign of finite values."""
    assert actual.dtype == expected.dtype == torch.bfloat16
    actual_bits = actual.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    expected_bits = expected.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    actual_ordered = torch.where(
        actual_bits & 0x8000 != 0,
        0x8000 - (actual_bits & 0x7FFF),
        0x8000 + actual_bits,
    )
    expected_ordered = torch.where(
        expected_bits & 0x8000 != 0,
        0x8000 - (expected_bits & 0x7FFF),
        0x8000 + expected_bits,
    )
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    ulps = (actual_ordered - expected_ordered).abs()
    matches = (
        (finite & (ulps <= 1))
        | (torch.isnan(actual) & torch.isnan(expected))
        | (actual == expected)
    )
    if not bool(matches.all()):
        max_ulp = ulps[finite].max().item() if bool(finite.any()) else "no finite pairs"
        raise AssertionError(f"BF16 values differ by more than 1 ULP; max={max_ulp}")


@pytest.mark.parametrize(
    "reference,adjacent,two_ulps",
    [(0x3F80, 0x3F81, 0x3F82), (0xBF80, 0xBF7F, 0xBF7E)],
)
def test_bf16_ulp_boundary(reference, adjacent, two_ulps):
    def bf16_bits(bits):
        return (
            torch.tensor([bits], dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        )

    assert_within_one_bf16_ulp(bf16_bits(adjacent), bf16_bits(reference))
    with pytest.raises(AssertionError, match="more than 1 ULP"):
        assert_within_one_bf16_ulp(bf16_bits(two_ulps), bf16_bits(reference))


@pytest.mark.parametrize("rows,hidden", [(1, 96), (7, 128), (128, 5376)])
def test_rmsnorm_indexed_scale_shift_bit_exact_canary(rows, hidden):
    torch.manual_seed(1117)
    x, weight, shift, scale, indices = make_inputs(rows, hidden, torch.int32, False)
    expected = eager(x, weight, shift, scale, indices, 1e-5)
    actual = rmsnorm_indexed_scale_shift(x, weight, shift, scale, indices, 1e-5)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize(
    "rows,hidden", [(1, 96), (7, 128), (128, 5376), (128, 8192), (16464, 5376)]
)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("seed", [1117, 1119])
def test_rmsnorm_indexed_scale_shift_matches_eager(rows, hidden, index_dtype, seed):
    torch.manual_seed(seed)
    strided = rows != 1
    x, weight, shift, scale, indices = make_inputs(rows, hidden, index_dtype, strided)
    x_before = x.clone()
    expected = eager(x, weight, shift, scale, indices, 1e-5)
    actual = rmsnorm_indexed_scale_shift(x, weight, shift, scale, indices, 1e-5)
    assert actual.shape == x.shape and actual.dtype == x.dtype
    assert actual.data_ptr() != x.data_ptr()
    assert torch.equal(x, x_before)
    assert_matches_eager(actual, expected)
    assert_within_one_bf16_ulp(actual, expected)


@pytest.mark.parametrize("hidden", [1, 3, 127, 129, 5375, 5377])
def test_rmsnorm_indexed_scale_shift_scalar_rms_path(hidden):
    torch.manual_seed(1120)
    inputs = make_inputs(7, hidden, torch.int32, True)
    expected = eager(*inputs, 1e-5)
    actual = rmsnorm_indexed_scale_shift(*inputs, 1e-5)
    assert_matches_eager(actual, expected)
    assert_within_one_bf16_ulp(actual, expected)


@pytest.mark.parametrize("hidden", [128, 5376])
def test_rmsnorm_indexed_scale_shift_empty_batch(hidden):
    x, weight, shift, scale, indices = make_inputs(0, hidden, torch.int32, True)
    actual = rmsnorm_indexed_scale_shift(x, weight, shift, scale, indices, 1e-5)
    assert actual.shape == x.shape and actual.dtype == x.dtype
    assert torch.equal(actual, eager(x, weight, shift, scale, indices, 1e-5))
    assert (
        rmsnorm_indexed_scale_shift(
            x, weight, shift[:0], scale[:0], indices, 1e-5
        ).shape
        == x.shape
    )


def test_rmsnorm_indexed_scale_shift_large_hidden_fallback():
    x, weight, shift, scale, indices = make_inputs(7, 8193, torch.int64, True)
    actual = rmsnorm_indexed_scale_shift(x, weight, shift, scale, indices, 1e-5)
    expected = eager(x, weight, shift, scale, indices, 1e-5)
    assert_matches_eager(actual, expected)
    assert_within_one_bf16_ulp(actual, expected)


@pytest.mark.parametrize("offset_x,offset_weight", [(True, False), (False, True)])
def test_rmsnorm_indexed_scale_shift_unaligned_storage(offset_x, offset_weight):
    torch.manual_seed(1121)
    rows, hidden = 7, 128
    x, weight, _, _, indices = make_inputs(rows, hidden, torch.int64, False)
    if offset_x:
        x = torch.randn(rows * hidden + 1, device="xpu", dtype=x.dtype)[1:].view(
            rows, hidden
        )
    if offset_weight:
        weight = torch.randn(hidden + 1, device="xpu", dtype=x.dtype)[1:]
    packed = torch.randn(3, 2 * hidden + 1, device="xpu", dtype=x.dtype)
    shift, scale = packed[:, 1 : hidden + 1], packed[:, hidden + 1 :]
    expected = eager(x, weight, shift, scale, indices, 1e-5)
    actual = rmsnorm_indexed_scale_shift(x, weight, shift, scale, indices, 1e-5)
    assert_matches_eager(actual, expected)
    assert_within_one_bf16_ulp(actual, expected)


@pytest.mark.parametrize(
    "rows,hidden", [(7, 12), (128, 4096), (512, 4096), (128, 5376)]
)
@pytest.mark.parametrize("eps", [1e-12, 0.03125, 1.0])
def test_rmsnorm_indexed_scale_shift_reduction_and_rounding(rows, hidden, eps):
    torch.manual_seed(1122)
    x, weight, shift, scale, indices = make_inputs(rows, hidden, torch.int32, True)
    # Different magnitudes expose FP32 reduction differences after BF16 rounding.
    exponents = torch.randint(-8, 9, x.shape, device=x.device).float()
    x = (x.float() * torch.exp2(exponents)).bfloat16()
    scale[0].fill_(-1)
    scale[1].fill_(-1.0078125)
    expected = eager(x, weight, shift, scale, indices, eps)
    actual = rmsnorm_indexed_scale_shift(x, weight, shift, scale, indices, eps)
    assert_matches_eager(actual, expected)
    assert_within_one_bf16_ulp(actual, expected)


@pytest.mark.parametrize("hidden", [128, 5376])
@pytest.mark.parametrize("mode", ["zero", "small", "nonfinite"])
def test_rmsnorm_indexed_scale_shift_extreme_values(hidden, mode):
    torch.manual_seed(1123)
    x, weight, shift, scale, indices = make_inputs(128, hidden, torch.int64, True)
    if mode == "zero":
        x.zero_()
    elif mode == "small":
        x = (x.float() * 2**-60).bfloat16()
    else:
        values = torch.tensor(
            [0.0, -0.0, float("inf"), -float("inf"), float("nan"), 1.0, -1.0, 2**-133],
            device=x.device,
            dtype=x.dtype,
        )
        x.view(-1).copy_(values.repeat(x.numel() // values.numel()))
    expected = eager(x, weight, shift, scale, indices, 1e-40)
    actual = rmsnorm_indexed_scale_shift(x, weight, shift, scale, indices, 1e-40)
    assert_matches_eager(actual, expected)
    assert_within_one_bf16_ulp(actual, expected)


def test_rmsnorm_indexed_scale_shift_uses_current_stream():
    x, weight, shift, scale, indices = make_inputs(128, 5376, torch.int64, True)
    x.zero_()
    weight.fill_(1)
    shift.zero_()
    scale.zero_()
    torch.xpu._sleep(1)
    torch.xpu.synchronize()
    stream = torch.xpu.Stream()
    stream.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(stream):
        # Delay the fill on this stream; a launch on an unrelated stream can
        # then read the old x while this stream is still sleeping.
        torch.xpu._sleep(1_000_000_000)
        x.fill_(1)
        actual = rmsnorm_indexed_scale_shift(x, weight, shift, scale, indices, 1e-5)
        checksum = actual.float().sum()
    stream.synchronize()
    expected = eager(x, weight, shift, scale, indices, 1e-5)
    assert_matches_eager(actual, expected)
    torch.testing.assert_close(checksum, expected.float().sum(), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "case,message",
    [
        ("x_dtype", "requires contiguous BF16 XPU rows"),
        ("indices_length", "indices mismatch"),
        ("x_layout", "requires contiguous BF16 XPU rows"),
        ("indices_stride", "indices mismatch"),
        ("empty_table", "modulation table mismatch"),
    ],
)
def test_rmsnorm_indexed_scale_shift_rejects_invalid_inputs(case, message):
    x, weight, shift, scale, indices = make_inputs(7, 128, torch.int64, True)
    if case == "x_dtype":
        x = x.half()
    elif case == "indices_length":
        indices = indices[:-1]
    elif case == "x_layout":
        x = x.T.contiguous().T
    elif case == "indices_stride":
        indices = indices.repeat_interleave(2)[::2]
    else:
        shift, scale = shift[:0], scale[:0]
    with pytest.raises(RuntimeError, match=message):
        rmsnorm_indexed_scale_shift(x, weight, shift, scale, indices, 1e-5)


def test_rmsnorm_indexed_scale_shift_rejects_invalid_eps():
    x, weight, shift, scale, indices = make_inputs(7, 128, torch.int64, True)
    for eps in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(RuntimeError, match="eps must be positive"):
            rmsnorm_indexed_scale_shift(x, weight, shift, scale, indices, eps)


def test_rmsnorm_indexed_scale_shift_requires_stride_one_for_single_index():
    x, weight, shift, scale, _ = make_inputs(1, 128, torch.int64, True)
    indices = torch.zeros(2, device="xpu", dtype=torch.int64)[::2]
    assert indices.is_contiguous() and indices.stride(0) == 2
    with pytest.raises(RuntimeError, match="indices mismatch"):
        rmsnorm_indexed_scale_shift(x, weight, shift, scale, indices, 1e-5)


@pytest.mark.debug_checks
@pytest.mark.parametrize("bad_index", [-1, 3])
def test_rmsnorm_indexed_scale_shift_rejects_out_of_range_index(bad_index):
    x, weight, shift, scale, indices = make_inputs(7, 128, torch.int64, True)
    indices[0] = bad_index
    with pytest.raises(RuntimeError, match="indices must be in"):
        rmsnorm_indexed_scale_shift(x, weight, shift, scale, indices, 1e-5)
