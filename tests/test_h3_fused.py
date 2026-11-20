"""Correctness and contract checks for indexed H3 RMSNorm/AdaLN."""

import pytest
import torch
import torch.nn.functional as F
from sgl_kernel.h3_fused import (
    can_use_h3_rmsnorm_indexed_adaln,
    h3_rmsnorm_indexed_adaln,
)

pytestmark = pytest.mark.arch("xe20")


def make_inputs(rows, hidden, index_dtype, strided_table, strided_indices):
    x = torch.randn(rows, hidden, device="xpu", dtype=torch.bfloat16)
    weight = torch.randn(hidden, device="xpu", dtype=torch.bfloat16)
    if strided_table:
        packed = torch.randn(3, 6 * hidden, device="xpu", dtype=torch.bfloat16)
        shift, scale = packed.chunk(6, dim=-1)[:2]
    else:
        shift = torch.randn(3, hidden, device="xpu", dtype=torch.bfloat16)
        scale = torch.randn_like(shift)
    if strided_indices:
        storage = torch.randint(0, 3, (rows, 2), device="xpu", dtype=index_dtype)
        indices = storage[:, 0]
    else:
        indices = torch.randint(0, 3, (rows,), device="xpu", dtype=index_dtype)
    return x, weight, shift, scale, indices


def eager(x, weight, shift, scale, indices, eps):
    normalized = F.rms_norm(x, (x.shape[-1],), weight, eps)
    return normalized * (
        1 + scale.index_select(0, indices.long())
    ) + shift.index_select(0, indices.long())


@pytest.mark.skipif(not torch.xpu.is_available(), reason="requires Intel XPU")
@pytest.mark.parametrize(
    "rows,hidden", [(1, 96), (7, 128), (128, 5376), (128, 8192), (16464, 5376)]
)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("seed", [1117, 1119])
def test_indexed_rms_adaln_matches_eager(rows, hidden, index_dtype, seed):
    torch.manual_seed(seed)
    strided = rows != 1
    x, weight, shift, scale, indices = make_inputs(
        rows, hidden, index_dtype, strided, strided
    )
    x_before = x.clone()
    expected = eager(x, weight, shift, scale, indices, 1e-5)
    actual = h3_rmsnorm_indexed_adaln(x, weight, shift, scale, indices, 1e-5)
    assert actual.shape == x.shape and actual.dtype == x.dtype
    assert actual.data_ptr() != x.data_ptr()
    assert torch.equal(x, x_before)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.skipif(not torch.xpu.is_available(), reason="requires Intel XPU")
@pytest.mark.parametrize("hidden", [1, 3, 127, 129, 5375, 5377])
def test_indexed_rms_adaln_scalar_rms_path(hidden):
    torch.manual_seed(1120)
    inputs = make_inputs(7, hidden, torch.int32, True, True)
    expected = eager(*inputs, 1e-5)
    actual = h3_rmsnorm_indexed_adaln(*inputs, 1e-5)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.skipif(not torch.xpu.is_available(), reason="requires Intel XPU")
@pytest.mark.parametrize("offset_x,offset_weight", [(True, False), (False, True)])
def test_indexed_rms_adaln_unaligned_storage(offset_x, offset_weight):
    torch.manual_seed(1121)
    rows, hidden = 7, 128
    x, weight, _, _, indices = make_inputs(rows, hidden, torch.int64, False, True)
    if offset_x:
        x = torch.randn(rows * hidden + 1, device="xpu", dtype=x.dtype)[1:].view(
            rows, hidden
        )
    if offset_weight:
        weight = torch.randn(hidden + 1, device="xpu", dtype=x.dtype)[1:]
    packed = torch.randn(3, 2 * hidden + 1, device="xpu", dtype=x.dtype)
    shift, scale = packed[:, 1 : hidden + 1], packed[:, hidden + 1 :]
    assert can_use_h3_rmsnorm_indexed_adaln(x, weight, shift, scale, indices)
    expected = eager(x, weight, shift, scale, indices, 1e-5)
    actual = h3_rmsnorm_indexed_adaln(x, weight, shift, scale, indices, 1e-5)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.skipif(not torch.xpu.is_available(), reason="requires Intel XPU")
@pytest.mark.parametrize(
    "rows,hidden", [(7, 12), (128, 4096), (512, 4096), (128, 5376)]
)
@pytest.mark.parametrize("eps", [1e-12, 0.03125, 1.0])
def test_indexed_rms_adaln_reduction_and_rounding(rows, hidden, eps):
    torch.manual_seed(1122)
    x, weight, shift, scale, indices = make_inputs(
        rows, hidden, torch.int32, True, True
    )
    # Different magnitudes expose FP32 reduction differences after BF16 rounding.
    exponents = torch.randint(-8, 9, x.shape, device=x.device).float()
    x = (x.float() * torch.exp2(exponents)).bfloat16()
    scale[0].fill_(-1)
    scale[1].fill_(-1.0078125)
    expected = eager(x, weight, shift, scale, indices, eps)
    actual = h3_rmsnorm_indexed_adaln(x, weight, shift, scale, indices, eps)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.skipif(not torch.xpu.is_available(), reason="requires Intel XPU")
@pytest.mark.parametrize("hidden", [128, 5376])
@pytest.mark.parametrize("mode", ["zero", "small", "nonfinite"])
def test_indexed_rms_adaln_extreme_values(hidden, mode):
    torch.manual_seed(1123)
    x, weight, shift, scale, indices = make_inputs(128, hidden, torch.int64, True, True)
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
    actual = h3_rmsnorm_indexed_adaln(x, weight, shift, scale, indices, 1e-40)
    # Integer views also compare signed zeros and the BF16 NaN encoding.
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.skipif(not torch.xpu.is_available(), reason="requires Intel XPU")
def test_indexed_rms_adaln_uses_current_stream():
    x, weight, shift, scale, indices = make_inputs(128, 5376, torch.int64, True, True)
    stream = torch.xpu.Stream()
    stream.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(stream):
        actual = h3_rmsnorm_indexed_adaln(x, weight, shift, scale, indices, 1e-5)
        checksum = actual.float().sum()
    stream.synchronize()
    expected = eager(x, weight, shift, scale, indices, 1e-5)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert torch.isfinite(checksum)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="requires Intel XPU")
def test_indexed_rms_adaln_rejects_invalid_inputs():
    x, weight, shift, scale, indices = make_inputs(7, 128, torch.int64, True, False)
    for invalid in (
        (x.half(), weight, shift, scale, indices),
        (x, weight, shift, scale, indices[:-1]),
        (x[:, ::2], weight, shift, scale, indices),
        (x, weight, shift[:0], scale[:0], indices),
    ):
        assert not can_use_h3_rmsnorm_indexed_adaln(*invalid)
        with pytest.raises(ValueError, match="unsupported H3 RMSNorm/AdaLN"):
            h3_rmsnorm_indexed_adaln(*invalid, 1e-5)
        with pytest.raises(RuntimeError, match="H3 RMS AdaLN"):
            torch.ops.sgl_kernel.h3_indexed_rms_adaln(*invalid, 1e-5)
    for eps in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(RuntimeError, match="eps must be positive"):
            h3_rmsnorm_indexed_adaln(x, weight, shift, scale, indices, eps)
