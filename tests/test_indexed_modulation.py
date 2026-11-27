import pytest
import sgl_kernel
import torch

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "xpu") or not torch.xpu.is_available(),
    reason="XPU is not available",
)


def _round_bf16_to_fp32(value: torch.Tensor) -> torch.Tensor:
    return value.to(torch.bfloat16).to(torch.float32)


def _scale_shift_ref(x, shift, scale, indices):
    out = x.float().clone()
    selected_shift = shift.index_select(0, indices.long()).float()
    selected_scale = scale.index_select(0, indices.long()).float()
    one_plus_scale = _round_bf16_to_fp32(1.0 + selected_scale)
    scaled = _round_bf16_to_fp32(out * one_plus_scale)
    return (scaled + selected_shift).to(torch.bfloat16)


def _gate_ref(x, gate, other, indices):
    selected_gate = gate.index_select(0, indices.long()).float()
    gated = _round_bf16_to_fp32(selected_gate * other.float())
    return (x.float() + gated).to(torch.bfloat16)


@pytest.mark.parametrize(
    "rows,hidden,num_indices",
    [
        (0, 100, 4),
        (3, 0, 5),
        (7, 96, 5),
        (13, 5376, 3),
        (16, 128, 4),
        (64, 97, 4),
        (128, 98, 4),
    ],
)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_indexed_scale_shift_bf16(rows, hidden, num_indices, index_dtype):
    torch.manual_seed(0)
    device = "xpu"
    x = torch.randn(rows, hidden, device=device, dtype=torch.bfloat16)
    shift = torch.randn(num_indices, hidden, device=device, dtype=torch.bfloat16)
    scale = torch.randn(num_indices, hidden, device=device, dtype=torch.bfloat16)
    indices = (torch.arange(rows, device=device) % num_indices).to(index_dtype)

    expected = _scale_shift_ref(x, shift, scale, indices)
    actual = x.clone()
    returned = sgl_kernel.indexed_scale_shift_bf16_(actual, shift, scale, indices)
    torch.xpu.synchronize()

    assert returned is actual
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "rows,hidden,num_indices",
    [
        (0, 100, 4),
        (3, 0, 5),
        (7, 96, 5),
        (13, 5376, 3),
        (16, 128, 4),
        (64, 97, 4),
        (128, 98, 4),
    ],
)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_indexed_gate_bf16(rows, hidden, num_indices, index_dtype):
    torch.manual_seed(1)
    device = "xpu"
    x = torch.randn(rows, hidden, device=device, dtype=torch.bfloat16)
    gate = torch.randn(num_indices, hidden, device=device, dtype=torch.bfloat16)
    other = torch.randn(rows, hidden, device=device, dtype=torch.bfloat16)
    indices = (torch.arange(rows, device=device) % num_indices).to(index_dtype)

    expected = _gate_ref(x, gate, other, indices)
    actual = x.clone()
    returned = sgl_kernel.indexed_gate_bf16_(actual, gate, other, indices)
    torch.xpu.synchronize()

    assert returned is actual
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


###############################################################################
# Unaligned storage-offset and row-stride tests
###############################################################################


def _make_offset_2d(rows, hidden, offset, device, dtype=torch.bfloat16):
    """Create a 2D contiguous view with a non-zero element storage offset.

    The resulting tensor has stride(0) == hidden and stride(1) == 1, so stride
    divisibility alone would still pick N=8 when hidden % 8 == 0; only checking
    base pointer alignment caps the vector width (e.g. offset=1 -> 2B aligned
    -> N=1, offset=2 -> 4B aligned -> N=2, offset=4 -> 8B aligned -> N=4).
    """
    buf = torch.randn(rows * hidden + offset, device=device, dtype=dtype)
    view = buf[offset:].view(rows, hidden)
    assert view.storage_offset() == offset
    assert view.stride(0) == hidden
    assert view.stride(1) == 1
    if offset % 8 != 0:
        assert view.data_ptr() % 16 != 0
    return view


def _make_strided_2d(rows, hidden, extra_cols, device, dtype=torch.bfloat16):
    """Create a last-dim-contiguous 2D slice whose row stride is hidden + extra_cols."""
    full = torch.randn(rows, hidden + extra_cols, device=device, dtype=dtype)
    view = full[:, :hidden]
    assert view.stride(0) == hidden + extra_cols
    assert view.stride(1) == 1
    return view


@pytest.mark.parametrize("target", ["x", "shift", "scale", "all"])
@pytest.mark.parametrize("offset", [1, 2, 4])
def test_indexed_scale_shift_bf16_unaligned_offset(target, offset):
    torch.manual_seed(2)
    device = "xpu"
    rows, hidden, num_indices = 16, 128, 4
    x = _make_offset_2d(rows, hidden, offset if target in ("x", "all") else 0, device)
    shift = _make_offset_2d(
        num_indices, hidden, offset if target in ("shift", "all") else 0, device
    )
    scale = _make_offset_2d(
        num_indices, hidden, offset if target in ("scale", "all") else 0, device
    )
    indices = (torch.arange(rows, device=device) % num_indices).to(torch.int64)

    expected = _scale_shift_ref(x, shift, scale, indices)
    actual = _make_offset_2d(
        rows, hidden, offset if target in ("x", "all") else 0, device
    )
    actual.copy_(x)
    returned = sgl_kernel.indexed_scale_shift_bf16_(actual, shift, scale, indices)
    torch.xpu.synchronize()

    assert returned is actual
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("target", ["x", "gate", "other", "all"])
@pytest.mark.parametrize("offset", [1, 2, 4])
def test_indexed_gate_bf16_unaligned_offset(target, offset):
    torch.manual_seed(3)
    device = "xpu"
    rows, hidden, num_indices = 16, 128, 4
    x = _make_offset_2d(rows, hidden, offset if target in ("x", "all") else 0, device)
    gate = _make_offset_2d(
        num_indices, hidden, offset if target in ("gate", "all") else 0, device
    )
    other = _make_offset_2d(
        rows, hidden, offset if target in ("other", "all") else 0, device
    )
    indices = (torch.arange(rows, device=device) % num_indices).to(torch.int64)

    expected = _gate_ref(x, gate, other, indices)
    actual = _make_offset_2d(
        rows, hidden, offset if target in ("x", "all") else 0, device
    )
    actual.copy_(x)
    returned = sgl_kernel.indexed_gate_bf16_(actual, gate, other, indices)
    torch.xpu.synchronize()

    assert returned is actual
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("extra_cols", [1, 2, 4])
def test_indexed_scale_shift_bf16_unaligned_row_strides(extra_cols):
    torch.manual_seed(4)
    device = "xpu"
    rows, hidden, num_indices = 16, 128, 4
    x = _make_strided_2d(rows, hidden, extra_cols, device)
    shift = _make_strided_2d(num_indices, hidden, extra_cols, device)
    scale = _make_strided_2d(num_indices, hidden, extra_cols, device)
    indices = (torch.arange(rows, device=device) % num_indices).to(torch.int64)

    expected = _scale_shift_ref(x, shift, scale, indices)
    actual = _make_strided_2d(rows, hidden, extra_cols, device)
    actual.copy_(x)
    returned = sgl_kernel.indexed_scale_shift_bf16_(actual, shift, scale, indices)
    torch.xpu.synchronize()

    assert returned is actual
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("extra_cols", [1, 2, 4])
def test_indexed_gate_bf16_unaligned_row_strides(extra_cols):
    torch.manual_seed(5)
    device = "xpu"
    rows, hidden, num_indices = 16, 128, 4
    x = _make_strided_2d(rows, hidden, extra_cols, device)
    gate = _make_strided_2d(num_indices, hidden, extra_cols, device)
    other = _make_strided_2d(rows, hidden, extra_cols, device)
    indices = (torch.arange(rows, device=device) % num_indices).to(torch.int64)

    expected = _gate_ref(x, gate, other, indices)
    actual = _make_strided_2d(rows, hidden, extra_cols, device)
    actual.copy_(x)
    returned = sgl_kernel.indexed_gate_bf16_(actual, gate, other, indices)
    torch.xpu.synchronize()

    assert returned is actual
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
