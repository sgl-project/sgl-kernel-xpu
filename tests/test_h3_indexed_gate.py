"""Correctness and contract checks for the H3 indexed gated residual."""

import pytest
import torch
from sgl_kernel.h3_fused import can_use_h3_indexed_gate, h3_indexed_gate

pytestmark = pytest.mark.arch("xe20")


@pytest.mark.skipif(not torch.xpu.is_available(), reason="requires Intel XPU")
@pytest.mark.parametrize(
    "rows,hidden",
    [
        (128, 2048),
        (128, 5376),
        (4096, 4096),
        (2048, 8192),
        (8192, 5376),
        (16464, 5376),
    ],
)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("magnitude", [0.25, 1.0, 30.0])
def test_indexed_gate_matches_eager_in_place_and_out_of_place(
    rows, hidden, index_dtype, magnitude
):
    torch.manual_seed(1110 + rows)
    x = torch.randn(rows, hidden, device="xpu", dtype=torch.bfloat16) * magnitude
    other = torch.randn_like(x) * magnitude
    packed = torch.randn(3, 6 * hidden, device="xpu", dtype=torch.bfloat16) * magnitude
    gate = packed.chunk(6, dim=-1)[2]
    storage = torch.randint(0, 3, (rows, 2), device="xpu", dtype=index_dtype)
    indices = storage[:, 0]
    expected = x + gate.index_select(0, indices.long()) * other

    actual = h3_indexed_gate(x, gate, other, indices)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    in_place = x.clone()
    returned = h3_indexed_gate(in_place, gate, other, indices, inplace=True)
    assert returned.data_ptr() == in_place.data_ptr()
    assert torch.equal(in_place.view(torch.int16), expected.view(torch.int16))


@pytest.mark.skipif(not torch.xpu.is_available(), reason="requires Intel XPU")
def test_indexed_gate_unaligned_table_fallback_is_bit_exact():
    torch.manual_seed(1119)
    rows, hidden = 128, 5376
    x = torch.randn(rows, hidden, device="xpu", dtype=torch.bfloat16)
    other = torch.randn_like(x)
    storage = torch.randn(3, hidden + 1, device="xpu", dtype=torch.bfloat16)
    gate = storage[:, 1:]
    indices = torch.randint(0, 3, (rows,), device="xpu", dtype=torch.int64)
    expected = x + gate.index_select(0, indices) * other
    assert torch.equal(
        h3_indexed_gate(x, gate, other, indices).view(torch.int16),
        expected.view(torch.int16),
    )
    in_place = x.clone()
    h3_indexed_gate(in_place, gate, other, indices, inplace=True)
    assert torch.equal(in_place.view(torch.int16), expected.view(torch.int16))


@pytest.mark.skipif(not torch.xpu.is_available(), reason="requires Intel XPU")
def test_indexed_gate_rejects_invalid_metadata():
    x = torch.randn(7, 128, device="xpu", dtype=torch.bfloat16)
    gate = torch.randn(3, 128, device="xpu", dtype=torch.bfloat16)
    other = torch.randn_like(x)
    indices = torch.zeros(7, device="xpu", dtype=torch.int64)
    invalid = (x, gate, other, indices[:-1])
    assert not can_use_h3_indexed_gate(*invalid)
    with pytest.raises(ValueError, match="unsupported H3 indexed gate"):
        h3_indexed_gate(*invalid)
