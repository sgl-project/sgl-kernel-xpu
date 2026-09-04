import pytest
import torch

import sgl_kernel

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "xpu") or not torch.xpu.is_available(), reason="XPU is not available"
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
    "rows,hidden,num_indices", [(0, 128, 4), (7, 96, 5), (13, 128, 3)]
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
    "rows,hidden,num_indices", [(0, 128, 4), (7, 96, 5), (13, 128, 3)]
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