import os

import pytest
import torch
import torch.nn.functional as F
from sgl_kernel.inkling_dense_mlp import (
    inkling_down,
    inkling_gate_up,
    repack_inkling_down_weight,
    repack_inkling_gate_up_weight,
)

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="XPU is required",
)


def _relative_error(got: torch.Tensor, expected: torch.Tensor) -> float:
    got_f = got.float()
    expected_f = expected.float()
    return ((got_f - expected_f).abs() / expected_f.abs().clamp_min(1.0)).max().item()


def _make_case(m: int, hidden: int = 256, intermediate: int = 256):
    torch.manual_seed(20260923 + m)
    x = torch.randn((m, hidden), device="xpu", dtype=torch.bfloat16) * 0.1
    gate_up_weight = (
        torch.randn((2 * intermediate, hidden), device="xpu", dtype=torch.bfloat16)
        * 0.1
    )
    down_weight = (
        torch.randn((hidden, intermediate), device="xpu", dtype=torch.bfloat16) * 0.1
    )
    return x, gate_up_weight, down_weight


@pytest.mark.parametrize("m", [1, 17, 127, 193, 1025])
def test_inkling_dense_mlp_dispatch_and_tails(m: int):
    x, gate_up_weight, down_weight = _make_case(m)
    packed_gate_up = repack_inkling_gate_up_weight(gate_up_weight)
    packed_down = repack_inkling_down_weight(down_weight)

    activated = inkling_gate_up(x, packed_gate_up)
    got = inkling_down(activated, packed_down)

    projected = F.linear(x, gate_up_weight)
    gate, up = projected.chunk(2, dim=-1)
    expected_activated = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
    expected_down = F.linear(activated, down_weight)
    expected_module = F.linear(expected_activated, down_weight)
    torch.xpu.synchronize()

    assert activated.shape == (m, gate_up_weight.shape[0] // 2)
    assert activated.stride() == (activated.shape[1] + 64, 1)
    assert got.is_contiguous()
    assert _relative_error(activated, expected_activated) <= 1.0 / 64.0
    assert _relative_error(got, expected_down) <= 1.0 / 128.0
    assert _relative_error(got, expected_module) <= 1.0 / 64.0


def test_inkling_dense_mlp_out_variants():
    x, gate_up_weight, down_weight = _make_case(37)
    packed_gate_up = repack_inkling_gate_up_weight(gate_up_weight)
    packed_down = repack_inkling_down_weight(down_weight)
    intermediate = gate_up_weight.shape[0] // 2
    activated_storage = torch.empty(
        (x.shape[0], intermediate + 64), device="xpu", dtype=torch.bfloat16
    )
    activated = activated_storage[:, :intermediate]
    output = torch.empty_like(x)

    assert inkling_gate_up(x, packed_gate_up, out=activated) is activated
    assert inkling_down(activated, packed_down, out=output) is output
    torch.xpu.synchronize()


def test_inkling_dense_mlp_empty_m():
    x, gate_up_weight, down_weight = _make_case(0)
    activated = inkling_gate_up(x, repack_inkling_gate_up_weight(gate_up_weight))
    output = inkling_down(activated, repack_inkling_down_weight(down_weight))
    assert activated.shape == (0, 256)
    assert output.shape == (0, 256)


def test_inkling_dense_mlp_rejects_fp16():
    gate_up_weight = torch.empty((512, 256), device="xpu", dtype=torch.float16)
    down_weight = torch.empty((256, 256), device="xpu", dtype=torch.float16)
    with pytest.raises(TypeError, match="bfloat16"):
        repack_inkling_gate_up_weight(gate_up_weight)
    with pytest.raises(TypeError, match="bfloat16"):
        repack_inkling_down_weight(down_weight)


@pytest.mark.skipif(
    os.environ.get("SGL_KERNEL_TEST_PRODUCTION") != "1",
    reason="set SGL_KERNEL_TEST_PRODUCTION=1 for the 4096x6144 production case",
)
def test_inkling_dense_mlp_production_shape():
    m, hidden, intermediate = 4096, 6144, 6144
    x, gate_up_weight, down_weight = _make_case(m, hidden, intermediate)
    packed_gate_up = repack_inkling_gate_up_weight(gate_up_weight)
    packed_down = repack_inkling_down_weight(down_weight)

    activated = inkling_gate_up(x, packed_gate_up)
    got = inkling_down(activated, packed_down)
    projected = F.linear(x, gate_up_weight)
    gate, up = projected.chunk(2, dim=-1)
    expected_activated = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
    # Check down against the activation it actually consumes. Comparing it only
    # to the ideal gate/up reference incorrectly charges amplified gate/up
    # rounding differences to the down kernel at long K.
    expected_down = torch.matmul(activated.float(), down_weight.float().t()).to(
        torch.bfloat16
    )
    expected_module = torch.matmul(
        expected_activated.float(), down_weight.float().t()
    ).to(torch.bfloat16)
    torch.xpu.synchronize()

    assert _relative_error(activated, expected_activated) <= 1.0 / 64.0
    assert _relative_error(got, expected_down) <= 1.0 / 128.0
    assert _relative_error(got, expected_module) <= 1.0 / 16.0
