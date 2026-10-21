import pytest
import sgl_kernel
import torch

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available() or not sgl_kernel.is_xe3_arch(),
    reason="Xe35 kernels require compute capability 3.5",
)


def _softcap_tolerance(dtype):
    return {
        torch.bfloat16: 5e-3,
        torch.float16: 1e-3,
        torch.float32: 3e-5,
    }[dtype]


def test_softcap_logits_strided():
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    source = torch.randn(3, 17, device="xpu", dtype=torch.float32)
    logits = source[:, :13]
    before = logits.clone()
    tail = source[:, 13:].clone()
    torch.ops.sgl_kernel.softcap_logits_xe35(logits, 30.0)
    torch.testing.assert_close(logits, torch.tanh(before / 30.0) * 30.0)
    torch.testing.assert_close(source[:, 13:], tail, atol=0, rtol=0)


@pytest.mark.parametrize("width", [32, 33])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_softcap_logits_contiguous(width, dtype):
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    source = (
        torch.linspace(-90, 90, 4 * width, device="xpu").to(dtype).reshape(4, width)
    )
    logits = source.clone()
    torch.ops.sgl_kernel.softcap_logits_xe35(logits, 30.0)
    expected = torch.tanh(source.cpu().float() / 30.0) * 30.0
    tolerance = _softcap_tolerance(dtype)
    torch.testing.assert_close(
        logits.float().cpu(), expected, rtol=tolerance, atol=tolerance
    )


def test_softcap_logits_vector_accuracy_across_zero_and_saturation():
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    source = torch.linspace(-900.0, 900.0, 2048, device="xpu", dtype=torch.float32)
    source[:8] = torch.tensor(
        [0.0, 1e-6, -1e-6, 0.001, -0.001, 2.99, 3.01, -3.01],
        device="xpu",
        dtype=torch.float32,
    )
    logits = source.clone().reshape(8, 256)
    torch.ops.sgl_kernel.softcap_logits_xe35(logits, 30.0)
    torch.testing.assert_close(
        logits, (torch.tanh(source / 30.0) * 30.0).reshape(8, 256), rtol=3e-5, atol=3e-5
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("width", [33, 4097, 202048])
def test_softcap_logits_independent_reference(width, dtype):
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    source_cpu = (
        torch.linspace(-100, 100, 8 * width, dtype=torch.float64)
        .to(dtype)
        .reshape(8, width)
    )
    expected = 20 * torch.tanh(source_cpu.double() / 20)
    logits = source_cpu.to("xpu")
    torch.ops.sgl_kernel.softcap_logits_xe35(logits, 20.0)
    tolerance = _softcap_tolerance(dtype)
    torch.testing.assert_close(
        logits.cpu().double(), expected, atol=tolerance, rtol=tolerance
    )
