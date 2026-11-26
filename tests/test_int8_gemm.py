import sys

import pytest
import torch
import utils
from sgl_kernel import int8_scaled_mm
from utils import is_xe2_device

device = utils.get_device()


def to_int8(tensor: torch.Tensor) -> torch.Tensor:
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)


def torch_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias):
    o = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    o = o * scale_a.reshape(-1, 1) * scale_b.reshape(1, -1)
    if bias is not None:
        o = o + bias.reshape(1, -1)
    return o.to(out_dtype)


def _test_accuracy_once(M, N, K, with_bias, out_dtype, device):
    a = to_int8(torch.randn((M, K), device=device) * 5)
    b = to_int8(torch.randn((N, K), device=device).t() * 5)
    scale_a = torch.randn((M,), device=device, dtype=torch.float32)
    scale_b = torch.randn((N,), device=device, dtype=torch.float32)
    if with_bias:
        bias = torch.randn((N,), device=device, dtype=out_dtype) * 10
    else:
        bias = None
    o = int8_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)
    o1 = torch_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)
    torch.testing.assert_close(o, o1)


@pytest.mark.skipif(
    not is_xe2_device(),
    reason="int8_scaled_mm XPU kernel is currently implemented for Xe2 (BMG)",
)
@pytest.mark.parametrize("M", [1, 7, 16, 32, 64, 128, 256, 512, 1024, 4096])
@pytest.mark.parametrize("N", [16, 128, 512, 1024, 4096, 8192])
@pytest.mark.parametrize("K", [512, 1024, 4096, 8192])
@pytest.mark.parametrize("with_bias", [True, False])
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16])
def test_accuracy(M, N, K, with_bias, out_dtype):
    _test_accuracy_once(M, N, K, with_bias, out_dtype, device)


@pytest.mark.skipif(
    not is_xe2_device(),
    reason="int8_scaled_mm XPU kernel is currently implemented for Xe2 (BMG)",
)
@pytest.mark.parametrize(
    "M,N,K",
    [
        (0, 128, 64),
        (1, 8, 16),
        (1, 16, 48),
        (7, 24, 48),
        (65, 24, 80),
        (137, 120, 112),
        (192, 4096, 4096),
        (1, 14336, 4096),
        (1024, 4096, 14336),
    ],
)
@pytest.mark.parametrize("with_bias", [True, False])
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16])
def test_edge_shapes(M, N, K, with_bias, out_dtype):
    _test_accuracy_once(M, N, K, with_bias, out_dtype, device)


@pytest.mark.skipif(
    not is_xe2_device(),
    reason="int8_scaled_mm XPU kernel is currently implemented for Xe2 (BMG)",
)
def test_2d_scales_and_unaligned_views():
    M, N, K = 73, 256, 512
    # Create padded tensors and take unaligned sliced views (offset != 64B aligned)
    a_full = to_int8(torch.randn((M + 3, K + 16), device=device) * 5)
    b_full = to_int8(torch.randn((N + 5, K + 16), device=device) * 5)
    a = a_full[1 : 1 + M, 4 : 4 + K]
    b = b_full[1 : 1 + N, 4 : 4 + K].t()
    scale_a = torch.randn((M, 1), device=device, dtype=torch.float32)
    scale_b = torch.randn((N, 1), device=device, dtype=torch.float32)
    bias = torch.randn((N,), device=device, dtype=torch.bfloat16) * 5

    o = int8_scaled_mm(a, b, scale_a, scale_b, torch.bfloat16, bias)
    o_ref = torch_scaled_mm(a, b, scale_a, scale_b, torch.bfloat16, bias)
    torch.testing.assert_close(o, o_ref)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
