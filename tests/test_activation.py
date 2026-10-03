# Adapted from https://github.com/flashinfer-ai/flashinfer/blob/4e8eb1879f9c3ba6d75511e5893183bf8f289a62/tests/test_activation.py

import sys

import pytest
import sgl_kernel
import torch
import utils

device = utils.get_device()
ref_device = utils.get_reference_device()


@pytest.mark.parametrize("dim", [128, 256, 512, 2048, 4096, 11008, 16384])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("seq_len", [1, 2, 4, 8, 16, 32, 64, 128, 512])
def test_fused_silu_mul(dim, batch_size, seq_len):
    x = torch.randn(batch_size, seq_len, 2 * dim).to(ref_device).to(torch.float16)
    y_ref = x[..., dim:] * torch.nn.functional.silu(x[..., :dim])
    y = sgl_kernel.silu_and_mul(x.to(device))
    torch.testing.assert_close(y_ref.cpu(), y.cpu(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "rows,width",
    [
        (2047, 256),
        (2048, 256),
        (4095, 512),
        (4096, 512),
        (4095, 768),
        (4096, 768),
        (6143, 768),
        (6144, 768),
        (6176, 772),
        (8192, 1024),
        (8192, 2048),
        (8192, 4096),
        (24576, 768),
    ],
)
def test_fused_silu_mul_large_routed_batch(dtype, rows, width):
    x = torch.randn(rows, 2 * width, dtype=dtype, device=device)
    out = sgl_kernel.silu_and_mul(x)
    ref = (torch.nn.functional.silu(x[:, :width].float()) * x[:, width:].float()).to(
        dtype
    )
    # CUDA activation/test_activation.py uses 1e-2 for BF16. One BF16 ULP
    # exceeds 5e-3 for some values even with float accumulation.
    tolerance = 1e-2 if dtype == torch.bfloat16 else 1e-3
    torch.testing.assert_close(out, ref, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dim", [128, 256, 512, 2048, 4096, 11008, 16384])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("seq_len", [1, 2, 4, 8, 16, 32, 64, 128, 512])
def test_fused_gelu_tanh_mul(dim, batch_size, seq_len):
    x = torch.randn(batch_size, seq_len, 2 * dim).to(ref_device).to(torch.float16)
    y_ref = x[..., dim:] * torch.nn.functional.gelu(x[..., :dim], approximate="tanh")
    y = sgl_kernel.gelu_tanh_and_mul(x.to(device))
    torch.testing.assert_close(y_ref.cpu(), y.cpu(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dim", [128, 256, 512, 2048, 4096, 11008, 16384])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("seq_len", [1, 2, 4, 8, 16, 32, 64, 128, 512])
def test_fused_gelu_mul(dim, batch_size, seq_len):
    x = torch.randn(batch_size, seq_len, 2 * dim).to(ref_device).to(torch.float16)
    y_ref = x[..., dim:] * torch.nn.functional.gelu(x[..., :dim], approximate="none")
    y = sgl_kernel.gelu_and_mul(x.to(device))
    torch.testing.assert_close(y_ref.cpu(), y.cpu(), rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
