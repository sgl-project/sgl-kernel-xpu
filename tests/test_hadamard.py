import math

import pytest
import torch
import torch.nn.functional as F
from sgl_kernel import hadamard_transform


def _fwht(x: torch.Tensor) -> torch.Tensor:
    """Unnormalized Walsh-Hadamard transform over the last dim (Sylvester order).

    O(n log n) butterfly; avoids materializing the n x n matrix (4 GiB at 32768).
    """
    n = x.shape[-1]
    h = 1
    while h < n:
        x = x.reshape(-1, n // (2 * h), 2, h)
        x = torch.stack((x[:, :, 0] + x[:, :, 1], x[:, :, 0] - x[:, :, 1]), dim=2)
        h *= 2
    return x.reshape(-1, n)


def _ref_torch_impl(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    # min log_dim of 3 matches the kernel, which always pads up to dim >= 8.
    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    log_dim = max(3, math.ceil(math.log2(max(dim, 1))))
    dim_padded = 1 << log_dim
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    out = (_fwht(x.float()) * scale).to(x.dtype)
    if dim_padded != dim:
        out = out[:, :dim]
    return out.reshape(x_shape)


def _setup_inputs(bs: int, dim: int, dtype: torch.dtype) -> torch.Tensor:
    torch.manual_seed(0)
    stream = torch.xpu.Stream()
    torch.xpu.set_stream(stream)
    return torch.randn(bs, dim, device="xpu", dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "dim",
    [1, 2, 4, 8, 16, 32, 36, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
)
@torch.inference_mode()
def test_hadamard_transform(dim: int, dtype: torch.dtype) -> None:
    if not torch.xpu.is_available():
        pytest.skip("XPU is required for SYCL hadamard accuracy comparison")

    if dtype == torch.float32:
        rtol, atol = 3e-4, 3e-3
    elif dtype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    else:  # float16
        rtol, atol = 3e-3, 5e-3

    batch_size = 15
    x = _setup_inputs(batch_size, dim, dtype)

    scale = 1 / math.sqrt(dim)

    out_sycl = hadamard_transform(x, scale=scale)
    out_ref = _ref_torch_impl(x, scale=scale)

    torch.testing.assert_close(
        out_sycl.float(),
        out_ref.float(),
        rtol=rtol,
        atol=atol,
        msg="SYCL hadamard output mismatch vs torch reference",
    )


if __name__ == "__main__":
    pytest.main([__file__])
