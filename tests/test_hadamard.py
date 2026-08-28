import math
from functools import lru_cache

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from scipy.linalg import hadamard
from sgl_kernel import hadamard_transform
from utils import get_device


# Cache the HOST-side int8 matrix only, never a device tensor. The cost being
# avoided is scipy's build (~10s at dim=32768, and every case calls the
# reference twice), which is host work; caching the device copy instead would
# pin it for the whole session, and the conftest's empty_cache() fixture cannot
# free memory that a cache still references. At dim=32768 an fp32 device matrix
# is 4 GiB, which alone would not fit alongside the rest on an 11 GB card.
#
# int8 is safe here: Hadamard entries are +-1, so the cast below is
# bit-identical to scipy's float64 matrix at 1/8th the host memory.
@lru_cache(maxsize=2)
def _hadamard_matrix_cpu(dim_padded):
    """Host-side +-1 Hadamard matrix, cached per padded dim."""
    if hadamard is None:
        raise ImportError("Please install scipy")
    return torch.from_numpy(hadamard(dim_padded, dtype=np.int8))


def _hadamard_matrix(dim_padded, dtype, device):
    """Hadamard matrix on *device* with *dtype*, freshly allocated each call.

    Deliberately uncached so the caller's tensor is released between tests.
    """
    return _hadamard_matrix_cpu(dim_padded).to(device=device, dtype=dtype)


def hadamard_transform_ref(x, scale=1.0):
    """
    x: (..., dim)
    out: (..., dim)
    """
    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    log_dim = math.ceil(math.log2(dim))
    dim_padded = 2**log_dim
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    out = F.linear(x, _hadamard_matrix(dim_padded, x.dtype, x.device))
    out = out * scale
    return out[..., :dim].reshape(*x_shape)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "dim",
    [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
)
def test_fast_hadamard_transform(dim, dtype):
    device = get_device()

    if dtype == torch.float32:
        rtol, atol = 3e-4, 3e-3
    elif dtype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    else:  # float16
        rtol, atol = 3e-3, 5e-3

    torch.random.manual_seed(0)
    batch_size = 15

    x = torch.randn(batch_size, dim, device=device, dtype=dtype)
    x_ref = x.detach().clone().to(torch.float32)
    x_pt = x.detach().clone()

    scale = 1 / math.sqrt(dim)

    out = hadamard_transform(x, scale=scale)
    out_ref = hadamard_transform_ref(x_ref, scale=scale)
    out_pt = hadamard_transform_ref(x_pt, scale=scale)

    torch.testing.assert_close(
        out_pt.float(),
        out_ref,
        rtol=rtol,
        atol=atol,
        msg="Reference implementations mismatch",
    )
    torch.testing.assert_close(
        out.float(),
        out_ref,
        rtol=rtol,
        atol=atol,
        msg="fast_hadamard_transform output mismatch",
    )


if __name__ == "__main__":
    pytest.main([__file__])
