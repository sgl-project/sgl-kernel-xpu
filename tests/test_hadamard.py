import math
from functools import lru_cache

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy.linalg import hadamard
from sgl_kernel import hadamard_transform


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
    return torch.from_numpy(hadamard(dim_padded, dtype=np.int8))


def _hadamard_matrix(dim_padded, dtype, device):
    """Hadamard matrix on *device* with *dtype*, freshly allocated each call.

    Deliberately uncached so the caller's tensor is released between tests.
    """
    return _hadamard_matrix_cpu(dim_padded).to(device=device, dtype=dtype)


def _ref_torch_impl(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    # min log_dim of 3 matches the kernel, which always pads up to dim >= 8.
    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    log_dim = max(3, math.ceil(math.log2(max(dim, 1))))
    dim_padded = 1 << log_dim
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    out = F.linear(x, _hadamard_matrix(dim_padded, x.dtype, x.device))
    out = out * scale
    if dim_padded != dim:
        out = out[:, :dim]
    return out.reshape(x_shape)


def _bench(fn, *, warmup: int = 5, iters: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()

    times = []
    for _ in range(iters):
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.xpu.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bs", [132, 1024])
@pytest.mark.parametrize("dim", [36, 1024, 4096, 16384])
@torch.inference_mode()
def test_hadamard_transform_perf(bs: int, dim: int, dtype: torch.dtype) -> None:
    if not torch.xpu.is_available():
        pytest.skip("XPU is required for SYCL hadamard performance comparison")

    x = _setup_inputs(bs, dim, dtype)
    scale = 1 / math.sqrt(dim)

    t_ref = _bench(lambda: _ref_torch_impl(x, scale=scale))
    t_sycl = _bench(lambda: hadamard_transform(x, scale=scale))

    assert (
        t_sycl < t_ref
    ), f"sycl ({t_sycl:.3f} ms) not faster than torch ({t_ref:.3f} ms), {bs=}, {dim=}, {dtype=}"


if __name__ == "__main__":
    pytest.main([__file__])
