import math

import pytest
import torch
import torch.nn.functional as F
from sgl_kernel import hadamard_transform


def _ref_torch_impl(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    assert x.dim() == 2
    bs, dim = x.shape
    log_dim = max(3, math.ceil(math.log2(max(dim, 1))))
    dim_padded = 1 << log_dim

    if dim_padded != dim:
        out = F.pad(x, (0, dim_padded - dim))
    else:
        out = x.clone()

    h = 1
    while h < dim_padded:
        out = out.view(bs, dim_padded // (2 * h), 2, h)
        a = out[:, :, 0, :]
        b = out[:, :, 1, :]
        out = torch.stack((a + b, a - b), dim=2).view(bs, dim_padded)
        h *= 2

    out = out * scale
    if dim_padded != dim:
        out = out[:, :dim]
    return out


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
    [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
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
@pytest.mark.parametrize("dim", [1024, 4096, 16384])
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
