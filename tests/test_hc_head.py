import pytest
import torch
import torch.nn.functional as F
from sgl_kernel import fused_hc_head

pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU required")

device = torch.device("xpu")


def hc_head_torch_ref(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    shape, dtype = x.size(), x.dtype
    x_flat = x.flatten(1).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x_flat, hc_fn.float()) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale.float() + hc_base.float()) + hc_eps
    y = torch.sum(pre.unsqueeze(-1) * x.float().view(shape), dim=1)
    return y.to(dtype)


@pytest.mark.parametrize("T", [1, 16, 48, 128, 768, 1024])
@pytest.mark.parametrize("hidden_size", [4096, 7168])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_hc_head(T, hidden_size, dtype):
    torch.manual_seed(42)

    hc_mult = 4
    x = torch.randn(T, hc_mult, hidden_size, dtype=dtype, device=f"{device}:0")
    hc_fn = torch.randn(
        hc_mult, hc_mult * hidden_size, dtype=torch.float32, device=f"{device}:0"
    )
    hc_scale = torch.randn(1, dtype=torch.float32, device=f"{device}:0")
    hc_base = torch.randn(hc_mult, dtype=torch.float32, device=f"{device}:0")

    norm_eps = 1e-6
    hc_eps = 1e-6

    expected = hc_head_torch_ref(x, hc_fn, hc_scale, hc_base, norm_eps, hc_eps)
    actual = fused_hc_head(
        x, hc_fn, hc_scale, hc_base, norm_eps=norm_eps, hc_eps=hc_eps
    )

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
