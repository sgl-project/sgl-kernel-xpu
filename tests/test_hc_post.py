import pytest
import torch
import utils
from sgl_kernel import hc_post

device = utils.get_device()
HC_MULT = 4


def hc_post_torch_impl(x, residual, post, comb):
    return (
        post.unsqueeze(-1) * x.unsqueeze(1)
        + (comb.unsqueeze(-1) * residual.unsqueeze(2)).sum(dim=1)
    ).type_as(x)


def _make_inputs(T, D, device, seed=42):
    torch.manual_seed(seed)
    x = torch.randn(T, D, dtype=torch.bfloat16, device=device)
    residual = torch.randn(T, HC_MULT, D, dtype=torch.bfloat16, device=device)
    post = torch.rand(T, HC_MULT, dtype=torch.float32, device=device) * 2.0
    comb = torch.rand(T, HC_MULT, HC_MULT, dtype=torch.float32, device=device)
    comb = comb / comb.sum(dim=-1, keepdim=True)
    return x, residual, post, comb


@pytest.mark.parametrize("T", [16, 48, 128, 768, 885, 1021, 1024, 1280, 2047])
@pytest.mark.parametrize("D", [4096])
def test_hc_post_kernel(T, D):
    x, residual, post_layer_mix, comb_res_mix = _make_inputs(T, D, device=f"{device}:0")

    expected = hc_post_torch_impl(x, residual, post_layer_mix, comb_res_mix)

    out = hc_post(x, residual, post_layer_mix, comb_res_mix)

    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)
