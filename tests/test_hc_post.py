import pytest
import torch
import utils

device = utils.get_device()
HC_MULT = 4


def hc_post_torch_impl(x, residual, post, comb):
    return (
        post.unsqueeze(-1) * x.unsqueeze(1)
        + (comb.unsqueeze(-1) * residual.unsqueeze(2)).sum(dim=1)
    ).type_as(x)

def _make_inputs(T, D, device, seed=42):
    torch.manual_seed(seed)
    print("x..")
    x = torch.randn(T, D, dtype=torch.bfloat16, device=device)
    print("residual..")
    residual = torch.randn(T, HC_MULT, D, dtype=torch.bfloat16, device=device)
    print("post..")
    post = torch.rand(T, HC_MULT, dtype=torch.float32, device=device) * 2.0
    print("comb..")
    comb = torch.rand(T, HC_MULT, HC_MULT, dtype=torch.float32, device=device)
    comb = comb / comb.sum(dim=-1, keepdim=True)
    return x, residual, post, comb


@pytest.mark.parametrize("T", [16, 48, 128, 768, 885, 1021, 1024, 1280, 2047])
@pytest.mark.parametrize("D", [4096])
def test_hc_post_torch_only(T, D):
    print("make inputs")
    # x, residual, post, comb = _make_inputs(T, D, device=f"{device}:0")
    x, residual, post, comb = _make_inputs(T, D, device="cpu")
    print("running torch implementation")
    output = hc_post_torch_impl(x, residual, post, comb)
    print("done")
    assert output.shape == (T, HC_MULT, D)
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()