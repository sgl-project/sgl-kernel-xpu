import itertools

import pytest
import torch
from sgl_kernel import moe_sum_reduce


@pytest.mark.parametrize("num_tokens", [5, 16, 128])
@pytest.mark.parametrize("num_experts", [4, 8, 32])
@pytest.mark.parametrize("top_k", [2, 4, 8])
@pytest.mark.parametrize("hidden_dims", [16, 32, 64])
def test_moe_sum_reduce(num_tokens, num_experts, top_k, hidden_dims):
    torch.manual_seed(41)

    def moe_reduce_reference(inp, scaling):
        if inp.dim() == 3:
            out = inp.sum(dim=1) * float(scaling)
        elif inp.dim() == 2:
            out = inp * float(scaling)
        else:
            raise ValueError(f"Unexpected input dim {inp.dim()}")
        return out

    # routing that generate unique tokens
    inp = torch.randn((num_tokens, top_k, hidden_dims))
    out = torch.empty((num_tokens, hidden_dims))
    scaling = 1.0 / float(top_k)
    expected = moe_reduce_reference(inp, scaling)

    device = "xpu"
    inp_xpu = inp.clone().to(device)
    output_xpu = torch.empty((num_tokens, hidden_dims), device=device)
    moe_sum_reduce(inp_xpu, output_xpu, scaling)
    torch.testing.assert_close(expected, output_xpu.to("cpu"))
