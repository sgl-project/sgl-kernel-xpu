import itertools
import sys
from typing import Optional, Tuple

import pytest
import torch
import utils
from sgl_kernel import sgl_per_token_quant_fp8

device = utils.get_device()
# XPU is never HIP/ROCm, so the fp8 dtype is always e4m3fn (matches the sibling
# test_per_tensor_quant_fp8.py, and avoids importing sglang which isn't
# installed in the kernel CI image).
fp8_type_ = torch.float8_e4m3fn


def torch_per_token_quant_fp8(tensor, inv_scale):
    # The reference implementation that fully aligns to
    # the kernel being tested.
    finfo = torch.finfo(torch.float8_e4m3fn)
    inv_scale = inv_scale.view(-1, 1)
    scale = inv_scale.reciprocal()
    qweight = (tensor.to(torch.float32) * scale).clamp(min=finfo.min, max=finfo.max)
    qweight = qweight.to(torch.float8_e4m3fn)
    return qweight


def sglang_per_token_quant_fp8(
    input: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scale = torch.zeros(input.size(0), device=input.device, dtype=torch.float32)
    output = torch.empty_like(input, device=input.device, dtype=fp8_type_)

    sgl_per_token_quant_fp8(input, output, scale)
    scale = scale.reshape(-1, 1)

    return output, scale


# num_tokens sweep spans both kernels: small batches (<=512) take the
# work-group-per-token path; large batches (8192) take the sub-group-per-token
# warp path. hidden_dims include non-16-divisible sizes to exercise the
# vec-width fallback.
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "num_tokens,hidden_dim",
    list(itertools.product([128, 256, 512, 8192], [512, 1076, 1368, 2048, 4096])),
)
def test_per_token_quant_compare_implementations(
    dtype: torch.dtype,
    num_tokens: int,
    hidden_dim: int,
):
    torch.manual_seed(42)
    x = torch.rand((num_tokens, hidden_dim), dtype=dtype, device=device)

    sglang_out, sglang_scale = sglang_per_token_quant_fp8(x)
    torch_out = torch_per_token_quant_fp8(x, sglang_scale)

    torch.testing.assert_close(
        sglang_out.float(), torch_out.float(), rtol=1e-3, atol=1e-3
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
