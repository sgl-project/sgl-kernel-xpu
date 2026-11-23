import itertools
import sys

import pytest
import torch
import utils
from sgl_kernel import sgl_per_token_quant_int8

device = utils.get_device()


def torch_per_token_quant_int8(
    tensor: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
    cal_sum: bool = False,
):
    values = tensor.to(torch.float32)
    absmax = values.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
    scale = (absmax / 127.0).to(scale_dtype)
    scaled = (values * (127.0 / absmax)).double()
    # Round half away from zero in float64 to match sycl::round.
    qweight = torch.copysign((scaled.abs() + 0.5).floor(), scaled).to(torch.int8)
    if cal_sum:
        token_sum = values.sum(dim=-1).to(tensor.dtype)
        return qweight, scale, token_sum
    return qweight, scale


# num_tokens and hidden_dim sweep spans the cached warp kernel (large batch,
# hidden_dim <= 1024), single-pass register-cached block kernels (WG 128/256/512),
# and non-16-divisible tail handling.
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("scale_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("cal_sum", [False, True])
@pytest.mark.parametrize(
    "num_tokens,hidden_dim",
    list(
        itertools.product(
            [1, 128, 256, 512, 8192],
            [512, 1076, 1367, 2048, 4096, 5120, 8192, 65536],
        )
    ),
)
def test_per_token_quant_compare_implementations(
    dtype: torch.dtype,
    scale_dtype: torch.dtype,
    cal_sum: bool,
    num_tokens: int,
    hidden_dim: int,
):
    torch.manual_seed(42)
    x = torch.randn((num_tokens, hidden_dim), dtype=dtype, device=device)

    if scale_dtype == torch.float32 and not cal_sum:
        sglang_outputs = sgl_per_token_quant_int8(x)
    else:
        sglang_outputs = sgl_per_token_quant_int8(
            x, scale_dtype=scale_dtype, cal_sum=cal_sum
        )
    torch_outputs = torch_per_token_quant_int8(
        x, scale_dtype=scale_dtype, cal_sum=cal_sum
    )

    assert len(sglang_outputs) == (3 if cal_sum else 2)
    assert sglang_outputs[0].shape == x.shape
    assert sglang_outputs[0].dtype == torch.int8
    assert sglang_outputs[1].shape == x.shape[:-1] + (1,)
    assert sglang_outputs[1].dtype == scale_dtype

    scale_tol = utils.precision[scale_dtype]
    torch.testing.assert_close(sglang_outputs[0], torch_outputs[0], rtol=0, atol=1)
    torch.testing.assert_close(
        sglang_outputs[1], torch_outputs[1], rtol=scale_tol, atol=scale_tol
    )
    if cal_sum:
        assert sglang_outputs[2].shape == x.shape[:-1]
        assert sglang_outputs[2].dtype == dtype
        sum_tol = utils.precision[dtype]
        torch.testing.assert_close(
            sglang_outputs[2].float(),
            torch_outputs[2].float(),
            rtol=max(sum_tol, 1e-4),
            atol=max(sum_tol, 1e-3),
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
