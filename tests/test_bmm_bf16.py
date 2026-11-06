import sys

import pytest
import torch
import torch.nn.functional as F
from sgl_kernel import bmm_bf16


@pytest.mark.parametrize("in_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("res_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "shape",
    [
        (16, 48, 64, 80),
        (4, 256, 128, 256),
        (1, 512, 512, 512),
    ],
)
def test_bmm_bf16(shape, res_dtype, in_dtype):
    L, M, K, N = shape

    A = torch.randn([L, M, K], device="xpu", dtype=in_dtype)
    B = torch.randn([L, K, N], device="xpu", dtype=in_dtype)

    out = bmm_bf16(A, B, dtype=res_dtype)
    assert out.shape == (L, M, N)
    assert out.dtype == res_dtype

    reference = torch.bmm(A.float(), B.float()).to(res_dtype)
    cos_sim = F.cosine_similarity(
        reference.reshape(-1).float(), out.reshape(-1).float(), dim=0
    )
    assert cos_sim > 0.99, f"cos_sim={cos_sim.item()}"


@pytest.mark.parametrize("in_dtype", [torch.bfloat16, torch.float16])
def test_bmm_bf16_preallocated_out(in_dtype):
    L, M, K, N = 2, 64, 64, 128
    A = torch.randn([L, M, K], device="xpu", dtype=in_dtype)
    B = torch.randn([L, K, N], device="xpu", dtype=in_dtype)
    out = torch.empty([L, M, N], device="xpu", dtype=in_dtype)
    ret = bmm_bf16(A, B, out=out)
    assert ret.data_ptr() == out.data_ptr()

    reference = torch.bmm(A.float(), B.float()).to(in_dtype)
    cos_sim = F.cosine_similarity(
        reference.reshape(-1).float(), out.reshape(-1).float(), dim=0
    )
    assert cos_sim > 0.99


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
