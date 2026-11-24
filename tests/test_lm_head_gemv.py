import pytest
import torch

if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
    pytest.skip("XPU not available", allow_module_level=True)

from sgl_kernel import lm_head_gemv


@pytest.mark.parametrize("m", [1, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_lm_head_gemv_matches_matmul(m, dtype):
    k, n = 1024, 4096
    hidden = torch.randn(m, k, device="xpu", dtype=dtype)
    weight = torch.randn(n, k, device="xpu", dtype=dtype)
    out = lm_head_gemv(hidden, weight)
    ref = hidden @ weight.t()
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
