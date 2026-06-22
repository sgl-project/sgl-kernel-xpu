import pytest
import torch
import utils
from sgl_kernel import hc_pre_gemm_sqr_sum

device = utils.get_device()


def _hc_pre_gemm_sqr_sum_torch(A: torch.Tensor, B: torch.Tensor):
    C = A @ B.t()
    sqr_sum_ref = (A * A).sum(dim=1)
    return C, sqr_sum_ref


def _make_inputs(M, K, N, n_splits, a_dtype, b_dtype, device, seed=42):
    torch.manual_seed(seed)
    A = torch.randn(M, K, dtype=a_dtype, device=device)
    B = torch.randn(N, K, dtype=b_dtype, device=device)  # B is [N, K]
    C = torch.empty(n_splits, M, N, dtype=torch.float32, device=device)
    sqr_sum = torch.empty(n_splits, M, dtype=torch.float32, device=device)
    return A, B, C, sqr_sum


@pytest.mark.parametrize(
    "M", [16, 48, 128, 512, 896, 1021, 1024, 1034, 1038, 1518, 2048]
)
def test_hc_pre_gemm_sqr_sum(M):
    torch.manual_seed(42)
    N = 24
    K = 16384
    n_splits = 32 if M <= 2048 else 1
    A, B, C_xpu, sqr_sum_xpu = _make_inputs(
        M, K, N, n_splits, torch.bfloat16, torch.float32, device=f"{device}:0"
    )

    hc_pre_gemm_sqr_sum(C_xpu, sqr_sum_xpu, A, B)
    torch.xpu.synchronize()

    # CPU reference
    C_ref, sqr_sum_ref = _hc_pre_gemm_sqr_sum_torch(A.cpu().float(), B.cpu().float())

    # Reduce the K-split partials over the split axis
    C_xpu_fused = C_xpu.cpu().sum(dim=0)  # [n_splits, M, N] -> [M, N]
    sqr_sum_xpu_fused = sqr_sum_xpu.cpu().sum(dim=0)  # [n_splits, M] -> [M]

    assert torch.allclose(C_xpu_fused, C_ref, atol=2e-4, rtol=2e-4), (
        f"C mismatch: max={(C_xpu_fused - C_ref).abs().max():.3e} "
        f"(M={M} K={K} N={N} n_splits={n_splits})"
    )
    assert torch.allclose(sqr_sum_xpu_fused, sqr_sum_ref, atol=2e-4, rtol=2e-4), (
        f"sqr_sum mismatch: max={(sqr_sum_xpu_fused - sqr_sum_ref).abs().max():.3e} "
        f"(M={M} K={K} N={N} n_splits={n_splits})"
    )
