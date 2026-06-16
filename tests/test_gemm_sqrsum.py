import pytest
import torch
import utils
from sgl_kernel import gemm_sqrsum

device = utils.get_device()


def _gemm_sqrsum_torch(A: torch.Tensor, B: torch.Tensor):
    C = A @ B.t()
    sqrsum_ref = (A * A).sum(dim=1)
    return C, sqrsum_ref


def _make_inputs(M, K, N, n_splits, a_dtype, b_dtype, device, seed=42):
    torch.manual_seed(seed)
    A = torch.randn(M, K, dtype=a_dtype, device=device)
    B = torch.randn(N, K, dtype=b_dtype, device=device)  # B is [N, K]
    C = torch.empty(n_splits, M, N, dtype=torch.float32, device=device)
    sqrsum = torch.empty(n_splits, M, dtype=torch.float32, device=device)
    return A, B, C, sqrsum


@pytest.mark.parametrize(
    "M", [16, 48, 128, 512, 896, 1021, 1024, 1034, 1038, 1518, 2048]
)
def test_gemm_sqrsum_production(M):
    torch.manual_seed(42)
    N = 24
    K = 16384
    n_splits = 32 if M <= 2048 else 1
    A, B, C_xpu, sqrsum_xpu = _make_inputs(
        M, K, N, n_splits, torch.bfloat16, torch.float32, device=f"{device}:0"
    )

    gemm_sqrsum(C_xpu, sqrsum_xpu, A, B)
    torch.xpu.synchronize()

    # CPU reference
    C_ref, sqrsum_ref = _gemm_sqrsum_torch(A.cpu().float(), B.cpu().float())

    # Reduce the K-split partials over the split axis
    C_xpu_fused = C_xpu.cpu().sum(dim=0)  # [n_splits, M, N] -> [M, N]
    sqrsum_xpu_fused = sqrsum_xpu.cpu().sum(dim=0)  # [n_splits, M] -> [M]

    # tf32 keeps ~10 mantissa bits; over a K-deep accumulation the absolute error grows ~ 2^-10 * sqrt(K)
    atol = 2e-2 * max(1, K) ** 0.5
    assert torch.allclose(C_xpu_fused, C_ref, atol=atol, rtol=2e-2), (
        f"C mismatch: max={(C_xpu_fused - C_ref).abs().max():.3e} "
        f"(M={M} K={K} N={N} n_splits={n_splits})"
    )
    assert torch.allclose(sqrsum_xpu_fused, sqrsum_ref, atol=atol, rtol=2e-2), (
        f"sqrsum mismatch: max={(sqrsum_xpu_fused - sqrsum_ref).abs().max():.3e} "
        f"(M={M} K={K} N={N} n_splits={n_splits})"
    )
