import pytest
import torch
import utils
from sgl_kernel import gemm_sqrsum

device = utils.get_device()

# Production mhc_pre GEMM+sqrsum stage dims (see HCPreFuse.cpp / the mhc_pre driver):
#   A      [M, K]  bf16   (residual.view(M, hc_hidden)); hc_hidden = hc_mult * hidden
#   B      [N, K]  fp32   (fn = [24, 16384] = [N, K])
#   C      [M, N]  fp32   (gemm_out_mul),  C[m,n] = sum_k A[m,k]*B[n,k] = A @ B^T
#   sqrsum [M]     fp32   (gemm_out_sqrsum), sqrsum[m] = sum_k A[m,k]^2
# N (=hc_mult3=24) and K (=hc_mult*hidden=16384) are fixed; only token count M varies.
PROD_N = 24
PROD_K = 16384

# The ragged (not tile-aligned) M values are the ones that exercise partial-tile
# masking; the small-M ones (<=128) exercise the split-K path.
PROD_M = [16, 48, 128, 512, 896, 1021, 1024, 1034, 1038, 1518, 2048]


def _gemm_sqrsum_torch(A: torch.Tensor, B: torch.Tensor):
    """Pure-torch fp32 reference: C = A @ B^T, sqrsum[m] = sum_k A[m,k]^2."""
    A32 = A.float()
    B32 = B.float()
    C_ref = A32 @ B32.t()  # [M, N]
    sqrsum_ref = (A32 * A32).sum(dim=1)  # [M]
    return C_ref, sqrsum_ref


def _make_inputs(M, K, N, a_dtype, b_dtype, device, seed=42):
    torch.manual_seed(seed)
    A = torch.randn(M, K, dtype=a_dtype, device=device)
    B = torch.randn(N, K, dtype=b_dtype, device=device)  # B is [N, K]
    C = torch.empty(M, N, dtype=torch.float32, device=device)
    sqrsum = torch.empty(M, dtype=torch.float32, device=device)
    return A, B, C, sqrsum


def _check(M, K, N, A, B, C, sqrsum):
    C_ref, sqrsum_ref = _gemm_sqrsum_torch(A.cpu(), B.cpu())
    C_t = C.cpu()
    sq_t = sqrsum.cpu()

    # Coverage: the whole output must be written (a partial tile / launch-dim bug
    # leaves rows or columns untouched). This check exposed the original
    # compat::dim3 axis-swap bug, so keep it.
    rows_cov = int((C_t.abs().sum(1) > 0).sum())
    cols_cov = int((C_t.abs().sum(0) > 0).sum())
    assert rows_cov == M, f"only {rows_cov}/{M} rows written (partial-tile/launch bug)"
    assert cols_cov == N, f"only {cols_cov}/{N} cols written (partial-tile/launch bug)"

    # tf32 keeps ~10 mantissa bits; over a K-deep accumulation the absolute error
    # grows ~ 2^-10 * sqrt(K), so scale the tolerance with sqrt(K).
    atol = 2e-2 * max(1, K) ** 0.5
    assert torch.allclose(C_t, C_ref, atol=atol, rtol=2e-2), (
        f"C mismatch: max={(C_t - C_ref).abs().max():.3e} "
        f"(M={M} K={K} N={N})"
    )
    assert torch.allclose(sq_t, sqrsum_ref, atol=atol, rtol=2e-2), (
        f"sqrsum mismatch: max={(sq_t - sqrsum_ref).abs().max():.3e} "
        f"(M={M} K={K} N={N})"
    )


@pytest.mark.parametrize("M", PROD_M)
def test_gemm_sqrsum_production(M):
    """Production mhc_pre shapes: bf16(A) x fp32(B), N=24, K=16384, ragged M."""
    A, B, C, sqrsum = _make_inputs(
        M, PROD_K, PROD_N, torch.bfloat16, torch.float32, device=f"{device}:0"
    )
    gemm_sqrsum(C, sqrsum, A, B)
    torch.xpu.synchronize()
    _check(M, PROD_K, PROD_N, A, B, C, sqrsum)


@pytest.mark.parametrize(
    "M,K,N",
    [
        (256, 256, 256),  # single tile
        (512, 512, 256),  # multi-tile M,N
        (256, 512, 512),  # multi-K
    ],
)
def test_gemm_sqrsum_tile_aligned(M, K, N):
    """Tile-aligned bf16(A) x fp32(B) -> fp32 tf32 path."""
    A, B, C, sqrsum = _make_inputs(
        M, K, N, torch.bfloat16, torch.float32, device=f"{device}:0"
    )
    gemm_sqrsum(C, sqrsum, A, B)
    torch.xpu.synchronize()
    _check(M, K, N, A, B, C, sqrsum)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_gemm_sqrsum_same_dtype(dtype):
    """Non-production A/B same-dtype combos all route through the tf32 launcher
    (inputs widened to fp32, then reinterpreted to tf32 at load)."""
    M, K, N = 256, 256, 256
    A, B, C, sqrsum = _make_inputs(M, K, N, dtype, dtype, device=f"{device}:0")
    gemm_sqrsum(C, sqrsum, A, B)
    torch.xpu.synchronize()
    _check(M, K, N, A, B, C, sqrsum)
