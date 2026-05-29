import pytest
import torch
import utils
from sgl_kernel import hc_pre_gemm

device = utils.get_device()


@pytest.mark.parametrize("T", [16, 128, 512, 1024])
@pytest.mark.parametrize("K", [4096, 16384])
@pytest.mark.parametrize("N", [24, 32])
def test_hc_pre_gemm_correctness(T, K, N):
    """Test HC Pre GEMM against PyTorch reference"""

    # Create inputs
    A = torch.randn(T, K, dtype=torch.bfloat16, device=f"{device}:0")
    # B must be column-major for CUTLASS: create [N, K] then transpose gives [K, N] with stride (1, N)
    B_colmajor = torch.randn(N, K, dtype=torch.float32, device=f"{device}:0")
    B = B_colmajor.t()  # [K, N] with column-major stride
    C = torch.empty(T, N, dtype=torch.float32, device=f"{device}:0")

    # Run kernel
    hc_pre_gemm(A, B, C)

    # Reference: PyTorch matmul
    A_fp32 = A.to(torch.float32)
    C_ref = torch.matmul(A_fp32, B)

    # Compare
    diff = (C - C_ref).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"\nShape: [{T}, {K}] @ [{K}, {N}]")
    print(f"C[0,:5] (kernel):    {C[0, :5]}")
    print(f"C_ref[0,:5] (torch): {C_ref[0, :5]}")
    print(f"Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")

    torch.testing.assert_close(
        C,
        C_ref,
        rtol=2e-2,
        atol=1.5,  # Relaxed for bf16: large K causes larger absolute error
        msg=f"HC Pre GEMM output mismatch for shape [{T}, {K}] @ [{K}, {N}] - max_diff={max_diff:.6f}",
    )
    print(f"✓ Test passed for T={T}, K={K}, N={N}")


def test_hc_pre_gemm_shapes():
    """Test various input shapes"""
    test_cases = [
        (1, 16384, 24),  # Single token
        (16, 16384, 24),  # MHC Pre typical case
        (512, 16384, 24),  # Batch
        (2048, 16384, 24),  # Large batch
    ]

    for T, K, N in test_cases:
        A = torch.randn(T, K, dtype=torch.bfloat16, device=f"{device}:0")
        B = torch.randn(K, N, dtype=torch.float32, device=f"{device}:0")
        C = torch.empty(T, N, dtype=torch.float32, device=f"{device}:0")

        hc_pre_gemm(A, B, C)

        # Verify output shape
        assert C.shape == (
            T,
            N,
        ), f"Output shape mismatch: expected [{T}, {N}], got {C.shape}"
        print(f"✓ Shape test passed for [{T}, {K}] @ [{K}, {N}]")


def test_hc_pre_gemm_dtype_checks():
    """Test that dtype checks work"""
    T, K, N = 16, 1024, 24

    # Test wrong dtype for A
    with pytest.raises(RuntimeError, match="A must be bfloat16"):
        A = torch.randn(T, K, dtype=torch.float32, device=f"{device}:0")
        B = torch.randn(K, N, dtype=torch.float32, device=f"{device}:0")
        C = torch.empty(T, N, dtype=torch.float32, device=f"{device}:0")
        hc_pre_gemm(A, B, C)

    # Test wrong dtype for B
    with pytest.raises(RuntimeError, match="B must be float32"):
        A = torch.randn(T, K, dtype=torch.bfloat16, device=f"{device}:0")
        B = torch.randn(K, N, dtype=torch.bfloat16, device=f"{device}:0")
        C = torch.empty(T, N, dtype=torch.float32, device=f"{device}:0")
        hc_pre_gemm(A, B, C)

    # Test wrong dtype for C
    with pytest.raises(RuntimeError, match="C must be float32"):
        A = torch.randn(T, K, dtype=torch.bfloat16, device=f"{device}:0")
        B = torch.randn(K, N, dtype=torch.float32, device=f"{device}:0")
        C = torch.empty(T, N, dtype=torch.bfloat16, device=f"{device}:0")
        hc_pre_gemm(A, B, C)

    print("✓ Dtype check tests passed")


def test_hc_pre_gemm_dimension_checks():
    """Test that dimension checks work"""

    # Test K dimension mismatch
    with pytest.raises(RuntimeError, match="K dimension mismatch"):
        A = torch.randn(16, 1024, dtype=torch.bfloat16, device=f"{device}:0")
        B = torch.randn(2048, 24, dtype=torch.float32, device=f"{device}:0")
        C = torch.empty(16, 24, dtype=torch.float32, device=f"{device}:0")
        hc_pre_gemm(A, B, C)

    # Test C dimension mismatch
    with pytest.raises(RuntimeError, match="C dimension mismatch"):
        A = torch.randn(16, 1024, dtype=torch.bfloat16, device=f"{device}:0")
        B = torch.randn(1024, 24, dtype=torch.float32, device=f"{device}:0")
        C = torch.empty(16, 32, dtype=torch.float32, device=f"{device}:0")  # Wrong N
        hc_pre_gemm(A, B, C)

    print("✓ Dimension check tests passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
