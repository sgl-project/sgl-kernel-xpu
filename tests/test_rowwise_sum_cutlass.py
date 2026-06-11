import pytest
import torch
from sgl_kernel import row_wise_square_sum


def test_row_wise_sum_basic():
    """Test basic row-wise square sum."""
    device = "xpu"
    M, N = 128, 512

    A = torch.randn(M, N, dtype=torch.float32, device=device)
    result = row_wise_square_sum(A)

    # Reference: PyTorch's square sum
    expected = (A ** 2).sum(dim=1)

    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)
    print(f"✓ Basic test passed: M={M}, N={N}")


def test_row_wise_sum_various_shapes():
    """Test with various matrix shapes."""
    device = "xpu"
    test_shapes = [
        (1, 256),      # Single row
        (16, 64),      # Small
        (64, 1024),    # Wide
        (1024, 128),   # Tall
        (256, 4096),   # Large
    ]

    for M, N in test_shapes:
        A = torch.randn(M, N, dtype=torch.float32, device=device)
        result = row_wise_square_sum(A)
        expected = (A ** 2).sum(dim=1)

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)
        print(f"✓ Shape ({M}, {N}) passed")


def test_row_wise_sum_all_ones():
    """Test with all ones - easy to verify."""
    device = "xpu"
    M, N = 64, 256

    A = torch.ones(M, N, dtype=torch.float32, device=device)
    result = row_wise_square_sum(A)

    # All rows should have square sum = N (since 1^2 = 1)
    expected = torch.full((M,), float(N), dtype=torch.float32, device=device)

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)
    print(f"✓ All-ones test passed: each row square sum = {N}")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_row_wise_sum_dtypes(dtype):
    """Test with different data types."""
    device = "xpu"
    M, N = 64, 512

    A = torch.randn(M, N, dtype=dtype, device=device)
    result = row_wise_square_sum(A)

    expected = (A ** 2).sum(dim=1)

    # Adjust tolerance for lower precision types
    if dtype == torch.bfloat16:
        rtol, atol = 1e-2, 1e-2
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3
    else:
        rtol, atol = 1e-4, 1e-5

    torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)
    print(f"✓ Dtype {dtype} passed")


if __name__ == "__main__":
    print("Testing CUTLASS-style row-wise square sum kernel...")
    test_row_wise_sum_basic()
    test_row_wise_sum_various_shapes()
    test_row_wise_sum_all_ones()
    for dtype in [torch.float32, torch.bfloat16, torch.float16]:
        test_row_wise_sum_dtypes(dtype)
    print("\n✅ All tests passed!")
