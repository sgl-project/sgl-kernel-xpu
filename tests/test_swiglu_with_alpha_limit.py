import itertools
import sys

import pytest
import torch
from sgl_kernel import swiglu_with_alpha_and_limit


def swiglu_with_alpha_and_limit_ref(x, gemm1_alpha, gemm1_limit):
    """Reference implementation using native PyTorch"""
    gate, up = x[..., ::2], x[..., 1::2]
    gate = gate.clamp(min=None, max=gemm1_limit)
    up = up.clamp(min=-gemm1_limit, max=gemm1_limit)
    return gate * torch.sigmoid(gate * gemm1_alpha) * (up + 1)


@pytest.mark.parametrize(
    "batch_size, hidden_size, alpha, limit",
    list(
        itertools.product(
            [1, 16, 128, 512, 1024],  # batch_size
            [64, 128, 256, 512, 1024, 2048, 4096],  # hidden_size (must be even)
            [0.5, 1.0, 2.0],  # alpha
            [1.0, 5.0, 10.0],  # limit
        )
    ),
)
def test_swiglu_with_alpha_and_limit(batch_size, hidden_size, alpha, limit):
    # Ensure hidden_size is even for gate/up split
    if hidden_size % 2 != 0:
        pytest.skip("hidden_size must be even")

    x = torch.randn((batch_size, hidden_size), dtype=torch.float32, device="xpu")

    # Call the kernel
    output = swiglu_with_alpha_and_limit(x, alpha, limit)

    # Reference implementation
    output_ref = swiglu_with_alpha_and_limit_ref(x, alpha, limit)

    # Verify the outputs match
    assert torch.allclose(
        output_ref, output, atol=1e-4, rtol=1e-4
    ), f"Output mismatch: max_diff={torch.max(torch.abs(output_ref - output))}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
