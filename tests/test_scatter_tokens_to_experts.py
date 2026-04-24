"""
Test scatter_tokens_to_experts kernel with bf16 and fp16 data types.
"""

import pytest
import torch
from sgl_kernel import scatter_tokens_to_experts


def scatter_tokens_to_experts_ref(input_tensor, src2dst_map, num_output_tokens, topk):
    """
    Reference implementation of scatter_tokens_to_experts on CPU.

    Args:
        input_tensor: [num_tokens, hidden_dim] - input tokens
        src2dst_map: [num_tokens * topk] - maps each (token, k) pair to destination row
        num_output_tokens: total output rows (num_tokens * topk)
        topk: number of experts per token

    Returns:
        output_tensor: [num_output_tokens, hidden_dim] - scattered tokens
    """
    num_tokens, hidden_dim = input_tensor.shape
    output_tensor = torch.zeros(
        (num_output_tokens, hidden_dim),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )

    for token_id in range(num_tokens):
        for k in range(topk):
            src_row = token_id
            dst_row = src2dst_map[token_id * topk + k].item()
            output_tensor[dst_row] = input_tensor[src_row]

    return output_tensor


@pytest.mark.parametrize("num_tokens", [1, 4, 16, 32, 128, 512])
@pytest.mark.parametrize("hidden_dim", [64, 128, 256, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("topk", [1, 2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float8_e4m3fn])
def test_scatter_tokens_to_experts_basic(num_tokens, hidden_dim, topk, dtype):
    """Test scatter_tokens_to_experts with various configurations."""
    device = "xpu"
    torch.manual_seed(42)

    # Create input tensor
    # torch.randn doesn't support FP8, so generate in fp32 and convert
    if dtype == torch.float8_e4m3fn:
        input_tensor = torch.randn(
            num_tokens, hidden_dim, dtype=torch.float32, device=device
        ).to(dtype)
    else:
        input_tensor = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)

    # Create src2dst_map: simple sequential mapping for testing
    # Each token scatters to topk consecutive output rows
    num_output_tokens = num_tokens * topk
    src2dst_map = torch.arange(num_output_tokens, dtype=torch.int32, device=device)

    # Create output tensor
    output_tensor = torch.empty(
        num_output_tokens, hidden_dim, dtype=dtype, device=device
    )

    # Run XPU kernel
    scatter_tokens_to_experts(input_tensor, src2dst_map, output_tensor)

    # Run reference implementation on CPU
    input_cpu = input_tensor.cpu()
    src2dst_map_cpu = src2dst_map.cpu()
    output_ref = scatter_tokens_to_experts_ref(
        input_cpu, src2dst_map_cpu, num_output_tokens, topk
    )

    # Compare results: scatter is a pure copy operation, so we expect bit-exact equality
    output_cpu = output_tensor.cpu()

    assert torch.equal(output_cpu, output_ref), (
        f"Scatter kernel should produce bit-exact copy. "
        f"Mismatch for num_tokens={num_tokens}, hidden_dim={hidden_dim}, topk={topk}, dtype={dtype}. "
        f"This indicates a kernel bug since no arithmetic is performed."
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float8_e4m3fn])
def test_scatter_tokens_to_experts_permuted(dtype):
    """Test scatter_tokens_to_experts with non-sequential permutation."""
    device = "xpu"
    torch.manual_seed(42)

    num_tokens = 16
    hidden_dim = 512
    topk = 4
    num_output_tokens = num_tokens * topk

    # Create input tensor
    # torch.randn doesn't support FP8, so generate in fp32 and convert
    if dtype == torch.float8_e4m3fn:
        input_tensor = torch.randn(
            num_tokens, hidden_dim, dtype=torch.float32, device=device
        ).to(dtype)
    else:
        input_tensor = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)

    # Create a random permutation for src2dst_map
    src2dst_map = torch.randperm(num_output_tokens, dtype=torch.int32, device=device)

    # Create output tensor
    output_tensor = torch.empty(
        num_output_tokens, hidden_dim, dtype=dtype, device=device
    )

    # Run XPU kernel
    scatter_tokens_to_experts(input_tensor, src2dst_map, output_tensor)

    # Run reference implementation on CPU
    input_cpu = input_tensor.cpu()
    src2dst_map_cpu = src2dst_map.cpu()
    output_ref = scatter_tokens_to_experts_ref(
        input_cpu, src2dst_map_cpu, num_output_tokens, topk
    )

    # Compare results: scatter is a pure copy operation, so we expect bit-exact equality
    output_cpu = output_tensor.cpu()

    assert torch.equal(output_cpu, output_ref), (
        f"Scatter kernel should produce bit-exact copy. "
        f"Mismatch for permuted test with dtype={dtype}. "
        f"This indicates a kernel bug since no arithmetic is performed."
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float8_e4m3fn])
def test_scatter_tokens_to_experts_edge_cases(dtype):
    """Test edge cases: single token, single expert, large hidden_dim."""
    device = "xpu"

    test_cases = [
        # (num_tokens, hidden_dim, topk)
        (1, 64, 1),  # Single token, single expert
        (1, 4096, 8),  # Single token, large hidden_dim
        (128, 16, 1),  # Many tokens, small hidden_dim
        (64, 8192, 2),  # Large hidden_dim
    ]

    for num_tokens, hidden_dim, topk in test_cases:
        torch.manual_seed(42)

        # Create input tensor
        # torch.randn doesn't support FP8, so generate in fp32 and convert
        if dtype == torch.float8_e4m3fn:
            input_tensor = torch.randn(
                num_tokens, hidden_dim, dtype=torch.float32, device=device
            ).to(dtype)
        else:
            input_tensor = torch.randn(
                num_tokens, hidden_dim, dtype=dtype, device=device
            )

        # Create src2dst_map
        num_output_tokens = num_tokens * topk
        src2dst_map = torch.arange(num_output_tokens, dtype=torch.int32, device=device)

        # Create output tensor
        output_tensor = torch.empty(
            num_output_tokens, hidden_dim, dtype=dtype, device=device
        )

        # Run XPU kernel
        scatter_tokens_to_experts(input_tensor, src2dst_map, output_tensor)

        # Run reference implementation on CPU
        input_cpu = input_tensor.cpu()
        src2dst_map_cpu = src2dst_map.cpu()
        output_ref = scatter_tokens_to_experts_ref(
            input_cpu, src2dst_map_cpu, num_output_tokens, topk
        )

        # Compare results: scatter is a pure copy operation, so we expect bit-exact equality
        output_cpu = output_tensor.cpu()

        assert torch.equal(output_cpu, output_ref), (
            f"Scatter kernel should produce bit-exact copy. "
            f"Mismatch for edge case: num_tokens={num_tokens}, hidden_dim={hidden_dim}, topk={topk}, dtype={dtype}. "
            f"This indicates a kernel bug since no arithmetic is performed."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
