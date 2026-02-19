import sys
from typing import Optional

import pytest
import torch
import triton
import triton.language as tl
from sgl_kernel import merge_state, merge_state_v2


# Naive PyTorch Implements of Merge Attn States
def merge_state_torch(
    prefix_output: torch.Tensor,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    prefix_lse: torch.Tensor,  # [NUM_TOKENS, NUM_HEADS]
    suffix_output: torch.Tensor,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    suffix_lse: torch.Tensor,  # [NUM_TOKENS, NUM_HEADS]
    output: Optional[torch.Tensor] = None,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    output_lse: Optional[torch.Tensor] = None,  # [NUM_TOKENS, NUM_HEADS]
):
    # Avoid creating new tensors if they are already provided
    if output is None:
        output = torch.empty_like(prefix_output)
    if output_lse is None:
        output_lse = torch.empty_like(prefix_lse)
    p_lse = prefix_lse
    s_lse = suffix_lse
    # inf -> -inf
    p_lse[p_lse == torch.inf] = -torch.inf
    s_lse[s_lse == torch.inf] = -torch.inf
    # max_lse [NUM_HEADS, NUM_TOKENS]
    max_lse = torch.maximum(p_lse, s_lse)
    p_lse = p_lse - max_lse
    s_lse = s_lse - max_lse
    p_lse_exp = torch.exp(p_lse)
    s_lse_exp = torch.exp(s_lse)
    out_se = p_lse_exp + s_lse_exp
    if output_lse is not None:
        output_lse = torch.log(out_se) + max_lse
    p_scale = p_lse_exp / out_se
    s_scale = s_lse_exp / out_se
    p_scale = p_scale.unsqueeze(2)  # [NUM_TOKENS, NUM_HEADS, 1]
    s_scale = s_scale.unsqueeze(2)  # [NUM_TOKENS, NUM_HEADS, 1]
    output = prefix_output * p_scale + suffix_output * s_scale
    return output, output_lse


@pytest.mark.parametrize("num_tokens", [256, 512, 613, 1024, 1536])
@pytest.mark.parametrize("num_query_heads", [8, 16, 32])
@pytest.mark.parametrize("head_size", [32, 48, 64, 128, 256])
@pytest.mark.parametrize("output_dtype", [torch.half, torch.bfloat16])
@torch.inference_mode()
def test_merge_attn_states(
    num_tokens: int, num_query_heads: int, head_size: int, output_dtype: torch.dtype
):
    # prefix_lse and suffix_lse contain inf and normal values
    prefix_lse = torch.randn(
        num_tokens, num_query_heads, dtype=torch.float32, device="cpu"
    )
    suffix_lse = torch.randn(
        num_tokens, num_query_heads, dtype=torch.float32, device="cpu"
    )

    # Generate boolean masks
    mask_prefix = torch.rand(num_tokens, num_query_heads) < 0.1
    mask_suffix = torch.rand(num_tokens, num_query_heads) < 0.1
    # Ensure that the same position is not True at the same time
    combined_mask = torch.logical_and(mask_prefix, mask_suffix)
    mask_prefix = torch.logical_and(mask_prefix, ~combined_mask)
    mask_suffix = torch.logical_and(mask_suffix, ~combined_mask)

    prefix_lse[mask_prefix] = float("inf")
    suffix_lse[mask_suffix] = float("inf")

    # Other input tensors (need to be initialized but
    # no actual calculation needed)
    output = torch.zeros(
        (num_tokens, num_query_heads, head_size), dtype=output_dtype, device="cpu"
    )
    output_lse = torch.zeros(
        (num_tokens, num_query_heads), dtype=torch.float32, device="cpu"
    )
    prefix_output = torch.randn(
        (num_tokens, num_query_heads, head_size), dtype=output_dtype, device="cpu"
    )
    suffix_output = torch.randn(
        (num_tokens, num_query_heads, head_size), dtype=output_dtype, device="cpu"
    )

    # 1. Run the Torch kernel (CPU)
    output_torch, output_lse_torch = merge_state_torch(
        prefix_output.clone(),
        prefix_lse.clone(),
        suffix_output.clone(),
        suffix_lse.clone(),
        output.clone(),
        output_lse.clone(),
    )

    device = "xpu"
    # 2. Run the merge_state V2 kernel (XPU)
    output_v2, output_lse_v2 = merge_state_v2(
        prefix_output.clone().to(device),
        prefix_lse.clone().to(device),
        suffix_output.clone().to(device),
        suffix_lse.clone().to(device),
        output.clone().to(device),
        output_lse.clone().to(device),
    )

    # 3. Correctness compare
    rtol = 1e-2 if output_dtype == torch.bfloat16 else 1e-3
    torch.testing.assert_close(
        output_torch.float(), output_v2.to("cpu").float(), atol=1e-3, rtol=rtol
    )
    torch.testing.assert_close(
        output_lse_torch.float(), output_lse_v2.to("cpu").float(), atol=1e-3, rtol=rtol
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
