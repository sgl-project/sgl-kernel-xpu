import itertools
import sys

import pytest
import torch
import utils
from sgl_kernel import topk_softmax

device = utils.get_device()


@pytest.mark.parametrize(
    "num_tokens, num_experts, topk",
    list(
        itertools.product(
            [1, 16, 128, 512, 1024, 2048],  # num_tokens
            [4, 8, 16, 32, 64, 128, 256],  # num_experts
            [1, 2, 4],  # topk
        )
    ),
)
def test_topk_softmax(num_tokens, num_experts, topk):
    gating_output = torch.randn(
        (num_tokens, num_experts), dtype=torch.float32, device=device
    )

    topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device=device)
    topk_indices = torch.empty((num_tokens, topk), dtype=torch.int32, device=device)
    token_expert_indices = torch.empty(
        (num_tokens, topk), dtype=torch.int32, device=device
    )

    topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
    )

    # Native torch implementation
    softmax_output = torch.softmax(gating_output, dim=-1)
    topk_weights_ref, topk_indices_ref = torch.topk(softmax_output, topk, dim=-1)

    # Verify the top-k weights and indices match the torch native ones
    assert torch.allclose(
        topk_weights_ref, topk_weights, atol=1e-3, rtol=1e-3
    ), f"Weights mismatch: torch={topk_indices_ref} vs SGLang={topk_weights}"

    assert torch.allclose(
        topk_indices_ref.int(), topk_indices, atol=0, rtol=0
    ), f"Indices mismatch: torch={topk_indices_ref}, SGLang={topk_indices}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
