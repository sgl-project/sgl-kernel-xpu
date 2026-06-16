import sys
from typing import Optional

import pytest
import sgl_kernel
import torch
import torch.nn.functional as F
import utils

device = utils.get_device()


def fused_topk_sigmoid_torch_native(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    correction_bias: Optional[torch.Tensor],
):
    assert (
        hidden_states.shape[0] == gating_output.shape[0]
    ), f"Number of tokens mismatch, {hidden_states.shape=} vs {gating_output.shape=}"
    if correction_bias is not None:
        n_routed_experts = gating_output.shape[-1]
        scores = F.sigmoid(gating_output)
        scores_for_choice = scores.view(
            -1, n_routed_experts
        ) + correction_bias.unsqueeze(0)
        topk_ids = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_ids)
    else:
        M, _ = hidden_states.shape
        topk_weights = torch.empty(
            M, topk, dtype=torch.float32, device=hidden_states.device
        )
        topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
        topk_weights = F.sigmoid(gating_output.float())
        topk_weights, topk_ids = torch.topk(topk_weights, topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids


# float32 covers the fp32 gating-logits path: some models (e.g.
# Nemotron-3-Nano MoE) emit float32 router logits. Before the kernel dispatch
# was widened from AT_DISPATCH_REDUCED_FLOATING_TYPES to also cover Float, this
# call raised "not implemented for 'Float'". The kernel upcasts to float
# internally, so fp32 in must match the float reference within the same
# tolerance as the reduced-float path.
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("n_token", [2, 32, 4096])
@pytest.mark.parametrize("n_expert", [8, 32, 256])
@pytest.mark.parametrize("n_topk", [1, 2, 4])
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("with_correction_bias", [False, True])
def test_topk_sigmoid(
    dtype, n_token, n_topk, n_expert, renormalize, with_correction_bias
):
    torch.manual_seed(1024)

    # expand gating_output by M, otherwise bfloat16 fall into same value after truncating
    hidden_states = torch.randn(n_token, 100, device=device, dtype=dtype)
    gating_output = torch.randn(n_token, n_expert, device=device, dtype=dtype)
    correction_bias = None
    if with_correction_bias:
        correction_bias = torch.randn((n_expert), dtype=torch.float32, device=device)

    ref_token_weights, ref_topk_indices = fused_topk_sigmoid_torch_native(
        hidden_states.float(),
        gating_output.float(),
        n_topk,
        renormalize,
        correction_bias,
    )

    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    M, _ = hidden_states.shape

    topk_weights = torch.empty(
        M, n_topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_indices = torch.empty(
        M, n_topk, dtype=torch.int32, device=hidden_states.device
    )

    sgl_kernel.topk_sigmoid(
        topk_weights,
        topk_indices,
        gating_output,
        renormalize,
        correction_bias,
    )

    # Compare the results
    res = torch.zeros(n_token, n_expert, dtype=torch.float, device=hidden_states.device)
    ref = torch.zeros(n_token, n_expert, dtype=torch.float, device=hidden_states.device)
    res.scatter_(1, topk_indices.long(), topk_weights)
    ref.scatter_(1, ref_topk_indices.long(), ref_token_weights)

    # Increase the tolerance for this kernel for bf16 and fp16 inputs
    atol = 3e-3
    rtol = 1e-3
    torch.testing.assert_close(res, ref, atol=atol, rtol=rtol)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
