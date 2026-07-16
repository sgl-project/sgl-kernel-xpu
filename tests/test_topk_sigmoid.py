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
    routed_scaling_factor: float = 1.0,
    num_fused_shared_experts: int = 0,
):
    assert (
        hidden_states.shape[0] == gating_output.shape[0]
    ), f"Number of tokens mismatch, {hidden_states.shape=} vs {gating_output.shape=}"
    routed_topk = topk - num_fused_shared_experts
    assert routed_topk > 0, "routed_topk must be positive"

    if correction_bias is not None:
        n_routed_experts = gating_output.shape[-1]
        scores = F.sigmoid(gating_output)
        scores_for_choice = scores.view(
            -1, n_routed_experts
        ) + correction_bias.unsqueeze(0)
        topk_ids = torch.topk(scores_for_choice, k=routed_topk, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_ids)
    else:
        topk_weights = F.sigmoid(gating_output.float())
        topk_weights, topk_ids = torch.topk(topk_weights, routed_topk, dim=-1)

    row_sum = topk_weights.sum(dim=-1, keepdim=True)

    if renormalize:
        topk_weights = topk_weights * (routed_scaling_factor / (row_sum + 1e-20))

    if num_fused_shared_experts > 0:
        shared_id = torch.full(
            (topk_ids.shape[0], 1),
            gating_output.shape[-1],
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )
        if renormalize:
            shared_weight = torch.ones(
                (topk_weights.shape[0], 1),
                dtype=topk_weights.dtype,
                device=topk_weights.device,
            )
        else:
            shared_weight = row_sum / routed_scaling_factor

        topk_ids = torch.cat([topk_ids, shared_id], dim=-1)
        topk_weights = torch.cat([topk_weights, shared_weight], dim=-1)

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
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("with_correction_bias", [False, True])
@pytest.mark.parametrize("routed_scaling_factor", [1.0, 2.5])
@pytest.mark.parametrize(
    "n_topk,num_fused_shared_experts",
    [(1, 0), (2, 0), (2, 1), (4, 0), (4, 1)],
)
def test_topk_sigmoid(
    dtype,
    n_token,
    n_topk,
    n_expert,
    renormalize,
    with_correction_bias,
    routed_scaling_factor,
    num_fused_shared_experts,
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
        routed_scaling_factor=routed_scaling_factor,
        num_fused_shared_experts=num_fused_shared_experts,
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
        routed_scaling_factor,
        num_fused_shared_experts,
    )

    # Compare the results; shared expert occupies slot n_expert (beyond routed range)
    n_out_experts = n_expert + num_fused_shared_experts
    res = torch.zeros(
        n_token, n_out_experts, dtype=torch.float, device=hidden_states.device
    )
    ref = torch.zeros(
        n_token, n_out_experts, dtype=torch.float, device=hidden_states.device
    )
    res.scatter_(1, topk_indices.long(), topk_weights)
    ref.scatter_(1, ref_topk_indices.long(), ref_token_weights)

    # Increase the tolerance for this kernel for bf16 and fp16 inputs
    atol = 3e-3
    rtol = 1e-3
    torch.testing.assert_close(res, ref, atol=atol, rtol=rtol)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
